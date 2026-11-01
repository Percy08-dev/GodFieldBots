"""Bounded synthetic GRU behavior; these are component, not learning-quality, proofs."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from gf_cli.config import LoadedConfig, load_config
from gf_common.domain import CandidateSet, ContractError, DecisionInput, digest
from gf_common.schedule import serialize_schedule
from gf_models.arrays import load_arrays, sha256
from gf_models.config import (
    GRUModelConfig,
    SequenceInferenceConfig,
    SequencePPOTrainingConfig,
    gru_model_config,
    sequence_ppo_training_config,
)
from gf_models.encoder import feature_names
from gf_models.features import critic_indices, encode_decision
from gf_models.gru.model import make_network, model_storage_bytes, save_artifact
from gf_models.gru.policy import GRUPolicy
from gf_models.torch_runtime import weights
from gf_training.collection import FinalizedCollection
from gf_training.journal import atomic_json
from gf_training.lineage import checkpoint_path, owned_binding, read_json, validate_chain
from gf_training.native_learning import BulkRollout, RolloutChunk
from gf_training.ppo import gae, normalize_advantages
from gf_training.ppo_run import train_ppo
from gf_training.sequence_numeric import Learner, microbatches, packed
from gf_training.sequence_rollout import read_rollout, write_rollout
from tests.lightgbm_support import synthetic_run
from tests.ppo_support import synthetic_server_identity


def bulk(learner, lengths=(4, 0, 7), *, seed=3, candidates=3):
    torch = learner.torch
    rng = np.random.default_rng(seed)
    chunks = []
    all_advantages = []
    for length in lengths:
        state = rng.normal(size=(length, len(critic_indices(learner.model.encoder)))).astype(np.float32)
        actor = rng.normal(size=(length * candidates, len(feature_names(learner.model.encoder)))).astype(np.float32)
        old_logs, old_values = np.empty(length), np.empty(length)
        cache = learner.network.create_cache()
        with torch.no_grad():
            for index in range(length):
                token = torch.from_numpy(state[index])[None]
                context = learner.network.step(token, cache)
                logs, value = learner.network.heads(context, token,
                                                    torch.from_numpy(actor[index * candidates:(index + 1) * candidates])[None],
                                                    torch.ones((1, candidates), dtype=torch.bool))
                old_logs[index], old_values[index] = float(logs[0, 0]), float(value[0])
        advantage, target = gae(old_values, 1) if length else (np.empty(0), np.empty(0))
        all_advantages.append(advantage)
        chunks.append(RolloutChunk(actor, state, np.arange(length + 1, dtype=np.int64) * candidates,
                                   np.zeros(length, dtype=np.int64), old_logs, old_values, advantage, target,
                                   np.array([0, length], dtype=np.int64)))
    normalized = normalize_advantages(np.concatenate(all_advantages))
    output, start = [], 0
    for chunk in chunks:
        stop = start + len(chunk.selected)
        output.append(replace(chunk, advantage=normalized[start:stop]))
        start = stop
    return BulkRollout(tuple(output), normalized)


def persisted(tmp_path, learner, lengths=(4, 0, 7), name='rollout', **options):
    source = bulk(learner, lengths, **options)
    directory = tmp_path / name
    write_rollout(directory, source, learner.model)
    return read_rollout(directory, sha256(directory / 'rollout.json'), learner.model)


@pytest.mark.parametrize('window', [1, 31, 32, 33, 128, 1024])
def test_sequence_config_strict_window_and_roundtrip(window):
    value = {**asdict(SequencePPOTrainingConfig()), 'sequence_length': window}
    assert sequence_ppo_training_config(value).sequence_length == window


@pytest.mark.parametrize('window', [True, 32., 0, -1, 1025, '32'])
def test_sequence_config_rejects_noninteger_or_out_of_range_window(window):
    with pytest.raises(ContractError):
        sequence_ppo_training_config({**asdict(SequencePPOTrainingConfig()), 'sequence_length': window})


def test_config_dispatch_and_mlp_rejects_sequence_training():
    for path in ('configs/experiments/gru-ppo.toml', 'configs/experiments/gru-ppo-cpu.toml'):
        value = load_config(path)
        assert isinstance(value.config.model, GRUModelConfig)
        assert isinstance(value.config.training, SequencePPOTrainingConfig)
    with pytest.raises(ContractError):
        load_config('configs/experiments/ppo-cpu.toml', {'training': {'sequence_length': 32}})
    config = asdict(GRUModelConfig())
    config['network']['hidden_size'] = True
    with pytest.raises(ContractError):
        gru_model_config(config)
    config = asdict(GRUModelConfig())
    config['inference']['cache_budget_bytes'] = True
    with pytest.raises(ContractError):
        gru_model_config(config)


@pytest.mark.parametrize('window', [1, 31, 32, 33, 128, 1024])
def test_split_forward_and_gradient_boundary(window):
    torch = pytest.importorskip('torch')
    network = make_network(GRUModelConfig(), 17)
    length = min(window + 2, 35)
    tokens = torch.randn(2, length, len(critic_indices(GRUModelConfig().encoder)), requires_grad=True)
    actor = torch.randn(2, 3, len(feature_names(GRUModelConfig().encoder)))
    mask = torch.ones((2, 3), dtype=torch.bool)
    prefix = max(0, length - window)
    full = network(tokens, actor, mask)
    split = network.split(tokens, actor, mask, prefix)
    for actual, expected in zip(split, full, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    (split[0][:, 0].sum() + split[1].sum()).backward()
    assert torch.count_nonzero(tokens.grad[:, :prefix]) == 0
    assert torch.count_nonzero(tokens.grad[:, prefix:]) > 0
    assert torch.count_nonzero(network.projection.weight.grad) > 0
    assert torch.count_nonzero(network.recurrent.weight_hh_l0.grad) > 0
    if not prefix:
        split_grads = [parameter.grad.clone() for parameter in network.parameters()]
        network.zero_grad(set_to_none=True)
        (full[0][:, 0].sum() + full[1].sum()).backward()
        for expected, parameter in zip(split_grads, network.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)


def test_gru_gate_initialization_rng_and_critic_candidate_independence():
    torch = pytest.importorskip('torch')
    before = torch.get_rng_state().clone()
    network = make_network(GRUModelConfig(), 17)
    assert torch.equal(before, torch.get_rng_state())
    assert model_storage_bytes(GRUModelConfig()) == sum(value.numel() * value.element_size() for value in network.state_dict().values())
    for name, parameter in network.recurrent.named_parameters():
        if name.startswith('weight'):
            for gate in parameter.chunk(3, 0):
                torch.testing.assert_close(gate @ gate.T, torch.eye(128), atol=2e-6, rtol=1e-5)
        else:
            assert torch.count_nonzero(parameter) == 0
    state = torch.randn(1, 4, len(critic_indices(GRUModelConfig().encoder)))
    actor = torch.randn(1, 3, len(feature_names(GRUModelConfig().encoder)))
    logs, value = network(state, actor, torch.ones((1, 3), dtype=torch.bool))
    reordered, other = network(state, actor.flip(1), torch.ones((1, 3), dtype=torch.bool))
    torch.testing.assert_close(reordered, logs.flip(1))
    torch.testing.assert_close(other, value, atol=0, rtol=0)
    _, padded_value = network(state, torch.cat((actor, torch.randn_like(actor)), 1), torch.ones((1, 6), dtype=torch.bool))
    torch.testing.assert_close(padded_value, value, atol=0, rtol=0)


def test_banks_shards_empty_episode_no_window_duplication_and_hash_boundaries(tmp_path, monkeypatch):
    learner = Learner(GRUModelConfig(), 17, 'cpu')
    import gf_training.sequence_rollout as module
    monkeypatch.setattr(module, 'MAX_ARRAY_BYTES', 20000)
    rollout = persisted(tmp_path, learner)
    manifest = read_json(tmp_path / 'rollout' / 'rollout.json')
    assert [bank['length'] for bank in manifest['banks']] == [4, 0, 7]
    assert len(manifest['shards']) > 3
    assert sum(bank.nbytes for bank in rollout.banks) == 11 * 373 * 4
    tokens, _ = rollout.endpoint(0)
    np.testing.assert_array_equal(tokens, rollout.banks[0][:1])
    np.testing.assert_array_equal(rollout.endpoint(4)[0], rollout.banks[2][:1])
    assert all(sum(item['bytes'] for item in shard['metadata']['arrays'].values()) <= 20000 for shard in manifest['shards'])
    manifest['shards'][-1]['bank_sha256'] = '0' * 64
    atomic_json(tmp_path / 'rollout' / 'rollout.json', manifest)
    with pytest.raises(ContractError, match='hash'):
        read_rollout(tmp_path / 'rollout', sha256(tmp_path / 'rollout' / 'rollout.json'), learner.model)


def test_maximum_history_and_no_future_or_cross_episode_inputs(tmp_path):
    learner = Learner(GRUModelConfig(), 17, 'cpu')
    rollout = persisted(tmp_path, learner, (1024, 0, 2))
    assert len(rollout.endpoint(1023)[0]) == 1024
    assert len(rollout.endpoint(1024)[0]) == 1
    assert all(len(rollout.endpoint(i)[0]) <= 1024 for i in (0, 31, 32, 1023, 1024, 1025))
    # Changing future states cannot affect the earlier endpoint's prefix.
    early = rollout.endpoint(30)[0].copy()
    rollout.banks[0].flags.writeable = True
    rollout.banks[0][31:] = 1e3
    np.testing.assert_array_equal(early, rollout.endpoint(30)[0])
    with pytest.raises(ContractError, match='resource'):
        bulk(learner, (1025,))
    # The persistence boundary separately rejects a directly supplied oversized
    # bank, without relying on a caller to have used the policy cache first.
    source = bulk(learner, (1024,))
    chunk = source.chunks[0]
    oversized = BulkRollout((replace(chunk, state=np.concatenate((chunk.state, chunk.state[-1:])),
                                     episode_offsets=np.array([0, 1025], dtype=np.int64)),), source.advantage)
    with pytest.raises(ContractError):
        write_rollout(tmp_path / 'oversized', oversized, learner.model)


def test_microbatch_grouping_local_bounds_and_uneven_gradient_weights(tmp_path):
    torch = pytest.importorskip('torch')
    learner = Learner(GRUModelConfig(), 17, 'cpu', 2)
    rollout = persisted(tmp_path, learner, (4, 4, 4))
    logical = [7, 3, 6, 2, 11, 10, 4, 0]
    groups = microbatches(logical, rollout, 2, slots=6)
    assert sorted(i for group in groups for i in group) == sorted(logical)
    assert groups[0] == [7, 3] and groups[1] == [11]
    assert all(len(group) * 3 <= 6 for group in groups)
    with pytest.raises(ContractError, match='resource'):
        microbatches([0], rollout, 2, input_bytes=1)
    gradients = []
    from gf_training.ppo_math import losses
    for slots in (4096, 3):
        learner.network.zero_grad(set_to_none=True)
        for indices in microbatches(logical, rollout, 2, slots=slots):
            tokens, actor, mask = packed(rollout, indices, 'cpu')
            logs, values = learner.network.split(tokens, actor, mask, max(0, tokens.shape[1] - 2))
            selected = torch.from_numpy(rollout.vector(indices, 'selected', np.int64))
            loss, _ = losses(logs, values, mask, selected,
                             *(torch.from_numpy(rollout.vector(indices, key)) for key in ('old_logprob', 'advantage', 'target')))
            (loss * len(indices) / len(logical)).backward()
        gradients.append([parameter.grad.clone() for parameter in learner.network.parameters()])
    for expected, actual in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_whole_batch_parity_mandatory_cpu_split_before_adam_and_resume_identity(tmp_path, monkeypatch):
    learner = Learner(GRUModelConfig(), 17, 'cpu', 2)
    rollout = persisted(tmp_path, learner)
    original = learner.network.split
    split_calls = []
    def split(tokens, actor, mask, prefix):
        split_calls.append((prefix, learner.torch.is_grad_enabled()))
        return original(tokens, actor, mask, prefix)
    monkeypatch.setattr(learner.network, 'split', split)
    result = learner.parity(rollout)
    assert result['decisions'] == len(rollout)
    assert all(grad for _, grad in split_calls) and any(prefix for prefix, _ in split_calls)
    before = weights(learner.network)
    bad = rollout.shards[-1].old_value
    bad.flags.writeable = True
    bad[-1] += .1
    with pytest.raises(ContractError, match='parity'):
        learner.update(rollout)
    assert learner.steps == 0 and not learner.optimizer.state
    for key, value in weights(learner.network).items():
        np.testing.assert_array_equal(value, before[key])
    bad[-1] -= .1
    measured = learner.update(rollout)
    assert measured['cuda_parity_peak_allocated_bytes'] == measured['cuda_parity_peak_reserved_bytes'] == 0
    assert measured['cuda_peak_scope'] == 'optimizer update after parity'
    learner.save(tmp_path / 'generation', 'a' * 64)
    assert read_json(tmp_path / 'generation' / 'inference.json')['schema_version'] == 2
    restored = Learner(GRUModelConfig(), 999, 'cpu', 2)
    restored.restore(tmp_path / 'generation')
    with pytest.raises(ContractError, match='mismatch'):
        Learner(GRUModelConfig(), 17, 'cpu', 3).restore(tmp_path / 'generation')
    next_rollout = persisted(tmp_path, learner, name='next-rollout')
    learner.update(next_rollout)
    restored.update(next_rollout)
    for key, value in weights(learner.network).items():
        np.testing.assert_array_equal(value, weights(restored.network)[key])
    learner.save(tmp_path / 'final-a', 'a' * 64)
    restored.save(tmp_path / 'final-b', 'a' * 64)
    for key, value in load_arrays(tmp_path / 'final-a' / 'resume.npz', read_json(tmp_path / 'final-a' / 'resume.json')['arrays']).items():
        np.testing.assert_array_equal(value, load_arrays(tmp_path / 'final-b' / 'resume.npz', read_json(tmp_path / 'final-b' / 'resume.json')['arrays'])[key])


def test_policy_incremental_reset_history_bound_and_budget(tmp_path, make_input):
    from gf_common.actions import PublicCandidateGenerator
    from gf_common.domain import ActionConfig
    config = GRUModelConfig()
    network = make_network(config, 17)
    save_artifact(network, tmp_path / 'policy', config, 'a' * 64)
    policy = GRUPolicy(str(tmp_path / 'policy'), sha256(tmp_path / 'policy' / 'inference.json'))
    policy.reset(123)
    value = make_input()
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    hidden_pointer = policy.hidden.data_ptr()
    first = policy.select(value, candidates)
    policy.select(value, candidates)
    assert policy.position == 2 and policy.hidden.data_ptr() == hidden_pointer
    policy.reset(123)
    assert policy.select(value, candidates) == first
    policy.position = 1024
    with pytest.raises(ContractError, match='resource'):
        policy.select(value, candidates)
    assert policy.position == 1024
    tiny = replace(config, inference=SequenceInferenceConfig(cache_budget_bytes=1))
    save_artifact(network, tmp_path / 'tiny', tiny, 'a' * 64)
    with pytest.raises(ContractError, match='budget'):
        GRUPolicy(str(tmp_path / 'tiny'), sha256(tmp_path / 'tiny' / 'inference.json'))
    assert policy.reserved_bytes <= config.inference.cache_budget_bytes


def synthetic_sequence_collector(make_input, model_id='gru'):
    async def collect(plan):
        manifest = synthetic_run(plan.output, make_input, seeds=tuple(group.seed for group in plan.groups), planned=plan.groups,
                                 split='train' if plan.kind == 'ppo_batch' else 'validation', both_viewers=True)
        for path in plan.output.glob('*/player-*.jsonl'):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            binding = plan.bindings[records[0]['policy']]
            if binding.model_id != model_id:
                continue
            options = binding.policy_options
            from gf_models.baselines import make_policy
            policy = make_policy(model_id, options)
            policy.reset(123)
            history = []
            for record in records:
                if record['kind'] != 'decision':
                    continue
                raw = record['input']
                value = DecisionInput(raw['observation'], tuple(raw['events']), raw['history_end'], tuple(raw['typed_metadata']), raw['catalog'], raw['action_options'])
                raw_candidates = record['candidates'].copy()
                candidates = CandidateSet(tuple(raw_candidates.pop('actions')), **raw_candidates)
                selection = policy.select(value, candidates)
                actor, state = encode_decision(value, candidates.actions, policy.config.encoder)
                index, torch = record['selection']['index'], policy.torch
                history.append(state)
                with torch.no_grad():
                    logs, values = policy.network(torch.from_numpy(np.stack(history))[None],
                                                  torch.from_numpy(actor)[None], torch.ones((1, len(actor)), dtype=torch.bool))
                record['selection'] = asdict(replace(selection, index=index, logprob=float(logs[0, index]), value=float(values[0])))
            path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
        parent = read_json(Path(plan.linkage['parent_run_path']) / 'run.json')
        integration = parent['identity']['integration']
        manifest.update(binary_sha256=integration['binary_sha256'], server_config_sha256=integration['server_config_sha256'],
                        simulator={'revision': integration['source_revision'], 'spec_sha256': integration['source_spec_sha256']}, code=parent['identity']['code'])
        manifest.update(schema_version=4, collection_kind=plan.kind, ppo=plan.linkage,
                        schedule=serialize_schedule(plan.groups, 4), policy_bindings={key: asdict(value) for key, value in plan.bindings.items()})
        atomic_json(plan.output / 'manifest.json', manifest)
        return FinalizedCollection(plan.output, read_json(plan.output / 'report.json'))
    return collect


def synthetic_gru_collector(make_input):
    return synthetic_sequence_collector(make_input, 'gru')


def test_real_coordinator_worker_gru_two_updates_pause_resume_and_owned_bindings(tmp_path, make_input):
    loaded = load_config('configs/experiments/gru-ppo-cpu.toml', {
        'preprocessing': {'backend': 'python', 'reader': 'legacy'}, 'server': synthetic_server_identity(tmp_path)})
    config = replace(loaded.config, training=replace(loaded.config.training, max_updates=2, sequence_length=2))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    directory = tmp_path / 'parent'
    collector = synthetic_gru_collector(make_input)
    result = asyncio.run(train_ppo(loaded, collector, directory, pause_after_update=1))
    assert result['status'] == 'paused' and result['updates'] == 1
    label, binding = owned_binding(directory, 1)
    assert label.startswith('gru-') and binding.model_id == 'gru'
    changed = replace(config, training=replace(config.training, sequence_length=3))
    with pytest.raises(ContractError, match='mismatch'):
        asyncio.run(train_ppo(LoadedConfig(changed, asdict(changed), digest(asdict(changed)), ()), collector, directory, mode='resume'))
    result = asyncio.run(train_ppo(loaded, collector, directory, mode='resume'))
    assert result['status'] == 'complete' and result['updates'] == 2
    validate_chain(directory)
    assert read_json(directory / 'numeric' / 'batch-000000' / 'rollout.json')['schema_version'] == 2
    assert read_json(directory / 'updates' / '000002.json')['metrics']['parity']['checks'][-1] == 'learner-split'
    assert checkpoint_path(directory, 2).joinpath('resume.npz').is_file()
    # A healthy complete GRU parent is still outside the legacy fixed-data
    # training contract. Supply the actual chain validation evidence to the
    # source scanner and require rejection before any collection is loaded.
    from gf_training.fixed_data import build_dataset
    with pytest.raises(ContractError, match='requires MLP PPO parents'):
        build_dataset((directory, tmp_path / 'z-second', tmp_path / 'z-third'),
                      (tmp_path / 'v1', tmp_path / 'v2', tmp_path / 'v3'), tmp_path / 'dataset.json')



@pytest.mark.parametrize('failure', [False, True])
def test_worker_releases_each_rollout_before_save_ack_next_load_and_on_failure(monkeypatch, capsys, failure):
    import io
    import weakref

    import gf_training.ppo_worker as worker
    import gf_training.sequence_numeric as numeric
    import gf_training.sequence_rollout as storage
    references = []
    class OwnedBatch:
        pass
    def read(*args):
        assert all(reference() is None for reference in references)
        value = OwnedBatch()
        references.append(weakref.ref(value))
        return value
    class NumericalChild:
        def __init__(self, *args):
            self.calls = 0
        def update(self, value):
            assert references[-1]() is value
            self.calls += 1
            del value
            if failure and self.calls == 2:
                raise ContractError('synthetic update failure')
            return {}
        def save(self, *args):
            assert all(reference() is None for reference in references)
            return {}
        def hardware(self):
            return {}
    requests = [{'command': 'initialize', 'model': asdict(GRUModelConfig()), 'seed': 17, 'device': 'cpu',
                 'output': '/tmp/unused', 'action_hash': 'a' * 64, 'sequence_length': 32}]
    requests += [{'command': 'update', 'batch': '/tmp/unused', 'batch_hash': 'a' * 64,
                  'output': '/tmp/unused', 'action_hash': 'a' * 64} for _ in range(2)]
    requests += [{'command': 'close'}]
    monkeypatch.setattr(storage, 'read_rollout', read)
    monkeypatch.setattr(numeric, 'Learner', NumericalChild)
    monkeypatch.setattr(worker.sys, 'stdin', io.StringIO(''.join(json.dumps(value) + '\n' for value in requests)))
    if failure:
        with pytest.raises(ContractError, match='synthetic'):
            worker.main()
    else:
        worker.main()
    assert len(references) == 2 and all(reference() is None for reference in references)
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(replies) == (2 if failure else 4)


def test_sequence_metrics_accumulate_in_order_without_retaining_microbatch_dicts(tmp_path, monkeypatch):
    import weakref

    import gf_training.sequence_numeric as numeric
    learner = Learner(GRUModelConfig(), 17, 'cpu', 2)
    rollout = persisted(tmp_path, learner)
    original = numeric.losses
    references, totals = [], {}
    denominator = 0
    class Metrics(dict):
        pass
    def tracked(*args):
        nonlocal denominator
        # Only the immediately previous local can survive evaluation of the
        # next RHS. Earlier diagnostic dictionaries must already be retired.
        assert sum(reference() is not None for reference in references) <= 1
        loss, metrics = original(*args)
        count = len(args[1])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value * count
        denominator += count
        result = Metrics(metrics)
        references.append(weakref.ref(result))
        return loss, result
    monkeypatch.setattr(numeric, 'losses', tracked)
    measured = learner.update(rollout)
    assert len(references) > 2 and all(reference() is None for reference in references)
    for key, value in totals.items():
        assert measured[key] == value / denominator
