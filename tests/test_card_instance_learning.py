"""Numerical/lifecycle fixtures; these do not authorize a qualified game Source."""
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured import CandidateTable, Selection
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.arrays import sha256
from gf_models.structured_artifacts import semantics_v3
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_runtime import (
    admit_planned_game_training_v3,
    learner_process_bytes,
    own_game_worker_settings_v3,
)
from gf_models.structured_sealed import SealedScan, inspect_retained_game_artifact_v3
from gf_training.structured_numeric import GameLearner
from gf_training.structured_rollout import (
    _episode_correspondence_digest,
    _source_correspondence,
    _write_rollout,
    game_episode_records,
    game_rollout,
    read_game_rollout,
    read_rollout,
    write_rollout,
)
from tests.model3_support import LAYOUT, model3, raw_inputs
from tests.test_structured_learning import assert_same_learners

RULES = card_instance_rules()
ACTIONS = ActionConfigV3(action_contract=LAYOUT.action_contract, max_candidates=64)
RESUME = digest({'numerical-fixture': 'not-source-authority'})


def learner_for(temporal='mlp', hand='attention', device='cpu'):
    if device != 'cpu' and os.environ.get('GF_RUN_CUDA') != '1':
        pytest.skip('actual HOST CUDA checks require GF_RUN_CUDA=1')
    return GameLearner(model3(hand, temporal), 17, InferenceBindingV2(device=device, max_candidates=64, candidate_chunk=2),
                       RESUME, actions=ACTIONS, rules=RULES)


def records_for(learner, *, count=3, empty=False, outcome=None, kind='natural'):
    _, _, frames, bank, _ = raw_inputs(count=count, empty=empty)
    tables, selected = [], []
    for index, frame in enumerate(frames):
        actions = ([{'type': 'pray'}] if empty else [
            {'type': 'play_sequence', 'ordered_card_refs': [frame.references[0]], 'target': 2},
            {'type': 'discard', 'card_refs': [frame.references[1]]}])
        table = CandidateTable.from_actions(frame, actions, action_contract=ACTIONS.action_contract,
            purpose='qualified_game', synthetic_capacity=False)
        tokens = tuple(tokenize_frame(row, bank.view(), layout=LAYOUT) for row in frames[:index + 1])
        candidates = tokenize_table(table, frame, 18, layout=LAYOUT)
        with learner.torch.no_grad():
            logs, value, _ = learner.network(tokens, candidates)
        selection = index % len(actions)
        selected.append(Selection(table.table_hash, selection, 'numerical-fixture-policy', float(logs[selection].cpu()), float(value.cpu())))
        tables.append(table)
    return game_episode_records(bank.view(), tuple(frames), tuple(tables), tuple(selected), kind,
        {'win': {'winner': 1}} if outcome is None and kind == 'natural' else outcome,
        None if kind == 'natural' else 'fixture interruption', actions=ACTIONS, rules=RULES)


def rollout_for(learner, **options):
    return game_rollout((records_for(learner, **options),), learner.model, actions=ACTIONS, rules=RULES)


def correspondence(rollout):
    return _source_correspondence('a' * 64, semantics_v3(
            rollout.model, layout=LAYOUT, action_contract=ACTIONS.action_contract, rules=RULES),
        tuple(_episode_correspondence_digest(item.records) for item in rollout.episodes), game=True)


@pytest.mark.parametrize('hand', ['direct', 'deepsets', 'attention'])
@pytest.mark.parametrize('temporal', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_game4_raw_numeric_parity_and_rng_neutral(tmp_path, temporal, hand, device):
    learner = learner_for(temporal, hand, device)
    rollout = rollout_for(learner)
    from gf_models.structured_artifacts import semantics_v3
    declaration = semantics_v3(learner.model, layout=LAYOUT, action_contract=ACTIONS.action_contract, rules=RULES)
    target = tmp_path / 'numeric'
    manifest = _write_rollout(target, rollout, shard_bytes=32768)
    assert manifest['schema_version'] == 4 and manifest['data']['schema_version'] == 1
    restored = read_game_rollout(target, sha256(target / 'rollout.json'), learner.model, actions=ACTIONS,
        rules=RULES, declaration=declaration, expected_correspondence=correspondence(rollout))
    assert restored.episodes[0].records == rollout.episodes[0].records
    np.testing.assert_array_equal(restored.episodes[0].advantage, rollout.episodes[0].advantage)
    before = learner.torch.get_rng_state().clone()
    cuda_before = [value.clone() for value in learner.torch.cuda.get_rng_state_all()] if device != 'cpu' else []
    shuffle = canonical(learner.shuffle.bit_generator.state)
    assert learner.parity(restored)['decisions'] == 3
    assert learner.torch.equal(before, learner.torch.get_rng_state())
    assert all(learner.torch.equal(left, right) for left, right in zip(cuda_before, learner.torch.cuda.get_rng_state_all() if cuda_before else [], strict=True))
    assert canonical(learner.shuffle.bit_generator.state) == shuffle
    with pytest.raises(ContractError):
        write_rollout(tmp_path / 'legacy-write', rollout)
    with pytest.raises(ContractError):
        read_rollout(target, sha256(target / 'rollout.json'), learner.model,
            action_contract=ACTIONS.action_contract, purpose='qualified_game', synthetic_capacity=False)


@pytest.mark.parametrize('temporal', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_game_two_updates_resume_full_named_state(tmp_path, temporal, device):
    left = learner_for(temporal, device=device)
    left.update(rollout_for(left))
    directory = tmp_path / 'pair'
    result = left.save(directory, shard_bytes=1024**2)
    pair = inspect_retained_game_artifact_v3(directory, expected_inference_digest=result['inference_digest'],
        actions=ACTIONS, rules=RULES, scan=SealedScan(max_owned_bytes=256 * 1024**2, deadline=time.monotonic() + 30))
    assert pair.resume['optimizer_steps'] == left.steps and pair.resume['schema_version'] == 4
    batch = rollout_for(left)
    right = learner_for(temporal, device=device)
    right.restore(directory)
    assert_same_learners(left, right)
    left.update(batch)
    expected_cpu = left.torch.get_rng_state().clone()
    expected_cuda = [value.clone() for value in left.torch.cuda.get_rng_state_all()] if device != 'cpu' else []
    right.restore(directory)
    right.update(batch)
    assert_same_learners(left, right)
    assert right.torch.equal(expected_cpu, right.torch.get_rng_state())
    assert all(right.torch.equal(a, b) for a, b in zip(expected_cuda, right.torch.cuda.get_rng_state_all() if expected_cuda else [], strict=True))


def test_saved_output_mismatch_fails_before_rng_optimizer_mutation():
    learner = learner_for()
    record = records_for(learner)
    bad = replace(record.selections[0], logprob=record.selections[0].logprob + .1)
    record = game_episode_records(record.bank, record.frames, record.tables, (bad, *record.selections[1:]),
        record.end_kind, record.outcome, actions=ACTIONS, rules=RULES)
    rollout = game_rollout((record,), learner.model, actions=ACTIONS, rules=RULES)
    torch_rng = learner.torch.get_rng_state().clone()
    shuffle = canonical(learner.shuffle.bit_generator.state)
    parameters = {key: value.clone() for key, value in learner.network.state_dict().items()}
    with pytest.raises(ContractError, match='parity'):
        learner.update(rollout)
    assert learner.steps == 0 and not learner.optimizer.state
    assert learner.torch.equal(torch_rng, learner.torch.get_rng_state())
    assert canonical(learner.shuffle.bit_generator.state) == shuffle
    assert all(learner.torch.equal(value, learner.network.state_dict()[key]) for key, value in parameters.items())


def test_empty_natural_and_interrupted_episode_reward_semantics():
    learner = learner_for()
    record = records_for(learner)
    empty = game_episode_records(record.bank, (), (), (), 'natural', {'win': {'winner': 2}}, actions=ACTIONS, rules=RULES)
    rollout = game_rollout((empty,), learner.model, actions=ACTIONS, rules=RULES)
    with pytest.raises(ContractError, match='nonempty'):
        rollout.require_learning()
    interrupted = records_for(learner, kind='truncated')
    cohort = game_rollout((interrupted,), learner.model, actions=ACTIONS, rules=RULES)
    with pytest.raises(ContractError, match='natural'):
        cohort.require_learning()
    winner = rollout_for(learner, count=1)
    loser = rollout_for(learner, count=1, outcome={'win': {'winner': 2}})
    assert winner.vector(0)[-1] == 1 and loser.vector(0)[-1] == -1


def test_numeric_rejects_correspondence_and_version_before_arrays(tmp_path, monkeypatch):
    learner = learner_for()
    rollout = rollout_for(learner)
    from gf_models.structured_artifacts import semantics_v3
    declaration = semantics_v3(learner.model, layout=LAYOUT, action_contract=ACTIONS.action_contract, rules=RULES)
    target = tmp_path / 'numeric'
    manifest = _write_rollout(target, rollout)
    import gf_training.structured_rollout as module
    def forbidden(*args, **kwargs):
        pytest.fail('mixed context reached array admission')
    monkeypatch.setattr(module, 'inspect_bundle', forbidden)
    manifest['schema_version'] = 3
    (target / 'rollout.json').write_text(canonical(manifest))
    with pytest.raises(ContractError, match='recipe'):
        read_game_rollout(target, sha256(target / 'rollout.json'), learner.model, actions=ACTIONS,
            rules=RULES, declaration=declaration, expected_correspondence=correspondence(rollout))


def test_planned_admission_matches_actual_worker_formula_without_artifact(tmp_path):
    from tests.test_card_instance_neural import saved
    owner, _label, execution, _, model, binding, _ = saved(tmp_path)
    try:
        settings = own_game_worker_settings_v3(model.model_id, ACTIONS, execution.policy_options, rules=RULES)
        learner = replace(binding, device='cuda:0')
        expected = learner_process_bytes(model, learner, layout=LAYOUT)
        actual = admit_planned_game_training_v3(ACTIONS, model, binding, learner, rules=RULES,
            game_concurrency=2, global_cuda_bytes=12 * 1024**3, max_workers=4)
        assert actual['worker_resident_bytes'] == 4 * settings[2]
        assert actual['decision_temporary_bytes'] == 2 * settings[3]
        assert actual['learner_cuda_bytes'] == expected
    finally:
        owner.release()


@pytest.mark.parametrize('temporal', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_wire4_initialize_restore_raw_request_refusal_and_reap(tmp_path, temporal, device):
    import asyncio

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.structured_learning_process import GameLearningProcess
    learner = learner_for(temporal, device=device)
    learner.update(rollout_for(learner))
    original = learner.save(tmp_path / 'direct')
    async def run():
        ledger = ReservationLedger()
        child = GameLearningProcess(tmp_path, 1024**2, learner.model, learner.binding, digest(asdict(ACTIONS)), RESUME,
            actions=ACTIONS, ledger=ledger, cuda_visible_devices='0', cleanup_deadline=time.monotonic() + 60)
        await child.start()
        try:
            result = await child.initialize(17, tmp_path / 'unused', restore=tmp_path / 'direct', deadline=time.monotonic() + 60)
            assert result['protocol_version'] == 4 and child.steps == 4
            assert result['artifact']['files'] == original['files']
            assert child.identity['execution']['cuda_visible_devices'] == ('' if device == 'cpu' else '0')
            assert not (tmp_path / 'unused').exists()
            with pytest.raises(ContractError, match='Source-issued'):
                await child.update({}, tmp_path / 'forbidden', deadline=time.monotonic() + 60)
            with pytest.raises(ContractError, match='raw request'):
                await child.request({'protocol_version': 4, 'command': 'update', 'identity': child.identity,
                    'batch': 'missing', 'batch_digest': 'a' * 64, 'expected_policy_version': 'synthetic',
                    'output': 'missing', 'transaction_id': 'b' * 64, 'projection': {}}, time.monotonic() + 60)
            assert child.steps == 4 and not ledger.updating
        finally:
            await child.close(time.monotonic() + 60)
        assert child.process.returncode == 0 and child.lease.released and ledger.learner is None
    asyncio.run(run())


def test_projection_closed_generation_steps_before_any_numerical_use():
    from gf_training.structured_numeric_worker import validate_projection
    value = {'recipe': 'card-instance-source-update-v1', 'parent_run_id': 'fixture', 'batch_id': 'batch-000002',
        'collection_generation': 2, 'expected_optimizer_steps': 8, 'expected_policy_version': 'fixture-policy',
        'collection_manifest_sha256': 'a' * 64, 'numeric_manifest_sha256': 'b' * 64, 'numeric_files_sha256': 'c' * 64,
        'source_correspondence': {'recipe': 'structured-source-cohort-correspondence-v2', 'control_fingerprint': 'a' * 64,
            'semantics_sha256': 'b' * 64, 'episode_count': 16, 'episode_digests': ['c' * 64] * 16, 'binding_sha256': 'd' * 64}}
    for mutation in ({'expected_optimizer_steps': 4}, {'expected_optimizer_steps': True}, {'collection_generation': True},
                     {'batch_id': 'batch-000001'}, {'extra': 'forbidden'}):
        with pytest.raises(ContractError):
            validate_projection({**value, **mutation}, expected_steps=8, policy='fixture-policy', batch_digest='b' * 64)


@pytest.mark.parametrize('mode', ['fail-spawn', 'cancel-spawn'])
def test_game_learner_start_failure_and_cancellation_preserve_reap_owner(tmp_path, monkeypatch, mode):
    import asyncio

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.structured_learning_process import GameLearningProcess
    learner = learner_for()
    actual_spawn = asyncio.create_subprocess_exec
    spawned = []
    async def run():
        ledger = ReservationLedger()
        child = GameLearningProcess(tmp_path, 1024**2, learner.model, learner.binding, digest(asdict(ACTIONS)), RESUME,
            actions=ACTIONS, ledger=ledger, cuda_visible_devices='', cleanup_deadline=time.monotonic() + 60)
        if mode == 'fail-spawn':
            async def fail(*args, **kwargs):
                raise OSError('injected game numerical spawn failure')
            monkeypatch.setattr(asyncio, 'create_subprocess_exec', fail)
            with pytest.raises(OSError, match='injected'):
                await child.start()
        else:
            returned, release = asyncio.Event(), asyncio.Event()
            async def delayed(*args, **kwargs):
                process = await actual_spawn(*args, **kwargs)
                spawned.append(process)
                returned.set()
                await release.wait()
                return process
            monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
            starting = asyncio.create_task(child.start())
            try:
                await asyncio.wait_for(returned.wait(), 10)
                starting.cancel()
                await asyncio.sleep(0)
                assert ledger.learner._cleanup_owner is child and child.lease.creation_pending
                with pytest.raises(ContractError):
                    child.lease.cancel_before_spawn()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await starting
            finally:
                release.set()
                await asyncio.gather(starting, return_exceptions=True)
                for process in spawned:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
            assert spawned[0].returncode is not None
        assert ledger.learner is None and child.lease.released and not ledger.updating
    asyncio.run(run())
