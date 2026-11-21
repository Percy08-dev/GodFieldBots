import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError


def test_trajectory_evidence_uses_learner_seat_even_when_checkpoints_match():
    from gf_common.structured import Selection
    from gf_training.rust_player_storage import _AcceptedTrainingRow, _verify_training_trajectory

    policy = SimpleNamespace(native={'checkpoint': 'a' * 64, 'epoch': 0})
    context = SimpleNamespace(run_id='run', trial='batch', episode_id='episode',
        learner_viewer=2, policies=(policy, policy))
    frame = SimpleNamespace(revision=7, frame_hash='b' * 64)
    rows = [_AcceptedTrainingRow(frame, Selection('c' * 64, 1, 'learner', -.75, .25), 2)]

    def require_identity(*args):
        if args != ('run', 'batch', 'episode', 'game', 2, 'a' * 64, 0, 1, None):
            raise ValueError('different raw identity')

    def require_row(*args):
        if args != (0, 7, 'b' * 64, 'c' * 64, 2, 1, -.75, .25):
            raise ValueError('different raw selection')

    native = SimpleNamespace(require_identity=require_identity, require_row=require_row)
    _verify_training_trajectory(native, context, 'game', 'draw', rows)
    with pytest.raises(ContractError, match='learner/Journal'):
        _verify_training_trajectory(native, SimpleNamespace(**(vars(context) | {'learner_viewer': 1})),
            'game', 'draw', rows)
    with pytest.raises(ContractError, match='learner/Journal'):
        _verify_training_trajectory(native, context, 'game', {'win': {'winner': 1}}, rows)
    with pytest.raises(ContractError, match='learner/Journal'):
        _verify_training_trajectory(native, context, 'game', 'draw',
            [_AcceptedTrainingRow(frame, Selection('c' * 64, 0, 'learner', -.75, .25), 2)])


def test_rust_slab_transfers_borrow_storage_and_keep_empty_operands():
    pytest.importorskip('torch')
    from gf_training.rust_ppo_rollout import _RustTrainingRollout

    backing = np.arange(8, dtype=np.int64)
    backing.flags.writeable = False

    def pack(locations, deadline, part):
        assert (locations, deadline, part) == ([(1, 0)], 42., 'frames')
        return [backing], {'left': (0, 0, [1, 4]), 'empty': (0, 4, [1, 0, 2]),
            'right': (0, 4, [1, 4])}, [0]

    rollout = _RustTrainingRollout.__new__(_RustTrainingRollout)
    rollout.locations = ((1, 0),)
    rollout._state, rollout._deadline = 'PERSISTED', 42.
    rollout._native = SimpleNamespace(pack=pack)
    network = SimpleNamespace(device='cpu', config=SimpleNamespace(resources=SimpleNamespace(max_tensor_bytes=32)))
    window = rollout.pack(network, (0,), 'frames')
    assert window.frame_count == 1 and window.transfer_count == 2 and window.event_lengths == (0,)
    assert window.operands['left'].tolist() == [[0, 1, 2, 3]]
    assert tuple(window.operands['empty'].shape) == (1, 0, 2)
    assert window.operands['right'].tolist() == [[4, 5, 6, 7]]
    assert window.operands['right'].data_ptr() == backing.ctypes.data + 32


@pytest.mark.parametrize('sealed,alive', [(False, False), (True, True), (True, False)])
def test_writer_cannot_release_a_complete_trajectory_without_verified_seal_and_join(sealed, alive):
    from gf_training.rust_player_storage import RustPlayerStorage

    writer = RustPlayerStorage.__new__(RustPlayerStorage)
    writer.training = True
    writer._training_episodes = {'episode': object()}
    writer._training_seal_verified = False
    writer._sealed = sealed
    writer._thread = SimpleNamespace(is_alive=lambda: alive)
    with pytest.raises(ContractError, match='actual sealed learner trajectories'):
        writer.take_training_episodes(('episode',))
    assert list(writer._training_episodes) == ['episode']


def test_optimizer_admission_rechecks_pack_after_numeric_file_validation(tmp_path, monkeypatch):
    from gf_common.domain import ActionConfig, digest
    from gf_common.group_pack import StorageSettings
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_source import RustPlayerSource

    deadline = time.monotonic() + 15
    pack_path = tmp_path / 'collection.sqlite3'
    pack_path.write_bytes(b'original recorded pack')
    numeric_path = tmp_path / 'rollout.json'
    numeric_path.write_bytes(b'{}')
    pack, numeric = OriginalFile(pack_path, deadline), OriginalFile(numeric_path, deadline)
    try:
        source = RustPlayerSource.__new__(RustPlayerSource)
        source.state = 'PERSISTED'
        source.plan = SimpleNamespace(execution_deadline=deadline, linkage={'collection_generation': 0})
        source.parent = SimpleNamespace(optimizer_steps=0, generation=0, require=lambda: None)
        source.collection = SimpleNamespace(pack_verification={}, require=lambda *args: pack.recheck())
        source._rollout = SimpleNamespace(model=object(), action_config_hash=digest({}))
        source._episode_digests = ()
        monkeypatch.setattr(source, '_digests', tuple)
        source._numeric_originals = {'rollout.json': numeric}
        source._numeric_metadata = '{}'
        source.numeric_files = {'rollout.json': numeric.sha256}
        learner = SimpleNamespace(steps=0, model=source._rollout.model)
        local = SimpleNamespace(learner=learner,
            _update_from_source=lambda _: pytest.fail('mutated Pack reached optimizer'))
        config = SimpleNamespace(actions=ActionConfig(), preprocessing=SimpleNamespace(max_source_file_mib=1))
        source._rollout.action_config_hash = digest(asdict(config.actions))
        source.registry = SimpleNamespace(hpo=False, local=local, directory=tmp_path, pending=source,
            current=source.parent, config=config, state={'updates': 0}, budget=StorageBudget(StorageSettings(), deadline))

        def hash_then_mutate():
            pack_path.write_bytes(b'changed during numeric hash')
            return numeric.sha256

        monkeypatch.setattr(numeric, 'hash', hash_then_mutate)
        with pytest.raises(ValueError, match='original file path identity changed'):
            source.consume(tmp_path / 'artifacts/000001')
        assert learner.steps == 0 and source.state == 'FAILED' and source.registry.failed
    finally:
        numeric.close()
        pack.close()


@pytest.mark.parametrize('headroom', [None, 2**63 - 1])
def test_exact_native_capacity_refines_only_an_overflowing_gpu_group(headroom):
    pytest.importorskip('torch')
    from gf_training.hpo_update_planning import plan_groups
    from tests.test_card_instance_learning import learner_for, rollout_for

    learner = learner_for()
    reference = rollout_for(learner)
    rollout = SimpleNamespace(endpoint=lambda _: reference.endpoint(0), resident_bytes=reference.resident_bytes)
    logical = [3, 1, 0, 2]
    kwargs = {'variant': 'microbatch', 'reservation_bytes': learner.binding.managed_budget_bytes,
        'layout': learner.network.layout, 'packed_device_headroom': headroom}
    expected = plan_groups(rollout, logical, learner.model, learner.binding, 1, **kwargs)
    roomy = plan_groups(rollout, logical, learner.model, learner.binding, 1, **kwargs,
        _native_pack_fits=lambda indices, staging: True)
    assert roomy == expected and [group.indices for group in expected] == [tuple(logical)]
    tight = plan_groups(rollout, logical, learner.model, learner.binding, 1, **kwargs,
        _native_pack_fits=lambda indices, staging: len(indices) <= 2)
    assert [group.indices for group in tight] == [(3, 1), (0, 2)]


@pytest.mark.parametrize('model_version', ['3', '4'])
def test_small_native_singleton_budget_refuses_before_shuffle_or_adam(model_version, monkeypatch):
    pytest.importorskip('torch')

    from gf_common.domain import canonical
    from gf_training.rust_ppo_rollout import _RustTrainingRollout

    if model_version == '4':
        from tests.test_hpo_numeric import learner_for, rollout_for
    else:
        from tests.test_card_instance_learning import learner_for, rollout_for
    learner = learner_for()
    reference = rollout_for(learner)
    native = _RustTrainingRollout.__new__(_RustTrainingRollout)
    native._state, native._deadline = 'PERSISTED', 42.
    native._native = SimpleNamespace(_pack_fits=lambda parts, deadline: False)
    native.locations = tuple(reference.locations)
    native.used_decisions = len(reference)
    native.model, native.input_binding = reference.model, reference.input_binding
    native.action_config_hash, native.resident_bytes = reference.action_config_hash, reference.resident_bytes
    monkeypatch.setattr(native, 'endpoint', reference.endpoint)
    before = canonical(learner.shuffle.bit_generator.state)
    if model_version == '4':
        learner._owned_source_budget = native._budget = object()
        native.training, native.generation, native._snapshot_owner = learner.training, 0, None
        monkeypatch.setattr(learner, '_update_numeric', lambda *args, **kwargs: pytest.fail('capacity reached trainer'))
        update = learner.update_owned
    else:
        monkeypatch.setattr(learner, '_update_ordinary_model3', lambda *args, **kwargs: pytest.fail('capacity reached trainer'))
        update = learner._update_owned_model3
    with pytest.raises(ContractError, match='native PPO singleton padding'):
        update(native)
    assert learner.steps == 0 and not learner.optimizer.state
    assert canonical(learner.shuffle.bit_generator.state) == before


def test_model4_cuda_singleton_checks_actual_staged_host_lifetimes(monkeypatch):
    pytest.importorskip('torch')

    from gf_models.structured_learning import FrameStagingPlan
    from gf_training import hpo_update_planning
    from gf_training.rust_ppo_rollout import _RustTrainingRollout
    from tests.test_hpo_numeric import learner_for, rollout_for

    learner = learner_for()
    reference = rollout_for(learner, selected=((0, 0),))
    learner.binding = replace(learner.binding, device='cuda:0')
    monkeypatch.setattr(hpo_update_planning, 'execution_plan', lambda *args, **kwargs:
        hpo_update_planning.ExecutionPlan(FrameStagingPlan((), ((0,),)), {}))
    native = _RustTrainingRollout.__new__(_RustTrainingRollout)
    native._state, native._deadline = 'PERSISTED', 42.
    native.locations, native.used_decisions = tuple(reference.locations), 1
    native.resident_bytes = reference.resident_bytes
    monkeypatch.setattr(native, 'endpoint', reference.endpoint)
    checked = []

    def fits(parts, deadline):
        checked.append((parts, deadline))
        return parts[0][1] == 'all'

    native._native = SimpleNamespace(_pack_fits=fits)
    with pytest.raises(ContractError, match='native PPO singleton padding'):
        learner._validate_singleton_capacity(native)
    assert checked == [([([(0, 0)], 'candidates'), ([(0, 0)], 'frames')], 42.)]


@pytest.mark.parametrize('device,variant', [('cpu', 'microbatch'), ('cuda:0', 'packed')])
def test_model4_scalar_rejects_a_staged_only_model_or_device_plan(device, variant, monkeypatch):
    pytest.importorskip('torch')

    from gf_models.structured_learning import FrameStagingPlan
    from gf_training import hpo_update_planning
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    from gf_training.rust_ppo_rollout import _RustTrainingRollout
    from tests.test_hpo_numeric import learner_for, rollout_for

    learner = learner_for()
    reference = rollout_for(learner, selected=((0, 0),))
    learner.binding = replace(learner.binding, device=device)
    learner._training_execution = TrainingExecutionSpec(variant=variant)
    monkeypatch.setattr(hpo_update_planning, 'execution_plan', lambda *args, **kwargs:
        hpo_update_planning.ExecutionPlan(FrameStagingPlan((), ((0,),)), {}))

    def reject_unsplit(rollout, indices, *_args, **_kwargs):
        assert rollout is native and indices == (0,)
        raise hpo_update_planning.UpdateCapacityError('actual unsplit scalar admission refused')

    monkeypatch.setattr(hpo_update_planning, 'admit_group', reject_unsplit)
    native = _RustTrainingRollout.__new__(_RustTrainingRollout)
    native._state, native._deadline = 'PERSISTED', 42.
    native.locations, native.used_decisions = tuple(reference.locations), 1
    native.resident_bytes = reference.resident_bytes
    monkeypatch.setattr(native, 'endpoint', reference.endpoint)
    native._native = SimpleNamespace(_pack_fits=lambda *args: True)
    with pytest.raises(ContractError, match='actual unsplit scalar admission refused'):
        learner._validate_singleton_capacity(native)


def audit_model3_live_rollout(source, learner, mode):
    """Optional supervisor component. Check actual same-input arrays and losses on CPU."""
    import torch

    from gf_common.domain import digest
    from gf_common.group_pack_game import storage_binding
    from gf_models.structured_artifacts import semantics_v3
    from gf_training.hpo_update_planning import plan_groups
    from gf_training.structured_rollout import _source_correspondence, read_game_rollout

    native = source._rollout
    context = native._context
    proof = source.proof
    declaration = semantics_v3(learner.model, layout=context.layout,
        action_contract=context.actions.action_contract, rules=context.rules)
    storage = storage_binding()
    expected = _source_correspondence(digest(proof), declaration, tuple(proof['episode_digests']),
        game=True, storage_binding=storage)
    # The historical Python reader independently recomputes each token bundle,
    # GAE and target from the bounded raw arrays and writer-verified digests.
    with source.registry.budget.reserve(0) as scratch:
        reference = read_game_rollout(source.numeric_directory, source.numeric_files['rollout.json'],
            learner.model, actions=context.actions, rules=context.rules, declaration=declaration,
            expected_correspondence=expected, storage_binding=storage, _ppo_source=mode,
            check=source._packing_deadline, reserve=scratch.resize,
            max_owned_bytes=source.registry.budget.limit - source.registry.budget.used)
        assert native.locations == tuple(reference.locations)
        assert len(reference) > 128
        order = np.random.Generator(np.random.PCG64(17)).permutation(len(reference)).tolist()
        report = {'decisions': len(order), 'loss_tolerance': 1e-5, 'widths': {}}
        with torch.no_grad():
            for width in (64, 128):
                errors = []
                assert len(order) % width
                for start in (0, len(order) - len(order) % width):
                    logical = order[start:start + width]
                    groups = plan_groups(reference, logical, learner.model, learner.binding, 1,
                        variant='microbatch', reservation_bytes=learner.binding.managed_budget_bytes,
                        layout=context.layout, mlp_width=16, packed_device_headroom=2**63 - 1)
                    left = right = 0.
                    for group in groups:
                        loss, _ = learner._packed_model3_loss(native, group.indices)
                        old, _ = learner._packed_model3_loss(reference, group.indices)
                        torch.testing.assert_close(loss, old, atol=1e-5, rtol=1e-5)
                        left += len(group.indices) * float(loss) / len(logical)
                        right += len(group.indices) * float(old) / len(logical)
                    np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
                    errors.append(abs(left - right))
                report['widths'][str(width)] = {'first_loss_error': errors[0], 'tail_loss_error': errors[1],
                    'tail_decisions': len(order) % width}
            row = order[0]
            loss, _ = learner._scalar_model3_loss(native, row)
            old, _ = learner._scalar_model3_loss(reference, row)
            torch.testing.assert_close(loss, old, atol=1e-5, rtol=1e-5)
            report['scalar_loss_error'] = abs(float(loss) - float(old))
        del reference
        return report


@pytest.mark.skipif(not os.environ.get('GFB_RUST_PPO_MODEL4_PARITY_PARENT'), reason='requires a retained native Model4 parent')
def test_model4_partial_budget_saved_array_first_loss():
    """Replay the first logical loss from the actual frozen checkpoint on CPU."""
    import torch

    from gf_cli.config import load_rust_player_effective
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import digest
    from gf_common.group_pack_game import storage_binding
    from gf_common.rust_ppo_source import source_mode
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.sharded_arrays import read_bundle
    from gf_models.structured_artifacts import load_game_inference_v4
    from gf_models.structured_learning import pack_window
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_numeric import hpo_losses
    from gf_training.hpo_rollout import read_hpo_rollout
    from gf_training.hpo_update_planning import group_frames, plan_groups
    from gf_training.structured_rollout import (
        CANDIDATE_FIELDS,
        EVENT_FIELDS,
        FRAME_FIELDS,
        _hpo_semantics,
        _RolloutContext,
        _source_correspondence,
    )

    parent = Path(os.environ['GFB_RUST_PPO_MODEL4_PARITY_PARENT'])
    state = json.loads((parent / 'run.json').read_bytes())
    update = json.loads((parent / 'updates/000001.json').read_bytes())
    previous = json.loads((parent / 'generations/000000.json').read_bytes())
    proof = update['source_proof']
    loaded = load_rust_player_effective(parent / '.rust-player/effective-config.json',
        expected_sha256=state['config_sha256'])
    mode = source_mode(loaded.effective)
    assert mode is not None
    config = loaded.config
    rules = card_instance_rules()
    context = _RolloutContext(config.actions, rules)
    model = config.model.architecture(layout=layout)
    training = hpo_training_config(config.hpo_spec['training_recipe'])
    binding = replace(config.model.inference, device='cpu')
    network, _ = load_game_inference_v4(parent / 'artifacts/000000', expected_digest=previous['inference_sha256'],
        model=model, binding=binding, action_config_hash=digest(asdict(config.actions)),
        action_contract=config.actions.action_contract, rules=rules)
    storage = storage_binding()
    correspondence = _source_correspondence(digest(proof), _hpo_semantics(model, context),
        tuple(proof['episode_digests']), game=True, hpo=True, storage_binding=storage)
    numeric = parent / update['numeric']
    reference = read_hpo_rollout(numeric, update['numeric_sha256'], model, training,
        actions=config.actions, rules=rules, expected_selection=proof['selection'],
        expected_snapshot_identity=proof['snapshot_identity'], expected_correspondence=correspondence,
        storage_binding=storage, _ppo_source=mode)
    assert 0 < reference.used_decisions < reference.collected_decisions
    manifest = json.loads((numeric / 'rollout.json').read_bytes())
    stored = read_bundle(numeric, manifest['data'])
    order = np.random.Generator(np.random.PCG64(state['seed'])).permutation(len(reference)).tolist()
    logical = order[:training.minibatch_size]
    groups = plan_groups(reference, logical, model, binding, training.tbptt_length,
        variant='microbatch', reservation_bytes=binding.managed_budget_bytes, layout=layout)

    def saved_endpoint(index):
        episode, decision = reference.locations[index]
        prefix, event = f'e.{episode}.f.{decision}', f'e.{episode}.event-tokens.{decision}'
        frame = SimpleNamespace(**{name: stored[f'{prefix}.{name}'] for name in FRAME_FIELDS},
            **{name: stored[f'{event}.{name}'] for name in EVENT_FIELDS})
        candidates = SimpleNamespace(**{name: stored[f'{prefix}.candidate.{name}'] for name in CANDIDATE_FIELDS})
        vector = tuple(stored[f'e.{episode}.{name}'][decision].item()
            for name in ('selected', 'old_logprob', 'old_value', 'advantage', 'target'))
        return (frame,), candidates, vector

    def group_loss(window, size):
        logs, values = network._learning_forward(window, [1] * size, 1)
        raw = window.operands
        return hpo_losses(logs, values, raw['mask'], raw['selected'], *raw['vectors'], training=training)[0]

    actual = expected = 0.
    with torch.no_grad():
        for group in groups:
            endpoints = [reference.endpoint(index) for index in group.indices]
            original = pack_window(network, group_frames(endpoints, model, False),
                [candidates for _, candidates in endpoints], [reference.vector(i) for i in group.indices])
            saved = [saved_endpoint(index) for index in group.indices]
            archived = pack_window(network, [frames[-1] for frames, _, _ in saved],
                [candidates for _, candidates, _ in saved], [vector for _, _, vector in saved])
            size = len(group.indices)
            left, right = group_loss(archived, size), group_loss(original, size)
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
            actual += size * float(left) / len(logical)
            expected += size * float(right) / len(logical)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    print(json.dumps({'used_decisions': reference.used_decisions, 'collected_decisions': reference.collected_decisions,
        'first_logical_decisions': len(logical), 'actual': actual, 'python_reference': expected,
        'abs_loss_error': abs(actual - expected)}, sort_keys=True))
