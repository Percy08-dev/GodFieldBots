"""Mixed fixture Sources and actual CPU components; no executed-game evidence."""
import asyncio
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError, canonical, digest
from tests.test_hpo_native import candidate  # noqa: F401 - explicit private binary
from tests.test_refresh_opponents import DESCRIPTOR


def test_mixed_candidate_capability(request):
    module = request.getfixturevalue('candidate')
    from gf_training.native_backend import require_native_refresh_opponents
    require_native_refresh_opponents(module)


@pytest.mark.parametrize('hpo', [False, True])
def test_complete_mixed_original_source_with_configured_baseline_and_genuine_learner(tmp_path, monkeypatch, request, hpo):
    from gf_models.arrays import sha256
    from gf_training.card_instance_source import _open_batch, recorded_plan
    from gf_training.hpo_source import recorded_hpo_plan
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import _ChainScan
    from tests.card_instance_learning_support import initial_parent, original_collection
    from tests.hpo_support import original_hpo_collection, variable_initial_hpo_chain
    if hpo:
        module = request.getfixturevalue('candidate')
        monkeypatch.setattr('gf_training.native_backend.load_native', lambda: module)
    directory = tmp_path / 'parent'
    if hpo:
        state, records, learner = variable_initial_hpo_chain(directory, games=16, training_opponents=DESCRIPTOR)
        from gf_common.preprocessing import preparation_config
        settings = preparation_config(state['effective_config']['preprocessing'])
    with run_lock(directory, create=not hpo):
        if not hpo:
            learner, state, records, settings = initial_parent(directory, training_opponents=DESCRIPTOR)
        deadline = time.monotonic()+90
        plan, label, _ = (recorded_hpo_plan if hpo else recorded_plan)(directory, state, records, 0, deadline)
        assert len([value for value in plan.bindings.values() if value.model_id == 'heuristic_refresh']) == 1
        (original_hpo_collection if hpo else original_collection)(plan, state, learner)
        with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=deadline) as scan:
            if hpo:
                from gf_training.hpo_lineage import validate_hpo_chain
                validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=deadline, scan=scan)
            else:
                from gf_training.structured_lineage import _validate_component_chain
                _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan, _game=True)
            with _open_batch(plan, state, records, scan, settings, online=None,
                    max_owned_bytes=1024**3, deadline=deadline) as source:
                assert source.learner_label == label and source.proof['games'] == 16
                assert source.proof['qualified'] is False
                numeric = directory / 'numeric/batch-000000'
                source.write_numeric(numeric)
                rollout = source.read_numeric(numeric, sha256(numeric/'rollout.json'))
                rollout.require_learning()
                assert len(rollout) == (3 if hpo else 16)
                assert source.decision_count == 16
                assert source.native_inputs is None


@pytest.mark.parametrize('native', [False, True])
def test_ordinary_mixed_cpu_update_save_resume_and_changed_threshold_refuses(tmp_path, monkeypatch, request, native):
    from gf_cli.config import LoadedConfig, RefreshOpponentsExperimentConfig
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    from gf_common.preprocessing import PreparationConfig
    from gf_models.arrays import sha256
    from gf_models.sharded_arrays import read_json
    from gf_models.structured_binding import ConfiguredNeuralModelV3
    from gf_training.card_instance_ppo import _GameOperations, train_card_instance_ppo
    from gf_training.card_instance_source import recorded_plan, validate_game_parent
    from gf_training.collection import FinalizedCollection
    from gf_training.structured_numeric import GameLearner
    from tests.card_instance_learning_support import original_collection
    from tests.structured_ppo_support import configuration
    from tests.test_card_instance_baseline_runtime import receipt
    from tests.test_card_instance_learning import ACTIONS, RULES, learner_for
    from tests.test_structured_numeric_worker import assert_artifacts_equal
    module = request.getfixturevalue('candidate') if native else None
    if native:
        monkeypatch.setattr('gf_training.native_backend.load_native', lambda: module)
    base = configuration(tmp_path)
    learner = learner_for()
    model = ConfiguredNeuralModelV3(**{key: getattr(learner.model, key) for key in learner.model.__dataclass_fields__}, inference=learner.binding)
    config = RefreshOpponentsExperimentConfig(**{key: getattr(base.config, key) for key in base.config.__dataclass_fields__},
        training_opponents=deepcopy(DESCRIPTOR))
    config = replace(config, actions=ACTIONS, model=model,
        preprocessing=PreparationConfig(backend='rust' if native else 'python', reader='participant-structured-v2', memory_budget_mib=1024))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    monkeypatch.setattr(_GameOperations, 'preflight', lambda self, config, deadline: None)
    def integration(self, config, deadline):
        return {'binary_sha256': receipt()['binary']['sha256'], 'server_config_sha256': sha256(Path(config.server.config)),
            'declared_simulator_revision': ACTIONS.action_contract.simulator_revision,
            'source_revision': ACTIONS.action_contract.simulator_revision,
            'source_spec_sha256': dict(CORRECTED_REVIEWED_SPEC_HASHES)}
    monkeypatch.setattr(_GameOperations, 'integration_identity', integration)
    async def collector(plan, ledger):
        child = ledger.learner._cleanup_owner
        state = read_json(child.directory/'run.json')
        generation = plan.linkage['collection_generation']
        controls = [read_json(child.directory/'generations'/f'{number:06}.json') for number in range(generation+1)]
        expected, _, _ = recorded_plan(child.directory, state, controls, generation, time.monotonic()+60)
        assert plan.groups == expected.groups
        assert plan.bindings == expected.bindings and plan.linkage == expected.linkage
        frozen = GameLearner(learner.model, 17, learner.binding, state['resume_identity'], actions=ACTIONS, rules=RULES)
        frozen.restore(child.directory/'artifacts'/f'{generation:06}')
        original_collection(plan, state, frozen)
        return FinalizedCollection(plan.output, {'comparison_valid': True})
    continuous, resumed = tmp_path/'continuous', tmp_path/'resumed'
    result = asyncio.run(train_card_instance_ppo(loaded, collector, continuous))
    paused = asyncio.run(train_card_instance_ppo(loaded, collector, resumed, pause_after_update=1))
    assert result['updates'] == 2 and paused['status'] == 'paused'
    changed = replace(config, training_opponents={**DESCRIPTOR, 'refresh': {'version': '1', 'attack_threshold': 13}})
    with pytest.raises(ContractError):
        asyncio.run(train_card_instance_ppo(LoadedConfig(changed, asdict(changed), digest(asdict(changed)), ()), collector, resumed, mode='resume'))
    done = asyncio.run(train_card_instance_ppo(loaded, collector, resumed, mode='resume'))
    assert done['updates'] == 2 and done['status'] == 'complete'
    assert_artifacts_equal(continuous/'artifacts/000002', resumed/'artifacts/000002')
    for directory in (continuous, resumed):
        state, records = validate_game_parent(directory, max_owned_bytes=1024**3, deadline=time.monotonic()+90)
        assert len(records) == 3 and state['effective_config']['training_opponents'] == DESCRIPTOR
        assert state['accepted_decisions'] == 32
    if native:
        compare_recorded_arrays(module, resumed, config, tmp_path/'regenerated', hpo=False)


@pytest.mark.parametrize('native', [False, True])
def test_hpo_mixed_cpu_component_common_loop_ack_save_two_updates_resume(tmp_path, monkeypatch, request, native):
    from gf_cli.config import LoadedConfig
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
    from gf_training.hpo_ppo import train_hpo_ppo
    from tests.test_hpo_coordinator import collector_for, controlled_run
    module = request.getfixturevalue('candidate')
    monkeypatch.setattr('gf_training.native_backend.load_native', lambda: module)
    loaded, model, recipe, binding, run_id = controlled_run(tmp_path, monkeypatch, budget=32, max_updates=2, native=native)
    recipe = replace(recipe, collection_games=16)
    spec = deepcopy(loaded.config.hpo_spec)
    spec.update(schema_version=2, training_opponents=deepcopy(DESCRIPTOR), training_recipe=asdict(recipe),
        effective_settings_sha256=effective_settings(model, recipe, {}, layout=LAYOUT).identity)
    config = replace(loaded.config, hpo_spec=spec)
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), loaded.inputs)
    collector = collector_for(loaded, model, recipe, binding)
    job = admit_gpu_job(JobContext('fixture', 'mixed-training', 'training', run_id, None),
        global_cuda_bytes=config.run.global_cuda_budget_bytes, max_workers=4)
    try:
        paused = asyncio.run(train_hpo_ppo(loaded, collector, tmp_path/'parent', gpu_job=job, pause_after_update=1))
        assert paused.report['status'] == 'paused' and paused.report['used_decisions'] == 16
    finally:
        job.close()
    job = admit_gpu_job(JobContext('fixture', 'mixed-resume', 'training', run_id, None),
        global_cuda_bytes=config.run.global_cuda_budget_bytes, max_workers=4)
    try:
        changed_spec = deepcopy(spec)
        changed_spec['training_opponents']['refresh']['attack_threshold'] = 13
        changed = replace(config, hpo_spec=changed_spec)
        with pytest.raises(ContractError):
            asyncio.run(train_hpo_ppo(LoadedConfig(changed, asdict(changed), digest(asdict(changed)), ()),
                collector, tmp_path/'parent', gpu_job=job, mode='resume'))
        done = asyncio.run(train_hpo_ppo(loaded, collector, tmp_path/'parent', gpu_job=job, mode='resume'))
        assert done.report['status'] == 'complete' and done.report['used_decisions'] == 32
        state = read_json(tmp_path/'parent/run.json')
        assert state['effective_config']['hpo_spec']['training_opponents'] == DESCRIPTOR
        assert state['hpo_spec_sha256'] == digest(spec)
        for generation in (1, 2):
            update = read_json(tmp_path/'parent/updates'/f'{generation:06}.json')
            assert len([v for v in update['policy_bindings'].values() if v['model_id'] == 'heuristic_refresh']) == 1
        if native:
            compare_recorded_arrays(module, tmp_path/'parent', config, tmp_path/'regenerated', hpo=True)
    finally:
        job.close()


def compare_recorded_arrays(native, directory, config, output, *, hpo):
    """Independently decode/reconstruct recorded native inputs against saved arrays."""
    import numpy as np

    from gf_models.sharded_arrays import read_bundle, read_json
    from gf_training.native_card_instance import _NativeGameInputs
    plan = {'schema_version': 5 if hpo else 3, 'mode': 'historical_parent',
        'parents': [{'directory': str(directory), 'require_complete': True}], 'collections': []}
    with _NativeGameInputs(plan, config.preprocessing, actions=config.actions,
            max_owned_bytes=1024**3, deadline=time.monotonic()+90, native=native) as owner:
        assert len(owner.summary['collections']) == 2
        for index in range(2):
            actual = owner.compare_recorded(index, output/str(index))
            location = directory/'numeric'/f'batch-{index:06}'
            original = read_json(location/'rollout.json')
            left = read_bundle(location, original['data'])
            right = read_bundle(output/str(index), actual['data'])
            assert set(left) == set(right)
            for name in left:
                np.testing.assert_array_equal(left[name], right[name], err_msg=name)
        assert owner.summary['qualified'] is False
    assert owner.cleanup_complete


@pytest.mark.parametrize('hpo', [False, True])
@pytest.mark.parametrize('mutation', ['pair', 'registry'])
def test_unadmitted_native_original_mixed_roster_tampering_refuses(tmp_path, request, hpo, mutation):
    """Actual FFI structural refusal; this does not create a functional HPO job."""
    import json

    from gf_training.card_instance_source import recorded_plan
    from gf_training.hpo_source import recorded_hpo_plan
    from gf_training.journal import atomic_json
    from gf_training.lineage import run_lock
    from tests.card_instance_learning_support import initial_parent, original_collection
    from tests.hpo_support import original_hpo_collection, variable_initial_hpo_chain
    native = request.getfixturevalue('candidate')
    directory = tmp_path/'parent'
    if hpo:
        state, records, learner = variable_initial_hpo_chain(directory, games=16, training_opponents=DESCRIPTOR, native=native)
    with run_lock(directory, create=not hpo):
        if not hpo:
            learner, state, records, _ = initial_parent(directory, training_opponents=DESCRIPTOR, native=native)
        plan, _, _ = (recorded_hpo_plan if hpo else recorded_plan)(directory, state, records, 0, time.monotonic()+60)
        manifest = (original_hpo_collection if hpo else original_collection)(plan, state, learner)
        raw_plan = {'schema_version': 5 if hpo else 3, 'mode': 'live_cohort',
            'parents': [{'directory': str(directory), 'require_complete': False}],
            'collections': [{'directory': str(plan.output), 'parent_index': 0}], 'cohort_collection_index': 0}
        settings = {'memory_bytes': 1024**3, 'scratch_bytes': 512*1024**2,
            'max_source_file_bytes': 64*1024**2, 'max_record_bytes': 8*1024**2, 'chunk_target_bytes': 8*1024**2}
        def open_session():
            return native.CardInstanceNativeSession(canonical(raw_plan).encode(), canonical(settings).encode(), time.monotonic()+60)
        owner = open_session()
        try:
            assert json.loads(owner.control_summary())['qualified'] is False
        finally:
            owner.close()
        if mutation == 'pair':
            for episode in manifest['schedule'][3]['episodes']:
                episode['policies'] = ['heuristic-card-instance-v2' if label.startswith('heuristic_refresh-') else label
                    for label in episode['policies']]
            manifest['linkage']['schedule_digest'] = digest(manifest['schedule'])
            expected = 'mixed pair opponent'
        else:
            label = next(label for label, binding in manifest['policy_bindings'].items() if binding['model_id'] == 'heuristic_refresh')
            from gf_common.card_instance_rules import card_instance_rules
            from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
            from tests.test_card_instance_learning import ACTIONS
            changed_label = refresh_policy_label(ACTIONS, card_instance_rules(), RefreshOptions(attack_threshold=13))
            binding = manifest['policy_bindings'].pop(label)
            binding['policy_options']['attack_threshold'] = 13
            manifest['policy_bindings'][changed_label] = binding
            manifest['policies'] = [changed_label if value == label else value for value in manifest['policies']]
            for group in manifest['schedule']:
                for episode in group['episodes']:
                    episode['policies'] = [changed_label if value == label else value for value in episode['policies']]
            manifest['linkage']['schedule_digest'] = digest(manifest['schedule'])
            manifest['linkage']['policy_registry_digest'] = digest(manifest['policy_bindings'])
            expected = 'training refresh registry'
        atomic_json(plan.output/'manifest.json', manifest)
        with pytest.raises((ContractError, RuntimeError, ValueError), match=expected):
            open_session()


def reconstruction_parent(tmp_path, *, mixed):
    """Saved CPU artifact copies isolate metadata reconstruction, not learning."""
    import shutil

    import numpy as np

    from gf_training.collection import training_groups
    from gf_training.journal import atomic_json
    from gf_training.lineage import run_lock
    from tests.card_instance_learning_support import initial_parent
    directory = tmp_path/'parent'
    with run_lock(directory, create=True):
        learner, state, records, _ = initial_parent(directory, training_opponents=DESCRIPTOR if mixed else None)
    # Bounded diagnostic metadata makes repeated JSON admission consequential.
    # It is outside the policy/config and carries no learning/source authority.
    state['identity']['fixture_diagnostic_padding'] = 'x'*(1024**2)
    atomic_json(directory/'run.json', state)
    rng = np.random.Generator(np.random.PCG64(state['seed']))
    generations = []
    for number in range(3):
        if number:
            shutil.copytree(directory/'artifacts/000000', directory/'artifacts'/f'{number:06}')
        generations.append({'generation': number, 'artifact_status': 'retained',
            'inference_sha256': records[0]['inference_sha256'], 'resume_sha256': records[0]['resume_sha256'],
            'artifact_files_sha256': digest(records[0]['artifact_files']),
            'parent_controls': {'schedule_opponent_rng': deepcopy(rng.bit_generator.state),
                'snapshot_pool': [0], 'collection_position': number}})
        training_groups(state['seed'], number, 'learner', ('snapshot',), rng, actions_version=3)
    from gf_models.arrays import sha256
    parent = {'parent_index': 0, 'directory': str(directory), 'run_sha256': sha256(directory/'run.json'),
        'model': asdict(learner.model), 'execution': state['execution'], 'generations': generations,
        'state': {key: state[key] for key in ('run_id', 'seed', 'config_sha256', 'resume_identity')}}
    return parent


@pytest.mark.parametrize('mixed', [False, True])
def test_corrected_multigeneration_reconstruction_admits_original_config_once(tmp_path, monkeypatch, mixed):
    from gf_models.structured_sealed import SealedScan, inspect_retained_game_artifact_v3
    from gf_training.native_structured_session import _verify_parent_controls
    from tests.test_card_instance_learning import ACTIONS, RULES
    parent = reconstruction_parent(tmp_path, mixed=mixed)
    directory = Path(parent['directory'])
    deadline = time.monotonic()+60
    # Admit the actual files once to derive a one-config budget independently of
    # the reconstruction loop. No SealedScan cache or accounting is substituted.
    reference = SealedScan(max_owned_bytes=1024**3, deadline=deadline)
    largest_temporary = 0
    charge = reference.charge
    def measured_charge(size, *, temporary=False):
        nonlocal largest_temporary
        if temporary:
            largest_temporary = max(largest_temporary, size)
        return charge(size, temporary=temporary)
    monkeypatch.setattr(reference, 'charge', measured_charge)
    reference.object(directory/'run.json', parent['run_sha256'])
    for generation in parent['generations']:
        inspect_retained_game_artifact_v3(directory/'artifacts'/f"{generation['generation']:06}",
            expected_inference_digest=generation['inference_sha256'], actions=ACTIONS, rules=RULES, scan=reference)
    reference.finalize()
    budget = reference.retained + largest_temporary
    assert 32*(directory/'run.json').stat().st_size+65536 > largest_temporary
    scan = SealedScan(max_owned_bytes=budget, deadline=deadline)
    read_original = scan.object
    original_reads = []
    def observed_object(path, expected=None):
        if path == directory/'run.json':
            original_reads.append(expected)
        return read_original(path, expected)
    monkeypatch.setattr(scan, 'object', observed_object)
    _, _, schedules, registries = _verify_parent_controls([parent], ACTIONS.action_contract, scan, game_actions=ACTIONS)
    assert len(schedules) == len(registries) == 3
    assert original_reads == [parent['run_sha256']]
    assert scan.retained == reference.retained and scan.retained <= budget
    scan.finalize()


@pytest.mark.parametrize('mutation', ['expected-hash', 'config-digest', 'after-read'])
def test_corrected_reconstruction_original_config_hash_and_final_recheck_refuse(tmp_path, monkeypatch, mutation):
    from gf_models import structured_sealed
    from gf_models.structured_sealed import SealedScan
    from gf_training.native_structured_session import _verify_parent_controls
    from tests.test_card_instance_learning import ACTIONS
    parent = reconstruction_parent(tmp_path, mixed=True)
    original = Path(parent['directory'])/'run.json'
    scan = SealedScan(max_owned_bytes=1024**3, deadline=time.monotonic()+60)
    if mutation == 'expected-hash':
        parent['run_sha256'] = 'f'*64
    elif mutation == 'config-digest':
        parent['state']['config_sha256'] = 'e'*64
    else:
        inspect = structured_sealed.inspect_retained_game_artifact_v3
        changed = False
        def mutate_after_original_read(*args, **kwargs):
            nonlocal changed
            value = inspect(*args, **kwargs)
            if not changed:
                assert original in scan.files
                original.write_bytes(original.read_bytes()+b'\n')
                changed = True
            return value
        monkeypatch.setattr(structured_sealed, 'inspect_retained_game_artifact_v3', mutate_after_original_read)
    with pytest.raises(ContractError, match='hash mismatch|Config differs|changed'):
        _verify_parent_controls([parent], ACTIONS.action_contract, scan, game_actions=ACTIONS)
        scan.finalize()
