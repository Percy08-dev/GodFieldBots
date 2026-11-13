"""Complete original and derived HPO correspondence; no experiment claim."""
import copy
from dataclasses import asdict

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models.arrays import atomic_json, sha256
from tests.test_card_instance_learning import ACTIONS, RULES, records_for
from tests.test_group_pack_decoded_retirement import issued  # noqa: F401 - actual packed lifecycle fixture
from tests.test_hpo_numeric import learner_for


def test_complete_original_source_preserves_schedule_and_selected_correspondence(tmp_path):
    import time

    from gf_common.preprocessing import preparation_config
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.hpo_support import original_hpo_collection
    with run_lock(tmp_path / 'parent', create=True):
        directory = tmp_path / 'parent'
        # run_lock creates the directory; fixture owns the contents.
        state, records, learner = initial_hpo_chain(directory / 'contents', decision_budget=3)
        directory = directory / 'contents'
        plan, _, _ = recorded_hpo_plan(directory, state, records, 0, time.monotonic()+60)
        original_hpo_collection(plan, state, learner)
        with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=time.monotonic()+60) as scan:
            from gf_training.hpo_lineage import validate_hpo_chain
            validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=scan.deadline, scan=scan)
            with open_hpo_batch(plan, state, records, scan,
                    preparation_config(state['effective_config']['preprocessing']), online=None,
                    max_owned_bytes=1024**3, deadline=time.monotonic()+60) as source:
                assert len(source.records) == 8 and source.decision_count == 8
                assert source.selection['used_decisions'] == 3
                assert source.proof['accepted_decisions'] > source.decision_count
                numeric = directory / 'numeric' / 'batch-000000'
                manifest = source.write_numeric(numeric)
                rollout = source.read_numeric(numeric, sha256(numeric / 'rollout.json'))
                assert len(rollout) == 3 and rollout.collected_decisions == 8
                assert manifest['selection'] == source.selection
                with pytest.raises(ContractError): source.issue(numeric, {'rollout.json': 'a'*64})


@pytest.mark.parametrize('mutation',[False,True])
def test_parent_source_numeric_preparse_phase_and_sealed_file_guard(tmp_path,monkeypatch,mutation):
    import hashlib
    import time
    from contextlib import nullcontext

    from gf_common.preprocessing import preparation_config
    from gf_models import structured_sealed
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_support import (
        original_hpo_chain,
        original_hpo_collection,
        variable_initial_hpo_chain,
    )
    directory=tmp_path/'parent'
    if mutation:
        state,records,learner=variable_initial_hpo_chain(directory,games=8,decision_budget=8)
    else:
        state,records,_=original_hpo_chain(directory,updates=1,numeric_budget_bytes=1,
            _unqualified_numeric_components=True)
    numeric=directory/'numeric/batch-000000'
    path=numeric/'rollout.json'
    deadline=time.monotonic()+60
    plan,_,_=recorded_hpo_plan(directory,state,records,0,deadline)
    if mutation:
        original_hpo_collection(plan,state,learner)
    with _ChainScan(directory=directory,max_owned_bytes=1024**3,deadline=deadline) as scan:
        validate_hpo_chain(directory,max_owned_bytes=1024**3,deadline=deadline,scan=scan)
        attempted=False
        # A changed original remains poisoned at final SAME-scan recheck.
        with (
            pytest.raises(ContractError) if mutation else nullcontext(),
            open_hpo_batch(plan,state,records,scan,preparation_config(state['effective_config']['preprocessing']),
                online=None,max_owned_bytes=1024**3,deadline=deadline) as source,
        ):
            if mutation:
                source.write_numeric(numeric)
                assert path not in scan.files
            expected=sha256(path)
            parsed=[]
            original_load=structured_sealed.json.loads
            def observed(raw,*args,**kwargs):
                data=raw.encode() if isinstance(raw,str) else raw
                if hashlib.sha256(data).hexdigest()==expected:
                    parsed.append(len(data))
                return original_load(raw,*args,**kwargs)
            monkeypatch.setattr(structured_sealed.json,'loads',observed)
            if mutation:
                original_object=scan.object
                def replaced(target,*args,**kwargs):
                    if target==path:
                        # Same bytes/hash, different actual inode between
                        # preliminary admission and the owned object read.
                        replacement=numeric/'replacement.tmp'
                        replacement.write_bytes(path.read_bytes())
                        replacement.replace(path)
                    return original_object(target,*args,**kwargs)
                monkeypatch.setattr(scan,'object',replaced)
            attempted=True
            with pytest.raises(ContractError):
                source.read_numeric(numeric,expected)
            assert parsed==[]
            assert source._numeric is None
        assert attempted


def test_numeric5_complete_history_roundtrip_selected_targets_and_tamper(tmp_path):
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_rollout import HpoRollout, read_hpo_rollout, write_hpo_rollout
    learner = learner_for('gru')
    records = (records_for(learner, count=5),)
    selection = selection_for_counts((5,), learning_seed=17, batch=0, decision_budget=2, used_before=0)
    rollout = HpoRollout(records, learner.model, learner.training, actions=ACTIONS, rules=RULES,
        selected_locations=tuple((0, index) for index in selection['indices']))
    manifest = write_hpo_rollout(tmp_path / 'numeric', rollout, selection=selection)
    assert manifest['schema_version'] == 5 and manifest['decision_count'] == 2
    assert manifest['training_recipe'] == asdict(learner.training)
    decoded = read_hpo_rollout(tmp_path / 'numeric', sha256(tmp_path / 'numeric/rollout.json'),
        learner.model, learner.training, actions=ACTIONS, rules=RULES, expected_selection=selection,
        expected_records=records)
    assert decoded.collected_decisions == 5 and decoded.used_decisions == 2
    for index in range(2): assert decoded.vector(index) == rollout.vector(index)
    np.testing.assert_array_equal(decoded.episodes[0].target, rollout.episodes[0].target)
    changed = copy.deepcopy(manifest)
    changed['selection']['derived_seed'] += 1
    atomic_json(tmp_path / 'numeric/rollout.json', changed)
    with pytest.raises(ContractError):
        read_hpo_rollout(tmp_path / 'numeric', sha256(tmp_path / 'numeric/rollout.json'),
            learner.model, learner.training, actions=ACTIONS, rules=RULES, expected_selection=selection,
            expected_records=records)


@pytest.mark.parametrize('tensor', ['e.0.advantage', 'e.0.target', 'e.0.old_value', 'e.0.f.0.state_numerics'])
def test_rehashed_numeric5_math_and_original_token_tamper_refused(tmp_path, tensor):
    from gf_models.sharded_arrays import read_bundle
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_rollout import HpoRollout, read_hpo_rollout, write_hpo_rollout
    from tests.test_structured_learning import rewritten_rollout
    learner = learner_for('gru')
    records = (records_for(learner, count=5),)
    selection = selection_for_counts((5,), learning_seed=17, batch=0, decision_budget=1, used_before=0)
    rollout = HpoRollout(records, learner.model, learner.training, actions=ACTIONS, rules=RULES,
        selected_locations=tuple((0, index) for index in selection['indices']))
    directory = tmp_path/'numeric'
    manifest = write_hpo_rollout(directory, rollout, selection=selection)
    arrays = read_bundle(directory, manifest['data'])
    arrays[tensor] = arrays[tensor].copy()
    arrays[tensor].flat[0] += 1
    checksum = rewritten_rollout(directory, manifest, arrays)
    with pytest.raises(ContractError):
        read_hpo_rollout(directory, checksum, learner.model, learner.training, actions=ACTIONS,
            rules=RULES, expected_selection=selection, expected_records=records)


def test_structural_cpu_parent_cannot_become_production_source(tmp_path):
    import time

    from gf_training.hpo_source import validate_hpo_parent
    from tests.hpo_lineage_support import initial_hpo_chain
    directory = tmp_path/'parent'
    initial_hpo_chain(directory)
    with pytest.raises(ContractError, match='actual CUDA'):
        validate_hpo_parent(directory, require_complete=False, max_owned_bytes=1024**3,
            deadline=time.monotonic()+60)


def test_hpo_retirement_freezes_distinct_collected_used_and_nonlegacy_delta(issued, monkeypatch):  # noqa: F811 - fixture imported above
    """Inert lifecycle owner, no collection or optimizer permission is created."""
    from gf_common.domain import canonical
    from gf_common.group_pack_game import storage_binding
    from gf_training.card_instance_source import IssuedGameProjection
    from gf_training.hpo_collection import selection_for_counts
    old, original, reference = issued
    source = old.source
    # This inert component fixture predates actual GPU-job authority. Bypass
    # ONLY that new boundary; all original process/lease/delta/custody checks stay.
    monkeypatch.setattr(type(source.online),'_require_hpo_job',lambda self: None)
    source.state.update(schema_version=5, run_variant='hpo-ppo-v1', model_version='4',
        storage_binding=storage_binding())
    source._hpo_selection = canonical(selection_for_counts((236,), learning_seed=17,
        batch=0, decision_budget=1, used_before=0)).encode()
    expected = {'expected_optimizer_steps': 0, 'collection_generation': 0, 'expected_optimizer_step_delta': 3}
    handle = IssuedGameProjection(source, old.process, canonical(expected).encode(), old.numeric, {})
    handle.completed = True
    source.issued = source.online.issued = handle
    handle.process.steps = 3
    assert (handle._verified_collected, handle._verified_decisions, handle._verified_delta) == (236, 1, 3)
    for name in ('_verified_collected', '_verified_decisions', '_verified_delta'):
        with pytest.raises(AttributeError): setattr(handle, name, 1)
    handle.retire_packed_decoded()
    assert reference() is None and source.records is None
    handle.require(handle.process, after=True)
    handle.process.steps = 4
    with pytest.raises(ContractError): handle.require(handle.process, after=True)
    assert original.artifacts._originals.database_file.fd >= 0


def test_fully_pruned_historical_originals_keep_same_proof_and_correspondence(tmp_path):
    import time

    from gf_common.preprocessing import preparation_config
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_lineage_support import prune_fixture_files
    from tests.hpo_support import original_hpo_chain
    directory = tmp_path/'parent'
    state, records, _ = original_hpo_chain(directory)
    def inspect():
        deadline = time.monotonic()+60
        plan, _, _ = recorded_hpo_plan(directory, state, records, 1, deadline)
        with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=deadline) as scan:
            validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=deadline, scan=scan)
            with open_hpo_batch(plan, state, records, scan, preparation_config(state['effective_config']['preprocessing']),
                    online=None, max_owned_bytes=1024**3, deadline=deadline) as source:
                rollout = source.read_numeric(directory/'numeric/batch-000001', sha256(directory/'numeric/batch-000001/rollout.json'))
                assert rollout.used_decisions == 8
                return source.proof, source.correspondence()
    before = inspect()
    prune_fixture_files(directory, 1)
    assert inspect() == before
    from gf_models.sharded_arrays import read_json
    registry = read_json(directory/'artifact-registry.json')
    registry['generations']['1']['files']['inference.json']['deletion_receipt']['sha256'] = 'a'*64
    atomic_json(directory/'artifact-registry.json', registry)
    with pytest.raises(ContractError): inspect()


def test_explicit_hpo_registry_admits_declared_pool_beyond_five_with_same_custody(tmp_path):
    import time

    from gf_common.domain import RuntimePolicyBinding
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_binding import policy_label_game_v4, safe_options_game_v4
    learner = learner_for()
    registry = {'random-card-instance-v2': RuntimePolicyBinding('random'),
        'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')}
    for number in range(7):
        directory = tmp_path/f'g{number}'
        saved = learner.save(directory, code_revision=f'{number:040x}')
        safe = safe_options_game_v4({'version': '4', 'checkpoint': str(directory),
            'expected_manifest_digest': saved['inference_digest'], 'model': asdict(learner.model),
            'inference': asdict(learner.binding), 'cuda_visible_devices': '',
            'action_config_hash': learner.action_config_hash, **learner.input_binding}, actions=ACTIONS, rules=RULES)
        label = policy_label_game_v4(learner.model, learner.binding, saved['inference_digest'],
            action_contract=ACTIONS.action_contract, rules=RULES, action_config_hash=learner.action_config_hash,
            cuda_visible_devices='')
        registry[label] = RuntimePolicyBinding('mlp', safe)
    options = {'rules': RULES, 'max_owned_bytes': 1024**3,
        'preflight_deadline': time.monotonic()+60, 'finalization_deadline': time.monotonic()+120}
    with pytest.raises(ContractError): RetainedGameArtifacts.for_registry(registry, ACTIONS, **options)
    with pytest.raises(ContractError): RetainedGameArtifacts.for_hpo_registry(registry, ACTIONS, pool_size=5, **options)
    owner = RetainedGameArtifacts.for_hpo_registry(registry, ACTIONS, pool_size=6, **options)
    try:
        assert len(owner.finalize(deadline=time.monotonic()+60)['policies']) == 7
    finally: owner.release()


def test_missing_retained_inference_cannot_use_historical_fallback(tmp_path):
    import time

    from gf_training.hpo_source import original_inference
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_lineage_support import initial_hpo_chain
    directory = tmp_path/'parent'
    state, records, _ = initial_hpo_chain(directory)
    (directory/'artifacts/000000/inference.json').unlink()
    with (_ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=time.monotonic()+60) as scan,
            pytest.raises((ContractError, FileNotFoundError))):
        original_inference(scan, state, records[0], 0, live=False)


def test_variable_original_batch_preserves_pair_ten_schedule_order_and_viewer(tmp_path):
    import time

    from gf_common.preprocessing import preparation_config
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_support import original_hpo_collection, variable_initial_hpo_chain
    directory = tmp_path/'parent'
    state, records, learner = variable_initial_hpo_chain(directory)
    deadline = time.monotonic()+60
    plan, _, _ = recorded_hpo_plan(directory, state, records, 0, deadline)
    original_hpo_collection(plan, state, learner)
    with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=deadline) as scan:
        validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=deadline, scan=scan)
        with open_hpo_batch(plan, state, records, scan, preparation_config(state['effective_config']['preprocessing']),
                online=None, max_owned_bytes=1024**3, deadline=deadline) as source:
            expected = [(spec.episode_id, spec.learner_viewer) for group in plan.groups for spec in group.episodes]
            actual = [(record.bank.episode.rsplit('-', 1)[0], record.bank.actor) for record in source.records]
            assert actual == expected and len(actual) == 24
            assert actual[20][0] == 'b000000-p10-seat0'


@pytest.mark.parametrize('wrong_seed', [None, 999])
def test_original_g0_schedule_rng_is_seed_bound_before_collection_or_production_gate(tmp_path, monkeypatch, wrong_seed):
    import time

    from gf_models.sharded_arrays import read_json
    from gf_training import hpo_source
    from gf_training.hpo_lineage import validate_hpo_chain
    from tests.hpo_lineage_support import initial_hpo_chain, rehash_chain
    directory = tmp_path/'parent'
    initial_hpo_chain(directory)
    if wrong_seed is not None:
        record = read_json(directory/'generations/000000.json')
        record['parent_controls']['schedule_opponent_rng'] = np.random.Generator(np.random.PCG64(wrong_seed)).bit_generator.state
        atomic_json(directory/'generations/000000.json', record)
        rehash_chain(directory)
    # The structural reader deliberately owns only PCG shape; full Source must
    # bind the original seed even at G0 before there is a collection to replay.
    state, records = validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic()+60)
    production_checks = []
    monkeypatch.setattr(hpo_source, 'require_hpo_production', lambda _: production_checks.append('checked'))
    def verify():
        return hpo_source.validate_hpo_parent(directory, require_complete=False,
            max_owned_bytes=1024**3, deadline=time.monotonic()+60)
    if wrong_seed is None:
        assert verify()[0]['updates'] == 0 and production_checks == ['checked']
        assert hpo_source.recorded_hpo_plan(directory, state, records, 0, time.monotonic()+60)[0].groups
        for invalid_generation in (1, True):
            with pytest.raises(ContractError):
                hpo_source.recorded_hpo_plan(directory, state, records, invalid_generation, time.monotonic()+60)
    else:
        with pytest.raises(ContractError, match='generation0.*seed'): verify()
        assert production_checks == []
        with pytest.raises(ContractError, match='generation0.*seed'):
            hpo_source.recorded_hpo_plan(directory, state, records, 0, time.monotonic()+60)
