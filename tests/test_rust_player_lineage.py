"""Strict new metadata boundaries; data fixtures grant no live Source authority."""

from copy import deepcopy

import pytest

from gf_common.domain import ContractError
from tests.test_rust_player_config import runtime


def operation_data():
    from gf_common.rust_player import deadline_bits

    return {"schema_version": 2, "kind": "rust-player-operation-v2",
        "operation_id": "operation", "run_id": "parent", "mode": "ordinary",
        "operation_type": "ordinary-train", "config_path": "/tmp/effective.json",
        "config_sha256": "a" * 64, "runtime_binding_sha256": "b" * 64,
        "parent_directory": "/tmp/parent", "expected_context": None,
        "deadlines": {name: deadline_bits(value) for name, value in zip(
            ("admission_bits", "execution_bits", "term_bits", "kill_bits", "finish_bits"),
            (100., 110., 120., 130., 140.), strict=True)},
        "execution_request": {"mode": "fresh", "pause_after_update": 1}}


def producer_data():
    return {"schema_version": 1, "kind": "rust-player-producer-v1",
        **dict.fromkeys(("python_source_digest", "python_runtime_digest", "extension_sha256",
            "supervisor_sha256", "sampling_runtime_sha256"), "a" * 64),
        "build": {"version": "fixture", "rustc": "fixture", "target": "fixture",
            "profile": "release", "rustflags": "", "cargo_lock_sha256": "b" * 64,
            "native_source_sha256": "c" * 64, "abi": "cp314-gil", "allocator": "system"}}


def test_closed_runtime_binding_preserves_actual_data_and_excludes_invocation_observations():
    from gf_common import rust_player

    first = rust_player.runtime_binding(runtime(), producer_data())
    second = rust_player.runtime_binding(runtime(), producer_data())
    assert first == second
    assert set(first) == {"schema_version", "kind", "runtime", "producer", "sampling_runtime_sha256"}
    assert first["runtime"] == runtime(formation_wait_ms=0.)
    assert rust_player.validate_runtime_binding(first) == first
    first["invocation_id"] = "extra"
    with pytest.raises(ContractError):
        rust_player.validate_runtime_binding(first)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("kind", "shared-service"),
    ("sampling_runtime_sha256", "d" * 64), ("producer", {}),
])
def test_runtime_binding_refuses_wrong_kind_version_sampling_and_producer(field, value):
    from gf_common import rust_player

    bound = rust_player.runtime_binding(runtime(), producer_data())
    bound[field] = value
    with pytest.raises(ContractError):
        rust_player.validate_runtime_binding(bound)


def test_resume_identity_binds_complete_inputs_and_distinct_mode_recipe():
    from gf_common.rust_player import runtime_binding
    from gf_training.rust_player_lineage import resume_identity

    values = {"effective_config": {"run": {"name": "parent"}},
        "identity": {"player_runtime": runtime_binding(runtime(), producer_data()), "code": {"revision": "source"}},
        "execution": {"learner": {"device": "cpu"}},
        "classification": {"purpose": "qualified_game"}, "storage_binding": {"format": "sqlite-group-pack-v1"}}
    original = resume_identity(**values)
    assert original == resume_identity(**deepcopy(values))
    assert original != resume_identity(**values, hpo=True)
    for field in values:
        changed = deepcopy(values)
        changed[field]["different_actual_input"] = True
        assert original != resume_identity(**changed)


def test_operation2_explicit_execution_request_and_operation1_remains_closed():
    from gf_common.domain import digest
    from gf_common.rust_player import validate_operation

    operation = operation_data()
    assert validate_operation(operation) == operation
    changed = deepcopy(operation)
    changed["execution_request"] = {"mode": "resume", "pause_after_update": 2}
    assert digest(changed) != digest(operation)
    legacy = deepcopy(operation)
    legacy.update(schema_version=1, kind="rust-player-operation-v1")
    with pytest.raises(ContractError):
        validate_operation(legacy)
    del legacy["execution_request"]
    assert validate_operation(legacy) == legacy


@pytest.mark.parametrize("execution_request", [
    None, {}, {"mode": "infer", "pause_after_update": 1},
    {"mode": "fresh", "pause_after_update": True},
    {"mode": "resume", "pause_after_update": 0},
    {"mode": "fresh", "pause_after_update": 1, "unrecorded": True},
])
def test_operation2_refuses_missing_unknown_or_boolean_execution_controls(execution_request):
    from gf_common.rust_player import validate_operation

    operation = operation_data()
    operation["execution_request"] = execution_request
    with pytest.raises(ContractError):
        validate_operation(operation)


def test_initial_hpo_pause_requires_the_distinct_prepared_and_completion_records():
    from gf_common.domain import digest
    from gf_common.rust_player import BUDGET_KEYS, validate_operation, validate_prepared
    from gf_training.rust_player_lineage import validate_completion

    operation = operation_data()
    operation["execution_request"]["pause_after_update"] = 0
    with pytest.raises(ContractError):
        validate_operation(operation)
    context = {"campaign_id": "campaign", "operation_id": operation["operation_id"],
        "kind": "training", "run_id": operation["run_id"], "evaluation_id": None}
    operation.update(mode="hpo", operation_type="hpo-train", expected_context=context)
    assert validate_operation(operation) == operation
    identities = {key: operation[key] for key in (
        "operation_id", "run_id", "config_sha256", "runtime_binding_sha256")}
    identities.update(supervision_id="a" * 32, invocation_id="b" * 32)
    admission = {"schema_version": 1, "kind": "rust-player-admission-v1", **identities,
        "nonce": "c" * 32, "supervisor": {"pid": 1, "start_ticks": 1},
        "host": {"pid": 2, "pgid": 2, "start_ticks": 2}, "context": context,
        "budget": dict.fromkeys(BUDGET_KEYS, 1), "deadlines": operation["deadlines"],
        "producer": producer_data(), "global_ticket": "d" * 32}
    prepared = {"schema_version": 2, "kind": "rust-player-initial-pause-prepared-v2", **identities,
        "admission_sha256": digest(admission), "state": "paused", "committed_update": 0,
        "generation": 0, "state_sha256": "e" * 64, "generation_sha256": "e" * 64,
        "artifacts_sha256": "e" * 64, "source_manifest_sha256": None,
        "close_ack_path": "/tmp/close.json", "close_ack_sha256": "e" * 64}
    assert validate_prepared(prepared, admission, operation=operation) == prepared
    with pytest.raises(ContractError):
        validate_prepared(prepared, admission)
    with pytest.raises(ContractError):
        validate_prepared(prepared, admission, operation=operation | {
            "execution_request": {"mode": "fresh", "pause_after_update": 1}})
    with pytest.raises(ContractError):
        validate_prepared(prepared | {"source_manifest_sha256": "e" * 64}, admission, operation=operation)
    completion = {"schema_version": 2, "kind": "rust-player-initial-pause-completion-v2", **identities,
        "operation_sha256": digest(operation), "prepared_sha256": digest(prepared),
        "exit_sha256": "f" * 64, "state_sha256": prepared["state_sha256"],
        "generation_sha256": prepared["generation_sha256"], "source_manifest_sha256": None,
        "state": "paused", "committed_update": 0, "audit_path": "/tmp/audit.json",
        "audit_sha256": "f" * 64, "eligible": False, "parent_metadata_validation": None}
    assert validate_completion(completion) == completion
    with pytest.raises(ContractError):
        validate_completion(completion | {"eligible": True})


def test_checkpoint_time_cap_uses_the_original_admission_clock():
    from gf_cli.rust_player_campaign import _checkpoint_time_cap_boundary
    from gf_common.rust_player import deadline_bits

    operation = operation_data()
    with pytest.raises(ContractError, match="actual exhausted"):
        _checkpoint_time_cap_boundary("a" * 64, operation, 99.0)
    assert _checkpoint_time_cap_boundary("a" * 64, operation, 101.0) == {
        "phase": "checkpoint-time-cap",
        "awaiting_sha256": "a" * 64,
        "admission_bits": operation["deadlines"]["admission_bits"],
        "observed_bits": deadline_bits(101.0),
    }


@pytest.mark.parametrize("record_type,ordinary,hpo_version", [
    ("collection", 10, 11), ("proof", 4, 7), ("generation", 4, 6),
    ("update", 6, 8), ("registry", 5, 8),
])
def test_new_record_shape_is_closed_per_kind_and_never_authorizes_input(record_type, ordinary, hpo_version):
    from gf_training.rust_player_lineage import record_fields, validate_record_shape

    for hpo, version in ((False, ordinary), (True, hpo_version)):
        # Data-shape fixture only. No private issuer accepts this dictionary.
        row = dict.fromkeys(record_fields(record_type, hpo=hpo))
        row["schema_version"] = version
        if record_type == "collection":
            row.update(kind="card-instance-collection", command="collect")
        elif record_type == "proof":
            row.update(kind="rust-player-hpo-source-proof-v1" if hpo else "rust-player-source-proof-v1",
                qualified=False, parent_qualification="not_performed")
        if hpo and record_type in ('collection', 'proof'):
            row['encoder'] = 'canonical-python'
        assert validate_record_shape(row, record_type, hpo=hpo) is row
        for bad_version in (True, version + 100):
            bad = row | {"schema_version": bad_version}
            with pytest.raises(ContractError):
                validate_record_shape(bad, record_type, hpo=hpo)
        with pytest.raises(ContractError):
            validate_record_shape(row | {"unrecorded_authority": True}, record_type, hpo=hpo)
    with pytest.raises(ContractError):
        record_fields("unknown")


def test_actual_saved_policy_needs_generation_write_and_preserves_resume_alias_custody(tmp_path):
    import time

    import torch

    from gf_models.rust_player import RustPlayerInference
    from gf_training.hpo_owned_lineage import OwnedControlWriter
    from gf_training.lineage import run_lock
    from gf_training.rust_player_learning import LocalHpoLearner
    from gf_training.rust_player_lineage import record_fields
    from gf_training.rust_player_registry import RustSavedPolicy
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget

    current = learner_for()
    account = budget()
    local = LocalHpoLearner(current.model, current.training, 17, current.binding, RESUME,
        actions=ACTIONS, rules=RULES, budget=account)
    pair = local.save(tmp_path / "artifacts" / "000000")
    with run_lock(tmp_path) as owner:
        writer = OwnedControlWriter(owner, budget=account, limit=32 * 1024**2)
        policy = RustSavedPolicy.prepare(pair, budget=account, generation=0,
            run_id="component-parent", runtime_binding_sha256="a" * 64,
            actions=ACTIONS, rules=RULES, inference=current.binding)
        with pytest.raises(ContractError, match="committed"):
            policy.require()
        row = dict.fromkeys(record_fields("generation", hpo=True))
        row.update(schema_version=6, origin="producer-owned-v1", encoder="canonical-python",
            run_id="component-parent", generation=0, parent_sha256=None, snapshot_identity=policy.identity,
            inference_sha256=pair.inference_digest, resume_sha256=pair.resume_digest,
            artifact_files=pair.files, update_sha256=None, parent_controls={}, optimizer_steps=0,
            used_decisions=0, collected_decisions=0, accepted_decisions=0, runtime_binding_sha256="a" * 64)
        path = tmp_path / "generations" / "000000.json"
        path.parent.mkdir()
        checksum = writer.write(path, row, deadline=time.monotonic() + 30, immutable=True)
        policy.commit(writer, path, checksum)
        from gf_models.card_instance_artifacts import validate_neural_metadata
        safe, label = validate_neural_metadata(policy.journal_metadata(), actions=ACTIONS, rules=RULES)
        assert (label, safe) == (policy.policy_binding()[0], policy.policy_binding()[1].policy_options)
        alias = pair.snapshot.resume.array(pair.snapshot.resume.names[0])
        before = account.used
        with pytest.raises(BufferError):
            policy.retire_resume()
        assert account.used == before
        del alias
        policy.retire_resume()
        assert account.used < before
        policy.require()
        inference_service = RustPlayerInference(device="cpu", cache_entries=2, cache_weight_bytes=64 * 1024**2)
        rng = torch.get_rng_state().clone()
        inference_service.publish_banks(policy.model, policy.binding, policy.inference, pair.inference_digest)
        assert torch.equal(rng, torch.get_rng_state())
        for name, value in local.learner.network.state_dict().items():
            torch.testing.assert_close(inference_service.registry.networks[pair.inference_digest].state_dict()[name],
                value, rtol=0, atol=0)
        holder = object()
        lease = policy.retain(holder)
        with pytest.raises(ContractError, match="consumer"):
            policy.close()
        lease.close(holder)
        policy.close()
    assert account.used == 0


def test_rust_source_refuses_metadata_without_actual_collector_owner():
    from gf_training.rust_player_source import RustPlayerSource

    with pytest.raises(ContractError, match="actual"):
        RustPlayerSource.project({"schema_version": 10, "status": "complete"})
    with pytest.raises(ContractError, match="actual"):
        RustPlayerSource()


def test_source_capability_copy_is_refused_before_any_payload_access():
    import copy

    from gf_training.rust_player_source import RustPlayerSource
    # A partial cleanup object has no payload or Source authority to copy.
    partial = RustPlayerSource.__new__(RustPlayerSource)
    with pytest.raises(TypeError, match='Source authority'):
        copy.copy(partial)
    with pytest.raises(TypeError, match='Source authority'):
        copy.deepcopy(partial)


@pytest.mark.parametrize('hpo,version', [(False, 6), (True, 8)])
def test_new_update_accepts_only_closed_embedded_device_plan(hpo, version):
    from gf_models.rust_player_device import DeviceUpdatePlan
    from gf_training.rust_player_lineage import record_fields, validate_record_shape
    row = dict.fromkeys(record_fields('update', hpo=hpo))
    row.update(schema_version=version, generation=1)
    plan = DeviceUpdatePlan('actual-source-op', 1, 'cpu', 'cuda:0',
        10000, 10000, 0, 10, 0, 100, 0, 200).record()
    row['device_plan'] = plan
    assert validate_record_shape(row, 'update', hpo=hpo) is row
    with pytest.raises(ContractError):
        validate_record_shape(row | {'device_plan': plan | {'extra': 0}}, 'update', hpo=hpo)
    with pytest.raises(ContractError):
        validate_record_shape(row | {'device_plan': plan | {'generation': 2}}, 'update', hpo=hpo)


def test_cold_plan_data_binds_source_saved_receipts_and_holds_temporary_loans(tmp_path):
    import time
    from types import SimpleNamespace

    from gf_common.domain import digest
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount
    from gf_models.structured_config import resource_estimate
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import _audit_source_device_plan
    from gf_training.rust_player_supervision import write_control_record
    from tests.test_rust_player_device import checkpoint_profile
    from tests.test_rust_player_supervisor import _new_namespace_records
    # Saved receipt DATA with real files/loans; no HostAdmission/Cold capability.
    saved, _ = _new_namespace_records()
    admission = saved['host_admission']
    admission['budget']['managed_cuda_bytes'] = 8 * 1024**3
    profile = checkpoint_profile()
    model = profile.model
    configured = SimpleNamespace(inference=profile.binding, architecture=lambda **kwargs: model)
    loaded = SimpleNamespace(sha256=admission['config_sha256'], config=SimpleNamespace(
        model=configured, training=SimpleNamespace(device='cuda:0'),
        hpo_spec={'schema_version': 6, 'training_recipe': {'tbptt_length': 1}}))
    operation = operation_data()
    operation.update(operation_id=admission['operation_id'], run_id=admission['run_id'], mode='hpo', operation_type='hpo-train',
        config_path=str(tmp_path / 'config.json'), config_sha256=loaded.sha256,
        runtime_binding_sha256=admission['runtime_binding_sha256'], expected_context=admission['context'], deadlines=admission['deadlines'])
    op_path, admit_path = tmp_path / 'source-operation.json', tmp_path / 'source-admission.json'
    op_sha, admit_sha = write_control_record(op_path, operation), write_control_record(admit_path, admission)
    source_id = admission['invocation_id']
    state = {'run_id': admission['run_id'], 'runtime_binding_sha256': admission['runtime_binding_sha256'],
        'identity': {'player_runtime': {'producer': admission['producer']}},
        'invocations': {source_id: {'operation_path': str(op_path), 'operation_sha256': op_sha,
            'admission_path': str(admit_path), 'admission_sha256': admit_sha}}}
    geometry = CudaCheckpointProfile(model, configured.inference, layout=profile.layout)
    logical = resource_estimate(model, configured.inference, suffix_decisions=1, layout=profile.layout)['total_bytes']
    account = RustPlayerDeviceAccount(global_bytes=admission['budget']['managed_cuda_bytes'], learner_device='cuda:0',
        serving_device='cuda:0', learner_profile=geometry, admitted_profiles=(geometry,))
    hold = account.begin_update(logical_bytes=logical, operation_id=operation['operation_id'], generation=1)
    update = {'generation': 1, 'device_plan': hold.plan.record()}
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    loans = []
    try:
        _audit_source_device_plan(update, loaded=loaded, state=state, source_invocation=source_id,
            record_root=tmp_path, pool=pool, deadline=pool.deadline, loans=loans)
        assert loans and pool.budget.used > pool._base_charge
        with pytest.raises(ContractError):
            _audit_source_device_plan(update | {'device_plan': update['device_plan'] | {'operation_id': 'current-resume'}},
                loaded=loaded, state=state, source_invocation=source_id, record_root=tmp_path,
                pool=pool, deadline=pool.deadline, loans=loans)
        assert digest(admission) == admit_sha
    finally:
        for loan in loans:
            loan.close()
        pool.close()
        hold.complete()


@pytest.mark.parametrize('escaped', ['operation', 'admission'])
def test_source_receipt_parent_traversal_refuses_both_paths_before_any_read(tmp_path, monkeypatch, escaped):
    from types import SimpleNamespace

    from gf_training import rust_player_completion as completion
    root = tmp_path / 'session'
    paths = {'operation': root / 'operation.json', 'admission': root / 'admission.json'}
    paths[escaped] = root / '..' / 'foreign' / (escaped + '.json')
    entry = {'operation_path': str(paths['operation']), 'admission_path': str(paths['admission']),
        'operation_sha256': 'a' * 64, 'admission_sha256': 'b' * 64}
    loaded = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(device='cuda:0'),
        model=SimpleNamespace(inference=SimpleNamespace(device='cuda:0'))))
    def forbidden(*args, **kwargs):
        raise AssertionError('receipt read occurred before namespace rejection')
    monkeypatch.setattr(completion, '_read', forbidden)
    with pytest.raises(ContractError, match='record root'):
        completion._audit_source_device_plan({}, loaded=loaded, state={'invocations': {'source': entry}},
            source_invocation='source', record_root=root, pool=SimpleNamespace(budget=None), deadline=1., loans=[])


def test_rust_common_operations_require_actual_same_host_control_and_admission():
    from gf_training.rust_player_learning import RustGameOperations
    with pytest.raises(ContractError, match='actual'):
        RustGameOperations(None, None, None)


def test_model4_rust_numeric_encoding_refuses_metadata_without_actual_source():
    from gf_training.hpo_owned_rollout import encode_rust_batch
    with pytest.raises(ContractError, match='actual'):
        encode_rust_batch({'schema_version': 11}, selected_locations=(), selection={}, numeric_budget_bytes=1)


def test_exact_d_parent_refuses_generic_metadata_or_ordinary_constructor():
    from gf_training.rust_player_completion import ColdFinalizedRustParent
    with pytest.raises(ContractError, match='actual'):
        ColdFinalizedRustParent(None, {'eligible': True, 'state': 'complete'})


def test_resume_audit_refuses_prior_receipts_without_current_live_owners():
    from gf_training.rust_player_restore import RustResumeAudit
    with pytest.raises(ContractError, match='actual'):
        RustResumeAudit(None, None, None, loaded=None, pool=None, writer=None)


def test_adopted_resume_metadata_keeps_same_loan_through_control_last_use():
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_registry import RustRunRegistry
    from gf_training.rust_player_restore import RustResumeAudit
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    audit = RustResumeAudit.__new__(RustResumeAudit)
    audit._closed, audit.scan = False, None
    audit.policies, audit._partial_banks, audit._decodes, audit._held = {}, [], [], {}
    audit.state = {'generation': 1}
    audit.current = {'artifact_files': {'inference.json': 'a' * 64}}
    audit.records, audit.registry, audit.artifact = {1: audit.current}, {'generations': {'1': audit.current}}, None
    loan = pool.budget.reserve(4096)
    audit._loans = [loan]
    registry = RustRunRegistry.__new__(RustRunRegistry)
    registry.state = audit.state
    registry._adopt_restored_metadata(audit)  # DATA handoff only, no admission/capability.
    before = pool.budget.used
    audit.close()
    assert pool.budget.used == before and not loan.closed
    assert audit.registry == {} and registry.records[1]['artifact_files']
    # The actual prepare consumer reads the SAME adopted rows before retirement.
    assert registry.registry['generations']['1'] is registry.records[1]
    rows, state = registry.records, registry.state
    registry._retire_restored_metadata()
    assert not rows and not state and loan.closed
    assert pool.budget.used == before - 4096
    pool.close()
    assert pool.budget.used == 0


@pytest.mark.parametrize('uncertain', [False, True])
def test_late_raw_failure_keeps_actual_fd_and_graph_loan_until_tb_retirement(tmp_path, monkeypatch, uncertain):
    import time
    import weakref
    from types import SimpleNamespace

    import numpy as np

    from gf_common.domain import digest
    from gf_training import card_instance_collection_source, structured_source
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import _audit_raw_originals
    from gf_training.rust_player_source import RustPlayerOriginals
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    raw = RustPlayerOriginals.__new__(RustPlayerOriginals)
    raw.root, raw._descriptor_sha256 = tmp_path, 'd' * 64
    path = tmp_path / 'original.json'
    path.write_text('{}')
    raw._cold_descriptor = OriginalFile(path, pool.deadline)
    descriptor = raw._cold_descriptor.fd
    raw.source = None
    raw._entries, raw.files, raw._closed = {}, {}, False
    raw._reservations = [pool.budget.reserve(4096)]
    backing = SimpleNamespace(descriptor={'database_sha256': 'a', 'seal_id': 'seal'},
        validated={'seal': {'logical_sha256': 'logical'}})
    class DataSource:
        _originals = backing
        def close(self):
            pass
    raw.source = DataSource()
    manifest = {'config': {}, 'code': {}, 'config_hash': 'config', 'runtime_binding_sha256': 'runtime',
        'player_producer': {}, 'capture_authority': {}, 'logical_retirement': {}}
    raw.object = lambda location: manifest
    storage = {'format': 'sqlite-group-pack-v1', 'descriptor_path': str(path),
        'descriptor_sha256': raw._descriptor_sha256, 'database_sha256': 'a', 'seal_id': 'seal', 'logical_sha256': 'logical'}
    proof = {'manifest_sha256': digest(manifest), 'player_producer': {}, 'storage': storage,
        'capture_authority_sha256': digest({}), 'collection_retirement_sha256': digest({})}
    update = {'collection_manifest_sha256': digest(manifest)}
    loaded = SimpleNamespace(effective={}, sha256='config', config=SimpleNamespace(actions=None))
    state = {'identity': {'code': {}}, 'runtime_binding_sha256': 'runtime'}
    monkeypatch.setattr(card_instance_collection_source, 'inspect_rust_game_manifest', lambda *a, **k: ([], []))
    references = []
    def fail_late_raw(*args, **kwargs):
        decoded_graph = np.arange(512, dtype=np.int64)
        references.append(weakref.ref(decoded_graph))
        raise ContractError('late Replay/correspondence failure')
    monkeypatch.setattr(structured_source, '_inspect_collection_content', fail_late_raw)
    before = pool.budget.used
    try:
        with pytest.raises(ContractError, match='late Replay') as failed:
            _audit_raw_originals(raw, loaded=loaded, state=state, previous={}, update=update,
                proof=proof, generation=1, hpo=False, expected=path)
        assert references[0]() is not None and pool.budget.used == before
        import os
        os.fstat(descriptor)
        cleanup = failed.value.cleanup_owner
        if uncertain:
            actual_close = raw._cold_descriptor.close
            def fail_close():
                raise OSError('raw original close uncertain')
            raw._cold_descriptor.close = fail_close
            with pytest.raises(OSError, match='uncertain'):
                cleanup.close()
            assert pool.budget.used == before and not cleanup.closed
            raw._cold_descriptor.close = actual_close
        cleanup.close()
        assert references[0]() is None and pool.budget.used == before - 4096
        cleanup.close()
    finally:
        raw.close()
        pool.close()


def test_failed_control_last_use_retires_stopped_aliases_before_metadata_refund():
    import time
    import weakref

    import numpy as np

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_learning import RustGameOperations
    from gf_training.rust_player_registry import RustRunRegistry
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    registry = RustRunRegistry.__new__(RustRunRegistry)
    registry.state = {'data': np.arange(512)}
    reference = weakref.ref(registry.state['data'])
    registry.records, registry.registry = {}, {}
    loan = pool.budget.reserve(4096)
    registry._restored_metadata_loans = [loan]
    ops = RustGameOperations.__new__(RustGameOperations)
    ops.pool, ops.registry, ops.resume_audit = pool, registry, None
    def fail(data):
        retained_array = data['data']
        assert retained_array.size == 512
        raise ValueError('control consumer failed')
    try:
        fail(registry.state)
    except ValueError as failure:
        ops._failure_traceback = failure.__traceback__
    actual_close = loan.close
    def last_use():
        assert reference() is None
        actual_close()
    loan.close = last_use
    ops.close_control_resources()
    assert pool.budget.used == 0


@pytest.mark.parametrize('stage', ['load', 'restore', 'compare', 'finalize'])
def test_pre_adoption_restore_failure_drops_first_aliases_before_audit_refund(stage):
    import time
    import weakref
    from types import SimpleNamespace

    import numpy as np

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_learning import RustGameOperations
    from gf_training.rust_player_registry import RustRunRegistry
    from gf_training.rust_player_restore import RustResumeAudit
    # Cleanup-only coordinator: I/O fault doubles never create a restored policy,
    # admission or Source. Exercise actual restore_initial's pre-adoption order.
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    audit = RustResumeAudit.__new__(RustResumeAudit)
    audit._closed, audit.policies, audit._partial_banks, audit._decodes, audit._held = False, {}, [], [], {}
    audit.state = {'updates': 1, 'data': np.arange(512)}
    reference = weakref.ref(audit.state['data'])
    audit.current = audit.artifact = None
    audit.records, audit.registry = {}, {}
    loan = pool.budget.reserve(4096)
    audit._loans = [loan]
    registry = RustRunRegistry.__new__(RustRunRegistry)
    registry.state = audit.state  # Actual first-alias relation, before adoption.
    registry.records, registry.registry, registry._restored_metadata_loans = {}, {}, []
    ops = RustGameOperations.__new__(RustGameOperations)
    ops.state, ops.registry, ops.resume_audit, ops.pool = audit.state, registry, audit, pool
    def boundary(name):
        if name == stage:
            funded_array = audit.state['data']
            assert funded_array.size == 512
            raise OSError('pre-adoption ' + name)
    audit.load = lambda local: (boundary('load'), {1: object()})[1]
    local = SimpleNamespace(restore_policy=lambda policy: boundary('restore'))
    ops.verify_restored_state = lambda local, policy: boundary('compare')
    audit.scan = SimpleNamespace(finalize=lambda: boundary('finalize'), close=lambda: None)
    with pytest.raises(OSError, match='pre-adoption') as failed:
        ops.restore_initial(local, audit.state, deadline=pool.deadline)
    ops._failure_traceback = failed.value.__traceback__
    actual_close = loan.close
    def last_use():
        assert not ops.state and not registry.state
        assert reference() is None
        actual_close()
    loan.close = last_use
    ops.close_control_resources()
    assert audit._closed and pool.closed and pool.budget.used == 0


def test_cold_close_drops_control_graphs_before_refunding_same_account(tmp_path):
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import ColdFinalizedRustInvocation
    # Cleanup-only partial owner: no Cold/Source/Parent capability is issued.
    owner = ColdFinalizedRustInvocation.__new__(ColdFinalizedRustInvocation)
    owner.pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    owner.closed, owner._workspace, owner._held = False, None, []
    owner.artifact = owner.audit = owner.completion = {'borrowed': ['data']}
    owner._validated_state = {'borrowed': ['data']}
    owner._control_loans = [owner.pool.budget.reserve(4096)]
    actual = owner._control_loans[0].close
    def check_last_use():
        assert owner._validated_state is None
        assert owner.artifact is owner.audit is owner.completion is None
        actual()
    owner._control_loans[0].close = check_last_use
    owner.close()
    owner.close()
    assert owner.closed and owner.pool.closed and owner.pool.budget.used == 0


def test_cold_uncertain_original_close_retains_control_grant_and_owner(tmp_path):
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import ColdFinalizedRustInvocation
    path = tmp_path / 'original.json'
    path.write_text('{}')
    original = OriginalFile(path, time.monotonic() + 30)
    owner = ColdFinalizedRustInvocation.__new__(ColdFinalizedRustInvocation)
    owner.pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    owner.closed, owner._workspace, owner._held = False, None, [original]
    owner.artifact = owner.audit = owner.completion = owner._validated_state = None
    owner._control_loans = [owner.pool.budget.reserve(4096)]
    before, actual = owner.pool.budget.used, original.close
    def fail():
        raise OSError('unconfirmed original close')
    original.close = fail
    with pytest.raises(OSError, match='unconfirmed'):
        owner.close()
    assert not owner.closed and not owner.pool.closed and owner.pool.budget.used == before
    assert owner._held == [original]
    original.close = actual
    owner.close()
    assert owner.pool.budget.used == 0


@pytest.mark.parametrize('uncertain', [False, True])
def test_provisional_parent_metadata_fd_closes_before_outer_loan_refund(tmp_path, uncertain):
    import os
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import (
        ColdFinalizedRustInvocation,
        ColdFinalizedRustParent,
    )
    path = tmp_path / 'metadata.json'
    path.write_text('{}')
    original = OriginalFile(path, time.monotonic() + 30)
    descriptor, actual_close = original.fd, original.close
    invocation = ColdFinalizedRustInvocation.__new__(ColdFinalizedRustInvocation)
    invocation.pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    invocation.closed, invocation._workspace, invocation._held = False, None, []
    invocation.artifact = invocation.audit = invocation.completion = invocation._validated_state = None
    invocation._control_loans = [invocation.pool.budget.reserve(4096)]
    parent = ColdFinalizedRustParent.__new__(ColdFinalizedRustParent)
    parent._ready, parent._failed, parent._closed = False, True, False
    parent._invocation, parent._metadata_original = invocation, original
    parent._state = parent._files = parent._metadata = b'data'
    invocation._provisional_owner = parent
    before = invocation.pool.budget.used
    def fail():
        raise OSError('metadata FD close uncertain')
    if uncertain:
        original.close = fail
    try:
        if uncertain:
            with pytest.raises(OSError, match='uncertain'):
                invocation.close()
            assert invocation.pool.budget.used == before and not invocation.closed
            assert invocation._provisional_owner is parent
            os.fstat(descriptor)
            original.close = actual_close
        invocation.close()
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert invocation.closed and invocation.pool.budget.used == 0
        assert parent._failed and not parent._ready
        parent.close()  # Idempotent; never Parent→invocation→Parent recursion.
        with pytest.raises(ContractError):
            parent.require()
    finally:
        original.close = actual_close
        original.close()


def test_separate_parent_metadata_failure_is_terminal_and_retains_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gf_common.domain import canonical
    from gf_training import rust_player_completion as completion
    # Exercise only failed post-ACK I/O on a partial, unqualified owner. No fake
    # ColdContinuation, Source or qualified Parent is constructed by this test.
    owner = completion.ColdFinalizedRustParent.__new__(completion.ColdFinalizedRustParent)
    owner._ready, owner._failed, owner._closed = False, False, False
    owner._cleanup_children = []
    owner._state = canonical({'run_id': 'run', 'seed': 17, 'config_sha256': 'a' * 64,
        'updates': 1, 'used_decisions': 64, 'runtime_binding_sha256': 'b' * 64,
        'effective_config': {'hpo_spec': {'effective_settings_sha256': 'c' * 64}}}).encode()
    owner._state_sha256 = owner._generation_sha256 = owner._generations_sha256 = 'd' * 64
    invocation = SimpleNamespace(_parent_directory=tmp_path, artifact=SimpleNamespace(
        files={'inference.json': 'e' * 64}), completion_sha256='f' * 64,
        cold=SimpleNamespace(_control=SimpleNamespace(root=tmp_path)),
        completion={'invocation_id': 'inv', 'prepared_sha256': '1' * 64,
            'exit_sha256': '2' * 64, 'audit_sha256': '3' * 64}, closed=False)
    owner._invocation = invocation
    def fail(path, body):
        raise OSError('metadata publication failed after completion ACK')
    monkeypatch.setattr(completion, 'write_control_record', fail)
    with pytest.raises(OSError, match='after completion ACK') as failed:
        owner._issue_metadata()
    assert failed.value.cleanup_owner is owner and owner._invocation is invocation
    assert owner._failed and not owner._ready and not invocation.closed
    with pytest.raises(ContractError, match='qualified'):
        owner.require()
    assert not list(tmp_path.iterdir())


def test_parent_metadata_original_mutation_refuses_before_returning_data(tmp_path):
    import time
    from types import SimpleNamespace

    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import ColdFinalizedRustParent
    path = tmp_path / 'parent-metadata.json'
    path.write_text('{"actual":"original"}')
    original = OriginalFile(path, time.monotonic() + 30)
    owner = ColdFinalizedRustParent.__new__(ColdFinalizedRustParent)
    owner._closed, owner._failed, owner._ready = False, False, True
    owner._metadata_original = original
    owner._invocation = SimpleNamespace(closed=False, _held=[], completion_sha256='a' * 64,
        cold=SimpleNamespace(_finalized_sha256='a' * 64))
    try:
        path.write_text('{"actual":"mutated!"}')
        with pytest.raises(ValueError, match='identity changed'):
            owner.require()
    finally:
        original.close()


def test_rust_run_reader_refuses_legacy_variant_and_unknown_outer_fields():
    from gf_training.rust_player_lineage import validate_run
    for raw in ({'schema_version': 5, 'run_variant': 'hpo-ppo-owned-v1'},
                {'schema_version': 5, 'run_variant': 'ppo-rust-player-v1', 'eligible': True}):
        with pytest.raises(ContractError):
            validate_run(raw)


def test_saved_actual_run5_requires_resume_identity_and_exact_config(tmp_path):
    import json
    from pathlib import Path

    from gf_cli.config import load_rust_player_effective
    from gf_common.domain import canonical
    from gf_training.rust_player_lineage import validate_run
    # Data-only projection of actual common case e3a1... saved Run5. Only the
    # unrelated per-module identity inventory was omitted; no live authority.
    path = Path(__file__).parent / 'fixtures/rust_player_run5_common_metadata.json'
    state = json.loads(path.read_bytes())
    config_path = tmp_path / 'config.json'
    config_path.write_text(canonical(state['effective_config']))
    loaded = load_rust_player_effective(config_path, expected_sha256=state['config_sha256'])
    assert loaded.effective == state['effective_config']
    assert validate_run(state) is state
    missing = dict(state)
    del missing['resume_identity']
    with pytest.raises(ContractError):
        validate_run(missing)
    with pytest.raises(ContractError):
        validate_run(state | {'unrecorded_eligibility': True})
    with pytest.raises(ContractError):
        validate_run(state | {'config_sha256': '0' * 64})


def test_ordinary_saved_policy_respects_existing_bank_codec_bound_with_larger_source_account(tmp_path):
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_learning import LocalGameLearner
    from gf_training.rust_player_registry import RustSavedPolicy
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES, learner_for

    current = learner_for()
    pool = PackedArtifactPool(max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    local = LocalGameLearner(current.model, 17, current.binding, RESUME, actions=ACTIONS, rules=RULES)
    pair = local.save(tmp_path / '000000')
    policy = RustSavedPolicy.prepare(pair, budget=pool.budget, generation=0, run_id='ordinary-component',
        runtime_binding_sha256='a' * 64, actions=ACTIONS, rules=RULES, inference=current.binding)
    assert policy.inference.names == tuple(sorted(local.learner.network.state_dict()))
    policy.close()
    pool.close()
    assert pool.budget.used == 0


@pytest.mark.parametrize('source_gib,scan_gib', [(8, 8), (16, 12)])
def test_cold_saved_scan_borrows_only_its_supported_share_of_source(source_gib, scan_gib):
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import ColdFinalizedRustInvocation

    deadline = time.monotonic() + 30
    gib = 1024**3
    pool = PackedArtifactPool(max_owned_bytes=source_gib * gib, deadline=deadline)
    # The saved scanning boundary consumes no supervision authority.
    owner = ColdFinalizedRustInvocation.__new__(ColdFinalizedRustInvocation)
    owner.pool = pool
    retained = pool.budget.reserve(4096)
    before = pool.budget.used
    expected = min(scan_gib * gib, source_gib * gib - before)
    try:
        scan = owner._open_saved_scan(deadline)
        assert pool.budget.limit == source_gib * gib
        assert owner._scan is scan and scan.limit == expected
        assert owner._workspace.size == expected
        assert pool.budget.used == before + expected
    finally:
        workspace = getattr(owner, '_workspace', None)
        if workspace is not None:
            workspace.close()
        retained.close()
        pool.close()
