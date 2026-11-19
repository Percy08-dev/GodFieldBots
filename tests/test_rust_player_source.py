"""Native Source ownership controls; component fixtures grant no training quality."""

import copy
import time

import pytest

from gf_common.group_pack import StorageSettings
from gf_training.group_pack_resources import StorageBudget
from gf_training.rust_player_storage import RustWriterInventory, _WriterWorkspace
from tests.test_rust_player_storage import fix1_completed_storage  # noqa: F401 -- real native/writer pytest fixture


def _failed_operation_source_data(*, hpo=False):
    """Real CPU funding + cleanup receiver, never Source/admission authority."""
    import weakref
    from types import SimpleNamespace

    import numpy as np

    from gf_models.rust_player_device import RustPlayerDeviceAccount
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_learning import RustGameOperations, RustHpoOperations
    from gf_training.rust_player_source import RustPlayerSource
    from tests.test_rust_player_device import checkpoint_profile
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    local = SimpleNamespace(device_account=account)
    hold = account.begin_update(logical_bytes=10**9, operation_id='cleanup-data', generation=1, owner=local)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    source = RustPlayerSource.__new__(RustPlayerSource)
    source.state, source._rollout, source._records = 'FAILED', np.arange(512), ()
    reference = weakref.ref(source._rollout)
    source.manifest = {'logical_retirement': {'service_settlement_sha256': 'a' * 64}}
    source._numeric, source._numeric_originals = None, {}
    source._proof_json = source._numeric_metadata = ''
    source._episode_digests, source.numeric_files = (), {}
    source.collection = SimpleNamespace(require=lambda *args: None, close=lambda: None)
    source._parent_lease = SimpleNamespace(close=lambda owner: None)
    source._charges = [pool.budget.reserve(4096)]
    source._pool_hold = pool.hold_close(source)
    registry = SimpleNamespace(local=local, pending=source, hpo=hpo)
    source.registry = registry
    ops = (RustHpoOperations if hpo else RustGameOperations).__new__(RustHpoOperations if hpo else RustGameOperations)
    ops.registry, ops.device_account = registry, account
    return source, ops, hold, pool, reference


@pytest.mark.parametrize('nested_uncertain', [False, True])
def test_projection_failure_keeps_source_funding_through_external_tb_clear(tmp_path, monkeypatch, nested_uncertain):
    import traceback

    import numpy as np

    from gf_common.ram_bank import RamBankBuilder
    from gf_training.rust_player_source import RustPlayerSource
    source, ops, hold, pool, reference = _failed_operation_source_data()
    bank = RamBankBuilder({'value': np.arange(512, dtype=np.float32)}, pool.budget).seal()
    monkeypatch.setattr(RustPlayerSource, 'project', classmethod(lambda cls, collection: source))
    monkeypatch.setattr(RustPlayerSource, 'persist', lambda self, path: {})
    monkeypatch.setattr(RustPlayerSource, 'proof', property(lambda self: {}))
    before = pool.budget.used
    def numerical_failure():
        actual_rollout = source._rollout
        actual_view = bank.array('value')
        assert actual_rollout.size == 512
        assert actual_view.size == 512
        try:
            raise ValueError('original numerical failure')
        except ValueError as error:
            error.cleanup_owner = bank
            hold.fail(error)
            error.cleanup_owner = source  # Actual outer Source overwrite occurs later.
            raise
    with (pytest.raises(ValueError, match='original numerical') as failed,
            ops.project(source.collection, object(), tmp_path, None, pool.deadline)):
        numerical_failure()
    traceback.clear_frames(failed.value.__traceback__)
    assert failed.value is hold.error and failed.value.source_cleanup_owner is source
    assert source.registry.pending is source
    assert source.state != 'CLOSED' and pool.budget.used == before and reference() is not None
    with pytest.raises(ValueError):
        hold.retire_failed(gpu_settled=False)
    with pytest.raises(BufferError, match='operation'):
        ops.close_batch()
    assert pool.budget.used == before
    if nested_uncertain:
        actual_close = bank.close
        def uncertain():
            raise OSError('actual nested close uncertain')
        monkeypatch.setattr(bank, 'close', uncertain)
        with pytest.raises(OSError, match='uncertain') as cleanup:
            hold.retire_failed(gpu_settled=True)
        assert hold.retirement_errors[-1] is cleanup.value
        assert pool.budget.used == before and hold.state == 'FAILED_HELD'
        with pytest.raises(BufferError):
            ops.close_batch()
        assert not source._charges[0].closed
        monkeypatch.setattr(bank, 'close', actual_close)
    hold.retire_failed(gpu_settled=True)
    assert not source._charges[0].closed  # Hold retirement never refunds outer Source.
    ops.close_batch()
    assert source.state == 'CLOSED' and reference() is None
    pool.close()
    assert pool.budget.used == 0


def test_hpo_close_batch_after_refused_retirement_preserves_actual_source_charge():
    import traceback

    source, ops, hold, pool, reference = _failed_operation_source_data(hpo=True)
    before = pool.budget.used
    def fail():
        actual_data_graph = source._rollout
        assert actual_data_graph.size == 512
        try:
            raise ValueError('held data')
        except ValueError as error:
            hold.fail(error)
    fail()
    traceback.clear_frames(hold.error.__traceback__)
    with pytest.raises(ValueError):
        hold.retire_failed(gpu_settled=False)
    with pytest.raises(BufferError, match='operation'):
        ops.close_batch()
    assert source.state != 'CLOSED' and pool.budget.used == before and reference() is not None
    hold.retire_failed(gpu_settled=True)
    ops.close_batch()
    assert reference() is None
    pool.close()
    assert pool.budget.used == 0


def test_no_account_source_cleanup_keeps_existing_refund_behavior():
    source, ops, hold, pool, reference = _failed_operation_source_data()
    hold.complete()
    source.registry.local.device_account = None
    ops.close_batch()
    assert source.state == 'CLOSED' and reference() is None
    pool.close()
    assert pool.budget.used == 0


def test_inventory_immutable_rows_and_actual_last_owner_workspace_retirement():
    account = StorageBudget(StorageSettings(), time.monotonic() + 30)
    workspace = _WriterWorkspace(account.reserve(4096))
    inventory = RustWriterInventory((("actual/indexed-member", 17, "a" * 64),), workspace)
    rows = inventory.rows
    assert rows == (("actual/indexed-member", 17, "a" * 64),)
    with pytest.raises(TypeError):
        rows[0] = ("changed", 1, "b" * 64)
    with pytest.raises(TypeError):
        copy.copy(inventory)
    workspace.close()
    assert account.used == 4096
    del rows
    inventory.close()
    inventory.close()
    assert account.used == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = inventory.rows


def test_inventory_exception_finalization_keeps_original_grant_until_last_owner():
    account = StorageBudget(StorageSettings(), time.monotonic() + 30)
    workspace = _WriterWorkspace(account.reserve(4096))
    first = RustWriterInventory((("first", 1, "a" * 64),), workspace)
    second = RustWriterInventory((("second", 2, "b" * 64),), workspace)
    workspace.close()
    first.close()
    assert account.used == 4096
    del second
    assert account.used == 0


def test_actual_native_writer_descriptor_opens_same_original_journal_data(fix1_completed_storage):  # noqa: F811 -- imported pytest fixture
    import hashlib
    import json

    from gf_common.preprocessing import PreparationConfig
    from gf_training import rust_player_source
    from gf_training.group_pack_pool import PackedArtifactPool

    writer, runtime, _ = fix1_completed_storage
    locator = writer.seal()
    writer.join()
    pending = writer.take_pending_owner()
    assert runtime.join()["joined"]
    pool = PackedArtifactPool(max_owned_bytes=512 * 1024**2, deadline=writer.deadline)
    owner = rust_player_source.RustPlayerOriginals(
        pending, pool=pool, settings=PreparationConfig(), deadline=writer.deadline
    )
    try:
        assert owner.budget is pool.budget
        raw = owner.read(owner.root / "planned/player-1.jsonl")
        rows = [json.loads(row) for row in raw.splitlines()]
        assert rows[0]["schema_version"] == 6
        assert rows[-1]["end_kind"] == "truncated"
        assert owner.files["planned/player-1.jsonl"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert owner.source._originals.descriptor_file.path == locator.path
        pending.recheck()
    finally:
        del raw, rows
        owner.close()
        pending.close()
        pool.close()
    assert pool.budget.used == 0


@pytest.mark.parametrize("fault", ["duplicate", "unplanned", "controls", "partial", "admitted"])
def test_deferred_actual_context_binding_refuses_wrong_or_late_originals(tmp_path, fault):
    from contextlib import suppress
    from dataclasses import replace

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.rust_player_storage import (
        RustPlayerStorage,
        StorageGroup,
        episode_encoding_plan,
    )
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config
    from tests.test_rust_player_storage import journal_context

    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram",
        budget_bytes=512 * 1024**2, queue_bytes=4 * 1024**2)
    account = StorageBudget(settings, time.monotonic() + 30)
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    writer = None
    try:
        with managed(tmp_path / "server") as (startup, receipt):
            specs = tuple(game_spec(startup, receipt, episode=f"pair-{i}") for i in range(2))
            contexts = tuple(journal_context(spec) for spec in specs)
            writer = RustPlayerStorage(runtime, tmp_path / "collection", run_id="run",
                contexts=(), groups=(StorageGroup("pair", ("pair-0", "pair-1"), {}),),
                settings=settings, workspace=account, workspace_bytes=256 * 1024**2,
                deadline=time.monotonic() + 30, cleanup_deadline=time.monotonic() + 35,
                deferred_contexts=True, context_plans=tuple(episode_encoding_plan(c) for c in contexts))
            submitted = contexts
            if fault == "duplicate":
                writer.bind_contexts(contexts)
            elif fault == "unplanned":
                submitted = (replace(contexts[0], episode_id="foreign"), contexts[1])
            elif fault == "controls":
                submitted = (replace(contexts[0], trial="changed"), contexts[1])
            elif fault == "partial":
                submitted = contexts[:1]
            elif fault == "admitted":
                runtime.start([specs[0]])
                assert runtime.episode_admission_attempted("pair-0")
            with pytest.raises(RuntimeError, match="storage failed"):
                writer.bind_contexts(submitted)
            assert not writer._sealed
    finally:
        runtime.cancel("deferred-context negative control complete")
        runtime.join()
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()
            assert not writer._thread.is_alive()
        assert account.used == 0


@pytest.mark.parametrize("fault", ["empty", "unexecuted", "profile"])
def test_fix1_deferred_binding_validates_all_executed_receipts_atomically(tmp_path, fault):
    from contextlib import suppress
    from dataclasses import replace

    from gf_common.card_instance import corrected_card_instance_contract
    from gf_common.domain import clone
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.rust_player_storage import (
        RustPlayerStorage,
        StorageGroup,
        episode_encoding_plan,
    )
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config
    from tests.test_rust_player_storage import journal_context

    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram",
        budget_bytes=512 * 1024**2, queue_bytes=4 * 1024**2)
    account = StorageBudget(settings, time.monotonic() + 30)
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    writer = None
    try:
        with managed(tmp_path / "server") as (startup, receipt):
            contexts = tuple(journal_context(game_spec(startup, receipt, episode=f"pair-{i}"))
                for i in range(2))
            evidence = clone(contexts[1].runtime_evidence)
            if fault == "empty":
                evidence = {}
            elif fault == "unexecuted":
                evidence["executed"] = None
            else:
                contexts = (contexts[0], replace(contexts[1],
                    actions=replace(contexts[1].actions, action_contract=corrected_card_instance_contract())))
            contexts = (contexts[0], replace(contexts[1], runtime_evidence=evidence))
            writer = RustPlayerStorage(runtime, tmp_path / "collection", run_id="run",
                contexts=(), groups=(StorageGroup("pair", ("pair-0", "pair-1"), {}),),
                settings=settings, workspace=account, workspace_bytes=256 * 1024**2,
                deadline=time.monotonic() + 30, cleanup_deadline=time.monotonic() + 35,
                deferred_contexts=True, context_plans=tuple(episode_encoding_plan(c) for c in contexts))
            with pytest.raises(RuntimeError, match="storage failed"):
                writer.bind_contexts(contexts)
            assert writer.encoders == {}
            assert not runtime.episode_admission_attempted("pair-0")
            assert not runtime.episode_admission_attempted("pair-1")
    finally:
        runtime.cancel("invalid deferred receipt fixture cleanup")
        runtime.join()
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()
        assert account.used == 0


@pytest.mark.parametrize("baseline_kind,wrong_cost", [
    ("random", False), ("heuristic", False), ("refresh", False), ("random", True),
])
def test_mixed_actor_uses_own_numeric_costs_and_canonical_journal_recipe(tmp_path, baseline_kind, wrong_cost):
    """Real mixed native capture; component data only, no new Source capability."""
    import random
    from dataclasses import asdict, replace

    from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
    from gf_common.domain import ContractError, canonical, digest
    from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
    from gf_common.structured import EventBank
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_wire import game_generated_dict_v3, parse_delta, parse_frame
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.card_instance_baselines import CardInstanceBaseline
    from gf_models.refresh_heuristic import RefreshHeuristicPolicy
    from gf_models.structured_artifacts import save_game_inference_v3
    from gf_models.structured_binding import ConfiguredNeuralModelV3
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.rust_player_storage import (
        CanonicalEpisodeEncoder,
        PolicyJournalContext,
        RecordingArtifacts,
        RecordingBudget,
        decode_original,
    )
    from tests.test_rust_player_runtime import actual_service, game_spec, managed, runtime_config
    from tests.test_rust_player_storage import journal_context

    service, network = actual_service()
    with managed(tmp_path / "server") as (startup, receipt):
        spec = game_spec(startup, receipt)
        spec.update(max_decisions=4, clock=[0.] * 4096)
        context = journal_context(spec)
        directory = tmp_path / "policy"
        model, binding = network.config, network.binding
        save_game_inference_v3(directory, network, model, digest(asdict(context.actions)),
            action_contract=context.actions.action_contract, rules=context.rules)
        configured = ConfiguredNeuralModelV3(
            **{key: getattr(model, key) for key in model.__dataclass_fields__},
            checkpoint=str(directory), inference=binding)
        artifacts = RetainedGameArtifacts((configured,), context.actions, rules=context.rules,
            max_owned_bytes=256 * 1024**2, preflight_deadline=spec["deadline"],
            finalization_deadline=spec["cleanup_deadline"])
        runtime = RustPlayerRuntime(runtime_config() | {"runtime_identity": service.runtime}, service=service)
        sink = RecordingArtifacts(RecordingBudget(runtime.recording_account(), spec["deadline"]))
        encoder = None
        try:
            label, execution = artifacts.binding(configured)
            caps, costs, _, _ = own_game_worker_settings_v3("mlp", context.actions,
                execution.policy_options, rules=context.rules)
            baseline_id = "heuristic_refresh" if baseline_kind == "refresh" else baseline_kind
            options = asdict(RefreshOptions()) if baseline_kind == "refresh" else None
            base_caps, base_costs, _, _ = own_game_worker_settings_v3(
                baseline_id, context.actions, options, rules=context.rules)
            assert asdict(costs) != asdict(base_costs)
            neural = {"kind": "mlp", "seed": 17, "checkpoint": "a" * 64, "epoch": 0,
                "limits": None, "attack_threshold": 12, "numeric_costs": asdict(costs)}
            # Match the real immutable service's auto limits in both actual contexts.
            neural["limits"] = {"max_decisions": model.temporal.max_decisions,
                "max_episode_events": model.event.max_episode, "max_interval_events": model.event.max_interval,
                "max_candidates": binding.max_candidates, "max_positions": model.max_positions,
                "aggregate_window": model.event.aggregate_window}
            baseline = {"kind": baseline_kind, "seed": 18, "checkpoint": None, "epoch": 0,
                "limits": None, "attack_threshold": 12, "numeric_costs": asdict(base_costs)}
            baseline_journal = {"policy": refresh_policy_label(context.actions, context.rules, RefreshOptions())
                if options is not None else baseline_id + "-card-instance-v2", "model_id": baseline_id,
                "caps": asdict(base_caps), "numeric_costs": asdict(base_costs), "sampling_seed": 18}
            if options is not None:
                baseline_journal["baseline_options"] = options
            expected_baseline = (RefreshHeuristicPolicy(context.actions, rules=context.rules, options=RefreshOptions())
                if options is not None else CardInstanceBaseline(baseline_kind, context.actions, rules=context.rules))
            expected_baseline.reset(18)
            spec["policies"] = [neural, baseline]
            policies = (PolicyJournalContext(neural, {"policy": label, "model_id": "mlp",
                "caps": asdict(caps), "numeric_costs": asdict(NumericCosts()) if wrong_cost else asdict(costs),
                "sampling_seed": 17, "neural": artifacts.journal_metadata(label)}),
                PolicyJournalContext(baseline, baseline_journal))
            encoder = CanonicalEpisodeEncoder(replace(context, policies=policies, clock=spec["clock"]), sink)
            runtime.start([spec])
            seen = set()
            banks = {viewer: EventBank(viewer, context.episode_id, layout=layout) for viewer in (1, 2)}
            while encoder.terminal is None:
                bundle = runtime.next_storage(spec["deadline"])
                if wrong_cost:
                    with pytest.raises(ContractError, match="numeric costs"):
                        encoder.consume(bundle)
                    runtime.storage_ack(bundle, failure="wrong per-policy cost context")
                    del bundle
                    break
                data = decode_original(bundle)
                encoder.consume(bundle)
                if bundle.kind == "pending":
                    actor = data["actor"]
                    seen.add(actor)
                    bank = banks[actor]
                    bank.append(parse_delta(data["delta"], layout=layout))
                    frame = parse_frame(data["frame"], layout=layout)
                    rng = random.Random()
                    state = data["search_rng_before"]
                    rng.setstate((state[0], tuple(state[1]), state[2]))
                    expected = PublicCandidateGeneratorV3(context.actions, rules=context.rules,
                        clock=lambda: 0.).generate_game(frame, bank.view(), rng, costs if actor == 1 else base_costs)
                    assert canonical(game_generated_dict_v3(expected,
                        action_contract=context.actions.action_contract, rules=context.rules)) == canonical(data["generated"])
                    assert data["generated"]["report"]["numeric_costs_hash"] == digest(asdict(costs if actor == 1 else base_costs))
                    if actor == 2:
                        selection = expected_baseline.select_frame(frame, bank.view(), expected)
                        assert selection.index == data["selected"]
                        assert selection.validate(expected.table, frame) == data["command"]["action"]
                runtime.storage_ack(bundle)
                del bundle
            if not wrong_cost:
                runtime.wait(spec["deadline"])
                assert seen == {1, 2}
        finally:
            runtime.cancel("mixed-cost component cleanup")
            runtime.join()
            if encoder is not None:
                encoder.retire()
            sink.close()
            artifacts.release()
@pytest.mark.parametrize('kind', ['cold', 'resume', 'metadata'])
@pytest.mark.parametrize('uncertain', [False, True])
def test_nested_original_constructor_failure_retains_actual_fd_before_tb_clear(tmp_path, monkeypatch, kind, uncertain):
    import os
    import time
    import weakref
    from types import SimpleNamespace

    from gf_common.domain import canonical
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import (
        ColdFinalizedRustInvocation,
        ColdFinalizedRustParent,
    )
    from gf_training.rust_player_source import RustColdOriginals, RustResumeOriginals
    # Partial cleanup owners ONLY: no Cold/Pending/Source/qualified Parent cap.
    # Enter the production file-I/O portion after authority checks, and force
    # the real OriginalFile constructor to fail before assignment can finish.
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    loan = pool.budget.reserve(4096)
    outer = ColdFinalizedRustInvocation.__new__(ColdFinalizedRustInvocation)
    outer.closed, outer.pool, outer._workspace, outer._held = False, pool, None, []
    outer.artifact = SimpleNamespace(files={'inference.json': 'e' * 64})
    outer.audit, outer._validated_state = None, None
    outer._control_loans = [loan]
    outer._parent_directory, outer.completion_sha256 = tmp_path, 'f' * 64
    outer.cold = SimpleNamespace(_control=SimpleNamespace(root=tmp_path))
    outer.completion = {'invocation_id': 'inv', 'prepared_sha256': '1' * 64,
        'exit_sha256': '2' * 64, 'audit_sha256': '3' * 64}
    if kind == 'metadata':
        owner = ColdFinalizedRustParent.__new__(ColdFinalizedRustParent)
        owner._invocation, owner._metadata_original = outer, None
        owner._closed, owner._failed, owner._ready = False, False, False
        owner._state = canonical({'run_id': 'run', 'seed': 17, 'config_sha256': 'a' * 64,
            'updates': 1, 'used_decisions': 64, 'runtime_binding_sha256': 'b' * 64,
            'effective_config': {'hpo_spec': {'effective_settings_sha256': 'c' * 64}}}).encode()
        owner._state_sha256 = owner._generation_sha256 = owner._generations_sha256 = 'd' * 64
        owner._cleanup_children = []
        enter = owner._issue_metadata
    else:
        owner = (RustColdOriginals if kind == 'cold' else RustResumeOriginals).__new__(
            RustColdOriginals if kind == 'cold' else RustResumeOriginals)
        path = tmp_path / 'descriptor.json'
        path.write_text('{}')
        owner.pool, owner.budget, owner.deadline = pool, pool.budget, pool.deadline
        owner.root, owner._descriptor_path, owner._descriptor_sha256 = tmp_path, path, 'a' * 64
        owner.source = owner._cold_descriptor = None
        owner.files, owner._entries, owner._reservations, owner._cleanup_children = {}, {}, [], []
        owner._closed = False
        enter = owner._open_sealed_originals
    descriptors, references = [], []
    actual_close = OriginalFile.close
    def fail_hash(original):
        descriptors.append(original.fd)
        references.append(weakref.ref(original))
        raise OSError('actual constructor hash failure')
    def fail_close(original):
        raise OSError('actual inner close uncertain')
    monkeypatch.setattr(OriginalFile, 'hash', fail_hash)
    monkeypatch.setattr(OriginalFile, 'close', fail_close)
    try:
        with pytest.raises(OSError, match='constructor hash') as failed:
            enter()
        outer._provisional_owner = failed.value.cleanup_owner
        outer._failure_traceback = failed.value.__traceback__
        before = pool.budget.used
        if uncertain:
            with pytest.raises(OSError, match='close uncertain'):
                outer.close()
            assert references[0]() is not None and not outer.closed
            assert pool.budget.used == before and not loan.closed
            os.fstat(descriptors[0])
        monkeypatch.setattr(OriginalFile, 'close', actual_close)
        outer.close()
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        assert loan.closed and outer.closed and pool.budget.used == 0
        if kind == 'metadata':
            assert owner._failed and not owner._ready
    finally:
        # RED probes must not leak the real kernel FD when the old wrapper loses it.
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError:
                pass
            else:
                os.close(descriptor)
