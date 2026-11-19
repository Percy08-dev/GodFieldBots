"""Native Source ownership controls; component fixtures grant no training quality."""

import copy
import time

import pytest

from gf_common.group_pack import StorageSettings
from gf_training.group_pack_resources import StorageBudget
from gf_training.rust_player_storage import RustWriterInventory, _WriterWorkspace
from tests.test_rust_player_storage import fix1_completed_storage  # noqa: F401 -- real native/writer pytest fixture


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
