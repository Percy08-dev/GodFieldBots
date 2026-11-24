"""Actual database-thread and native-original encoding behavior."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from gf_common.group_pack import MIB, PackMember, StorageSettings, encode_pack, unpack_raw


def test_database_owner_same_thread_actual_seal_and_original_reopen(tmp_path, monkeypatch):
    from gf_training import group_pack_worker as module

    calls = []
    actual_new = module.new_database

    def create(*args, **kwargs):
        calls.append(threading.get_ident())
        return actual_new(*args, **kwargs)

    monkeypatch.setattr(module, "new_database", create)
    directory = tmp_path / "collection"
    directory.mkdir()
    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram")
    values = asdict(settings)
    values.update(db_root=str(settings.db_root), ram_root=str(settings.ram_root))
    pack = encode_pack((PackMember("journal", "group/e1", b"original"),))
    deadline = time.monotonic() + 30

    def run():
        owner = module.GroupPackDatabase(max_owned_bytes=512 * MIB)
        ordinal = 0

        def command(kind, payload=None, **fields):
            nonlocal ordinal
            result = owner.execute(
                {"version": 1, "request_id": ordinal, "command": kind, **fields}, payload
            )
            ordinal += 1
            return result

        command(
            "init",
            directory=str(directory),
            run_id="thread-run",
            settings=values,
            deadline=deadline,
        )
        command(
            "submit",
            pack,
            group_id="group",
            raw_bytes=len(unpack_raw(pack)),
            record={},
            episodes=[{"episode_id": "e1", "game_id": "actual", "record": {}}],
        )
        command("control", b"control-original", name="manifest.json")
        # The real default SQLite thread check proves ownership has not escaped.
        db = owner.db
        descriptor = command("seal", publication="pending")
        assert owner.db is None and owner.database_original.fd == -1
        return descriptor, db, threading.get_ident()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="storage-proof") as pool:
        descriptor, closed_db, identity = pool.submit(run).result()
    assert calls == [identity] and identity != threading.get_ident()
    with pytest.raises(sqlite3.ProgrammingError, match="same thread"):
        closed_db.execute("SELECT 1")
    original = module.OriginalFile(directory / module.DATABASE, deadline)
    try:
        inspected = module.inspect_packed(
            directory / module.DATABASE,
            deadline=deadline,
            max_owned_bytes=512 * MIB,
            allow_pending=True,
            without_descriptor=True,
            database_original=original,
        )
        assert inspected["database_sha256"] == descriptor["database_sha256"]
        assert inspected["validated"]["seal"]["seal_id"] == descriptor["seal_id"]
    finally:
        original.close()


def test_one_native_recording_account_external_resize_and_explicit_release():
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from tests.test_rust_player_runtime import runtime_config

    runtime = RustPlayerRuntime(runtime_config())
    try:
        first = runtime.recording_account()
        second = runtime.recording_account()
        assert first.same_account(second) and first.used == first.native_bytes == 0
        lease = first.reserve(first.limit - 7)
        assert second.external_bytes == first.limit - 7
        with pytest.raises((ValueError, MemoryError), match="budget"):
            second.reserve(8)
        lease.resize(12)
        assert first.used == first.external_bytes == 12 and first.native_bytes == 0
        lease.close()
        lease.close()
        assert first.used == 0 and first.peak == first.limit - 7
        with pytest.raises(ValueError, match="closed"):
            lease.resize(1)
    finally:
        runtime.cancel("account test complete")
        runtime.join()


def test_database_transitional_readonly_survives_failed_original_close(tmp_path, monkeypatch):
    from gf_training import group_pack_worker as module

    originals = []
    injected = False
    actual = module.OriginalFile

    class Observed(actual):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            originals.append(self)

        def close(self):
            nonlocal injected
            if self is originals[0] and len(originals) == 2 and not injected:
                injected = True
                raise OSError("original close failed before release")
            return super().close()

    monkeypatch.setattr(module, "OriginalFile", Observed)
    directory = tmp_path / "collection"
    directory.mkdir()
    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram")
    values = asdict(settings) | {
        "db_root": str(settings.db_root),
        "ram_root": str(settings.ram_root),
    }

    def run():
        owner = module.GroupPackDatabase(max_owned_bytes=512 * MIB)
        owner.execute(
            {
                "version": 1,
                "request_id": 0,
                "command": "init",
                "directory": str(directory),
                "run_id": "failed-transition",
                "settings": values,
                "deadline": time.monotonic() + 30,
            }
        )
        pack = encode_pack((PackMember("journal", "g/e", b"original"),))
        owner.execute(
            {
                "version": 1,
                "request_id": 1,
                "command": "submit",
                "group_id": "g",
                "raw_bytes": len(unpack_raw(pack)),
                "record": {},
                "episodes": [],
            },
            pack,
        )
        with pytest.raises(OSError, match="original close failed"):
            owner.execute(
                {
                    "version": 1,
                    "request_id": 2,
                    "command": "seal-producer",
                    "publication": "pending",
                }
            )
        failures = owner.fail()
        try:
            assert failures == []
            assert len(originals) == 2 and all(original.fd == -1 for original in originals)
        finally:
            for original in originals:
                original.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run).result()


def journal_context(spec):
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import CardInstanceRules
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from gf_training.rust_player_storage import EpisodeEncodingContext, PolicyJournalContext

    actions, rules = action_config_v3(spec["actions"]), CardInstanceRules(**spec["rules"])
    policies = []
    for policy in spec["policies"]:
        native = {"checkpoint": None, "epoch": 0, "limits": None, "attack_threshold": 12} | policy
        caps, costs, _, _ = own_game_worker_settings_v3(policy["kind"], actions, None, rules=rules)
        policies.append(
            PolicyJournalContext(
                native,
                {
                    "policy": policy["kind"] + "-card-instance-v2",
                    "model_id": policy["kind"],
                    "caps": asdict(caps),
                    "numeric_costs": asdict(costs),
                    "sampling_seed": policy["seed"],
                },
            )
        )
    return EpisodeEncodingContext(
        spec["run"],
        spec["trial"],
        spec["episode"],
        actions,
        rules,
        spec["numeric_costs"],
        spec["runtime_evidence"],
        tuple(policies),
        spec["deadline"],
        spec.get("clock"),
    )


@pytest.mark.parametrize("decisions", [8])
def test_actual_native_originals_encode_journal6_and_both_replay_views(
    tmp_path, monkeypatch, decisions
):
    import asyncio
    import json
    from dataclasses import replace
    from pathlib import Path

    from gf_common.card_instance import (
        corrected_card_instance_contract,
        shared_card_instance_contract,
    )
    from gf_common.structured import EventBank
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_simulator_client.process import ManagedServer
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.card_instance_journal import read_card_instance_journal
    from gf_training.rust_player_storage import (
        CanonicalEpisodeEncoder,
        RecordingArtifacts,
        RecordingBudget,
    )
    from tests.test_rust_player_runtime import assert_source_match, game_spec, runtime_config

    simulator = Path("/tmp/gf-issue72-shared-inference")
    binary = Path("/tmp/gfb-shared-simulator-072dabb/target/release/gf-test-server")
    if not binary.exists():
        pytest.skip("approved external Simulator release unavailable")
    assert_source_match()

    async def run():
        evidence = CardInstanceRuntimeEvidence(
            binary, simulator, contract=shared_card_instance_contract()
        )
        evidence.preflight()
        server = ManagedServer(
            str(binary),
            str(Path(__file__).parents[1] / "configs/test-server.toml"),
            "00000000-0000-0000-0000-000000000017",
            1,
            tmp_path / "server",
        )
        runtime = RustPlayerRuntime(runtime_config())
        sink = RecordingArtifacts(
            RecordingBudget(runtime.recording_account(), time.monotonic() + 30)
        )
        reference_sink = RecordingArtifacts(sink.budget)
        try:
            startup = await server.start()
            spec = game_spec(startup, evidence.verify_executed(server.process))
            spec["max_decisions"] = decisions
            context = journal_context(spec)
            encoder = CanonicalEpisodeEncoder(context, sink)
            assert encoder._native_candidates
            old_profile = replace(context, actions=replace(
                context.actions, action_contract=corrected_card_instance_contract(),
            ))
            assert not CanonicalEpisodeEncoder(old_profile, reference_sink)._native_candidates
            reference = CanonicalEpisodeEncoder(context, reference_sink)
            reference._native_candidates = False  # compare both codecs on the same exact native owner
            runtime.start([spec])
            account_checked = False

            def unexpected_snapshot(*args, **kwargs):
                pytest.fail("live native pending rebuilt the validated event bank")

            while encoder.terminal is None:
                bundle = runtime.next_storage(time.monotonic() + 20)
                if not account_checked:
                    account = runtime.recording_account()
                    native_used = account.native_bytes
                    assert native_used >= bundle.reserved_bytes
                    loan = account.reserve(account.limit - account.used)
                    with pytest.raises(MemoryError, match="budget"):
                        account.reserve(1)
                    loan.close()
                    assert account.used == native_used
                    account_checked = True
                if bundle.kind == "pending":
                    with monkeypatch.context() as live_pending:
                        live_pending.setattr(EventBank, "view", unexpected_snapshot)
                        encoder.consume(bundle)
                else:
                    encoder.consume(bundle)
                reference.consume(bundle)
                runtime.storage_ack(bundle)
                del bundle
            runtime.wait(time.monotonic() + 10)
            runtime.join()
            members = sink.members()
            files = {member.name: member.data for member in members}
            assert files == {member.name: member.data for member in reference_sink.members()}
            assert sum(name.endswith(".jsonl") for name in files) == 2
            assert sum("replay-player-" in name for name in files) == 2
            directory = tmp_path / context.episode_id
            directory.mkdir()
            for name, data in files.items():
                (tmp_path / name).write_bytes(data)
            for viewer in (1, 2):
                data = read_card_instance_journal(
                    directory / f"player-{viewer}.jsonl",
                    actions=context.actions,
                    rules=context.rules,
                )
                assert data.header["bank_episode"] == context.episode_id
                assert len(data.decisions) == (0 if decisions == 0 else 4)
                rows = [
                    json.loads(line)
                    for line in files[f"{context.episode_id}/player-{viewer}.jsonl"].splitlines()
                ]
                assert rows[-1]["end_kind"] == "truncated" and rows[-1]["outcome"] is None
            del members, files, data
            encoder.retire()
            del encoder
            reference.retire()
            reference_sink.close()
            sink.close()
            assert runtime.recording_account().used == 0
        finally:
            runtime.cancel("encoder test complete")
            runtime.join()
            reference_sink.close()
            sink.close()
            await server.stop()

    asyncio.run(run())


@pytest.mark.parametrize("deferred", [False, True])
def test_real_storage_thread_stages_partial_pair_and_seals_originals(tmp_path, monkeypatch, deferred):
    import json

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training import group_pack_worker as database_module
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.group_pack_source import PackedOriginals
    from gf_training.rust_player_storage import (
        RustPlayerStorage,
        StorageGroup,
        episode_encoding_plan,
    )
    from tests.test_rust_player_runtime import (
        assert_source_match,
        game_spec,
        managed,
        runtime_config,
    )

    assert_source_match()
    create_threads = []
    actual = database_module.new_database

    def create(*args, **kwargs):
        create_threads.append(threading.get_ident())
        return actual(*args, **kwargs)

    monkeypatch.setattr(database_module, "new_database", create)
    settings = StorageSettings(
        db_root=tmp_path / "catalog",
        ram_root=tmp_path / "ram",
        budget_bytes=512 * MIB,
        queue_bytes=4 * MIB,
    )
    workspace = StorageBudget(settings, time.monotonic() + 30, name="source-workspace")
    runtime = RustPlayerRuntime(
        runtime_config() | {"storage_bytes": settings.budget_bytes, "game_concurrency": 1}
    )
    writer = None
    pending = None
    try:
        with managed(tmp_path / "server") as (startup, receipt):
            first, second = [game_spec(startup, receipt, episode=f"pair-{i}") for i in range(2)]
            contexts = tuple(journal_context(spec) for spec in (first, second))
            writer = RustPlayerStorage(
                runtime,
                tmp_path / "collection",
                run_id="run",
                contexts=() if deferred else contexts,
                groups=(StorageGroup("pair", ("pair-0", "pair-1"), {"paired": True}),),
                settings=settings,
                workspace=workspace,
                workspace_bytes=256 * MIB,
                deadline=time.monotonic() + 30,
                cleanup_deadline=time.monotonic() + 40,
                deferred_contexts=deferred,
                context_plans=tuple(episode_encoding_plan(c) for c in contexts) if deferred else (),
            )
            if deferred:
                assert writer.encoders == {}
                writer.bind_contexts(contexts)
                assert set(writer.encoders) == {"pair-0", "pair-1"}
            for ordinal, spec in enumerate((first, second)):
                runtime.start([spec])
                while True:
                    bundle = runtime.next_storage(time.monotonic() + 20)
                    terminal = bundle.kind == "terminal"
                    writer.accept(bundle)
                    runtime.storage_ack(bundle)
                    del bundle
                    if terminal:
                        break
                runtime.wait(time.monotonic() + 10)
                if ordinal == 0:
                    assert writer.committed == set()
                    assert runtime.recording_account().native_bytes > 0
                    assert writer._thread.is_alive()
                    with sqlite3.connect(tmp_path / "collection/collection.sqlite3") as db:
                        assert db.execute("select count(*) from groups").fetchone() == (0,)
                    db.close()
                    with pytest.raises(ValueError, match="complete"):
                        writer.inventory()
            inventory = writer.inventory()
            rows = inventory.rows
            assert rows and type(rows) is tuple
            with sqlite3.connect(tmp_path / "collection/collection.sqlite3") as db:
                assert rows == tuple(db.execute(
                    "SELECT name,length,sha256 FROM members ORDER BY name"
                ))
            db.close()
            assert not any(name in ("manifest.json", "report.json") for name, _, _ in rows)
            writer.store_control("manifest.json", b'{"diagnostic_component":true}')
            locator = writer.seal()
            report = writer.join()
            assert report["joined"] and report["sealed"] and report["committed_groups"] == 1
            assert (
                create_threads == [writer.thread_identity]
                and create_threads[0] != threading.get_ident()
            )
            assert runtime.join()["joined"]
            assert workspace.used == 256 * MIB
            del rows
            inventory.close()
            assert workspace.used == 0
            pending = writer.take_pending_owner()
            pending.recheck()
            originals = PackedOriginals(
                locator.path,
                deadline=time.monotonic() + 20,
                max_owned_bytes=512 * MIB,
                allow_pending=True,
            )
            originals.recheck()
            originals.close()
            view = pending.view()
            alias = view[1:]
            assert view.readonly and hashlib_sha(view) == pending.descriptor_sha256
            charge = runtime.recording_account().external_bytes
            assert charge == len(view)
            pending.close()
            assert runtime.recording_account().external_bytes == charge
            del view
            assert runtime.recording_account().external_bytes == charge
            del alias
            assert runtime.recording_account().used == 0
            with pytest.raises(RuntimeError, match="closed"):
                pending.recheck()
            assert json.loads(locator.path.read_bytes())["publication"] == "pending"
            assert not (locator.path.parent / "collection.sqlite3-wal").exists()
    finally:
        runtime.cancel("writer test complete")
        runtime.join()
        if writer is not None:
            writer.close()
        if pending is not None:
            pending.close()


def hashlib_sha(view):
    import hashlib

    return hashlib.sha256(view).hexdigest()


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate", "sql", "publication", "command", "interval-prefix", "mixed-profile",
        "old-writer-new-table", "new-writer-old-table",
    ],
)
def test_actual_storage_failure_cancels_and_never_publishes_healthy_seal(
    tmp_path, monkeypatch, failure
):
    import os
    from contextlib import suppress
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace

    from gf_common import card_instance
    from gf_common.domain import ContractError, digest
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training import group_pack_worker as database_module
    from gf_training import rust_player_storage as storage_module
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.rust_player_storage import (
        RustPendingDescriptor,
        RustPlayerStorage,
        StorageGroup,
    )
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config

    release = {}
    foreign_contract = None
    if failure in ("old-writer-new-table", "new-writer-old-table"):
        old_writer = failure == "old-writer-new-table"
        env = "GFB_P41_TEST" if old_writer else "GFB_DREAM_COMPAT_TEST"
        source, binary = os.environ.get(f"{env}_SOURCE"), os.environ.get(f"{env}_SERVER")
        if not source or not binary:
            pytest.skip("explicit managed Dream release binary/approved source required")
        old = card_instance.strength_powder_card_instance_contract()
        new = card_instance.dream_compat_card_instance_contract()
        own, foreign_contract = (old, new) if old_writer else (new, old)
        release = {"source": Path(source), "binary": Path(binary), "contract": own}
    elif failure == "mixed-profile":
        foreign_contract = card_instance.strength_powder_card_instance_contract()

    rejects_pending = failure in ("command", "interval-prefix") or foreign_contract is not None
    prefix_rejected = False
    if failure == "interval-prefix":
        begin_native_frame = storage_module._Journal._begin_native_frame

        def wrong_prefix(journal, bundle, frame, delta, report, selection, command):
            nonlocal prefix_rejected
            start_hash = "0" * 64
            assert start_hash != frame.interval_start_hash
            bad_frame = replace(
                frame, interval_start_hash=start_hash,
                frame_hash=digest(frame._binding() | {"interval_start_hash": start_hash}),
            )
            table = storage_module.decode_original(bundle)["generated"]["table"]
            table.pop("table_hash")
            table_hash = digest(table | {"frame_hash": bad_frame.frame_hash})

            def checked_candidates(max_positions, size):
                _, _, selected, count, elements, action, payload, sha256 = (
                    bundle._journal_candidates(max_positions, size)
                )
                return (
                    bad_frame.frame_hash, table_hash, selected, count, elements, action, payload,
                    sha256,
                )

            before = {name: len(chunks) for name, chunks in journal.sink.chunks.items()}
            with pytest.raises(ContractError, match="interval prefix mismatch") as rejected:
                begin_native_frame(
                    journal, SimpleNamespace(_journal_candidates=checked_candidates), bad_frame,
                    delta, replace(report, table_hash=table_hash),
                    replace(selection, table_hash=table_hash), command,
                )
            assert journal.count == 0 and not journal.pending
            assert {name: len(chunks) for name, chunks in journal.sink.chunks.items()} == before
            prefix_rejected = True
            raise rejected.value

        monkeypatch.setattr(storage_module._Journal, "_begin_native_frame", wrong_prefix)
    elif rejects_pending:
        decode = storage_module.decode_original

        def wrong_pending(bundle):
            value = decode(bundle)
            if bundle.kind == "pending":
                if failure == "command":
                    value["command"]["action"] = {"type": "invalid test action"}
                else:
                    value["generated"]["table"]["action_contract"] = asdict(foreign_contract)
            return value

        monkeypatch.setattr(storage_module, "decode_original", wrong_pending)
    elif failure == "sql":

        def fail_sql(*args, **kwargs):
            raise OSError("private SQL injection")

        monkeypatch.setattr(database_module, "submit_group", fail_sql)
    elif failure == "publication":
        actual = RustPendingDescriptor.recheck
        injected = False

        def fail_publication(owner):
            nonlocal injected
            if not injected:
                injected = True
                raise OSError("private publication injection")
            actual(owner)

        monkeypatch.setattr(RustPendingDescriptor, "recheck", fail_publication)
    settings = StorageSettings(
        db_root=tmp_path / "catalog",
        ram_root=tmp_path / "ram",
        budget_bytes=512 * MIB,
        queue_bytes=4 * MIB,
    )
    workspace = StorageBudget(settings, time.monotonic() + 20, name="source-workspace")
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    writer = None
    held = None
    try:
        with managed(tmp_path / "server", **release) as (startup, receipt):
            spec = game_spec(startup, receipt)
            spec["actions"]["action_contract"] = receipt["action_contract"]
            writer = RustPlayerStorage(
                runtime,
                tmp_path / "collection",
                run_id="run",
                contexts=(journal_context(spec),),
                groups=(StorageGroup("g", (spec["episode"],), {}),),
                settings=settings,
                workspace=workspace,
                workspace_bytes=256 * MIB,
                deadline=time.monotonic() + 20,
                cleanup_deadline=time.monotonic() + 25,
            )
            runtime.start([spec])
            while True:
                bundle = runtime.next_storage(time.monotonic() + 10)
                if rejects_pending and bundle.kind == "pending":
                    held = bundle.view()
                    with pytest.raises(RuntimeError, match="storage failed") as caught:
                        writer.accept(bundle)
                    assert failure != "interval-prefix" or prefix_rejected
                    caught.value.__traceback__ = None  # caller's exception retained the original
                    del caught, bundle
                    break
                writer.accept(bundle)
                if held is None:
                    held = bundle.view()
                runtime.storage_ack(bundle)
                if failure == "duplicate":
                    with pytest.raises(RuntimeError, match="storage failed") as caught:
                        writer.accept(bundle)
                    assert "private" not in str(caught.value)
                    caught.value.__traceback__ = None
                    del caught
                    break
                terminal = bundle.kind == "terminal"
                del bundle
                if terminal:
                    break
            if failure in ("sql", "publication"):
                with pytest.raises(RuntimeError, match="storage failed"):
                    writer.seal()
            with pytest.raises(RuntimeError, match="storage failed"):
                writer.close()
            assert not writer._thread.is_alive() and workspace.used == 0
            assert not writer._sealed
            pending = tmp_path / "collection/collection.pending.json"
            if pending.exists():
                assert pending.read_bytes().startswith(b"GFB-QUARANTINED")
            runtime.join()
            assert runtime.recording_account().native_bytes >= held.nbytes
            held = None
            if failure == "duplicate":
                del bundle
            assert runtime.recording_account().used == 0
    finally:
        held = None
        runtime.cancel("fault test complete")
        runtime.join()
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()


@pytest.mark.parametrize("mode", ["stale_hints", "natural", "abort_turn_limit", "abort_timeout"])
def test_actual_terminal_dispositions_and_zero_act_bank_identity(tmp_path, mode):
    import json
    from contextlib import nullcontext

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.rust_player_storage import (
        CanonicalEpisodeEncoder,
        RecordingArtifacts,
        RecordingBudget,
    )
    from tests.test_rust_player_runtime import fault_proxy, game_spec, managed, runtime_config

    with managed(tmp_path) as (startup, receipt):  # noqa: SIM117 -- dependent proxy choice
        with (
            nullcontext((startup, None)) if mode == "natural" else fault_proxy(startup, mode) as (
                routed,
                state,
            )
        ):
            spec = game_spec(routed, receipt)
            spec["max_decisions"] = 500 if mode == "natural" else 8
            context = journal_context(spec)
            runtime = RustPlayerRuntime(runtime_config())
            sink = RecordingArtifacts(
                RecordingBudget(runtime.recording_account(), time.monotonic() + 30)
            )
            encoder = CanonicalEpisodeEncoder(context, sink)
            try:
                runtime.start([spec])
                while encoder.terminal is None:
                    bundle = runtime.next_storage(time.monotonic() + 20)
                    encoder.consume(bundle)
                    runtime.storage_ack(bundle)
                    del bundle
                runtime.wait(time.monotonic() + 10)
                runtime.join()
                members = sink.members()
                files = {m.name: m.data for m in members}
                report = encoder.terminal
                expected = {
                    "stale_hints": "incomplete",
                    "natural": "natural",
                    "abort_turn_limit": "truncated",
                    "abort_timeout": "aborted",
                }[mode]
                assert report["end_kind"] == expected
                for viewer in (1, 2):
                    rows = [
                        json.loads(line)
                        for line in files[f"planned/player-{viewer}.jsonl"].splitlines()
                    ]
                    assert rows[0]["bank_episode"] == "planned" and rows[0]["viewer"] == viewer
                    assert rows[-1]["end_kind"] == expected
                    assert (rows[-1]["terminal_reward"] is not None) == (mode == "natural")
                    if mode == "stale_hints":
                        assert not any(row["kind"] == "decision" for row in rows)
                terminal = json.loads(files["planned/native-terminal.json"])
                if mode == "stale_hints":
                    assert report["decisions"] == 0 and report["error"]
                    assert [(bank["actor"], bank["episode"]) for bank in terminal["banks"]] == [
                        (1, "planned"),
                        (2, "planned"),
                    ]
                    assert not any(
                        path.endswith("/actions") for method, path, body in state["calls"]
                    )
                else:
                    assert report["decisions"] > 0
                del members, files, terminal, rows
                encoder.retire()
                sink.close()
                assert runtime.recording_account().used == 0
            finally:
                runtime.cancel("disposition test cleanup")
                runtime.join()
                encoder.retire()
                sink.close()


def test_actual_mlp_original_selected_scalars_roundtrip_in_canonical_journal(tmp_path):
    import json
    from dataclasses import replace

    from gf_common.domain import digest
    from gf_models import structured_artifacts
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_binding import ConfiguredNeuralModelV3
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.rust_player_storage import (
        PolicyJournalContext,
        RustPlayerStorage,
        StorageGroup,
    )
    from tests.test_rust_player_runtime import actual_service, game_spec, managed, runtime_config

    service, network = actual_service()
    writer = pending = None
    artifact_owner = None
    with managed(tmp_path / "server") as (startup, receipt):
        spec = game_spec(startup, receipt)
        baseline_context = journal_context(spec)
        model, binding = network.config, network.binding
        directory = tmp_path / "inference"
        structured_artifacts.save_game_inference_v3(
            directory,
            network,
            model,
            digest(asdict(baseline_context.actions)),
            action_contract=baseline_context.actions.action_contract,
            rules=baseline_context.rules,
        )
        # Nested model settings are typed dataclasses, not recursively serialized dicts.
        configured = ConfiguredNeuralModelV3(
            **{key: getattr(model, key) for key in model.__dataclass_fields__},
            checkpoint=str(directory),
            inference=binding,
        )
        artifact_owner = RetainedGameArtifacts(
            (configured,),
            baseline_context.actions,
            rules=baseline_context.rules,
            max_owned_bytes=256 * MIB,
            preflight_deadline=time.monotonic() + 20,
            finalization_deadline=time.monotonic() + 30,
        )
        label, execution = artifact_owner.binding(configured)
        caps, costs, _, _ = own_game_worker_settings_v3(
            "mlp", baseline_context.actions, execution.policy_options, rules=baseline_context.rules
        )
        spec["numeric_costs"] = asdict(costs)
        limits = {
            "max_decisions": model.temporal.max_decisions,
            "max_episode_events": model.event.max_episode,
            "max_interval_events": model.event.max_interval,
            "max_candidates": binding.max_candidates,
            "max_positions": model.max_positions,
            "aggregate_window": model.event.aggregate_window,
        }
        policies = []
        for viewer in (1, 2):
            native = {
                "kind": "mlp",
                "seed": 16 + viewer,
                "checkpoint": "a" * 64,
                "epoch": 0,
                "limits": limits,
                "attack_threshold": 12,
            }
            spec["policies"][viewer - 1] = native
            policies.append(
                PolicyJournalContext(
                    native,
                    {
                        "policy": label,
                        "model_id": "mlp",
                        "caps": asdict(caps),
                        "numeric_costs": asdict(costs),
                        "sampling_seed": 16 + viewer,
                        "neural": artifact_owner.journal_metadata(label),
                    },
                )
            )
        context = replace(baseline_context, numeric_costs=asdict(costs), policies=tuple(policies))
        from gf_common.group_pack import decode_pack
        from gf_training.group_pack_resources import StorageBudget

        settings = StorageSettings(
            db_root=tmp_path / "catalog",
            ram_root=tmp_path / "ram",
            budget_bytes=512 * MIB,
            queue_bytes=4 * MIB,
        )
        runtime = RustPlayerRuntime(
            runtime_config()
            | {"runtime_identity": service.runtime, "storage_bytes": settings.budget_bytes},
            service=service,
        )
        workspace = StorageBudget(settings, time.monotonic() + 30, name="source-workspace")
        writer = RustPlayerStorage(
            runtime,
            tmp_path / "collection",
            run_id="run",
            contexts=(context,),
            groups=(StorageGroup("g", (context.episode_id,), {}),),
            settings=settings,
            workspace=workspace,
            workspace_bytes=256 * MIB,
            deadline=time.monotonic() + 30,
            cleanup_deadline=time.monotonic() + 40,
        )
        observed = {}
        actual_evaluate = service.evaluate

        def evaluate(batch):
            answers = actual_evaluate(batch)
            for answer in answers:
                identity = json.loads(answer["identity"])
                observed[identity["player"], identity["request"]] = (
                    float(answer["logs"][answer["sample"]]),
                    float(answer["value"]),
                )
            return answers

        service.evaluate = evaluate
        try:
            runtime.start([spec])
            while True:
                bundle = runtime.next_storage(time.monotonic() + 20)
                terminal = bundle.kind == "terminal"
                writer.accept(bundle)
                runtime.storage_ack(bundle)
                del bundle
                if terminal:
                    break
            runtime.wait(time.monotonic() + 10)
            runtime.join()
            locator = writer.seal()
            writer.join()
            pending = writer.take_pending_owner()
            db = sqlite3.connect(locator.path.parent / "collection.sqlite3")
            try:
                compressed = db.execute("select pack from groups").fetchone()[0]
            finally:
                db.close()
            members = decode_pack(compressed)
            del compressed
            for member in members:
                if member.name.endswith(".jsonl"):
                    rows = [json.loads(line) for line in member.data.splitlines()]
                    viewer = rows[0]["viewer"]
                    assert rows[0]["model_version"] == "3"
                    for row in rows:
                        if row["kind"] == "decision":
                            pair = row["selection"]["logprob"], row["selection"]["value"]
                            assert pair == observed[viewer, row["decision"]]
            del members, member, rows
            pending.close()
            assert runtime.recording_account().used == 0
        finally:
            runtime.cancel("MLP canonical test cleanup")
            runtime.join()
            writer.close()
            if pending is not None:
                pending.close()
            artifact_owner.release()


def test_database_initialization_unconfirmed_connection_retains_actual_owner(tmp_path, monkeypatch):
    from gf_training import group_pack_db as database
    from gf_training import group_pack_worker as worker

    actual_connect = sqlite3.connect
    actual_configure = database.configure
    configuration_failed = False
    attempts = []
    connections = []

    class Connection(sqlite3.Connection):
        def close(self):
            attempts.append((self, threading.get_ident()))
            if len(attempts) == 1:
                raise OSError("creation close unconfirmed before release")
            super().close()

    def connect(*args, **kwargs):
        connection = actual_connect(*args, **kwargs, factory=Connection)
        connections.append(connection)
        return connection

    def fail_configuration(*args, **kwargs):
        nonlocal configuration_failed
        if not configuration_failed:
            configuration_failed = True
            raise OSError("initial SQLite configuration failed")
        return actual_configure(*args, **kwargs)

    monkeypatch.setattr(database.sqlite3, "connect", connect)
    monkeypatch.setattr(database, "configure", fail_configuration)
    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram")
    values = asdict(settings) | {
        "db_root": str(settings.db_root),
        "ram_root": str(settings.ram_root),
    }
    directory = tmp_path / "collection"
    directory.mkdir()

    def run():
        owner = worker.GroupPackDatabase(max_owned_bytes=512 * MIB)
        with pytest.raises(OSError, match="initial SQLite") as caught:
            owner.execute(
                {
                    "version": 1,
                    "request_id": 0,
                    "command": "init",
                    "directory": str(directory),
                    "run_id": "creation-fault",
                    "settings": values,
                    "deadline": time.monotonic() + 20,
                }
            )
        assert owner.db is connections[0]
        caught.value.__traceback__ = None
        assert owner.fail() == [] and owner.db is None
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("select 1")
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=1) as executor:
        identity = executor.submit(run).result()
    assert [thread for connection, thread in attempts if connection is connections[0]] == [
        identity,
        identity,
    ]


def test_actual_writer_queue_cancel_wakes_blocked_producers_before_act(tmp_path, monkeypatch):
    from contextlib import suppress

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.rust_player_storage import (
        CanonicalEpisodeEncoder,
        RustPlayerStorage,
        StorageGroup,
    )
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config

    entered, release = threading.Event(), threading.Event()
    actual = CanonicalEpisodeEncoder.consume

    def blocked(owner, bundle):
        entered.set()
        assert release.wait(5)
        return actual(owner, bundle)

    monkeypatch.setattr(CanonicalEpisodeEncoder, "consume", blocked)
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": 512 * MIB})
    writer = None
    first = second = None
    try:
        with managed(tmp_path / "server", count=2) as (startup, receipt):
            specs = tuple(game_spec(startup, receipt, episode=f"q{i}") for i in range(2))
            runtime.start(list(specs))
            first, second = [runtime.next_storage(time.monotonic() + 10) for _ in range(2)]
            settings = StorageSettings(
                db_root=tmp_path / "catalog",
                ram_root=tmp_path / "ram",
                budget_bytes=512 * MIB,
                queue_bytes=max(first.reserved_bytes, second.reserved_bytes),
            )
            source = StorageBudget(settings, time.monotonic() + 20, name="source-workspace")
            writer = RustPlayerStorage(
                runtime,
                tmp_path / "collection",
                run_id="run",
                contexts=tuple(journal_context(s) for s in specs),
                groups=(StorageGroup("g", tuple(s["episode"] for s in specs), {}),),
                settings=settings,
                workspace=source,
                workspace_bytes=256 * MIB,
                deadline=time.monotonic() + 20,
                cleanup_deadline=time.monotonic() + 25,
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                one = pool.submit(writer.accept, first)
                assert entered.wait(3)
                two = pool.submit(writer.accept, second)
                with pytest.raises(TimeoutError):
                    two.result(0.03)
                writer.cancel("queue cancel test")
                release.set()
                for future in (one, two):
                    with pytest.raises(RuntimeError, match="storage failed") as caught:
                        future.result(3)
                    caught.value.__traceback__ = None
                    del caught
            with pytest.raises(RuntimeError, match="storage failed"):
                writer.close()
            assert not writer._thread.is_alive() and source.used == 0
            report = runtime.join()
            assert all(game["decisions"] == 0 for game in report["games"])
            first = second = None
            assert runtime.recording_account().used == 0
    finally:
        release.set()
        first = second = None
        runtime.cancel("queue test cleanup")
        runtime.join()
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()


def test_foreign_runtime_bundle_cannot_enter_another_recording_envelope(tmp_path):
    from contextlib import suppress

    from gf_common.domain import ContractError
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.rust_player_storage import RustPlayerStorage, StorageGroup
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config

    settings = StorageSettings(
        db_root=tmp_path / "catalog", ram_root=tmp_path / "ram", budget_bytes=512 * MIB
    )
    one = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    two = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    writer = None
    bundle = None
    try:
        with managed(tmp_path / "server") as (startup, receipt):
            spec = game_spec(startup, receipt)
            source = StorageBudget(settings, time.monotonic() + 20, name="source-workspace")
            writer = RustPlayerStorage(
                one,
                tmp_path / "collection",
                run_id="run",
                contexts=(journal_context(spec),),
                groups=(StorageGroup("g", (spec["episode"],), {}),),
                settings=settings,
                workspace=source,
                workspace_bytes=256 * MIB,
                deadline=time.monotonic() + 20,
                cleanup_deadline=time.monotonic() + 25,
            )
            two.start([spec])
            bundle = two.next_storage(time.monotonic() + 10)
            assert two.recording_account().owns(bundle)
            assert not one.recording_account().owns(bundle)
            with pytest.raises(ContractError, match="SAME-account") as caught:
                writer.accept(bundle)
            caught.value.__traceback__ = None
            del caught
            with pytest.raises(RuntimeError, match="storage failed"):
                writer.close()
            assert source.used == 0 and one.recording_account().used == 0
            assert two.recording_account().native_bytes >= bundle.reserved_bytes
            two.cancel("foreign owner test complete")
            assert two.join()["games"][0]["decisions"] == 0
            bundle = None
            assert two.recording_account().used == 0
    finally:
        bundle = None
        for runtime in (one, two):
            runtime.cancel("identity cleanup")
            runtime.join()
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()


def _fix1_idle_inputs(tmp_path, *, deadline=None):
    """SQLite/thread ownership fixture only; it issues no gameplay/Source data."""
    import json
    from pathlib import Path

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.rust_player_storage import StorageGroup
    from tests.test_rust_player_runtime import runtime_config

    settings = StorageSettings(
        db_root=tmp_path / "catalog",
        ram_root=tmp_path / "ram",
        budget_bytes=512 * MIB,
        queue_bytes=4 * MIB,
    )
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/rust_player_v5/originals.json").read_bytes()
    )["episodes"][0]
    stop = time.monotonic() + 20 if deadline is None else deadline
    spec = {
        "run": "run",
        "trial": "trial",
        "episode": "idle",
        "actions": fixture["actions"],
        "rules": fixture["rules"],
        "numeric_costs": fixture["numeric_costs"],
        "runtime_evidence": {},
        "policies": [{"kind": "heuristic", "seed": 17}, {"kind": "heuristic", "seed": 18}],
        "deadline": stop,
    }
    source = StorageBudget(settings, time.monotonic() + 20, name="fix1-source-workspace")
    return (
        runtime,
        source,
        {
            "runtime": runtime,
            "directory": tmp_path / "collection",
            "run_id": "run",
            "contexts": (journal_context(spec),),
            "groups": (StorageGroup("g", ("idle",), {}),),
            "settings": settings,
            "workspace": source,
            "workspace_bytes": 256 * MIB,
            "deadline": stop,
            "cleanup_deadline": time.monotonic() + 1,
        },
    )


def test_fix1_publication_constructor_nested_original_is_closed_on_storage_thread(
    tmp_path, monkeypatch
):
    import os
    from contextlib import suppress

    from gf_training import rust_player_storage as module

    opened = []
    attempts = []
    actual = module.OriginalFile

    class Original(actual):
        def recheck(self):
            super().recheck()
            if not opened:
                opened.append((self, self.fd))
                raise OSError("publication constructor recheck")

        def close(self):
            attempts.append(threading.get_ident())
            if len(attempts) == 1:
                raise OSError("first close before actual release")
            super().close()

    monkeypatch.setattr(module, "OriginalFile", Original)
    runtime, source, kwargs = _fix1_idle_inputs(tmp_path)
    owner = None
    try:
        with pytest.raises(RuntimeError, match="storage failed") as caught:
            module.RustPlayerStorage(**kwargs)
        owner = caught.value.cleanup_owner
        caught.value.__traceback__ = None
        del caught
        with suppress(RuntimeError):
            owner.close()
        assert not owner._thread.is_alive()
        assert len(opened) == 1
        with pytest.raises(OSError):
            os.fstat(opened[0][1])
        assert source.used == 0 and owner.cleanup_errors == ()
        assert len(attempts) >= 2 and set(attempts) == {owner.thread_identity}
    finally:
        if owner is not None:
            with suppress(RuntimeError):
                owner.close()
        # Old-candidate RED leaves this real non-SQL FD alive; safe test cleanup.
        for original, _ in opened:
            if original.fd >= 0:
                original.close()
        runtime.cancel("fix1 init cleanup")
        runtime.join()


@pytest.mark.parametrize("failure", ["wait-timeout", "remaining-race"])
def test_fix1_post_start_constructor_error_transfers_same_owner(tmp_path, monkeypatch, failure):
    from contextlib import suppress

    from gf_training import rust_player_storage as module

    entered, release = threading.Event(), threading.Event()
    observed = []
    actual_run = module.RustPlayerStorage._run
    actual_original = module.OriginalFile

    def track(owner):
        observed.append(owner)
        return actual_run(owner)

    class Original(actual_original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            entered.set()
            assert release.wait(3)

    monkeypatch.setattr(module.RustPlayerStorage, "_run", track)
    monkeypatch.setattr(module, "OriginalFile", Original)
    stop = time.monotonic() + 0.1 if failure == "wait-timeout" else time.monotonic() + 20
    runtime, source, kwargs = _fix1_idle_inputs(tmp_path, deadline=stop)
    if failure == "remaining-race":
        actual_remaining = module.remaining
        main = threading.get_ident()

        def expired(deadline):
            if threading.get_ident() == main and not entered.is_set():
                assert entered.wait(2)
                raise TimeoutError("forced post-start remaining expiry")
            return actual_remaining(deadline)

        monkeypatch.setattr(module, "remaining", expired)
    try:
        with pytest.raises(TimeoutError) as caught:
            module.RustPlayerStorage(**kwargs)
        assert entered.is_set() and len(observed) == 1
        owner = observed[0]
        assert owner._thread.is_alive() and source.used == 256 * MIB
        try:
            assert caught.value.cleanup_owner is owner
            assert owner._failure is not None
        finally:
            caught.value.__traceback__ = None
            del caught
            release.set()
        with suppress(RuntimeError, TimeoutError):
            owner.close()
        owner._thread.join(2)
        assert not owner._thread.is_alive()
        # A real expired execution deadline may leave failed-catalog cleanup
        # explicitly unhealthy; it must keep its SAME actual workspace owner.
        assert source.used == 0 or (owner.cleanup_errors and owner._workspace_lease is not None)
    finally:
        release.set()
        for owner in observed:
            with suppress(RuntimeError, TimeoutError):
                owner.close()
        runtime.cancel("fix1 constructor cleanup")
        runtime.join()


@pytest.fixture
def fix1_completed_storage(tmp_path):
    from contextlib import suppress

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.rust_player_storage import RustPlayerStorage, StorageGroup
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config

    settings = StorageSettings(
        db_root=tmp_path / "catalog",
        ram_root=tmp_path / "ram",
        budget_bytes=512 * MIB,
        queue_bytes=4 * MIB,
    )
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    source = StorageBudget(settings, time.monotonic() + 20, name="fix1-source-workspace")
    writer = None
    try:
        with managed(tmp_path / "server") as (startup, receipt):
            spec = game_spec(startup, receipt)
            spec["max_decisions"] = 1
            writer = RustPlayerStorage(
                runtime,
                tmp_path / "collection",
                run_id="run",
                contexts=(journal_context(spec),),
                groups=(StorageGroup("g", (spec["episode"],), {}),),
                settings=settings,
                workspace=source,
                workspace_bytes=256 * MIB,
                deadline=time.monotonic() + 20,
                cleanup_deadline=time.monotonic() + 25,
            )
            runtime.start([spec])
            while True:
                bundle = runtime.next_storage(time.monotonic() + 10)
                terminal = bundle.kind == "terminal"
                writer.accept(bundle)
                runtime.storage_ack(bundle)
                del bundle
                if terminal:
                    break
            runtime.wait(time.monotonic() + 10)
            # A following control command confirms the actual group SQL finished.
            writer.store_control("fix1-control", b"actual original")
            yield writer, runtime, source
    finally:
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()
            if writer._pending_owner is not None:
                with suppress(RuntimeError):
                    writer._pending_owner.close()
        runtime.cancel("fix1 live fixture cleanup")
        runtime.join()


def test_fix1_seal_inspection_nested_sqlite_owner_is_closed(fix1_completed_storage, monkeypatch):
    import os
    from contextlib import suppress

    from gf_training import group_pack_source as source_module

    writer, _runtime, source = fix1_completed_storage
    actual_recheck = source_module.PackedOriginals.recheck
    actual_close = source_module.PackedOriginals.close
    inspected = []
    fds = []
    closing = []

    def recheck(owner):
        actual_recheck(owner)
        if not inspected:
            inspected.append((owner, owner.db))
            # A real independently opened original inside the inspection owner
            # makes nested FD custody observable without assuming SQLite opens
            # a fresh number (its VFS may reuse an already opened inode).
            original = source_module.OriginalFile(
                owner.database_file.path, owner.deadline, mutable=True
            )
            owner._cleanup_children.append(original)
            fds.append(original.fd)
            raise OSError("inspection recheck failed")

    def close(owner):
        closing.append(threading.get_ident())
        if len(closing) == 1:
            raise OSError("inspection first close before release")
        return actual_close(owner)

    monkeypatch.setattr(source_module.PackedOriginals, "recheck", recheck)
    monkeypatch.setattr(source_module.PackedOriginals, "close", close)
    with pytest.raises(RuntimeError, match="storage failed"):
        writer.seal()
    with suppress(RuntimeError):
        writer.close()
    assert not writer._thread.is_alive() and inspected and fds
    assert inspected[0][0]._closed
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert set(closing) == {writer.thread_identity} and len(closing) >= 2
    assert source.used == 0 and writer.cleanup_errors == ()


def test_fix1_cancel_wins_after_actual_publication(fix1_completed_storage, monkeypatch):
    from contextlib import suppress

    from gf_training import rust_player_storage as module

    writer, _runtime, source = fix1_completed_storage
    reached, release = threading.Event(), threading.Event()
    actual = module.RustPlayerStorage._publish
    produced = []

    def publication(owner, descriptor):
        locator = actual(owner, descriptor)
        produced.append(locator)
        reached.set()
        assert release.wait(3)
        return locator

    monkeypatch.setattr(module.RustPlayerStorage, "_publish", publication)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(writer.seal)
            assert reached.wait(2)
            writer.cancel("cancel after actual publication")
            release.set()
            with pytest.raises(RuntimeError, match="storage failed") as caught:
                waiting.result(3)
            caught.value.__traceback__ = None
        with suppress(RuntimeError):
            writer.close()
        assert not writer._sealed and not writer._thread.is_alive()
        assert produced[0].path.read_bytes().startswith(b"GFB-QUARANTINED")
        assert writer._pending_owner is None and source.used == 0
    finally:
        release.set()


@pytest.mark.parametrize("boundary", ["submit", "columns"])
def test_fix1_chained_error_retires_private_buffer_before_refund(tmp_path, monkeypatch, boundary):
    import os
    import weakref
    from contextlib import suppress

    import numpy as np

    from gf_simulator_client.rust_player import RustPlayerRuntime
    from gf_training import group_pack_worker as worker
    from gf_training import rust_player_storage as module
    from gf_training.group_pack_resources import OwnedDescriptor, StorageBudget
    from tests.test_rust_player_runtime import game_spec, managed, runtime_config

    watched = []
    refund_liveness = []
    resources = []
    close_threads = []
    resource_path = tmp_path / "fault-owner"
    resource_path.write_bytes(b"actual fd original")

    class Original(OwnedDescriptor):
        def close(self):
            close_threads.append(threading.get_ident())
            super().close()

    def fail_with_buffer(data, sql=None):
        private_view = np.frombuffer(data, dtype=np.uint8)
        watched.append(weakref.ref(private_view))
        first = Original(os.open(resource_path, os.O_RDONLY))
        second = Original(os.open(resource_path, os.O_RDONLY))
        resources.extend([(first, first.fd), (second, second.fd)])
        try:
            if sql is None:
                raise OSError("inner codec failure")
            sql.execute("select * from fix1_absent_table")
        except (OSError, sqlite3.Error) as error:
            error.cleanup_owner = first
            wrapped = RuntimeError("outer wrapped backend failure")
            wrapped.__cause__ = error
        independent = OSError("independent context owner")
        independent.cleanup_owner = second
        wrapped.__context__ = independent
        independent.__cause__ = wrapped  # genuine exception-chain cycle
        raise wrapped

    actual_close = module._RecordingReservation.close

    def close(lease):
        if watched and not refund_liveness:
            refund_liveness.append(watched[-1]() is not None)
        return actual_close(lease)

    monkeypatch.setattr(module._RecordingReservation, "close", close)
    if boundary == "submit":

        def submit(db, run_id, group_id, pack, **kwargs):
            fail_with_buffer(pack, db)

        monkeypatch.setattr(worker, "submit_group", submit)
    else:
        actual_write = module.RecordingArtifacts.write

        def write(owner, name, data, reservation=None):
            if name.endswith(".columns"):
                fail_with_buffer(data)
            return actual_write(owner, name, data, reservation)

        monkeypatch.setattr(module.RecordingArtifacts, "write", write)
    settings = StorageSettings(
        db_root=tmp_path / "catalog",
        ram_root=tmp_path / "ram",
        budget_bytes=512 * MIB,
        queue_bytes=4 * MIB,
    )
    runtime = RustPlayerRuntime(runtime_config() | {"storage_bytes": settings.budget_bytes})
    source = StorageBudget(settings, time.monotonic() + 20, name="fix1-source-workspace")
    writer = None
    bundle = None
    try:
        with managed(tmp_path / "server") as (startup, receipt):
            spec = game_spec(startup, receipt)
            spec["max_decisions"] = 1
            writer = module.RustPlayerStorage(
                runtime,
                tmp_path / "collection",
                run_id="run",
                contexts=(journal_context(spec),),
                groups=(module.StorageGroup("g", (spec["episode"],), {}),),
                settings=settings,
                workspace=source,
                workspace_bytes=256 * MIB,
                deadline=time.monotonic() + 20,
                cleanup_deadline=time.monotonic() + 25,
            )
            runtime.start([spec])
            while True:
                try:
                    bundle = runtime.next_storage(time.monotonic() + 5)
                    terminal = bundle.kind == "terminal"
                    writer.accept(bundle)
                    runtime.storage_ack(bundle)
                    bundle = None
                    if terminal:
                        break
                except (RuntimeError, ValueError) as error:
                    error.__traceback__ = None
                    break
            with suppress(RuntimeError):
                writer.close()
            runtime.join()
            bundle = None
            assert refund_liveness == [False]
            assert watched and watched[-1]() is None
            assert not writer._thread.is_alive() and source.used == 0
            assert resources and set(close_threads) == {writer.thread_identity}
            for resource, fd in resources:
                assert resource.fd == -1
                with pytest.raises(OSError):
                    os.fstat(fd)
    finally:
        runtime.cancel("fix1 chained cleanup")
        runtime.join()
        bundle = None
        if writer is not None:
            with suppress(RuntimeError):
                writer.close()
        for resource, _ in resources:
            if resource.fd >= 0:
                resource.close()


def test_fix2_real_launch_interrupt_transfers_same_live_sqlite_owner(tmp_path, monkeypatch):
    import os
    from contextlib import suppress

    from gf_training import rust_player_storage as module

    runtime, source, kwargs = _fix1_idle_inputs(tmp_path)
    kwargs["cleanup_deadline"] = time.monotonic() + 5
    reached, release = threading.Event(), threading.Event()
    observed = []
    actual_run = module.RustPlayerStorage._run
    actual_start = threading.Thread.start

    class ReadyBarrier:
        def __init__(self, event):
            self.event = event

        def set(self):
            self.event.set()
            reached.set()
            assert release.wait(3)

        def wait(self, timeout=None):
            return self.event.wait(timeout)

    def run(owner):
        owner._ready = ReadyBarrier(owner._ready)
        observed.append(owner)
        return actual_run(owner)

    def start(thread):
        actual_start(thread)
        if thread.name == "gf-rust-player-storage":
            assert reached.wait(2)
            assert len(observed) == 1 and observed[0]._ready.wait(0)
            raise KeyboardInterrupt("interrupted AFTER real native thread launch")

    monkeypatch.setattr(module.RustPlayerStorage, "_run", run)
    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(KeyboardInterrupt, match="AFTER real native") as caught:
            module.RustPlayerStorage(**kwargs)
        owner = observed[0]
        assert owner._thread.is_alive() and owner._database.db is not None
        assert source.used == 256 * MIB
        fd = owner._publication_database.fd
        os.fstat(fd)
        try:
            assert caught.value.cleanup_owner is owner
            assert owner._failure is not None and owner._closing
            assert source.used == 256 * MIB  # barrier forbids premature cleanup
        finally:
            caught.value.__traceback__ = None
            del caught
            release.set()
        with suppress(RuntimeError):
            owner.close()
        assert not owner._thread.is_alive() and source.used == 0
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        release.set()
        for owner in observed:
            with suppress(RuntimeError):
                owner.close()
        runtime.cancel("fix2 launch cleanup")
        runtime.join()


def test_fix2_known_preprimitive_failure_is_not_reported_as_joined(tmp_path, monkeypatch):
    from gf_training import rust_player_storage as module

    runtime, source, kwargs = _fix1_idle_inputs(tmp_path)
    attempted = []

    def no_native_thread(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError("genuine native launch failure before spawn")

    monkeypatch.setattr(threading, "_start_joinable_thread", no_native_thread)
    try:
        with pytest.raises(RuntimeError, match="before spawn") as caught:
            module.RustPlayerStorage(**kwargs)
        owner = caught.value.cleanup_owner
        assert attempted == [True] and source.used == 0
        assert owner._database is None and owner._publication_database is None
        assert not kwargs["directory"].exists()
        # The caller knows this injected primitive never spawned. Production
        # treats an interrupted launch conservatively, without an ident/alive
        # shortcut that could misclassify a real native thread still starting.
        with pytest.raises(RuntimeError, match="before it is started"):
            owner.join()
        caught.value.__traceback__ = None
    finally:
        runtime.cancel("fix2 prelaunch cleanup")
        runtime.join()
