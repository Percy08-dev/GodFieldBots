"""Memory journals retain original bytes and share bounded storage ownership."""
from __future__ import annotations

import time

import pytest

from gf_common.group_pack import StorageSettings
from tests.test_card_instance_baseline_runtime import ACTIONS, receipt, writer


def test_memory_journal_matches_file_bytes_and_logical_evidence(tmp_path):
    from gf_training.card_instance_journal import read_card_instance_journal
    from gf_training.group_pack_memory import MemoryCardInstanceJournal, MemoryGroupArtifacts
    from gf_training.group_pack_resources import StorageBudget

    folder = tmp_path / "episode"
    folder.mkdir()
    physical, frame, delta, generated, selected, command = writer(folder)
    budget = StorageBudget(StorageSettings(), time.monotonic() + 30)
    sink = MemoryGroupArtifacts(budget)
    memory = MemoryCardInstanceJournal(folder, "episode", 1, physical.policy,
        run_id="run", game_id="game", actions=ACTIONS, bank_episode=physical.bank.episode,
        caps=physical.caps, numeric_costs=physical.header["numeric_costs"], model_id="heuristic",
        runtime_evidence=receipt(), sampling_seed=17, artifact_sink=sink)
    for journal in (physical, memory):
        journal.begin_frame(frame, delta, generated, selected, command, {})
        journal.finish("cancelled", None)
    snapshot = sink.snapshot()
    assert snapshot.read("episode/player-1.jsonl") == physical.final_path.read_bytes()
    for path in folder.glob("*.columns"):
        assert snapshot.read("episode/" + path.name) == path.read_bytes()
    parsed = read_card_instance_journal(memory.final_path, actions=ACTIONS,
        artifacts=snapshot, artifact_prefix="episode")
    old = read_card_instance_journal(physical.final_path, actions=ACTIONS)
    assert parsed.records == old.records
    assert parsed.decisions[0].generated.table == old.decisions[0].generated.table
    assert parsed.sidecars == ()
    assert parsed.logical_sidecars[0].kind == "logical-candidate-bytes-v1"
    assert parsed.logical_sidecars[0].sha256 == old.sidecars[0]["sha256"]
    with pytest.raises((ValueError, RuntimeError)):
        sink.write("episode/late", b"late")
    assert budget.used > 0
    snapshot.close()
    sink.close()
    assert budget.used == 0


def test_memory_store_budget_freeze_and_partial_preservation(tmp_path):
    from gf_training.group_pack_memory import MemoryGroupArtifacts
    from gf_training.group_pack_resources import StorageBudget

    budget = StorageBudget(StorageSettings(), time.monotonic() + 30)
    sink = MemoryGroupArtifacts(budget)
    sink.write("episode/player-1.partial.jsonl", b"header\n")
    sink.append("episode/player-1.partial.jsonl", b"decision\n")
    with pytest.raises(ValueError):
        sink.write("episode/player-1.partial.jsonl", b"overwrite")
    with pytest.raises(ValueError):
        sink.write("../outside", b"no")
    reservation = budget.reserve(budget.limit - budget.used)
    with pytest.raises(ValueError, match="budget"):
        sink.append("episode/player-1.partial.jsonl", b"unadmitted")
    reservation.close()
    snapshot = sink.snapshot()
    assert snapshot.read("episode/player-1.partial.jsonl") == b"header\ndecision\n"
    sink.close()
    assert budget.used > 0  # snapshot owns the chunks and joined original bytes.
    snapshot.close()
    assert budget.used == 0


@pytest.mark.parametrize("prefix", ["../episode", "other", "episode/nested", "/episode"])
def test_logical_reader_rejects_cross_episode_prefix(tmp_path, prefix):
    from gf_training.card_instance_journal import read_card_instance_journal
    from gf_training.group_pack_memory import MemoryGroupArtifacts
    from gf_training.group_pack_resources import StorageBudget

    budget = StorageBudget(StorageSettings(), time.monotonic() + 30)
    sink = MemoryGroupArtifacts(budget)
    sink.write("episode/player-1.jsonl", b"{}\n")
    snapshot = sink.snapshot()
    try:
        with pytest.raises(ValueError):
            read_card_instance_journal(tmp_path / "episode/player-1.jsonl", actions=ACTIONS,
                                      artifacts=snapshot, artifact_prefix=prefix)
    finally:
        snapshot.close()
        sink.close()


def test_storage_budget_reservations_share_ram_and_thread_safe_capacity():
    from concurrent.futures import ThreadPoolExecutor

    from gf_training.group_pack_resources import StorageBudget

    budget = StorageBudget(StorageSettings(), time.monotonic() + 30)
    budget.measure_ram(1024)
    with ThreadPoolExecutor(max_workers=4) as pool:
        held = list(pool.map(budget.reserve, [4096] * 20))
    assert budget.used == 20 * 4096 + 1024
    held[0].resize(8192)
    assert budget.peak_bytes == 21 * 4096 + 1024
    for item in held:
        item.close()
        item.close()
    assert budget.used == budget.ram_bytes == 1024
    budget.measure_ram(0)
    assert budget.used == 0


def test_filesystem_source_rejects_logical_candidate_evidence(tmp_path, monkeypatch):
    from dataclasses import replace

    from gf_training import card_instance_journal
    from tests.test_card_instance_source import structural_source

    original = card_instance_journal.read_card_instance_journal
    def logical(*args, **kwargs):
        return replace(original(*args, **kwargs), logical_sidecars=(object(),))
    monkeypatch.setattr(card_instance_journal, "read_card_instance_journal", logical)
    with pytest.raises(ValueError, match="filesystem Source rejects logical"), structural_source(tmp_path, decisions=False):
        pytest.fail("logical byte evidence was promoted through a filesystem Source")
