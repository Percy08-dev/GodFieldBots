"""Corrected game DTO/codec/journal/clean process and publication boundaries."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common import card_instance
from gf_common.candidate_columns import (
    decode_candidate_columns,
    decode_candidate_columns_v2,
    encode_candidate_columns,
    encode_candidate_columns_v2,
)
from gf_common.card_instance import (
    ActionConfigV3,
    card_instance_contract,
    corrected_card_instance_contract,
    strength_powder_card_instance_contract,
)
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.card_instance_runtime import (
    APPROVED_BINARY_SHA256,
    APPROVED_CHANGE_FILES,
    APPROVED_CHANGE_SHA256,
    APPROVED_SPEC_FILES,
    approved_runtime_build,
    validate_runtime_receipt,
)
from gf_common.card_instance_validation import decision_input, reconcile_replay, replay_closure
from gf_common.diverse_actions import (
    GameGeneratedCandidatesV3,
    GeneratedCandidatesV3,
    NumericCosts,
    PublicCandidateGeneratorV3,
)
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured import event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature, tokenize_frame
from gf_common.structured_wire import (
    game_generated_dict_v3,
    generated_dict_v3,
    parse_game_generated_v3,
    parse_generated_v3,
)
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.structured_runtime import WorkerCaps
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import CollectionPublication
from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal
from gf_training.structured_journal import read_structured_journal
from tests.card_instance_action_support import inputs
from tests.test_corrected_card_instance import records

ACTIONS = ActionConfigV3(
    action_contract=corrected_card_instance_contract(), max_sequence_length=4, max_candidates=50
)
RULES = card_instance_rules()


def receipt():
    contract = corrected_card_instance_contract()
    return {
        "schema_version": 1,
        "action_contract": asdict(contract),
        "build": {
            "binary_sha256": APPROVED_BINARY_SHA256,
            "spec_inventory_sha256": contract.spec_inventory_sha256,
            "change_inventory_sha256": APPROVED_CHANGE_SHA256,
            "spec_files": dict(APPROVED_SPEC_FILES),
            "change_files": dict(APPROVED_CHANGE_FILES),
        },
        "binary": {
            "path": "/test/runtime",
            "sha256": APPROVED_BINARY_SHA256,
            "identity": [1, 2, 3, 4, 5],
        },
        "source": {
            "path": "/test/source",
            "revision": contract.simulator_revision,
            "spec_files": dict(APPROVED_SPEC_FILES),
            "change_files": dict(APPROVED_CHANGE_FILES),
        },
        "executed": {
            "pid": 123,
            "start_time": 1,
            "sha256": APPROVED_BINARY_SHA256,
            "identity": [1, 2, 3, 4, 5],
        },
    }


def shared_runtime_receipt(contract):
    build = approved_runtime_build(contract)
    value = receipt()
    value.update(schema_version=2, action_contract=asdict(contract), build=build)
    value["binary"]["sha256"] = build["binary_sha256"]
    value["executed"]["sha256"] = build["binary_sha256"]
    value["source"].update(
        revision=contract.simulator_revision, dirty=False,
        spec_files=build["spec_files"], change_files=build["change_files"],
    )
    assert validate_runtime_receipt(value, require_executed=True) == value
    return value


def game(value=None, seed=17, *, actions=ACTIONS):
    frame, bank, delta = records(
        inputs([("fireball", "learned"), ("fireball", "unused"), ("spiritual-staff", None)], mp=20)
        if value is None
        else value
    )
    search_seed = int(
        digest({"recipe": "structured-search-stream-v1", "sampling_seed": seed})[:16], 16
    )
    generated = PublicCandidateGeneratorV3(actions, rules=RULES, clock=lambda: 0.0).generate_game(
        frame, bank.view(), random.Random(search_seed), NumericCosts()
    )
    return frame, bank, delta, generated


def writer(tmp_path, seed=17, *, value=None, actions=ACTIONS, runtime_evidence=None):
    frame, bank, delta, generated = game(value, seed=seed, actions=actions)
    policy = CardInstanceBaseline("heuristic", actions, rules=RULES)
    selected = policy.select_frame(frame, bank.view(), generated)
    command = {
        "command_id": str(uuid.uuid4()),
        "expected_revision": frame.revision,
        "action": selected.validate(generated.table, frame),
    }
    journal = CardInstanceJournal(
        tmp_path,
        "episode",
        1,
        policy.version,
        run_id="run",
        game_id="game",
        actions=actions,
        bank_episode=bank.episode,
        caps=asdict(WorkerCaps(max_positions=4, max_candidates=50)),
        numeric_costs=asdict(NumericCosts()),
        model_id="heuristic",
        runtime_evidence=receipt() if runtime_evidence is None else runtime_evidence,
        sampling_seed=seed,
    )
    return journal, frame, delta, generated, selected, command


def test_explicit_game_context_never_relabels_component_or_old_consumers():
    frame, bank, _, generated = game()
    assert generated.table.purpose == "qualified_game"
    with pytest.raises(ContractError):
        GeneratedCandidatesV3(generated.table, generated.report)
    with pytest.raises(ContractError):
        generated_dict_v3(generated, action_contract=ACTIONS.action_contract, rules=RULES)
    raw = game_generated_dict_v3(generated, action_contract=ACTIONS.action_contract, rules=RULES)
    assert (
        parse_game_generated_v3(raw, action_contract=ACTIONS.action_contract, rules=RULES)
        == generated
    )
    with pytest.raises(ContractError):
        parse_generated_v3(raw, action_contract=ACTIONS.action_contract, rules=RULES)
    component = PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0.0).generate(
        frame, bank.view(), random.Random(17), NumericCosts()
    )
    with pytest.raises(ContractError):
        CardInstanceBaseline("random", ACTIONS, rules=RULES).select_frame(
            frame, bank.view(), component
        )
    with pytest.raises(ContractError):
        PublicCandidateGeneratorV3(
            replace(ACTIONS, action_contract=card_instance_contract()), rules=RULES
        ).generate_game(frame, bank.view(), random.Random(17), NumericCosts())


@pytest.mark.parametrize("kind", ["random", "heuristic"])
def test_actual_clean_worker_returns_original_refs_and_no_ppo_stats(kind):
    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker(kind, 17, ACTIONS, ledger=ledger)
        frame, _, delta, _ = game()
        try:
            generated, selected = await worker.select_frame(frame, delta)
            assert selected.logprob is selected.value is None
            action = selected.validate(generated.table, frame)
            assert all(
                ref in frame.references
                for ref in action.get("ordered_card_refs", action.get("card_refs", []))
            )
            assert selected.policy_version == kind + "-card-instance-v2"
        finally:
            await worker.close()
        assert not ledger.workers
        assert worker.process.poll() is not None

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["child_init", "parent_ready", "child_select", "parent_selection"])
@pytest.mark.parametrize("mutation", ["missing", "preceding_signature"])
def test_protocol3_rejects_preceding_layout_at_every_handshake_before_candidates(
    monkeypatch, boundary, mutation
):
    import gf_simulator_client.card_instance_worker as workers

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("random", 17, ACTIONS, ledger=ledger)
        original = worker._roundtrip

        def alter(message):
            if mutation == "missing":
                del message["layout_signature"]
            else:
                message["layout_signature"] = "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad"

        async def corrupt(payload):
            if ((boundary == "child_init" and "kind" not in payload)
                or (boundary == "child_select" and payload.get("kind") == "select")):
                payload = copy.deepcopy(payload)
                alter(payload)
            received = await original(payload)
            if ((boundary == "parent_ready" and received.get("kind") == "ready")
                or (boundary == "parent_selection" and received.get("kind") == "selection")):
                alter(received)
            return received

        monkeypatch.setattr(worker, "_roundtrip", corrupt)
        monkeypatch.setattr(workers, "parse_game_generated_v3",
            lambda *args, **kwargs: pytest.fail("stale layout reached candidate acceptance"))
        frame, _, delta, _ = game()
        try:
            with pytest.raises((ContractError, RuntimeError)):
                await worker.select_frame(frame, delta)
        finally:
            await worker.close()
        assert worker.process.poll() is not None and not ledger.workers

    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["missing_layout", "stale_layout", "stale_token"])
def test_journal6_rejects_preceding_layout_header_before_bank_allocation(tmp_path, monkeypatch, mutation):
    import gf_training.card_instance_journal as journals

    journal, *_ = writer(tmp_path)
    journal.finish("incomplete", None)
    header = copy.deepcopy(journal.header)
    if mutation == "missing_layout":
        del header["layout_signature"]
    else:
        header["layout_signature" if mutation == "stale_layout" else "token_signature"] = (
            "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad")
    monkeypatch.setattr(journals, "EventBank",
        lambda *args, **kwargs: pytest.fail("stale journal reached bank allocation"))
    with pytest.raises(ContractError):
        journals._Context(ACTIONS, RULES).validate_header(header)


def test_journal6_rejects_valid_receipt_from_different_action_contract(tmp_path, monkeypatch):
    import gf_training.card_instance_journal as journals

    journal, *_ = writer(tmp_path)
    journal.finish("incomplete", None)
    header = copy.deepcopy(journal.header)
    header["runtime_evidence"] = shared_runtime_receipt(strength_powder_card_instance_contract())
    monkeypatch.setattr(journals, "EventBank",
        lambda *args, **kwargs: pytest.fail("foreign runtime reached bank allocation"))
    with pytest.raises(ContractError, match="runtime receipt action contract mismatch"):
        journals._Context(ACTIONS, RULES).validate_header(header)


def test_dream_compat_release_and_powder_behavior_are_pinned_independently():
    old = strength_powder_card_instance_contract()
    new = card_instance.dream_compat_card_instance_contract()
    assert new.profile_id == "card-instance-simulator-v2-shared-20261124"
    assert old.requires_powder_weapon and new.requires_powder_weapon
    assert old.simulator_revision != new.simulator_revision
    assert old.spec_inventory_sha256 != new.spec_inventory_sha256
    assert rule_signature(old, RULES) != rule_signature(new, RULES)
    old_receipt, new_receipt = (shared_runtime_receipt(profile) for profile in (old, new))
    assert old_receipt["binary"]["sha256"] != new_receipt["binary"]["sha256"]
    for profile, original, foreign in ((old, old_receipt, new_receipt), (new, new_receipt, old_receipt)):
        build = original["build"]
        assert digest(build["spec_files"]) == profile.spec_inventory_sha256
        assert digest(build["change_files"]) == build["change_inventory_sha256"]
        for field in ("build", "binary", "source", "executed", "action_contract"):
            with pytest.raises(ContractError):
                validate_runtime_receipt(
                    original | {field: foreign[field]}, require_executed=True, expected_contract=profile
                )


@pytest.mark.parametrize("use_new", [False, True], ids=["old", "new"])
def test_journal6_accepts_own_dream_release_and_rejects_foreign_receipt(tmp_path, use_new):
    old = strength_powder_card_instance_contract()
    new = card_instance.dream_compat_card_instance_contract()
    own, foreign = (new, old) if use_new else (old, new)
    actions = replace(ACTIONS, action_contract=own)
    journal, *_ = writer(tmp_path, actions=actions, runtime_evidence=shared_runtime_receipt(own))
    journal.finish("incomplete", None)
    original = read_card_instance_journal(journal.final_path, actions=actions)
    assert canonical(original.header) == canonical(journal.header)

    lines = journal.final_path.read_bytes().splitlines(keepends=True)
    header = json.loads(lines[0]) | {"runtime_evidence": shared_runtime_receipt(foreign)}
    lines[0] = canonical(header).encode() + b"\n"
    journal.final_path.write_bytes(b"".join(lines))
    with pytest.raises(ContractError, match="runtime receipt action contract mismatch"):
        read_card_instance_journal(journal.final_path, actions=actions)


def test_game_layout_signature_fixed128_is_separate_from_journal_aggregate_window(tmp_path):
    from gf_training.card_instance_journal import _Context

    journal, *_ = writer(tmp_path)
    journal.finish("incomplete", None)
    header = copy.deepcopy(journal.header)
    header.update(aggregate_window=1, token_signature=signature(1, layout=LAYOUT))
    assert header["layout_signature"] == signature(128, layout=LAYOUT)
    assert header["token_signature"] != header["layout_signature"]
    _, bank = _Context(ACTIONS, RULES).validate_header(header)
    assert bank.layout is LAYOUT


def test_guardian_events_survive_actual_worker_and_journal6_original_prefix(tmp_path):
    events = tuple({"type": family, "player": 1 + index % 2,
        "definition_id": "seafood-soup" if family == "guardian_acted" else "earth"}
        for index, family in enumerate(("guardian_set", "guardian_acted", "guardian_left")))
    value = replace(inputs([("steel-armor", None)], usable=()), events=events)
    journal, frame, delta, generated, selected, command = writer(tmp_path, value=value)
    index = journal.begin_frame(frame, delta, generated, selected, command, {})
    journal.finish("incomplete", None)
    data = read_card_instance_journal(journal.final_path, actions=ACTIONS)
    assert data.bank.events() == events
    assert data.bank.boundary == frame.boundary
    assert data.decisions[index].frame == frame
    tokens = tokenize_frame(frame, data.bank, layout=LAYOUT)
    assert tokens.history[359] == 2
    assert tokens.history[63 + LAYOUT.event_categories[1].index("seafood-soup")] == 1

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("random", 17, ACTIONS, ledger=ledger)
        try:
            generated, selected = await worker.select_frame(frame, delta)
            assert selected.validate(generated.table, frame) == {
                "type": "discard", "card_refs": list(frame.references)}
        finally:
            await worker.close()
        assert not ledger.workers

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    ["foreign_delta", "stale_frame", "profile", "reply_table", "reply_policy", "reply_stats"],
)
def test_malformed_parent_or_child_reply_reaps_and_releases(monkeypatch, mutation):
    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("random", 17, ACTIONS, ledger=ledger)
        frame, _, delta, _ = game()
        if mutation == "foreign_delta":
            delta = event_delta(
                2, delta.episode, delta.start, delta.previous_hash, delta.events, layout=LAYOUT
            )
        elif mutation == "stale_frame":
            worker._last_frame = frame
        elif mutation == "profile":
            frame = records(
                layout=__import__(
                    "gf_common.structured_layout", fromlist=["CARD_INSTANCE_LAYOUT"]
                ).CARD_INSTANCE_LAYOUT
            )[0]
        else:
            original = worker._roundtrip

            async def corrupt(payload):
                received = await original(payload)
                if received.get("kind") == "selection":
                    if mutation == "reply_table":
                        received["generated"]["table"]["frame_hash"] = "f" * 64
                    elif mutation == "reply_policy":
                        received["selection"]["policy_version"] = "foreign"
                    else:
                        received["selection"]["logprob"] = 0.0
                return received

            monkeypatch.setattr(worker, "_roundtrip", corrupt)
        with pytest.raises((ContractError, RuntimeError)):
            await worker.select_frame(frame, delta)
        assert worker.process.poll() is not None and not ledger.workers

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation", ["codec", "profile", "codebook", "offset", "kind", "padding", "hash", "slot"]
)
def test_codec2_rejects_mixed_and_rehashed_corruption(mutation):
    _, _, _, generated = game()
    payload, desc = encode_candidate_columns_v2(generated.table, "owned.columns", layout=LAYOUT)
    assert (
        decode_candidate_columns_v2(
            payload,
            desc,
            action_contract=ACTIONS.action_contract,
            purpose="qualified_game",
            synthetic_capacity=False,
            layout=LAYOUT,
        )
        == generated.table
    )
    if mutation == "codec":
        desc["codec"] = "candidate-columns-v1"
    elif mutation == "profile":
        desc["action_contract"] = asdict(card_instance_contract())
    elif mutation == "codebook":
        desc["codebook_sha256"] = "f" * 64
    elif mutation == "offset":
        desc["columns"][0]["offset"] = 1
    elif mutation == "hash":
        desc["sha256"] = "f" * 64
    else:
        raw = bytearray(payload)
        columns = {c["name"]: c for c in desc["columns"]}
        location = (
            columns["kinds" if mutation == "kind" else "slots"]["offset"]
            if mutation != "padding"
            else columns["numerics"]["offset"] - 1
        )
        raw[location] = 255
        payload = bytes(raw)
        desc["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ContractError):
        decode_candidate_columns_v2(
            payload,
            desc,
            action_contract=ACTIONS.action_contract,
            purpose="qualified_game",
            synthetic_capacity=False,
            layout=LAYOUT,
        )
    with pytest.raises(ContractError):
        encode_candidate_columns(generated.table, "owned.columns")
    with pytest.raises(ContractError):
        decode_candidate_columns(
            payload,
            desc,
            action_contract=ACTIONS.action_contract,
            purpose="qualified_game",
            synthetic_capacity=False,
        )


def test_journal6_transaction_pending_ack_and_sidecar_reconstruction(tmp_path):
    journal, frame, delta, generated, selected, command = writer(tmp_path)
    index = journal.begin_frame(frame, delta, generated, selected, command, {"generation_ms": 0.0})
    with pytest.raises(ContractError):
        journal.acknowledge(
            index,
            {
                "command_id": command["command_id"],
                "previous_revision": frame.revision,
                "state_revision": frame.revision + 2,
                "status": "active",
                "result": None,
            },
        )
    assert journal.pending == {0}
    journal.finish("incomplete", None)
    data = read_card_instance_journal(journal.final_path, actions=ACTIONS)
    assert data.accepted == () and data.end["pending_decisions"] == [0]
    assert data.decisions[0].generated.table == generated.table
    assert data.decisions[0].command == command
    with pytest.raises(ContractError):
        read_structured_journal(
            journal.final_path,
            action_contract=ACTIONS.action_contract,
            purpose="qualified_game",
            synthetic_capacity=False,
        )


def test_journal6_positive_ack_natural_and_corrupt_original_sidecar(tmp_path):
    journal, frame, delta, generated, selected, command = writer(tmp_path)
    index = journal.begin_frame(frame, delta, generated, selected, command, {})
    outcome = {"win": {"winner": 1}}
    journal.acknowledge(
        index,
        {
            "command_id": command["command_id"],
            "previous_revision": frame.revision,
            "state_revision": frame.revision + 1,
            "status": "terminal",
            "result": outcome,
        },
    )
    final = copy.deepcopy(frame.observation)
    final.update(
        phase="terminal",
        current_player=None,
        actionable=False,
        required_choice=None,
        typed_action_schema=[],
        result=outcome,
        state_revision=frame.revision + 1,
    )
    journal.finish("natural", outcome, final)
    data = read_card_instance_journal(journal.final_path, actions=ACTIONS)
    assert data.accepted == (0,) and data.end["terminal_reward"] == 1
    sidecar = next(tmp_path.glob("*.columns"))
    sidecar.write_bytes(b"bad")
    with pytest.raises(ContractError):
        read_card_instance_journal(journal.final_path, actions=ACTIONS)


@pytest.mark.parametrize("failure", ["sidecar", "row", "command", "stats"])
def test_journal_publication_failure_never_resolves_pending(tmp_path, monkeypatch, failure):
    journal, frame, delta, generated, selected, command = writer(tmp_path)
    if failure == "sidecar":
        monkeypatch.setattr(
            journal._context,
            "publish_candidates",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("sidecar")),
        )
    elif failure == "row":
        original = journal.write

        def write(row):
            if row["kind"] == "decision":
                raise OSError("row")
            original(row)

        monkeypatch.setattr(journal, "write", write)
    elif failure == "command":
        command["action"] = {"type": "pray"}
    else:
        selected = replace(selected, value=0.0)
    with pytest.raises((ContractError, OSError)):
        journal.begin_frame(frame, delta, generated, selected, command, {})
    assert not journal.pending and not journal.commands
    journal.close_incomplete()
    assert not journal.final_path.exists()


@pytest.mark.parametrize("field", ["build", "binary", "source", "executed"])
def test_runtime_receipt_approval_is_closed(field):
    raw = receipt()
    assert validate_runtime_receipt(raw, require_executed=True) == raw
    raw[field][
        "sha256"
        if field in ("binary", "executed")
        else "binary_sha256"
        if field == "build"
        else "revision"
    ] = "f" * 64
    with pytest.raises(ContractError):
        validate_runtime_receipt(raw, require_executed=True)


def test_exec_swap_cannot_be_hidden_by_restored_path_receipt(tmp_path):
    # Even a previously validated/restored pathname cannot vouch for this live Python child.
    owner = CardInstanceRuntimeEvidence(tmp_path / "restored-approved-path", tmp_path)
    owner._receipt = receipt()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        with pytest.raises(ContractError, match="executed binary"):
            owner.verify_executed(process)
    finally:
        process.kill()
        process.wait()


def test_publication_healthy_commit_and_late_deadline_owned_invalidation(tmp_path, monkeypatch):
    owner = CollectionPublication(
        tmp_path, work_deadline=time.monotonic() + 10, cleanup_deadline=time.monotonic() + 20
    )
    original = owner._check

    def late():
        original()
        if (tmp_path / "manifest.json").exists():
            raise TimeoutError("late")

    monkeypatch.setattr(owner, "_check", late)
    with pytest.raises(TimeoutError):
        owner.commit({"healthy": True}, {"schema_version": 5})
    assert (tmp_path / "report.json").read_bytes().startswith(b"!")
    assert not owner.files and owner.directory_fd is None


def test_publication_preserves_foreign_replaced_path_and_invalidates_held_inode(
    tmp_path, monkeypatch
):
    owner = CollectionPublication(
        tmp_path, work_deadline=time.monotonic() + 10, cleanup_deadline=time.monotonic() + 20
    )
    original = owner._write
    saved = tmp_path / "saved-report"

    def substitute(name, value):
        original(name, value)
        if name == "report.json":
            (tmp_path / name).rename(saved)
            (tmp_path / name).write_text("FOREIGN")
            raise OSError("after replace")

    monkeypatch.setattr(owner, "_write", substitute)
    with pytest.raises(OSError):
        owner.commit({"healthy": True}, {"schema_version": 5})
    assert (tmp_path / "report.json").read_text() == "FOREIGN" and saved.read_bytes().startswith(
        b"!"
    )


def test_replay_zero_no_action_aborted_and_concealed_are_not_acceptance():
    raw = json.loads(
        (Path(__file__).parent / "fixtures/card_instance_v2_runtime_replay.json").read_text()
    )
    reconcile_replay(raw, raw["viewer_player_id"], [], None)
    assert replay_closure(raw, raw["game_id"], raw["viewer_player_id"]) is None
    value = inputs([("fireball", "learned")])
    obs = value.observation
    raw["viewer_player_id"] = 1
    row = {
        "actor": 1,
        "observation": obs,
        "action_options": None,
        "visible_action": None,
        "visible_events": [],
        "aborted": False,
    }
    raw["turns"] = [{"turn": obs["turn"], "active_player": 1, "decisions": [row]}]
    reconcile_replay(raw, 1, [], None)
    row["visible_action"] = {"type": "concealed"}
    with pytest.raises(ContractError):
        reconcile_replay(raw, 1, [], None)
    row.update(aborted=True, visible_action=None)
    row["observation"] = copy.deepcopy(obs)
    row["observation"].update(actionable=False, typed_action_schema=[])
    reconcile_replay(raw, 1, [], None)
    with pytest.raises(ContractError):
        reconcile_replay(raw, 1, [{"type": "pray"}], None)


def test_input_builder_preserves_original_refs_order_and_own_catalog_only():
    value = inputs([("fireball", "learned"), ("fireball", "unused"), ("spiritual-staff", None)])
    rebuilt = decision_input(
        value.observation,
        list(value.events),
        value.history_end,
        value.catalog,
        {key: value.action_options[key] for key in ("state_revision", "action_options")},
    )
    assert rebuilt == value
    assert list(rebuilt.catalog) == sorted(
        {card["definition_id"] for card in value.observation["own"]["cards"]}
    )


def test_worker_scrubs_management_argv_env_and_only_sends_safe_own_inputs(monkeypatch):
    monkeypatch.setenv("GAME_SEED", "management-secret")
    monkeypatch.setenv("GF_SERVER_URL", "http://management-secret")

    async def run():
        worker = CardInstanceParticipantWorker("random", 17, ACTIONS, ledger=ReservationLedger())
        try:
            environment = Path(f"/proc/{worker.process.pid}/environ").read_bytes()
            argv = Path(f"/proc/{worker.process.pid}/cmdline").read_bytes()
            assert b"management-secret" not in environment + argv
            assert b"GAME_SEED" not in environment
            assert set(worker.initial) == {
                "protocol_version",
                "model_id",
                "sampling_seed",
                "actions",
                "rules",
                "caps",
                "numeric_costs",
                "action_contract",
                "rule_signature",
                "layout_signature",
                "purpose",
                "synthetic_capacity",
            }
            assert "factory" not in worker.initial
            frame, _, delta, _ = game()
            messages = []
            original = worker._roundtrip

            async def inspect(payload):
                messages.append(copy.deepcopy(payload))
                return await original(payload)

            monkeypatch.setattr(worker, "_roundtrip", inspect)
            await worker.select_frame(frame, delta)
            assert set(messages[-1]) == {
                "protocol_version",
                "kind",
                "frame",
                "delta",
                "action_contract",
                "rule_signature",
                "layout_signature",
                "purpose",
                "synthetic_capacity",
            }
            assert "management-secret" not in canonical(messages)
        finally:
            await worker.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_worker_deadline_or_cancel_reaps_before_reservation_release(monkeypatch, cancel):
    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("random", 17, ACTIONS, ledger=ledger)
        if not cancel:
            worker.timeout = 0.000001
        task = asyncio.create_task(worker.initialize())
        if cancel:
            await asyncio.sleep(0)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert worker.process.poll() is not None
        assert not ledger.workers

    asyncio.run(run())


def test_publication_success_has_no_reportable_checks_after_release(tmp_path, monkeypatch):
    owner = CollectionPublication(
        tmp_path, work_deadline=time.monotonic() + 10, cleanup_deadline=time.monotonic() + 20
    )
    original = owner._release

    def release():
        original()
        monkeypatch.setattr(owner, "_check", lambda: pytest.fail("post-release check"))

    monkeypatch.setattr(owner, "_release", release)
    owner.commit({"healthy": True}, {"schema_version": 5})
    assert owner.committed and not owner.files
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"schema_version": 5}


def test_unconfirmed_publication_revocation_retains_reachable_quarantine_owner(
    tmp_path, monkeypatch
):
    owner = CollectionPublication(
        tmp_path, work_deadline=time.monotonic() + 10, cleanup_deadline=0.0
    )
    original = owner._write

    def fail(name, value):
        original(name, value)
        raise OSError("publication fail")

    monkeypatch.setattr(owner, "_write", fail)
    with pytest.raises(OSError) as error:
        owner.commit({"healthy": True}, {"schema_version": 5})
    retained = error.value._card_instance_collection_cleanup_owner
    assert retained is owner and retained.quarantined and retained.files
    # Explicit test cleanup retries the same original owner with a test allowance.
    retained.cleanup_deadline = time.monotonic() + 10
    retained.revoke()
    assert not retained.files


def test_runtime_binary_mismatch_precedes_environment_output_or_popen(tmp_path, monkeypatch):
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    from gf_simulator_client.environment import EnvironmentSettings

    bad = tmp_path / "bad-binary"
    bad.write_bytes(b"not-approved")
    owner = CardInstanceRuntimeEvidence(bad, tmp_path)
    monkeypatch.setattr(owner, "_source", lambda _build: receipt()["source"])
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("mismatch reached process allocation")
    )
    out = tmp_path / "uncreated"
    settings = EnvironmentSettings(str(bad), "unused", out, "run", ACTIONS)
    with pytest.raises(ContractError):
        CardInstanceSimulatorEnvironment(
            settings, CardInstanceJournal, ledger=ReservationLedger(), runtime_evidence=owner
        )
    assert not out.exists()


def test_runtime_closed_approval_maps_and_source_recheck(monkeypatch, tmp_path):
    import gf_simulator_client.card_instance_runtime as module

    owner = CardInstanceRuntimeEvidence(tmp_path / "binary", tmp_path)
    expected = receipt()
    monkeypatch.setattr(
        module, "_read_hash", lambda path, deadline: (APPROVED_BINARY_SHA256, [1, 2, 3, 4, 5])
    )
    monkeypatch.setattr(owner, "_source", lambda _build: expected["source"])
    assert owner.preflight()["executed"] is None
    owner._receipt["executed"] = expected["executed"]
    assert owner.recheck()["executed"] == expected["executed"]
    monkeypatch.setattr(module, "_read_hash", lambda path, deadline: ("f" * 64, [1, 2, 3, 4, 5]))
    with pytest.raises(ContractError):
        owner.recheck()


def test_natural_reward_requires_both_replay_closures_before_journal_finish():
    from gf_common.domain import EpisodeResult
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment

    environment = object.__new__(CardInstanceSimulatorEnvironment)
    result = EpisodeResult("episode", "pair", "seed", ("random", "heuristic"))
    result.end_kind = "natural"
    result.outcome = "draw"

    class Sink:
        def finish(self, *args):
            pytest.fail("unreconciled natural reward reached journal")

    with pytest.raises(ContractError):
        environment._finish_journal(Sink(), result, None)


def restricted_game(value, actions):
    from gf_common.structured import CandidateTable

    frame, bank, _, generated = game(value)
    table = CandidateTable.from_actions(
        frame,
        actions(frame),
        action_contract=ACTIONS.action_contract,
        purpose="qualified_game",
        synthetic_capacity=False,
    )
    report = replace(
        generated.report,
        table_hash=table.table_hash,
        mandatory=0,
        selected=len(table.kinds),
        selected_elements=len(table.slots),
        discovered=len(table.kinds),
        channel_counts=(len(table.kinds), 0, 0, 0, 0, 0, 0),
        traversal_complete=True,
        presentation_complete=True,
    )
    return frame, bank, GameGeneratedCandidatesV3(table, report)


def test_heuristic_uses_adjacent_payment_and_new_attack_analysis():
    value = inputs(
        [
            ("magical-stick", None),
            ("fireball", "learned"),
            ("spiritual-staff", None),
            ("fireball", "unused"),
        ],
        mp=10,
    )
    frame, bank, generated = restricted_game(
        value,
        lambda f: [
            {
                "type": "play_sequence",
                "ordered_card_refs": [f.references[i] for i in (0, 1, 2, 3)],
                "target": 2,
            },
            {"type": "play_sequence", "ordered_card_refs": [f.references[0]], "target": 2},
        ],
    )
    selected = CardInstanceBaseline("heuristic", ACTIONS, rules=RULES).select_frame(
        frame, bank.view(), generated
    )
    # Only the immediately preceding Fireball is waived. The remaining payment
    # reduces the dynamic weapon; equal attack therefore favors the shorter row.
    assert selected.index == 1


def test_heuristic_suppresses_costcut_armor_defense_role():
    value = inputs(
        [("wall", "learned"), ("spiritual-hood", None), ("iron-armor", None)],
        mp=0,
        choice="defense",
    )
    frame, bank, generated = restricted_game(
        value,
        lambda f: [
            {"type": "respond", "ordered_card_refs": [f.references[0], f.references[1]]},
            {"type": "respond", "ordered_card_refs": [f.references[2]]},
        ],
    )
    selected = CardInstanceBaseline("heuristic", ACTIONS, rules=RULES).select_frame(
        frame, bank.view(), generated
    )
    assert selected.index == 1  # Hood's nine defense is suppressed by its CostCut role.


@pytest.mark.parametrize("mutation", [None, "wrapper_revision", "inner_revision", "inner_actor", "already_bound"])
def test_actual_raw_http_fixture_binds_own_view_and_builds_original_frame(mutation):
    from gf_common.card_instance_validation import catalog
    from gf_common.structured import DecisionFrame, EventBank, event_delta
    fixture_dir = Path(__file__).parent / "fixtures"
    def fixture(name):
        return json.loads((fixture_dir / ("card_instance_v2_" + name + ".json")).read_text())
    obs = fixture("runtime_observation")
    raw = fixture("runtime_options")
    feed = fixture("runtime_events")
    definitions = catalog(fixture("catalog"))
    assert set(raw) == {"state_revision", "action_options"}
    if mutation == "wrapper_revision":
        raw["state_revision"] += 1
    elif mutation == "inner_revision":
        raw["action_options"]["state_revision"] += 1
    elif mutation == "inner_actor":
        raw["action_options"]["actor"] = 3 - obs["viewer_player_id"]
    elif mutation == "already_bound":
        raw["viewer_player_id"] = obs["viewer_player_id"]
    before = copy.deepcopy(raw)
    if mutation is not None:
        with pytest.raises(ContractError):
            decision_input(obs, feed["events"], feed["cursor"], definitions, raw)
        assert raw == before
        return
    value = decision_input(obs, feed["events"], feed["cursor"], definitions, raw)
    assert raw == before  # raw HTTP fixture remains a separate original record.
    assert value.action_options == {**raw, "viewer_player_id": obs["viewer_player_id"]}
    value.action_options["action_options"]["candidate_groups"].clear()
    assert raw == before  # participant binding owns an independent clone.
    value = decision_input(obs, feed["events"], feed["cursor"], definitions, raw)
    bank = EventBank(obs["viewer_player_id"], "raw-http-binding", layout=LAYOUT)
    delta = event_delta(bank.actor, bank.episode, 0, bank.boundary.prefix_hash, value.events, layout=LAYOUT)
    bank.append(delta)
    frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
    assert frame.references == tuple(card["card_ref"] for card in obs["own"]["cards"])
    assert frame.raw["action_options"]["viewer_player_id"] == frame.actor
    assert frame.raw["action_options"]["state_revision"] == frame.revision
