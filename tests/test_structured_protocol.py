"""Real clean child boundary and parent-only typed journal5 transaction evidence."""
from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV2
from gf_common.domain import ContractError, digest
from gf_common.structured import ActionConfigV2, DecisionFrame, EventBank, EventDelta
from gf_common.structured_wire import (
    delta_dict,
    frame_dict,
    parse_delta,
    parse_frame,
)
from gf_common.validation import decision_input
from gf_models.structured_baselines import make_structured_policy
from gf_models.structured_binding import input_binding
from gf_models.structured_runtime import own_worker_settings
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from gf_training.structured_journal import StructuredJournal, read_structured_journal
from tests.structured_worker_fixtures import bad_hash, clean_probe, slow


def action_settings():
    return ActionConfigV2(max_sequence_length=2, max_candidates=30, max_expansions=2000, action_contract=engineering_contract())


def own_frame(value, bank, revision, events=()):
    delta = EventDelta(bank.actor, bank.episode, bank.boundary.end, bank.boundary.prefix_hash, events)
    interval_start = bank.boundary.end
    bank.append(delta)
    raw = value.to_dict()
    raw["observation"]["state_revision"] = revision
    raw["action_options"]["state_revision"] = revision
    current = decision_input(raw["observation"], list(bank.view().events()), raw["history_end"], raw["catalog"], raw["action_options"])
    frame = DecisionFrame.from_input(current, bank, interval_start)
    return frame, delta


def test_protocol2_real_clean_process_two_frames_null_values_and_reap(make_input, monkeypatch):
    monkeypatch.setenv("PRIVATE_SENTINEL", "must not cross")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-private-visibility")
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(clean_probe, "random", 2**64 - 1, action_settings(), ledger=ledger)
        try:
            assert len(ledger.workers) == 1
            await worker.initialize()
            assert worker.startup_elapsed_ms is not None
            bank = EventBank(1, "opaque-own-episode")
            value = make_input(hand=("bronze-club", "steel-armor"), learned=("fireball",))
            previous = None
            for revision in (0, 2):
                frame, delta = own_frame(value, bank, revision, [{"type": "action_accepted", "player": 1}])
                generated, selected = await worker.select_frame(frame, delta)
                assert selected.logprob is selected.value is None
                assert selected.validate(generated.table, frame) == generated.table.to_action(selected.index, frame)
                assert generated.report.numeric_costs_hash == digest(asdict(NumericCosts()))
                if previous is not None:
                    assert generated.report.rng_before == previous
                previous = generated.report.rng_after
                assert ledger.workers[worker.lease.key].decisions == {}
                assert len(ledger.workers) == 1  # reply never releases process residency/slot.
                assert worker.last_timings["parent_request_ms"] >= worker.last_timings["worker_decision_ms"]
            assert set(worker.initial) == {"protocol_version", "factory", "model_id", "sampling_seed", "actions", "policy_options", "caps", "numeric_costs",
                                           "action_contract", "purpose", "synthetic_capacity"}
        finally:
            await worker.close()
        assert worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "parent-private-visibility"


def test_bad_worker_table_hash_aborts_and_reaps_before_reservation_release(make_input):
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(bad_hash, "random", 0, action_settings(), ledger=ledger)
        bank = EventBank(1, "opaque")
        frame, delta = own_frame(make_input(), bank, 0)
        with pytest.raises(RuntimeError, match="mismatch"):
            await worker.select_frame(frame, delta)
        assert worker.closed and worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_worker_timeout_reaps_before_releasing_owned_slot(make_input):
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(slow, "random", 0, action_settings(), 1., ledger=ledger)
        await worker.initialize()
        worker.timeout = .1
        frame, delta = own_frame(make_input(), EventBank(1, "opaque"), 0)
        with pytest.raises(TimeoutError):
            await worker.select_frame(frame, delta)
        assert worker.closed and worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_parent_rejects_foreign_delta_and_gap_without_acting(make_input):
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(clean_probe, "random", 0, action_settings(), ledger=ledger)
        frame, delta = own_frame(make_input(), EventBank(1, "opaque"), 0)
        foreign = EventDelta(2, delta.episode, delta.start, delta.previous_hash, delta.events)
        with pytest.raises(ContractError, match="actor"):
            await worker.select_frame(frame, foreign)
        assert worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_global_reservations_cover_startup_reply_failure_and_update_ownership():
    ledger = ReservationLedger(100, 2, learner_bytes=20)
    first = ledger.reserve_worker(30)
    second = ledger.reserve_worker(30)
    with pytest.raises(ContractError, match="startup"):
        ledger.reserve_worker(0)
    process1, process2 = SimpleNamespace(pid=101, poll=lambda: None), SimpleNamespace(pid=102, poll=lambda: 0)
    first.spawned(process1)
    second.spawned(process2)
    scope = first.reserve_decision(20)
    assert ledger.cuda_bytes == 100
    with pytest.raises(ContractError, match="budget"):
        second.reserve_decision(1)
    first.reply(scope)
    assert ledger.cuda_bytes == 80 and len(ledger.workers) == 2
    with pytest.raises(ContractError, match="reap"):
        first.reaped(process1)
    assert not first.released
    with pytest.raises(ContractError, match="overlaps"):
        ledger.begin_update(0)
    process1.poll = lambda: 0
    first.reaped(process1)
    second.reaped(process2)
    ledger.begin_update(80)
    with pytest.raises(ContractError):
        ledger.reserve_worker(0)
    ledger.finish_update()
    pending = ledger.reserve_worker(1)
    pending.cancel_before_spawn()
    assert not ledger.workers and ledger.cuda_bytes == 20


def test_owned_wire_roundtrips_and_unknown_fields_fail(make_input):
    frame, delta = own_frame(make_input(), EventBank(1, "wire"), 0)
    assert parse_frame(json.loads(json.dumps(frame_dict(frame)))) == frame
    assert parse_delta(json.loads(json.dumps(delta_dict(delta)))) == delta
    with pytest.raises(ContractError):
        parse_frame({**frame_dict(frame), "events": []})
    with pytest.raises(ContractError):
        parse_delta({**delta_dict(delta), "sequence_options": []})


def journal_setup(make_input, tmp_path, seed=17, caps_override=None, model_id="random", model_version="1",
                  events=None, input_value=None, aggregate_window=128):
    import random
    settings = action_settings()
    caps, costs, _, _ = own_worker_settings("random", settings, None)
    bank = EventBank(1, "journal-owned-bank")
    frame, delta = own_frame(make_input(hand=(), learned=("fireball",)) if input_value is None else input_value,
                             bank, 0, [{"type": "turn_started", "turn": 1, "active_player": 1}] if events is None else events)
    search_seed = int(digest({"recipe": "structured-search-stream-v1", "sampling_seed": seed})[:16], 16)
    generated = PublicCandidateGeneratorV2(settings, clock=lambda: 0.).generate(frame, bank.view(), random.Random(search_seed), costs)
    policy = make_structured_policy("random", action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    policy.reset(seed)
    selection = policy.select_frame(frame, bank.view(), generated.table)
    declared_caps = asdict(caps)
    if caps_override:
        declared_caps.update(caps_override)
    writer = StructuredJournal(tmp_path, "episode", 1, policy.version, run_id="run", game_id="game",
        action_config_hash=digest(asdict(settings)), bank_episode=bank.episode, caps=declared_caps, numeric_costs=asdict(costs),
        model_id=model_id, model_version=model_version, sampling_seed=seed, aggregate_window=aggregate_window, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    command = {"command_id": str(uuid.uuid4()), "expected_revision": frame.revision,
               "action": selection.validate(generated.table, frame)}
    return writer, frame, delta, generated, selection, command


def test_journal5_sidecar_reference_append_ack_natural_source_inventory(make_input, tmp_path):
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path)
    index = writer.begin_frame(frame, delta, generated, selection, command, {"parent_request_ms": 1.})
    row = json.loads(writer.path.read_text().splitlines()[-1])
    assert "columns" in row["table"] and "kinds" not in row["table"] and "slots" not in row["table"]
    assert "table" not in row["search_report"] and "events" not in row["frame"]["payload"]
    acknowledgement = {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1,
                       "status": "terminal", "result": {"win": {"winner": 1}}}
    writer.acknowledge(index, acknowledgement)
    final = frame.observation
    final.update(state_revision=1, actionable=False, typed_action_schema=[], required_choice=None,
                 current_player=None, phase="terminal", result=acknowledgement["result"])
    writer.finish("natural", acknowledgement["result"], final)
    data = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    assert data.accepted == (0,) and data.end["terminal_reward"] == 1
    assert data.decisions[0].generated.table == generated.table
    assert data.decisions[0].selection.logprob is data.decisions[0].selection.value is None
    assert len(data.sidecars) == 1 and data.sidecars[0]["sha256"] == row["table"]["sha256"]
    assert data.bank.boundary == frame.boundary
    assert sum(record["kind"] == "events" for record in data.records) == 1


@pytest.mark.parametrize("failure", ["sidecar", "row"])
def test_journal5_publication_or_row_failure_returns_no_action_authorization(make_input, tmp_path, monkeypatch, failure):
    import gf_training.structured_journal as module
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path)
    if failure == "sidecar":
        def fail(*args, **kwargs):
            raise OSError("sidecar failed")
        monkeypatch.setattr(module, "publish_candidates", fail)
    else:
        actual = writer.write
        def fail(record):
            if record["kind"] == "decision":
                raise OSError("row failed")
            actual(record)
        monkeypatch.setattr(writer, "write", fail)
    try:
        with pytest.raises(OSError, match="failed"):
            writer.begin_frame(frame, delta, generated, selection, command, {})
        assert writer.count == 0 and not writer.pending
        if failure == "row":
            assert list(tmp_path.glob("*.columns"))  # retained unaccepted orphan evidence.
    finally:
        writer.close_incomplete()


def test_journal5_rejects_mutated_sidecar_and_interrupted_reward_invention(make_input, tmp_path):
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path)
    index = writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1,
                              "status": "active", "result": None})
    writer.finish("truncated", None)
    source = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    assert source.end["outcome"] is source.end["terminal_reward"] is None
    rows = copy.deepcopy(list(source.records))
    rows[-1]["terminal_reward"] = -1
    with pytest.raises(ContractError, match="invented"):
        read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    sidecar = tmp_path / rows[2]["table"]["file"]
    sidecar.write_bytes(sidecar.read_bytes() + b"corrupt")
    with pytest.raises(ContractError):
        read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def neural_component(tmp_path, kind="gru", actions=None):
    from gf_models.structured_binding import safe_options_v2
    from gf_training.structured_numeric import Learner
    from tests.test_structured_learning import learner_for
    settings = action_settings() if actions is None else actions
    base = learner_for(kind)
    learner = Learner(base.model, 17, base.binding, digest(asdict(settings)), digest({"component": "protocol2-only"}), action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    identity = learner.save(tmp_path / "component-checkpoint")
    options = safe_options_v2({"version": "2", "checkpoint": str(tmp_path / "component-checkpoint"),
        "expected_manifest_digest": identity["inference_digest"], "model": asdict(learner.model),
        "inference": asdict(learner.binding), "cuda_visible_devices": "", "allow_toy": True, **input_binding(engineering_contract(), "component", False)}, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    return learner, options


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_cpu_neural_clean_worker_two_frames_matches_full_frozen_network(make_input, tmp_path, kind):
    import math

    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    from gf_models.structured_binding import policy_label
    learner, options = neural_component(tmp_path, kind)
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(make_structured_policy, kind, 2**63, action_settings(), ledger=ledger, policy_options=options)
        try:
            await worker.initialize()
            assert worker.expected_policy == policy_label(learner.model, learner.binding, options["expected_manifest_digest"], cuda_visible_devices="")
            assert worker.initialized and worker.startup_elapsed_ms > 0
            bank, history = EventBank(1, "opaque-neural-component"), []
            value = make_input(hand=("bronze-club", "steel-armor"), learned=("fireball", "aura"))
            for revision in (0, 2):
                frame, delta = own_frame(value, bank, revision, [{"type": "action_accepted", "player": 1}])
                generated, selection = await worker.select_frame(frame, delta)
                history.append(tokenize_frame(frame, bank.view(), aggregate_window=learner.model.event.aggregate_window))
                candidates = tokenize_table(generated.table, frame, learner.model.max_positions,
                                            aggregate_window=learner.model.event.aggregate_window)
                with learner.torch.no_grad():
                    logs, expected_value, _ = learner.network(tuple(history), candidates)
                assert selection.logprob == pytest.approx(float(logs[selection.index]), abs=1e-5, rel=1e-5)
                assert selection.value == pytest.approx(float(expected_value), abs=1e-5, rel=1e-5)
                assert selection.policy_version == worker.expected_policy
                assert selection.validate(generated.table, frame) == generated.table.to_action(selection.index, frame)
                assert all(math.isfinite(value) and value >= 0 for value in worker.last_timings.values())
                assert {"tokenization_ms", "input_pack_transfer_host_ms", "forward_enqueue_host_ms", "sync_host_ms", "return_ms", "sampling_ms"} <= set(worker.last_timings)
                assert worker.lease.decisions == {} and len(ledger.workers) == 1
        finally:
            await worker.close()
        assert not ledger.workers and worker.process.poll() is not None
    asyncio.run(run())


@pytest.mark.parametrize("mismatch", ["action_hash", "manifest_hash", "actual_visibility"])
def test_neural_initialize_rejects_wrong_owned_binding_before_ready_and_reaps(tmp_path, mismatch):
    learner, options = neural_component(tmp_path)
    actions = action_settings()
    if mismatch == "action_hash":
        actions = __import__("dataclasses").replace(actions, max_candidates=actions.max_candidates - 1)
    elif mismatch == "manifest_hash":
        options["expected_manifest_digest"] = "0" * 64
    else:
        options["cuda_visible_devices"] = None
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(make_structured_policy, learner.model.model_id, 0, actions, ledger=ledger, policy_options=options)
        with pytest.raises(RuntimeError, match="hash|binding|visibility"):
            await worker.initialize()
        assert not worker.initialized and worker.closed and worker.process.poll() is not None
        assert not ledger.workers
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["table_hash", "policy", "config", "cost", "rng", "timing", "missing_timing", "unknown"])
def test_malformed_reply_reaps_child_and_releases_only_confirmed_owned_reservations(make_input, monkeypatch, mutation):
    import random

    from gf_common.structured_wire import generated_dict
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(clean_probe, "random", 0, action_settings(), ledger=ledger)
        await worker.initialize()
        bank = EventBank(1, "reply-component")
        frame, delta = own_frame(make_input(), bank, 0)
        rng = random.Random(0)
        rng.setstate(worker._search_state)
        generated = PublicCandidateGeneratorV2(action_settings(), clock=lambda: 0.).generate(frame, bank.view(), rng, worker.costs)
        policy = make_structured_policy("random", action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
        selection = policy.select_frame(frame, bank.view(), generated.table)
        reply = json.loads(json.dumps({"protocol_version": 2, "kind": "selection", "generated": generated_dict(generated),
            "selection": asdict(selection), "timings": {"generation_ms": 0., "policy_ms": 0., "worker_decision_ms": 0.}, **input_binding(engineering_contract(), "component", False)}))
        if mutation == "table_hash":
            reply["selection"]["table_hash"] = "0" * 64
        elif mutation == "policy":
            reply["selection"]["policy_version"] = "foreign-policy"
        elif mutation == "config":
            reply["generated"]["report"]["config_hash"] = "0" * 64
        elif mutation == "cost":
            reply["generated"]["report"]["numeric_costs_hash"] = "0" * 64
        elif mutation == "rng":
            reply["generated"]["report"]["rng_before"][1][0] ^= 1
        elif mutation == "timing":
            reply["timings"]["policy_ms"] = float("nan")
        elif mutation == "missing_timing":
            reply["timings"] = {}
        else:
            reply["unexpected"] = "forbidden"
        async def injected(payload):
            assert worker.lease.decisions  # reservation exists before request/response.
            return reply
        monkeypatch.setattr(worker, "_roundtrip", injected)
        with pytest.raises(ContractError):
            await worker.select_frame(frame, delta)
        assert worker.closed and worker.process.poll() is not None
        assert not ledger.workers and worker.lease.released
    asyncio.run(run())


def test_global_startup_residency_fails_before_popen_and_spawn_error_rolls_back(tmp_path, monkeypatch):
    import gf_simulator_client.structured_worker as module
    from gf_models.structured_binding import safe_options_v2
    from gf_models.structured_config import InferenceBindingV2
    from tests.test_structured_network import small_model
    model = small_model()
    binding = InferenceBindingV2(device="cuda:0")
    options = safe_options_v2({"version": "2", "checkpoint": str(tmp_path / "never-read"),
        "expected_manifest_digest": "a" * 64, "model": asdict(model), "inference": asdict(binding),
        "cuda_visible_devices": "0", "allow_toy": True, **input_binding(engineering_contract(), "component", False)}, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    ledger = ReservationLedger(global_cuda_bytes=1)
    def forbidden(*args, **kwargs):
        pytest.fail("Popen ran before shared startup residency admission")
    with monkeypatch.context() as patch:
        patch.setattr(module.subprocess, "Popen", forbidden)
        with pytest.raises(ContractError):
            StructuredParticipantWorker(make_structured_policy, "mlp", 0, action_settings(), ledger=ledger, policy_options=options)
    assert not ledger.workers
    def failed_spawn(*args, **kwargs):
        raise OSError("explicit spawn failure")
    monkeypatch.setattr(module.subprocess, "Popen", failed_spawn)
    with pytest.raises(OSError, match="spawn failure"):
        StructuredParticipantWorker(clean_probe, "random", 0, action_settings(), ledger=ledger)
    assert not ledger.workers


def test_inference_timing_counter_resets_disables_and_does_not_change_sampling(make_input, monkeypatch):
    import gf_models.structured_network as module
    from gf_models.structured_policy import StructuredPolicy
    from tests.test_structured_learning import learner_for, records_for
    learner = learner_for()
    record = records_for(make_input, learner, count=1)
    frame, table = record.frames[0], record.tables[0]
    policy = StructuredPolicy(learner.network, learner.model, learner.binding, "a" * 64, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    policy.reset(17)
    expected = policy.select_frame(frame, record.bank, table)
    policy.reset(17)
    learner.network._input_packing_seconds = 1_000_000.
    actual = policy.select_frame(frame, record.bank, table)
    assert actual == expected
    assert policy.last_timing["input_pack_transfer_host_ms"] < 1_000_000
    assert learner.network._measure_input_packing is False
    counter = learner.network._input_packing_seconds
    def forbidden():
        pytest.fail("training/gradient recomputation accumulated inference packing timers")
    monkeypatch.setattr(module.time, "perf_counter", forbidden)
    tokens, candidates = __import__("gf_training.structured_rollout", fromlist=["StructuredRollout"]).StructuredRollout(
        (record,), learner.model, learner.action_config_hash, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True).endpoint(0)
    result = learner.network(tokens, candidates)
    (-result[0][0] + result[1].square()).backward()
    assert learner.network._input_packing_seconds == counter
    assert learner.network._measure_input_packing is False


def test_explicit_unavailable_cuda_initialize_never_falls_back_to_cpu(tmp_path):
    from gf_models.structured_config import InferenceBindingV2
    learner, options = neural_component(tmp_path)
    options["inference"] = asdict(InferenceBindingV2(device="cuda:0"))
    options["cuda_visible_devices"] = ""  # explicit hide-all is valid syntax, unavailable execution.
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(make_structured_policy, learner.model.model_id, 0, action_settings(), ledger=ledger, policy_options=options)
        with pytest.raises(RuntimeError, match="CUDA|unavailable"):
            await worker.initialize()
        assert worker.initial["policy_options"]["inference"]["device"] == "cuda:0"
        assert not worker.initialized and worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_malformed_reply_keeps_slot_and_decision_reserved_until_delayed_owned_reap(make_input, monkeypatch):
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(clean_probe, "random", 0, action_settings(), ledger=ledger)
        await worker.initialize()
        entered, proceed = asyncio.Event(), asyncio.Event()
        actual_close = worker._close_owned
        async def delayed_close(timeout, started):
            entered.set()
            await proceed.wait()
            await actual_close(timeout, started)
        async def malformed(payload):
            return {"unexpected": "malformed"}
        monkeypatch.setattr(worker, "_roundtrip", malformed)
        monkeypatch.setattr(worker, "_close_owned", delayed_close)
        frame, delta = own_frame(make_input(), EventBank(1, "delayed-reap"), 0)
        task = asyncio.create_task(worker.select_frame(frame, delta))
        try:
            await asyncio.wait_for(entered.wait(), 1.)
            assert worker.process.poll() is None
            assert len(ledger.workers) == 1 and worker.lease.decisions and not worker.lease.released
        finally:
            proceed.set()
        with pytest.raises(ContractError):
            await task
        assert worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_inference_failure_disables_timing_and_preserves_sampling_rng(make_input, monkeypatch):
    from gf_models.structured_policy import StructuredPolicy
    from tests.test_structured_learning import learner_for, records_for
    learner = learner_for()
    record = records_for(make_input, learner, count=1)
    policy = StructuredPolicy(learner.network, learner.model, learner.binding, "a" * 64, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    policy.reset(17)
    before = policy.rng.get_state().clone()
    def failed_score(*args, **kwargs):
        raise RuntimeError("explicit scoring failure")
    with monkeypatch.context() as patch:
        patch.setattr(learner.network, "distribution", failed_score)
        with pytest.raises(RuntimeError, match="scoring failure"):
            policy.select_frame(record.frames[0], record.bank, record.tables[0])
    assert learner.network._measure_input_packing is False and policy.last_timing == {}
    assert learner.torch.equal(policy.rng.get_state(), before)
    policy.reset(17)
    result = policy.select_frame(record.frames[0], record.bank, record.tables[0])
    assert result.logprob is not None and result.value is not None


def test_neural_reply_with_missing_values_is_rejected_before_use(make_input, tmp_path, monkeypatch):
    import random

    from gf_common.structured import Selection
    from gf_common.structured_wire import generated_dict
    learner, options = neural_component(tmp_path)
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(make_structured_policy, learner.model.model_id, 0, action_settings(), ledger=ledger, policy_options=options)
        await worker.initialize()
        bank = EventBank(1, "neural-null-reply")
        frame, delta = own_frame(make_input(), bank, 0)
        rng = random.Random(0)
        rng.setstate(worker._search_state)
        generated = PublicCandidateGeneratorV2(action_settings(), clock=lambda: 0.).generate(frame, bank.view(), rng, worker.costs)
        invalid = Selection(generated.table.table_hash, 0, worker.expected_policy)
        async def null_reply(payload):
            return {"protocol_version": 2, "kind": "selection", "generated": generated_dict(generated),
                    "selection": asdict(invalid), "timings": {"generation_ms": 0., "policy_ms": 0., "worker_decision_ms": 0.}, **input_binding(engineering_contract(), "component", False)}
        monkeypatch.setattr(worker, "_roundtrip", null_reply)
        with pytest.raises(ContractError, match="finite logprob/value"):
            await worker.select_frame(frame, delta)
        assert worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["construct", "start", "cancel_construct", "cancel_start", "cancel_after_start"])
def test_thread_setup_failure_preserves_original_error_and_reaps_real_child(monkeypatch, failure):
    import gf_simulator_client.structured_worker as module
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(clean_probe, "random", 0, action_settings(), ledger=ledger)
        actual_thread, actual_start = module.threading.Thread, module.threading.Thread.start
        def construct(*args, **kwargs):
            if failure == "cancel_construct":
                raise asyncio.CancelledError("explicit constructor cancellation")
            raise RuntimeError("explicit constructor failure")
        def start(thread):
            if failure == "cancel_after_start":
                actual_start(thread)
                asyncio.current_task().cancel()
                return
            if failure == "cancel_start":
                raise asyncio.CancelledError("explicit start cancellation")
            raise RuntimeError("explicit start failure")
        with monkeypatch.context() as patch:
            if failure in ("construct", "cancel_construct"):
                patch.setattr(module.threading, "Thread", construct)
            else:
                patch.setattr(actual_thread, "start", start)
            expected = asyncio.CancelledError if failure.startswith("cancel") else RuntimeError
            with pytest.raises(expected):
                await worker.initialize()
        assert worker.closed and worker.process.poll() is not None
        assert worker.lease.released and not ledger.workers
        assert "abort_cleanup_error" not in worker.health
    asyncio.run(run())


def test_journal_writer_candidate_cap_rejects_before_sidecar_or_event_write(make_input, tmp_path, monkeypatch):
    import gf_training.structured_journal as module
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path, caps_override={"max_candidates": 1})
    before_rows, before_boundary = writer.path.read_bytes(), writer.bank.boundary
    def forbidden(*args, **kwargs):
        pytest.fail("sidecar ownership allocated before journal candidate cap")
    monkeypatch.setattr(module, "publish_candidates", forbidden)
    try:
        assert len(generated.table.kinds) > 1
        with pytest.raises(ContractError):
            writer.begin_frame(frame, delta, generated, selection, command, {})
        assert writer.path.read_bytes() == before_rows and writer.bank.boundary == before_boundary
        assert writer.count == 0 and not list(tmp_path.glob("*.columns"))
    finally:
        writer.close_incomplete()


def two_decision_journal(make_input, tmp_path, **settings):
    import random
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path, **settings)
    index = writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1,
                              "status": "active", "result": None})
    bank = EventBank(1, frame.episode)
    value = make_input(hand=(), learned=("fireball",))
    first, _ = own_frame(value, bank, 0, delta.events)
    assert first == frame
    second, next_delta = own_frame(value, bank, 1, [{"type": "action_accepted", "player": 1}])
    rng = random.Random(0)
    rng.setstate(generated.report.rng_after)
    costs = NumericCosts(**writer.header["numeric_costs"])
    next_generated = PublicCandidateGeneratorV2(action_settings(), clock=lambda: 0.).generate(second, bank.view(), rng, costs)
    policy = make_structured_policy("random", action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    policy.reset(17)
    policy.select_frame(first, bank.view(first.boundary), generated.table)
    next_selection = policy.select_frame(second, bank.view(), next_generated.table)
    next_command = {"command_id": str(uuid.uuid4()), "expected_revision": 1,
                    "action": next_selection.validate(next_generated.table, second)}
    return writer, second, next_delta, next_generated, next_selection, next_command


def test_journal_writer_decision_cap_rejects_before_owned_writes(make_input, tmp_path, monkeypatch):
    import gf_training.structured_journal as module
    writer, frame, delta, generated, selection, command = two_decision_journal(make_input, tmp_path)
    writer.caps["max_decisions"] = 1
    before_rows, before_boundary = writer.path.read_bytes(), writer.bank.boundary
    def forbidden(*args, **kwargs):
        pytest.fail("second sidecar allocated beyond journal decision cap")
    monkeypatch.setattr(module, "publish_candidates", forbidden)
    try:
        with pytest.raises(ContractError):
            writer.begin_frame(frame, delta, generated, selection, command, {})
        assert writer.path.read_bytes() == before_rows and writer.bank.boundary == before_boundary
        assert writer.count == 1
    finally:
        writer.close_incomplete()


@pytest.mark.parametrize("cap", ["candidates", "decisions"])
def test_reader_caps_reject_before_frame_or_sidecar_decode(make_input, tmp_path, monkeypatch, cap):
    import gf_training.structured_journal as module
    writer, frame, delta, generated, selection, command = two_decision_journal(make_input, tmp_path)
    index = writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 1, "state_revision": 2,
                              "status": "active", "result": None})
    writer.finish("incomplete", None)
    data = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    rows = copy.deepcopy(list(data.records))
    if cap == "candidates":
        rows[0]["caps"]["max_candidates"] = 1
        def forbidden(*args, **kwargs):
            pytest.fail("oversized table reached frame/sidecar decode")
        monkeypatch.setattr(module, "parse_frame", forbidden)
        monkeypatch.setattr(module, "read_candidates", forbidden)
    else:
        rows[0]["caps"]["max_decisions"] = 1
        actual = module.parse_frame
        calls = []
        def first_only(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                pytest.fail("second frame decoded beyond journal decision cap")
            return actual(*args, **kwargs)
        monkeypatch.setattr(module, "parse_frame", first_only)
    with pytest.raises(ContractError):
        read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


@pytest.mark.parametrize("kind", ["unknown", "unsupported", "infrastructure_failure", "", None])
def test_journal_unknown_end_kinds_rejected_by_writer_and_reader(make_input, tmp_path, kind):
    writer, _, _, _, _, _ = journal_setup(make_input, tmp_path)
    before = writer.path.read_bytes()
    with pytest.raises(ContractError, match="end kind"):
        writer.finish(kind, None)
    assert writer.path.read_bytes() == before
    writer.finish("incomplete", None)
    rows = [json.loads(line) for line in writer.final_path.read_text().splitlines()]
    rows[-1]["end_kind"] = kind
    with pytest.raises(ContractError, match="end kind"):
        read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


@pytest.mark.parametrize("version,model_id,value", [("2", "mlp", 2.), ("2", "mlp", True), ("2", "mlp", None),
    ("2", "mlp", "2"), ("2", "mlp", 3), ("1", "random", False), ("1", "random", 0),
    ("1", "random", "future"), ("1", "random", 2.)])
def test_generation_schema_is_exact_typed_role_value(make_input, tmp_path, version, model_id, value):
    writer, _, _, _, _, _ = journal_setup(make_input, tmp_path, model_id=model_id, model_version=version)
    writer.finish("incomplete", None)
    rows = [json.loads(line) for line in writer.final_path.read_text().splitlines()]
    assert rows[0]["generation_schema"] == (2 if version == "2" else None)
    rows[0]["generation_schema"] = value
    with pytest.raises(ContractError, match="generation schema"):
        read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def test_journal_neural_null_rejects_before_write_and_before_reader_sidecar_decode(make_input, tmp_path, monkeypatch):
    import gf_training.structured_journal as module
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path, model_id="mlp", model_version="2")
    before = writer.path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("neural null reached table publication/decode")
    with monkeypatch.context() as patch:
        patch.setattr(module, "publish_candidates", forbidden)
        with pytest.raises(ContractError, match="finite logprob/value"):
            writer.begin_frame(frame, delta, generated, selection, command, {})
    assert writer.path.read_bytes() == before and writer.bank.boundary.end == 0
    writer.close_incomplete()
    other = tmp_path / "baseline"
    other.mkdir()
    baseline, frame, delta, generated, selection, command = journal_setup(make_input, other)
    baseline.begin_frame(frame, delta, generated, selection, command, {})
    baseline.finish("incomplete", None)
    data = read_structured_journal(baseline.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    assert data.decisions[0].selection.logprob is data.decisions[0].selection.value is None
    rows = copy.deepcopy(list(data.records))
    rows[0].update(model_id="mlp", model_version="2", generation_schema=2)
    monkeypatch.setattr(module, "parse_frame", forbidden)
    monkeypatch.setattr(module, "read_candidates", forbidden)
    with pytest.raises(ContractError, match="finite logprob/value"):
        read_structured_journal(baseline.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


@pytest.mark.parametrize("failure", ["huge_timing", "cancel_validation", "cancel_ready"])
def test_initialize_validation_failures_reap_real_child_and_preserve_error(monkeypatch, failure):
    import gf_simulator_client.structured_worker as module
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(clean_probe, "random", 0, action_settings(), ledger=ledger)
        async def injected(payload):
            if failure == "cancel_ready":
                raise asyncio.CancelledError("explicit ready cancellation")
            return {"protocol_version": 2, "kind": "ready", "policy_version": worker.expected_policy,
                    "model_load_ms": 10**400 if failure == "huge_timing" else 0., **input_binding(engineering_contract(), "component", False)}
        def cancelled(*args, **kwargs):
            raise asyncio.CancelledError("explicit validation cancellation")
        monkeypatch.setattr(worker, "_roundtrip", injected)
        if failure == "cancel_validation":
            monkeypatch.setattr(module, "_timings", cancelled)
        expected = ContractError if failure == "huge_timing" else asyncio.CancelledError
        with pytest.raises(expected):
            await worker.initialize()
        assert not worker.initialized and worker.closed and worker.process.poll() is not None
        assert worker.lease.released and not ledger.workers
        assert "abort_cleanup_error" not in worker.health
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("command_id", "not-a-uuid"), ("command_id", None),
    ("command_id", "00000000-0000-0000-0000-00000000000A"),
    ("expected_revision", False), ("expected_revision", 0.)])
def test_journal_command_dto_rejected_before_sidecar_or_rows(make_input, tmp_path, monkeypatch, field, value):
    import gf_training.structured_journal as module
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path)
    before, boundary = writer.path.read_bytes(), writer.bank.boundary
    command[field] = value
    def forbidden(*args, **kwargs):
        pytest.fail("invalid command reached sidecar publication")
    monkeypatch.setattr(module, "publish_candidates", forbidden)
    try:
        with pytest.raises(ContractError):
            writer.begin_frame(frame, delta, generated, selection, command, {})
        assert writer.path.read_bytes() == before and writer.bank.boundary == boundary
        assert writer.count == 0
    finally:
        writer.close_incomplete()


@pytest.mark.parametrize("field,value", [("command_id", "not-a-uuid"), ("command_id", None),
    ("command_id", "00000000-0000-0000-0000-00000000000A"),
    ("expected_revision", False), ("expected_revision", 0.)])
def test_reader_rejects_bad_command_even_with_matching_positive_ack(make_input, tmp_path, field, value):
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path)
    index = writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1,
                               "status": "active", "result": None})
    writer.finish("truncated", None)
    data = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    rows = copy.deepcopy(list(data.records))
    decision = next(row for row in rows if row["kind"] == "decision")
    acknowledgement = next(row for row in rows if row["kind"] == "acknowledgement")
    decision["command"][field] = value
    if field == "command_id":
        acknowledgement["response"][field] = value
    with pytest.raises(ContractError):
        read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def test_journal_finish_preserves_final_created_during_end_write(make_input, tmp_path, monkeypatch):
    writer, _, _, _, _, _ = journal_setup(make_input, tmp_path)
    actual_write = writer.write
    def collision(row):
        if row["kind"] == "end":
            writer.final_path.write_bytes(b"preexisting-evidence")
        actual_write(row)
    monkeypatch.setattr(writer, "write", collision)
    with pytest.raises(FileExistsError):
        writer.finish("truncated", None)
    assert writer.final_path.read_bytes() == b"preexisting-evidence"
    assert writer.output.closed and writer.path.is_file()
    pending = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert pending[-1]["kind"] == "end" and pending[-1]["end_kind"] == "truncated"


@pytest.mark.parametrize("mutation", ["symlink", "replacement", "inplace", "hash_eof"])
def test_journal_finish_verifies_owned_object_through_publication(make_input, tmp_path, monkeypatch, mutation):
    import os

    import gf_training.structured_journal as module
    writer, _, _, _, _, _ = journal_setup(make_input, tmp_path)
    outside = tmp_path / "foreign-evidence"
    outside.write_bytes(b"foreign-evidence")
    actual_link, actual_pread = os.link, os.pread
    def link(source, destination, **kwargs):
        if mutation == "symlink":
            source.unlink()
            source.symlink_to(outside)
        elif mutation == "replacement":
            source.unlink()
            source.write_bytes(b"foreign-evidence")
        elif mutation == "inplace":
            source.write_bytes(b"foreign-evidence")
        return actual_link(source, destination, **kwargs)
    eof = []
    def pread(fd, count, offset):
        if mutation == "hash_eof" and count == 1:
            eof.append(fd)
            if len(eof) == 2:
                writer.final_path.write_bytes(b"")
        return actual_pread(fd, count, offset)
    monkeypatch.setattr(module.os, "link", link)
    monkeypatch.setattr(module.os, "pread", pread)
    with pytest.raises(ContractError):
        writer.finish("truncated", None)
    assert writer.output.closed and writer.path.exists()
    assert outside.read_bytes() == b"foreign-evidence"


@pytest.mark.parametrize("bound", ["interval", "delta", "episode", "positions"])
def test_journal_known_caps_reject_before_bank_decode_mutation_or_owned_io(make_input, tmp_path, monkeypatch, bound):
    import gf_training.structured_journal as module
    events = [{"type": "action_accepted", "player": 1}] * (2 if bound == "interval" else 1)
    caps = {"max_interval_events": 1} if bound in ("interval", "delta") else {"max_positions": 1} if bound == "positions" else {
        "max_interval_events": 1, "max_episode_events": 1}
    value = make_input(hand=("bronze-club",), learned=("aura",)) if bound == "positions" else None
    if bound == "episode":
        # A valid next interval fits its cap but exceeds the episode's absolute
        # endpoint; the first decision has already been positively acknowledged.
        writer, frame, delta, generated, selection, command = two_decision_journal(
            make_input, tmp_path, caps_override=caps, aggregate_window=1)
    else:
        writer, frame, delta, generated, selection, command = journal_setup(
            make_input, tmp_path, caps_override=caps, events=events, input_value=value)
    if bound == "delta":
        delta = EventDelta(delta.actor, delta.episode, delta.start, delta.previous_hash, events * 2)
    elif bound == "positions":
        assert max(b - a for a, b in zip(generated.table.offsets[:-1], generated.table.offsets[1:], strict=True)) == 2
    before, boundary, files = writer.path.read_bytes(), writer.bank.boundary, sorted(tmp_path.iterdir())
    count = writer.count
    def forbidden(*args, **kwargs):
        pytest.fail("known resource violation reached delta decode/bank mutation/sidecar publication")
    monkeypatch.setattr(EventDelta, "events", property(forbidden))
    monkeypatch.setattr(writer.bank, "append", forbidden)
    monkeypatch.setattr(module, "publish_candidates", forbidden)
    try:
        with pytest.raises(ContractError):
            writer.begin_frame(frame, delta, generated, selection, command, {})
        assert writer.path.read_bytes() == before and writer.bank.boundary == boundary
        assert writer.count == count and not writer.pending and sorted(tmp_path.iterdir()) == files
    finally:
        writer.close_incomplete()


@pytest.mark.parametrize("event_count", [0, 1, 2])
def test_journal_exact_event_cap_boundaries_roundtrip_and_reader_rejects_lower_cap(make_input, tmp_path, event_count):
    events = [{"type": "action_accepted", "player": 1}] * event_count
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path,
        caps_override={"max_interval_events": 2, "max_episode_events": 2, "max_decisions": 1, "max_positions": 2},
        events=events, aggregate_window=2)
    index = writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1,
                               "status": "active", "result": None})
    writer.finish("truncated", None)
    data = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    assert data.bank.boundary.end == event_count and data.accepted == (0,)
    assert data.decisions[0].generated.table == generated.table
    if event_count == 2:
        rows = copy.deepcopy(list(data.records))
        rows[0]["caps"]["max_interval_events"] = 1
        with pytest.raises(ContractError, match="interval cap"):
            read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


@pytest.mark.parametrize('error_type', [MemoryError, asyncio.CancelledError])
def test_ctor_preparation_clock_failure_precedes_reservation_and_spawn(monkeypatch, error_type):
    import gf_simulator_client.structured_worker as module
    ledger = ReservationLedger()
    original = error_type('injected synchronous clock failure')
    def failed_clock():
        raise original
    def forbidden(*args, **kwargs):
        pytest.fail('preparation error reached Popen')
    monkeypatch.setattr(module.time, 'monotonic', failed_clock)
    monkeypatch.setattr(module.subprocess, 'Popen', forbidden)
    with pytest.raises(error_type) as caught:
        StructuredParticipantWorker(clean_probe, 'random', 0, action_settings(), ledger=ledger)
    assert caught.value is original and not ledger.workers and ledger.serial == 0


@pytest.mark.parametrize('registered', [False, True])
@pytest.mark.parametrize('error_type', [MemoryError, asyncio.CancelledError])
def test_ctor_registration_failure_reaps_real_child_before_reservation_release(monkeypatch, registered, error_type):
    import gf_simulator_client.structured_worker as module
    from gf_simulator_client.structured_resources import WorkerReservation
    ledger = ReservationLedger()
    processes = []
    original = error_type('injected synchronous registration failure')
    actual_popen, actual_spawned = module.subprocess.Popen, WorkerReservation.spawned
    actual_cancel, actual_reaped = WorkerReservation.cancel_before_spawn, WorkerReservation.reaped
    def popen(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        processes.append(process)
        actual_wait = process.wait
        def wait(*args, **kwargs):
            assert ledger.workers and not next(iter(ledger.workers.values())).released
            return actual_wait(*args, **kwargs)
        monkeypatch.setattr(process, 'wait', wait)
        return process
    def spawned(lease, process):
        assert process.poll() is None and ledger.workers
        if registered:
            actual_spawned(lease, process)
        raise original
    def cancel(lease):
        assert processes[0].poll() is not None and processes[0].stdin.closed and processes[0].stdout.closed
        return actual_cancel(lease)
    def reaped(lease, process):
        assert process.poll() is not None and process.stdin.closed and process.stdout.closed
        return actual_reaped(lease, process)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(WorkerReservation, 'spawned', spawned)
    monkeypatch.setattr(WorkerReservation, 'cancel_before_spawn', cancel)
    monkeypatch.setattr(WorkerReservation, 'reaped', reaped)
    with pytest.raises(error_type) as caught:
        StructuredParticipantWorker(clean_probe, 'random', 0, action_settings(), ledger=ledger)
    assert caught.value is original and len(processes) == 1
    assert processes[0].poll() is not None and not ledger.workers


def test_ctor_unconfirmed_reap_keeps_reservation_until_real_test_cleanup(monkeypatch):
    import gf_simulator_client.structured_worker as module
    from gf_simulator_client.structured_resources import WorkerReservation
    ledger, processes = ReservationLedger(), []
    actual_popen = module.subprocess.Popen
    original = MemoryError('injected registration failure')
    def popen(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        processes.append((process, process.wait))
        def unconfirmed(*args, **kwargs):
            raise module.subprocess.TimeoutExpired('owned-child', 2)
        monkeypatch.setattr(process, 'wait', unconfirmed)
        return process
    def fail(*args, **kwargs):
        raise original
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(WorkerReservation, 'spawned', fail)
    try:
        with pytest.raises(MemoryError) as caught:
            StructuredParticipantWorker(clean_probe, 'random', 0, action_settings(), ledger=ledger)
        assert caught.value is original and len(ledger.workers) == 1
        lease = next(iter(ledger.workers.values()))
        assert not lease.released and lease.pid is None
        process, wait = processes[0]
        assert process.stdin.closed and process.stdout.closed
        assert any('startup reap failed' in note for note in original.__notes__)
    finally:
        # Confirm the actual killed child's reap; only then retire this test's
        # deliberately unconfirmed startup reservation. No fake child cleanup.
        if processes:
            process, wait = processes[0]
            wait(timeout=2)
            assert process.poll() is not None
            for lease in tuple(ledger.workers.values()):
                lease.cancel_before_spawn()
    assert not ledger.workers
