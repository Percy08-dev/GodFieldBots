"""Native owned serving and participant-local sampling component checks."""

import json
import time
from pathlib import Path

import gf_native
import numpy as np
import pytest

from gf_common.domain import canonical

FIXTURE = Path(__file__).parent / "fixtures/rust_player_v5/originals.json"


def control(index=0, *, game="g1", player=1, trial="trial", request=1, epoch=0, deadline=None):
    episode = json.loads(FIXTURE.read_text())["episodes"][index]
    decision = episode["decisions"][0]
    identity = {
        "run": "run",
        "trial": trial,
        "game": game,
        "player": player,
        "epoch": epoch,
        "request": request,
        "revision": decision["frame"]["revision"],
        "checkpoint": "a" * 64,
        "deadline": deadline or time.monotonic() + 30,
    }
    return {
        "identity": identity,
        "frame": decision["frame"],
        "events": decision["delta"]["events"],
        "actions": episode["actions"],
        "rules": episode["rules"],
        "table": decision["generated"]["table"],
        "window": 128,
        "max_positions": 18,
        "sampling_seed": 17,
    }


def test_native_builds_final_dtype_owned_row_and_empty_interval():
    dispatcher = gf_native.PlayerInferenceDispatcher(2, 1, 0.0, "torch-test")
    token = dispatcher.enqueue(canonical(control(3)).encode())
    batch = dispatcher.next_batch(time.monotonic() + 2)
    row = batch["rows"][0]
    assert row["inputs"]["interval_numerics"].shape[0] == 0
    assert row["inputs"]["state_numerics"].dtype == np.float32
    assert row["inputs"]["references"].dtype == np.int64
    assert row["inputs"]["card_mask"].dtype == np.bool_
    assert row["inputs"]["references"].shape[1] == 18
    assert row["rng_state"] is None and row["sampling_seed"] == 17
    dispatcher.cancel("test finished")
    with pytest.raises(ValueError):
        dispatcher.wait_result(token, time.monotonic() + 1)


def native_row(payload, *, runtime="torch-test"):
    dispatcher = gf_native.PlayerInferenceDispatcher(4, 1, 0.0, runtime)
    token = dispatcher.enqueue(canonical(payload).encode())
    batch = dispatcher.next_batch(time.monotonic() + 3)
    return dispatcher, token, batch


def expected_numeric(episode, decision, events):
    from types import SimpleNamespace

    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import parse_card_instance_rules
    from gf_common.structured import EventBank, event_delta
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    from gf_common.structured_wire import parse_frame, parse_game_generated_v3
    from gf_models.shared_numeric import numeric_row

    bank = EventBank(episode["actor"], episode["name"], layout=layout)
    bank.append(
        event_delta(
            episode["actor"], episode["name"], 0, bank.boundary.prefix_hash, events, layout=layout
        )
    )
    frame = parse_frame(decision["frame"], layout=layout)
    generated = parse_game_generated_v3(
        decision["generated"],
        action_contract=action_config_v3(episode["actions"]).action_contract,
        rules=parse_card_instance_rules(episode["rules"]),
    )
    tokens = tokenize_frame(frame, bank.view(), layout=layout)
    candidates = tokenize_table(generated.table, frame, 18, layout=layout)
    return numeric_row(
        tokens, candidates, SimpleNamespace(model_id="mlp", max_positions=18), layout=layout
    )


@pytest.mark.parametrize("index", range(9))
def test_original_tokens_final_arrays_match_unchanged_python(index):
    from gf_models.structured_learning import BOOL_FIELDS, LONG_FIELDS

    episode = json.loads(FIXTURE.read_text())["episodes"][index]
    events = []
    for request, decision in enumerate(episode["decisions"], 1):
        events.extend(decision["delta"]["events"])
        payload = control(index, request=request)
        payload.update(
            frame=decision["frame"], table=decision["generated"]["table"], events=list(events)
        )
        payload["identity"]["revision"] = decision["frame"]["revision"]
        dispatcher, _, batch = native_row(payload)
        expected = expected_numeric(episode, decision, events)
        actual = batch["rows"][0]["inputs"]
        assert set(actual) == set(expected["fields"])
        for name, values in expected["fields"].items():
            dtype = (
                np.int64 if name in LONG_FIELDS else np.bool_ if name in BOOL_FIELDS else np.float32
            )
            target = np.asarray(values, dtype=dtype).reshape(expected["shapes"][name])
            np.testing.assert_array_equal(actual[name], target, err_msg=name)
            assert actual[name].dtype == target.dtype and actual[name].flags.c_contiguous
        assert (
            batch["rows"][0]["buffer_bytes"]
            == sum(v.nbytes for v in actual.values()) + batch["rows"][0]["candidate_offsets"].nbytes
        )
        np.testing.assert_array_equal(
            batch["rows"][0]["candidate_offsets"], decision["generated"]["table"]["offsets"]
        )
        dispatcher.close()


def response(row, *, state=b"provisional", runtime="torch-test"):
    n = row["candidate_count"]
    return {
        "identity": row["identity"],
        "logs": np.full(n, -np.log(n), dtype=np.float32),
        "value": 0.0,
        "sample": 0,
        "next_state": state,
        "runtime": runtime,
    }


def test_cpu_tensor_alias_and_last_reference_lifetime():
    import gc
    import weakref

    import torch

    from gf_models.rust_player import tensor_views

    dispatcher, _, batch = native_row(control(2))
    array = batch["rows"][0]["inputs"]["references"]
    weak = weakref.ref(array)
    tensor = tensor_views({"x": array}, device="cpu")["x"]
    assert tensor.data_ptr() == array.ctypes.data
    target = tensor.clone()
    del batch, array
    dispatcher.close()
    del dispatcher
    gc.collect()
    assert weak() is not None
    torch.testing.assert_close(tensor, target)
    del tensor
    gc.collect()
    assert weak() is None


def test_rng_state_commits_once_and_late_duplicate_cannot_change_it():
    payload = control(2)
    dispatcher, token, batch = native_row(payload)
    row = batch["rows"][0]
    assert dispatcher.submit_results(batch["batch_id"], [response(row, state=b"committed")]) == 1
    assert dispatcher.wait_result(token, time.monotonic() + 1)["sample"] == 0
    with pytest.raises(ValueError):
        dispatcher.submit_results(batch["batch_id"], [response(row, state=b"wrong")])
    payload["identity"]["request"] += 1
    token = dispatcher.enqueue(canonical(payload).encode())
    later = dispatcher.next_batch(time.monotonic() + 1)
    assert later["rows"][0]["rng_state"] == b"committed"
    dispatcher.close()


@pytest.mark.parametrize(
    "key,value",
    [
        ("run", "foreign"),
        ("trial", "foreign"),
        ("game", "foreign"),
        ("player", 2),
        ("epoch", 1),
        ("request", 99),
        ("revision", 99),
        ("checkpoint", "f" * 64),
        ("deadline", 1.0),
    ],
)
def test_foreign_identity_never_commits(key, value):
    payload = control(2)
    dispatcher, token, batch = native_row(payload)
    row = batch["rows"][0]
    answer = response(row)
    identity = json.loads(answer["identity"])
    identity[key] = value
    answer["identity"] = canonical(identity).encode()
    with pytest.raises(ValueError):
        dispatcher.submit_results(batch["batch_id"], [answer])
    with pytest.raises(ValueError):
        dispatcher.wait_result(token, time.monotonic() + 1)
    payload["identity"]["request"] += 1
    dispatcher.enqueue(canonical(payload).encode())
    later = dispatcher.next_batch(time.monotonic() + 1)
    assert later["rows"][0]["rng_state"] is None
    dispatcher.close()


def test_expiry_between_sample_and_acceptance_retains_committed_state():
    payload = control(2)
    dispatcher, token, first = native_row(payload)
    dispatcher.submit_results(first["batch_id"], [response(first["rows"][0], state=b"previous")])
    dispatcher.wait_result(token, time.monotonic() + 1)
    payload["identity"].update(request=2, deadline=time.monotonic() + 0.08)
    token = dispatcher.enqueue(canonical(payload).encode())
    batch = dispatcher.next_batch(time.monotonic() + 1)
    time.sleep(0.1)
    assert dispatcher.submit_results(batch["batch_id"], [response(batch["rows"][0])]) == 0
    with pytest.raises(TimeoutError):
        dispatcher.wait_result(token, time.monotonic() + 1)
    payload["identity"].update(request=3, deadline=time.monotonic() + 2)
    dispatcher.enqueue(canonical(payload).encode())
    assert dispatcher.next_batch(time.monotonic() + 1)["rows"][0]["rng_state"] == b"previous"
    dispatcher.close()


def test_formation_uses_original_absolute_first_arrival_window():
    from concurrent.futures import ThreadPoolExecutor

    dispatcher = gf_native.PlayerInferenceDispatcher(3, 3, 180.0, "torch-test")
    dispatcher.enqueue(canonical(control(3)).encode())
    started = time.monotonic()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(dispatcher.next_batch, time.monotonic() + 2)
        time.sleep(0.12)
        dispatcher.enqueue(canonical(control(2, game="g2")).encode())
        batch = future.result(timeout=1)
    elapsed = time.monotonic() - started
    assert len(batch["rows"]) == 2 and 0.15 <= elapsed < 0.28
    dispatcher.close()


@pytest.mark.parametrize("operation", ["close", "cancel"])
def test_shutdown_wakes_full_queue_sender_and_result_waiter(operation):
    from concurrent.futures import ThreadPoolExecutor

    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, "torch-test")
    token = dispatcher.enqueue(canonical(control(3)).encode())
    with ThreadPoolExecutor(2) as pool:
        sender = pool.submit(dispatcher.enqueue, canonical(control(2, game="g2")).encode())
        waiter = pool.submit(dispatcher.wait_result, token, time.monotonic() + 5)
        time.sleep(0.03)
        getattr(dispatcher, operation)(*(["test cancel"] if operation == "cancel" else []))
        for future in (sender, waiter):
            with pytest.raises(ValueError if operation == "cancel" else RuntimeError):
                future.result(timeout=1)


def test_shutdown_wakes_empty_batch_receiver_and_releases_gil():
    from concurrent.futures import ThreadPoolExecutor

    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, "torch-test")
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(dispatcher.next_batch, time.monotonic() + 5)
        time.sleep(0.03)
        dispatcher.cancel("cancel idle receiver")
        with pytest.raises(ValueError):
            future.result(timeout=1)


def test_retired_epoch_cannot_be_readmitted_and_new_epoch_resets_sampling():
    payload = control(3)
    dispatcher, token, batch = native_row(payload)
    dispatcher.retire(canonical(payload["identity"]).encode())
    with pytest.raises(RuntimeError):
        dispatcher.wait_result(token, time.monotonic() + 1)
    with pytest.raises(ValueError):
        dispatcher.enqueue(canonical(payload).encode())
    assert dispatcher.submit_results(batch["batch_id"], [response(batch["rows"][0])]) == 0
    payload["identity"]["epoch"] += 1
    payload["sampling_seed"] = 99
    dispatcher.enqueue(canonical(payload).encode())
    assert dispatcher.next_batch(time.monotonic() + 1)["rows"][0]["rng_state"] is None
    dispatcher.close()


def for_player(payload, player):
    """Rebuild a valid participant frame/hash, never relabel original hash identities."""
    import copy
    from dataclasses import asdict

    from gf_common.card_instance import shared_card_instance_contract
    from gf_common.domain import DecisionInput
    from gf_common.structured import CandidateTable, DecisionFrame, EventBank, event_delta
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_wire import frame_dict

    if player == 1:
        return payload
    payload = copy.deepcopy(payload)
    original = json.loads(FIXTURE.read_text())["episodes"]
    decision = next(
        e["decisions"][0]
        for e in original
        if e["decisions"][0]["frame"]["frame_hash"] == payload["frame"]["frame_hash"]
    )
    raw = payload["frame"]["payload"]
    obs, hints = raw["observation"], raw["action_options"]
    for name in ("viewer_player_id", "active_player", "current_player", "first_player"):
        obs[name] = 3 - obs[name]
    obs["own"]["player_id"], obs["opponent"]["player_id"] = 2, 1
    hints["viewer_player_id"] = hints["action_options"]["actor"] = 2
    for event in payload["events"]:
        for name in ("player", "attacker", "target", "active_player"):
            if type(event.get(name)) is int:
                event[name] = 3 - event[name]
    value = DecisionInput(
        obs, tuple(payload["events"]), None, tuple(raw["typed_metadata"]), raw["catalog"], hints
    )
    bank = EventBank(2, payload["frame"]["episode"], layout=layout)
    bank.append(
        event_delta(2, bank.episode, 0, bank.boundary.prefix_hash, payload["events"], layout=layout)
    )
    frame = DecisionFrame.from_input(value, bank, 0, layout=layout)
    actions = copy.deepcopy(decision["ordered_actions"])
    for action in actions:
        if type(action.get("target")) is int:
            action["target"] = 3 - action["target"]
    table = CandidateTable.from_actions(
        frame,
        actions,
        18,
        action_contract=shared_card_instance_contract(),
        purpose="qualified_game",
        synthetic_capacity=False,
    )
    payload.update(frame=frame_dict(frame, layout=layout), table=asdict(table))
    payload["identity"]["player"] = 2
    return payload


def test_two_games_four_participants_interleave_out_of_order_batches_and_isolation():
    import torch

    from gf_models.rust_player import provisional_sample, sampling_runtime_identity

    runtime = sampling_runtime_identity()
    dispatcher = gf_native.PlayerInferenceDispatcher(2, 1, 0.0, runtime)
    baseline = {}
    for player in (1, 2):
        tokens = []
        batches = []
        for game, index in (("g1", 2), ("g2", 6)):
            payload = for_player(control(index, game=game), player)
            token = dispatcher.enqueue(canonical(payload).encode())
            batch = dispatcher.next_batch(time.monotonic() + 1)
            row = batch["rows"][0]
            if game == "g1":
                baseline[player] = {name: v.copy() for name, v in row["inputs"].items()}
            else:
                # Mutate service-owned buffers of the other row after native export.
                for array in row["inputs"].values():
                    array.fill(0)
            n = row["candidate_count"]
            values = torch.ones(n) / n
            sample, state = provisional_sample(
                values, state=row["rng_state"], seed=row["sampling_seed"], runtime=runtime
            )
            answer = response(row, state=state, runtime=runtime)
            answer["sample"] = sample
            tokens.append(token)
            batches.append((batch, answer))
        # Replies arrive in the opposite game order. g1 sees only its own answer.
        for batch, answer in reversed(batches):
            assert dispatcher.submit_results(batch["batch_id"], [answer]) == 1
        for token in tokens:
            result = dispatcher.wait_result(token, time.monotonic() + 1)
            assert json.loads(result["identity"])["player"] == player
    # Same game/player/request in a different trial starts from authorized seed.
    payload = control(2, trial="other-trial")
    dispatcher.enqueue(canonical(payload).encode())
    row = dispatcher.next_batch(time.monotonic() + 1)["rows"][0]
    assert row["rng_state"] is None
    for name, array in baseline[1].items():
        np.testing.assert_array_equal(row["inputs"][name], array)
    dispatcher.close()


def test_native_provisional_sampling_matches_original_and_rejects_bad_state_without_global_rng():
    import torch

    from gf_models.rust_player import provisional_sample, sampling_runtime_identity

    torch.manual_seed(919)
    before = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(17)
    values = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    state = None
    for _ in range(5):
        expected = int(torch.multinomial(values, 1, generator=generator).item())
        sample, state = provisional_sample(
            values, state=state, seed=17, runtime=sampling_runtime_identity()
        )
        assert sample == expected and state == generator.get_state().numpy().tobytes()
        assert torch.equal(before, torch.get_rng_state())
    for state, runtime in ((b"bad-state", sampling_runtime_identity()), (None, "foreign-runtime")):
        with pytest.raises(ValueError):
            provisional_sample(values, state=state, seed=17, runtime=runtime)
        assert torch.equal(before, torch.get_rng_state())


def inference_service():
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player import RustPlayerInference, serialized_torch
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network
    from tests.model3_support import model3

    service = RustPlayerInference(device="cpu", cache_entries=3, cache_weight_bytes=64 * 1024**2)
    with serialized_torch():
        network = make_network(
            model3(), 17, InferenceBindingV2(max_candidates=65536), layout=layout
        )
    service.publish(network, "a" * 64)
    return service, network


def test_real_registry_scoring_matches_original_and_pending_g0_survives_g1():
    import torch

    from gf_models.rust_player import serialized_torch
    from gf_models.shared_numeric import score_numeric_row, tensor_row

    service, network = inference_service()
    payload = control(2)
    dispatcher, token, batch = native_row(payload, runtime=service.runtime)
    with pytest.raises(ValueError):
        service.evict("a" * 64, dispatcher=dispatcher)
    with serialized_torch():
        service.publish(network, "b" * 64)
    service.retain("a" * 64)  # Frozen opponent is a distinct explicit holder.
    answers = service.evaluate(batch)
    episode = json.loads(FIXTURE.read_text())["episodes"][2]
    expected = expected_numeric(episode, episode["decisions"][0], [])
    with torch.no_grad():
        logs, value = score_numeric_row(network, tensor_row(expected))
    np.testing.assert_allclose(answers[0]["logs"], logs.numpy().reshape(-1), atol=1e-5, rtol=1e-5)
    assert answers[0]["value"] == pytest.approx(float(value.item()), abs=1e-5, rel=1e-5)
    assert dispatcher.submit_results(batch["batch_id"], answers) == 1
    dispatcher.wait_result(token, time.monotonic() + 1)
    with pytest.raises(ValueError):
        service.evict("a" * 64, dispatcher=dispatcher)
    service.release("a" * 64)
    service.evict("a" * 64, dispatcher=dispatcher)
    assert service.cache_state()["entries"] == 1
    dispatcher.close()


def test_publication_preserves_global_cpu_rng_and_serialization_keeps_intentional_learner_draws():
    import torch

    from gf_models.rust_player import serialized_torch

    service, network = inference_service()
    before = torch.get_rng_state().clone()
    service.publish(network, "b" * 64)
    assert torch.equal(before, torch.get_rng_state())
    with serialized_torch():
        torch.rand(1)
    assert not torch.equal(before, torch.get_rng_state())


def test_publication_preserves_initialized_cuda_rng():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware/build unavailable; CPU path must not initialize CUDA")
    torch.cuda.init()
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    service, network = inference_service()
    service.publish(network, "b" * 64)
    assert all(
        torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all(), strict=True)
    )


def test_serving_failure_wakes_native_request_and_records_failed_stage():
    service, _ = inference_service()
    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, service.runtime)
    payload = control(3)
    payload["identity"]["checkpoint"] = "f" * 64
    token = dispatcher.enqueue(canonical(payload).encode())
    with pytest.raises(ValueError):
        service.serve_next(dispatcher, deadline=time.monotonic() + 1)
    with pytest.raises(ValueError):
        dispatcher.wait_result(token, time.monotonic() + 1)
    assert service.calls["failures"] == 1 and service.calls["next_batch"] == 1
    assert service.timings["native_wait"] > 0


def test_full_sender_wakes_at_original_inflight_expiry_without_receiver():
    from concurrent.futures import ThreadPoolExecutor

    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, "torch-test")
    dispatcher.enqueue(canonical(control(3, deadline=time.monotonic() + 0.12)).encode())
    started = time.monotonic()
    with ThreadPoolExecutor(1) as pool:
        sender = pool.submit(dispatcher.enqueue, canonical(control(2, game="g2")).encode())
        assert sender.result(timeout=0.8) >= 0
    assert time.monotonic() - started < 0.7
    dispatcher.close()


def test_real_scoring_and_rng_are_invariant_to_other_rows_geometry_values_and_order():
    service, _ = inference_service()
    baseline = None
    for other_index, order in ((3, ("g1", "g2")), (6, ("g2", "g1")), (4, ("g1", "g2"))):
        dispatcher = gf_native.PlayerInferenceDispatcher(2, 2, 0.0, service.runtime)
        tokens = []
        for game in order:
            tokens.append(
                dispatcher.enqueue(
                    canonical(control(2 if game == "g1" else other_index, game=game)).encode()
                )
            )
        batch = dispatcher.next_batch(time.monotonic() + 2)
        for row in batch["rows"]:
            if json.loads(row["identity"])["game"] == "g2":
                row["inputs"]["state_numerics"][:] *= 9
        answers = service.evaluate(batch)
        own = next(a for a in answers if json.loads(a["identity"])["game"] == "g1")
        if baseline is None:
            baseline = own
        else:
            np.testing.assert_allclose(own["logs"], baseline["logs"], atol=1e-5, rtol=1e-5)
            assert own["value"] == pytest.approx(baseline["value"], abs=1e-5, rel=1e-5)
            assert (
                own["sample"] == baseline["sample"] and own["next_state"] == baseline["next_state"]
            )
        assert dispatcher.submit_results(batch["batch_id"], answers) == 2
        for token in tokens:
            dispatcher.wait_result(token, time.monotonic() + 1)
        dispatcher.close()


@pytest.mark.parametrize("damage", ["count", "order", "sample", "logs", "runtime", "state"])
def test_invalid_ordered_response_retires_batch_without_any_rng_commit(damage):
    dispatcher = gf_native.PlayerInferenceDispatcher(2, 2, 0.0, "torch-test")
    payloads = [control(2, game="g1"), control(3, game="g2")]
    tokens = [dispatcher.enqueue(canonical(p).encode()) for p in payloads]
    batch = dispatcher.next_batch(time.monotonic() + 1)
    answers = [response(row) for row in batch["rows"]]
    if damage == "count":
        answers.pop()
    elif damage == "order":
        answers.reverse()
    elif damage == "sample":
        answers[0]["sample"] = 999999
    elif damage == "logs":
        answers[0]["logs"] = np.zeros(1, dtype=np.float32)
    elif damage == "runtime":
        answers[0]["runtime"] = "foreign"
    else:
        answers[0]["next_state"] = b""
    with pytest.raises(ValueError):
        dispatcher.submit_results(batch["batch_id"], answers)
    for token in tokens:
        with pytest.raises(ValueError):
            dispatcher.wait_result(token, time.monotonic() + 1)
    for payload in payloads:
        payload["identity"]["request"] += 1
        dispatcher.enqueue(canonical(payload).encode())
    assert all(
        row["rng_state"] is None for row in dispatcher.next_batch(time.monotonic() + 1)["rows"]
    )
    dispatcher.close()


def test_maximum_serving_candidate_and_position_geometry_matches_python():
    import random
    from dataclasses import asdict

    from gf_common.card_instance import ActionConfigV3, shared_card_instance_contract
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.diverse_actions import NumericCosts
    from gf_common.domain import digest
    from gf_common.structured import DecisionFrame, EventBank
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_wire import frame_dict
    from tests.card_instance_action_support import inputs

    value = inputs([("abyss-dart", None)] * 18, mp=0)
    bank = EventBank(1, "maximum", layout=layout)
    frame = DecisionFrame.from_input(value, bank, 0, layout=layout)
    actions = ActionConfigV3(
        action_contract=shared_card_instance_contract(),
        max_sequence_length=4,
        max_candidates=65536,
        max_temporary_candidates=80000,
        max_expansions=250000,
        count_states=1,
    )
    search = {
        "frame": frame_dict(frame, layout=layout),
        "events": [],
        "actions": asdict(actions),
        "rules": asdict(card_instance_rules()),
        "numeric_costs": asdict(NumericCosts()),
        "rng_state": random.Random(17).getstate(),
        "clock": [0.0] * 400000,
    }
    generated = json.loads(gf_native.player_search_v5(canonical(search).encode()))["generated"]
    payload = control(3)
    payload.update(
        frame=search["frame"],
        events=[],
        actions=search["actions"],
        rules=search["rules"],
        table=generated["table"],
        max_positions=128,
    )
    dispatcher, _, batch = native_row(payload)
    row = batch["rows"][0]
    refs = row["inputs"]["references"]
    assert row["candidate_count"] == 65536 and refs.shape == (65536, 128)
    table = generated["table"]
    expected = np.full((65536, 128), 18, dtype=np.int64)
    for i, (start, stop) in enumerate(
        zip(table["offsets"][:-1], table["offsets"][1:], strict=True)
    ):
        expected[i, : stop - start] = table["slots"][start:stop]
    np.testing.assert_array_equal(refs, expected)
    assert digest(table)  # Original table/hash was validated at admission.
    dispatcher.close()


def test_retiring_first_request_does_not_extend_existing_formation_window():
    from concurrent.futures import ThreadPoolExecutor

    dispatcher = gf_native.PlayerInferenceDispatcher(3, 3, 220.0, "torch-test")
    first = control(3)
    dispatcher.enqueue(canonical(first).encode())
    started = time.monotonic()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(dispatcher.next_batch, time.monotonic() + 2)
        time.sleep(0.12)
        dispatcher.enqueue(canonical(control(2, game="g2")).encode())
        dispatcher.retire(canonical(first["identity"]).encode())
        batch = future.result(timeout=1)
    assert len(batch["rows"]) == 1
    assert time.monotonic() - started < 0.31
    dispatcher.close()


def test_already_aged_queued_request_does_not_restart_wait_when_service_returns():
    dispatcher = gf_native.PlayerInferenceDispatcher(1, 2, 140.0, "torch-test")
    dispatcher.enqueue(canonical(control(3)).encode())
    time.sleep(0.18)
    at = time.monotonic()
    batch = dispatcher.next_batch(time.monotonic() + 1)
    assert time.monotonic() - at < 0.08 and len(batch["rows"]) == 1
    dispatcher.close()


def test_publication_restores_rng_even_if_loader_changes_defaults_then_raises(monkeypatch):
    import torch

    service, network = inference_service()
    before = torch.get_rng_state().clone()
    cuda = (
        [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
    )

    def failing_publication(self, weights):
        torch.manual_seed(99211)
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(type(service.registry), "load_policy_weights", failing_publication)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        service.publish(network, "b" * 64)
    assert torch.equal(before, torch.get_rng_state())
    assert all(
        torch.equal(a, b)
        for a, b in zip(cuda, torch.cuda.get_rng_state_all() if cuda else [], strict=True)
    )
    assert "b" * 64 not in service.references


def test_real_cuda_owned_transfer_and_row_scoring_match_original_fp32():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware/build unavailable")
    from gf_models.rust_player import RustPlayerInference
    from gf_models.shared_numeric import score_numeric_row, tensor_row

    _, network = inference_service()
    service = RustPlayerInference(
        device="cuda:0", cache_entries=2, cache_weight_bytes=64 * 1024**2, visibility="0"
    )
    service.publish(network, "a" * 64)
    dispatcher, token, batch = native_row(control(2), runtime=service.runtime)
    answers = service.evaluate(batch)
    episode = json.loads(FIXTURE.read_text())["episodes"][2]
    expected = tensor_row(expected_numeric(episode, episode["decisions"][0], [])).to("cuda:0")
    with torch.no_grad():
        logs, value = score_numeric_row(service.registry.networks["a" * 64], expected)
    np.testing.assert_allclose(
        answers[0]["logs"], logs.cpu().numpy().reshape(-1), atol=1e-5, rtol=1e-5
    )
    assert answers[0]["value"] == pytest.approx(float(value.item()), abs=1e-5, rel=1e-5)
    assert dispatcher.submit_results(batch["batch_id"], answers) == 1
    dispatcher.wait_result(token, time.monotonic() + 1)
    assert service.timings["h2d"] > 0 and service.timings["return"] > 0
    dispatcher.close()


def test_cpu_registry_does_not_initialize_cuda_in_a_fresh_process():
    import subprocess
    import sys

    code = """
import torch
from gf_models.rust_player import RustPlayerInference
assert not torch.cuda.is_initialized()
RustPlayerInference(device='cpu',cache_entries=1,cache_weight_bytes=1048576)
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20, capture_output=True)


def test_retire_erases_previously_committed_state_before_new_epoch_handoff():
    payload = control(2)
    dispatcher, token, batch = native_row(payload)
    assert (
        dispatcher.submit_results(
            batch["batch_id"], [response(batch["rows"][0], state=b"old-private-state")]
        )
        == 1
    )
    dispatcher.wait_result(token, time.monotonic() + 1)
    payload["identity"]["request"] = 2
    token = dispatcher.enqueue(canonical(payload).encode())
    second = dispatcher.next_batch(time.monotonic() + 1)
    assert second["rows"][0]["rng_state"] == b"old-private-state"
    dispatcher.retire(canonical(payload["identity"]).encode())
    with pytest.raises(RuntimeError):
        dispatcher.wait_result(token, time.monotonic() + 1)
    assert (
        dispatcher.submit_results(second["batch_id"], [response(second["rows"][0], state=b"late")])
        == 0
    )
    payload["identity"].update(epoch=1, request=1)
    payload["sampling_seed"] = 99
    dispatcher.enqueue(canonical(payload).encode())
    new = dispatcher.next_batch(time.monotonic() + 1)["rows"][0]
    assert new["rng_state"] is None and new["sampling_seed"] == 99
    dispatcher.close()


def test_retire_maximum_epoch_clears_ownership_without_reopening_same_epoch():
    payload = control(2, epoch=2**64 - 1)
    dispatcher, token, batch = native_row(payload)
    dispatcher.retire(canonical(payload["identity"]).encode())
    with pytest.raises(RuntimeError):
        dispatcher.wait_result(token, time.monotonic() + 1)
    assert dispatcher.submit_results(batch["batch_id"], [response(batch["rows"][0])]) == 0
    with pytest.raises(ValueError):
        dispatcher.enqueue(canonical(payload).encode())
    dispatcher.release_game("run", "trial", "g1")
    dispatcher.close()


def test_fix1_retired_blocked_sync_producer_cannot_admit_or_revive():
    from concurrent.futures import ThreadPoolExecutor

    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, "torch-test")
    first, second = control(3, game="g1"), control(3, game="g2")
    dispatcher.enqueue(canonical(first).encode())
    try:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(dispatcher.enqueue, canonical(second).encode())
            time.sleep(0.04)
            assert not future.done()
            with pytest.raises(ValueError):
                dispatcher.release_game("run", "trial", "g2")
            dispatcher.retire(canonical(second["identity"]).encode())
            dispatcher.retire(canonical(first["identity"]).encode())
            with pytest.raises(ValueError):
                future.result(timeout=1)
        second["identity"]["epoch"] = 1
        token = dispatcher.enqueue(canonical(second).encode())
        batch = dispatcher.next_batch(time.monotonic() + 1)
        assert batch["rows"][0]["rng_state"] is None
        assert dispatcher.submit_results(batch["batch_id"], [response(batch["rows"][0])]) == 1
        dispatcher.wait_result(token, time.monotonic() + 1)
    finally:
        dispatcher.close()


def test_fix1_retire_before_first_admission_refuses_same_epoch():
    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, "torch-test")
    payload = control(3)
    dispatcher.retire(canonical(payload["identity"]).encode())
    try:
        with pytest.raises(ValueError):
            dispatcher.enqueue(canonical(payload).encode())
        payload["identity"]["epoch"] += 1
        dispatcher.enqueue(canonical(payload).encode())
        dispatcher.next_batch(time.monotonic() + 1)
    finally:
        dispatcher.close()


@pytest.mark.parametrize("operation", ["next_batch", "wait_result", "enqueue", "formation"])
def test_fix1_unsupported_timer_is_checked_without_poisoning_in_child(operation):
    import subprocess
    import sys

    code = """
import os,sys,time
import gf_native
from gf_common.domain import canonical
from tests.test_rust_player_inference import control,response
operation=sys.argv[1]
d=gf_native.PlayerInferenceDispatcher(1,1,0.,'torch-test')
try:
    if operation=='next_batch':d.next_batch(sys.float_info.max)
    elif operation=='wait_result':
        token=d.enqueue(canonical(control(3)).encode())
        d.wait_result(token,sys.float_info.max)
    elif operation=='enqueue':
        d.enqueue(canonical(control(3,deadline=sys.float_info.max)).encode())
    else:gf_native.PlayerInferenceDispatcher(1,1,sys.float_info.max,'torch-test')
except ValueError:pass
except BaseException:os._exit(42)
else:os._exit(43)
if operation=='wait_result':
    batch=d.next_batch(time.monotonic()+1)
    assert d.submit_results(batch['batch_id'],[response(batch['rows'][0])])==1
    assert d.wait_result(token,time.monotonic()+1)['sample']==0
else:
    try:d.next_batch(time.monotonic()+.001)
    except TimeoutError:pass
    except BaseException:os._exit(44)
    d.enqueue(canonical(control(3)).encode())
    d.next_batch(time.monotonic()+1)
d.cancel('checked timer cleanup')
d.close()
if operation=='next_batch':
    from concurrent.futures import ThreadPoolExecutor
    d=gf_native.PlayerInferenceDispatcher(1,1,0.,'torch-test')
    token=d.enqueue(canonical(control(3)).encode())
    with ThreadPoolExecutor(2) as pool:
        waiter=pool.submit(d.wait_result,token,time.monotonic()+5)
        sender=pool.submit(d.enqueue,canonical(control(3,game='g2')).encode())
        time.sleep(.03)
        try:d.next_batch(sys.float_info.max)
        except ValueError:pass
        except BaseException:os._exit(45)
        d.cancel('preserved blocked waiters')
        for future in (waiter,sender):
            try:future.result(timeout=1)
            except ValueError:pass
            else:os._exit(46)
    # A supported long clock value is cancelable, without an invented short cap.
    d=gf_native.PlayerInferenceDispatcher(1,1,0.,'torch-test')
    with ThreadPoolExecutor(1) as pool:
        receiver=pool.submit(d.next_batch,time.monotonic()+1e10)
        time.sleep(.03)
        assert not receiver.done()
        d.cancel('supported long timer')
        try:receiver.result(timeout=1)
        except ValueError:pass
        else:os._exit(47)
"""
    subprocess.run(
        [sys.executable, "-c", code, operation], check=True, timeout=10, capture_output=True
    )
