"""Rust-owned numeric rows against the original participant cache."""
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import gf_native
import numpy as np
import pytest

from gf_common.card_instance import action_config_v3, sacrifice_card_instance_contract
from gf_common.card_instance_rules import parse_card_instance_rules
from gf_common.domain import ContractError, canonical
from gf_common.structured import EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_common.structured_wire import parse_frame, parse_game_generated_v3
from tests.test_rust_player_cuda import model4
from tests.test_rust_player_inference import FIXTURE, control
from tests.test_rust_player_runtime import game_spec, managed, runtime_config

torch = pytest.importorskip("torch")


def sequence_network(kind, *, seed=17, candidates=32):
    from gf_models.hpo_config import normalize_model_v4
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network

    model = model4()
    model = replace(model, model_id=kind, max_positions=18,
        event=replace(model.event, aggregate_window=128),
        temporal=replace(model.temporal, kind=kind, num_layers=2,
                         num_heads=1, feedforward_size=8))
    return make_network(normalize_model_v4(model), seed,
                        InferenceBindingV2(max_candidates=candidates), layout=LAYOUT)


def numeric_history(index, *, runtime, game="g1"):
    episode = json.loads(FIXTURE.read_text())["episodes"][index]
    bank = EventBank(episode["actor"], episode["name"], layout=LAYOUT)
    for position, decision in enumerate(episode["decisions"]):
        delta = decision["delta"]["events"]
        bank.append(event_delta(episode["actor"], episode["name"], bank.boundary.end,
            bank.boundary.prefix_hash, delta, layout=LAYOUT))
        frame = parse_frame(decision["frame"], layout=LAYOUT)
        generated = parse_game_generated_v3(decision["generated"],
            action_contract=action_config_v3(episode["actions"]).action_contract,
            rules=parse_card_instance_rules(episode["rules"]))
        payload = control(index, game=game, request=position)
        payload.update(frame=decision["frame"], table=decision["generated"]["table"],
                       events=list(bank.view().events()))
        payload["identity"]["revision"] = decision["frame"]["revision"]
        config = runtime_config() | {"runtime_identity": runtime, "sequence_inference": True}
        owner = gf_native.PlayerRuntime(canonical(config).encode())
        owner.dispatcher.enqueue(canonical(payload).encode())
        batch = owner.dispatcher.next_work(time.monotonic() + 3)
        owner.join()
        row = batch["rows"][0]
        assert (row["event_start"], row["event_end"]) == (frame.interval_start, frame.boundary.end)
        np.testing.assert_array_equal(row["candidate_offsets"], decision["generated"]["table"]["offsets"])
        yield batch, tokenize_frame(frame, bank.view(), layout=LAYOUT), tokenize_table(
            generated.table, frame, 18, layout=LAYOUT)


@pytest.fixture(params=[
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        os.environ.get("GF_RUN_CUDA") != "1", reason="explicit physical CUDA checks required")),
])
def device(request):
    if request.param == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def sequence_service(network, device, *, entries=1):
    from gf_common.rust_player import BATCHED_TORCH_SEQUENCE_INFERENCE
    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount

    account = None
    if device != "cpu":
        profile = CudaCheckpointProfile(network.config, network.binding, layout=LAYOUT)
        account = RustPlayerDeviceAccount(global_bytes=1024**3, learner_device="cpu",
            serving_device=device, learner_profile=profile, admitted_profiles=(profile,))
    service = RustPlayerInference(device=device, cache_entries=entries, cache_weight_bytes=1 << 24,
        inference_recipe=BATCHED_TORCH_SEQUENCE_INFERENCE, device_account=account,
        visibility=None if device == "cpu" else "0")
    service.publish(network, "a" * 64)
    return service


def _cache_values(state):
    if state.hidden is not None:
        return (state.hidden,)
    return state.keys[..., :state.position, :], state.values[..., :state.position, :]


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_scalar_numeric_cached_sequence_matches_original_and_only_commits_accepted_rows(kind, device):
    from gf_models.rust_player import provisional_sample

    reference = sequence_network(kind)
    service = sequence_service(reference, device)
    service.bind_collection(("a" * 64,), game_concurrency=1, max_batch_size=1, scheduled_sequence_seats=1)
    reference = reference.to(device)
    cache = reference.new_cache()
    rng = None
    history = numeric_history(8, runtime=service.runtime)
    for position, (batch, frame, candidates) in enumerate(history):
        row = batch["rows"][0]
        row["rng_state"] = rng
        with torch.no_grad():
            context, cards = reference.step(frame, cache)
            logits = reference.factorized_logits(context, cards, candidates, checkpoint=False)
            logs = logits.log_softmax(-1)
            value = reference.value(context)
        expected_sample, expected_rng = provisional_sample(logs.exp(), state=rng, seed=17,
                                                           runtime=service.runtime)
        if position == 1:
            rejected, pending = service.evaluate_sequence(batch)
            service.accept_sequence(pending, [False])
        answers, pending = service.evaluate_sequence(batch)
        result = answers[0]
        np.testing.assert_allclose(result["logs"], logs.cpu().numpy(), rtol=1e-5, atol=1e-5)
        assert result["value"] == pytest.approx(value.item(), rel=1e-5, abs=1e-5)
        assert result["sample"] == expected_sample and result["next_state"] == expected_rng
        if position == 1:
            np.testing.assert_array_equal(result["logs"], rejected[0]["logs"])
        service.accept_sequence(pending, [True])
        rng = result["next_state"]
        state = next(iter(service._sequence.seats.values()))
        expected_storage = (cache.past,) if kind == "gru" else (
            cache.keys[..., :cache.position, :], cache.values[..., :cache.position, :])
        for actual, expected in zip(_cache_values(state), expected_storage, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    service.retire_sequence_game(("run", "trial", "g1"))
    first, _, _ = next(numeric_history(8, runtime=service.runtime))
    answers, pending = service.evaluate_sequence(first)
    assert answers[0]["next_state"]
    service.accept_sequence(pending, [True])


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_fixed_width_group_is_invariant_to_neighbor_order_and_rejection(kind, device):
    network = sequence_network(kind)

    def service(*, width=4):
        owner = sequence_service(network, device)
        owner.bind_collection(("a" * 64,), game_concurrency=4, max_batch_size=width,
                              scheduled_sequence_seats=2)
        owner.begin_collection_metrics()
        return owner

    scalar = service(width=1)
    reference, changed = service(), service()
    target = list(numeric_history(8, runtime=reference.runtime))
    for position, (batch, _, _) in enumerate(target):
        first = batch["rows"][0]
        neighbor = first.copy()
        identity = json.loads(neighbor["identity"])
        identity["game"] = "other-game"
        neighbor["identity"] = json.dumps(identity).encode()
        main, left = reference.evaluate_sequence(batch | {"rows": [first, neighbor]})
        altered = neighbor | {"inputs": neighbor["inputs"] |
            {"state_numerics": neighbor["inputs"]["state_numerics"] + 5}}
        moved, right = changed.evaluate_sequence(batch | {"rows": [altered, first]} if position == 0 else batch)
        expected, single = scalar.evaluate_sequence(batch)
        np.testing.assert_allclose(main[0]["logs"], expected[0]["logs"], rtol=1e-5, atol=1e-5)
        assert main[0]["value"] == pytest.approx(expected[0]["value"], rel=1e-5, abs=1e-5)
        assert main[0]["sample"] == expected[0]["sample"] and main[0]["next_state"] == expected[0]["next_state"]
        scalar.accept_sequence(single, [True])
        moved_row = moved[-1]
        np.testing.assert_array_equal(main[0]["logs"].view(np.uint32), moved_row["logs"].view(np.uint32))
        assert main[0]["value"] == moved_row["value"]
        assert main[0]["sample"] == moved_row["sample"]
        assert main[0]["next_state"] == moved_row["next_state"]
        reference.accept_sequence(left, [True, True])
        changed.accept_sequence(right, [False, True] if position == 0 else [True])
        for actual, expected in zip(_cache_values(left[0].state), _cache_values(right[-1].state), strict=True):
            # CUDA is the deployed grouped recipe. CPU GEMM uses different
            # accumulation order across matrix rows for this small fixture.
            tolerance = 1e-5 if device == "cpu" else 0
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert reference._batch_counts == {2: len(target)}
    metrics = changed.end_collection_metrics()
    assert metrics["batch_width"] == 4
    assert metrics["useful_size_counts"] == {"2": 1, "1": len(target) - 1}
    assert metrics["dummy_rows"] == 3 * len(target) - 1


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GFB_RUN_INTEGRATION") != "1", reason="requires actual Simulator sockets")
def test_failed_retirement_keeps_abnormal_terminal_and_never_admits_replacements(
        tmp_path, monkeypatch, device):
    """Use the actual production service loop, native scheduler and Simulator."""
    import threading

    from gf_simulator_client.rust_player import RustPlayerRuntime

    source = Path("/home/percy/develop/GodFieldBots/runs/pestle-mirage-compat/simulator")
    contract = sacrifice_card_instance_contract()
    service = sequence_service(sequence_network("gru", candidates=64), device)
    service.bind_collection(("a" * 64,), game_concurrency=1, max_batch_size=1,
                            scheduled_sequence_seats=4)
    runtime = RustPlayerRuntime(runtime_config() | {"runtime_identity": service.runtime,
        "game_concurrency": 1}, service=service)
    retiring, release = threading.Event(), threading.Event()
    real_retire = service.retire_sequence_game

    def fail_retirement(key):
        retiring.set()
        if not release.wait(5):
            raise RuntimeError("test did not release retirement")
        if device == "cpu":
            raise RuntimeError("injected retirement failure")
        real_retire(key)

    monkeypatch.setattr(service, "retire_sequence_game", fail_retirement)
    if device != "cpu":
        real_fence = torch.cuda.synchronize

        def fail_fence(device):
            if retiring.is_set():
                raise RuntimeError("injected retirement failure")
            real_fence(device)

        monkeypatch.setattr(torch.cuda, "synchronize", fail_fence)

    try:
        with managed(tmp_path, count=3, source=source, binary=source / "target/release/gf-test-server",
                     contract=contract) as (start, receipt):
            def spec(name):
                game = game_spec(start, receipt, episode=name, kind="gru")
                game["actions"]["action_contract"] = asdict(contract)
                game["max_decisions"] = 1
                for policy in game["policies"]:
                    policy["checkpoint"] = "a" * 64
                return game

            runtime.start([spec("first")])
            pending = runtime.next_storage(time.monotonic() + 10)
            assert pending.kind == "pending"
            runtime.storage_ack(pending)
            acknowledged = runtime.next_storage(time.monotonic() + 10)
            assert acknowledged.kind == "ack"
            runtime.storage_ack(acknowledged)
            assert retiring.wait(5)
            # Queue while the old lifetime still occupies the only slot.
            runtime.start([spec("queued")])
            release.set()
            terminals = {}
            for _ in range(2):
                bundle = runtime.next_storage(time.monotonic() + 10)
                assert bundle.kind == "terminal"
                terminals[bundle.episode] = json.loads(memoryview(bundle.view()).tobytes())["report"]
                runtime.storage_ack(bundle)
            runtime.wait(time.monotonic() + 5)
            assert terminals["first"]["game"]
            assert not terminals["first"]["inference_released"]
            assert any("inference release" in error for error in terminals["first"]["cleanup_errors"])
            assert terminals["queued"]["game"] is None
            assert terminals["queued"]["end_kind"] == "cancelled"
            with pytest.raises(ValueError, match="runtime admission closed"):
                runtime.start([spec("late")])
            assert service.references["a" * 64] > 0
            with pytest.raises(RuntimeError, match="injected retirement failure"):
                runtime.join()
            assert not runtime._service_thread.is_alive()
            if device == "cpu":
                assert service.references["a" * 64] == 0 and not service._sequence.seats
            else:
                assert service.references["a" * 64] > 0 and service._sequence.seats
                assert service.device_account._sequence_cache_bytes > 0
    finally:
        release.set()
        if not runtime._joined:
            runtime.cancel("test cleanup")
            try:
                runtime.join()
            except RuntimeError:
                pass


def _routing(batch, **changes):
    row = batch["rows"][0]
    identity = json.loads(row["identity"])
    return batch | {"rows": [row | {"identity": canonical(identity | changes).encode()}]}


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_different_history_positions_and_immutable_routes_isolate_sequence_state(kind, device):
    network = sequence_network(kind)
    service = sequence_service(network, device, entries=2)
    service.publish(sequence_network(kind, seed=91), "b" * 64)
    service.bind_collection(("a" * 64, "b" * 64), game_concurrency=4, max_batch_size=4,
                            scheduled_sequence_seats=8)
    first, second, third, fourth = [batch for batch, _, _ in numeric_history(8, runtime=service.runtime)]
    for batch in (first, second, third):
        _, steps = service.evaluate_sequence(batch)
        service.accept_sequence(steps, [True])
    expected, old_steps = service.evaluate_sequence(fourth)
    previous = tuple(tensor.clone() for tensor in _cache_values(old_steps[0].state))
    independent = {name: value for name, value in (
        ("run", "other-run"), ("trial", "other-trial"), ("game", "other-game"),
        ("player", 2), ("epoch", 1), ("checkpoint", "b" * 64))}
    for name, value in independent.items():
        with pytest.raises(ContractError, match="continuity"):
            service.evaluate_sequence(_routing(second, **{name: value}))
    varied = _routing(first, game="other-game")
    mixed, steps = service.evaluate_sequence(fourth | {"rows": [varied["rows"][0], fourth["rows"][0]]})
    np.testing.assert_array_equal(expected[0]["logs"], mixed[1]["logs"])
    assert expected[0]["value"] == mixed[1]["value"]
    assert expected[0]["next_state"] == mixed[1]["next_state"]
    for before, after in zip(previous, _cache_values(steps[1].state), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    new, new_steps = service.evaluate_sequence(_routing(first, checkpoint="b" * 64, game="other-game"))
    assert new
    service.accept_sequence(new_steps, [True])
    service.accept_sequence(steps, [False, True])
    assert steps[1].state.position == 4
    assert new_steps[0].state.position == 1
    assert steps[0].key not in service._sequence.seats


@pytest.mark.skipif(os.environ.get("GF_RUN_CUDA") != "1", reason="explicit physical CUDA checks required")
@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_retirement_returns_actual_cuda_cache_storage_and_collection_allowance(kind):
    from gf_models.rust_player_sequence import cache_bytes

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    network = sequence_network(kind)
    service = sequence_service(network, "cuda:0")
    account = service.device_account
    available = account.serving_tensor_allowance()
    service.bind_collection(("a" * 64,), game_concurrency=4, max_batch_size=4,
                            scheduled_sequence_seats=2)
    reserve = (4 if kind == "gru" else 2) * cache_bytes(network.config)
    assert account.serving_tensor_allowance() == available - reserve
    first, _, _ = next(numeric_history(8, runtime=service.runtime))
    for _ in range(2):
        before = torch.cuda.memory_allocated()
        _, pending = service.evaluate_sequence(first)
        service.accept_sequence(pending, [True])
        del pending
        state = next(iter(service._sequence.seats.values()))
        assert sum(value.untyped_storage().nbytes() for value in _cache_values(state)) == cache_bytes(network.config)
        del state
        assert torch.cuda.memory_allocated() >= before + cache_bytes(network.config)
        service.retire_sequence_game(("run", "trial", "g1"))
        assert torch.cuda.memory_allocated() == before
    service.end_collection()
    assert account.serving_tensor_allowance() == available


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GFB_RUN_INTEGRATION") != "1", reason="requires actual Simulator sockets")
@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
def test_missing_retirement_acknowledgement_cancels_native_slots_at_original_deadline(tmp_path, scheduler):
    from tests.test_rust_player_inference import response

    source = Path("/home/percy/develop/GodFieldBots/runs/pestle-mirage-compat/simulator")
    contract = sacrifice_card_instance_contract()
    config = runtime_config(scheduler=scheduler) | {"game_concurrency": 1, "sequence_inference": True}
    native = gf_native.PlayerRuntime(canonical(config).encode())
    dispatcher = native.dispatcher
    try:
        with managed(tmp_path, count=3, source=source, binary=source / "target/release/gf-test-server",
                     contract=contract) as (start, receipt):
            def spec(name):
                game = game_spec(start, receipt, episode=name, kind="gru")
                game["actions"]["action_contract"] = asdict(contract)
                game["max_decisions"] = 1
                for policy in game["policies"]:
                    policy.update(checkpoint="a" * 64, limits={"max_decisions": 1024,
                        "max_episode_events": 128, "max_interval_events": 64,
                        "max_candidates": 64, "max_positions": 18, "aggregate_window": 128})
                return game

            first = spec("first")
            first.update(deadline=time.monotonic() + 3, cleanup_deadline=time.monotonic() + 4)
            native.start(canonical([first]).encode())
            batch = dispatcher.next_work(time.monotonic() + 2)
            assert dispatcher.submit_rows(batch["batch_id"],
                [response(batch["rows"][0], runtime=config["runtime_identity"])]) == [True]
            for kind in ("pending", "ack"):
                bundle = native.next_storage(time.monotonic() + 2)
                assert bundle.kind == kind
                native.storage_ack(bundle)
            work = dispatcher.next_work(time.monotonic() + 2)
            ticket, key = work["retirement"]
            assert key[:2] == ("run", "trial")
            native.start(canonical([spec("queued")]).encode())
            # Leave the original retirement ticket outstanding on purpose.
            native.wait(time.monotonic() + 7)
            with pytest.raises(TimeoutError, match="native preparation deadline"):
                dispatcher.ack_retirement(ticket)
            with pytest.raises(ValueError, match="runtime admission closed"):
                native.start(canonical([spec("late")]).encode())
            first, queued = [row["report"] for row in json.loads(native.completed_reports(
                [("run", "trial", "first"), ("run", "trial", "queued")]))]
            assert first["game"] == key[2] and not first["inference_released"]
            assert any("inference release" in item for item in first["cleanup_errors"])
            assert queued["end_kind"] == "cancelled" and queued["game"] is None
    finally:
        native.cancel("test cleanup")
        native.join()


def test_transformer_temporal_estimate_charges_gather_and_scaled_key_prefix():
    from gf_models.rust_player_sequence import temporal_scratch

    model = sequence_network("transformer").config
    model = replace(model, temporal=replace(model.temporal, hidden_size=1024,
        num_heads=1, num_layers=1, feedforward_size=8))
    scalar, largest_scalar = temporal_scratch(model, rows=1, position=1023, grouped=False)
    grouped, largest_grouped = temporal_scratch(model, rows=8, position=1023, grouped=True)
    # A single layer reaches 4 MiB per row for each of K and V. Math SDPA
    # allocates one more whole K tensor while both group prefixes are live.
    assert scalar > 4 * 1024**2
    assert grouped > 3 * 32 * 1024**2
    assert largest_scalar == 4 * 1024**2
    assert largest_grouped == 32 * 1024**2
