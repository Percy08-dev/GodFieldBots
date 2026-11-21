"""Owned fixed-geometry inference checks using actual native operands."""

import json
import struct
import time
from dataclasses import replace

import gf_native
import numpy as np
import pytest

from gf_common.domain import ContractError, canonical
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from tests.model3_support import model3
from tests.test_rust_player_cuda import model4
from tests.test_rust_player_inference import control, for_player, native_row


def _model(version):
    if version == 3:
        model = model3()
    else:
        model = replace(model4(), max_positions=18)
    return replace(model, event=replace(model.event, num_layers=2))


def _network(version, seed=17, *, workload=False):
    from gf_models.structured_config import InferenceBindingV2, ModelConfigV3, ModelConfigV4
    from gf_models.structured_network import make_network

    if not workload:
        return make_network(_model(version), seed, InferenceBindingV2(
            candidate_chunk=128, max_candidates=2048), layout=LAYOUT)
    # Ordinary's repository default and the measured Model4 HPO architecture.
    model = ModelConfigV3()
    if version == 4:
        model = ModelConfigV4()
        model = replace(model, state=replace(model.state, hidden_sizes=(192,)),
            hand=replace(model.hand, hidden_sizes=(192,), latent_size=192),
            temporal=replace(model.temporal, hidden_size=192, num_layers=2),
            heads=replace(model.heads, actor_hidden_sizes=(192,), critic_hidden_sizes=(192,)))
    return make_network(model, seed, InferenceBindingV2(), layout=LAYOUT)


def _batch(runtime, cases, *, capacity=4):
    dispatcher = gf_native.PlayerInferenceDispatcher(capacity, capacity, 0.0, runtime)
    tokens = []
    for case, name, checkpoint in cases:
        payload = control(case, game=name)
        payload["identity"]["checkpoint"] = checkpoint
        tokens.append(dispatcher.enqueue(canonical(payload).encode()))
    return dispatcher, tokens, dispatcher.next_batch(time.monotonic() + 5)


def _assert_same(actual, expected):
    np.testing.assert_array_equal(actual["logs"].view(np.uint32), expected["logs"].view(np.uint32))
    assert struct.pack("f", actual["value"]) == struct.pack("f", expected["value"])
    assert actual["sample"] == expected["sample"]
    assert actual["next_state"] == expected["next_state"]


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("case", [0, 2, 3, 6])
def test_batch_no_grad_agrees_with_frozen_cpu_scalar_reference(version, case):
    import torch

    from gf_models.rust_player import tensor_views
    from gf_models.rust_player_batch import (
        BatchGeometry,
        BatchInputs,
        pack_native_rows,
        score_numeric_batch,
    )
    from gf_models.shared_numeric import score_numeric_row

    network = _network(version)
    d, _, batch = native_row(control(case))
    try:
        row = batch["rows"][0]
        operands = row["inputs"]
        shape = BatchGeometry(4, 2048, 8).shape(
            len(operands["kinds"]), len(operands["interval_numerics"]))
        packed = pack_native_rows([row], BatchInputs.compile(shape, positions=18, layout=LAYOUT))
        counts = len(operands["kinds"])
        with torch.no_grad():
            expected, expected_value = score_numeric_row(network, tensor_views(operands, device="cpu"))
            logs, probabilities, values = score_numeric_batch(network, packed.transfer(device="cpu", roots={}))
        assert not logs.requires_grad and not probabilities.requires_grad and not values.requires_grad
        np.testing.assert_allclose(logs[0, :counts].numpy(), expected.reshape(-1).numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(values[0].item(), expected_value.item(), atol=1e-5, rtol=1e-5)
        assert all(torch.count_nonzero(probabilities[i]).item() == 1 for i in (1, 2, 3))
    finally:
        d.close()


def test_runtime_recipe_is_opt_in_without_rewriting_existing_declaration():
    from gf_common.rust_player import player_runtime
    from tests.test_rust_player_config import runtime

    old = runtime()
    assert player_runtime(old, game_concurrency=old["game_concurrency"]) == old | {"formation_wait_ms": 0.0}
    assert player_runtime(old | {"inference_recipe": "batched-torch-v1"},
        game_concurrency=old["game_concurrency"])["inference_recipe"] == "batched-torch-v1"
    for invalid in (None, "scalar", "unknown"):
        with pytest.raises(ContractError):
            player_runtime(old | {"inference_recipe": invalid}, game_concurrency=old["game_concurrency"])


def test_checkpoint_capacity_uses_fixed_geometry_and_observed_residency():
    from gf_models.rust_player_batch import BatchGeometry, BatchShape, estimate_batch

    model = _model(3)
    binding = _network(3).binding
    geometry = BatchGeometry(4, 15, 6)
    assert geometry.shape(13, 5) == BatchShape(4, 15, 6)
    assert geometry.shape(1, 0) == BatchShape(4, 1, 0)
    assert geometry.shape(3, 1) == BatchShape(4, 4, 1)
    with pytest.raises(ContractError):
        geometry.shape(16, 1)
    shape = geometry.shape(13, 5)
    estimate = estimate_batch(model, binding, layout=LAYOUT, shape=shape)
    dispatcher, _, batch = native_row(control(0))
    dispatcher.close()
    operands = batch["rows"][0]["inputs"]
    exact_leaves = 4 * (sum(v.nbytes for k, v in operands.items() if k not in (
        "kinds", "targets", "numerics", "present", "references") and not k.startswith("interval_"))
        + 15 * (8 + 8 + 16 + 16 + 18 * 8) + 6 * (2 * 11 * 4 + len(LAYOUT.event_categories) * 8) + 15 + 8)
    assert estimate.input_bytes == exact_leaves
    assert estimate.peak_bytes > estimate.input_bytes
    assert estimate.largest_tensor_bytes >= 4 * 15 * model.max_positions * model.heads.actor_hidden_sizes[0] * 4


def _cuda_services(version, *, capacity=4, workload=False, publish_next=True):
    import torch

    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware/build unavailable")
    net = _network(version, workload=workload)
    profile = CudaCheckpointProfile(net.config, net.binding, layout=LAYOUT)
    account = RustPlayerDeviceAccount(global_bytes=1024**3, learner_device="cpu",
        serving_device="cuda:0", learner_profile=profile, admitted_profiles=(profile,))
    scalar = RustPlayerInference(device="cuda:0", cache_entries=2, cache_weight_bytes=64 * 1024**2,
        visibility="0")
    batched = RustPlayerInference(device="cuda:0", cache_entries=2, cache_weight_bytes=64 * 1024**2,
        visibility="0", device_account=account, inference_recipe="batched-torch-v1")
    a, b = "a" * 64, "b" * 64
    for service in (scalar, batched):
        service.publish(net, a)
        if publish_next:
            service.publish(_network(version, seed=321, workload=workload), b)
    batched.bind_collection((a, b) if publish_next else (a,),
        game_concurrency=capacity, max_batch_size=4 * capacity)
    return scalar, batched, account


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("capacity", [4, 16])
def test_cuda_rows_are_bit_independent_and_keep_original_tokens(version, capacity):
    from gf_models.rust_player import RustPlayerInference

    scalar, batched, _ = _cuda_services(version, capacity=capacity, workload=capacity == 16)
    a, b = "a" * 64, "b" * 64
    frozen_cpu = None
    if capacity == 16:
        frozen_cpu = RustPlayerInference(device="cpu", cache_entries=2, cache_weight_bytes=64 * 1024**2)
        frozen_cpu.publish(_network(version, workload=True), a)
        frozen_cpu.publish(_network(version, seed=321, workload=True), b)
    baseline = None
    variants = [
        [(0, "own", a)],
        [(1, "first", a), (0, "own", a)],
        [(1, "first", a), (0, "own", a), (3, "empty", a), (6, "large", b)],
        [(1, "first", a), (0, "own", a), (1, "third", a), (0, "fourth", a)],
        [(1, "first", b), (1, "second", a), (0, "third", a), (0, "own", a)],
        [(index % 2, f"full-{index}", a) for index in range(capacity - 1)] + [(0, "own", a)],
    ]
    for cases in variants:
        dispatcher, tokens, batch = _batch(batched.runtime, cases, capacity=capacity)
        try:
            for row in batch["rows"]:
                if json.loads(row["identity"])["game"] != "own":
                    row["inputs"]["state_numerics"][:] *= 13
            answers = batched.evaluate(batch)
            assert [answer["identity"] for answer in answers] == [row["identity"] for row in batch["rows"]]
            assert dispatcher.submit_results(batch["batch_id"], answers) == len(cases)
            for token in tokens:
                dispatcher.wait_result(token, time.monotonic() + 2)
            for actual, expected in zip(answers, scalar.evaluate(batch), strict=True):
                np.testing.assert_allclose(actual["logs"], expected["logs"], atol=1e-5, rtol=1e-5)
                assert actual["value"] == pytest.approx(expected["value"], abs=1e-5, rel=1e-5)
                assert actual["sample"] == expected["sample"]
                assert actual["next_state"] == expected["next_state"]
            if frozen_cpu is not None:
                for actual, expected in zip(answers, frozen_cpu.evaluate(batch), strict=True):
                    np.testing.assert_allclose(actual["logs"], expected["logs"], atol=1e-5, rtol=1e-5)
                    assert actual["value"] == pytest.approx(expected["value"], abs=1e-5, rel=1e-5)
            own = next(answer for answer in answers if json.loads(answer["identity"])["game"] == "own")
            if baseline is None:
                baseline = own
            else:
                _assert_same(own, baseline)
        finally:
            dispatcher.close()


def test_cuda_over_budget_shape_falls_back_before_batch_transfer(monkeypatch):
    from gf_models.rust_player_batch import PackedRows

    scalar, batched, account = _cuda_services(3)
    a = "a" * 64
    geometry, plan = batched._collection[a]
    small, large = plan[geometry.shape(1, 0)], plan[geometry.shape(2048, 0)]
    assert small and large and small.peak_bytes < large.peak_bytes
    # Keep real device residency and leave enough headroom for a scalar large
    # request while setting the batch precisely one byte over the allowance.
    account.global_bytes = account.host_fixed_bytes + account.learner_tensor_bytes + (
        account.retained_cache_tensor_bytes + large.peak_bytes - 1)
    batched.bind_collection((a,), game_concurrency=4, max_batch_size=16)
    assert batched._collection[a][1][geometry.shape(1, 0)]
    assert batched._collection[a][1][geometry.shape(2048, 0)] is None
    permitted = PackedRows.transfer
    transferred = []

    def transfer(self, **options):
        transferred.append(next(field.shape for field in self.inputs.fields if field.name == "kinds"))
        return permitted(self, **options)

    monkeypatch.setattr(PackedRows, "transfer", transfer)
    dispatcher, _, batch = _batch(batched.runtime, [(6, "large1", a), (3, "small", a), (6, "large2", a)])
    try:
        actual = batched.evaluate(batch)
        expected = scalar.evaluate(batch)
        assert transferred == [(4, 1)]
        _assert_same(actual[0], expected[0])
        _assert_same(actual[2], expected[2])
    finally:
        dispatcher.close()


def test_cuda_partial_batch_transfer_is_held_until_worker_stops_and_fences(monkeypatch):
    import threading

    import torch

    _, service, account = _cuda_services(3)
    d, _, batch = _batch(service.runtime, [(0, "own", "a" * 64)])
    original_to = torch.Tensor.to
    attempts = 0
    failures = []

    def fail_second_h2d(tensor, destination, *args, **kwargs):
        nonlocal attempts
        if destination == "cuda:0":
            attempts += 1
            if attempts == 2:
                raise RuntimeError("second batched H2D failed")
        return original_to(tensor, destination, *args, **kwargs)

    def evaluate():
        try:
            service.evaluate(batch)
        except RuntimeError as error:
            failures.append(str(error))

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(torch.Tensor, "to", fail_second_h2d)
            worker = threading.Thread(target=evaluate)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive()
        assert failures == ["second batched H2D failed"] and attempts == 2
        hold = account._active
        assert hold is not None and hold.state == "FAILED_HELD"
        assert len(hold.references) == 1 and next(iter(hold.references.values())).is_cuda
        assert hold._failure_traceback is not None
        service.retire_device_failure(dispatcher=d)
        assert account._active is None and hold.references is None and hold._failure_traceback is None
    finally:
        d.close()


def test_cuda_retired_token_drops_old_checkpoint_reply_without_rng_commit():
    scalar, service, _ = _cuda_services(4)
    a, b = "a" * 64, "b" * 64
    stale = control(0, game="retired")
    d = gf_native.PlayerInferenceDispatcher(4, 4, 0.0, service.runtime)
    try:
        old_token = d.enqueue(canonical(stale).encode())
        active = control(1, game="current")
        active["identity"]["checkpoint"] = b
        new_token = d.enqueue(canonical(active).encode())
        batch = d.next_batch(time.monotonic() + 2)
        answers = service.evaluate(batch)
        old_reference = scalar.evaluate(batch)[0]
        np.testing.assert_allclose(answers[0]["logs"], old_reference["logs"], atol=1e-5, rtol=1e-5)
        d.retire(canonical(stale["identity"]).encode())
        assert d.submit_results(batch["batch_id"], answers) == 1
        with pytest.raises(RuntimeError):
            d.wait_result(old_token, time.monotonic() + 2)
        assert d.wait_result(new_token, time.monotonic() + 2)["sample"] == answers[1]["sample"]
        stale["identity"].update(epoch=1, checkpoint=a)
        d.enqueue(canonical(stale).encode())
        latest = d.next_batch(time.monotonic() + 2)
        assert latest["rows"][0]["rng_state"] is None
    finally:
        d.close()


def test_cuda_collection_peaks_are_scoped_and_keep_process_high_water():
    import torch

    scalar, batched, _ = _cuda_services(3)
    for service in (scalar, batched):
        # Deliberately create a peak outside the collection.
        outside = torch.empty(24 * 1024**2, device="cuda:0", dtype=torch.float32)
        old_allocated = torch.cuda.max_memory_allocated(0)
        del outside
        service.begin_collection_metrics()
        d, _, batch = _batch(service.runtime, [(0, "own", "a" * 64)])
        try:
            service.evaluate(batch)
            scope = service.end_collection_metrics()
        finally:
            d.close()
        assert scope["cuda_peak_allocated_bytes"] < old_allocated
        assert scope["cuda_peak_reserved_bytes"] >= scope["cuda_peak_allocated_bytes"]
        assert service.process_cuda_peaks()[0] >= old_allocated
        if service is scalar:
            assert scope["scalar_rows"] == 1 and scope["batched_groups"] == 0
        else:
            assert scope["useful_size_counts"] == {"1": 1} and scope["dummy_rows"] == 3


@pytest.mark.parametrize("limit", ["model", "binding"])
def test_checkpoint_logical_limit_gates_shape_even_when_host_fits(limit):
    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_batch import estimate_batch
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount

    network = _network(3)
    model, binding = network.config, network.binding
    cap = 780 * 1024**2
    if limit == "model":
        model = replace(model, resources=replace(model.resources, max_owned_bytes=cap))
    else:
        binding = replace(binding, managed_budget_bytes=cap)
    profile = CudaCheckpointProfile(model, binding, layout=LAYOUT)
    account = RustPlayerDeviceAccount(global_bytes=1024**3, learner_device="cpu",
        serving_device="cuda:0", learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache("checkpoint", profile, {("cuda", 0, 1, profile.network_bytes): profile.network_bytes})
    service = RustPlayerInference.__new__(RustPlayerInference)
    service.device, service.device_account, service.inference_recipe = "cuda:0", account, "batched-torch-v1"
    service.bind_collection(("checkpoint",), game_concurrency=64, max_batch_size=64)
    geometry, plan = service._collection["checkpoint"]
    large = geometry.shape(2048, 0)
    estimate = estimate_batch(model, binding, layout=LAYOUT, shape=large)
    assert estimate.peak_bytes < account.serving_tensor_allowance()
    assert estimate.largest_tensor_bytes < model.resources.max_tensor_bytes
    assert plan[geometry.shape(1, 0)] is not None
    assert plan[large] is None


def test_cuda_same_game_seats_and_reused_game_ids_keep_own_rng_namespace():
    _, service, _ = _cuda_services(3)
    a = "a" * 64

    def roundtrip(dispatcher, payload, expected_state):
        token = dispatcher.enqueue(canonical(payload).encode())
        batch = dispatcher.next_batch(time.monotonic() + 3)
        row = batch["rows"][0]
        assert row["rng_state"] == expected_state
        answer, = service.evaluate(batch)
        assert dispatcher.submit_results(batch["batch_id"], [answer]) == 1
        accepted = dispatcher.wait_result(token, time.monotonic() + 3)
        assert json.loads(accepted["identity"]) == payload["identity"]
        assert accepted["sample"] == answer["sample"]
        return answer

    def fresh_dispatcher():
        return gf_native.PlayerInferenceDispatcher(4, 4, 0.0, service.runtime)

    first = control(2, game="reused", trial="primary")
    opponent = for_player(control(4, game="reused", trial="primary"), 2)
    opponent["sampling_seed"] = 233
    assert opponent["frame"]["actor"] == opponent["identity"]["player"] == 2
    alternate = control(2, game="reused", trial="shadow")
    dispatcher = fresh_dispatcher()
    try:
        own_first = roundtrip(dispatcher, first, None)
        other_first = roundtrip(dispatcher, opponent, None)
        shadow_first = roundtrip(dispatcher, alternate, None)
        _assert_same(shadow_first, own_first)

        opponent["identity"]["request"] += 1
        other_second = roundtrip(dispatcher, opponent, other_first["next_state"])
        first["identity"]["request"] += 1
        own_second = roundtrip(dispatcher, first, own_first["next_state"])
        alternate["identity"]["request"] += 1
        shadow_second = roundtrip(dispatcher, alternate, shadow_first["next_state"])
        _assert_same(own_second, shadow_second)
        assert own_second["next_state"] != other_second["next_state"]
    finally:
        dispatcher.close()

    # The same run/trial/game/player/checkpoint identifiers in a new runtime
    # execution start from the new dispatcher's original authorized seed.
    replacement = fresh_dispatcher()
    try:
        repeated = control(2, game="reused", trial="primary")
        assert repeated["identity"]["checkpoint"] == a
        _assert_same(roundtrip(replacement, repeated, None), own_first)
    finally:
        replacement.close()


def test_cpu_opt_in_still_uses_original_scalar_recipe():
    from gf_models.rust_player import RustPlayerInference

    net = _network(3)
    baseline = RustPlayerInference(device="cpu", cache_entries=1, cache_weight_bytes=64 * 1024**2)
    opted_in = RustPlayerInference(device="cpu", cache_entries=1, cache_weight_bytes=64 * 1024**2,
        inference_recipe="batched-torch-v1")
    a = "a" * 64
    for service in (baseline, opted_in):
        service.publish(net, a)
    opted_in.bind_collection((a,), game_concurrency=16, max_batch_size=64)
    opted_in.begin_collection_metrics()
    d, _, batch = _batch(opted_in.runtime, [(0, "own", a), (1, "other", a)])
    try:
        for actual, expected in zip(opted_in.evaluate(batch), baseline.evaluate(batch), strict=True):
            _assert_same(actual, expected)
        scope = opted_in.end_collection_metrics()
        assert scope["recipe"] == "scalar-torch-v1" and scope["scalar_rows"] == 2
        assert scope["batch_width"] is None and scope["cuda_peak_allocated_bytes"] is None
    finally:
        d.close()


def test_cuda_pending_old_checkpoint_completes_across_new_publication():
    scalar, service, _ = _cuda_services(3, publish_next=False)
    a, b = "a" * 64, "b" * 64
    old = control(0, game="old")
    dispatcher = gf_native.PlayerInferenceDispatcher(4, 4, 0.0, service.runtime)
    try:
        # Native takes its real reference before the later immutable publish.
        old_token = dispatcher.enqueue(canonical(old).encode())
        assert dispatcher.checkpoint_references() == {a: 1}
        service.end_collection()
        next_policy = _network(3, seed=321)
        service.publish(next_policy, b)
        scalar.publish(next_policy, b)
        service.bind_collection((a, b), game_concurrency=4, max_batch_size=16)
        new = control(0, game="current")
        new["identity"]["checkpoint"] = b
        new_token = dispatcher.enqueue(canonical(new).encode())
        formed = dispatcher.next_batch(time.monotonic() + 2)
        answers = service.evaluate(formed)
        originals = scalar.evaluate(formed)
        assert [json.loads(row["identity"])["checkpoint"] for row in answers] == [a, b]
        assert not np.array_equal(answers[0]["logs"], answers[1]["logs"])
        for actual, expected in zip(answers, originals, strict=True):
            np.testing.assert_allclose(actual["logs"], expected["logs"], atol=1e-5, rtol=1e-5)
            assert actual["value"] == pytest.approx(expected["value"], abs=1e-5, rel=1e-5)
            assert actual["next_state"] == expected["next_state"]
        assert dispatcher.submit_results(formed["batch_id"], answers) == 2
        for token, answer in zip((old_token, new_token), answers, strict=True):
            actual = dispatcher.wait_result(token, time.monotonic() + 2)
            assert actual["identity"] == answer["identity"] and actual["sample"] == answer["sample"]
        assert dispatcher.checkpoint_references() == {}
    finally:
        dispatcher.close()


def test_cuda_multichunk_index_lifetimes_veto_group_before_transfer(monkeypatch):
    import torch

    from gf_models.rust_player_batch import PackedRows

    scalar, service, account = _cuda_services(3, capacity=16, publish_next=False)
    a = "a" * 64
    # In the pinned CUDA implementation, differently strided broadcast
    # indices coexist with gather output. Strided reference slices copy too.
    batch_width, chunk, positions, hidden = 16, 128, 18, 16
    projected = torch.zeros((batch_width, positions, 19, hidden), device="cuda:0")
    refs = torch.zeros((batch_width, 512, positions), device="cuda:0", dtype=torch.long)
    rows = torch.arange(batch_width, device="cuda:0")[:, None, None]
    pos = torch.arange(positions, device="cuda:0")[None]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    gather = projected[rows, pos, refs[:, :chunk]]
    torch.cuda.synchronize()
    index_peak = torch.cuda.max_memory_allocated() - before - gather.nbytes
    assert index_peak == 3 * 8 * batch_width * chunk * positions
    del gather, projected, refs, rows, pos

    # Full group needs more than 6 MiB once the three index tensors overlap.
    # A scalar 408-candidate request still fits.
    allowance = 6 * 1024**2
    account.global_bytes = account.host_fixed_bytes + account.retained_cache_tensor_bytes + allowance
    service.bind_collection((a,), game_concurrency=16, max_batch_size=64)
    geometry, plan = service._collection[a]
    assert plan[geometry.shape(3, 4)] is not None
    assert plan[geometry.shape(408, 0)] is None

    def forbidden_transfer(*args, **kwargs):
        pytest.fail("over-budget group attempted a batched transfer")

    monkeypatch.setattr(PackedRows, "transfer", forbidden_transfer)
    d, _, batch = _batch(service.runtime, [(7, "multi-1", a), (7, "multi-2", a)], capacity=16)
    try:
        for actual, expected in zip(service.evaluate(batch), scalar.evaluate(batch), strict=True):
            _assert_same(actual, expected)
    finally:
        d.close()


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("restriction", ["default", "explicit", "host", "model", "binding"])
def test_width_32_batch_temporary_limit_preserves_other_admission_limits(version, restriction):
    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_batch import BatchGeometry, estimate_batch
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount

    network = _network(version, workload=True)
    model, binding = network.config, network.binding
    geometry = BatchGeometry(32, binding.max_candidates, model.event.max_interval)
    widened, oversized = geometry.shape(408, 0), geometry.shape(32768, 0)
    fitted = estimate_batch(model, binding, layout=LAYOUT, shape=widened)
    rejected = estimate_batch(model, binding, layout=LAYOUT, shape=oversized)
    assert 64 * 1024**2 < fitted.largest_tensor_bytes < 128 * 1024**2
    assert rejected.largest_tensor_bytes > 128 * 1024**2
    profile = CudaCheckpointProfile(model, binding, layout=LAYOUT)
    tight = profile.fixed_bytes + profile.network_bytes + fitted.peak_bytes - 1
    if restriction == "explicit":
        model = replace(model, resources=replace(model.resources, max_tensor_bytes=60 * 1024**2))
    elif restriction == "model":
        model = replace(model, resources=replace(model.resources, max_owned_bytes=tight))
    elif restriction == "binding":
        binding = replace(binding, managed_budget_bytes=tight)
    profile = CudaCheckpointProfile(model, binding, layout=LAYOUT)
    account = RustPlayerDeviceAccount(global_bytes=tight if restriction == "host" else 12 * 1024**3,
        learner_device="cpu", serving_device="cuda:0", learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache("checkpoint", profile, {("cuda", 0, 1, profile.network_bytes): profile.network_bytes})
    service = RustPlayerInference.__new__(RustPlayerInference)
    service.device, service.device_account, service.inference_recipe = "cuda:0", account, "batched-torch-v1"

    service.bind_collection(("checkpoint",), game_concurrency=32, max_batch_size=64)
    bound_geometry, plan = service._collection["checkpoint"]
    assert bound_geometry == geometry
    assert plan[geometry.shape(32, 0)] is not None
    assert (plan[widened] is not None) == (restriction == "default")
    assert plan[oversized] is None


@pytest.mark.parametrize("version", [3, 4])
def test_cuda_width_32_newly_admitted_shape_matches_scalar_and_is_bit_independent(version):
    from gf_models.rust_player import RustPlayerInference

    width, checkpoint = 32, "a" * 64
    scalar, batched, _ = _cuda_services(version, capacity=width, workload=True, publish_next=False)
    reference = RustPlayerInference(device="cpu", cache_entries=1, cache_weight_bytes=64 * 1024**2)
    reference.publish(_network(version, workload=True), checkpoint)
    own = (7, "own", checkpoint)  # 408 real candidates, fixed bucket width 512.
    baseline = None
    for cases, counts in (
        ([own], {"1": 1}),
        ([(0, "different-events", checkpoint), (6, "different-count", checkpoint),
          (7, "peer", checkpoint), own], {"1": 2, "2": 1}),
        ([(7, f"peer-{index}", checkpoint) for index in range(width - 1)] + [own], {"32": 1}),
    ):
        dispatcher, tokens, batch = _batch(batched.runtime, cases, capacity=width)
        try:
            own_index = next(index for index, row in enumerate(batch["rows"])
                             if json.loads(row["identity"])["game"] == "own")
            for index, row in enumerate(batch["rows"]):
                if index != own_index:
                    row["inputs"]["state_numerics"][:] *= index + 2
            batched.begin_collection_metrics()
            answers = batched.evaluate(batch)
            metrics = batched.end_collection_metrics()
            assert metrics["scalar_rows"] == 0 and metrics["useful_size_counts"] == counts
            actual = answers[own_index]
            single = batch | {"rows": [batch["rows"][own_index]]}
            for service in (scalar, reference):
                expected, = service.evaluate(single)
                np.testing.assert_allclose(actual["logs"], expected["logs"], atol=1e-5, rtol=1e-5)
                assert actual["value"] == pytest.approx(expected["value"], abs=1e-5, rel=1e-5)
                assert actual["sample"] == expected["sample"]
                assert actual["next_state"] == expected["next_state"]
            if baseline is None:
                baseline = actual
            else:
                _assert_same(actual, baseline)
            assert dispatcher.submit_results(batch["batch_id"], answers) == len(cases)
            for token in tokens:
                dispatcher.wait_result(token, time.monotonic() + 2)
        finally:
            dispatcher.close()
