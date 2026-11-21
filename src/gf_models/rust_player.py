"""Owned native serving views and immutable MLP inference, without a broker.

The host configures Torch/device visibility before concurrency. Learner network
initialization, update, restore and RNG/snapshot capture use serialized_torch().
Only native participant custody commits sampling state; this module keeps none.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from gf_common.domain import ContractError, digest
from gf_common.rust_player import BATCHED_TORCH_INFERENCE, RUST_BATCHED_TORCH_GRAPH_RECIPE
from gf_models.rust_player_batch import BatchEstimate, BatchGeometry, BatchInputs, BatchShape
from gf_models.structured_config import MAX_TENSOR_BYTES
from gf_models.torch_runtime import serialized_torch

if TYPE_CHECKING:
    from gf_models._rust_player_graph import CollectionCudaGraph

_CUDA_BATCH_TEMPORARY_TENSOR_BYTES = 128 * 1024**2


@cache
def sampling_runtime_identity() -> str:
    import torch

    from gf_models.torch_runtime import runtime_metadata

    metadata = runtime_metadata()
    if torch.__version__ != metadata["torch"]:
        raise ContractError("Torch module/distribution sampling runtime mismatch")
    return digest(
        {"format": "cpu-torch-generator-v1", "runtime": metadata, "module_cuda": torch.version.cuda}
    )


@contextmanager
def _preserve_global_rng() -> Iterator[None]:
    import torch

    # fork_rng always saves CPU. Only enumerate CUDA after its real host init;
    # an ordinary CPU registry never initializes unrelated CUDA devices.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices, device_type="cuda"):
        yield


def tensor_views(arrays: dict[str, np.ndarray], *, device: str) -> dict[str, Any]:
    """CPU aliases retain NumPy capsule owners; initial H2D is synchronous."""
    import torch

    return {
        name: torch.from_numpy(array).to(device, non_blocking=False)
        for name, array in arrays.items()
    }


def _participant_sampling_generator(*, state: bytes | None, seed: int, runtime: str) -> Any:
    import torch

    if runtime != sampling_runtime_identity():
        raise ContractError("native participant Torch sampling runtime mismatch")
    generator = torch.Generator(device="cpu")
    if state is None:
        generator.manual_seed(seed)
    else:
        # Torch owns state validation. Copy untrusted immutable bytes to a writable
        # own buffer; never alter the global generator on an invalid restore.
        try:
            generator.set_state(torch.from_numpy(np.frombuffer(state, dtype=np.uint8).copy()))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ContractError("invalid native participant Torch state") from exc
    return generator


def provisional_sample(
    probabilities: Any, *, state: bytes | None, seed: int, runtime: str
) -> tuple[int, bytes]:
    import torch

    generator = _participant_sampling_generator(state=state, seed=seed, runtime=runtime)
    values = probabilities.detach().to("cpu", dtype=torch.float32, non_blocking=False).reshape(-1)
    if (
        not bool(torch.isfinite(values).all())
        or bool((values < 0).any())
        or not bool(values.sum() > 0)
    ):
        raise ContractError("invalid own sampling probabilities")
    sample = int(torch.multinomial(values, 1, generator=generator).item())
    return sample, generator.get_state().numpy().tobytes()


def _sample_prepared_cpu(
    probabilities: Any, *, state: bytes | None, seed: int, runtime: str
) -> tuple[int, bytes]:
    """Sample one FP32 row from the grouped path's completed CPU transfer."""
    import torch

    generator = _participant_sampling_generator(state=state, seed=seed, runtime=runtime)
    # Torch validates finite, nonnegative weights with a nonzero sum.
    try:
        selected = torch.multinomial(probabilities, 1, generator=generator)
    except RuntimeError as exc:
        raise ContractError("invalid own sampling probabilities") from exc
    return int(selected.item()), generator.get_state().numpy().tobytes()


class RustPlayerInference:
    """One immutable shared_registry instance and explicit checkpoint references.

    Actor admissions live in the native dispatcher; frozen-policy holders call
    retain/release. Eviction checks both custody domains under the shared lock.
    No participant cache, async bridge, or second batch formation lives here.
    """

    def __init__(
        self,
        *,
        device: str,
        cache_entries: int,
        cache_weight_bytes: int,
        visibility: str | None = None,
        device_account: Any = None,
        inference_recipe: str | None = None,
    ) -> None:
        from gf_models.shared_registry import make_registry

        if device_account is not None and device_account.serving_device != device:
            raise ContractError('actual inference/account serving role differs')
        if inference_recipe not in (None, BATCHED_TORCH_INFERENCE, RUST_BATCHED_TORCH_GRAPH_RECIPE):
            raise ContractError("unknown Rust player inference recipe")
        if inference_recipe is not None and device != 'cpu' and device_account is None:
            raise ContractError("batched CUDA inference needs the actual device account")
        self.device, self.device_account = device, device_account
        self.inference_recipe = inference_recipe
        self._collection: dict[str, tuple[BatchGeometry, dict[BatchShape, BatchEstimate | None]]] = {}
        self._collection_graph: CollectionCudaGraph | None = None
        self._batch_counts: dict[int, int] | None = None
        self._scalar_rows = 0
        self._graph_replays = 0
        self._measurement_batch_width: int | None = None
        self._previous_cuda_peaks = (0, 0)
        with serialized_torch():
            hold = device_account.begin_backend(owner=self) if device_account is not None and device != 'cpu' else None
            try:
                with _preserve_global_rng():
                    self.registry = make_registry(device=device, cache_entries=cache_entries,
                        cache_weight_bytes=cache_weight_bytes, visibility=visibility)
                if hold is not None:
                    hold.complete()
            except BaseException as error:
                if hold is not None:
                    hold.fail(error)
                raise
        self.performance_callback: Any = None
        self.runtime = sampling_runtime_identity()
        self.references: dict[str, int] = {}
        self.timings = {
            name: 0.0
            for name in (
                "native_wait",
                "cpu_share",
                "h2d",
                "forward",
                "sampling",
                "return",
                "native_submit",
                "assembly",
            )
        }
        self.calls = {"next_batch": 0, "submit_results": 0, "failures": 0, "rows": 0,
            "accepted": 0, "expired": 0}

    def publish(self, network: Any, checkpoint: str) -> None:
        from gf_models.shared_registry import publication

        with serialized_torch():
            self._publish_payload(network.config, network.binding, publication(network, checkpoint), checkpoint)

    def publish_banks(self, model: Any, binding: Any, banks: Any, checkpoint: str) -> None:
        from gf_models.shared_registry import bank_publication

        with serialized_torch():
            self._publish_payload(model, binding, bank_publication(model, binding, banks, checkpoint), checkpoint)

    def _publish_payload(self, model: Any, binding: Any, payload: Any, checkpoint: str) -> None:
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.rust_player_device import CudaCheckpointProfile

        with serialized_torch():
            account = self.device_account if self.device != 'cpu' else None
            profile = (account.cached_profile(checkpoint) or CudaCheckpointProfile(model, binding, layout=layout)
                if account is not None else None)
            hold = account.begin_construction(checkpoint, profile, owner=self) if account is not None else None
            was_present = checkpoint in self.registry.networks
            try:
                with _preserve_global_rng():
                    self.registry.load_policy_weights(payload)
                if self.device != 'cpu':
                    import torch
                    torch.cuda.synchronize(self.device)
                if account is not None:
                    account.refresh_cache(checkpoint, profile, self.registry.networks[checkpoint])
                if hold is not None:
                    hold.complete()
                self.references[checkpoint] = 0
            except BaseException as error:
                if hold is not None:
                    self._failed_publication = checkpoint, was_present
                    hold.fail(error, references=payload)
                raise

    def retire_device_failure(self, *, dispatcher: Any = None) -> None:
        """Only after the original producer stopped; actual GPU fence precedes credit."""
        from tensordict import NonTensorData, TensorDict

        with serialized_torch():
            account = getattr(self, 'device_account', None)
            hold = account._active if account is not None else None
            if self.collection_graph_active and (dispatcher is None
                    or dispatcher.checkpoint_references() or any(self.references.values())):
                raise ContractError('actual collection graph consumers must retire before device cleanup')
            if account is None or hold is None or hold.owner is not self:
                if self.collection_graph_active:
                    if hold is not None:
                        raise ContractError('another actual operation still holds the device')
                    self.end_collection()
                return
            if hold.state != 'FAILED_HELD':
                raise ContractError('inference device operation has not stopped')
            if self.device != 'cpu':
                import torch
                torch.cuda.synchronize(self.device)
            provisional = getattr(self, '_failed_publication', None)
            if provisional is not None:
                checkpoint, was_present = provisional
                if not was_present and checkpoint in self.registry.networks:
                    if self.references.get(checkpoint, 0) or (dispatcher is not None
                            and dispatcher.checkpoint_references().get(checkpoint, 0)):
                        raise ContractError('failed provisional network still has actual consumers')
                    self.registry.load_policy_weights(TensorDict({'publication': NonTensorData(
                        {'command': 'EVICT', 'checkpoint': checkpoint})}, batch_size=[]))
                    if checkpoint in self.registry.networks:
                        raise ContractError('failed provisional network removal remains uncertain')
                    account.discard_failed_cache(checkpoint)
            hold.retire_failed(gpu_settled=True)
            self._failed_publication = None
            if self.collection_graph_active:
                self.end_collection()

    def retain(self, checkpoint: str) -> None:
        with serialized_torch():
            if checkpoint not in self.references:
                raise ContractError("unpublished native checkpoint reference")
            self.references[checkpoint] += 1

    def release(self, checkpoint: str) -> None:
        with serialized_torch():
            if self.references.get(checkpoint, 0) <= 0:
                raise ContractError("unbalanced native checkpoint release")
            self.references[checkpoint] -= 1

    def evict(self, checkpoint: str, *, dispatcher: Any) -> None:
        from tensordict import NonTensorData, TensorDict

        with serialized_torch():
            if self.references.get(checkpoint, 0) or dispatcher.checkpoint_references().get(
                checkpoint, 0
            ):
                raise ContractError("native checkpoint still referenced")
            if self.collection_graph_active:
                raise ContractError("collection CUDA Graph must retire before checkpoint eviction")
            self.registry.load_policy_weights(
                TensorDict(
                    {"publication": NonTensorData({"command": "EVICT", "checkpoint": checkpoint})},
                    batch_size=[],
                )
            )
            account = self.device_account if self.device != 'cpu' else None
            if account is not None:
                import torch
                torch.cuda.synchronize(self.device)
                account.evict_cache(checkpoint, gpu_settled=True)
            del self.references[checkpoint]

    def cache_state(self) -> dict[str, Any]:
        with serialized_torch():
            return {
                "references": dict(self.references),
                "entries": len(self.registry.networks),
                "weight_bytes": sum(
                    p.numel() * p.element_size()
                    for net in self.registry.networks.values()
                    for p in net.parameters()
                ),
            }

    @property
    def collection_graph_active(self) -> bool:
        return getattr(self, "_collection_graph", None) is not None

    def bind_collection(self, checkpoints: tuple[str, ...], *, game_concurrency: int,
                        max_batch_size: int, learner_checkpoint: str | None = None) -> None:
        """Freeze every scheduled checkpoint's shape decisions before actor admission."""
        from gf_models.rust_player_batch import estimate_batch

        if self.inference_recipe is None or self.device == "cpu":
            return
        with serialized_torch():
            graph_recipe = self.inference_recipe == RUST_BATCHED_TORCH_GRAPH_RECIPE
            if graph_recipe and getattr(self, "_collection_graph", None) is not None:
                raise ContractError("previous collection CUDA Graph remains live")
            account = self.device_account
            available = account.serving_tensor_allowance()
            plans = {}
            for checkpoint in checkpoints:
                profile = account.cached_profile(checkpoint)
                if profile is None:
                    raise ContractError("unpublished native collection checkpoint")
                model, binding = profile.model, profile.binding
                logical = min(model.resources.max_owned_bytes, binding.managed_budget_bytes)
                allowance = min(available, logical - profile.fixed_bytes - profile.network_bytes)
                tensor_limit = model.resources.max_tensor_bytes
                # Only CUDA batch temporaries receive the wider default cap.
                if tensor_limit == MAX_TENSOR_BYTES:
                    tensor_limit = _CUDA_BATCH_TEMPORARY_TENSOR_BYTES
                geometry = BatchGeometry(min(game_concurrency, max_batch_size),
                    binding.max_candidates, model.event.max_interval,
                    candidate_floor=64 if graph_recipe else 1, event_floor=16 if graph_recipe else 1)
                shapes = {}
                for shape in geometry.shapes():
                    estimate = estimate_batch(model, binding, layout=profile.layout, shape=shape,
                                              row_profile=profile.row)
                    shapes[shape] = (estimate if estimate.peak_bytes <= allowance and
                                     estimate.largest_tensor_bytes <= tensor_limit else None)
                plans[checkpoint] = geometry, shapes
            self._collection = plans
            if graph_recipe and learner_checkpoint is not None and learner_checkpoint in plans:
                self._prepare_collection_graph(learner_checkpoint, available)

    def _prepare_collection_graph(self, checkpoint: str, available: int) -> None:
        import torch

        from gf_models._rust_player_graph import (
            COLLECTION_GRAPH_SHAPE,
            CollectionCudaGraph,
            expected_collection_pool,
        )

        estimate = self._collection[checkpoint][1].get(COLLECTION_GRAPH_SHAPE)
        if estimate is None:
            return
        account = self.device_account
        profile = account.cached_profile(checkpoint)
        expected_pool = expected_collection_pool(profile, torch=torch, device=self.device)
        if expected_pool is None:
            return
        model_limit = (min(profile.model.resources.max_owned_bytes, profile.binding.managed_budget_bytes)
                       - profile.fixed_bytes - profile.network_bytes)
        graph = CollectionCudaGraph(checkpoint, estimate, self.device)
        host_peak = checkpoint_peak = graph.replay_bytes
        for candidate, (_, shapes) in self._collection.items():
            candidate_profile = account.cached_profile(candidate)
            for shape, eager in shapes.items():
                if eager is None:
                    # Native row peaks grow with the real C/E lengths within
                    # each bucket. Only unadmitted batch buckets can use rows.
                    schema = BatchInputs.compile(BatchShape(1, shape.candidates, shape.events),
                        positions=candidate_profile.model.max_positions, layout=candidate_profile.layout)
                    # The native row omits the batched candidate mask and last-event index.
                    leaf_bytes = sum(schema.bank_bytes) - shape.candidates - 8
                    peak = candidate_profile.row.estimate(interval_events=shape.events,
                        candidates=shape.candidates, input_leaf_bytes=leaf_bytes).peak_bytes
                    host_peak = max(host_peak, peak)
                    if candidate == checkpoint:
                        checkpoint_peak = max(checkpoint_peak, peak)
                elif candidate != checkpoint or shape != COLLECTION_GRAPH_SHAPE:
                    host_peak = max(host_peak, eager.peak_bytes)
                    if candidate == checkpoint:
                        checkpoint_peak = max(checkpoint_peak, eager.peak_bytes)

        ceiling = min(available - host_peak, model_limit - checkpoint_peak)
        expected = estimate.input_bytes + expected_pool
        if expected > ceiling:
            return
        preflight = max(estimate.peak_bytes, expected + graph.replay_bytes)
        hold = account.begin_batch(preflight, owner=self)
        self._collection_graph = graph
        hold.references = graph
        try:
            measured = graph.capture(torch, self.registry.networks[checkpoint])
            if measured <= ceiling:
                account.retain_collection_graph(hold, measured)
            else:
                self._drop_collection_graph(graph, torch)
            hold.references = None
            hold.complete()
        except BaseException as error:
            hold.fail(error, references=graph)
            raise

    def _drop_collection_graph(self, graph: CollectionCudaGraph, torch: Any) -> None:
        graph.retire(torch)
        self.device_account.clear_collection_graph()
        self._collection_graph = None

    def end_collection(self) -> None:
        with serialized_torch():
            if self.device != "cpu":
                import torch

                graph = self._collection_graph
                account = self.device_account
                if graph is not None and (account._active is not None or any(self.references.values())):
                    raise ContractError("actual collection graph consumers must retire before device cleanup")
                # A terminal retry on a poisoned account has already retired
                # the failed operation's stopped traceback and temporary aliases.
                hold = (account.begin_graph_retirement(owner=self)
                        if graph is not None and not account._poisoned else None)
                if hold is not None:
                    hold.references = graph
                try:
                    torch.cuda.synchronize(self.device)
                    if graph is not None:
                        self._drop_collection_graph(graph, torch)
                    if hold is not None:
                        hold.references = None
                        hold.complete()
                except BaseException as error:
                    if hold is not None:
                        hold.fail(error, references=graph)
                    raise
            self._collection.clear()

    def begin_collection_metrics(self) -> None:
        with serialized_torch():
            if self.device != "cpu":
                import torch

                self._previous_cuda_peaks = self.process_cuda_peaks()
                torch.cuda.reset_peak_memory_stats(self.device)
            first = next(iter(self._collection.values()), None)
            self._measurement_batch_width = first[0].rows if first is not None else None
            self._scalar_rows = 0
            self._graph_replays = 0
            self._batch_counts = {}

    def end_collection_metrics(self) -> dict[str, Any] | None:
        with serialized_torch():
            counts, self._batch_counts = self._batch_counts, None
            if counts is None:
                return None
            groups = sum(counts.values())
            useful = sum(size * count for size, count in counts.items())
            width = self._measurement_batch_width
            allocated = reserved = None
            if self.device != "cpu":
                import torch

                allocated = torch.cuda.max_memory_allocated(self.device)
                reserved = torch.cuda.max_memory_reserved(self.device)
            recipe = self.inference_recipe if width is not None else "scalar-torch-v1"
            totals = {"recipe": recipe,
                    "batch_width": width, "batched_groups": groups, "useful_rows": useful,
                    "dummy_rows": (width or 0) * groups - useful, "scalar_rows": self._scalar_rows,
                    "useful_size_counts": {str(size): count for size, count in counts.items()},
                    "cuda_peak_allocated_bytes": allocated, "cuda_peak_reserved_bytes": reserved}
            if recipe == RUST_BATCHED_TORCH_GRAPH_RECIPE:
                totals["graph_replays"] = self._graph_replays
            return totals

    def process_cuda_peaks(self) -> tuple[int, int]:
        """Preserve the cumulative series across opt-in collection peak resets."""
        import torch

        with serialized_torch():
            allocated, reserved = self._previous_cuda_peaks
            return (max(allocated, torch.cuda.max_memory_allocated(self.device)),
                    max(reserved, torch.cuda.max_memory_reserved(self.device)))

    def evaluate(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Provisional results only. Caller must pass them to native acceptance."""
        import torch

        if batch["runtime"] != self.runtime:
            raise ContractError("native inference batch runtime mismatch")
        with serialized_torch(), torch.no_grad():
            groups: dict[tuple[str, BatchShape], list[tuple[int, dict[str, Any]]]] = {}
            answers = []
            collection = getattr(self, "_collection", {})
            for index, row in enumerate(batch["rows"]):
                self.timings["assembly"] += row["assembly_seconds"]
                checkpoint = json.loads(row["identity"])["checkpoint"]
                if checkpoint not in self.registry.networks:
                    raise ContractError("unpublished native checkpoint")
                if not collection:
                    answers.append(self._evaluate_row(row, checkpoint))
                    continue
                geometry, _ = collection[checkpoint]
                operands = row["inputs"]
                shape = geometry.shape(len(operands["kinds"]), len(operands["interval_numerics"]))
                groups.setdefault((checkpoint, shape), []).append((index, row))
                answers.append({})
            for (checkpoint, shape), indexed in groups.items():
                _, shapes = collection[checkpoint]
                estimate = shapes[shape]
                if estimate is None:
                    for index, row in indexed:
                        answers[index] = self._evaluate_row(row, checkpoint)
                else:
                    scored = self._evaluate_group([row for _, row in indexed], checkpoint, estimate)
                    for (index, _), answer in zip(indexed, scored, strict=True):
                        answers[index] = answer
            return answers

    def _evaluate_row(self, row: dict[str, Any], checkpoint: str) -> dict[str, Any]:
        import torch

        from gf_models.shared_numeric import score_numeric_row

        account = getattr(self, 'device_account', None) if self.device != 'cpu' else None
        hold = (account.begin_row(checkpoint,
            interval_events=row['inputs']['interval_numerics'].shape[0],
            candidates=row['inputs']['kinds'].shape[0],
            input_leaf_bytes=sum(value.nbytes for value in row['inputs'].values()), owner=self)
            if account is not None else None)
        inputs: dict[str, Any] = {}
        if hold is not None:
            hold.references = inputs  # Retain BEFORE a partial H2D can fail.
        try:
            at = time.perf_counter()
            try:
                cpu = tensor_views(row["inputs"], device="cpu")
            finally:
                self.timings["cpu_share"] += time.perf_counter() - at
            at = time.perf_counter()
            try:
                for name, cpu_value in cpu.items():
                    inputs[name] = cpu_value.to(self.device, non_blocking=False)
            finally:
                self.timings["h2d"] += time.perf_counter() - at
            at = time.perf_counter()
            try:
                if self.performance_callback is None:
                    logs, value = score_numeric_row(self.registry.networks[checkpoint], inputs)
                else:
                    with torch.profiler.record_function("gf-native-row-forward"):
                        logs, value = score_numeric_row(self.registry.networks[checkpoint], inputs)
                if self.device != "cpu":
                    torch.cuda.synchronize(self.device)
            finally:
                self.timings["forward"] += time.perf_counter() - at
            if self.performance_callback is not None:
                self.performance_callback.after_forward(scored_rows=1, boundary="native-post-row-forward")
            at = time.perf_counter()
            try:
                sample, next_state = provisional_sample(
                    logs.exp(),
                    state=row["rng_state"],
                    seed=row["sampling_seed"],
                    runtime=self.runtime,
                )
            finally:
                self.timings["sampling"] += time.perf_counter() - at
            at = time.perf_counter()
            try:
                logs = logs.detach().to("cpu", non_blocking=False)
                value = value.detach().to("cpu", non_blocking=False)
                answer = {
                    "identity": row["identity"],
                    "logs": logs.reshape(-1).numpy(),
                    "value": float(value.item()),
                    "sample": sample,
                    "next_state": next_state,
                    "runtime": self.runtime,
                }
                self.calls["rows"] += 1
                if getattr(self, "_batch_counts", None) is not None:
                    self._scalar_rows += 1
            finally:
                self.timings["return"] += time.perf_counter() - at
            # Forward synchronization and blocking CPU result transfers
            # have completed. Last GPU input references retire INSIDE
            # this same Torch lock, before another row/update can start.
            inputs.clear()
            if hold is not None:
                hold.references = None
                hold.complete()
        except BaseException as error:
            if hold is not None:
                hold.fail(error, references=inputs)
            raise
        return answer

    def _evaluate_group(self, rows: list[dict[str, Any]], checkpoint: str,
                        estimate: BatchEstimate) -> list[dict[str, Any]]:
        import torch

        from gf_models.rust_player_batch import pack_native_rows, score_numeric_batch

        at = time.perf_counter()
        packed = pack_native_rows(rows, estimate.inputs)
        self.timings["cpu_share"] += time.perf_counter() - at
        graph = self._collection_graph
        if graph is not None and (graph.checkpoint != checkpoint or graph.estimate is not estimate):
            graph = None
        hold = self.device_account.begin_batch(estimate.peak_bytes if graph is None else graph.replay_bytes, owner=self)
        roots: dict[np.dtype[Any], Any] = {}
        hold.references = roots if graph is None else graph
        try:
            at = time.perf_counter()
            try:
                if graph is None:
                    inputs = packed.transfer(device=self.device, roots=roots)
                else:
                    graph.transfer(torch, packed)
            finally:
                self.timings["h2d"] += time.perf_counter() - at
            at = time.perf_counter()
            try:
                if graph is not None:
                    logs, probabilities, values = graph.replay()
                elif self.performance_callback is None:
                    logs, probabilities, values = score_numeric_batch(self.registry.networks[checkpoint], inputs)
                else:
                    with torch.profiler.record_function("gf-native-batch-forward"):
                        logs, probabilities, values = score_numeric_batch(self.registry.networks[checkpoint], inputs)
                # One blocking D2H also fences this serialized forward.
                size = logs.shape[1]
                combined = torch.cat((logs, probabilities, values[:, None]), dim=1)
                output = combined.to("cpu", non_blocking=False)
                del logs, probabilities, values, combined
                if graph is None:
                    del inputs
            finally:
                self.timings["forward"] += time.perf_counter() - at
            if self.performance_callback is not None:
                self.performance_callback.after_forward(scored_rows=len(rows), boundary="native-post-row-forward")
            roots.clear()
            hold.references = None
            hold.complete()
        except BaseException as error:
            hold.fail(error, references=roots if graph is None else graph)
            raise

        answers = []
        returned = output.numpy()
        for index, row in enumerate(rows):
            n = len(row["inputs"]["kinds"])
            logs_cpu, value = returned[index, :n], float(returned[index, -1])
            if not np.isfinite(logs_cpu).all() or not np.isfinite(value):
                raise ContractError("nonfinite shared MLP output")
            at = time.perf_counter()
            try:
                sample, next_state = _sample_prepared_cpu(
                    output[index, size:size + n],
                    state=row["rng_state"],
                    seed=row["sampling_seed"],
                    runtime=self.runtime,
                )
            finally:
                self.timings["sampling"] += time.perf_counter() - at
            at = time.perf_counter()
            answers.append({"identity": row["identity"], "logs": logs_cpu,
                            "value": value, "sample": sample, "next_state": next_state,
                            "runtime": self.runtime})
            self.timings["return"] += time.perf_counter() - at
        self.calls["rows"] += len(rows)
        if self._batch_counts is not None:
            self._batch_counts[len(rows)] = self._batch_counts.get(len(rows), 0) + 1
            if graph is not None:
                self._graph_replays += 1
        return answers

    def serve_next(self, dispatcher: Any, *, deadline: float) -> int:
        """Consume the sole Rust formation window, score, then atomically submit."""
        try:
            at = time.perf_counter()
            try:
                self.calls["next_batch"] += 1
                batch = dispatcher.next_batch(deadline)
            finally:
                self.timings["native_wait"] += time.perf_counter() - at
            answers = self.evaluate(batch)
            at = time.perf_counter()
            try:
                self.calls["submit_results"] += 1
                return int(dispatcher.submit_results(batch["batch_id"], answers))
            finally:
                self.timings["native_submit"] += time.perf_counter() - at
        except BaseException:
            self.calls["failures"] += 1
            dispatcher.cancel("native inference service failed")
            raise
