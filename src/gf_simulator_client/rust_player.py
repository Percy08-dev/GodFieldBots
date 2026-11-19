"""Owned Rust game runtime. Python owns processes, immutable inference and writer custody."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import gf_native

from gf_common.domain import ContractError, canonical


class RustPlayerRuntime:
    """Explicit online owner; storage acceptance is never a persistence receipt.

    The writer must keep each StorageBundle (or its read-only view) through its
    actual last use. storage_ack transfers that same reservation; it only lets
    the actor progress. No writer or archive-success substitute is installed.
    """

    def __init__(self, config: dict[str, Any], *, service: Any = None) -> None:
        self._native = gf_native.PlayerRuntime(canonical(config).encode())
        self.dispatcher = self._native.dispatcher
        self.service = service
        self._holds: dict[tuple[str, str, str], list[str]] = {}
        self._active_routes: dict[tuple[str, str, str], int] = {}
        self._settled = threading.Condition()
        self._service_thread: threading.Thread | None = None
        self._service_error: str | None = None
        self._closing = threading.Event()
        self._joined = False
        self._joining = False
        self._quiesced = False
        self._report: dict[str, Any] | None = None
        self._join_error: str | None = None
        self._lock = threading.RLock()
        self._join_done = threading.Condition(self._lock)
        if service is not None:
            from gf_models.rust_player import RustPlayerInference

            if (
                type(service) is not RustPlayerInference
                or service.runtime != config["runtime_identity"]
            ):
                raise ContractError("runtime requires same immutable inference service identity")

    def _serve(self) -> None:
        in_batch = False
        try:
            while True:
                at = time.perf_counter()
                try:
                    self.service.calls["next_batch"] += 1
                    batch = self.dispatcher.next_batch(10_000_000_000.0)
                finally:
                    self.service.timings["native_wait"] += time.perf_counter() - at
                in_batch = True
                routes = []
                for row in batch["rows"]:
                    identity = json.loads(row["identity"])
                    routes.append((identity["run"], identity["trial"], identity["game"]))
                del row
                with self._settled:
                    for route in routes:
                        self._active_routes[route] = self._active_routes.get(route, 0) + 1
                answers = None
                try:
                    answers = self.service.evaluate(batch)
                    at = time.perf_counter()
                    try:
                        self.service.calls["submit_results"] += 1
                        self.dispatcher.submit_results(batch["batch_id"], answers)
                    finally:
                        self.service.timings["native_submit"] += time.perf_counter() - at
                except BaseException as exc:
                    # A failing evaluate/submit traceback can still own Tensors
                    # after these local references are deleted. Publish failure
                    # under the SAME settlement lock before any route reaches
                    # zero; those holds can only release after original join.
                    with self._settled:
                        self._service_error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    # Success has synchronous CPU/device return and drops all
                    # original local owners here. Failure is already marked and
                    # route retirement never claims physical Tensor settlement.
                    del batch, answers
                    with self._settled:
                        for route in routes:
                            count = self._active_routes[route] - 1
                            if count:
                                self._active_routes[route] = count
                            else:
                                del self._active_routes[route]
                        self._settled.notify_all()
                in_batch = False
        except BaseException as exc:  # noqa: BLE001 - every service exit must wake original actors.
            if in_batch or not self._closing.is_set():
                # Retaining exception tracebacks could itself retain private
                # arrays. Only the management owner receives this text.
                with self._settled:
                    if self._service_error is None:
                        self._service_error = f"{type(exc).__name__}: {exc}"
                self.service.calls["failures"] += 1
                self.dispatcher.cancel("native inference service failed")
                self._native.cancel("inference service failed")

    def start(self, games: list[dict[str, Any]]) -> None:
        """Admit a precomputed wave; idle original runtimes remain alive between waves."""
        with self._lock:
            if self._joined or self._joining or self._quiesced:
                raise ContractError("native runtime admission closed")
            games = [
                {**game, "policies": [dict(policy) for policy in game["policies"]]}
                for game in games
            ]
            for game in games:
                for policy in game["policies"]:
                    if policy["kind"] == "mlp":
                        if self.service is None:
                            raise ContractError(
                                "MLP descriptor requires actual same-service retain"
                            )
                        # This actual obligation precedes descriptor admission.
                        # Keep even on uncertain start failure until original join.
                        checkpoint = policy["checkpoint"]
                        self.service.retain(checkpoint)
                        self._holds.setdefault(
                            (game["run"], game["trial"], game["episode"]), []
                        ).append(checkpoint)
                        from gf_models.rust_player import serialized_torch

                        with serialized_torch():
                            network = self.service.registry.networks[checkpoint]
                            model, binding = network.config, network.binding
                            if model.model_id != "mlp":
                                raise ContractError("native player supports configured MLP only")
                            limits = {
                                "max_decisions": model.temporal.max_decisions,
                                "max_episode_events": model.event.max_episode,
                                "max_interval_events": model.event.max_interval,
                                "max_candidates": binding.max_candidates,
                                "max_positions": model.max_positions,
                                "aggregate_window": model.event.aggregate_window,
                            }
                            if "limits" in policy and policy["limits"] != limits:
                                raise ContractError(
                                    "descriptor differs from actual model/binding limits"
                                )
                            policy["limits"] = limits
            # Existing live-process owner produces the exact schema2 receipt;
            # representation validation is separate from native policy input.
            from gf_common.card_instance_runtime import validate_runtime_receipt

            for game in games:
                # Native baseline construction errors still precede HTTP admission.
                if "runtime_evidence" in game:
                    validate_runtime_receipt(game["runtime_evidence"], require_executed=True)
                    from gf_simulator_client.http import EFFECTIVE_KEYS, validate_info

                    server = game["server"]
                    info = validate_info(server["startup_info"])
                    if {key: info[key] for key in EFFECTIVE_KEYS} != server[
                        "expected_effective"
                    ] or info["seed_fingerprint"] != server["expected_fingerprint"]:
                        raise ContractError("managed startup/effective config mismatch")
            if self.service is not None and self._service_thread is None:
                self._service_thread = threading.Thread(
                    target=self._serve, name="gf-player-inference"
                )
                self._service_thread.start()
            self._native.start(canonical(games).encode())

    def next_storage(self, deadline: float) -> Any:
        return self._native.next_storage(deadline)

    def storage_ack(self, bundle: Any, *, failure: str | None = None) -> None:
        with self._lock:
            self._native.storage_ack(bundle, failure)
            if failure is not None:
                self.cancel(f"writer failed: {failure}")
                return
            key = bundle.release_key
            if key is not None and self.service is not None:
                route = (key[0], key[1], bundle.game)
                with self._settled:
                    while self._active_routes.get(route, 0):
                        self._settled.wait()
                if self._service_error is None:
                    for checkpoint in self._holds.pop(key, []):
                        self.service.release(checkpoint)

    def recording_account(self) -> Any:
        """Return the SAME native recording account, without another pool/charge."""
        return self._native.recording_account()

    def storage_bytes(self) -> int:
        return int(self._native.storage_bytes())

    def wait(self, deadline: float) -> None:
        """Wait for the current submitted wave without ending idle runtimes."""
        self._native.wait(deadline)

    def quiesce(self) -> None:
        """Stop new admissions while existing actors and original I/O tasks finish."""
        with self._lock:
            self._quiesced = True
            self._native.quiesce()

    def cancel(self, reason: str) -> None:
        self._native.cancel(reason)

    def join(self) -> dict[str, Any]:
        """Join original actors and service before releasing actual checkpoint holds."""
        with self._join_done:
            while self._joining:
                self._join_done.wait()
            if self._join_error is not None:
                raise RuntimeError(self._join_error)
            if self._report is not None:
                return self._report
            self._joining = True
            self._closing.set()
        # Writer acknowledgements must progress while original actors join.
        try:
            try:
                report = json.loads(self._native.join())
            finally:
                if self._service_thread is not None:
                    self._service_thread.join()
            with self._lock:
                for checkpoints in self._holds.values():
                    for checkpoint in checkpoints:
                        self.service.release(checkpoint)
                self._holds.clear()
            report["python_service_threads"] = int(self._service_thread is not None)
            if self.service is not None:
                import torch

                report["torch_threads"] = {
                    "intraop": torch.get_num_threads(),
                    "interop": torch.get_num_interop_threads(),
                }
            else:
                report["torch_threads"] = None
            report["torch_thread_counts_kind"] = (
                "configured Torch API counts; OS ownership not measured"
            )
            report["io_executor_threads"] = report["player_threads"]
            report["batch_formation_owner"] = (
                "same Python service thread" if self.service is not None else None
            )
            if self._service_error is not None:
                raise RuntimeError(f"native inference service failed: {self._service_error}")
            with self._lock:
                self._report = report
            return report
        except BaseException as exc:
            with self._lock:
                self._join_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self._join_done:
                self._joined = True
                self._joining = False
                self._join_done.notify_all()
