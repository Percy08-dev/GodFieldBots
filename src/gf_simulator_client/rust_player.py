"""Owned Rust game runtime. Python owns processes, immutable inference and writer custody."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import gf_native

from gf_common.domain import ContractError, canonical
from gf_common.rust_player import LEARNED_POLICY_KINDS, SEQUENCE_MODEL_KINDS


class RustPlayerRuntime:
    """Explicit online owner; storage acceptance is never a persistence receipt.

    The writer must keep each StorageBundle (or its read-only view) through its
    actual last use. storage_ack transfers that same reservation; it only lets
    the actor progress. No writer or archive-success substitute is installed.
    """

    def __init__(self, config: dict[str, Any], *, service: Any = None) -> None:
        from gf_simulator_client.performance_diagnostics import host_diagnostics
        self.diagnostics = host_diagnostics()
        self._measurement_scope: dict[str, Any] | None = None
        self._measurement_stages: dict[str, float] | None = None
        self._performance_wait_started: float | None = None
        self._profile_status: Any = None
        self._profile_reported = False
        if service is not None:
            from gf_models.rust_player import RustPlayerInference

            if (
                type(service) is not RustPlayerInference
                or service.runtime != config["runtime_identity"]
            ):
                raise ContractError("runtime requires same immutable inference service identity")

        self._sequence = service is not None and service.sequence_enabled
        if self._sequence:
            config = config | {"sequence_inference": True}
        elif config.get("sequence_inference"):
            raise ContractError("sequence dispatcher requires its actual sequence service")
        self._native = gf_native.PlayerRuntime(canonical(config).encode())
        self.dispatcher = self._native.dispatcher
        self.service = service
        self._holds: dict[tuple[str, str, str], list[str]] = {}
        self._active_routes: dict[tuple[str, str, str], int] = {}
        self._game_admission_attempts: set[tuple[str, str, str]] = set()
        self._settled = threading.Condition()
        self._service_thread: threading.Thread | None = None
        self._service_error: str | None = None
        self._closing = threading.Event()
        self._native_joined = False
        self._joined = False
        self._joining = False
        self._quiesced = False
        self._report: dict[str, Any] | None = None
        self._join_error: str | None = None
        self._lock = threading.RLock()
        self._join_done = threading.Condition(self._lock)
        if self.diagnostics is not None:
            self.measurement_begin(self.diagnostics.current_scope or self.diagnostics.scope(phase='startup'))

    def _stage_snapshot(self) -> dict[str, float]:
        if self.service is None:
            return {}
        stages = dict(self.service.timings)
        if self._performance_wait_started is not None:
            stages['native_wait'] += time.perf_counter() - self._performance_wait_started
        return stages

    def measurement_begin(self, scope: dict[str, Any]) -> None:
        if self.diagnostics is None:
            return
        if self._measurement_scope is not None:
            self.measurement_end()
        try:
            with self._settled:
                if self._active_routes or self._holds:
                    raise ValueError('measurement overlaps unsettled actual inference')
                self._native.measurement_begin(canonical(scope).encode())
                self._measurement_scope = dict(scope)
                self._measurement_stages = self._stage_snapshot() if self.service is not None else None
                if self.service is not None and scope['phase'] == 'collection':
                    self.service.begin_collection_metrics()
                if self.diagnostics.profile_recipe == 'forward-window-v1' and self._profile_status is None and self.service is not None:
                    from gf_simulator_client.performance_profile import (
                        ForwardProfile,
                        ProfileStatus,
                    )
                    self._profile_status = ProfileStatus.local()
                    self.service.performance_callback = ForwardProfile(self.diagnostics.directory,
                        self.diagnostics.observation_id, self._profile_status, role='native-inference')
                if self._profile_status is not None and scope['phase'] == 'collection':
                    self._profile_status.begin(scope)
        except Exception as error:  # noqa: BLE001 - diagnostic failure is never policy/Source authority.
            self.diagnostics.error('native_measurement_begin:' + type(error).__name__)

    def measurement_end(self, *, complete: bool = True) -> None:
        if self.diagnostics is None or self._measurement_scope is None:
            return
        try:
            # Logical END is independent of fallible native snapshots/decoding/publication.
            if self._profile_status is not None and self._measurement_scope['phase'] == 'collection':
                self._profile_status.end()
            with self._settled:
                if complete and (self._active_routes or self._holds):
                    raise ValueError('measurement overlaps unsettled actual inference')
                inference = (self.service.end_collection_metrics()
                    if self.service is not None and self._measurement_scope['phase'] == 'collection' else None)
                data = json.loads(self._native.measurement_end(canonical(self._measurement_scope).encode(), complete))
                self.diagnostics.metrics(data, role='native-runtime')
                if self._measurement_stages is not None:
                    from gf_common.performance_metrics import PhaseMetrics
                    stages = self._stage_snapshot()
                    stage_metrics = PhaseMetrics(scope=self._measurement_scope,
                        start_ns=data['start_monotonic_ns'], counters=(), histograms=())
                    for name, start in self._measurement_stages.items():
                        seconds = stages[name] - start
                        if seconds < 0:
                            stage_metrics.error('invalid_stage_interval:' + name)
                        else:
                            stage_metrics.stages[name] = {'method': 'synchronized-host-wall-v1', 'seconds': seconds}
                    metrics = stage_metrics.snapshot(end_ns=data['end_monotonic_ns'], complete=complete)
                    if inference is not None:
                        metrics['native_inference'] = inference
                    self.diagnostics.metrics(metrics, role='native-inference')
                if (self._profile_status is not None and self._measurement_scope['phase'] == 'collection'
                        and not self._profile_reported):
                    import torch

                    from gf_simulator_client.performance_profile import missing_profile_result
                    result = self._profile_status.read()['result']
                    if result is None:
                        result = missing_profile_result(observation_id=self.diagnostics.observation_id,
                            owner=self.diagnostics.owner | {'role': 'native-inference'}, clock=self.diagnostics.clock,
                            scope=self._measurement_scope, boundary='native-post-row-forward', torch_version=torch.__version__)
                    self.diagnostics.write(result)
                    self._profile_reported = True
        except Exception as error:  # noqa: BLE001 - diagnostic failure is never policy/Source authority.
            self.diagnostics.error('native_measurement_end:' + type(error).__name__)
        finally:
            self._measurement_scope = self._measurement_stages = None

    def _serve(self) -> None:
        in_batch = False
        try:
            while True:
                at = time.perf_counter()
                if self.diagnostics is not None:
                    with self._settled:
                        self._performance_wait_started = at
                try:
                    self.service.calls["next_batch"] += 1
                    batch = (self.dispatcher.next_work(10_000_000_000.0) if self._sequence
                             else self.dispatcher.next_batch(10_000_000_000.0))
                finally:
                    if self.diagnostics is None:
                        self.service.timings["native_wait"] += time.perf_counter() - at
                    else:
                        with self._settled:
                            self.service.timings["native_wait"] += time.perf_counter() - at
                            self._performance_wait_started = None
                in_batch = True
                if self._sequence and "retirement" in batch:
                    ticket, key = batch["retirement"]
                    self.service.retire_sequence_game(key)
                    self.dispatcher.ack_retirement(ticket)
                    in_batch = False
                    continue
                routes = []
                for row in batch["rows"]:
                    identity = json.loads(row["identity"])
                    routes.append((identity["run"], identity["trial"], identity["game"]))
                del row
                with self._settled:
                    for route in routes:
                        self._active_routes[route] = self._active_routes.get(route, 0) + 1
                answers = pending = None
                try:
                    if self._sequence:
                        answers, pending = self.service.evaluate_sequence(batch)
                    else:
                        answers = self.service.evaluate(batch)
                    at = time.perf_counter()
                    try:
                        self.service.calls["submit_results"] += 1
                        if self._sequence:
                            accepted = self.dispatcher.submit_rows(batch["batch_id"], answers)
                            self.service.accept_sequence(pending, accepted)
                            delivered = sum(accepted)
                        else:
                            delivered = self.dispatcher.submit_results(batch["batch_id"], answers)
                        self.service.calls["accepted"] += delivered
                        self.service.calls["expired"] += len(answers) - delivered
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
                    del batch, answers, pending
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
                if self._sequence:
                    self._native.cancel_inference("inference service failed")
                else:
                    self.dispatcher.cancel("native inference service failed")
                    self._native.cancel("inference service failed")

    def start(self, games: list[dict[str, Any]]) -> None:
        """Admit a precomputed wave under its original absolute deadlines.

        Each game carries cleanup_deadline bounded by the supervised cooperative
        cutoff. HTTP cleanup and terminal storage share that SAME deadline.
        Optional per-policy numeric_costs overrides the GameSpec default for
        that acting participant only; production supplies its own model recipe.
        Idle original runtimes remain alive between waves.
        """
        with self._lock:
            if self._joined or self._joining or self._quiesced:
                raise ContractError("native runtime admission closed")
            games = [
                {**game, "policies": [dict(policy) for policy in game["policies"]]}
                for game in games
            ]
            for game in games:
                for policy in game["policies"]:
                    if policy["kind"] in LEARNED_POLICY_KINDS:
                        if self.service is None:
                            raise ContractError(
                                "learned descriptor requires actual same-service retain"
                            )
                        from gf_models.rust_player import serialized_torch

                        checkpoint = policy["checkpoint"]
                        with serialized_torch():
                            if checkpoint not in self.service.registry.networks:
                                raise ContractError("unpublished native checkpoint reference")
                            network = self.service.registry.networks[checkpoint]
                            model, binding = network.config, network.binding
                            sequence_model = model.model_id in SEQUENCE_MODEL_KINDS
                            if model.model_id != policy["kind"] or sequence_model and not self._sequence:
                                raise ContractError("native descriptor kind differs from published model")
                            if (sequence_model and self.service.device != "cpu"
                                    and checkpoint not in self.service._sequence_groups):
                                raise ContractError("native sequence descriptor lacks a funded collection envelope")
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
                            # Preserve the obligation through uncertain native admission.
                            self.service.retain(checkpoint)
                            self._holds.setdefault(
                                (game["run"], game["trial"], game["episode"]), []
                            ).append(checkpoint)
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
            self._game_admission_attempts.update(
                (game["run"], game["trial"], game["episode"]) for game in games)
            self._native.start(canonical(games).encode())

    def episode_admission_attempted(self, episode: str) -> bool:
        """Compatibility-only text observation; Source uses full context keys."""
        with self._lock:
            return any(key[2] == episode for key in self._game_admission_attempts)

    def game_admission_attempted(self, run: str, trial: str, episode: str) -> bool:
        with self._lock:
            return (run, trial, episode) in self._game_admission_attempts

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

    def completed_reports(self, keys: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        """Actual full-key settled observations; no worker shutdown or Source authority."""
        return json.loads(self._native.completed_reports(keys))

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
                self._native_joined = True
            finally:
                if self._service_thread is not None:
                    self._service_thread.join()
            if self._sequence:
                # Original producers and exported rows have settled. A failed
                # fence must leave the admission charges and checkpoint holds.
                self.service.settle_joined_sequence_service(dispatcher=self.dispatcher)
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
            self.measurement_end(complete=self._join_error is None)
            with self._join_done:
                self._joined = True
                self._joining = False
                self._join_done.notify_all()
