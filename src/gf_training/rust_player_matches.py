"""Concrete managed Rust match execution; no registry or publication authority."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gf_common.domain import ContractError, EpisodeGroup, Json, clone, digest
from gf_common.hpo_progress import ProgressCallback, observe
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.http_diagnostics import HttpFacts, HttpPairCapture
from gf_simulator_client.process import ManagedServer
from gf_simulator_client.rust_player import RustPlayerRuntime
from gf_training.group_pack_resources import remaining
from gf_training.rust_player_storage import EpisodeEncodingContext, RustPlayerStorage

if TYPE_CHECKING:
    from gf_common.hpo_storage import HPODiagnosticsConfig


@dataclass(frozen=True)
class RustMatchResult:
    """Existing execution facts by reference; not capture/Source/role authority."""
    accepted: list[Json]
    bound_contexts: dict[int, str]
    completed_groups: dict[int, Json]
    receipts: dict[int, Json]
    keys: list[tuple[str, str, str]]
    reports: list[Json]
    total: int


class RustMatchPump:
    """Retain original server handles/history beside one borrowed runtime."""

    def __init__(self, runtime: RustPlayerRuntime) -> None:
        self.runtime = runtime
        self.servers: dict[int, ManagedServer] = {}
        self.server_history: list[Json] = []
        self._pending_http: dict[ManagedServer, tuple[HttpFacts, Json]] = {}

    async def settle_http(self, server: ManagedServer, *, deadline: float, abnormal: bool) -> Json | None:
        pair = self._pending_http.get(server)
        if pair is None:
            return None
        facts, history = pair
        info = await server.settle_http(deadline=deadline, abnormal=abnormal, **vars(facts))
        history["http_diagnostics"] = info
        if info["status"] not in ("stop_unconfirmed", "settlement_unconfirmed"):
            del self._pending_http[server]
        return info

    async def execute(self, groups: tuple[EpisodeGroup, ...], contexts: dict[str, Json],
                      writer: RustPlayerStorage, *, server: Any, actions: Any,
                      max_decisions: int, output: Path, execution_deadline: float,
                      cleanup_deadline: float, game_concurrency: int, run_id: str,
                      batch_id: str, evidence: CardInstanceRuntimeEvidence,
                      http_diagnostics: HPODiagnosticsConfig | None = None,
                      operation_id: str | None = None,
                      progress: ProgressCallback | None = None,
                      player_worker_threads: int = 1) -> RustMatchResult:
        """Caller already preflighted; errors return through its original cleanup."""
        accepted: list[Json] = []
        bound_contexts: dict[int, str] = {}
        completed_groups: dict[int, Json] = {}
        current_specs: dict[str, Json] = {}
        positions: dict[int, int] = {}
        receipts: dict[int, Json] = {}
        startups: dict[int, Json] = {}
        terminal_http: dict[str, tuple[int, int, int, bool]] = {}
        abnormal_pairs: set[int] = set()
        total = sum(len(group.episodes) for group in groups)
        full_wave = 0 < total <= game_concurrency
        native_journal = writer._rollout is not None
        if native_journal and not full_wave:
            raise ContractError("native Journal reader requires prebound full cohort")
        parallel_start = native_journal and player_worker_threads > 1
        next_group = 0
        drain_error: BaseException | None = None

        def cancel_owner(cancel: Callable[[str], None], error: BaseException) -> None:
            try:
                cancel('match journal drain failed')
            except BaseException as cleanup:  # noqa: BLE001 - preserve the first failure.
                error.add_note('Journal cancellation failed: ' + type(cleanup).__name__)

        def check_drain() -> None:
            if drain_error is not None:
                raise drain_error from None

        def start_games(specs: list[dict[str, Any]]) -> None:
            check_drain()
            try:
                self.runtime.start(specs)
            except Exception:
                # Native admission may have observed cancellation first.
                check_drain()
                raise

        def register_group(index: int) -> tuple[ManagedServer, Json]:
            group = groups[index]
            capture = (None if http_diagnostics is None else HttpPairCapture.configured(
                http_diagnostics, operation_id=operation_id or run_id, collection_id=batch_id,
                pair_id=group.episodes[0].pair_id, episode_count=len(group.episodes)))
            managed = ManagedServer(server.binary, server.config, group.seed,
                len(group.episodes), output / 'servers' / group.episodes[0].episode_id,
                startup_seconds=min(server.startup_seconds, remaining(execution_deadline)),
                shutdown_seconds=server.shutdown_seconds, cleanup_deadline=cleanup_deadline,
                strict_cleanup_deadline=True,
                **({} if capture is None else {"http_capture": capture}))
            self.servers[index] = managed
            history: Json = {'server': managed, 'runtime_evidence': None}
            self.server_history.append(history)
            if capture is not None:
                self._pending_http[managed] = (HttpFacts(), history)
            return managed, history

        async def start_server(index: int, managed: ManagedServer, history: Json) -> None:
            startup = await managed.start()
            process = managed.process

            def retain_receipt(receipt: Json) -> None:
                history['runtime_evidence'] = receipt
                startups[index], receipts[index], positions[index] = startup, receipt, 0

            if not parallel_start:
                retain_receipt(evidence.verify_executed(process))
                return

            original = evidence.prepare_executed_read(process, deadline=execution_deadline)
            read = asyncio.create_task(asyncio.to_thread(original.read))
            try:
                receipt = await asyncio.shield(read)
            except asyncio.CancelledError as primary:
                # Do not hand this live server to cleanup while its original
                # executable is still being read.
                while not read.done():
                    try:
                        await asyncio.shield(read)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:  # noqa: BLE001 - retrieve the completed result below.
                        break
                try:
                    receipt = read.result()
                    evidence.confirm_executed_read(managed.process, original)
                except BaseException as failure:  # noqa: BLE001 - preserve the original cancellation.
                    primary.add_note("executed receipt not retained: " + type(failure).__name__)
                else:
                    retain_receipt(receipt)
                raise
            evidence.confirm_executed_read(managed.process, original)
            retain_receipt(receipt)

        async def bind_group(index: int) -> None:
            group = groups[index]
            startup, receipt = startups[index], receipts[index]
            real_contexts = tuple(EpisodeEncodingContext(**contexts[episode.episode_id],
                runtime_evidence=receipt) for episode in group.episodes)
            await asyncio.to_thread(writer.bind_contexts, real_contexts)
            bound_contexts[index] = digest([asdict(context) for context in real_contexts])
            for episode in group.episodes:
                value = contexts[episode.episode_id]
                current_specs[episode.episode_id] = {'episode': episode.episode_id, 'run': value['run_id'],
                    'trial': value['trial'], 'actions': asdict(actions),
                    'rules': asdict(value['rules']), 'numeric_costs': value['numeric_costs'],
                    'policies': [policy.native for policy in value['policies']],
                    'max_decisions': max_decisions, 'deadline': execution_deadline,
                    'cleanup_deadline': cleanup_deadline, 'runtime_evidence': receipt,
                    'server': {'base_url': startup['base_url'],
                        'expected_fingerprint': startup['test_server_info']['seed_fingerprint'],
                        'expected_effective': startup['test_server_config'], 'startup_info': startup['test_server_info'],
                        'request_seconds': server.request_seconds}}
                if 'learner_viewer' in value:
                    current_specs[episode.episode_id]['learner_viewer'] = value['learner_viewer']
            if not full_wave:
                start_games([current_specs[group.episodes[0].episode_id]])

        async def admit_group(index: int) -> None:
            managed, history = register_group(index)
            await start_server(index, managed, history)
            await bind_group(index)

        initial_groups = min(len(groups), game_concurrency)
        if parallel_start:
            for offset in range(0, initial_groups, player_worker_threads):
                cohort = range(offset, min(offset + player_worker_threads, initial_groups))
                owners = [(index, *register_group(index)) for index in cohort]
                try:
                    async with asyncio.TaskGroup() as startup_tasks:
                        for index, managed, history in owners:
                            startup_tasks.create_task(start_server(index, managed, history))
                except ExceptionGroup as failure:
                    if len(failure.exceptions) == 1 and not isinstance(failure.exceptions[0], BaseExceptionGroup):
                        original = failure.exceptions[0]
                        raise original from original.__cause__
                    raise
            # Every startup has settled. Bind the original contexts in plan
            # order before the single native admission of the whole wave.
            for index in range(initial_groups):
                await bind_group(index)
            next_group = initial_groups
        else:
            while next_group < initial_groups:
                await admit_group(next_group)
                next_group += 1
        if full_wave:
            # Repeated-seed games have independent IDs and use the same seed.
            # Submit only after the original writer contexts are bound.
            writer._finish_rollout_registration()
            start_games([current_specs[episode.episode_id] for group in groups for episode in group.episodes])
        episode_group = {episode.episode_id: index for index, group in enumerate(groups) for episode in group.episodes}
        loop = asyncio.get_running_loop()
        terminal_ready = asyncio.Event()

        def drain_storage() -> None:
            nonlocal drain_error
            terminal_count = 0
            rollout = None
            deferred_releases: list[tuple[tuple[str, str, str], str]] = []
            try:
                rollout = writer._begin_rollout_read()
                defer_release = rollout is not None and full_wave
                while terminal_count < total:
                    if rollout is None:
                        bundle, packet = self.runtime.next_storage(execution_deadline), None
                    else:
                        packet = rollout.next_terminal(execution_deadline)
                        accepted.extend(json.loads(packet.prior_receipts()))
                        bundle = packet.bundle()
                    row = {'id': bundle.id, 'kind': bundle.kind, 'episode': bundle.episode,
                        'game': bundle.game, 'transport_sha256': bundle.transport_sha256, 'bytes': bundle.reserved_bytes}
                    if packet is None:
                        writer.accept(bundle)
                    else:
                        writer.accept_rollout(bundle, packet)
                    if defer_release:
                        owner = self.runtime._storage_ack_defer_release(bundle)
                        if owner is not None:
                            deferred_releases.append(owner)
                    else:
                        self.runtime.storage_ack(bundle)
                    if http_diagnostics is not None and row['kind'] == 'terminal':
                        facts = bundle._http_diagnostics()
                        if facts is None:
                            raise ContractError('Rust terminal bundle is missing its HTTP diagnostic facts')
                        terminal_http[row['episode']] = facts
                    # One producer appends completed metadata. The event-loop cursor
                    # only reads this prefix and never owns a native bundle.
                    accepted.append(row)
                    del bundle, packet
                    if row['kind'] == 'terminal':
                        terminal_count += 1
                        loop.call_soon_threadsafe(terminal_ready.set)
                if defer_release:
                    self.runtime._settle_deferred_storage_releases(
                        deferred_releases, execution_deadline,
                    )
            except BaseException as error:
                # Publish the cause and stop native admission even if the event loop
                # has not yet delivered the executor future's completion callback.
                drain_error = error
                cancel_owner(self.runtime.cancel, error)
                if rollout is not None:
                    cancel_owner(lambda _: writer._finish_rollout_read(failed=True), error)
                raise
            if rollout is not None:
                writer._finish_rollout_read(failed=False)

        drain = loop.run_in_executor(None, drain_storage)
        drain.add_done_callback(lambda _: terminal_ready.set())
        terminal_count = cursor = 0
        try:
            while terminal_count < total:
                terminal_ready.clear()
                check_drain()
                while cursor < len(accepted):
                    row = accepted[cursor]
                    cursor += 1
                    if row['kind'] == 'terminal':
                        break
                else:
                    await terminal_ready.wait()
                    continue
                terminal_count += 1
                if progress is not None:
                    # Terminal receipts include abnormal closures. Storage ACK
                    # precedes the event-loop observation, not durable sealing.
                    observe(progress, {'batch_games_ended': terminal_count})
                index = episode_group[row['episode']]
                positions[index] += 1
                group = groups[index]
                managed = self.servers[index]
                if http_diagnostics is not None:
                    responses, errors, network, abnormal = terminal_http.pop(row["episode"])
                    facts, _ = self._pending_http[managed]
                    facts.responses += responses
                    facts.error_responses += errors
                    facts.transport_errors += network
                    if abnormal:
                        abnormal_pairs.add(index)
                if positions[index] == len(group.episodes):
                    health = await managed.stop()
                    completed_groups[index] = {'runtime_evidence': receipts[index],
                        'startup': startups[index], 'expected_effective': managed.expected, 'process': clone(health)}
                    diagnostic = await self.settle_http(managed, deadline=cleanup_deadline,
                                                        abnormal=index in abnormal_pairs)
                    if diagnostic is not None:
                        completed_groups[index]["http_diagnostics"] = diagnostic
                    if next_group < len(groups):
                        check_drain()
                        await admit_group(next_group)
                        next_group += 1
                elif not full_wave:
                    start_games([current_specs[group.episodes[positions[index]].episode_id]])
            await asyncio.shield(drain)
        except BaseException as error:
            for cancel in (self.runtime.cancel, writer.cancel):
                cancel_owner(cancel, error)
            # The caller owns final cleanup. Keep the original reader alive
            # until it has actually unwound, including on a second cancellation.
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    continue
                except BaseException:  # noqa: BLE001 - retrieve the completed outcome below.
                    break
            try:
                drain.result()
            except BaseException as cleanup:  # noqa: BLE001 - preserve the first failure.
                if cleanup is not error:
                    error.add_note('Journal drain exit: ' + type(cleanup).__name__)
            raise
        await asyncio.to_thread(self.runtime.wait, execution_deadline)
        keys = [(run_id, batch_id, episode.episode_id) for group in groups for episode in group.episodes]
        reports = self.runtime.completed_reports(keys)
        return RustMatchResult(accepted, bound_contexts, completed_groups, receipts, keys, reports, total)
