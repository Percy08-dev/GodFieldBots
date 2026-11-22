"""Concrete managed Rust match execution; no registry or publication authority."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import EpisodeGroup, Json, clone, digest
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.process import ManagedServer
from gf_simulator_client.rust_player import RustPlayerRuntime
from gf_training.group_pack_resources import remaining
from gf_training.rust_player_storage import EpisodeEncodingContext, RustPlayerStorage


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

    async def execute(self, groups: tuple[EpisodeGroup, ...], contexts: dict[str, Json],
                      writer: RustPlayerStorage, *, server: Any, actions: Any,
                      max_decisions: int, output: Path, execution_deadline: float,
                      cleanup_deadline: float, game_concurrency: int, run_id: str,
                      batch_id: str, evidence: CardInstanceRuntimeEvidence) -> RustMatchResult:
        """Caller already preflighted; errors return through its original cleanup."""
        accepted: list[Json] = []
        bound_contexts: dict[int, str] = {}
        completed_groups: dict[int, Json] = {}
        current_specs: dict[str, Json] = {}
        positions: dict[int, int] = {}
        receipts: dict[int, Json] = {}
        startups: dict[int, Json] = {}
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

        async def admit_group(index: int) -> None:
            group = groups[index]
            managed = ManagedServer(server.binary, server.config, group.seed,
                len(group.episodes), output / 'servers' / group.episodes[0].episode_id,
                startup_seconds=min(server.startup_seconds, remaining(execution_deadline)),
                shutdown_seconds=server.shutdown_seconds, cleanup_deadline=cleanup_deadline,
                strict_cleanup_deadline=True)
            self.servers[index] = managed
            history: Json = {'server': managed, 'runtime_evidence': None}
            self.server_history.append(history)
            startup = await managed.start()
            receipt = evidence.verify_executed(managed.process)
            history['runtime_evidence'] = receipt
            startups[index], receipts[index], positions[index] = startup, receipt, 0
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
            start_games([current_specs[group.episodes[0].episode_id]])

        while next_group < min(len(groups), game_concurrency):
            await admit_group(next_group)
            next_group += 1
        episode_group = {episode.episode_id: index for index, group in enumerate(groups) for episode in group.episodes}
        total = sum(len(group.episodes) for group in groups)
        loop = asyncio.get_running_loop()
        terminal_ready = asyncio.Event()

        def drain_storage() -> None:
            nonlocal drain_error
            terminal_count = 0
            try:
                while terminal_count < total:
                    bundle = self.runtime.next_storage(execution_deadline)
                    row = {'id': bundle.id, 'kind': bundle.kind, 'episode': bundle.episode,
                        'game': bundle.game, 'transport_sha256': bundle.transport_sha256, 'bytes': bundle.reserved_bytes}
                    writer.accept(bundle)
                    self.runtime.storage_ack(bundle)
                    # One producer appends completed metadata. The event-loop cursor
                    # only reads this prefix and never owns a native bundle.
                    accepted.append(row)
                    del bundle
                    if row['kind'] == 'terminal':
                        terminal_count += 1
                        loop.call_soon_threadsafe(terminal_ready.set)
            except BaseException as error:
                # Publish the cause and stop native admission even if the event loop
                # has not yet delivered the executor future's completion callback.
                drain_error = error
                cancel_owner(self.runtime.cancel, error)
                raise

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
                index = episode_group[row['episode']]
                positions[index] += 1
                group = groups[index]
                if positions[index] < len(group.episodes):
                    start_games([current_specs[group.episodes[positions[index]].episode_id]])
                else:
                    managed = self.servers[index]
                    health = await managed.stop()
                    completed_groups[index] = {'runtime_evidence': receipts[index],
                        'startup': startups[index], 'expected_effective': managed.expected, 'process': clone(health)}
                    if next_group < len(groups):
                        check_drain()
                        await admit_group(next_group)
                        next_group += 1
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
