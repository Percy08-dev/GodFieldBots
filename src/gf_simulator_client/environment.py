"""Game/group ownership. Journal/model factories are injected by the CLI root."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from gf_common.domain import (
    ActionConfig,
    CandidateSet,
    ContractError,
    DecisionInput,
    EpisodeGroup,
    EpisodeResult,
    EpisodeSpec,
    GroupResult,
    Json,
    Policy,
    PolicyDecision,
    UnsupportedContract,
    canonical,
    digest,
)
from gf_common.validation import (
    decision_input,
    projected_abort_kind,
    reconcile_replay,
    replay_closure,
)
from gf_simulator_client.http import (
    ApiError,
    AttestationError,
    AttestedClient,
    DeliveryUnknown,
    seed_fingerprint,
)
from gf_simulator_client.process import ManagedServer, server_effective
from gf_simulator_client.workers import ParticipantWorker


class Journal(Protocol):
    def begin(self, value: DecisionInput, candidates: CandidateSet, selection: PolicyDecision,
              command: Json, elapsed_ms: float) -> int: ...
    def acknowledge(self, index: int, acknowledgement: Json) -> None: ...
    def finish(self, end_kind: str, outcome: Any, final_observation: Json | None = None,
               simulator_abort_reason: str | None = None) -> None: ...
    def close_incomplete(self) -> None: ...


@dataclass(frozen=True)
class EnvironmentSettings:
    binary: str
    server_config: str
    output_dir: Path
    run_id: str
    actions: ActionConfig
    fixed_seed_experiment: bool = True
    expected_fingerprint: str | None = None
    external_url: str | None = None
    startup_seconds: float = 15
    request_seconds: float = 10
    worker_seconds: float = 10
    shutdown_seconds: float = 30
    transport_retries: int = 2
    max_decisions: int = 20000
    admission_deadline: float = float("inf")
    execution_deadline: float = float("inf")
    wall_deadline: float = float("inf")
    work_deadline: float = float("inf")
    replay_deadline: float = float("inf")


def _failure(exc: BaseException) -> str:
    if isinstance(exc, UnsupportedContract):
        return "unsupported_contract"
    if isinstance(exc, AttestationError):
        return "attestation_failure"
    if isinstance(exc, DeliveryUnknown):
        return "transport_failure"
    if isinstance(exc, ApiError):
        code = exc.body.get("error", {}).get("code") if isinstance(exc.body, dict) else None
        if code == "INVALID_ACTION":
            return "rejection"
        if code == "GAME_ABORTED":
            return "abort"
        if exc.status == 409:
            return "conflict"
        return "infrastructure_failure"
    if isinstance(exc, TimeoutError):
        return "deadline"
    if isinstance(exc, ContractError):
        return "contract_failure"
    if isinstance(exc, OSError):
        return "infrastructure_failure"
    return "worker_failure"


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    temporary.replace(path)




class SimulatorEnvironment:
    def __init__(self, settings: EnvironmentSettings, policy_factory: Callable[[str], Policy],
                 journal_factory: Callable[..., Journal]) -> None:
        self.settings = settings
        self.policy_factory = policy_factory
        self.journal_factory = journal_factory

    async def _episode(self, client: AttestedClient, definitions: Json, group: EpisodeGroup,
                       spec: EpisodeSpec) -> EpisodeResult:
        started = time.monotonic()
        result = EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies)
        directory = self.settings.output_dir / spec.episode_id
        directory.mkdir(parents=True, exist_ok=False)
        journals: dict[int, Journal] = {}
        workers: dict[int, ParticipantWorker] = {}
        accepted: dict[int, list[Json]] = {1: [], 2: []}
        final: dict[int, Json] = {}
        game: str | None = None
        try:
            client.deadline = self.settings.execution_deadline
            async with asyncio.timeout_at(self.settings.execution_deadline):
                created = await client.create()
                game = created["game_id"]
                result.game_id = game
                for viewer in (1, 2):
                    joined = await client.join(game, str(uuid.uuid4()))
                    if joined["player_id"] != viewer:
                        raise ContractError("unexpected participant allocation")
                    try:
                        journals[viewer] = self.journal_factory(directory, spec.episode_id, viewer, spec.policies[viewer - 1],
                                                                   run_id=self.settings.run_id, game_id=game,
                                                                   action_config_hash=digest(asdict(self.settings.actions)),
                                                                   use_action_options=self.settings.actions.use_action_options)
                    except OSError as exc:
                        result.health["save_failure"] = str(exc)
                        raise
                    workers[viewer] = ParticipantWorker(self.policy_factory, spec.policies[viewer - 1],
                                                        spec.sampling_seeds[viewer - 1], self.settings.actions,
                                                        self.settings.worker_seconds)
                histories: dict[int, list[Json]] = {1: [], 2: []}
                cursors: dict[int, str | None] = {1: None, 2: None}
                obs = await client.observe(game, 1)
                result.first_player = obs["first_player"]
                while True:
                    if obs["result"] is not None:
                        result.outcome = obs["result"]
                        result.end_kind = "natural"
                        break
                    if result.decisions >= self.settings.max_decisions:
                        result.end_kind = "truncated"
                        result.failure = "decision_limit"
                        break
                    actor = obs["current_player"]
                    if actor not in (1, 2):
                        raise UnsupportedContract("no supported acting participant")
                    if obs["viewer_player_id"] != actor:
                        obs = await client.observe(game, actor)
                    if not obs["actionable"]:
                        raise UnsupportedContract("current actor has no typed input contract")
                    events, cursors[actor] = await client.events(game, actor, cursors[actor])
                    histories[actor].extend(events)
                    options = await client.actor_options(game, actor, obs) if self.settings.actions.use_action_options else None
                    value = decision_input(obs, histories[actor], cursors[actor], definitions, options)
                    selection_started = time.monotonic()
                    candidates, selected = await workers[actor].select(value)
                    elapsed_ms = (time.monotonic() - selection_started) * 1000
                    payload = {"command_id": str(uuid.uuid4()), "expected_revision": obs["state_revision"],
                               "action": candidates.actions[selected.index]}
                    try:
                        index = journals[actor].begin(value, candidates, selected, payload, elapsed_ms)
                    except OSError as exc:
                        result.health["save_failure"] = str(exc)
                        raise
                    result.candidate_counts.append(len(candidates.actions))
                    result.search_latency_ms.append(candidates.latency_ms)
                    result.search_attempts.append(candidates.attempts)
                    acknowledgement = await client.act(game, actor, payload)
                    # Keep known outcome even if subsequent recording fails.
                    if acknowledgement["status"] == "terminal":
                        if acknowledgement["result"] is None:
                            raise ContractError("terminal ack lacks result")
                        result.outcome = acknowledgement["result"]
                        result.end_kind = "natural"
                    elif acknowledgement["status"] == "aborted":
                        result.end_kind = "aborted"
                    accepted[actor].append(payload["action"])
                    result.decisions += 1
                    try:
                        journals[actor].acknowledge(index, acknowledgement)
                    except OSError as exc:
                        result.health["save_failure"] = str(exc)
                        raise
                    if acknowledgement["status"] != "active":
                        break
                    obs = await client.observe(game, actor)
        except asyncio.CancelledError:
            result.end_kind = "cancelled" if result.outcome is None else result.end_kind
            result.failure = "cancelled"
        except (OSError, ValueError, RuntimeError, TimeoutError, EOFError) as exc:
            failure = "save_failure" if result.health.get("save_failure") else _failure(exc)
            if result.outcome is None:
                result.end_kind = "aborted" if failure == "abort" else "truncated" if failure == "deadline" else "incomplete"
            result.failure = failure
            try:
                if time.monotonic() >= min(self.settings.wall_deadline, self.settings.work_deadline):
                    raise TimeoutError("failure persistence deadline")
                _save(directory / "failure.json", {"category": failure, "message": str(exc),
                                                    "diagnostic": exc.body if isinstance(exc, ApiError) else None})
            except (OSError, TimeoutError) as save_error:
                result.health["save_failure"] = str(save_error)
        finally:
            await self._shield_cleanup(self._finalize_episode(client, game, directory, result, journals,
                                                              workers, accepted, final), result.health)
            if result.health.get("finalization_cancelled"):
                result.failure = result.failure or "cancelled"
            result.elapsed_seconds = time.monotonic() - started
            if time.monotonic() >= min(self.settings.wall_deadline, self.settings.work_deadline):
                result.health["finalization_deadline"] = "wall allowance exhausted"
            try:
                if time.monotonic() >= min(self.settings.wall_deadline, self.settings.work_deadline):
                    raise TimeoutError("result persistence deadline")
                _save(directory / "result.json", asdict(result))
                if time.monotonic() > min(self.settings.wall_deadline, self.settings.work_deadline):
                    result.health["budget_overrun"] = "result persistence exceeded wall deadline"
            except (OSError, TimeoutError) as exc:
                result.health["save_failure"] = str(exc)
        return result

    async def _shield_cleanup(self, cleanup: Any, health: Json) -> None:
        # Once a result is known, caller cancellation must not erase it or strand
        # workers. Every awaited stage inside cleanup has its remaining budget.
        task = asyncio.create_task(cleanup)
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                health["finalization_cancelled"] = "cancellation received while cleaning up"
                if task.done():
                    if task.cancelled():
                        health["cleanup_failure"] = "cleanup task cancelled"
                    return

    async def _finalize_episode(self, client: AttestedClient, game: str | None, directory: Path,
                                result: EpisodeResult, journals: dict[int, Journal],
                                workers: dict[int, ParticipantWorker], accepted: dict[int, list[Json]],
                                final: dict[int, Json]) -> None:
        def remaining() -> float:
            return max(0.0, min(self.settings.wall_deadline, self.settings.work_deadline) - time.monotonic())

        try:
            client.deadline = min(self.settings.replay_deadline, min(self.settings.wall_deadline, self.settings.work_deadline))
            if game is not None and not client.poisoned and time.monotonic() < min(self.settings.replay_deadline, min(self.settings.wall_deadline, self.settings.work_deadline)):
                async with asyncio.timeout_at(min(self.settings.replay_deadline, min(self.settings.wall_deadline, self.settings.work_deadline))):
                    if result.outcome is not None:
                        for viewer in (1, 2):
                            final[viewer] = await client.observe(game, viewer)
                            if final[viewer]["result"] != result.outcome:
                                raise ContractError("terminal observation mismatch")
                    closures = []
                    for viewer in (1, 2):
                        replay = await client.replay(game, viewer)
                        if replay.get("game_id") != game:
                            raise ContractError("replay game identity mismatch")
                        if time.monotonic() >= min(self.settings.replay_deadline, min(self.settings.wall_deadline, self.settings.work_deadline)):
                            raise TimeoutError("replay persistence deadline")
                        _save(directory / f"replay-player-{viewer}.json", replay)
                        reconcile_replay(replay, viewer, accepted[viewer], result.outcome)
                        closures.append(replay_closure(replay, game, viewer))
                    if closures[0] != closures[1]:
                        raise ContractError("participant replay closure identities disagree")
                    closure = closures[0]
                    if result.outcome is None and closure is not None and closure["type"] == "aborted":
                        result.simulator_abort_reason = closure["reason"]
                        result.end_kind = projected_abort_kind(closure["reason"])
                    elif result.end_kind == "aborted" and closure is None:
                        raise ContractError("simulator abort lacks reconciled closure evidence")
                    elif result.outcome is None and closure is not None:
                        raise ContractError("terminal replay cannot fabricate an unknown natural outcome")
                result.replay_reconciled = True
            elif game is not None:
                result.health["replay_or_finalization_failure"] = "connection poisoned or finalization deadline"
        except (OSError, ValueError, RuntimeError, TimeoutError, asyncio.CancelledError) as exc:
            result.health["replay_or_finalization_failure"] = str(exc) or type(exc).__name__
        finally:
            for viewer, journal in journals.items():
                try:
                    if remaining() <= 0 or result.outcome is not None and viewer not in final:
                        journal.close_incomplete()
                        result.health["save_failure"] = "no terminal observation or journal finalization budget"
                    else:
                        journal.finish(result.end_kind, result.outcome, final.get(viewer), result.simulator_abort_reason)
                        if remaining() <= 0:
                            result.health["budget_overrun"] = "journal persistence exceeded wall deadline"
                except (OSError, ValueError, RuntimeError) as exc:
                    result.health["save_failure"] = str(exc)
                    try:
                        journal.close_incomplete()
                    except (OSError, ValueError, RuntimeError) as close_error:
                        result.health["journal_close_failure"] = str(close_error)
            closed = await asyncio.gather(*(worker.close(timeout=remaining()) for worker in workers.values()),
                                          return_exceptions=True)
            for worker in workers.values():
                if any(getattr(worker, "health", {}).values()):
                    result.health["worker_cleanup_failure"] = str(worker.health)
            for failure in closed:
                if isinstance(failure, BaseException):
                    result.health["worker_cleanup_failure"] = str(failure)
            if game is not None:
                if not client.poisoned and remaining() > 0:
                    try:
                        async with asyncio.timeout(remaining()):
                            client.deadline = min(self.settings.wall_deadline, self.settings.work_deadline)
                            await client.delete(game)
                    except (OSError, ValueError, RuntimeError, TimeoutError, asyncio.CancelledError) as exc:
                        result.health["delete_failure"] = str(exc) or type(exc).__name__
                else:
                    result.health["delete_failure"] = "connection poisoned or finalization deadline"


    async def run_group(self, group: EpisodeGroup) -> GroupResult:
        results: list[EpisodeResult] = []
        health: Json = {}
        server: ManagedServer | None = None
        client: AttestedClient | None = None
        try:
            if time.monotonic() >= self.settings.admission_deadline:
                raise TimeoutError("group admission deadline")
            startup = None
            if self.settings.external_url is None:
                server = ManagedServer(self.settings.binary, self.settings.server_config, group.seed,
                                       len(group.episodes), self.settings.output_dir / f"server-{group.episodes[0].episode_id}",
                                       self.settings.startup_seconds, self.settings.shutdown_seconds,
                                       cleanup_deadline=min(self.settings.wall_deadline, self.settings.work_deadline))
                async with asyncio.timeout_at(self.settings.execution_deadline):
                    startup = await server.start()
                health["startup"] = startup
                url = startup["base_url"]
                expected = server.expected
            else:
                url = self.settings.external_url
                expected = server_effective(self.settings.server_config, None)
            health["expected_effective"] = expected
            client = AttestedClient(url, expected_fingerprint=self.settings.expected_fingerprint or seed_fingerprint(group.seed),
                                    fixed_seed_experiment=self.settings.fixed_seed_experiment,
                                    expected_effective=expected, startup_info=startup["test_server_info"] if startup else None,
                                    request_seconds=self.settings.request_seconds, retries=self.settings.transport_retries,
                                    deadline=self.settings.execution_deadline)
            await client.attest()
            definitions = await client.public_catalog()
            for spec in group.episodes:
                if time.monotonic() >= self.settings.admission_deadline:
                    raise TimeoutError("game admission deadline")
                result = await self._episode(client, definitions, group, spec)
                results.append(result)
                if result.failure == "cancelled":
                    raise asyncio.CancelledError
                if client.poisoned:
                    raise AttestationError("connection poisoned; remaining games forbidden")
        except asyncio.CancelledError:
            health["failure"] = "cancelled"
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            health["failure"] = _failure(exc)
            health["message"] = str(exc)
        finally:
            async def close_group() -> None:
                if client:
                    try:
                        async with asyncio.timeout(max(0.0, min(self.settings.wall_deadline, self.settings.work_deadline) - time.monotonic())):
                            await client.close()
                    except (TimeoutError, OSError):
                        health["client_close_failure"] = "close deadline/failure"
                if server:
                    health["process"] = await server.stop(max(0.0, min(self.settings.wall_deadline, self.settings.work_deadline) - time.monotonic()))
                    if health["process"]["exit_code"] != 0 or health["process"]["shutdown"] != "stdin_eof" or health["process"]["log_error"]:
                        for episode in results:
                            episode.health["process_failure"] = str(health["process"])
                if time.monotonic() > min(self.settings.wall_deadline, self.settings.work_deadline):
                    health["budget_overrun_seconds"] = time.monotonic() - min(self.settings.wall_deadline, self.settings.work_deadline)
                    for episode in results:
                        episode.health["budget_overrun"] = "bounded cleanup escalation exceeded wall allowance"
            await self._shield_cleanup(close_group(), health)
            recorded = {r.episode_id for r in results}
            for spec in group.episodes:
                if spec.episode_id not in recorded:
                    results.append(EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                                 end_kind="incomplete", failure=health.get("failure", "not_admitted")))
        return GroupResult(results, health)
