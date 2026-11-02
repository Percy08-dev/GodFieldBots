"""Parent journal5 with once-only event deltas and typed candidate sidecar links."""
from __future__ import annotations

import math
import os
import random
import re
import stat
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from gf_common.action_contract import ActionContract
from gf_common.diverse_actions import GENERATOR_V2, GeneratedCandidates, NumericCosts
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    ContractError,
    Json,
    canonical,
    closed,
    digest,
    integer,
)
from gf_common.structured import DecisionFrame, EventBank, EventBankView, EventDelta, Selection
from gf_common.structured_tokens import recorded_aggregate_recipe, signature
from gf_common.structured_wire import (
    delta_dict,
    frame_dict,
    parse_delta,
    parse_frame,
    parse_report,
    parse_selection,
)
from gf_common.validation import observation, projected_abort_kind, result
from gf_models.structured_binding import check_input_binding, input_binding
from gf_training.candidate_storage import (
    _bounded_hash,
    _identity,
    _open_regular,
    publish_candidates,
    read_candidates,
    sidecar_inventory,
)
from gf_training.journal import REWARD_VERSION, ParticipantJournal, _acknowledgement, _reward

STRUCTURED_JOURNAL_SCHEMA = 5
STRUCTURED_FEATURE_VERSION = "participant-structured-v1"
END_KINDS = ("natural", "truncated", "aborted", "cancelled", "incomplete")


class StructuredJournal(ParticipantJournal):
    def __init__(self, directory: Path, episode_id: str, viewer: int, policy: str, *, run_id: str, game_id: str,
                 action_config_hash: str, bank_episode: str, caps: Json, numeric_costs: Json,
                 model_id: str, model_version: str, aggregate_window: int = 128,
                 sampling_seed: int = 0, use_action_options: bool = True, action_contract: ActionContract,
                 purpose: str, synthetic_capacity: bool) -> None:
        classification = input_binding(action_contract, purpose, synthetic_capacity)
        if synthetic_capacity:
            raise ContractError("journal5 game proposals reject synthetic capacity")
        self.action_contract, self.purpose = action_contract, purpose
        from gf_models.structured_runtime import WorkerCaps
        caps = asdict(WorkerCaps(**closed(caps, set(WorkerCaps.__dataclass_fields__))))
        numeric_costs = asdict(NumericCosts(**closed(numeric_costs, set(NumericCosts.__dataclass_fields__))))
        integer(aggregate_window, 1, min(65536, caps["max_episode_events"]))
        if type(action_config_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", action_config_hash) is None:
            raise ContractError("journal5 action hash required")
        if model_version == "2":
            if model_id not in ("mlp", "gru", "transformer", "lightgbm"):
                raise ContractError("journal5 model2 requires a supported neural kind")
        elif model_version != "1" or model_id not in ("random", "heuristic"):
            raise ContractError("journal5 baseline/model version mix")
        if use_action_options is not True:
            raise ContractError("journal5 requires actor atomic hints")
        integer(viewer, 1, 2)
        integer(sampling_seed)
        self.final_path = directory / f"player-{viewer}.jsonl"
        self.path = directory / f"player-{viewer}.partial.jsonl"
        if self.final_path.exists() or self.final_path.is_symlink():
            raise ContractError("journal5 final path already exists")
        self.output = self.path.open("x+", encoding="utf-8")
        created = os.fstat(self.output.fileno())
        self._owned_inode = (created.st_dev, created.st_ino)
        self.viewer, self.count, self.use_action_options = viewer, 0, True
        self.pending: set[int] = set()
        self.commands: dict[int, Json] = {}
        self.lifecycle_ack: Json | None = None
        self.last_accepted: int | None = None
        self.last_frame: DecisionFrame | None = None
        self.bank = EventBank(viewer, bank_episode, caps["max_episode_events"])
        self.caps, self.policy = caps, policy
        self.header = {**classification, "kind": "header", "schema_version": 5, "feature_version": STRUCTURED_FEATURE_VERSION,
                       "input_recipe": STRUCTURED_FEATURE_VERSION, "reward_version": REWARD_VERSION,
                       "policy_input_version": POLICY_INPUT_VERSION, "end_projection_version": END_PROJECTION_VERSION,
                       "catalog": CATALOG_META, "episode_id": episode_id, "viewer": viewer, "policy": policy,
                       "run_id": run_id, "game_id": game_id, "action_config_hash": action_config_hash,
                       "generator_version": GENERATOR_V2, "bank_episode": bank_episode, "caps": caps,
                       "numeric_costs": numeric_costs, "model_id": model_id, "model_version": model_version,
                       "generation_schema": 2 if model_version == "2" and model_id != "lightgbm" else None,
                       "aggregate_window": aggregate_window, "token_signature": signature(aggregate_window),
                       "sampling_seed": sampling_seed, "use_action_options": True}
        self.write(self.header)
        self.last_storage_ms = 0.
        search_seed = int(digest({"recipe": "structured-search-stream-v1", "sampling_seed": sampling_seed})[:16], 16)
        self.last_search_state = random.Random(search_seed).getstate()

    def begin(self, *args: Any, **kwargs: Any) -> int:
        raise ContractError("journal5 requires explicit frame/delta/compact-table begin_frame")

    def begin_frame(self, frame: DecisionFrame, delta: EventDelta, generated: GeneratedCandidates,
                    selection: Selection, command: Json, timings: Json) -> int:
        started = time.perf_counter()
        check_input_binding({"action_contract": asdict(generated.table.action_contract), "purpose": generated.table.purpose,
                             "synthetic_capacity": generated.table.synthetic_capacity}, action_contract=self.action_contract,
                            purpose=self.purpose, synthetic_capacity=False)
        if self.caps["max_decisions"] is not None:
            integer(self.count + 1, 1, self.caps["max_decisions"])
        integer(len(generated.table.kinds), 1, self.caps["max_candidates"])
        integer(frame.boundary.end, 0, self.caps["max_episode_events"])
        interval_count = frame.boundary.end - frame.interval_start
        integer(interval_count, 0, self.caps["max_interval_events"])
        # Count already-owned encoded events without decoding/copying the delta.
        delta_count = len(delta._events)
        integer(delta_count, 0, self.caps["max_interval_events"])
        integer(delta.start + delta_count, 0, self.caps["max_episode_events"])
        if delta_count != interval_count:
            raise ContractError("journal5 delta count/frame interval mismatch")
        for start, end in zip(generated.table.offsets[:-1], generated.table.offsets[1:], strict=True):
            integer(end - start, 0, self.caps["max_positions"])
        if self.header["model_id"] in ("mlp", "gru", "transformer") and (selection.logprob is None or selection.value is None):
            raise ContractError("journal5 neural selection requires finite logprob/value")
        if self.header["model_id"] == "lightgbm" and (selection.logprob is not None or selection.value is not None):
            raise ContractError("journal5 Tree2 policy has no policy logprob/critic value")
        if self.lifecycle_ack is not None or self.pending:
            raise ContractError("journal5 decision after pending/lifecycle input")
        if frame.actor != self.viewer or frame.episode != self.bank.episode or delta.start != frame.interval_start:
            raise ContractError("journal5 own actor/frame/delta mismatch")
        previous = 0 if self.last_frame is None else self.last_frame.boundary.end
        if frame.interval_start != previous or self.last_frame is not None and frame.revision <= self.last_frame.revision:
            raise ContractError("journal5 decision interval/revision continuity mismatch")
        action = selection.validate(generated.table, frame)
        _command(command, frame, action)
        self.bank.append(delta)
        if self.bank.boundary != frame.boundary:
            raise ContractError("journal5 delta must match exact frame boundary")
        frame.bank_view(self.bank.view())
        if selection.policy_version != self.policy or generated.report.config_hash != self.header["action_config_hash"]:
            raise ContractError("journal5 policy/search configuration mismatch")
        if generated.report.rng_before != self.last_search_state:
            raise ContractError("journal5 independent search RNG continuity mismatch")
        if generated.report.numeric_costs_hash != digest(self.header["numeric_costs"]):
            raise ContractError("journal5 numeric cost binding mismatch")
        _host_timings(timings)
        index = self.count
        descriptor = publish_candidates(self.path.parent, frame, generated.table, index,
                                        max_positions=self.caps["max_positions"], action_contract=self.action_contract,
                                        purpose=self.purpose, synthetic_capacity=False)
        # Sidecar is closed/verified/published BEFORE any referencing JSONL row.
        if self.last_accepted is not None:
            self.write({"kind": "transition", "decision": self.last_accepted, "next_decision": index,
                        "reward": 0, "terminal": False})
            self.last_accepted = None
        self.write({"kind": "events", "delta": delta_dict(delta), "boundary": {"end": frame.boundary.end,
                    "prefix_hash": frame.boundary.prefix_hash}})
        self.write({"kind": "decision", "decision": index, "frame": frame_dict(frame),
                    "table": descriptor, "search_report": asdict(generated.report), "selection": asdict(selection),
                    "command": command, "timings": timings})
        self.pending.add(index)
        self.commands[index] = command
        self.count += 1
        self.last_frame = frame
        self.last_search_state = generated.report.rng_after
        self.last_storage_ms = (time.perf_counter() - started) * 1000
        return index

    def finish(self, end_kind: str, outcome: Any, final_observation: Json | None = None,
               simulator_abort_reason: str | None = None) -> None:
        if end_kind not in END_KINDS:
            raise ContractError("invalid journal5 end kind")
        if self.final_path.exists() or self.final_path.is_symlink():
            raise ContractError("journal5 refuses to overwrite existing final evidence")
        if simulator_abort_reason is not None and end_kind != projected_abort_kind(simulator_abort_reason):
            raise ContractError("simulator abort cause/projected kind mismatch")
        if self.lifecycle_ack is not None:
            expected = "natural" if self.lifecycle_ack["status"] == "terminal" else projected_abort_kind(simulator_abort_reason) if simulator_abort_reason is not None else "aborted"
            if end_kind != expected or self.lifecycle_ack["result"] != outcome:
                raise ContractError("lifecycle acknowledgement/end mismatch")
        if end_kind != "natural" and final_observation is not None:
            raise ContractError("interrupted journal cannot have final observation")
        reward = None
        if end_kind == "natural":
            reward = _reward(outcome, self.viewer)
            if (final_observation is None or observation(final_observation, self.viewer)["phase"] != "terminal"
                    or final_observation["result"] != outcome or self.pending):
                raise ContractError("natural end requires observed participant final state")
            if self.last_accepted is not None:
                self.write({"kind": "transition", "decision": self.last_accepted, "next_decision": None,
                            "reward": reward, "terminal": True})
        try:
            self.write({"kind": "end", "end_kind": end_kind, "outcome": outcome if end_kind == "natural" else None,
                        "terminal_reward": reward, "final_observation": final_observation,
                        "pending_decisions": sorted(self.pending), "simulator_abort_reason": simulator_abort_reason})
            self.output.flush()
            os.fsync(self.output.fileno())
            owned = os.fstat(self.output.fileno())
            current = self.path.lstat()
            if (not stat.S_ISREG(owned.st_mode) or (owned.st_dev, owned.st_ino) != self._owned_inode
                    or not stat.S_ISREG(current.st_mode) or _identity(current) != _identity(owned)):
                raise ContractError("journal5 pending path differs from owned file")
            expected_hash = _bounded_hash(cast(BinaryIO, self.output.buffer), owned.st_size)
            if _identity(os.fstat(self.output.fileno())) != _identity(owned):
                raise ContractError("journal5 pending file changed during verification")
            # Existing final objects remain evidence even if they appear during
            # the end-row write. Keep the owned pending FD through publication.
            os.link(self.path, self.final_path, follow_symlinks=False)
            with _open_regular(self.final_path, owned.st_size) as published:
                before = os.fstat(published.fileno())
                if ((before.st_dev, before.st_ino) != self._owned_inode or before.st_mtime_ns != owned.st_mtime_ns
                        or _bounded_hash(published, owned.st_size) != expected_hash):
                    raise ContractError("journal5 published object/content differs from owned file")
                after = os.fstat(published.fileno())
                final, pending = self.final_path.lstat(), self.path.lstat()
                if (_identity(after) != _identity(before) or not stat.S_ISREG(final.st_mode)
                        or _identity(final) != _identity(before) or not stat.S_ISREG(pending.st_mode)
                        or _identity(pending) != _identity(before)):
                    raise ContractError("journal5 publication changed during verification")
            self.path.unlink()
        finally:
            self.output.close()


@dataclass(frozen=True)
class StructuredDecision:
    frame: DecisionFrame
    generated: GeneratedCandidates
    selection: Selection
    command: Json
    timings: Json


@dataclass(frozen=True)
class StructuredJournalData:
    header: Json
    records: tuple[Json, ...]
    bank: EventBankView
    decisions: tuple[StructuredDecision, ...]
    accepted: tuple[int, ...]
    end: Json
    sidecars: tuple[Json, ...]


def _command(command: Json, frame: DecisionFrame, action: Json) -> None:
    closed(command, {"command_id", "expected_revision", "action"})
    try:
        if str(uuid.UUID(command["command_id"])) != command["command_id"]:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError("journal5 canonical command UUID required") from exc
    integer(command["expected_revision"])
    if command["expected_revision"] != frame.revision or canonical(command["action"]) != canonical(action):
        raise ContractError("journal5 command not reconstructed from original frame")


def _host_timings(value: Any) -> None:
    if not isinstance(value, dict) or any(type(item) not in (int, float) or item < 0 or item > sys.float_info.max or not math.isfinite(item) for item in value.values()):
        raise ContractError("finite nonnegative journal5 host timings required")


def read_structured_journal(path: Path, *, deadline: float = float("inf"),
                            records: list[Json] | None = None, max_jsonl_bytes: int = 64 * 1024**2,
                            max_record_bytes: int = 8 * 1024**2, action_contract: ActionContract,
                            purpose: str, synthetic_capacity: bool) -> StructuredJournalData:
    """Return structural component records/evidence, NOT source eligibility.

    Whole-run, Replay, split and whole-PPO-parent qualification remain mandatory
    in the source layer; this result alone authorizes no training or evaluation.
    """
    classification = input_binding(action_contract, purpose, synthetic_capacity)
    if synthetic_capacity:
        raise ContractError("journal5 game proposals reject synthetic capacity")
    import json

    from gf_models.structured_runtime import WorkerCaps
    if records is None:
        integer(max_jsonl_bytes, 1)
        integer(max_record_bytes, 1, max_jsonl_bytes)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_jsonl_bytes:
                raise ContractError("journal5 regular bounded file required")
            records = []
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("journal5 read deadline")
                line = source.readline(max_record_bytes + 1)
                if not line:
                    break
                if len(line) > max_record_bytes:
                    raise ContractError("journal5 JSONL record exceeds configured bound")
                records.append(json.loads(line))
            after = os.fstat(source.fileno())
            if any(getattr(before, field) != getattr(after, field) for field in
                   ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                raise ContractError("journal5 file changed while reading")
    if not records or records[0].get("kind") != "header":
        raise ContractError("journal5 header required")
    header = records[0]
    closed(header, {"kind", "schema_version", "feature_version", "input_recipe", "reward_version", "policy_input_version",
                    "end_projection_version", "catalog", "episode_id", "viewer", "policy", "run_id", "game_id",
                    "action_config_hash", "generator_version", "bank_episode", "caps", "numeric_costs", "model_id",
                    "model_version", "generation_schema", "aggregate_window", "token_signature", "sampling_seed", "use_action_options"} | set(classification))
    check_input_binding({key: header[key] for key in classification}, action_contract=action_contract,
                        purpose=purpose, synthetic_capacity=synthetic_capacity)
    if (type(header["schema_version"]) is not int or header["schema_version"] != 5
            or header["feature_version"] != STRUCTURED_FEATURE_VERSION or header["input_recipe"] != STRUCTURED_FEATURE_VERSION
            or header["reward_version"] != REWARD_VERSION or header["policy_input_version"] != POLICY_INPUT_VERSION
            or header["end_projection_version"] != END_PROJECTION_VERSION or canonical(header["catalog"]) != canonical(CATALOG_META)
            or header["generator_version"] != GENERATOR_V2 or header["use_action_options"] is not True):
        raise ContractError("journal5 semantic identity mismatch")
    recorded_aggregate_recipe(header['token_signature'], header['aggregate_window'])
    integer(header["viewer"], 1, 2)
    integer(header["sampling_seed"])
    for key in ("episode_id", "policy", "run_id", "game_id", "action_config_hash", "bank_episode"):
        if type(header[key]) is not str or not header[key]:
            raise ContractError("journal5 identity string required")
    caps = WorkerCaps(**closed(header["caps"], set(WorkerCaps.__dataclass_fields__)))
    NumericCosts(**closed(header["numeric_costs"], set(NumericCosts.__dataclass_fields__)))
    if header["model_id"] not in ("random", "heuristic", "mlp", "gru", "transformer", "lightgbm") or header["model_version"] not in ("1", "2"):
        raise ContractError("unknown journal5 policy model")
    if header["model_version"] == "2" and header["model_id"] == "lightgbm":
        if header["generation_schema"] is not None:
            raise ContractError("journal5 Tree2 is non-PPO and requires null generation schema")
    elif header["model_version"] == "2":
        if type(header["generation_schema"]) is not int or header["generation_schema"] != 2 or header["model_id"] not in ("mlp", "gru", "transformer"):
            raise ContractError("journal5 neural generation schema must be integer2")
    elif header["generation_schema"] is not None or header["model_id"] not in ("random", "heuristic"):
        raise ContractError("journal5 baseline generation schema must be None")
    integer(header["aggregate_window"], 1, min(65536, caps.max_episode_events))
    if re.fullmatch(r"[0-9a-f]{64}", header["action_config_hash"]) is None:
        raise ContractError("journal5 action hash required")
    bank = EventBank(header["viewer"], header["bank_episode"], caps.max_episode_events)
    search_seed = int(digest({"recipe": "structured-search-stream-v1", "sampling_seed": header["sampling_seed"]})[:16], 16)
    expected_search_state = random.Random(search_seed).getstate()
    decisions: list[StructuredDecision] = []
    accepted: list[int] = []
    inventories: list[Json] = []
    pending = last_accepted = awaiting_next = None
    terminal_transition = lifecycle = end = None
    last_frame = None
    delta_pending = False
    commands: dict[int, Json] = {}
    for row in records[1:]:
        if time.monotonic() >= deadline:
            raise TimeoutError("journal5 projection deadline")
        if end is not None:
            raise ContractError("records after journal5 end")
        kind = row.get("kind")
        if kind == "events":
            if caps.max_decisions is not None:
                integer(len(decisions) + 1, 1, caps.max_decisions)
            closed(row, {"kind", "delta", "boundary"})
            if pending is not None or delta_pending or lifecycle is not None:
                raise ContractError("journal5 event delta ordering mismatch")
            delta = parse_delta(row["delta"])
            if len(delta._events) > caps.max_interval_events:
                raise ContractError("journal5 event interval cap exceeded")
            boundary = bank.append(delta)
            if canonical(row["boundary"]) != canonical({"end": boundary.end, "prefix_hash": boundary.prefix_hash}):
                raise ContractError("journal5 delta boundary mismatch")
            delta_pending = True
        elif kind == "decision":
            closed(row, {"kind", "decision", "frame", "table", "search_report", "selection", "command", "timings"})
            index = row["decision"]
            integer(index, 0)
            if caps.max_decisions is not None:
                integer(index + 1, 1, caps.max_decisions)
            if not isinstance(row["table"], dict):
                raise ContractError("journal5 compact table descriptor required")
            integer(row["table"].get("count"), 1, caps.max_candidates)
            selection = parse_selection(row["selection"])
            if header["model_id"] == "lightgbm" and (selection.logprob is not None or selection.value is not None):
                raise ContractError("journal5 Tree2 policy has no policy logprob/critic value")
            if header["model_id"] in ("mlp", "gru", "transformer") and (selection.logprob is None or selection.value is None):
                raise ContractError("journal5 neural selection requires finite logprob/value")
            if index != len(decisions) or pending is not None or last_accepted is not None or terminal_transition is not None or lifecycle is not None or not delta_pending:
                raise ContractError("journal5 decision ordering mismatch")
            if awaiting_next is not None and awaiting_next != index:
                raise ContractError("journal5 transition/next mismatch")
            awaiting_next = None
            frame = parse_frame(row["frame"])
            integer(frame.boundary.end, 0, caps.max_episode_events)
            integer(frame.boundary.end - frame.interval_start, 0, caps.max_interval_events)
            if frame.boundary != bank.boundary:
                raise ContractError("journal5 current frame/delta boundary mismatch")
            frame.bank_view(bank.view())
            if frame.interval_start != (0 if last_frame is None else last_frame.boundary.end) or last_frame is not None and frame.revision <= last_frame.revision:
                raise ContractError("journal5 frame interval/revision mismatch")
            table = read_candidates(path.parent, row["table"], frame, index, max_positions=caps.max_positions, action_contract=action_contract,
                                    purpose=purpose, synthetic_capacity=False)
            report = parse_report(row["search_report"], action_contract=action_contract)
            generated = GeneratedCandidates(table, report)
            if report.rng_before != expected_search_state:
                raise ContractError("journal5 search RNG stream discontinuity")
            expected_search_state = report.rng_after
            action = selection.validate(table, frame)
            if (selection.policy_version != header["policy"] or report.config_hash != header["action_config_hash"]
                    or report.numeric_costs_hash != digest(header["numeric_costs"])):
                raise ContractError("journal5 selection/report identity mismatch")
            command = row["command"]
            _command(command, frame, action)
            _host_timings(row["timings"])
            commands[index] = command
            decisions.append(StructuredDecision(frame, generated, selection, command, row["timings"]))
            inventories.append(sidecar_inventory(path.parent, row["table"], frame, index, max_positions=caps.max_positions, action_contract=action_contract,
                                    purpose=purpose, synthetic_capacity=False))
            pending, last_frame, delta_pending = index, frame, False
        elif kind == "acknowledgement":
            closed(row, {"kind", "decision", "response"})
            if pending is None or row["decision"] != pending:
                raise ContractError("journal5 acknowledgement without pending decision")
            integer(row["decision"], 0)
            _acknowledgement(row["response"], commands[pending])
            accepted.append(pending)
            last_accepted, pending = pending, None
            if row["response"]["status"] != "active":
                lifecycle = row["response"]
        elif kind == "transition":
            closed(row, {"kind", "decision", "next_decision", "reward", "terminal"})
            integer(row["decision"], 0)
            integer(row["reward"], -1, 1)
            if row["next_decision"] is not None:
                integer(row["next_decision"], 0)
            if pending is not None or last_accepted is None or row["decision"] != last_accepted or type(row["terminal"]) is not bool:
                raise ContractError("journal5 transition lacks positive accepted action")
            if row["terminal"]:
                if row["next_decision"] is not None:
                    raise ContractError("journal5 terminal next pointer")
                terminal_transition = row
            else:
                if row["reward"] != 0 or row["next_decision"] != len(decisions):
                    raise ContractError("journal5 intermediate transition mismatch")
                awaiting_next = row["next_decision"]
            last_accepted = None
        elif kind == "end":
            closed(row, {"kind", "end_kind", "outcome", "terminal_reward", "final_observation", "pending_decisions", "simulator_abort_reason"})
            if row["end_kind"] not in END_KINDS:
                raise ContractError("invalid journal5 end kind")
            if row["pending_decisions"] != ([] if pending is None else [pending]) or awaiting_next is not None or delta_pending:
                raise ContractError("journal5 end pending/transition mismatch")
            if row["simulator_abort_reason"] is not None and row["end_kind"] != projected_abort_kind(row["simulator_abort_reason"]):
                raise ContractError("journal5 abort classification mismatch")
            if row["end_kind"] == "natural":
                result(row["outcome"])
                reward = _reward(row["outcome"], header["viewer"])
                integer(row["terminal_reward"], -1, 1)
                final = observation(row["final_observation"], header["viewer"])
                if pending is not None or final["phase"] != "terminal" or final["result"] != row["outcome"] or row["terminal_reward"] != reward:
                    raise ContractError("journal5 natural closure lacks observed participant outcome")
                if accepted and (terminal_transition is None or terminal_transition["decision"] != accepted[-1] or terminal_transition["reward"] != reward):
                    raise ContractError("journal5 final accepted action lacks natural reward")
            elif row["outcome"] is not None or row["terminal_reward"] is not None or row["final_observation"] is not None or terminal_transition is not None:
                raise ContractError("interrupted journal5 invented natural reward/state")
            if lifecycle is not None:
                expected = "natural" if lifecycle["status"] == "terminal" else projected_abort_kind(row["simulator_abort_reason"]) if row["simulator_abort_reason"] is not None else "aborted"
                if row["end_kind"] != expected or row["outcome"] != lifecycle["result"]:
                    raise ContractError("journal5 lifecycle/end mismatch")
            end = row
        else:
            raise ContractError("unknown journal5 record kind")
    if end is None:
        raise ContractError("journal5 final closure required")
    return StructuredJournalData(header, tuple(records), bank.view(), tuple(decisions), tuple(accepted), end, tuple(inventories))
