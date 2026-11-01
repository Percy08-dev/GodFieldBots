"""Append-only participant evidence; run management belongs in a separate manifest."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.actions import GENERATOR_VERSION, SCORER_VERSION
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    CandidateSet,
    ContractError,
    DecisionInput,
    Json,
    PolicyDecision,
    canonical,
    closed,
    digest,
    group_health_failures,
)
from gf_common.validation import (
    observation,
    projected_abort_kind,
    reconcile_replay,
    replay_closure,
    result,
    validate_worker_input,
)

JOURNAL_SCHEMA = 4
FEATURE_VERSION = "participant-json-v2"
REWARD_VERSION = "natural-terminal-v1"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(canonical(value) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


class ParticipantJournal:
    def __init__(self, directory: Path, episode_id: str, viewer: int, policy: str, *,
                 run_id: str, game_id: str, action_config_hash: str, use_action_options: bool = True) -> None:
        if type(use_action_options) is not bool:
            raise ContractError("invalid journal hints mode")
        self.final_path = directory / f"player-{viewer}.jsonl"
        self.path = directory / f"player-{viewer}.partial.jsonl"
        self.output = self.path.open("x", encoding="utf-8")
        self.use_action_options = use_action_options
        self.viewer = viewer
        self.count = 0
        self.pending: set[int] = set()
        self.commands: dict[int, Json] = {}
        self.lifecycle_ack: Json | None = None
        self.last_accepted: int | None = None
        self.write({"kind": "header", "schema_version": JOURNAL_SCHEMA, "feature_version": FEATURE_VERSION,
                    "reward_version": REWARD_VERSION, "policy_input_version": POLICY_INPUT_VERSION,
                    "use_action_options": use_action_options, "end_projection_version": END_PROJECTION_VERSION, "catalog": CATALOG_META,
                    "episode_id": episode_id, "viewer": viewer, "policy": policy, "run_id": run_id,
                    "game_id": game_id, "action_config_hash": action_config_hash,
                    "generator_version": GENERATOR_VERSION, "scorer_version": SCORER_VERSION})

    def write(self, record: Json) -> None:
        self.output.write(canonical(record) + "\n")
        self.output.flush()

    def begin(self, value: DecisionInput, candidates: CandidateSet, selection: PolicyDecision,
              command: Json, elapsed_ms: float) -> int:
        validate_worker_input(value, self.use_action_options)
        if self.lifecycle_ack is not None:
            raise ContractError("decision after lifecycle acknowledgement")
        if value.observation["viewer_player_id"] != self.viewer:
            raise ContractError("journal viewer mismatch")
        index = self.count
        if self.last_accepted is not None:
            self.write({"kind": "transition", "decision": self.last_accepted, "next_decision": index,
                        "reward": 0, "terminal": False})
            self.last_accepted = None
        self.write({"kind": "decision", "decision": index, "input": value.to_dict(),
                    "candidates": candidates.to_dict(), "selection": asdict(selection),
                    "command": command, "selection_elapsed_ms": elapsed_ms})
        self.pending.add(index)
        self.commands[index] = command
        self.count += 1
        return index

    def acknowledge(self, index: int, acknowledgement: Json) -> None:
        if index not in self.pending:
            raise ContractError("acknowledgement without pending decision")
        _acknowledgement(acknowledgement, self.commands[index])
        self.write({"kind": "acknowledgement", "decision": index, "response": acknowledgement})
        self.pending.remove(index)
        self.last_accepted = index
        if acknowledgement["status"] != "active":
            self.lifecycle_ack = acknowledgement

    def finish(self, end_kind: str, outcome: Any, final_observation: Json | None = None,
               simulator_abort_reason: str | None = None) -> None:
        if simulator_abort_reason is not None and end_kind != projected_abort_kind(simulator_abort_reason):
            raise ContractError("simulator abort cause/projected kind mismatch")
        if self.lifecycle_ack is not None:
            expected_end = "natural" if self.lifecycle_ack["status"] == "terminal" else projected_abort_kind(simulator_abort_reason) if simulator_abort_reason is not None else "aborted"
            if end_kind != expected_end or self.lifecycle_ack["result"] != outcome:
                raise ContractError("lifecycle acknowledgement/end mismatch")
        if end_kind != "natural" and final_observation is not None:
            raise ContractError("interrupted journal cannot have final observation")
        reward: int | None = None
        if end_kind == "natural":
            if outcome == "draw":
                reward = 0
            elif isinstance(outcome, dict) and "win" in outcome:
                reward = 1 if outcome["win"]["winner"] == self.viewer else -1
            else:
                raise ContractError("natural end requires outcome")
            if final_observation is None or observation(final_observation, self.viewer)["phase"] != "terminal" or final_observation["result"] != outcome or self.pending:
                raise ContractError("natural end requires observed participant final state")
            if self.last_accepted is not None:
                self.write({"kind": "transition", "decision": self.last_accepted, "next_decision": None,
                            "reward": reward, "terminal": True})
        self.write({"kind": "end", "end_kind": end_kind, "outcome": outcome if end_kind == "natural" else None,
                    "terminal_reward": reward, "final_observation": final_observation,
                    "pending_decisions": sorted(self.pending), "simulator_abort_reason": simulator_abort_reason})
        self.output.flush()
        os.fsync(self.output.fileno())
        self.output.close()
        self.path.replace(self.final_path)

    def close_incomplete(self) -> None:
        if not self.output.closed:
            try:
                self.output.flush()
            finally:
                self.output.close()


def _reward(outcome: Any, viewer: int) -> int:
    result(outcome)
    if outcome is None:
        raise ContractError("missing natural result")
    return 0 if outcome == "draw" else 1 if outcome["win"]["winner"] == viewer else -1


def _acknowledgement(response: Any, command: Json) -> None:
    from gf_common.domain import integer
    closed(response, {"command_id", "previous_revision", "state_revision", "status", "result"})
    if response["command_id"] != command["command_id"]:
        raise ContractError("acknowledgement command mismatch")
    integer(response["previous_revision"])
    integer(response["state_revision"])
    if response["previous_revision"] != command["expected_revision"] or response["state_revision"] != command["expected_revision"] + 1:
        raise ContractError("acknowledgement revision mismatch")
    if response["status"] not in ("active", "terminal", "aborted"):
        raise ContractError("invalid accepted acknowledgement status")
    result(response["result"])
    if (response["status"] == "terminal") != (response["result"] is not None):
        raise ContractError("acknowledgement outcome/status mismatch")


def _projection_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("journal projection deadline")


def _projection_hash(path: Path, deadline: float) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            _projection_deadline(deadline)
            hasher.update(chunk)
    _projection_deadline(deadline)
    return hasher.hexdigest()


def read_journal(path: Path, *, deadline: float = float("inf"), records: list[Json] | None = None,
                 typed_decisions: dict[int, DecisionInput] | None = None) -> list[Json]:
    from gf_common.domain import integer
    if records is None:
        records = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                _projection_deadline(deadline)
                records.append(json.loads(line))
    if not records or records[0].get("kind") != "header":
        raise ContractError("missing journal header")
    header = records[0]
    if type(header.get("schema_version")) is not int or header["schema_version"] != JOURNAL_SCHEMA:
        raise ContractError("incompatible journal versions (schema 1, schema 2 and schema 3 are not reinterpreted)")
    closed(header, {"kind", "schema_version", "feature_version", "reward_version", "catalog", "episode_id", "viewer", "policy", "run_id", "game_id", "action_config_hash", "generator_version", "scorer_version", "end_projection_version", "policy_input_version", "use_action_options"})
    if type(header["schema_version"]) is not int or header["schema_version"] != JOURNAL_SCHEMA or header["feature_version"] != FEATURE_VERSION or header["reward_version"] != REWARD_VERSION or header["end_projection_version"] != END_PROJECTION_VERSION or canonical(header["catalog"]) != canonical(CATALOG_META):
        raise ContractError("incompatible journal versions (schema 1, schema 2 and schema 3 are not reinterpreted)")
    if header["policy_input_version"] != POLICY_INPUT_VERSION or type(header["use_action_options"]) is not bool:
        raise ContractError("incompatible policy input version/mode")
    if header["generator_version"] != GENERATOR_VERSION or header["scorer_version"] != SCORER_VERSION:
        raise ContractError("incompatible search versions")
    integer(header["viewer"], 1, 2)
    if not all(isinstance(header[k], str) and header[k] for k in ("episode_id", "policy", "run_id", "game_id", "action_config_hash")):
        raise ContractError("invalid journal identity")
    decisions: dict[int, Json] = {}
    accepted: list[int] = []
    pending: int | None = None
    last_accepted: int | None = None
    awaiting_next: int | None = None
    terminal_transition: Json | None = None
    ended = False
    last_revision = 0
    lifecycle_ack: Json | None = None
    for record in records[1:]:
        _projection_deadline(deadline)
        if ended:
            raise ContractError("records after journal end")
        kind = record.get("kind")
        if kind == "decision":
            closed(record, {"kind", "decision", "input", "candidates", "selection", "command", "selection_elapsed_ms"})
            index = record["decision"]
            integer(index)
            if index != len(decisions) or pending is not None or last_accepted is not None or terminal_transition is not None or lifecycle_ack is not None:
                raise ContractError("invalid decision ordering")
            if awaiting_next is not None and awaiting_next != index:
                raise ContractError("transition/next decision mismatch")
            awaiting_next = None
            raw = record["input"]
            closed(raw, {"observation", "events", "history_end", "typed_metadata", "catalog", "action_options"})
            value = validate_worker_input(DecisionInput(raw["observation"], tuple(raw["events"]), raw["history_end"], tuple(raw["typed_metadata"]), raw["catalog"], raw["action_options"]), header["use_action_options"])
            if value.observation["viewer_player_id"] != header["viewer"]:
                raise ContractError("mixed participant journal")
            candidates = record["candidates"]
            closed(candidates, {"actions", "generator_version", "scorer_version", "config_hash", "attempts", "public_prunes", "approximate_drops", "latency_ms", "cutoff", "cutoff_reasons"})
            if candidates["generator_version"] != header["generator_version"] or candidates["scorer_version"] != header["scorer_version"] or candidates["config_hash"] != header["action_config_hash"]:
                raise ContractError("incompatible candidate versions/config")
            selection = record["selection"]
            closed(selection, {"index", "policy_version", "logprob", "value"})
            integer(selection["index"])
            if not candidates["actions"] or selection["index"] >= len(candidates["actions"]):
                raise ContractError("invalid stored candidate selection")
            for field in ("logprob", "value"):
                number = selection[field]
                if number is not None and (type(number) not in (int, float) or not math.isfinite(number)):
                    raise ContractError("invalid policy numeric output")
            command = closed(record["command"], {"command_id", "expected_revision", "action"})
            try:
                if str(uuid.UUID(command["command_id"])) != command["command_id"]:
                    raise ValueError
            except (ValueError, TypeError, AttributeError) as exc:
                raise ContractError("invalid command UUID") from exc
            integer(command["expected_revision"])
            if command["expected_revision"] < last_revision:
                raise ContractError("participant revision moved backwards")
            if command["expected_revision"] != value.observation["state_revision"] or command["action"] != candidates["actions"][selection["index"]]:
                raise ContractError("command/input/candidate mismatch")
            if typed_decisions is not None:
                typed_decisions[index] = value
            decisions[index] = record
            pending = index
        elif kind == "acknowledgement":
            closed(record, {"kind", "decision", "response"})
            index = record["decision"]
            integer(index)
            if index != pending:
                raise ContractError("acknowledgement without unique pending decision")
            _acknowledgement(record["response"], decisions[index]["command"])
            if record["response"]["status"] != "active":
                lifecycle_ack = record["response"]
            last_revision = record["response"]["state_revision"]
            accepted.append(index)
            pending = None
            last_accepted = index
        elif kind == "transition":
            closed(record, {"kind", "decision", "next_decision", "reward", "terminal"})
            index = record["decision"]
            integer(index)
            if index != last_accepted or terminal_transition is not None or type(record["terminal"]) is not bool:
                raise ContractError("transition lacks last accepted action")
            if record["terminal"]:
                integer(record["reward"], -1, 1)
                if record["next_decision"] is not None:
                    raise ContractError("terminal transition has next decision")
                terminal_transition = record
            else:
                if lifecycle_ack is not None:
                    raise ContractError("continuing transition after lifecycle acknowledgement")
                if type(record["reward"]) is not int or record["reward"] != 0 or type(record["next_decision"]) is not int or record["next_decision"] != len(decisions):
                    raise ContractError("invalid continuing transition")
                awaiting_next = record["next_decision"]
            last_accepted = None
        elif kind == "end":
            closed(record, {"kind", "end_kind", "outcome", "terminal_reward", "final_observation", "pending_decisions", "simulator_abort_reason"})
            if record["end_kind"] not in ("natural", "truncated", "aborted", "cancelled", "incomplete") or awaiting_next is not None:
                raise ContractError("invalid journal end/order")
            if record["pending_decisions"] != ([] if pending is None else [pending]):
                raise ContractError("pending request ledger mismatch")
            reason = record["simulator_abort_reason"]
            if reason is not None and record["end_kind"] != projected_abort_kind(reason):
                raise ContractError("simulator abort cause/projected kind mismatch")
            if lifecycle_ack is not None:
                expected_end = "natural" if lifecycle_ack["status"] == "terminal" else projected_abort_kind(reason) if reason is not None else "aborted"
                if record["end_kind"] != expected_end or record["outcome"] != lifecycle_ack["result"]:
                    raise ContractError("lifecycle acknowledgement/end mismatch")
            if record["end_kind"] == "natural":
                reward = _reward(record["outcome"], header["viewer"])
                final = observation(record["final_observation"], header["viewer"])
                if pending is not None or final["phase"] != "terminal" or final["result"] != record["outcome"] or type(record["terminal_reward"]) is not int or record["terminal_reward"] != reward:
                    raise ContractError("natural result/observation/reward mismatch")
                if accepted and (terminal_transition is None or terminal_transition["decision"] != accepted[-1] or terminal_transition["reward"] != reward):
                    raise ContractError("terminal transition/perspective mismatch")
                if not accepted and terminal_transition is not None:
                    raise ContractError("terminal transition without accepted action")
            elif record["terminal_reward"] is not None or record["outcome"] is not None or terminal_transition is not None or record["final_observation"] is not None:
                raise ContractError("interrupted journal has terminal label")
            ended = True
        else:
            raise ContractError("unknown journal record")
    if terminal_transition is not None and not ended:
        raise ContractError("terminal transition lacks matching natural end")
    if awaiting_next is not None:
        raise ContractError("transition lacks recorded next decision")
    return records


def _manifest_source(source: Path, records: list[Json], *, online_context: Any = None, parent_cache: Any = None, locks: Any = None, deadline: float = float("inf"), session: Any = None) -> tuple[Json, Json, Json, Json]:
    manifest_path = source.parent.parent / "manifest.json"
    try:
        manifest = session.read_json(manifest_path) if session is not None else json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError("journal requires its finalized linked run manifest") from exc
    header = records[0]
    if header['episode_id'] != source.parent.name:
        raise ContractError('journal/directory episode identity mismatch')
    if manifest.get("schema_version") not in (3, 4) or type(manifest.get("schema_version")) is not int or manifest.get("status") not in ("complete", "incomplete", "interrupted"):
        raise ContractError("run manifest must be finalized schema 3")
    if manifest["schema_version"] == 4:
        from gf_training.lineage import validate_collection_parent
        if session is not None:
            if online_context is not None:
                raise ContractError("offline preparation cannot reuse online authorization")
            session.validate_collection(manifest, source.parent.parent)
        else:
            validate_collection_parent(manifest, source.parent.parent, online_context=online_context, parent_cache=parent_cache, locks=locks, deadline=deadline)
    if manifest.get("run_id") != header["run_id"] or manifest.get("journal_schema_version") != JOURNAL_SCHEMA or manifest.get("feature_version") != FEATURE_VERSION or manifest.get("reward_policy_version") != REWARD_VERSION or manifest.get("end_projection_version") != END_PROJECTION_VERSION or canonical(manifest.get("catalog")) != canonical(CATALOG_META):
        raise ContractError("journal/manifest identity or semantic versions mismatch")
    from gf_common.domain import SIMULATOR_REVISION
    if manifest.get("simulator_contract_revision") != SIMULATOR_REVISION or manifest.get("observation_schema_version") != "1" or type(manifest.get("action_schema_version")) is not int or manifest["action_schema_version"] != 1:
        raise ContractError("incompatible source simulator semantics")
    if manifest.get("policy_input_version") != POLICY_INPUT_VERSION or manifest.get("reviewed_simulator_spec_sha256") != REVIEWED_SPEC_HASHES:
        raise ContractError("incompatible policy input/spec snapshot provenance")
    config = manifest.get("effective_config", {})
    if digest(config) != manifest.get("config_sha256") or digest(config.get("actions")) != header["action_config_hash"]:
        raise ContractError("source configuration hash mismatch")
    if type(config.get("actions", {}).get("use_action_options")) is not bool or config["actions"]["use_action_options"] != header["use_action_options"]:
        raise ContractError("source ActionOptions mode mismatch")
    if config.get("actions", {}).get("scorer") != "public_heuristic_v1":
        raise ContractError("source search scorer mismatch")
    split = manifest.get("split")
    if split not in ("train", "validation", "test") or config.get("run", {}).get("split") != split:
        raise ContractError("source declared split mismatch")
    matching = [e for e in manifest.get("episodes", []) if e.get("episode_id") == header["episode_id"]]
    if len(matching) != 1:
        raise ContractError("journal episode missing or duplicated in manifest")
    episode = matching[0]
    if "simulator_abort_reason" not in episode:
        raise ContractError("missing projected abort cause")
    if episode.get("game_id") != header["game_id"] or episode.get("policies", [])[header["viewer"] - 1] != header["policy"]:
        raise ContractError("source game/model linkage mismatch")
    schedule_matches = [(group, spec) for group in manifest.get("schedule", []) for spec in group.get("episodes", []) if spec.get("episode_id") == header["episode_id"]]
    if len(schedule_matches) != 1:
        raise ContractError("source scheduled identity missing or duplicated")
    group, spec = schedule_matches[0]
    if episode.get("seed") != group.get("seed") or episode.get("pair_id") != spec.get("pair_id") or episode.get("policies") != spec.get("policies"):
        raise ContractError("source schedule/result mismatch")
    linked_group_health = [g for g in manifest.get("groups", []) if header["episode_id"] in g.get("episode_ids", [])]
    group_health = linked_group_health[0] if len(linked_group_health) == 1 else {"failure": "missing group health"}
    lineage = {"run_id": header["run_id"], "game_id": header["game_id"], "episode_id": header["episode_id"],
               "seed": group["seed"], "pair_id": spec["pair_id"], "ancestry": group.get("ancestry"), "split": split}
    if manifest["schema_version"] == 4:
        lineage["learner_viewer"] = spec["learner_viewer"]
    return manifest, episode, group_health, lineage




def _source_report(source: Path, manifest: Json, *, session: Any = None) -> tuple[Json, list[str]]:
    """A complete manifest alone cannot override a durably invalidated report."""
    path = source.parent.parent / "report.json"
    diagnostic: Json = {"path": str(path.resolve()), "expected_sha256": manifest.get("report_sha256"),
                        "sha256": None, "comparison_valid": None}
    reasons = []
    try:
        if session is not None:
            report = session.read_json(path)
            diagnostic["sha256"] = session.hash_file(path)
        else:
            encoded = path.read_bytes()
            diagnostic["sha256"] = hashlib.sha256(encoded).hexdigest()
            report = json.loads(encoded)
        if not isinstance(manifest.get("report_sha256"), str) or diagnostic["sha256"] != manifest["report_sha256"]:
            reasons.append("report_hash_mismatch")
        if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
            raise ContractError("invalid linked report schema")
        for field in ("failures", "health_failures", "group_health_failures"):
            if not isinstance(report.get(field), dict):
                raise ContractError("linked report missing failure/health ledger")
        diagnostic["comparison_valid"] = report.get("comparison_valid")
        diagnostic["failures"] = report["failures"]
        diagnostic["health_failures"] = report["health_failures"]
        diagnostic["group_health_failures"] = report["group_health_failures"]
        if report.get("comparison_valid") is not True or any(report[field] for field in ("failures", "health_failures", "group_health_failures")) or report.get("budget_overrun_seconds") or report.get("report_status") == "minimal":
            reasons.append("report_not_successful")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reasons.append("report_missing_or_malformed")
        diagnostic["error"] = str(exc)
    if any(manifest.get(field) for field in ("finalization_failure", "budget_overrun_seconds", "correction_save_failures")):
        reasons.append("run_finalization_failure")
    return diagnostic, reasons


@dataclass(frozen=True)
class AcceptedDecision:
    value: DecisionInput
    action: Json
    target: float
    identity: tuple[str, str, int, int]
    split: str


def _projection_sources(sources: list[Path], *, deadline: float, session: Any,
                        capture_typed: bool) -> Iterator[tuple[Path, list[Json], dict[int, DecisionInput], Any]]:
    ordered = sorted(sources, key=lambda path: str(path.resolve()))
    if session is None:
        for source in ordered:
            yield source, read_journal(source, deadline=deadline), {}, None
        return
    groups: dict[Path, list[Path]] = {}
    for source in ordered:
        groups.setdefault(source.parent.resolve(), []).append(source.resolve())
    for directory, journals in groups.items():
        # Offline source admission is always a complete two-view episode.
        if {path.name for path in journals} != {"player-1.jsonl", "player-2.jsonl"}:
            raise ContractError("preparation requires both participant journals per episode")
        paths = tuple(journals) + tuple(directory / f"replay-player-{viewer}.json" for viewer in (1, 2))
        with session.episode(paths) as payload:
            validated = []
            for source in journals:
                records = payload[source]
                typed: dict[int, DecisionInput] = {}
                if capture_typed:
                    manifest = session.read_json(source.parent.parent / "manifest.json")
                    specs = [spec for group in manifest.get("schedule", []) for spec in group.get("episodes", [])
                             if spec.get("episode_id") == records[0].get("episode_id")]
                    selected = len(specs) == 1 and specs[0].get("learner_viewer", records[0].get("viewer")) == records[0].get("viewer")
                else:
                    selected = False
                read_journal(source, deadline=deadline, records=records, typed_decisions=typed if selected else None)
                session.diagnostics["journal_validations"] += 1
                validated.append((source, records, typed, payload))
            session.settle_episode((payload, validated))
            yield from validated
            for _, records, typed, _ in validated:
                records.clear()
                typed.clear()
            validated.clear()


def _project_journals(sources: list[Path], *, deadline: float = float("inf"), online_context: Any = None,
                      parent_cache: Any = None, locks: Any = None, session: Any = None,
                      decision_sink: Callable[[AcceptedDecision], None] | None = None,
                      materialize: bool = True) -> tuple[list[Json], Json]:
    """Shared original-source validation/projection for export and training."""
    if not sources:
        raise ContractError("at least one linked participant journal required")
    from gf_models.sharded_arrays import read_json as bounded_source_json
    from gf_training.structured_source import reject_engineering_source
    for manifest_path in {source.parent.parent / "manifest.json" for source in sources}:
        try:
            original = session.read_json(manifest_path) if session is not None else bounded_source_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise ContractError("journal requires its finalized linked run manifest") from exc
        reject_engineering_source(original)
    rows = []
    source_metadata = []
    manifest_metadata: dict[str, Json] = {}
    identities: set[tuple[str, str, int]] = set()
    for source, records, typed, payload in _projection_sources(sources, deadline=deadline, session=session,
                                                              capture_typed=decision_sink is not None):
        _projection_deadline(deadline)
        header = records[0]
        identity = (header["run_id"], header["game_id"], header["viewer"])
        if identity in identities:
            raise ContractError("duplicate source participant identity")
        identities.add(identity)
        manifest, episode, group_health, lineage = _manifest_source(source, records, online_context=online_context, parent_cache=parent_cache, locks=locks, deadline=deadline, session=session)
        manifest_path = source.parent.parent / "manifest.json"
        manifest_hash = session.hash_file(manifest_path) if session is not None else _projection_hash(manifest_path, deadline)
        prior = manifest_metadata.get(header["run_id"])
        if prior is not None and prior["sha256"] != manifest_hash:
            raise ContractError("run identity refers to different manifests")
        manifest_metadata[header["run_id"]] = {
            "path": str(manifest_path.resolve()), "sha256": manifest_hash,
            "config_sha256": manifest["config_sha256"], "effective_config": manifest["effective_config"],
            "config_inputs": manifest.get("config_inputs"), "code": manifest.get("code"),
            "simulator": manifest.get("simulator"), "reviewed_simulator_spec_sha256": manifest["reviewed_simulator_spec_sha256"],
            "policy_input_version": POLICY_INPUT_VERSION, "simulator_contract_revision": manifest["simulator_contract_revision"],
            "binary_sha256": manifest.get("binary_sha256"), "binary_build_revision": manifest.get("binary_build_revision"),
            "status": manifest["status"], "split": manifest["split"], "end_projection_version": END_PROJECTION_VERSION,
        }
        acknowledgements = {r["decision"]: r for r in records if r["kind"] == "acknowledgement"}
        transitions = {r["decision"]: r for r in records if r["kind"] == "transition"}
        decisions = [r for r in records if r["kind"] == "decision"]
        if manifest["schema_version"] == 4 and header["viewer"] == lineage["learner_viewer"]:
            from gf_models.ppo_identity import selection_version
            expected_policy = selection_version(header["policy"])
            if any(record["selection"]["policy_version"] != expected_policy for record in decisions):
                raise ContractError("PPO learner selection differs from frozen inference identity")
        end = next((r for r in records if r["kind"] == "end"), None)
        report_diagnostic, reasons = _source_report(source, manifest, session=session)
        manifest_metadata[header["run_id"]]["report"] = report_diagnostic
        if manifest["status"] != "complete":
            reasons.append("run_not_complete")
        if end is None or end["end_kind"] != "natural" or episode.get("end_kind") != "natural":
            reasons.append("not_natural")
        if end is not None and (end["end_kind"] != episode.get("end_kind") or end["simulator_abort_reason"] != episode.get("simulator_abort_reason")):
            raise ContractError("journal/manifest end projection mismatch")
        if end is not None and end["end_kind"] == "natural" and end["outcome"] != episode.get("outcome"):
            raise ContractError("journal/manifest terminal outcome mismatch")
        if episode.get("failure") or any(episode.get("health", {}).values()) or group_health_failures(group_health):
            reasons.append("episode_or_group_health")
        if len(acknowledgements) != len(decisions) or end is None or end["pending_decisions"]:
            reasons.append("pending_acknowledgements")
        replay_path = source.parent / f"replay-player-{header['viewer']}.json"
        replay_hash = None
        try:
            if payload is not None:
                replay = payload[replay_path.resolve()]
            else:
                replay_bytes = replay_path.read_bytes()
                replay = json.loads(replay_bytes)
            if replay.get("game_id") != header["game_id"]:
                raise ContractError("replay game identity mismatch")
            accepted = [r["command"]["action"] for r in decisions if r["decision"] in acknowledgements]
            reconcile_replay(replay, header["viewer"], accepted, episode.get("outcome"))
            closure = replay_closure(replay, header["game_id"], header["viewer"])
            if episode.get("simulator_abort_reason") is not None:
                other_viewer = 3 - header["viewer"]
                other_replay = payload[(source.parent / f"replay-player-{other_viewer}.json").resolve()] if payload is not None else json.loads((source.parent / f"replay-player-{other_viewer}.json").read_bytes())
                other_closure = replay_closure(other_replay, header["game_id"], other_viewer)
                if closure != other_closure or closure is None or closure["type"] != "aborted" or closure["reason"] != episode["simulator_abort_reason"] or projected_abort_kind(closure["reason"]) != episode["end_kind"]:
                    raise ContractError("replay/manifest abort projection mismatch")
            elif closure is not None and closure["type"] == "aborted" and episode.get("replay_reconciled"):
                raise ContractError("reconciled abort is missing diagnostic cause")
            replay_hash = session.hash_file(replay_path) if session is not None else hashlib.sha256(replay_bytes).hexdigest()
            if not episode.get("replay_reconciled"):
                reasons.append("owner_replay_not_reconciled")
        except (OSError, ValueError, TypeError, KeyError):
            reasons.append("replay_mismatch_or_missing")
        eligible = not reasons
        source_metadata.append({"path": str(source.resolve()), "sha256": session.hash_file(source) if session is not None else _projection_hash(source, deadline),
                                "manifest_sha256": manifest_hash, "header": header, "lineage": lineage,
                                "decision_count": len(decisions), "end": end, "episode_result": episode,
                                "group_health": group_health, "replay_sha256": replay_hash, "report": report_diagnostic,
                                "eligible": eligible, "ineligible_reasons": sorted(set(reasons))})
        if session is not None:
            session.retain(source_metadata[-1])
        for record in decisions:
            _projection_deadline(deadline)
            index = record["decision"]
            if decision_sink is not None and index in typed and eligible and index in acknowledgements:
                assert end is not None
                decision_sink(AcceptedDecision(typed[index], record["command"]["action"],
                                              float(end["terminal_reward"]), (*identity, index), lineage["split"]))
            if not materialize:
                continue
            rows.append({"run_id": header["run_id"], "game_id": header["game_id"],
                         "episode_id": header["episode_id"], "viewer": header["viewer"], "decision": index,
                         "split": lineage["split"], "eligible": eligible,
                         "input_json": canonical(record["input"]), "candidate_set_json": canonical(record["candidates"]),
                         "selection_json": canonical(record["selection"]), "command_json": canonical(record["command"]),
                         "acknowledgement_json": canonical(acknowledgements[index]) if index in acknowledgements else None,
                         "transition_json": canonical(transitions[index]) if index in transitions else None})
    return rows, projection_metadata(source_metadata, manifest_metadata, deadline=deadline)


def projection_metadata(source_metadata: list[Json], manifest_metadata: dict[str, Json], *,
                        deadline: float = float("inf")) -> Json:
    """One global identity/split/proof recipe for streaming and legacy adapters."""
    from gf_training.preparation import canonical_digest as source_digest
    source_metadata = sorted(source_metadata, key=lambda source: source["path"])
    identities = [(source["header"]["run_id"], source["header"]["game_id"], source["header"]["viewer"])
                  for source in source_metadata]
    if len(set(identities)) != len(identities):
        raise ContractError("duplicate source participant identity")
    lineage_entries = [source["lineage"] for source in source_metadata]
    # Union all equal seeds and common fork ancestries across source runs before
    # checking declared splits. Repeated views and repeated runs stay together.
    parents: dict[str, str] = {}
    def find(item: str) -> str:
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]
    for entry in lineage_entries:
        _projection_deadline(deadline)
        keys = ["seed:" + entry["seed"]]
        if entry["ancestry"] is not None:
            keys.append("ancestry:" + entry["ancestry"])
        roots = sorted(find(key) for key in keys)
        for root in roots:
            parents[root] = roots[0]
    lineage_groups: dict[str, list[Json]] = {}
    for entry in lineage_entries:
        lineage_groups.setdefault(find("seed:" + entry["seed"]), []).append(entry)
    for entries in lineage_groups.values():
        if len({e["split"] for e in entries}) != 1:
            raise ContractError("equal-seed/common-ancestry sources have conflicting declared splits")
    metadata = {"policy_input_version": POLICY_INPUT_VERSION, "derived_schema_version": 4, "journal_schema_version": JOURNAL_SCHEMA, "end_projection_version": END_PROJECTION_VERSION,
                "feature_version": FEATURE_VERSION, "reward_version": REWARD_VERSION, "catalog": CATALOG_META,
                "sources": source_metadata, "manifests": manifest_metadata,
                "lineage_groups": [entries for _, entries in sorted(lineage_groups.items())],
                "source_set_sha256": source_digest(source_metadata, deadline=deadline),
                "rows": sum(source["decision_count"] for source in source_metadata),
                "eligible_rows": sum(source["decision_count"] for source in source_metadata if source["eligible"])}
    _projection_deadline(deadline)
    return metadata


def project_journals(sources: list[Path], *, deadline: float = float("inf"), online_context: Any = None) -> tuple[list[Json], Json]:
    from contextlib import ExitStack
    with ExitStack() as locks:
        return _project_journals(sources, deadline=deadline, online_context=online_context, parent_cache={}, locks=locks)


def export_parquet(sources: list[Path], destination: Path) -> Json:
    """Write exact validated participant projection and diagnostic sidecar."""
    rows, metadata = project_journals(sources)
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([*[(name, pa.string()) for name in ("run_id", "game_id", "episode_id")],
                        ("viewer", pa.int64()), ("decision", pa.int64()), ("split", pa.string()), ("eligible", pa.bool_()),
                        *[(name, pa.string()) for name in ("input_json", "candidate_set_json", "selection_json", "command_json", "acknowledgement_json", "transition_json")]])
    table = pa.Table.from_pylist(rows, schema=schema)
    # Management provenance is only in the sidecar, never feature-bearing rows
    # or Parquet schema metadata accessible to a feature loader.
    table = table.replace_schema_metadata({b"godfieldbots": canonical({"policy_input_version": POLICY_INPUT_VERSION, "derived_schema_version": 4, "end_projection_version": END_PROJECTION_VERSION, "feature_version": FEATURE_VERSION, "reward_version": REWARD_VERSION}).encode()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False, write_statistics=True)
    temporary.replace(destination)
    atomic_json(destination.with_suffix(destination.suffix + ".manifest.json"), metadata)
    return metadata


def assign_splits(episodes: list[Json], salt: str = "split-v1") -> dict[str, str]:
    """Union seed and fork ancestry before assigning; both views share episode ID."""
    parents: dict[str, str] = {}

    def find(item: str) -> str:
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]

    for episode in episodes:
        identities = ["seed:" + episode["seed"]]
        if episode.get("ancestry"):
            identities.append("ancestry:" + episode["ancestry"])
        roots = sorted(find(item) for item in identities)
        for root in roots:
            parents[root] = roots[0]
    groups: dict[str, list[str]] = {}
    for identity in parents:
        groups.setdefault(find(identity), []).append(identity)
    assignment = {}
    for episode in episodes:
        root = find("seed:" + episode["seed"])
        bucket = int(digest([salt, sorted(groups[root])])[:16], 16) % 100
        assignment[episode["episode_id"]] = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
    return assignment
