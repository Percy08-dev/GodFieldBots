"""Participant-safe values. No simulator, storage, or model dependencies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

Json = dict[str, Any]
SIMULATOR_REVISION = "ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6"
RULE_SET = "true-match-1v1-v1"
POLICY_INPUT_VERSION = "player-atomic-options-v1"
REVIEWED_SPEC_HASHES = {
    "docs/spec/00-specification-overview.md": "a2ef3bbf2d8a1e88300d94f75bccfd1956a9b0ea18c14dce2dc0c1c41845a03d",
    "docs/spec/01-server-spec.md": "0833200db8682a42f2d13b9759457c014f1e460fea7fcf321aa75f447ffecee8",
    "docs/spec/03-game-engine.md": "2756b6c9524c97600568e823a71ad028298be8d9446e66043ead5bb1f77ce205",
    "docs/spec/05-observation.md": "8d8326c1eed4a8d2f81c822f1ed6c6b2782dfec309f1efbcc250d3c4f727cda9",
    "docs/spec/06-persistence-history.md": "21135b53299cfcf4da5bf0ddbaf00379e0cba117310518b5cd5adda95b28c2ea",
    "docs/spec/07-external-interfaces-adapters.md": "342a8d59cc33494c9fbc91e565c7f710578cbf8a0fa59ee373fa0cea533d2d62",
    "docs/spec/08-test-requirements.md": "2d3f41349ff796e27590a6dd70b469fb2d547134ff5d659aba770fb02cb9a731",
    "docs/spec/09-gameplay-rules.md": "156c73425aab90c362a35b6393bc0e673dcac9de1e0979e8ae83ed5a0d370948",
    "docs/spec/10-cli-client.md": "31a0ad7fcc310d987d54bef1e6805aff3fdb31cb28e18b563f715206eb06588b",
}

END_PROJECTION_VERSION = "simulator-closure-v1"
CATALOG_HASH = "481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d"
CATALOG_META: Json = {
    "catalog_schema_version": 1, "catalog_version": "public-ja-v2",
    "rule_set_version": RULE_SET, "locale": "ja", "content_sha256": CATALOG_HASH,
    "definition_count": 296,
}


class ContractError(ValueError):
    pass


class UnsupportedContract(ContractError):
    pass


class EmptyCandidates(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def clone(value: Any) -> Any:
    return json.loads(canonical(value))


def closed(value: Any, required: set[str], optional: set[str] | None = None) -> Json:
    if not isinstance(value, dict) or set(value) - required - (optional or set()) or required - set(value):
        raise ContractError(f"invalid fields; required={sorted(required)}")
    return value


def integer(value: Any, low: int = 0, high: int = 2**64 - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ContractError(f"integer outside {low}..{high}")
    return value


@dataclass(frozen=True)
class DecisionInput:
    observation: Json
    events: tuple[Json, ...]
    history_end: str | None
    typed_metadata: tuple[Json, ...]
    catalog: Json
    action_options: Json | None = None

    def to_dict(self) -> Json:
        return clone(asdict(self))


@dataclass(frozen=True)
class ActionConfig:
    use_action_options: bool = True
    strategy: str = "beam"
    scorer: str = "public_heuristic_v1"
    beam_width: int = 64
    max_candidates: int = 512
    max_expansions: int = 20_000
    max_sequence_length: int = 3
    exchange_allocations: tuple[str, ...] = ("balanced", "hp", "mp", "money")
    targets: tuple[str, ...] = ("opponent", "self")


@dataclass(frozen=True)
class CandidateSet:
    actions: tuple[Json, ...]
    generator_version: str
    scorer_version: str
    config_hash: str
    attempts: int
    public_prunes: int
    approximate_drops: int
    latency_ms: float
    cutoff: bool
    cutoff_reasons: tuple[str, ...] = ()

    def to_dict(self) -> Json:
        return clone(asdict(self))


@dataclass(frozen=True)
class PolicyDecision:
    index: int
    policy_version: str
    logprob: float | None = None
    value: float | None = None


class CandidateGenerator(Protocol):
    def generate(self, decision_input: DecisionInput) -> CandidateSet: ...


class Policy(Protocol):
    def reset(self, sampling_seed: int) -> None: ...
    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision: ...


def validated_policy_options(value: Any) -> Json:
    import re
    options = closed(value, {"checkpoint", "expected_manifest_digest"}, {"allow_toy"})
    if not isinstance(options["checkpoint"], str) or not options["checkpoint"] or not isinstance(options["expected_manifest_digest"], str) or re.fullmatch(r"[0-9a-f]{64}", options["expected_manifest_digest"]) is None:
        raise ContractError("invalid safe checkpoint binding")
    if "allow_toy" in options and type(options["allow_toy"]) is not bool:
        raise ContractError("invalid direct toy permission")
    return clone(options)


@dataclass(frozen=True)
class RuntimePolicyBinding:
    model_id: str
    policy_options: Json | None = None


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    policies: tuple[str, str]
    sampling_seeds: tuple[int, int]
    pair_id: str
    learner_viewer: int | None = None


@dataclass(frozen=True)
class EpisodeGroup:
    seed: str
    episodes: tuple[EpisodeSpec, ...]
    ancestry: str | None = None


@dataclass
class EpisodeResult:
    episode_id: str
    pair_id: str
    seed: str
    policies: tuple[str, str]
    game_id: str | None = None
    outcome: Any = None
    end_kind: str = "incomplete"
    simulator_abort_reason: str | None = None
    failure: str | None = None
    health: dict[str, str | None] = field(default_factory=dict)
    replay_reconciled: bool = False
    first_player: int | None = None
    decisions: int = 0
    elapsed_seconds: float = 0.0
    candidate_counts: list[int] = field(default_factory=list)
    search_latency_ms: list[float] = field(default_factory=list)
    inference_latency_ms: list[float] = field(default_factory=list)
    inference_latency_by_policy: dict[str, list[float]] = field(default_factory=dict)
    search_attempts: list[int] = field(default_factory=list)


@dataclass
class GroupResult:
    episodes: list[EpisodeResult]
    health: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    async def run_group(self, group: EpisodeGroup) -> GroupResult: ...


def group_health_failures(value: Json) -> tuple[str, ...]:
    failures = [key for key in ("failure", "client_close_failure", "budget_overrun_seconds", "finalization_cancelled", "cleanup_failure") if value.get(key)]
    process = value.get("process")
    if process is not None:
        if process.get("exit_code") != 0:
            failures.append("process_exit")
        if process.get("shutdown") != "stdin_eof":
            failures.append("process_shutdown")
        failures.extend("process_" + key for key in ("log_error", "reap_failure", "budget_overrun_seconds") if process.get(key))
    return tuple(failures)
