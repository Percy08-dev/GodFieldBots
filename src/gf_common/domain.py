"""Participant-safe values. No simulator, storage, or model dependencies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

Json = dict[str, Any]
SIMULATOR_REVISION = "44a26e3a16b808e826deb332fa72acfc5037d3ef"
RULE_SET = "true-match-1v1-v1"
POLICY_INPUT_VERSION = "player-atomic-options-v1"
REVIEWED_SPEC_HASHES = {
    "docs/spec/00-specification-overview.md": "a2ef3bbf2d8a1e88300d94f75bccfd1956a9b0ea18c14dce2dc0c1c41845a03d",
    "docs/spec/07-external-interfaces-adapters.md": "057577a3a3678cdc76a427bf578bd50295433fc0f096e50be3352a1f6e38062b",
    "docs/spec/08-test-requirements.md": "179d18cf95f1d8e8f4c925dc1998ae367a9e86fd32effbf9fe69272d359d48b7",
}
END_PROJECTION_VERSION = "simulator-closure-v1"
CATALOG_HASH = "4bb3ac87e4567798767f7c74a9f9280208adf931277c6afc7deeb865ef4d0673"
CATALOG_META: Json = {
    "catalog_schema_version": 1, "catalog_version": "public-ja-v1",
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


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    policies: tuple[str, str]
    sampling_seeds: tuple[int, int]
    pair_id: str


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
