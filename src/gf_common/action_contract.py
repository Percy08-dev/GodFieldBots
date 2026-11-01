"""Closed owner-proposal compatibility metadata, never policy features or authority.

Engineering component work does not qualify a Simulator experiment or source.
The baseline Simulator/catalog/spec pins remain independent and unchanged.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import NoReturn

from gf_common.domain import ContractError, Json, closed, integer

ENGINEERING_PROFILE_ID = "owner-proposal-engineering-20261101-v1"
OWNER_CONFIRMATION_SHA256 = "a0e6acb7fdc708ce4bd1b11f23d5dd3f0e41373cc1c1c1bef0a43a3e3efc2cb2"
ADOPTED_RULES_SHA256 = "f4132843bb700947d51e19b57bed1b3bc9bcaf5f63c0cfa60998f1ddc3d6d978"
PROFILE_SHA256 = "4f18089923ab8b6b4b6346d5adfbe0ca5fba401ead3a6d7c186c9596327c7a16"
GAME_REFERENCE_CAPACITY = 18
PENDING_COVERAGE = (
    "same-definition distinct normal-attack miracle combination exception",
    "mixed utility with weapon or utility and order",
    "pure Doll and CutCost armor attack-support placement/companions",
    "Tear/Herb explicit-opponent versus user-fixed effect mapping",
    "guardian-present or unknown-guardian use",
    "underfull soap/broom removal",
    "opponent broom without a permitted visible count",
    "per-definition omitted/null defaults",
    "normative target-document discrepancy",
    "runtime conformance and reconciled reflection opportunity contract",
)


def validate_record_purpose(purpose: str, contract: ActionContract) -> None:
    if type(contract) is not ActionContract:
        raise ContractError("explicit closed action contract required")
    if type(purpose) is not str or purpose not in ("component", "qualified_game"):
        raise ContractError("explicit component or qualified_game purpose required")
    if purpose == "qualified_game":
        require_qualified_contract(contract)


@dataclass(frozen=True)
class ActionContract:
    profile_id: str
    owner_confirmation_sha256: str
    adopted_rules_sha256: str
    eligibility: str

    def __post_init__(self) -> None:
        values = (self.profile_id, self.owner_confirmation_sha256, self.adopted_rules_sha256, self.eligibility)
        if any(type(value) is not str for value in values) or values != (
                ENGINEERING_PROFILE_ID, OWNER_CONFIRMATION_SHA256, ADOPTED_RULES_SHA256, "engineering_only"):
            raise ContractError("unknown action contract/profile or unsupported eligibility")


def engineering_contract() -> ActionContract:
    """Explicit constructor for new engineering components, never a reader default."""
    return ActionContract(ENGINEERING_PROFILE_ID, OWNER_CONFIRMATION_SHA256, ADOPTED_RULES_SHA256, "engineering_only")


def parse_action_contract(raw: Json, *, expected: ActionContract | None = None) -> ActionContract:
    contract = ActionContract(**closed(raw, set(ActionContract.__dataclass_fields__)))
    if expected is not None and (type(expected) is not ActionContract or contract != expected):
        raise ContractError("action contract differs from authoritative Config")
    return contract


def require_qualified_contract(contract: ActionContract) -> NoReturn:
    """No user purpose/producer flag can upgrade engineering source eligibility."""
    if type(contract) is not ActionContract:
        raise ContractError("explicit action contract required")
    parse_action_contract(asdict(contract))
    raise ContractError("engineering_only action contract prohibits real Simulator execution and qualified sources")


def validate_game_references(rows: Iterable[Iterable[int]]) -> None:
    """Physical slot uniqueness within each Action; no restriction across decisions.

    Slot identity is source+hand instance or learned definition in the owned
    reference table. Structural128-position component records use a separate
    synthetic purpose; this guard describes the reachable unique gameplay domain.
    """
    for row in rows:
        seen: set[int] = set()
        for slot in row:
            integer(slot, 0, GAME_REFERENCE_CAPACITY - 1)
            if slot in seen or len(seen) >= GAME_REFERENCE_CAPACITY:
                raise ContractError("game Action repeats a physical reference or exceeds eighteen references")
            seen.add(slot)
