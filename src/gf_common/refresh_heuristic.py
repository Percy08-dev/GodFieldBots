"""Closed, stdlib-only identity for the corrected refresh baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules, rule_signature
from gf_common.card_instance_runtime import require_corrected_contract
from gf_common.domain import ContractError, Json, closed, digest


@dataclass(frozen=True)
class RefreshOptions:
    version: str = "1"
    attack_threshold: int = 12

    def __post_init__(self) -> None:
        if type(self.version) is not str or self.version != "1":
            raise ContractError("unsupported refresh recipe")
        if type(self.attack_threshold) is not int or self.attack_threshold <= 0:
            raise ContractError("refresh attack threshold must be a positive integer")


def parse_refresh_options(value: Json) -> RefreshOptions:
    return RefreshOptions(**closed(value, set(RefreshOptions.__dataclass_fields__)))


def refresh_policy_label(
    actions: ActionConfigV3, rules: CardInstanceRules, options: RefreshOptions
) -> str:
    if type(actions) is not ActionConfigV3 or type(options) is not RefreshOptions:
        raise ContractError("closed refresh actions/options required")
    require_corrected_contract(actions.action_contract)
    parse_refresh_options(asdict(options))
    identity = {
        "options": asdict(options),
        "action_contract": asdict(actions.action_contract),
        "rule_signature": rule_signature(actions.action_contract, rules),
        "action_config_hash": digest(asdict(actions)),
    }
    return "heuristic_refresh-card-instance-v2-" + digest(identity)
