"""Closed approved runtime receipts. Pure diagnostics, never policy/Source authority."""

from __future__ import annotations

from dataclasses import asdict
from types import MappingProxyType
from typing import Any

from gf_common.card_instance import (
    CORRECTED_REVIEWED_SPEC_HASHES,
    CardInstanceContract,
    is_corrected_contract,
    is_shared_contract,
    parse_card_instance_contract,
    reviewed_specs,
)
from gf_common.domain import ContractError, Json, closed, digest, integer

APPROVED_BINARY_SHA256 = "76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d"
SHARED_APPROVED_BINARY_SHA256 = "76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d"
APPROVED_CHANGE_SHA256 = "e34b7160f956b6510243077d418a0a96f34f6e271077021641b2fd50a9870346"
APPROVED_SPEC_FILES = MappingProxyType(dict(CORRECTED_REVIEWED_SPEC_HASHES))
APPROVED_CHANGE_FILES = MappingProxyType(
    {
        "src/engine/resolution/entry.rs": "3432945727970ebdb7a9ade9b03aee4641a36cdebab5312542092c38e0d6e663",
        "src/engine/resolution/turn.rs": "853bab7afcaecbf95980f0781b7d2e20d2d5d612d57fa02006e674fa089bd80c",
        "src/engine/resolution/tests.rs": "66b59de141f1f940c572bd9501376646514fc82690ad774100b1fc976439a0b4",
        "tests/http_contract.rs": "fbbcc074c111d245aea137f0755caea248614c89b7c13f44ad7f75efdf491fd1",
        "tests/canonical_v2.rs": "0978a7d531908d2408a77c30207648af4a7aba7635f9b1eaa375266fa35df205",
        "docs/spec/03-game-engine.md": "2441745566f553d8ce01806b64f88d4978209cd75e2ba0699fe71dc1fa022a04",
        "docs/spec/06-persistence-history.md": "3ac16199cb0a15289a1324f1f375a581a32e7889bcb73efa142dc24235853f64",
        "docs/spec/09-gameplay-rules.md": "6a11ad81e5df99d4962a57f7cf45b620f46624c7bb04628514600930e9543d5d",
        "docs/spec/godfield_rule_verification_register.md": "d892bafb621be33dc24455af2ab07d173f1127736e78da3e86a72134520084da",
    }
)
STRENGTH_POWDER_BINARY_SHA256 = "45ac711b2f172dfd89c23edf30ba0045ab0704c0215a6995d4e8626af1cfb923"
STRENGTH_POWDER_CHANGE_SHA256 = "fca9be8a30ae4969047f1136211acba2ee427a0f15df35aa8212f17afa36bf81"
STRENGTH_POWDER_CHANGE_FILES = MappingProxyType(
    {
        "docs/spec/09-gameplay-rules.md":
            "b6cf7e55c078df3eb893312e30215e58127117792ebce1cc0ce250b51fcbaefa",
        "src/engine/catalog/mod.rs":
            "8f61cc4d4719d82c33ecc5aed90337eeb8e416b6c88787e5f69dea90d2f33977",
        "src/engine/resolution/choice.rs":
            "4636ba93454bb54d115a39239d59198b9ece27c671f92339e76b64fb3831ddc7",
        "src/engine/resolution/tests.rs":
            "0e86413e5678292b1237d0e542179f18a6013b88e634ef3245ab6f98109515ed",
        "tests/fixtures/full_game_policy_rng_baseline.txt":
            "270d4f8495c8869f38f48fadb5bf61636d04ba275aa90d011a475810ddb6c4e6",
        "tests/fixtures/full_game_v2_baseline.txt":
            "c205750ea1c6cf934429592087294f1881ea99f0bb241b1f582be30c4af2c04b",
    }
)


def require_corrected_contract(contract: CardInstanceContract) -> None:
    if type(contract) is not CardInstanceContract or not is_corrected_contract(contract):
        raise ContractError("corrected runtime requires the exact approved contract")


def approved_binary_sha256(contract: CardInstanceContract) -> str:
    require_corrected_contract(contract)
    if contract.requires_powder_weapon:
        return STRENGTH_POWDER_BINARY_SHA256
    if is_shared_contract(contract):
        return SHARED_APPROVED_BINARY_SHA256
    return APPROVED_BINARY_SHA256


def approved_runtime_build(contract: CardInstanceContract) -> Json:
    binary_sha256 = approved_binary_sha256(contract)
    change_hash, change_files = (
        (STRENGTH_POWDER_CHANGE_SHA256, STRENGTH_POWDER_CHANGE_FILES)
        if contract.requires_powder_weapon
        else (APPROVED_CHANGE_SHA256, APPROVED_CHANGE_FILES)
    )
    return {
        "binary_sha256": binary_sha256,
        "spec_inventory_sha256": contract.spec_inventory_sha256,
        "change_inventory_sha256": change_hash,
        "spec_files": reviewed_specs(contract),
        "change_files": dict(change_files),
    }

def game_binding(contract: CardInstanceContract, rules_signature: str) -> Json:
    from gf_common.card_instance_rules import card_instance_rules, rule_signature
    from gf_common.structured_layout import layout_for_contract
    from gf_common.structured_tokens import signature

    require_corrected_contract(contract)
    if rules_signature != rule_signature(contract, card_instance_rules()):
        raise ContractError("corrected game rule signature mismatch")
    return {
        "action_contract": asdict(contract),
        "rule_signature": rules_signature,
        # Fixed128 window identifies structure across worker versions. Actual
        # model/journal token signatures additionally bind their own window.
        "layout_signature": signature(128, layout=layout_for_contract(contract)),
        "purpose": "qualified_game",
        "synthetic_capacity": False,
    }


def check_game_binding(raw: Json, contract: CardInstanceContract, rules_signature: str) -> None:
    expected = game_binding(contract, rules_signature)
    closed(raw, set(expected))
    if digest(raw) != digest(expected):
        raise ContractError("corrected game input binding mismatch")


def _identity(raw: Any) -> None:
    if type(raw) not in (list, tuple) or len(raw) != 5:
        raise ContractError("runtime file identity required")
    for number in raw:
        integer(number, 0)


def validate_runtime_receipt(
    raw: Json,
    *,
    require_executed: bool = False,
    expected_contract: CardInstanceContract | None = None,
) -> Json:
    """Validate representation/pins only; only the live owner can attest reads."""
    from gf_common.domain import clone

    closed(raw, {"schema_version", "action_contract", "build", "binary", "source", "executed"})
    if type(raw["schema_version"]) is not int or raw["schema_version"] not in (1,2):
        raise ContractError("runtime receipt schema required")
    contract = parse_card_instance_contract(raw["action_contract"])
    if expected_contract is not None and contract != expected_contract:
        raise ContractError("runtime receipt action contract mismatch")
    shared = is_shared_contract(contract)
    if raw["schema_version"] != (2 if shared else 1):
        raise ContractError("runtime receipt/profile version mismatch")
    build = approved_runtime_build(contract)
    if closed(raw["build"], set(build)) != build:
        raise ContractError("unapproved runtime release tuple/file inventory")
    binary_sha = build["binary_sha256"]
    binary = closed(raw["binary"], {"path", "sha256", "identity"})
    source = closed(raw["source"], {"path", "revision", "spec_files", "change_files"} | ({"dirty"} if shared else set()))
    if shared and source["dirty"] is not False:
        raise ContractError("shared runtime requires clean pinned source")
    if (
        binary["sha256"] != binary_sha
        or source["revision"] != contract.simulator_revision
        or source["spec_files"] != build["spec_files"]
        or source["change_files"] != build["change_files"]
    ):
        raise ContractError("runtime actual code/source differs from approved release")
    for path in (binary["path"], source["path"]):
        if type(path) is not str or not path.startswith("/"):
            raise ContractError("absolute runtime diagnostic path required")
    _identity(binary["identity"])
    executed = raw["executed"]
    if executed is None:
        if require_executed:
            raise ContractError("live executed-code receipt required before Create")
    else:
        closed(executed, {"pid", "start_time", "sha256", "identity"})
        integer(executed["pid"], 1)
        integer(executed["start_time"], 1)
        _identity(executed["identity"])
        if executed["sha256"] != binary_sha:
            raise ContractError("actual executed binary differs from approved release")
    return clone(raw)
