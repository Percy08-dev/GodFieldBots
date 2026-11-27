"""Closed approved runtime receipts. Pure diagnostics, never policy/Source authority."""

from __future__ import annotations

from dataclasses import asdict
from types import MappingProxyType
from typing import Any

from gf_common.card_instance import (
    _DEFENSE_V3_RELEASE,
    CORRECTED_REVIEWED_SPEC_HASHES,
    DREAM_COMPAT_PROFILE,
    MOON_DREAM_PROFILE,
    MOON_DREAM_REVIEWED_SPEC_CHANGES,
    PESTLE_MIRAGE_PROFILE,
    PESTLE_MIRAGE_REVIEWED_SPEC_CHANGES,
    SACRIFICE_PROFILE,
    STRENGTH_POWDER_PROFILE,
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
DREAM_COMPAT_BINARY_SHA256 = "f95fadbc8eaad23fbe63e17ac9bdb0ebeca4b237956dedeab09d86f30cfd94bc"
DREAM_COMPAT_CHANGE_SHA256 = "eb56add45d4b4e89af3baf44ec25b0a909886229ed0dffe6d41e20b9c0019951"
DREAM_COMPAT_CHANGE_FILES = MappingProxyType(
    {
        "docs/spec/09-gameplay-rules.md":
            "74dacf30ee3fbbef9a8b34a10f09fd8e21a8f22385195fbad91dcfbc52324bc1",
        "docs/spec/godfield_rule_verification_register.md":
            "b00c6823bf2b0a862fe953fa8dab2263eb3cbf2207bdaa7e37d63bfa1c50faa0",
        "src/engine/catalog/mod.rs":
            "a430a9c6fdb5a1369ef2131ff1d62f65bb558b62a69e78ea6f77f414bac85130",
        "src/engine/resolution/tests.rs":
            "efd66e08332c544ab10f164b5e8cfc133739b9b485b057766d8ddc9b79fb4643",
        "tests/fixtures/full_game_v2_baseline.txt":
            "bbf514c1bdec6330a209643c23dad3654eea6779fd6ff78a70a8f06f725e35fa",
    }
)
MOON_DREAM_BINARY_SHA256 = "15137c72e670a90a64fd474030ee9024c91b97b1282c8d103db895f54a2bfa37"
MOON_DREAM_CHANGE_SHA256 = "2ac3dd61dd077420b404decede74aaee17d16ec5d5185b5fac0068725d3f663e"
MOON_DREAM_CHANGE_FILES = MappingProxyType(
    {
        **MOON_DREAM_REVIEWED_SPEC_CHANGES,
        "src/engine/catalog/mod.rs":
            "a430a9c6fdb5a1369ef2131ff1d62f65bb558b62a69e78ea6f77f414bac85130",
        "src/engine/resolution/choice.rs":
            "4636ba93454bb54d115a39239d59198b9ece27c671f92339e76b64fb3831ddc7",
        "src/engine/resolution/status.rs":
            "85fe7567dc6877e012f903c77ed0acf7de27fff16815799644b2f4a8ed772d45",
        "src/engine/resolution/tests.rs":
            "35034432e5d59bc8519d7d46b7900eb9049b6055bd2710b444293d03b9969cb7",
        "tests/fixtures/full_game_policy_rng_baseline.txt":
            "270d4f8495c8869f38f48fadb5bf61636d04ba275aa90d011a475810ddb6c4e6",
        "tests/fixtures/full_game_v2_baseline.txt":
            "bbf514c1bdec6330a209643c23dad3654eea6779fd6ff78a70a8f06f725e35fa",
        "tests/full_game_validation.rs":
            "852629fb5cc9f3629cfd5b9c73ed219f83b3a868b09712965e77241c2402be30",
    }
)

PESTLE_MIRAGE_BINARY_SHA256 = "500d55cd8eb03126f65d3c24fe58b6091bf8acdb2fc20aed76622a8679f4000d"
PESTLE_MIRAGE_CHANGE_SHA256 = "eb801942ca482a6d6e381442faff2857fa45f70c16432ad517ac190724785342"
PESTLE_MIRAGE_CHANGE_FILES = MappingProxyType(
    {
        **MOON_DREAM_CHANGE_FILES,
        **PESTLE_MIRAGE_REVIEWED_SPEC_CHANGES,
        "src/engine/model/state.rs":
            "d95b1e4a761c4d3c7f43e42872bba7eb9560ae08f5dba7e068846e34fb114472",
        "src/engine/resolution/choice.rs":
            "a29f6b7692cff865c50295ac061cccc707e655cdb3cfb980f307edf14a647ab4",
        "src/engine/resolution/combat.rs":
            "fa19b9f3816436c754e3ef510e4d528ce21e1b30af13191acb28b86b9eae3d17",
        "src/engine/resolution/mod.rs":
            "0e0c47d4ef1bc7b12ce09f03365ca238e39f49e88cf5bf3357796e99459443a4",
        "src/engine/resolution/status.rs":
            "6b4ad16a0f2b6d4ee1f648b2b9f9ebd45eebbfe83526b86220720e0a7c50527f",
        "src/engine/resolution/tests.rs":
            "683ad4cd7d251b8c09e8fc75e32a87ceda167393cdc9aac6908e0eb12642d626",
        "src/server/history/record/mod.rs":
            "1f28fe9a5496c5718bde3fe50da34ae997c97f04df4d6c67b459dbc63f3cc43a",
    }
)


def require_corrected_contract(contract: CardInstanceContract) -> None:
    if type(contract) is not CardInstanceContract or not is_corrected_contract(contract):
        raise ContractError("corrected runtime requires the exact approved contract")


def approved_binary_sha256(contract: CardInstanceContract) -> str:
    require_corrected_contract(contract)
    if contract.requires_direct_defense:
        return str(_DEFENSE_V3_RELEASE["binary_sha256"])
    if contract.profile_id in (PESTLE_MIRAGE_PROFILE, SACRIFICE_PROFILE):
        return PESTLE_MIRAGE_BINARY_SHA256
    if contract.profile_id == MOON_DREAM_PROFILE:
        return MOON_DREAM_BINARY_SHA256
    if contract.profile_id == DREAM_COMPAT_PROFILE:
        return DREAM_COMPAT_BINARY_SHA256
    if contract.profile_id == STRENGTH_POWDER_PROFILE:
        return STRENGTH_POWDER_BINARY_SHA256
    if is_shared_contract(contract):
        return SHARED_APPROVED_BINARY_SHA256
    return APPROVED_BINARY_SHA256


def approved_runtime_build(contract: CardInstanceContract) -> Json:
    binary_sha256 = approved_binary_sha256(contract)
    if contract.requires_direct_defense:
        change_hash, change_files = (
            _DEFENSE_V3_RELEASE["change_inventory_sha256"], _DEFENSE_V3_RELEASE["change_files"]
        )
    elif contract.profile_id in (PESTLE_MIRAGE_PROFILE, SACRIFICE_PROFILE):
        change_hash, change_files = PESTLE_MIRAGE_CHANGE_SHA256, PESTLE_MIRAGE_CHANGE_FILES
    elif contract.profile_id == MOON_DREAM_PROFILE:
        change_hash, change_files = MOON_DREAM_CHANGE_SHA256, MOON_DREAM_CHANGE_FILES
    elif contract.profile_id == DREAM_COMPAT_PROFILE:
        change_hash, change_files = DREAM_COMPAT_CHANGE_SHA256, DREAM_COMPAT_CHANGE_FILES
    elif contract.profile_id == STRENGTH_POWDER_PROFILE:
        change_hash, change_files = STRENGTH_POWDER_CHANGE_SHA256, STRENGTH_POWDER_CHANGE_FILES
    else:
        change_hash, change_files = APPROVED_CHANGE_SHA256, APPROVED_CHANGE_FILES
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
