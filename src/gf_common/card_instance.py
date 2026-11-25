"""Pinned card-instance contract metadata, independent of transport and models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer

SIMULATOR_REVISION = "07e8185f98c503aa41200a80ea554b4fd21088ad"
CARD_INSTANCE_SIMULATOR_REVISION = SIMULATOR_REVISION
CORRECTED_SIMULATOR_REVISION = "91d5926a7572db327eaea3751e7c24f428120996"
SHARED_SIMULATOR_REVISION = "072dabbca1458c9526897781d2fd64edeeb4f9ca"
SHARED_SPEC_INVENTORY_HASH = "da4bde0ce82d91dc10c4a8a7f012c1cb95b0079113344453b956f638fb1d39d4"
CORRECTED_SPEC_INVENTORY_HASH = "131890dfe579bc6951f5189e460971d33658339106e219a2e00b1a289a0534d0"
STRENGTH_POWDER_PROFILE = "card-instance-simulator-v2-shared-20261123"
STRENGTH_POWDER_SIMULATOR_REVISION = "3172da407e37a4ad57ac82e70a96910c3ac6fbb8"
STRENGTH_POWDER_SPEC_INVENTORY_HASH = "6403eef771b64c34e78eb653c2baefe65ceca5308051b506a7092151e9316c59"
DREAM_COMPAT_PROFILE = "card-instance-simulator-v2-shared-20261124"
DREAM_COMPAT_SIMULATOR_REVISION = "9e47d4c5f8cd3f60437f9aeabc95076c1dc04321"
DREAM_COMPAT_SPEC_INVENTORY_HASH = "afefd6f3b9113b813c90aaae7bb7b18f1e6299e52bf814ef34ec22112362ab5e"
MOON_DREAM_PROFILE = "card-instance-simulator-v2-shared-20261125"
MOON_DREAM_SIMULATOR_REVISION = "f394a7b57b0ffb12c6b5782ff52fdb8842ec7534"
MOON_DREAM_SPEC_INVENTORY_HASH = "c2a0e37607bff98e342b618c5057887cdf6adca981b81731245adf5853cdfc78"
MOON_DREAM_REVIEWED_SPEC_CHANGES = MappingProxyType({
    "docs/spec/03-game-engine.md": "5623712d584f7c9865cb294922a2904c58801398d0478f69427509aa08c92833",
    "docs/spec/06-persistence-history.md": "65c169afe9e1d6e4473b6017c0ef91ad9721e4a7657d14a9a4ae1b7323c8f9fe",
    "docs/spec/09-gameplay-rules.md": "b3b6d69e475fc02ef052815be4244fd394cdf9f6ba6de872af620ab09faea063",
    "docs/spec/godfield_rule_verification_register.md":
        "ddf7e9d94239176794fb5a8b84921d1fee02e8eced9aaaae7c1dda73db0b07da",
})
PESTLE_MIRAGE_PROFILE = "card-instance-simulator-v2-shared-20261125-pestle"
PESTLE_MIRAGE_SIMULATOR_REVISION = "6d50603dab3f6d2df34ea4e433bab7dca96f52c1"
PESTLE_MIRAGE_SPEC_INVENTORY_HASH = "05b68761d1c23a3784d04093b417875b2b5893739a1e6c73367a969dc83db753"
PESTLE_MIRAGE_REVIEWED_SPEC_CHANGES = MappingProxyType({
    "docs/spec/03-game-engine.md": "3a2e9595f7de0ecca297eee0af8eac94bed7cc05e5699da259ff58d8f350da95",
    "docs/spec/06-persistence-history.md": "af6a30d485195b95825f1a4e182508b7b24855fa750744e167e04f4074d44960",
    "docs/spec/08-test-requirements.md": "169e852529e6bc1cd3a0f08b163fa4554aac25d3e57f12a8f255ad960419ed74",
    "docs/spec/09-gameplay-rules.md": "02df399100b785ebc6fd2e19af5e3a859c38efb7252df25072c3ccd410f87e43",
    "docs/spec/godfield_rule_verification_register.md":
        "5a5269fcb62acbd1dd1507c38d9a65d29228cf9266929af4a84b82e10f6274b5",
})
SACRIFICE_PROFILE = "card-instance-simulator-v2-shared-20261125-sacrifice"
RULE_SET = "true-match-1v1-v2"
CATALOG_HASH = "f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6"
SPEC_INVENTORY_HASH = "8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5"
GENERATOR_VERSION = "public-card-instance-diverse-v5"
CATALOG_META: Json = {
    "catalog_schema_version": 2,
    "catalog_version": "public-ja-v3",
    "rule_set_version": RULE_SET,
    "locale": "ja",
    "content_sha256": CATALOG_HASH,
    "definition_count": 296,
}


@dataclass(frozen=True)
class CardInstanceContract:
    profile_id: str = "card-instance-simulator-v2-20261103"
    simulator_revision: str = SIMULATOR_REVISION
    rule_set_version: str = RULE_SET
    gameplay_api: str = "v2"
    card_ref_scheme: str = "aes256-card-ref-v1"
    catalog_schema_version: int = 2
    catalog_version: str = "public-ja-v3"
    catalog_sha256: str = CATALOG_HASH
    spec_inventory_sha256: str = SPEC_INVENTORY_HASH
    policy_input_version: str = "player-card-refs-v2"
    input_recipe: str = "participant-structured-v2"

    def __post_init__(self) -> None:
        if type(self.profile_id) is not str or self.profile_id not in _CONTRACT_PINS:
            raise ContractError("unknown card-instance contract profile")
        expected = _CONTRACT_PINS[self.profile_id]
        for name, pin in zip(_CONTRACT_FIELD_NAMES, expected, strict=True):
            value = getattr(self, name)
            if type(value) is not type(pin) or value != pin:
                raise ContractError("unknown card-instance contract pin")

    @property
    def requires_powder_weapon(self) -> bool:
        return self.profile_id in (
            STRENGTH_POWDER_PROFILE, DREAM_COMPAT_PROFILE, MOON_DREAM_PROFILE, PESTLE_MIRAGE_PROFILE,
            SACRIFICE_PROFILE,
        )

    @property
    def mirage_danger_requires_opponent(self) -> bool:
        return self.profile_id in (PESTLE_MIRAGE_PROFILE, SACRIFICE_PROFILE)

    @property
    def protects_sacrifice_artifacts(self) -> bool:
        return self.profile_id == SACRIFICE_PROFILE


_CONTRACT_FIELD_NAMES = tuple(CardInstanceContract.__dataclass_fields__)
_ORIGINAL_PIN: tuple[Any, ...] = tuple(
    descriptor.default for descriptor in CardInstanceContract.__dataclass_fields__.values()
)
_CORRECTED_PIN: tuple[Any, ...] = tuple(
    {
        "profile_id": "card-instance-simulator-v2-20261104",
        "simulator_revision": CORRECTED_SIMULATOR_REVISION,
        "spec_inventory_sha256": CORRECTED_SPEC_INVENTORY_HASH,
    }.get(name, default)
    for name, default in zip(_CONTRACT_FIELD_NAMES, _ORIGINAL_PIN, strict=True)
)
_SHARED_PIN: tuple[Any, ...] = tuple(
    {"profile_id": "card-instance-simulator-v2-shared-20261118",
     "simulator_revision": SHARED_SIMULATOR_REVISION,
     "spec_inventory_sha256": SHARED_SPEC_INVENTORY_HASH}.get(name, default)
    for name, default in zip(_CONTRACT_FIELD_NAMES, _CORRECTED_PIN, strict=True)
)
_STRENGTH_POWDER_PIN: tuple[Any, ...] = tuple(
    {
        "profile_id": STRENGTH_POWDER_PROFILE,
        "simulator_revision": STRENGTH_POWDER_SIMULATOR_REVISION,
        "spec_inventory_sha256": STRENGTH_POWDER_SPEC_INVENTORY_HASH,
    }.get(name, default)
    for name, default in zip(_CONTRACT_FIELD_NAMES, _SHARED_PIN, strict=True)
)
_DREAM_COMPAT_PIN: tuple[Any, ...] = tuple(
    {
        "profile_id": DREAM_COMPAT_PROFILE,
        "simulator_revision": DREAM_COMPAT_SIMULATOR_REVISION,
        "spec_inventory_sha256": DREAM_COMPAT_SPEC_INVENTORY_HASH,
    }.get(name, default)
    for name, default in zip(_CONTRACT_FIELD_NAMES, _STRENGTH_POWDER_PIN, strict=True)
)
_MOON_DREAM_PIN: tuple[Any, ...] = tuple(
    {
        "profile_id": MOON_DREAM_PROFILE,
        "simulator_revision": MOON_DREAM_SIMULATOR_REVISION,
        "spec_inventory_sha256": MOON_DREAM_SPEC_INVENTORY_HASH,
    }.get(name, default)
    for name, default in zip(_CONTRACT_FIELD_NAMES, _DREAM_COMPAT_PIN, strict=True)
)
_PESTLE_MIRAGE_PIN: tuple[Any, ...] = tuple(
    {
        "profile_id": PESTLE_MIRAGE_PROFILE,
        "simulator_revision": PESTLE_MIRAGE_SIMULATOR_REVISION,
        "spec_inventory_sha256": PESTLE_MIRAGE_SPEC_INVENTORY_HASH,
    }.get(name, default)
    for name, default in zip(_CONTRACT_FIELD_NAMES, _MOON_DREAM_PIN, strict=True)
)
_SACRIFICE_PIN: tuple[Any, ...] = tuple(
    SACRIFICE_PROFILE if name == "profile_id" else pin
    for name, pin in zip(_CONTRACT_FIELD_NAMES, _PESTLE_MIRAGE_PIN, strict=True)
)
_CONTRACT_PINS = MappingProxyType(
    {
        "card-instance-simulator-v2-20261103": _ORIGINAL_PIN,
        "card-instance-simulator-v2-20261104": _CORRECTED_PIN,
        "card-instance-simulator-v2-shared-20261118": _SHARED_PIN,
        STRENGTH_POWDER_PROFILE: _STRENGTH_POWDER_PIN,
        DREAM_COMPAT_PROFILE: _DREAM_COMPAT_PIN,
        MOON_DREAM_PROFILE: _MOON_DREAM_PIN,
        PESTLE_MIRAGE_PROFILE: _PESTLE_MIRAGE_PIN,
        SACRIFICE_PROFILE: _SACRIFICE_PIN,
    }
)


def card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract()


def corrected_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _CORRECTED_PIN, strict=True)))


def shared_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _SHARED_PIN, strict=True)))


def strength_powder_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _STRENGTH_POWDER_PIN, strict=True)))


def dream_compat_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _DREAM_COMPAT_PIN, strict=True)))


def moon_dream_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _MOON_DREAM_PIN, strict=True)))


def pestle_mirage_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _PESTLE_MIRAGE_PIN, strict=True)))


def sacrifice_card_instance_contract() -> CardInstanceContract:
    return CardInstanceContract(**dict(zip(_CONTRACT_FIELD_NAMES, _SACRIFICE_PIN, strict=True)))


def is_shared_contract(contract: object) -> bool:
    return type(contract) is CardInstanceContract and contract.profile_id in (
        "card-instance-simulator-v2-shared-20261118",
        STRENGTH_POWDER_PROFILE,
        DREAM_COMPAT_PROFILE,
        MOON_DREAM_PROFILE,
        PESTLE_MIRAGE_PROFILE,
        SACRIFICE_PROFILE,
    )


def is_corrected_contract(contract: object) -> bool:
    return type(contract) is CardInstanceContract and (
        contract.profile_id == "card-instance-simulator-v2-20261104"
        or is_shared_contract(contract)
    )


def reviewed_specs(contract: CardInstanceContract) -> dict[str, str]:
    parse_card_instance_contract(asdict(contract))
    if is_shared_contract(contract):
        specs = {
            **CORRECTED_REVIEWED_SPEC_HASHES,
            "docs/spec/07-external-interfaces-adapters.md":
                "8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2",
        }
        if contract.profile_id == STRENGTH_POWDER_PROFILE:
            specs["docs/spec/09-gameplay-rules.md"] = (
                "b6cf7e55c078df3eb893312e30215e58127117792ebce1cc0ce250b51fcbaefa"
            )
        elif contract.profile_id == DREAM_COMPAT_PROFILE:
            specs["docs/spec/09-gameplay-rules.md"] = (
                "74dacf30ee3fbbef9a8b34a10f09fd8e21a8f22385195fbad91dcfbc52324bc1"
            )
            specs["docs/spec/godfield_rule_verification_register.md"] = (
                "b00c6823bf2b0a862fe953fa8dab2263eb3cbf2207bdaa7e37d63bfa1c50faa0"
            )
        elif contract.profile_id == MOON_DREAM_PROFILE:
            specs.update(MOON_DREAM_REVIEWED_SPEC_CHANGES)
        elif contract.profile_id in (PESTLE_MIRAGE_PROFILE, SACRIFICE_PROFILE):
            specs.update(PESTLE_MIRAGE_REVIEWED_SPEC_CHANGES)
        return specs
    return dict(CORRECTED_REVIEWED_SPEC_HASHES if is_corrected_contract(contract) else REVIEWED_SPEC_HASHES)


def parse_card_instance_contract(raw: Json) -> CardInstanceContract:
    return CardInstanceContract(**closed(raw, set(CardInstanceContract.__dataclass_fields__)))


@dataclass(frozen=True)
class ActionConfigV3:
    action_contract: CardInstanceContract = field(kw_only=True)
    version: str = "3"
    generator: str = GENERATOR_VERSION
    use_action_options: bool = True
    targets: tuple[str, ...] = ("opponent", "self")
    max_sequence_length: int = 18
    max_candidates: int = 32768
    max_expansions: int = 262144
    max_temporary_candidates: int = 262144
    count_states: int = 16384
    count_milliseconds: int = 50
    search_milliseconds: int = 1000
    channel_weights: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)

    def __post_init__(self) -> None:
        if type(self.action_contract) is not CardInstanceContract:
            raise ContractError("explicit card-instance contract required")
        parse_card_instance_contract(asdict(self.action_contract))
        if (
            type(self.version) is not str
            or self.version != "3"
            or self.generator != GENERATOR_VERSION
        ):
            raise ContractError("unsupported card-instance action recipe")
        if self.use_action_options is not True:
            raise ContractError("actions3 requires acting-player atomic hints")
        if (
            type(self.targets) is not tuple
            or not self.targets
            or any(type(value) is not str for value in self.targets)
            or len(set(self.targets)) != len(self.targets)
            or set(self.targets) - {"self", "opponent"}
        ):
            raise ContractError("invalid card-instance targets")
        for name, maximum in (
            ("max_sequence_length", 128),
            ("max_candidates", 65536),
            ("max_expansions", 2**31 - 1),
            ("max_temporary_candidates", 2**31 - 1),
            ("count_states", 2**31 - 1),
            ("count_milliseconds", 1000),
            ("search_milliseconds", 10000),
        ):
            integer(getattr(self, name), 1, maximum)
        if type(self.channel_weights) is not tuple or len(self.channel_weights) != 7:
            raise ContractError("seven channel weights required")
        for weight in self.channel_weights:
            integer(weight, 1, 2**31 - 1)


def action_config_v3(raw: Json) -> ActionConfigV3:
    value = closed(raw, set(ActionConfigV3.__dataclass_fields__))
    for name in ("targets", "channel_weights"):
        if not isinstance(value[name], (list, tuple)):
            raise ContractError("action array required")
    return ActionConfigV3(
        **{
            **value,
            "action_contract": parse_card_instance_contract(value["action_contract"]),
            "targets": tuple(value["targets"]),
            "channel_weights": tuple(value["channel_weights"]),
        }
    )


# Exact normative source inventory at the pinned Simulator revision.
REVIEWED_SPEC_HASHES: dict[str, str] = {
    "docs/spec/00-specification-overview.md": "674346a7b3f798a4c02e5241e7c760170224670e52bf81f5fe0acf46c8189a3e",
    "docs/spec/01-server-spec.md": "4ab2072fafef4a7b400527cafaf933de8896c045da1e5943ed6d7e1dd1c655ba",
    "docs/spec/02-software-design.md": "74fb313ebd0322d39e359b192b07bc6b702952b10b8a8345720b11a5b6409533",
    "docs/spec/03-game-engine.md": "24e1b6f7ffe2d5eb78df3d553ffd969ab04bb24438719c7f815dc44a0fb09a37",
    "docs/spec/04-random-system.md": "54ace9374f8d6ba93612a35548e33ce542b6ae9271ebac5dde9aad6fbe847a19",
    "docs/spec/05-observation.md": "6a1cf52eab8e88f588020955bd281052eb2ac6b19612f5229ce4f86f178d6e8b",
    "docs/spec/06-persistence-history.md": "eb5db7bbd72d265af3cf90bbe30e5a0e59944bf6c42ef8feedafe3d24abfa99d",
    "docs/spec/07-external-interfaces-adapters.md": "5a179c97ef116d52ab1a4e52b82726fb45fb025cc6a6cbeeec4cf639718e9664",
    "docs/spec/08-test-requirements.md": "ad1b275112c1c019ca78be48831e8a5e4f1bf1472c0db0645c2359e3895c9e09",
    "docs/spec/09-gameplay-rules.md": "1f8ab2fa87748fb79d927dbcce3457b92ea2e05d825e70c78a51bedb7f31b536",
    "docs/spec/10-cli-client.md": "7394dddb8498bfead6f91a47b7cfdbd0b9179bc40dba1416e25acabe9b99734a",
    "docs/spec/card-definition-classification-v2.md": "62e6b88b237d46e808758f30e3964b4811c4e5f73786595840fda206fc79885c",
    "docs/spec/godfield_rule_verification_register.md": "a1a9e4aee2ec05ee490154cb1df143be6cd63df6865d2cbfca4f863fc6af4ce5",
}

# Corrected91d5926 normative inventory; historical07e map above stays unchanged.
CORRECTED_REVIEWED_SPEC_HASHES: dict[str, str] = {
    "docs/spec/00-specification-overview.md": "674346a7b3f798a4c02e5241e7c760170224670e52bf81f5fe0acf46c8189a3e",
    "docs/spec/01-server-spec.md": "4ab2072fafef4a7b400527cafaf933de8896c045da1e5943ed6d7e1dd1c655ba",
    "docs/spec/02-software-design.md": "74fb313ebd0322d39e359b192b07bc6b702952b10b8a8345720b11a5b6409533",
    "docs/spec/03-game-engine.md": "2441745566f553d8ce01806b64f88d4978209cd75e2ba0699fe71dc1fa022a04",
    "docs/spec/04-random-system.md": "54ace9374f8d6ba93612a35548e33ce542b6ae9271ebac5dde9aad6fbe847a19",
    "docs/spec/05-observation.md": "6a1cf52eab8e88f588020955bd281052eb2ac6b19612f5229ce4f86f178d6e8b",
    "docs/spec/06-persistence-history.md": "3ac16199cb0a15289a1324f1f375a581a32e7889bcb73efa142dc24235853f64",
    "docs/spec/07-external-interfaces-adapters.md": "5a179c97ef116d52ab1a4e52b82726fb45fb025cc6a6cbeeec4cf639718e9664",
    "docs/spec/08-test-requirements.md": "ad1b275112c1c019ca78be48831e8a5e4f1bf1472c0db0645c2359e3895c9e09",
    "docs/spec/09-gameplay-rules.md": "6a11ad81e5df99d4962a57f7cf45b620f46624c7bb04628514600930e9543d5d",
    "docs/spec/10-cli-client.md": "7394dddb8498bfead6f91a47b7cfdbd0b9179bc40dba1416e25acabe9b99734a",
    "docs/spec/card-definition-classification-v2.md": "62e6b88b237d46e808758f30e3964b4811c4e5f73786595840fda206fc79885c",
    "docs/spec/godfield_rule_verification_register.md": "d892bafb621be33dc24455af2ab07d173f1127736e78da3e86a72134520084da",
}
