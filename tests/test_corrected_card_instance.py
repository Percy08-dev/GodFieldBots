"""Explicit corrected profile adoption with frozen pre-adoption compatibility.

Original snapshot: runs/assessments/card-instance-corrected-adoption-20261104/
before-compatibility.json, SHA256
3f0768e9aed4118c0f18ee6f765b8892f2df9eaa1400b56b035712d9b88789c3.
Path hashes were captured at /home/percy/develop/GodFieldBots; normalize only
known effective Config path fields to that original parent for relocated tests.
"""

import hashlib
import json
import random
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path

import pytest

from gf_common.card_instance import (
    CORRECTED_REVIEWED_SPEC_HASHES,
    CORRECTED_SIMULATOR_REVISION,
    CORRECTED_SPEC_INVENTORY_HASH,
    ActionConfigV3,
    CardInstanceContract,
    action_config_v3,
    card_instance_contract,
    corrected_card_instance_contract,
    parse_card_instance_contract,
)
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import (
    CARD_INSTANCE_LAYOUT as OLD,
)
from gf_common.structured_layout import (
    CORRECTED_CARD_INSTANCE_LAYOUT as NEW,
)
from gf_common.structured_layout import (
    LEGACY_LAYOUT,
    is_card_instance_layout,
    layout_for_contract,
    require_layout,
)
from gf_common.structured_tokens import (
    signature,
    tokenize_frame,
    tokenize_table,
    validate_frame_tokens,
)
from gf_common.structured_wire import (
    delta_dict,
    frame_dict,
    generated_dict_v3,
    parse_delta,
    parse_frame,
    parse_generated_v3,
    parse_table,
)
from tests.card_instance_action_support import inputs
from tests.test_card_instance_contract import relocate_effective_paths

ROOT = Path(__file__).parents[1]


def records(value=None, *, layout=NEW, episode="corrected-adoption-fixed"):
    value = (
        inputs([("fireball", "learned"), ("spiritual-staff", None)], mp=0)
        if value is None
        else value
    )
    bank = EventBank(1, episode, layout=layout)
    delta = event_delta(1, episode, 0, bank.boundary.prefix_hash, value.events, layout=layout)
    bank.append(delta)
    frame = DecisionFrame.from_input(value, bank, 0, layout=layout)
    return frame, bank, delta


def generate(value=None, *, contract=None, **settings):
    contract = corrected_card_instance_contract() if contract is None else contract
    layout = layout_for_contract(contract)
    frame, bank, delta = records(value, layout=layout)
    result = PublicCandidateGeneratorV3(
        ActionConfigV3(action_contract=contract, **settings),
        rules=card_instance_rules(),
        clock=lambda: 0.0,
    ).generate(frame, bank.view(), random.Random(17), NumericCosts())
    result.table.validate(frame)
    actions = [result.table.to_action(i, frame) for i in range(len(result.table.kinds))]
    return result, actions, frame, bank, delta


def test_two_exact_full_profiles_same_type_and_unchanged_defaults():
    old, new = card_instance_contract(), corrected_card_instance_contract()
    assert type(old) is type(new) is CardInstanceContract
    assert old == CardInstanceContract()
    assert asdict(old) == BEFORE["contract"]
    assert len(fields(new)) == len(fields(old)) == 11
    assert new.profile_id == "card-instance-simulator-v2-20261104"
    assert new.simulator_revision == CORRECTED_SIMULATOR_REVISION
    assert new.spec_inventory_sha256 == CORRECTED_SPEC_INVENTORY_HASH
    assert digest(CORRECTED_REVIEWED_SPEC_HASHES) == CORRECTED_SPEC_INVENTORY_HASH
    for contract in (old, new):
        assert parse_card_instance_contract(asdict(contract)) == contract
        config = ActionConfigV3(action_contract=contract)
        assert action_config_v3(json.loads(json.dumps(asdict(config)))) == config
    for name in ("profile_id", "simulator_revision", "spec_inventory_sha256"):
        mixed = {**asdict(new), name: getattr(old, name)}
        with pytest.raises(ContractError):
            parse_card_instance_contract(mixed)
    for missing in asdict(new):
        raw = asdict(new)
        del raw[missing]
        with pytest.raises(ContractError):
            parse_card_instance_contract(raw)
    for name, value in (
        ("profile_id", "unknown"),
        ("profile_id", []),
        ("catalog_schema_version", True),
    ):
        with pytest.raises(ContractError):
            parse_card_instance_contract({**asdict(new), name: value})
    with pytest.raises(ContractError):
        CardInstanceContract(profile_id=new.profile_id)


def test_old_defaults_thirteen_config_hashes_tokens_rules_and_all_goldens_stay_exact():
    from gf_cli.config import ExperimentConfig, load_config

    assert digest(asdict(ExperimentConfig())) == BEFORE["default_config_sha256"]
    assert (
        digest(asdict(ActionConfigV3(action_contract=card_instance_contract())))
        == BEFORE["actions3_sha256"]
    )
    assert (
        rule_signature(card_instance_contract(), card_instance_rules())
        == BEFORE["old_rule_signature"]
    )
    for key, layout in (("legacy", LEGACY_LAYOUT), ("07e", OLD)):
        for window, expected in BEFORE["token_signatures"][key].items():
            assert signature(int(window), layout=layout) == expected
    assert len(BEFORE["experiments"]) == 13
    for name, expected in BEFORE["experiments"].items():
        loaded = load_config(ROOT / name)
        normalized = relocate_effective_paths(
            loaded.effective, source_parent=ROOT.parent, target_parent=Path("/home/percy/develop")
        )
        assert digest(normalized) == expected, name
    for name, expected in BEFORE["goldens"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_corrected_layout_preserves_card_fields_but_repairs_guardian_event_namespace():
    from gf_common import structured_vocab as vocab

    assert layout_for_contract(card_instance_contract()) is OLD
    assert layout_for_contract(corrected_card_instance_contract()) is NEW
    assert is_card_instance_layout(OLD) and is_card_instance_layout(NEW)
    assert not is_card_instance_layout(LEGACY_LAYOUT)
    assert NEW is not OLD and NEW.simulator_revision != OLD.simulator_revision
    for name in (
        "state_numeric_fields",
        "state_categories",
        "card_numeric_fields",
        "card_categories",
        "event_types",
        "event_numeric_fields",
        "action_kinds",
        "targets",
        "format_versions",
    ):
        assert getattr(NEW, name) is getattr(OLD, name)
    assert NEW.event_categories[1] == OLD.event_categories[1] + vocab.GUARDIANS
    assert NEW.event_categories[0] is OLD.event_categories[0]
    assert NEW.event_categories[2:] == OLD.event_categories[2:]
    assert NEW.event_width == OLD.event_width + len(vocab.GUARDIANS) == 405
    assert NEW.history_width == OLD.history_width + len(vocab.GUARDIANS) == 387
    with pytest.raises(ContractError):
        require_layout(replace(NEW))


def test_corrected_bank_and_frame_bind_complete_contract_and_frozen_class():
    value = inputs([("fireball", "learned"), ("spiritual-staff", None)], mp=0)
    old, old_bank, old_delta = records(value, layout=OLD)
    new, new_bank, new_delta = records(value)
    assert old.payload == new.payload
    assert old_bank.boundary.prefix_hash != new_bank.boundary.prefix_hash
    assert old.frame_hash != new.frame_hash
    assert new._binding()["action_contract"] == asdict(corrected_card_instance_contract())
    assert "action_contract" not in old._binding()
    assert new_bank.view()._prefixes[0] == digest(
        [NEW.input_recipe, "event-bank", 1, new.episode, asdict(NEW.action_contract)]
    )
    for current, previous in (
        (new, old),
        (new_bank.view(), old_bank.view()),
        (new_delta, old_delta),
    ):
        assert type(current) is not type(previous)
        assert fields(current) == fields(previous)
        assert set(current.__dataclass_fields__) == set(previous.__dataclass_fields__)
        assert "layout" not in asdict(current)
        with pytest.raises(FrozenInstanceError):
            current.layout = OLD
    assert replace(new).layout is NEW and replace(new_bank.view()).layout is NEW
    assert new_bank.view().prefix(new_bank.boundary).layout is NEW
    with pytest.raises(TypeError):
        replace(new_delta)


def test_cross_profile_frames_views_tokens_tables_and_generator_reject():
    old_result, _, old, old_bank, _ = generate(contract=card_instance_contract())
    result, _, frame, bank, _ = generate()
    assert old.payload == frame.payload
    for left, right in ((OLD, NEW), (NEW, OLD), (NEW, LEGACY_LAYOUT)):
        current, current_bank, _ = records(layout=left)
        raw = frame_dict(current, layout=left)
        with pytest.raises(ContractError):
            parse_frame(raw, layout=right)
        with pytest.raises(ContractError):
            frame_dict(current, layout=right)
        with pytest.raises(ContractError):
            tokenize_frame(current, current_bank.view(), layout=right)
    for source, view in ((frame, old_bank.view()), (old, bank.view())):
        with pytest.raises(ContractError):
            source.bank_view(view)
    with pytest.raises(ContractError):
        replace(bank.view(), _prefixes=old_bank.view()._prefixes)
    with pytest.raises(ContractError):
        DecisionFrame.from_input(inputs([]), bank, 0)
    tokens = tokenize_frame(frame, bank.view(), layout=NEW)
    for layout in (OLD, LEGACY_LAYOUT):
        with pytest.raises(ContractError):
            validate_frame_tokens(tokens, layout=layout)
    assert signature(layout=NEW) != signature(layout=OLD)
    for table, other in ((result.table, old), (old_result.table, frame)):
        with pytest.raises(ContractError):
            table.validate(other)
        with pytest.raises(ContractError):
            table.to_action(0, other)
    with pytest.raises(ContractError):
        PublicCandidateGeneratorV3(
            ActionConfigV3(action_contract=corrected_card_instance_contract()),
            rules=card_instance_rules(),
        ).generate(old, old_bank.view(), random.Random(17), NumericCosts())


def test_delta_parse_binds_expected_context_while_append_attests_profile_prefix():
    old = EventBank(1, "same-episode", layout=OLD)
    corrected = EventBank(1, "same-episode", layout=NEW)
    raw = delta_dict(
        event_delta(1, old.episode, 0, old.boundary.prefix_hash, [], layout=OLD), layout=OLD
    )
    parsed = parse_delta(raw, layout=NEW)
    assert parsed.layout is NEW  # Standalone parsing does not authenticate origin.
    with pytest.raises(ContractError):
        corrected.append(parsed)
    raw_new = delta_dict(
        event_delta(1, corrected.episode, 0, corrected.boundary.prefix_hash, [], layout=NEW),
        layout=NEW,
    )
    with pytest.raises(ContractError):
        old.append(parse_delta(raw_new, layout=OLD))
    corrected.append(parse_delta(raw_new, layout=NEW))


def test_corrected_event_prefix_excludes_future_and_requires_current_own_interval():
    first_event = {"type": "safe", "player": 1}
    value = replace(inputs([("fireball", "learned")], mp=0), events=(first_event,))
    first, bank, _ = records(value)
    before = tokenize_frame(first, bank.view(), layout=NEW)
    second_event = {"type": "artifact_used", "player": 2, "definition_id": "bronze-club"}
    bank.append(
        event_delta(1, bank.episode, 1, bank.boundary.prefix_hash, [second_event], layout=NEW)
    )
    assert tokenize_frame(first, bank.view(), layout=NEW) == before
    current = replace(value, events=(first_event, second_event))
    current.observation["state_revision"] = 1
    current.action_options["state_revision"] = 1
    current.action_options["action_options"]["state_revision"] = 1
    second = DecisionFrame.from_input(current, bank, first.boundary.end, layout=NEW)
    assert second.interval_start == 1 and second.boundary.end == 2
    assert second.bank_view(bank.view()).events(1) == (second_event,)
    with pytest.raises(ContractError):
        DecisionFrame.from_input(current, bank, second.boundary.end, layout=NEW)


def test_corrected_report_wire_requires_exact_expected_contract_and_rules():
    result, _, frame, _, _ = generate()
    binding = {
        "action_contract": corrected_card_instance_contract(),
        "rules": card_instance_rules(),
    }
    raw = json.loads(json.dumps(generated_dict_v3(result, **binding)))
    assert parse_generated_v3(raw, **binding) == result
    assert (
        parse_table(
            raw["table"],
            **{key: value for key, value in binding.items() if key == "action_contract"},
            purpose="component",
            synthetic_capacity=False,
            layout=NEW,
        )
        == result.table
    )
    assert result.report.rule_signature == rule_signature(
        binding["action_contract"], binding["rules"]
    )
    assert result.report.rule_signature != BEFORE["old_rule_signature"]
    with pytest.raises(ContractError):
        parse_generated_v3(
            raw, action_contract=card_instance_contract(), rules=card_instance_rules()
        )
    with pytest.raises(ContractError):
        generated_dict_v3(
            result, action_contract=card_instance_contract(), rules=card_instance_rules()
        )
    with pytest.raises(ContractError):
        parse_table(
            raw["table"],
            action_contract=corrected_card_instance_contract(),
            purpose="component",
            synthetic_capacity=False,
            layout=OLD,
        )
    assert frame.references == tuple(card["card_ref"] for card in frame.observation["own"]["cards"])


@pytest.mark.parametrize("name", ["flame", "fireball", "aura", "wall", "wind", "spring"])
@pytest.mark.parametrize("state", ["unused", "learned"])
@pytest.mark.parametrize("end_times", [False, True])
def test_corrected_all_miracle_discard_matrix_stays_in_candidates(name, state, end_times):
    result, actions, frame, _, _ = generate(
        inputs([(name, state)], mp=0, usable=(), end_times=end_times)
    )
    assert actions == [
        {"type": "sacrifice" if end_times else "discard", "card_refs": list(frame.references)}
    ]
    assert result.report.precount.complete == 1


@pytest.mark.parametrize(
    "name",
    [
        "smile-dew",
        "heart-dew",
        "romance-water",
        "galaxy-geyser",
        "smile-flower",
        "heart-flower",
        "romance-fragrance",
        "spring",
        "treasure",
        "tone",
        "song",
        "smile-shell",
        "heart-shell",
        "thump-thump-tear",
        "heaven-herb",
        "goddess-s-soap",
        "nocturnal-broom",
        "guardian-pot",
        "release",
    ],
)
def test_corrected_all_scoped_utility_targets(name):
    state = "learned" if name in ("spring", "treasure", "tone", "song", "release") else None
    _, actions, frame, _, _ = generate(inputs([(name, state)], discard=False))
    assert {canonical(action) for action in actions} == {
        canonical(
            {"type": "play_sequence", "ordered_card_refs": list(frame.references), "target": target}
        )
        for target in (1, 2)
    }


@pytest.mark.parametrize("flash", [False, True])
def test_corrected_flash_response_and_unpaid_costcut_paths(flash):
    _, actions, frame, _, _ = generate(
        inputs(
            [("wall", "learned"), ("spiritual-socks", None)], choice="defense", mp=0, flash=flash
        )
    )
    pair = {"type": "respond", "ordered_card_refs": list(frame.references)}
    assert (pair in actions) is not flash
    _, plays, original, _, _ = generate(
        inputs([("fireball", "learned"), ("spiritual-staff", None)], mp=0, flash=flash)
    )
    assert {
        "type": "play_sequence",
        "ordered_card_refs": list(original.references),
        "target": 2,
    } in plays


@pytest.mark.parametrize("layout", [OLD, NEW])
def test_both_profiles_reject_old_codec_numeric_and_submit_ready_synthetic(
    layout, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from gf_common import candidate_columns
    from gf_models.structured_config import ModelConfigV2
    from gf_training import structured_rollout as writer

    frame, _, _ = records(layout=layout)
    contract = layout.action_contract
    table = CandidateTable.from_actions(
        frame,
        [{"type": "pray"}],
        action_contract=contract,
        purpose="component",
        synthetic_capacity=False,
    )
    monkeypatch.setattr(
        candidate_columns,
        "column_layout",
        lambda *args, **kwargs: pytest.fail("must reject before column allocation"),
    )
    with pytest.raises(ContractError):
        candidate_columns.encode_candidate_columns(table, "wrong.bin")
    fake = SimpleNamespace(
        model=ModelConfigV2(), episodes=(SimpleNamespace(records=SimpleNamespace(tables=(table,))),)
    )
    monkeypatch.setattr(
        writer,
        "create_output_directory",
        lambda *args: pytest.fail("must reject before publication"),
    )
    with pytest.raises(ContractError):
        writer.write_rollout(tmp_path / "unsupported", fake)
    synthetic = CandidateTable.from_actions(
        frame,
        [{"type": "play_sequence", "ordered_card_refs": [frame.references[0]] * 128, "target": 2}],
        action_contract=contract,
        purpose="component",
        synthetic_capacity=True,
    )
    with pytest.raises(ContractError):
        synthetic.to_action(0, frame, validate=False)


def test_corrected_config_release_paths_pairing_and_guarded_ppo_integration(tmp_path):
    import asyncio

    import gf_cli
    from gf_cli.config import CorrectedServerConfigV2, ServerConfigV2, load_config

    preset = ROOT / "configs/experiments/card-instance-v2-corrected-contract.toml"
    loaded = load_config(preset, {"run": {"output_dir": str(tmp_path / "unused")}})
    assert type(loaded.config.server) is CorrectedServerConfigV2
    assert (
        type(load_config(ROOT / "configs/experiments/card-instance-v2-contract.toml").config.server)
        is ServerConfigV2
    )
    assert loaded.config.actions.action_contract == corrected_card_instance_contract()
    assert loaded.config.server.binary == str(
        (ROOT.parent / "GodFieldSimurator/target/release/gf-test-server").resolve()
    )
    assert loaded.config.server.simulator_revision == CORRECTED_SIMULATOR_REVISION
    for overrides in (
        {"server": {"simulator_revision": card_instance_contract().simulator_revision}},
        {"actions": {"action_contract": asdict(card_instance_contract())}},
        {"actions": {"action_contract": {"profile_id": card_instance_contract().profile_id}}},
    ):
        with pytest.raises(ContractError):
            load_config(preset, overrides)
    with pytest.raises(ContractError, match="not implemented"):
        asyncio.run(gf_cli.run_experiment(loaded, "collect", ledger=object()))
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("missing", ["revision", "profile"])
def test_corrected_config_requires_explicit_raw_full_pin(tmp_path, missing):
    from gf_cli.config import load_config

    action = (ROOT / "configs/actions/card-instance-v2-corrected.toml").read_text()
    if missing == "profile":
        action = "\n".join(
            line for line in action.splitlines() if not line.startswith("profile_id")
        )
    own_action = tmp_path / "actions.toml"
    own_action.write_text(action)
    text = 'schema_version=1\nkind="experiment"\n[references]\nactions="actions.toml"\n[server]\ngameplay_api="v2"\nfixed_seed_experiment=true\n'
    if missing != "revision":
        text += 'simulator_revision="' + CORRECTED_SIMULATOR_REVISION + '"\n'
    candidate = tmp_path / "config.toml"
    candidate.write_text(text)
    with pytest.raises(ContractError):
        load_config(candidate)


def test_corrected_minimal_config_default_paths_resolve_at_origin(tmp_path):
    from gf_cli.config import load_config

    candidate = tmp_path / "nested" / "config.toml"
    candidate.parent.mkdir()
    candidate.write_text(
        'schema_version=1\nkind="experiment"\n[references]\nactions='
        + json.dumps(str(ROOT / "configs/actions/card-instance-v2-corrected.toml"))
        + '\n[server]\ngameplay_api="v2"\nsimulator_revision="'
        + CORRECTED_SIMULATOR_REVISION
        + '"\nfixed_seed_experiment=true\n'
    )
    loaded = load_config(candidate)
    assert loaded.config.server.binary == str(
        (candidate.parent / "../GodFieldSimurator/target/release/gf-test-server").resolve()
    )
    assert loaded.config.server.config == str(candidate.parent / "configs/test-server.toml")
    assert loaded.config.run.output_dir == str(candidate.parent / "runs")


def test_corrected_raw_identity_goldens_preserved_and_incomplete_tokens_superseded():
    result, _, frame, bank, _ = generate()
    expected = json.loads(
        (ROOT / "tests/fixtures/corrected_card_instance_identity.json").read_text()
    )
    assert asdict(corrected_card_instance_contract()) == expected["contract"]
    assert bank.view()._prefixes[0] == expected["initial_prefix"]
    assert frame.frame_hash == expected["frame_hash"]
    assert result.table.table_hash == expected["table_hash"]
    assert signature(layout=NEW) != expected["token_signature"]
    assert signature(layout=NEW) == "ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f"
    assert result.report.rule_signature == expected["rule_signature"]
    tokens = tokenize_frame(frame, bank.view(), layout=NEW)
    assert digest(asdict(tokens)) != expected["frame_tokens_sha256"]
    assert digest(asdict(tokens)) == "2c3d318a4a4cb047f68c87b9721db7be0a5c9595e07e82fb4fa2b84bd4bb8d20"
    assert (
        digest(asdict(tokenize_table(result.table, frame, layout=NEW)))
        == "64e8778719518c7b6fddc0069a0894a238ca1e4710db4d4c864027cebf61c4be"
    )


BEFORE = {
    "base_revision": "be02f01ef4036e26e8b10c3b4825b2c605fdd7f1",
    "default_config_sha256": "bdb2b3663397693baf9bc059e9831c17ea91cbc85a3184e30d3d2f1aa6f0fa33",
    "actions3_sha256": "dbfc491028e3321dbcfd6a6ff52cfa9234b48b38a7199eca3c03a8fcfd688752",
    "contract": {
        "card_ref_scheme": "aes256-card-ref-v1",
        "catalog_schema_version": 2,
        "catalog_sha256": "f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6",
        "catalog_version": "public-ja-v3",
        "gameplay_api": "v2",
        "input_recipe": "participant-structured-v2",
        "policy_input_version": "player-card-refs-v2",
        "profile_id": "card-instance-simulator-v2-20261103",
        "rule_set_version": "true-match-1v1-v2",
        "simulator_revision": "07e8185f98c503aa41200a80ea554b4fd21088ad",
        "spec_inventory_sha256": "8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5",
    },
    "old_rule_signature": "1480f832b98c0b78b894a823681848b1dc11093b3a6a2aa462dd2069ed579a36",
    "token_signatures": {
        "07e": {
            "128": "ade08b702322a313b92a7cbff53a7875f3918000cc613c7c948174770c20f346",
            "2": "9cf03e53ed7e3d8257fbd462a507a7d2e3418634d833fd463410a59cbc0ba083",
        },
        "legacy": {
            "128": "4b4ae064f08192a322de83e24e4a93ef4191c7d73396997d59cd7710cd05cb04",
            "2": "ddcc85ccb548b7cdf109839033a01ab0c3b5cf96669ad0994b6a1941e7e02490",
        },
    },
    "goldens": {
        "tests/fixtures/card_instance_legacy_generator_golden.json": "ed0c5a377d7db69ce4f3662fbc9bac19a694d88127833259070bd132ac9a2dfd",
        "tests/fixtures/card_instance_legacy_structured_golden.json": "cd4d82af32f048358fa2f18cef43f12b0ff5a79c7b5fdfe46c282fa855218745",
        "tests/fixtures/card_instance_v2_tokens_golden.json": "d7a8f369f5adbe573182326926c456053f25251e10d51488cc864a805b03959e",
    },
    "experiments": {
        "configs/experiments/baseline.toml": "71e189b3dca2cbfa16560655354950a4cf3250c40202430909467b5281ddb70b",
        "configs/experiments/card-instance-v2-contract.toml": "2705f95c8b470b8e4464f76569353d51c2adcdf006e90c380795c5de40979e3c",
        "configs/experiments/gru-ppo-cpu.toml": "9c7fee094e3ea58aabe9d03910845950e9cb93516aa43f96aa12644f4901a830",
        "configs/experiments/gru-ppo.toml": "3cae8d42b7f983e215634ad9214f2693dd00bfbce9d2273bd1d1ac5c17ab80a8",
        "configs/experiments/lightgbm-fixed-history.toml": "aaec03ce9e1d9905f85f49198089b1ec2a7cf968036365cbc68199fc3826fa2a",
        "configs/experiments/lightgbm-fixed-observation.toml": "467c973d1b8f4132409639fccd2d68b4e9ce51258ca953da6ef2bc5a9d52c5a6",
        "configs/experiments/lightgbm.toml": "397d5a92dada69d867a7b1841055948b996cbfc61da306c13bb8ab76a1becd8a",
        "configs/experiments/pilot.toml": "1fb48a5493f0d183e326c7184018cbcb0a51cccd1e718ba9a61bf75131f32385",
        "configs/experiments/ppo-cpu.toml": "07207648d5936705e1f0c7c8efd60ef45d0494c2ad7c69c60779a88457f676ef",
        "configs/experiments/ppo.toml": "73a4d5326378043cf0f7421fbe06674bd7f14c8503d6f9eae39be15697ca8456",
        "configs/experiments/smoke.toml": "5834ff038d225c98c2a0ac6ef1e219a52122e61c7fdd59b7adde13d3ec4b9eb9",
        "configs/experiments/transformer-ppo-cpu.toml": "0e32e428cd91bb3b6c04dd90f1e7f92fdef8245ae4c39c91bcd1d1fbaa73265f",
        "configs/experiments/transformer-ppo.toml": "60df8fdb3eba6c266a99b194b7f8a4a18e2e919b6cf9b56e0f9393bf0acefc34",
    },
}
