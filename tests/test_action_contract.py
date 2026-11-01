"""Owner-profile compatibility and independent unique-reference component evidence."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli.config import load_config
from gf_common.action_contract import (
    ADOPTED_RULES_SHA256,
    OWNER_CONFIRMATION_SHA256,
    PENDING_COVERAGE,
    PROFILE_SHA256,
    ActionContract,
    engineering_contract,
    parse_action_contract,
    require_qualified_contract,
)
from gf_common.diverse_actions import GeneratedCandidates
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured import ActionConfigV2, CandidateTable, DecisionFrame
from tests.test_diverse_actions import generate
from tests.test_structured_records import frame_for

ROOT = Path(__file__).parents[1]


def test_packaged_profile_constants_match_immutable_canonical_evidence():
    owner = json.loads((ROOT / "docs/contracts/owner-20261101-utility-and-repetition.json").read_text())
    profile = json.loads((ROOT / "docs/contracts/owner-proposal-engineering-v1.json").read_text())
    assert digest(owner) == OWNER_CONFIRMATION_SHA256
    assert digest(profile) == PROFILE_SHA256
    assert digest({key: profile[key] for key in ("adopted_rules", "pending_coverage", "verification_claims")}) == ADOPTED_RULES_SHA256
    assert tuple(profile["pending_coverage"]) == PENDING_COVERAGE
    contract = engineering_contract()
    assert parse_action_contract(asdict(contract), expected=contract) == contract
    assert profile["profile_id"] == contract.profile_id


@pytest.mark.parametrize("field", list(ActionContract.__dataclass_fields__))
@pytest.mark.parametrize("value", [None, False, "", "0" * 64, "qualified_game"])
def test_unknown_zero_or_upgraded_contract_fields_reject(field, value):
    raw = asdict(engineering_contract())
    raw[field] = value
    with pytest.raises(ContractError):
        parse_action_contract(raw)


def test_required_component_contract_and_purpose_never_grant_source_authority(make_input):
    with pytest.raises(TypeError):
        ActionConfigV2()
    with pytest.raises(ContractError, match="engineering_only"):
        require_qualified_contract(engineering_contract())
    frame, _ = frame_for(make_input(learned=("fireball",)))
    action = {"type": "play_sequence", "ordered_elements": [frame.references[-1]] * 128}
    with pytest.raises(ContractError):
        CandidateTable.from_actions(frame, [action], action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    table = CandidateTable.from_actions(frame, [action], action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    assert table.synthetic_capacity and table.purpose == "component" and len(table.slots) == 128
    for purpose in ("qualified_game", "synthetic", "", None):
        with pytest.raises(ContractError):
            CandidateTable.from_actions(frame, [action], action_contract=engineering_contract(), purpose=purpose, synthetic_capacity=True)


def test_raw_config_requires_explicit_or_inherited_contract_before_default_merge(tmp_path):
    prefix = 'schema_version=1\nkind="experiment"\n[server]\nfixed_seed_experiment=true\n[preprocessing]\nreader="structured-v1"\n[actions]\nversion="2"\n'
    path = tmp_path / "experiment.toml"
    path.write_text(prefix)
    with pytest.raises(ContractError, match="explicitly declare"):
        load_config(path)
    contract = engineering_contract()
    preset = tmp_path / "actions.toml"
    lines = '\n'.join(f'{key}="{value}"' for key, value in asdict(contract).items())
    preset.write_text('schema_version=1\nkind="actions"\n[actions]\nversion="2"\n[actions.action_contract]\n' + lines)
    path.write_text(prefix + '[references]\nactions="actions.toml"\n')
    loaded = load_config(path, {"actions": {"max_candidates": 100}})
    assert loaded.config.actions.action_contract == contract and loaded.effective["actions"]["action_contract"] == asdict(contract)
    assert len(loaded.inputs) == 2
    with pytest.raises(ContractError):
        load_config(path, {"actions": {"action_contract": {"eligibility": "production"}}})


def test_independent_hand_copy_and_learned_reference_permutation_oracle(make_input):
    value = make_input(hand=("bronze-club", "fireball", "fireball"), learned=("aura",), mp=20)
    result, actions = generate(value, max_sequence_length=128)
    refs = (("hand_artifact", 1), ("hand_artifact", 2), ("hand_artifact", 3), ("learned_miracle", "aura"))
    expected = set()
    for length in range(1, 5):
        for sequence in permutations(refs, length):
            if sequence[0] == refs[3] or refs[0] in sequence[1:]:
                continue
            for target in (1, 2):
                expected.add((sequence, target))
    actual = {(tuple((item["source"], item.get("instance_id", item.get("definition_id"))) for item in action["ordered_elements"]), action["target"])
              for action in actions if action["type"] == "play_sequence"}
    assert actual == expected
    assert all(len(set(row)) == len(row) for row, _ in actual)
    assert result.table.synthetic_capacity is False and result.report.action_contract == result.table.action_contract
    assert result.report.omissions == PENDING_COVERAGE


def test_learned_reference_reuse_is_per_decision_and_normal_copy_exception_stays_pending(make_input):
    import random

    from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV2
    value = make_input(hand=("bronze-club",), learned=("fireball",))
    frame, bank = frame_for(value)
    config = ActionConfigV2(action_contract=engineering_contract())
    generator = PublicCandidateGeneratorV2(config, clock=lambda: 0.)
    first = generator.generate(frame, bank.view(), random.Random(0), NumericCosts())
    obs, hints = value.to_dict()["observation"], value.to_dict()["action_options"]
    obs["state_revision"] = hints["state_revision"] = 1
    later = DecisionFrame.from_input(replace(value, observation=obs, action_options=hints), bank, frame.boundary.end)
    second = generator.generate(later, bank.view(), random.Random(1), NumericCosts())
    for generated in (first, second):
        assert any(generated.table.slots[start:end] == (1,) for start, end in zip(generated.table.offsets[:-1], generated.table.offsets[1:], strict=True))
        assert all(len(set(generated.table.slots[start:end])) == end - start for start, end in zip(generated.table.offsets[:-1], generated.table.offsets[1:], strict=True))
    _, copies = generate(make_input(hand=("flame", "flame")), max_sequence_length=2)
    assert {tuple(item["instance_id"] for item in action["ordered_elements"]) for action in copies if action["type"] == "play_sequence"} == {(1,), (2,)}


@pytest.mark.parametrize("name", ["smile-dew", "heart-dew", "romance-water", "galaxy-geyser", "smile-flower", "heart-flower",
    "romance-fragrance", "spring", "treasure", "tone", "song", "smile-shell", "heart-shell"])
def test_adopted_recovery_resource_cleanse_singletons_have_explicit_targets_and_allow_noop(make_input, name):
    value = make_input(hand=(name,))
    obs = value.to_dict()["observation"]
    obs["own"]["hp"] = 99
    _, actions = generate(replace(value, observation=obs))
    plays = [action for action in actions if action["type"] == "play_sequence"]
    assert plays == [{"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}], "target": target} for target in (2, 1)]


@pytest.mark.parametrize("name", ["thump-thump-tear", "heaven-herb"])
def test_user_fixed_effect_utilities_select_self_without_inventing_recipient_fields(make_input, name):
    _, actions = generate(make_input(hand=(name,)))
    plays = [action for action in actions if action["type"] == "play_sequence"]
    assert plays == [{"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}], "target": 1}]


@pytest.mark.parametrize("name", ["guardian-pot", "release"])
@pytest.mark.parametrize("visible", ["absent", "present", "unknown"])
def test_guardian_singletons_require_explicit_public_absence(make_input, name, visible):
    value = make_input(hand=(name,))
    obs = value.to_dict()["observation"]
    obs["opponent"]["guardian"] = None
    if visible == "present":
        obs["opponent"]["guardian"] = "dragon"
    elif visible == "unknown":
        obs["own"]["status"]["fog"] = True
        obs["opponent"] = {key: obs["opponent"][key] for key in ("player_id", "learned_miracles", "ascended")}
    _, actions = generate(replace(value, observation=obs))
    plays = [action for action in actions if action["type"] == "play_sequence"]
    assert {action["target"] for action in plays} == ({1, 2} if visible == "absent" else {1})


@pytest.mark.parametrize("learned_count", [0, 1, 2])
def test_soap_requires_two_visible_learned_refs_per_chosen_target(make_input, learned_count):
    value = make_input(hand=("goddess-s-soap",), learned=("aura", "fireball"))
    obs = value.to_dict()["observation"]
    obs["opponent"]["learned_miracles"] = ["flame", "fireball"][:learned_count]
    _, actions = generate(replace(value, observation=obs))
    targets = {action["target"] for action in actions if action["type"] == "play_sequence" and action["ordered_elements"] == [{"source": "hand_artifact", "instance_id": 1}]}
    assert targets == ({1, 2} if learned_count == 2 else {1})


@pytest.mark.parametrize("remaining", [0, 2, 3])
def test_broom_uses_only_remaining_own_hand_count_and_never_opponent_count(make_input, remaining):
    _, actions = generate(make_input(hand=("nocturnal-broom",) + ("steel-armor",) * remaining))
    targets = {action["target"] for action in actions if action["type"] == "play_sequence"}
    assert targets == ({1} if remaining == 3 else set())


def test_generated_game_proposals_reject_synthetic_marker_even_with_consistent_hashes(make_input):
    result, _ = generate(make_input(hand=("bronze-club",)))
    fields = result.table._content()
    fields["synthetic_capacity"] = True
    synthetic = CandidateTable(**{**fields, "action_contract": engineering_contract()}, table_hash=digest(fields))
    with pytest.raises(ContractError):
        GeneratedCandidates(synthetic, replace(result.report, table_hash=synthetic.table_hash))


@pytest.mark.parametrize("entry", ["collect", "evaluate", "smoke", "benchmark", "arbitrary-real-operation", "ppo", "fit", "fixed_fit", "direct_env"])
def test_engineering_real_entry_guards_precede_files_processes_requests_or_learner(tmp_path, monkeypatch, entry):
    import gf_cli
    from gf_simulator_client import environment
    from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment
    actions = ActionConfigV2(action_contract=engineering_contract())
    loaded = SimpleNamespace(config=SimpleNamespace(actions=actions))
    def forbidden(*args, **kwargs):
        pytest.fail("engineering profile reached real I/O or learner construction")
    monkeypatch.setattr(environment, "ManagedServer", forbidden)
    monkeypatch.setattr(environment, "AttestedClient", forbidden)
    with pytest.raises(ContractError, match="engineering_only"):
        if entry == "direct_env":
            SimulatorEnvironment(EnvironmentSettings("unused", "unused", tmp_path / "never", "unused", actions), forbidden, forbidden)
        elif entry == "ppo":
            from gf_training.ppo_run import train_ppo
            asyncio.run(train_ppo(loaded, forbidden, tmp_path / "never"))
        elif entry == "fit":
            from gf_training.lightgbm_run import train_experiments
            asyncio.run(train_experiments(loaded))
        elif entry == "fixed_fit":
            from gf_training.lightgbm_run import train_fixed_data
            asyncio.run(train_fixed_data(loaded))
        else:
            asyncio.run(gf_cli.run_experiment(loaded, entry))
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("field", ["action_contract", "purpose", "synthetic_capacity"])
def test_codec_and_wire_require_explicit_contract_purpose_and_capacity_classification(make_input, field):
    from gf_common.candidate_columns import decode_candidate_columns, encode_candidate_columns
    from gf_common.structured_wire import parse_table
    frame, _ = frame_for(make_input(hand=("bronze-club",)))
    table = CandidateTable.from_actions(frame, [{"type": "play_sequence", "ordered_elements": list(frame.references)}],
        action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    payload, descriptor = encode_candidate_columns(table, "owned.columns")
    expected = {"action_contract": engineering_contract(), "purpose": "component", "synthetic_capacity": False}
    assert decode_candidate_columns(payload, descriptor, **expected) == table
    assert parse_table(asdict(table), **expected) == table
    missing = dict(descriptor)
    del missing[field]
    with pytest.raises(ContractError):
        decode_candidate_columns(payload, missing, **expected)
    raw = asdict(table)
    del raw[field]
    with pytest.raises(ContractError):
        parse_table(raw, **expected)


@pytest.mark.parametrize("mutation", ["profile", "purpose", "capacity", "bool_capacity"])
def test_rehashed_codec_and_wire_cannot_upgrade_or_misclassify_profiles(make_input, mutation):
    from gf_common.candidate_columns import decode_candidate_columns, encode_candidate_columns
    from gf_common.structured_wire import parse_table
    frame, _ = frame_for(make_input())
    table = CandidateTable.from_actions(frame, [{"type": "pray"}], action_contract=engineering_contract(),
                                        purpose="component", synthetic_capacity=False)
    payload, descriptor = encode_candidate_columns(table, "owned.columns")
    raw = asdict(table)
    for record in (descriptor, raw):
        if mutation == "profile":
            record["action_contract"]["eligibility"] = "production"
        elif mutation == "purpose":
            record["purpose"] = "qualified_game"
        else:
            record["synthetic_capacity"] = True if mutation == "capacity" else 0
    raw["table_hash"] = digest({key: value for key, value in raw.items() if key != "table_hash"})
    with pytest.raises(ContractError):
        parse_table(raw, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    with pytest.raises(ContractError):
        decode_candidate_columns(payload, descriptor, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def test_synthetic_codec_128_preserves_exact_payload_without_qualifying_game_refs(make_input):
    from gf_common.candidate_columns import decode_candidate_columns, encode_candidate_columns
    frame, _ = frame_for(make_input(learned=("fireball",)))
    table = CandidateTable.from_actions(frame, [{"type": "play_sequence", "ordered_elements": [frame.references[-1]] * 128}],
        action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    payload, descriptor = encode_candidate_columns(table, "synthetic.columns")
    assert decode_candidate_columns(payload, descriptor, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True) == table
    with pytest.raises(ContractError):
        decode_candidate_columns(payload, descriptor, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def test_game_wire_compares_config_contract_and_rejects_synthetic_capacity(make_input):
    from gf_common.structured_wire import generated_dict, parse_generated
    result, _ = generate(make_input(hand=("bronze-club",)))
    raw = generated_dict(result)
    assert parse_generated(raw, action_contract=engineering_contract()) == result
    for target in ("table", "report"):
        missing = json.loads(json.dumps(raw))
        del missing[target]["action_contract"]
        with pytest.raises(ContractError):
            parse_generated(missing, action_contract=engineering_contract())
    fields = result.table._content()
    fields["synthetic_capacity"] = True
    raw["table"] = {**fields, "table_hash": digest(fields)}
    raw["report"]["table_hash"] = raw["table"]["table_hash"]
    with pytest.raises(ContractError):
        parse_generated(raw, action_contract=engineering_contract())


@pytest.mark.parametrize("command", ["smoke", "collect", "evaluate", "benchmark", "train"])
def test_cli_real_commands_cannot_bypass_engineering_guard(tmp_path, monkeypatch, capsys, command):
    import gf_cli
    path = tmp_path / "engineering.toml"
    contract = '\n'.join(f'{key}="{value}"' for key, value in asdict(engineering_contract()).items())
    path.write_text('schema_version=1\nkind="experiment"\n[server]\nfixed_seed_experiment=true\n'
        '[preprocessing]\nreader="structured-v1"\n[actions]\nversion="2"\n[actions.action_contract]\n' + contract)
    def forbidden(*args, **kwargs):
        pytest.fail("real command passed the engineering profile gate")
    monkeypatch.setattr(gf_cli, "run_experiment", forbidden)
    arguments = [command, "--config", str(path)]
    if command == "train":
        arguments += ["--run-dir", str(tmp_path / "never")]
    assert gf_cli.main(arguments) == 2
    assert "engineering_only" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("name", ["tone", "song", "spring", "release"])
def test_new_miracle_utility_singleton_consumes_printed_mp_not_hidden_free_completion(make_input, definitions, name):
    cost = definitions[name].get("mp_cost", 0)
    assert cost > 0
    _, paid = generate(make_input(hand=(name,), mp=cost))
    assert any(action["type"] == "play_sequence" for action in paid)
    _, unpaid = generate(make_input(hand=(name,), mp=cost - 1))
    assert not any(action["type"] == "play_sequence" for action in unpaid)


@pytest.mark.parametrize("entry", ["group", "episode"])
def test_existing_environment_rechecks_profile_before_each_real_entry(tmp_path, entry):
    from gf_common.domain import ActionConfig, EpisodeGroup, EpisodeSpec
    from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment
    def forbidden(*args, **kwargs):
        pytest.fail("existing Environment entered real I/O with engineering profile")
    settings = EnvironmentSettings("unused", "unused", tmp_path / "never", "run", ActionConfig())
    environment = SimulatorEnvironment(settings, forbidden, forbidden)
    environment.settings = replace(settings, actions=ActionConfigV2(action_contract=engineering_contract()))
    spec = EpisodeSpec("never", ("random", "random"), (0, 1), "pair")
    group = EpisodeGroup("not-a-game-seed", (spec,))
    with pytest.raises(ContractError, match="engineering_only"):
        if entry == "group":
            asyncio.run(environment.run_group(group))
        else:
            asyncio.run(environment._episode(SimpleNamespace(), {}, group, spec))
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("source", ["learned", "hand"])
def test_new_frame_rejects_ambiguous_physical_refs_while_v1_dto_semantics_stay_unchanged(make_input, source):
    from gf_common.validation import validate_worker_input
    value = make_input(hand=("fireball", "fireball"), learned=("fireball",))
    if source == "learned":
        value.observation["own"]["learned_miracles"] = ["fireball", "fireball"]
    else:
        value.observation["own"]["hand"][1]["instance_id"] = value.observation["own"]["hand"][0]["instance_id"]
        value.action_options["action_options"]["usable_hand_elements"] = [1]
    # Preserve both older behaviors: learned duplicates were schema-valid;
    # physical hand IDs were already unique in the v1 DTO contract.
    if source == "learned":
        assert validate_worker_input(value, True) is not None
    else:
        with pytest.raises(ContractError, match="duplicate hand instance"):
            validate_worker_input(value, True)
    with pytest.raises(ContractError, match="duplicate physical|duplicate hand"):
        frame_for(value)


def test_unambiguous_sources_preserve_distinct_hand_copies_and_hand_vs_learned(make_input):
    value = make_input(hand=("fireball", "fireball"), learned=("fireball",))
    frame, _ = frame_for(value)
    assert len({canonical(ref) for ref in frame.references}) == 3
    _, actions = generate(value, max_sequence_length=3)
    triples = [action["ordered_elements"] for action in actions if action["type"] == "play_sequence" and len(action["ordered_elements"]) == 3]
    assert len(triples) == 12  # 3! distinct physical-source orders ×2 explicit targets.
    assert all(len({canonical(ref) for ref in row}) == 3 for row in triples)
