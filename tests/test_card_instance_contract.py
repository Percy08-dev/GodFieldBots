"""Original v2 public DTO fixtures and strict cross-version boundary checks."""

import copy
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from gf_common import card_instance_validation as wire
from gf_common.card_instance import (
    REVIEWED_SPEC_HASHES,
    SPEC_INVENTORY_HASH,
    ActionConfigV3,
    CardInstanceContract,
    action_config_v3,
    card_instance_contract,
    parse_card_instance_contract,
)
from gf_common.domain import ContractError, digest

REF1, REF2 = "c1_" + "01" * 16, "c1_" + "02" * 16


def fixture(kind):
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"card_instance_v2_{kind}.json").read_text()
    )


def relocate_effective_paths(effective, *, source_parent, target_parent):
    """Test-only origin mapping; production configuration/hashes stay untouched."""
    result = copy.deepcopy(effective)
    path_fields = {
        "run": ("output_dir",),
        "server": ("binary", "config", "source_dir"),
        "model": ("checkpoint",),
        "opponent": ("checkpoint",),
        "training": ("pilot_run", "dataset_manifest"),
    }

    def relocate(value):
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(source_parent):
            return str(target_parent / path.relative_to(source_parent))
        return value

    for table, names in path_fields.items():
        for name in names:
            if name in result[table]:
                result[table][name] = relocate(result[table][name])
    for name in ("train_runs", "validation_runs"):
        if name in result["training"]:
            result["training"][name] = [relocate(value) for value in result["training"][name]]
    return result


def observation(*, actor=1, viewer=1, fog=False):
    status = {"disease": None, "fog": fog, "flash": False, "dream": False, "dark_cloud": False}
    opponent = {
        "player_id": 3 - viewer,
        "learned_miracles": ["fireball", "fireball"],
        "ascended": False,
    }
    if not fog:
        opponent.update(hp=40, mp=0, money=0, status={**status, "fog": False})
    actionable = actor == viewer
    return {
        "observation_schema_version": "2",
        "rule_set_version": "true-match-1v1-v2",
        "card_ref_scheme": "aes256-card-ref-v1",
        "action_schema_version": 2,
        "actionable": actionable,
        "typed_action_schema": wire.action_descriptors("turn_action") if actionable else [],
        "state_revision": 7,
        "viewer_player_id": viewer,
        "turn": 4,
        "turns_until_end_of_world": 146,
        "first_player": 1,
        "active_player": actor,
        "current_player": actor,
        "phase": "awaiting_action",
        "required_choice": "turn_action",
        "end_times": False,
        "result": None,
        "own": {
            "player_id": viewer,
            "hp": 40,
            "mp": 0,
            "money": 0,
            "cards": [
                {"kind": "artifact", "card_ref": REF1, "definition_id": "bronze-club"},
                {
                    "kind": "miracle",
                    "card_ref": REF2,
                    "definition_id": "fireball",
                    "state": "learned",
                },
            ],
            "status": status,
            "guardian": None,
            "ascended": False,
        },
        "opponent": opponent,
    }


def options(obs=None):
    obs = observation() if obs is None else obs
    rows = [
        {"card_ref": card["card_ref"], "definition_id": card["definition_id"]}
        for card in obs["own"]["cards"]
    ]
    return {
        "action_options_schema_version": "2",
        "rule_set_version": "true-match-1v1-v2",
        "card_ref_scheme": "aes256-card-ref-v1",
        "actor": obs["viewer_player_id"],
        "state_revision": obs["state_revision"],
        "required_choice": obs["required_choice"],
        "usable_cards": rows,
        "candidate_groups": {"new-public-group": rows, "overlap": rows},
        "exchange_cards": [],
        "sell_cards": [],
        "buy_cards": [],
        "phenomenon_cards": [],
        "valid_targets": [1, 2],
        "can_pray": True,
        "can_discard": True,
        "can_sacrifice": False,
        "can_allow": False,
    }


def replay(*, aborted=False, concealed=False):
    obs = observation(actor=2 if concealed else 1)
    if aborted:
        obs.update(actionable=False, typed_action_schema=[])
    return {
        "game_id": "game",
        "viewer_player_id": 1,
        "requested_turn": None,
        "rule_set_version": "true-match-1v1-v2",
        "observation_schema_version": "2",
        "public_event_schema_version": "2",
        "card_ref_scheme": "aes256-card-ref-v1",
        "episode_end": {
            "type": "aborted",
            "reason": "turn_limit_exceeded",
            "player": None,
            "decision": {"turn": 4, "decision_index": None, "invalid_attempt_count": 0},
        }
        if aborted
        else None,
        "turns": [
            {
                "turn": 4,
                "active_player": 2 if concealed else 1,
                "decisions": [
                    {
                        "actor": 2 if concealed else 1,
                        "observation": obs,
                        "action_options": None if concealed else options(obs),
                        "aborted": aborted,
                        "visible_action": {"type": "concealed"}
                        if concealed
                        else {"type": "action", "action": {"type": "pray"}},
                        "visible_events": [{"type": "safe", "player": 1}],
                    }
                ],
            }
        ],
    }


def test_contract_closed_pin_inventory_and_search_roundtrip():
    contract = card_instance_contract()
    assert len(fields(contract)) == 11
    assert parse_card_instance_contract(asdict(contract)) == contract
    assert digest(REVIEWED_SPEC_HASHES) == SPEC_INVENTORY_HASH
    actions = ActionConfigV3(action_contract=contract)
    assert action_config_v3(json.loads(json.dumps(asdict(actions)))) == actions
    for field in fields(contract):
        raw = asdict(contract)
        raw[field.name] = True if field.name == "catalog_schema_version" else "unknown"
        with pytest.raises(ContractError):
            parse_card_instance_contract(raw)
    with pytest.raises(ContractError):
        parse_card_instance_contract({**asdict(contract), "eligibility": "qualified"})
    with pytest.raises(ContractError):
        ActionConfigV3(action_contract=asdict(contract))
    for name, value in (
        ("version", 3),
        ("generator", "public-atomic-diverse-v4"),
        ("use_action_options", 1),
        ("max_candidates", True),
        ("channel_weights", (1,)),
    ):
        with pytest.raises(ContractError):
            ActionConfigV3(action_contract=contract, **{name: value})


def test_captured_public_catalog_and_info_hashes():
    catalog = fixture("catalog")
    assert len(wire.catalog(catalog)) == 296
    assert wire.public_info(fixture("info")) == fixture("info")
    origin = fixture("origin")
    for filename, expected in origin["files"].items():
        assert (
            hashlib.sha256((Path(__file__).parent / "fixtures" / filename).read_bytes()).hexdigest()
            == expected
        )
    for candidate in (
        {**catalog, "catalog_schema_version": True},
        {**catalog, "catalog_schema_version": "2"},
    ):
        with pytest.raises(ContractError):
            wire.catalog(candidate)
    bad = copy.deepcopy(catalog)
    bad["definitions"][0]["classification"]["use_roles"] = ["unknown"]
    with pytest.raises(ContractError):
        wire.catalog(bad)


def test_original_runtime_observation_options_events_replay():
    obs = fixture("runtime_observation")
    assert wire.observation(obs) == obs
    assert wire.actor_action_options(fixture("runtime_options"), obs) == fixture("runtime_options")
    # The runtime harness records the client's returned events/cursor projection.
    wire.validate_events(fixture("runtime_events")["events"])
    original_replay = fixture("runtime_replay")
    assert wire.replay(original_replay, original_replay["game_id"], 1) == original_replay


def test_original_card_order_kind_state_and_fog_duplicate_definitions():
    for fog in (True, False):
        obs = observation(fog=fog)
        assert wire.observation(obs) == obs
    for mutate in (
        lambda o: o["own"]["cards"][0].update(state="unused"),
        lambda o: o["own"]["cards"][1].pop("state"),
        lambda o: o["own"]["cards"][1].update(card_ref=REF1),
        lambda o: o["own"]["cards"][1].update(card_ref=1),
        lambda o: o["own"].update(hand=[]),
        lambda o: o["own"]["status"].update(fog=True),
    ):
        obs = observation()
        mutate(obs)
        with pytest.raises(ContractError):
            wire.observation(obs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_schema_version", 2),
        ("action_schema_version", "2"),
        ("action_schema_version", True),
        ("rule_set_version", "true-match-1v1-v1"),
        ("actionable", 1),
        ("state_revision", True),
        ("card_ref_scheme", "unknown"),
    ],
)
def test_observation_mixed_types_and_versions_reject(field, value):
    obs = observation()
    obs[field] = value
    with pytest.raises(ContractError):
        wire.observation(obs)


def test_all_required_fields_and_exact_metadata_reject_omission():
    for key in observation():
        bad = observation()
        del bad[key]
        with pytest.raises(ContractError):
            wire.observation(bad)
    obs = observation()
    obs["typed_action_schema"][1]["optional_fields"] = {"target": {"type": "null"}}
    with pytest.raises(ContractError):
        wire.observation(obs)


def test_hint_binding_overlap_and_unknown_group_are_preserved():
    obs = observation()
    hints = options(obs)
    raw = {"state_revision": 7, "action_options": hints}
    assert wire.actor_action_options(raw, obs) == raw
    for mutate in (
        lambda h: h.update(actor=2),
        lambda h: h.update(state_revision=8),
        lambda h: h.update(action_options_schema_version=2),
        lambda h: h["usable_cards"][0].update(definition_id="actual-secret"),
        lambda h: h["usable_cards"][0].update(card_ref="c1_" + "ff" * 16),
        lambda h: h["usable_cards"].append(h["usable_cards"][0]),
        lambda h: h.update(can_pray=1),
    ):
        hints = options(obs)
        mutate(hints)
        with pytest.raises(ContractError):
            wire.action_options(hints, obs)
    for key in options():
        hints = options()
        del hints[key]
        with pytest.raises(ContractError):
            wire.action_options(hints, obs)


@pytest.mark.parametrize(
    "action",
    [
        {"type": "play_sequence", "ordered_card_refs": [REF1, REF2], "target": 2},
        {"type": "respond", "ordered_card_refs": []},
        {"type": "discard", "card_refs": [REF1]},
        {"type": "sacrifice", "card_refs": [REF1]},
        {"type": "exchange", "card_ref": REF1, "hp": -1, "mp": 0, "money": 1},
        {"type": "sell", "card_ref": REF1, "sold_card_ref": REF2, "target": 2},
        {"type": "buy", "card_ref": REF1, "target": 2},
        {"type": "invoke_phenomenon", "card_ref": REF1},
        {"type": "buy_decision", "accept": False},
        {"type": "pray"},
    ],
)
def test_public_action_syntax_and_dataset_string_wrapper(action):
    assert wire.validate_action(action) == action
    assert wire.dataset_action(action, action_schema_version="2") == action
    for version in (2, True, "1"):
        with pytest.raises(ContractError):
            wire.dataset_action(action, action_schema_version=version)


@pytest.mark.parametrize(
    "action",
    [
        {"type": "play_sequence", "ordered_card_refs": [REF1]},
        {"type": "play_sequence", "ordered_card_refs": [REF1], "target": None},
        {"type": "play_sequence", "ordered_card_refs": [REF1, REF1], "target": 2},
        {"type": "respond_sequence", "ordered_defenses": []},
        {"type": "respond", "ordered_defenses": []},
        {"type": "discard", "instance_ids": [1]},
        {"type": "exchange", "card_ref": REF1, "hp": 0, "mp": 0, "money": 0, "target": 1},
        {"type": "buy_decision", "accept": 1},
    ],
)
def test_old_or_malformed_actions_reject(action):
    with pytest.raises(ContractError):
        wire.validate_action(action)


@pytest.mark.parametrize(
    "refs",
    [
        [REF1, REF2, REF1],
        ["c1_" + format(index, "032x") for index in range(19)],
    ],
)
def test_dataset_rejected_attempt_preserves_reference_order_and_multiplicity(refs):
    attempt = {"type": "play_sequence", "ordered_card_refs": refs, "target": 2}
    assert wire.dataset_action(attempt, action_schema_version="2") == attempt
    assert wire.dataset_action(attempt, action_schema_version="2")["ordered_card_refs"] == refs
    with pytest.raises(ContractError):
        wire.validate_action(attempt)
    with pytest.raises(ContractError):
        wire.action_command(
            {
                "command_id": "00000000-0000-0000-0000-000000000001",
                "expected_revision": 7,
                "action": attempt,
            }
        )


@pytest.mark.parametrize(
    "attempt",
    [
        {"type": "play_sequence", "ordered_card_refs": [REF1], "target": None},
        {"type": "play_sequence", "ordered_card_refs": [1], "target": 2},
        {"type": "play_sequence", "ordered_card_refs": REF1, "target": 2},
        {"type": "respond_sequence", "ordered_defenses": []},
        {"type": "buy_decision", "accept": 1},
    ],
)
def test_dataset_attempt_still_rejects_invalid_public_representation(attempt):
    with pytest.raises(ContractError):
        wire.dataset_action(attempt, action_schema_version="2")


def test_dataset_observation_is_its_own_original_representation():
    original = observation()
    dataset = {key: value for key, value in original.items() if key not in wire._HTTP_FIELDS}
    assert wire.dataset_observation(dataset) == dataset
    with pytest.raises(ContractError):
        wire.observation(dataset)
    with pytest.raises(ContractError):
        wire.dataset_observation(original)


@pytest.mark.parametrize("aborted,concealed", [(False, False), (True, False), (False, True)])
def test_replay_context_and_abort_closure(aborted, concealed):
    value = replay(aborted=aborted, concealed=concealed)
    assert wire.replay(value, "game", 1) == value
    if aborted:
        with pytest.raises(ContractError):
            wire.observation(value["turns"][0]["decisions"][0]["observation"])
        value["turns"][0]["decisions"][0]["aborted"] = False
        with pytest.raises(ContractError):
            wire.replay(value, "game", 1)


def test_all_25_public_events_and_fog_omission():
    rows = [
        {"type": "turn_started", "turn": 1, "active_player": 1},
        {"type": "action_accepted", "player": 1},
        *[
            {"type": name, "player": 1, "definition_id": "fireball"}
            for name in (
                "artifact_used",
                "artifact_drawn",
                "devil_appeared",
                "artifact_discarded",
                "guardian_set",
                "guardian_left",
                "guardian_acted",
            )
        ],
        {"type": "miracle_used", "player": 1, "definition_id": "fireball", "mp_spent": 0},
        {"type": "attack_declared", "attacker": 1, "target": 2, "attack": -1},
        {"type": "attack_missed", "attacker": 1, "target": 2},
        {"type": "damage_applied", "player": 1, "amount": -1},
        {"type": "healed", "player": 1, "amount": 1},
        {"type": "safe", "player": 1},
        {"type": "resource_changed", "player": 1, "hp_delta": -1, "mp_delta": 1, "money_delta": 0},
        {"type": "buy_offered", "buyer": 1, "seller": 2, "definition_id": "fireball", "price": 0},
        {"type": "sale_offered", "seller": 1, "target": 2, "definition_id": "fireball", "price": 0},
        {
            "type": "artifact_transferred",
            "from": 1,
            "to": 2,
            "definition_id": "fireball",
            "price": 0,
        },
        {"type": "sun_amulet_activated", "player": 1},
        {"type": "status_changed", "player": 1, "status": "dream", "active": True},
        {"type": "phenomenon_invoked", "definition_id": "fireball"},
        {"type": "player_ascended", "player": 1},
        {"type": "player_defeated", "player": 1},
        {"type": "game_ended", "result": "draw"},
    ]
    assert len(rows) == 25
    wire.validate_events(rows)
    for row in (
        {"type": "safe", "player": True},
        {"type": "safe", "player": 1, "card_ref": REF1},
        {"type": "status_changed", "player": 1, "status": "unknown", "active": True},
    ):
        with pytest.raises(ContractError):
            wire.validate_events([row])


def test_card_instance_contract_type_rejects_legacy_values():
    with pytest.raises(ContractError):
        CardInstanceContract(catalog_schema_version=True)


def test_root_config_new_descriptor_and_frozen_twelve_old_hashes():
    from gf_cli.config import ExperimentConfig, ServerConfigV2, load_config

    root = Path(__file__).parents[1]
    baseline = fixture("legacy_config_hashes")
    assert digest(asdict(ExperimentConfig())) == baseline["default_sha256"]
    assert len(baseline["experiments"]) == 12
    captured_parent = Path(baseline["original_parent"])
    assert Path(baseline["original_root"]).parent == captured_parent
    for name, expected in baseline["experiments"].items():
        loaded = load_config(root / name)
        assert loaded.sha256 == digest(loaded.effective)
        normalized = relocate_effective_paths(
            loaded.effective, source_parent=root.parent, target_parent=captured_parent
        )
        assert digest(normalized) == expected
        if root == Path(baseline["original_root"]):
            assert loaded.sha256 == expected
    new = load_config(root / "configs/experiments/card-instance-v2-contract.toml")
    assert type(new.config.actions) is ActionConfigV3
    assert type(new.config.server) is ServerConfigV2
    assert new.config.server.simulator_revision == card_instance_contract().simulator_revision
    assert new.config.server.gameplay_api == "v2"


def test_legacy_twelve_hashes_survive_simulated_checkout_relocation(tmp_path):
    from gf_cli.config import load_config

    root = Path(__file__).parents[1]
    baseline = fixture("legacy_config_hashes")
    captured_parent = Path(baseline["original_parent"])
    relocated_parent = tmp_path / "relocated-projects"
    for name, expected in baseline["experiments"].items():
        original = load_config(root / name).effective
        relocated = relocate_effective_paths(
            original, source_parent=root.parent, target_parent=relocated_parent
        )
        assert digest(relocated) != digest(original)
        restored = relocate_effective_paths(
            relocated, source_parent=relocated_parent, target_parent=captured_parent
        )
        assert digest(restored) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"server": {"gameplay_api": "v1"}},
        {"server": {"simulator_revision": "unknown"}},
        {"actions": {"version": "2"}},
        {"model": {"version": "3", "model_id": "mlp"}},
        {"model": {"version": "1", "model_id": "mlp"}},
        {"opponent": {"version": "2", "model_id": "mlp"}},
        {"preprocessing": {"reader": "structured-v1"}},
        {"preprocessing": {"backend": "rust"}},
        {"actions": {"action_contract": {"eligibility": "qualified"}}},
    ],
)
def test_root_config_mixed_descriptor_or_unimplemented_pipeline_rejects(overrides):
    from gf_cli.config import load_config

    preset = Path(__file__).parents[1] / "configs/experiments/card-instance-v2-contract.toml"
    with pytest.raises(ContractError):
        load_config(preset, overrides)


@pytest.mark.parametrize("missing", ["gameplay_api", "simulator_revision", "action_contract"])
def test_root_config_requires_explicit_raw_descriptor_fields(tmp_path, missing):
    from gf_cli.config import load_config

    root = Path(__file__).parents[1]
    source = (root / "configs/experiments/card-instance-v2-contract.toml").read_text()
    source = source.replace(
        '"../actions/card-instance-v2.toml"',
        json.dumps(str(root / "configs/actions/card-instance-v2.toml")),
    )
    source = source.replace(
        '"../models/heuristic.toml"', json.dumps(str(root / "configs/models/heuristic.toml"))
    )
    source = source.replace(
        '"../models/random.toml"', json.dumps(str(root / "configs/models/random.toml"))
    )
    if missing == "action_contract":
        source = source.replace(
            "actions = " + json.dumps(str(root / "configs/actions/card-instance-v2.toml")), ""
        )
        source += '\n[actions]\nversion="3"\n'
    else:
        source = "\n".join(
            line for line in source.splitlines() if not line.startswith(missing + " =")
        )
    candidate = tmp_path / "config.toml"
    candidate.write_text(source)
    with pytest.raises(ContractError):
        load_config(candidate)


def test_root_minimal_config_resolves_new_default_paths_at_its_origin(tmp_path):
    from gf_cli.config import load_config

    root = Path(__file__).parents[1]
    candidate = tmp_path / "nested" / "minimal.toml"
    candidate.parent.mkdir()
    candidate.write_text(
        'schema_version=1\nkind="experiment"\n[references]\n'
        + "actions="
        + json.dumps(str(root / "configs/actions/card-instance-v2.toml"))
        + "\n"
        + '[server]\ngameplay_api="v2"\nsimulator_revision="'
        + card_instance_contract().simulator_revision
        + '"\nfixed_seed_experiment=true\n'
    )
    loaded = load_config(candidate)
    assert loaded.config.server.binary == str(
        (candidate.parent / "../GodFieldSimurator/target/debug/gf-test-server").resolve()
    )
    assert loaded.config.server.config == str(candidate.parent / "configs/test-server.toml")
    assert loaded.config.run.output_dir == str(candidate.parent / "runs")
    assert loaded.config.actions.action_contract == card_instance_contract()


def test_root_run_guards_precede_allocation_and_io(tmp_path, monkeypatch):
    import asyncio

    import gf_cli
    from gf_cli.config import load_config

    root = Path(__file__).parents[1]
    loaded = load_config(
        root / "configs/experiments/card-instance-v2-contract.toml",
        {"run": {"output_dir": str(tmp_path / "unused")}},
    )
    monkeypatch.setattr(
        gf_cli, "atomic_json", lambda *args, **kwargs: pytest.fail("must reject before publication")
    )
    for entry in (gf_cli.run_experiment, gf_cli._run_experiment):
        with pytest.raises(ContractError, match="not implemented"):
            asyncio.run(entry(loaded, "collect"))
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "command", ["smoke", "collect", "evaluate", "benchmark", "train", "validate-policy"]
)
def test_root_main_guards_actions3_before_execution(command, monkeypatch, capsys):
    import gf_cli

    monkeypatch.setattr(
        gf_cli,
        "run_experiment",
        lambda *args, **kwargs: pytest.fail("must reject before execution"),
    )
    preset = Path(__file__).parents[1] / "configs/experiments/card-instance-v2-contract.toml"
    args = [command, "--config", str(preset)]
    if command == "validate-policy":
        args += ["--parent-run", "unused", "--output", "unused"]
    assert gf_cli.main(args) == 2
    assert "not implemented" in json.loads(capsys.readouterr().err)["message"]


def test_direct_environment_actions3_guard_precedes_server_or_owner_creation(tmp_path, monkeypatch):
    from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment

    settings = EnvironmentSettings(
        "unused",
        "unused",
        tmp_path,
        "stage1",
        ActionConfigV3(action_contract=card_instance_contract()),
    )
    monkeypatch.setattr(
        SimulatorEnvironment,
        "_new_server",
        lambda *args, **kwargs: pytest.fail("must reject before server I/O"),
    )
    monkeypatch.setattr(
        SimulatorEnvironment,
        "_initialize",
        lambda *args, **kwargs: pytest.fail("must reject before owner initialization"),
    )
    with pytest.raises(ContractError, match="actions3 runtime is not implemented"):
        SimulatorEnvironment(settings, lambda *args: None, lambda *args: None)


def test_direct_legacy_worker_actions3_guard_precedes_factory_and_popen(monkeypatch):
    from gf_simulator_client import workers

    monkeypatch.setattr(
        workers.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must reject before Popen")
    )
    monkeypatch.setattr(
        workers,
        "factory_descriptor",
        lambda *args: pytest.fail("must reject before factory processing"),
    )
    with pytest.raises(ContractError, match="actions3 runtime is not implemented"):
        workers.ParticipantWorker(
            lambda *args: None,
            "random",
            17,
            ActionConfigV3(action_contract=card_instance_contract()),
        )
