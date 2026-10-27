"""ADR0005 singleton amendment, from permitted own-view HTTP fields only."""
import asyncio
import copy
import json
from dataclasses import replace

import pytest

from gf_common.actions import GENERATOR_VERSION, PublicActionSpace, PublicCandidateGenerator
from gf_common.domain import ActionConfig, ContractError, EmptyCandidates
from gf_common.validation import action_descriptors, actor_action_options, decision_input
from gf_models.baselines import make_policy
from gf_simulator_client.workers import ParticipantWorker
from gf_training.journal import read_journal
from tests.conftest import fixture
from tests.test_journal import journal_decision, new_journal


def captured_input(definitions):
    raw = fixture("magical_stick_atomic_public.json")
    obs = raw["observation"]
    return decision_input(obs, [], None, definitions, actor_action_options(raw["action_options_response"], obs, raw=True))


def singleton(target):
    return {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 10}], "target": target}


@pytest.mark.parametrize("strategy", ["enumerate", "beam"])
def test_captured_mp10_weapon_has_exact_two_standalone_actions(definitions, strategy):
    value = captured_input(definitions)
    assert value.observation["own"]["hand"] == [{"instance_id": 10, "definition_id": "magical-stick"}]
    assert value.observation["own"]["mp"] == 10
    assert value.catalog["magical-stick"]["ability"] == "atkBy2xMP"
    assert "attack" not in value.catalog["magical-stick"]
    config = ActionConfig(strategy=strategy)
    result = PublicCandidateGenerator(config).generate(value)
    assert result.actions == (singleton(2), singleton(1))  # configured opponent, self order
    assert result.attempts == 2 and result.public_prunes == 0
    assert result.generator_version == GENERATOR_VERSION == "public-atomic-subset-v3"
    assert not PublicActionSpace(value, config).offense


@pytest.mark.parametrize("targets,valid,expected", [
    (("opponent",), [1, 2], [2]),
    (("self",), [1, 2], [1]),
    (("self", "opponent"), [2], [2]),
    (("opponent", "self"), [1], [1]),
])
def test_target_configuration_intersects_atomic_targets(definitions, targets, valid, expected):
    value = captured_input(definitions)
    value.action_options["action_options"]["valid_targets"] = valid
    result = PublicCandidateGenerator(ActionConfig(targets=targets)).generate(value)
    assert result.actions == tuple(singleton(target) for target in expected)
    assert result.attempts == len(expected)


@pytest.mark.parametrize("gate", ["mask", "targets", "observation_only", "nonactionable", "defense", "syntax", "learned"])
def test_singleton_requires_current_supported_actor_syntax_mode_and_own_hand(definitions, gate):
    value = captured_input(definitions)
    settings = ActionConfig()
    if gate == "mask":
        value.action_options["action_options"]["usable_hand_elements"] = []
    elif gate == "targets":
        value.action_options["action_options"]["valid_targets"] = []
    elif gate == "observation_only":
        value = replace(value, action_options=None)
        settings = replace(settings, use_action_options=False)
    elif gate == "nonactionable":
        value.observation.update(current_player=2, actionable=False, typed_action_schema=[])
    elif gate == "defense":
        value.observation.update(required_choice="defense", phase="awaiting_response", typed_action_schema=action_descriptors("defense"))
        value.action_options["action_options"]["required_choice"] = "defense"
        value = replace(value, typed_metadata=tuple(value.observation["typed_action_schema"]))
    elif gate == "syntax":
        value = replace(value, typed_metadata=())
    else:
        # It is a hand-weapon extension, never an invented learned ability rule.
        value.observation["own"].update(hand=[], learned_miracles=["magical-stick"])
        value.action_options["action_options"].update(usable_hand_elements=[], usable_learned_miracles=["magical-stick"])
    with pytest.raises((ContractError, EmptyCandidates)):
        PublicCandidateGenerator(settings).generate(value)


def test_singleton_budget_counts_each_target_and_keeps_scorer_unchanged(definitions):
    value = captured_input(definitions)
    result = PublicCandidateGenerator(ActionConfig(max_expansions=1)).generate(value)
    assert result.actions == (singleton(2),)
    assert result.attempts == 1 and "max_expansions" in result.cutoff_reasons
    bounded = PublicCandidateGenerator(ActionConfig(max_candidates=1)).generate(value)
    assert bounded.attempts == 2 and bounded.approximate_drops == 1
    assert bounded.scorer_version == "public-content-v1"


def test_dynamic_weapon_never_enters_support_or_attack_combinations(make_input):
    value = make_input(("magical-stick", "bronze-club"), ("aura",))
    result = PublicCandidateGenerator(ActionConfig(strategy="enumerate", max_sequence_length=3)).generate(value)
    with_stick = [action for action in result.actions if any(element.get("instance_id") == 1 for element in action.get("ordered_elements", []))]
    assert with_stick == [{"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}], "target": target} for target in (2, 1)]
    assert any(len(action.get("ordered_elements", [])) > 1 for action in result.actions)
    assert not any(action["type"] == "pray" for action in result.actions)


def test_captured_case_runs_through_normal_worker_and_v3_journal(definitions, tmp_path):
    async def run():
        value = captured_input(definitions)
        worker = ParticipantWorker(make_policy, "heuristic", 0, ActionConfig(), 2)
        try:
            candidates, selected = await worker.select(value)
            assert candidates.actions == (singleton(2), singleton(1))
            assert candidates.actions[selected.index] in (singleton(1), singleton(2))
        finally:
            await worker.close()
        journal = new_journal(tmp_path)
        journal_decision(journal, value)
        journal.finish("incomplete", None)
        records = read_journal(journal.final_path)
        assert records[0]["schema_version"] == 4
        assert records[0]["generator_version"] == records[1]["candidates"]["generator_version"] == "public-atomic-subset-v3"
        assert records[1]["input"] == value.to_dict()
        old = copy.deepcopy(records)
        old[0]["generator_version"] = old[1]["candidates"]["generator_version"] = "public-atomic-subset-v2"
        journal.final_path.write_text("\n".join(json.dumps(record) for record in old))
        with pytest.raises(ContractError, match="incompatible search versions"):
            read_journal(journal.final_path)
    asyncio.run(run())


def test_normal_episode_owner_submits_only_approved_singleton(tmp_path, monkeypatch, definitions):
    from tests.episode_support import AuditedWorkers, PublicScript, persisted, play

    workers = AuditedWorkers(monkeypatch)
    script = PublicScript(actors=(1,))
    original = script.observation
    def observation(viewer):
        obs = original(viewer)
        if viewer == 1:
            obs["own"].update(hand=[{"instance_id": script.hand_id(1), "definition_id": "magical-stick"}], mp=10)
        return obs
    script.observation = observation
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    assert result.end_kind == "natural" and result.failure is None and result.decisions == 1
    assert script.accepted == [(1, {"type": "play_sequence", "ordered_elements": [
        {"source": "hand_artifact", "instance_id": script.hand_id(1)}], "target": 2})]
    assert len(workers.instances[0].inputs) == 1
    _, journals = persisted(tmp_path, spec)
    assert journals[1][1]["candidates"]["generator_version"] == "public-atomic-subset-v3"
