"""ADR0007: one narrowly actionable resolving state, shared across boundaries."""
import asyncio
import copy
import json
import uuid
from dataclasses import asdict

import pytest

from gf_common.domain import (
    SIMULATOR_REVISION,
    ActionConfig,
    ContractError,
    UnsupportedContract,
    digest,
)
from gf_common.validation import (
    action_descriptors,
    actor_action_options,
    decision_input,
    observation,
)
from gf_models.baselines import make_policy
from gf_simulator_client.workers import ParticipantWorker
from gf_training.journal import ParticipantJournal, export_parquet, read_journal
from tests.conftest import fixture
from tests.test_http import client_for, info, response
from tests.test_journal import journal_decision, link_manifest, new_journal


@pytest.mark.parametrize("phase", ["awaiting_action", "awaiting_response", "resolving", "terminal"])
@pytest.mark.parametrize("choice", [None, "turn_action", "defense", "buy_accept_reject"])
@pytest.mark.parametrize("actor", [True, False])
@pytest.mark.parametrize("ended", [True, False])
def test_phase_actor_choice_result_exact_metadata_matrix(phase, choice, actor, ended):
    obs = fixture("staged_buy_public.json")["observation"]
    obs.update(phase=phase, required_choice=choice, current_player=2 if actor else 1,
               result="draw" if ended else None)
    allowed_choices = {("awaiting_action", "turn_action"), ("awaiting_action", "defense"),
                       ("awaiting_action", "buy_accept_reject"), ("awaiting_response", "turn_action"),
                       ("awaiting_response", "defense"), ("awaiting_response", "buy_accept_reject"),
                       ("resolving", "buy_accept_reject")}
    expected = actor and not ended and (phase, choice) in allowed_choices
    obs.update(actionable=expected, typed_action_schema=action_descriptors(choice) if expected else [])
    assert observation(obs) == obs
    bad = copy.deepcopy(obs)
    bad["actionable"] = not expected
    with pytest.raises(UnsupportedContract):
        observation(bad)
    if choice is not None:
        bad = copy.deepcopy(obs)
        bad["typed_action_schema"] = [] if expected else action_descriptors(choice)
        with pytest.raises(UnsupportedContract):
            observation(bad)


def test_old_false_buyer_or_changed_descriptor_is_never_repaired():
    obs = fixture("staged_buy_public.json")["observation"]
    old = copy.deepcopy(obs)
    old.update(actionable=False, typed_action_schema=[])
    with pytest.raises(UnsupportedContract, match="typed metadata"):
        observation(old)
    bad = copy.deepcopy(obs)
    bad["typed_action_schema"][0]["required_fields"]["accept"]["type"] = "integer"
    with pytest.raises(UnsupportedContract):
        observation(bad)
    obs.update(required_choice="additional_choice", actionable=False, typed_action_schema=[])
    assert observation(obs) == obs  # unsupported automatic resolving state stays closed
    obs["phase"] = "awaiting_action"
    with pytest.raises(UnsupportedContract, match="additional_choice"):
        observation(obs)


@pytest.mark.parametrize("damage", ["stale", "missing", "foreign", "extra"])
def test_new_buyer_dedicated_hints_are_still_strict(definitions, damage):
    raw = fixture("staged_buy_public.json")
    options = raw["action_options_response"]
    if damage == "stale":
        options["state_revision"] -= 1
    elif damage == "missing":
        options["action_options"].pop("can_allow")
    elif damage == "foreign":
        options["action_options"]["usable_hand_elements"] = [99999]
    else:
        options["private_pending_buy"] = {}
    with pytest.raises(ContractError):
        actor_action_options(options, raw["observation"], raw=True)


def test_captured_buyer_crosses_strict_http_worker_journal_and_export(tmp_path, definitions):
    async def run():
        raw = fixture("staged_buy_public.json")
        server_info = info()
        paths = []
        def handler(request):
            paths.append(request.url.path)
            if request.url.path == "/debug/v1/test-server/info":
                return response(server_info)
            if request.url.path.endswith("/observations/2"):
                return response(raw["observation"])
            if request.url.path.endswith("/action-options/2"):
                return response(raw["action_options_response"])
            raise AssertionError(request.url.path)
        client = client_for(handler, server_info)
        worker = ParticipantWorker(make_policy, "random", 17, ActionConfig(), 3)
        try:
            await client.attest()
            obs = await client.observe("only-owned-game", 2)
            options = await client.actor_options("only-owned-game", 2, obs)
            value = decision_input(obs, [], None, definitions, options)
            candidates, selected = await worker.select(value)
            assert candidates.actions == ({"type": "buy_decision", "accept": False}, {"type": "buy_decision", "accept": True})
            assert candidates.actions[selected.index] == {"type": "buy_decision", "accept": True}
            assert options["action_options"]["can_allow"] is False
            directory = tmp_path / "episode"
            directory.mkdir()
            journal = ParticipantJournal(directory, "episode", 2, "random", run_id="run", game_id="game",
                                         action_config_hash=digest(asdict(ActionConfig())))
            command = {"command_id": str(uuid.uuid4()), "expected_revision": 7, "action": candidates.actions[selected.index]}
            index = journal.begin(value, candidates, selected, command, 1)
            journal.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 7,
                                        "state_revision": 8, "status": "active", "result": None})
            journal.finish("truncated", None)
            records = read_journal(journal.final_path)
            assert records[1]["input"] == value.to_dict()
            assert records[-1]["terminal_reward"] is None
            link_manifest(tmp_path, journal)
            if __import__("importlib").util.find_spec("pyarrow") is not None:
                metadata = export_parquet([journal.final_path], tmp_path / "buyer.parquet")
                assert metadata["rows"] == 1 and metadata["eligible_rows"] == 0
            assert paths == ["/debug/v1/test-server/info", "/v1/games/only-owned-game/observations/2", "/v1/games/only-owned-game/action-options/2"]
        finally:
            await worker.close()
            await client.close()
    asyncio.run(run())


def test_older_revision_linked_source_is_rejected_without_relabeling(tmp_path, make_input):
    directory = tmp_path / "episode"
    directory.mkdir()
    journal = new_journal(directory)
    journal_decision(journal, make_input())
    journal.finish("incomplete", None)
    manifest = link_manifest(tmp_path, journal)
    manifest["simulator_contract_revision"] = "44a26e3a16b808e826deb332fa72acfc5037d3ef"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert SIMULATOR_REVISION != manifest["simulator_contract_revision"]
    assert read_journal(journal.final_path)  # structural schema remains unchanged
    pytest.importorskip("pyarrow")
    with pytest.raises(ContractError, match="incompatible source simulator semantics"):
        export_parquet([journal.final_path], tmp_path / "unsupported.parquet")


def test_resolving_buyer_uses_current_participant_even_when_active_differs():
    obs = fixture("staged_buy_public.json")["observation"]
    obs["active_player"] = 1
    assert obs["current_player"] == obs["viewer_player_id"] == 2
    assert observation(obs)["actionable"] is True
    obs.update(current_player=None, actionable=False, typed_action_schema=[])
    assert observation(obs)["actionable"] is False
    with pytest.raises(ContractError):
        actor_action_options(fixture("staged_buy_public.json")["action_options_response"], obs, raw=True)
