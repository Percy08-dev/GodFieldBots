"""Authorized atomic hints, independently scripted by purpose; no sequence oracle."""
import asyncio
import copy
import json
from dataclasses import asdict, replace

import httpx
import pytest

from gf_cli.config import load_config
from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import (
    POLICY_INPUT_VERSION,
    ActionConfig,
    ContractError,
    EmptyCandidates,
    PolicyDecision,
    digest,
)
from gf_common.validation import actor_action_options, decision_input, validate_worker_input
from gf_models.baselines import make_policy
from gf_simulator_client.http import AttestationError
from gf_simulator_client.workers import ParticipantWorker
from gf_training.journal import ParticipantJournal, export_parquet, read_journal
from tests.conftest import fixture
from tests.episode_support import AuditedWorkers, PublicScript, environment, persisted, play
from tests.test_http import client_for, info, response
from tests.test_journal import journal_decision, link_manifest, new_journal


def hints(value, **changes):
    options = copy.deepcopy(value.action_options)
    options["action_options"].update(changes)
    return replace(value, action_options=options)


def generate(value, **settings):
    return PublicCandidateGenerator(ActionConfig(max_sequence_length=2, **settings)).generate(value)


def test_actual_wire_snapshot_is_closed_and_keeps_perceived_catalog(definitions):
    obs = fixture("actor_options_observation_public.json")
    raw = fixture("actor_options_action_options_response_public.json")
    envelope = actor_action_options(raw, obs, raw=True)
    assert envelope == {"viewer_player_id": obs["viewer_player_id"], **raw}
    assert envelope["action_options"] == raw["action_options"]
    value = decision_input(obs, [], None, definitions, envelope)
    assert value.to_dict()["action_options"] == envelope
    assert set(value.catalog) == {c["definition_id"] for c in obs["own"]["hand"]}
    assert not any(key in value.to_dict() for key in ("game_id", "server_attestation", "game_seed"))


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(secret=True),
    lambda v: v.pop("state_revision"),
    lambda v: v.update(state_revision=True),
    lambda v: v.update(state_revision=1),
    lambda v: v["action_options"].pop("can_pray"),
    lambda v: v["action_options"].update(secret=True),
    lambda v: v["action_options"].update(can_pray=1),
    lambda v: v["action_options"].update(required_choice="defense"),
    lambda v: v["action_options"].update(usable_hand_elements=[1, 1]),
    lambda v: v["action_options"].update(usable_hand_elements=[True]),
    lambda v: v["action_options"].update(usable_hand_elements=[2**64]),
    lambda v: v["action_options"].update(usable_hand_elements=[999999]),
    lambda v: v["action_options"].update(usable_hand_elements={"id": 1}),
    lambda v: v["action_options"].update(usable_learned_miracles=["foreign-miracle"]),
    lambda v: v["action_options"].update(usable_learned_miracles=[1]),
    lambda v: v["action_options"].update(exchange_artifacts=[999999]),
    lambda v: v["action_options"].update(sell_artifacts=[999999]),
    lambda v: v["action_options"].update(buy_artifacts=[999999]),
    lambda v: v["action_options"].update(valid_targets=[0]),
    lambda v: v["action_options"].update(valid_targets=[True]),
    lambda v: v["action_options"].update(valid_targets=[1, 1]),
])
def test_wire_strict_schema_types_references_and_revision(mutation):
    raw = fixture("actor_options_action_options_response_public.json")
    mutation(raw)
    with pytest.raises(ContractError):
        actor_action_options(raw, fixture("actor_options_observation_public.json"), raw=True)


def test_envelope_actor_and_learned_reference_validation(make_input):
    value = make_input(("bronze-club",), ("meteor",))
    assert validate_worker_input(value, True) == value
    for key, invalid in (("viewer_player_id", 2), ("viewer_player_id", True), ("state_revision", 1)):
        bad = copy.deepcopy(value.action_options)
        bad[key] = invalid
        with pytest.raises(ContractError):
            validate_worker_input(replace(value, action_options=bad), True)
    for body in ({"usable_learned_miracles": ["meteor", "meteor"]}, {"usable_learned_miracles": ["bronze-club"]}):
        with pytest.raises(ContractError):
            generate(hints(value, **body))
    nonactor = copy.deepcopy(value.observation)
    nonactor.update(current_player=2, actionable=False, typed_action_schema=[])
    with pytest.raises(ContractError, match="current actionable actor"):
        actor_action_options(value.action_options, nonactor)


def test_hints_presence_matches_explicit_mode_and_config_hash(make_input, tmp_path):
    value = make_input()
    without = replace(value, action_options=None)
    with pytest.raises(ContractError, match="presence/mode"):
        generate(without)
    with pytest.raises(ContractError, match="presence/mode"):
        generate(value, use_action_options=False)
    assert generate(without, use_action_options=False).actions
    enabled = load_config("configs/experiments/smoke.toml")
    disabled = load_config("configs/experiments/smoke.toml", {"actions": {"use_action_options": False}})
    assert enabled.config.actions.use_action_options and not disabled.config.actions.use_action_options
    assert enabled.sha256 != disabled.sha256
    with pytest.raises(ContractError):
        load_config("configs/experiments/smoke.toml", {"actions": {"use_action_options": 1}})
    with pytest.raises(ContractError):
        PublicCandidateGenerator(ActionConfig(use_action_options=1))


def test_offense_and_learned_masks_preserve_perceived_ids_in_dream(make_input):
    value = make_input(("bronze-club", "god-sword"), ("meteor",))
    value.observation["own"]["status"]["dream"] = True
    masked = hints(value, usable_hand_elements=[1], usable_learned_miracles=[], can_discard=False)
    candidates = generate(masked)
    assert all(a["ordered_elements"] == [{"source": "hand_artifact", "instance_id": 1}] for a in candidates.actions)
    assert masked.catalog["bronze-club"]["attack"] == value.catalog["bronze-club"]["attack"]
    assert set(masked.catalog) == {"bronze-club", "god-sword", "meteor"}
    with pytest.raises(EmptyCandidates):
        generate(hints(masked, usable_hand_elements=[]))


def test_defense_mask_and_allow_are_independent_basics_stay_reserved(make_input):
    value = make_input(("steel-armor", "iron-shield"), choice="defense")
    candidates = generate(hints(value, usable_hand_elements=[2], can_allow=False))
    assert candidates.actions == ({"type": "respond_sequence", "ordered_defenses": [{"source": "hand_artifact", "instance_id": 2}]},)
    empty = hints(value, usable_hand_elements=[], can_allow=True)
    assert generate(empty).actions == ({"type": "respond", "ordered_defenses": []},)
    with pytest.raises(EmptyCandidates):
        generate(hints(empty, can_allow=False))
    reserved = generate(hints(value, usable_hand_elements=[2]), max_candidates=1)
    assert reserved.actions == ({"type": "respond", "ordered_defenses": []},)


def test_can_pray_overrides_visible_weapon_guess_and_does_not_fallback(make_input):
    value = hints(make_input(("magical-stick",)), usable_hand_elements=[], can_pray=True, can_discard=False)
    assert generate(value).actions == ({"type": "pray"},)
    with pytest.raises(EmptyCandidates):
        generate(hints(value, can_pray=False))
    armor = hints(make_input(("steel-armor",)), can_pray=False, can_discard=False, usable_hand_elements=[])
    with pytest.raises(EmptyCandidates):
        generate(armor)


def test_trade_carriers_sold_objects_targets_and_discard_are_separate(make_input):
    value = make_input(("exchange", "sell", "buy", "steel-armor"))
    masked = hints(value, usable_hand_elements=[], exchange_artifacts=[], sell_artifacts=[2], buy_artifacts=[3],
                   valid_targets=[2], can_pray=False, can_discard=True)
    candidates = generate(masked).actions
    assert not any(a["type"] in ("play_sequence", "exchange") for a in candidates)
    assert any(a == {"type": "sell", "artifact": 2, "sold_artifact": 4, "target": 2} for a in candidates)
    assert any(a == {"type": "buy", "artifact": 3, "target": 2} for a in candidates)
    assert all(a["target"] == 2 for a in candidates if "target" in a)
    assert {"type": "discard", "instance_ids": [4]} in candidates
    assert not any(a["type"] == "discard" for a in generate(hints(masked, can_discard=False)).actions)
    assert not any("target" in a for a in generate(hints(masked, valid_targets=[])).actions)
    exchanges = generate(hints(masked, exchange_artifacts=[1], sell_artifacts=[], buy_artifacts=[])).actions
    assert any(a["type"] == "exchange" for a in exchanges)


def test_normal_discard_protected_cards_and_sacrifice_no_invented_extension(make_input):
    value = hints(make_input(("sun-amulet", "dangerous-mortar", "steel-armor")), usable_hand_elements=[], can_pray=False)
    assert generate(value).actions == ({"type": "discard", "instance_ids": [3]},)
    value.observation["end_times"] = True
    sacrifices = generate(value).actions
    assert {"type": "sacrifice", "instance_ids": [1]} in sacrifices
    assert {"type": "sacrifice", "instance_ids": [2]} in sacrifices
    with pytest.raises(EmptyCandidates):
        generate(hints(value, can_sacrifice=False))


def test_buy_decisions_ignore_can_allow_and_support_only_prefix_is_incomplete(make_input):
    buy = hints(make_input(choice="buy_accept_reject"), can_allow=False)
    assert generate(buy).actions == ({"type": "buy_decision", "accept": False}, {"type": "buy_decision", "accept": True})
    aura = hints(make_input((), ("aura",)), can_pray=False, can_discard=False)
    with pytest.raises(EmptyCandidates):
        generate(aura)


def test_dedicated_actor_fetch_attestation_scope_and_exact_binding(make_input):
    async def run():
        value = info()
        source = make_input()
        paths = []
        raw = {k: v for k, v in source.action_options.items() if k != "viewer_player_id"}
        def handler(request):
            paths.append(request.url.path)
            return response(value if request.url.path.startswith("/debug/") else raw)
        client = client_for(handler, value)
        try:
            with pytest.raises(AttestationError):
                await client.actor_options("one-game", 1, source.observation)
            await client.attest()
            assert await client.actor_options("one-game", 1, source.observation) == source.action_options
            assert paths == ["/debug/v1/test-server/info", "/v1/games/one-game/action-options/1"]
            with pytest.raises(ContractError):
                await client.actor_options("one-game", 2, source.observation)
            with pytest.raises(ContractError):
                await client.actor_options("one-game", True, source.observation)
            assert len(paths) == 2
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["stale", "foreign", "missing", "extra", "unavailable", "timeout", "cancel"])
def test_bad_dedicated_options_stop_before_policy_and_act(tmp_path, monkeypatch, definitions, mode):
    async def run():
        workers = AuditedWorkers(monkeypatch)
        script = PublicScript()
        original = script.handle
        entered = asyncio.Event()
        async def handler(request):
            if "/action-options/" not in request.url.path:
                return await original(request)
            entered.set()
            if mode == "unavailable":
                return httpx.Response(404, json={"error": {"code": "UNAVAILABLE"}})
            if mode == "timeout":
                raise httpx.ReadTimeout("options timeout", request=request)
            if mode == "cancel":
                await asyncio.Event().wait()
            result = await original(request)
            raw = result.json()
            if mode == "stale":
                raw["state_revision"] += 1
            elif mode == "foreign":
                raw["action_options"]["usable_hand_elements"] = [script.hand_id(1)]
            elif mode == "missing":
                raw.pop("action_options")
            else:
                raw["secret_state"] = True
            return httpx.Response(200, json=raw)
        script.handle = handler
        task = asyncio.create_task(play(tmp_path, definitions, script))
        if mode == "cancel":
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
        result, _, spec = await asyncio.wait_for(task, 3)
        assert result.failure and result.decisions == 0 and not script.attempts
        assert all(not worker.inputs and worker.closed for worker in workers.instances)
        _, journals = persisted(tmp_path, spec)
        assert all([r["kind"] for r in records] == ["header", "end"] for records in journals.values())
    asyncio.run(run())


def test_explicit_opt_out_never_fetches_options_and_delivers_null(tmp_path, monkeypatch, definitions):
    async def run():
        from gf_common.domain import EpisodeGroup, EpisodeSpec
        workers = AuditedWorkers(monkeypatch)
        script = PublicScript(actors=(2,))
        env = environment(tmp_path)
        env.settings = replace(env.settings, actions=replace(env.settings.actions, use_action_options=False))
        client = script.client()
        await client.attest()
        spec = EpisodeSpec("s0-m0-r0-seat0", ("random", "random"), (17, 19), "pair")
        try:
            result = await env._episode(client, definitions, EpisodeGroup("seed", (spec,)), spec)
        finally:
            await client.close()
        assert result.end_kind == "natural"
        assert not any("action-options" in path for _, path in script.requests)
        assert workers.instances[1].inputs[0]["action_options"] is None
        _, journals = persisted(tmp_path, spec)
        assert all(not records[0]["use_action_options"] for records in journals.values())
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["missing", "stale", "foreign", "extra"])
def test_bootstrap_revalidates_before_policy_select(make_input, mutation):
    async def run():
        value = make_input()
        if mutation == "missing":
            value = replace(value, action_options=None)
        elif mutation == "stale":
            value.action_options["state_revision"] += 1
        elif mutation == "foreign":
            value.action_options["action_options"]["usable_hand_elements"] = [999]
        else:
            value.action_options["secret"] = True
        worker = ParticipantWorker(make_policy, "heuristic", 0, ActionConfig(), 2)
        try:
            with pytest.raises(RuntimeError, match="ContractError"):
                await worker.select(value)
            assert worker.closed
        finally:
            await worker.close()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["missing", "stale", "foreign", "mode", "version"])
def test_journal_reader_revalidates_snapshot_mode_and_input_version(tmp_path, make_input, mutation):
    journal = new_journal(tmp_path)
    value = make_input()
    journal_decision(journal, value)
    journal.finish("incomplete", None)
    records = read_journal(journal.final_path)
    if mutation == "missing":
        records[1]["input"]["action_options"] = None
    elif mutation == "stale":
        records[1]["input"]["action_options"]["state_revision"] += 1
    elif mutation == "foreign":
        records[1]["input"]["action_options"]["action_options"]["usable_hand_elements"] = [999]
    elif mutation == "mode":
        records[0]["use_action_options"] = False
    else:
        records[0]["policy_input_version"] = "old"
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError):
        read_journal(journal.final_path)


def test_exact_snapshot_roundtrip_export_and_manifest_versions(tmp_path, make_input):
    pytest.importorskip("pyarrow")
    directory = tmp_path / "episode"
    directory.mkdir()
    journal = new_journal(directory)
    value = hints(make_input(("bronze-club", "god-sword")), usable_hand_elements=[2, 1], valid_targets=[2, 1])
    journal_decision(journal, value)
    journal.finish("incomplete", None)
    link_manifest(tmp_path, journal)
    metadata = export_parquet([journal.final_path], tmp_path / "data.parquet")
    import pyarrow.parquet as pq
    row = pq.read_table(tmp_path / "data.parquet").to_pylist()[0]
    assert json.loads(row["input_json"])["action_options"] == value.action_options
    assert metadata["derived_schema_version"] == metadata["journal_schema_version"] == 4
    assert metadata["policy_input_version"] == POLICY_INPUT_VERSION
    for mutation in ("mode", "version", "spec"):
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        if mutation == "mode":
            manifest["effective_config"]["actions"]["use_action_options"] = False
            manifest["config_sha256"] = digest(manifest["effective_config"])
        elif mutation == "version":
            manifest["schema_version"] = 2
        else:
            manifest["reviewed_simulator_spec_sha256"] = {}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ContractError):
            export_parquet([journal.final_path], tmp_path / "bad.parquet")
        link_manifest(tmp_path, journal)


def test_journal_writer_requires_declared_mode(tmp_path, make_input):
    value = make_input()
    config = ActionConfig(use_action_options=False)
    journal = ParticipantJournal(tmp_path, "episode", 1, "random", run_id="run", game_id="game",
                                 action_config_hash=digest(asdict(config)), use_action_options=False)
    candidates = generate(replace(value, action_options=None), use_action_options=False)
    try:
        with pytest.raises(ContractError, match="presence/mode"):
            journal.begin(value, candidates, PolicyDecision(0, "random-v1"), {}, 0)
    finally:
        journal.close_incomplete()


def test_observation_only_journal_and_export_preserve_null_mode(tmp_path, make_input):
    pytest.importorskip("pyarrow")
    directory = tmp_path / "episode"
    directory.mkdir()
    config = ActionConfig(use_action_options=False)
    journal = ParticipantJournal(directory, "episode", 1, "random", run_id="run", game_id="game",
                                 action_config_hash=digest(asdict(config)), use_action_options=False)
    value = replace(make_input(), action_options=None)
    candidates = PublicCandidateGenerator(config).generate(value)
    journal.begin(value, candidates, PolicyDecision(0, "random-v1"),
                  {"command_id": "00000000-0000-0000-0000-000000000001", "expected_revision": 0,
                   "action": candidates.actions[0]}, 0)
    journal.finish("incomplete", None)
    records = read_journal(journal.final_path)
    assert records[0]["use_action_options"] is False
    assert records[1]["input"]["action_options"] is None
    link_manifest(tmp_path, journal)
    export_parquet([journal.final_path], tmp_path / "data.parquet")


def test_mask_preselection_does_not_fabricate_attempts_and_wide_beam_agrees(make_input):
    value = hints(make_input(("bronze-club", "god-sword", "warrior-s-bow")),
                  usable_hand_elements=[1], usable_learned_miracles=[], can_discard=False,
                  valid_targets=[2])
    outputs = [PublicCandidateGenerator(ActionConfig(strategy=strategy, beam_width=1000,
               max_sequence_length=1)).generate(value) for strategy in ("enumerate", "beam")]
    assert outputs[0].actions == outputs[1].actions == ({"type": "play_sequence", "ordered_elements": [
        {"source": "hand_artifact", "instance_id": 1}], "target": 2},)
    assert outputs[0].attempts == outputs[1].attempts == 1
    assert outputs[0].public_prunes == outputs[1].public_prunes == 0
