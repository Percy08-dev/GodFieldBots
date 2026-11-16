"""Live environment lifecycle hooks, including ACK failure and zero-Act games."""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import EpisodeGroup, EpisodeSpec
from gf_simulator_client.environment import SimulatorEnvironment
from tests.episode_support import SEED, AuditedWorkers, PublicScript, environment


@pytest.mark.parametrize("mode,count", [("natural", 4), ("rejection", 0), ("unknown", 0)])
def test_actual_act_ack_hook_never_promotes_rejection_or_unknown(tmp_path, monkeypatch, definitions, mode, count):
    AuditedWorkers(monkeypatch)
    script = PublicScript(mode=mode)
    env = environment(tmp_path)
    seen = []
    monkeypatch.setattr(SimulatorEnvironment, "_episode_opened", lambda self, result: seen.append(("open", result.game_id)))
    def ack(self, result, viewer, command, acknowledgement):
        assert command["command_id"] == acknowledgement["command_id"]
        assert command["expected_revision"] == acknowledgement["previous_revision"]
        seen.append(("ack", viewer))
    monkeypatch.setattr(SimulatorEnvironment, "_act_acknowledged", ack)
    monkeypatch.setattr(SimulatorEnvironment, "_terminal_observed", lambda self, result, viewer, observation: seen.append(("terminal", viewer)))
    monkeypatch.setattr(SimulatorEnvironment, "_episode_closed", lambda self, result: seen.append(("close", result.end_kind)))
    async def run():
        client = script.client()
        await client.attest()
        spec = EpisodeSpec("episode", ("random", "random"), (1, 2), "pair")
        try:
            return await env._episode(client, definitions, EpisodeGroup(SEED, (spec,)), spec)
        finally:
            await client.close()
    result = asyncio.run(run())
    assert len([item for item in seen if item[0] == "ack"]) == count == result.decisions
    assert seen[0][0] == "open" and seen[-1][0] == "close"
    assert [item[1] for item in seen if item[0] == "terminal"] == ([1, 2] if mode == "natural" else [])


def test_no_act_terminal_pair_is_observed_before_journal_finish(tmp_path, monkeypatch, definitions):
    AuditedWorkers(monkeypatch)
    script = PublicScript()
    actual = script.observation
    def observation(viewer):
        script.terminal = True
        return actual(viewer)
    script.observation = observation
    seen = []
    monkeypatch.setattr(SimulatorEnvironment, "_terminal_observed", lambda self, result, viewer, observation: seen.append(viewer))
    finish = SimulatorEnvironment._finish_journal
    def finish_after_pair(self, *args):
        assert seen == [1, 2]
        return finish(self, *args)
    monkeypatch.setattr(SimulatorEnvironment, "_finish_journal", finish_after_pair)
    async def run():
        client = script.client()
        await client.attest()
        spec = EpisodeSpec("episode", ("random", "random"), (1, 2), "pair")
        try:
            return await environment(tmp_path)._episode(client, definitions, EpisodeGroup(SEED, (spec,)), spec)
        finally:
            await client.close()
    result = asyncio.run(run())
    assert result.end_kind == "natural" and result.decisions == 0 and not result.health


def test_corrected_group_clone_retains_injected_capture_and_closes_after_actual_body(tmp_path, monkeypatch):
    import gf_simulator_client.card_instance_environment as module
    from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
    from gf_common.domain import GroupResult, RuntimePolicyBinding
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.test_group_pack_runtime_environment import Evidence

    base = environment(tmp_path).settings
    settings = replace(base, actions=ActionConfigV3(action_contract=corrected_card_instance_contract()),
        policy_bindings={"random-card-instance-v2": RuntimePolicyBinding("random")})
    events = []
    class Capture:
        def close_group(self, group, result):
            events.append(("closed", group, result))
    capture = Capture()
    monkeypatch.setattr(module, "CardInstanceRuntimeEvidence", Evidence)
    env = module.CardInstanceSimulatorEnvironment(settings, lambda *a, **kw: None,
        ledger=ReservationLedger(), runtime_evidence=Evidence("unused", tmp_path), capture=capture)
    async def body(self, group):
        assert self is not env and self.capture is capture
        events.append(("body", group))
        return GroupResult([])
    monkeypatch.setattr(module.CardInstanceSimulatorEnvironment, "_run_group_body", body)
    spec = EpisodeSpec("episode", ("random-card-instance-v2",) * 2, (1, 2), "pair")
    group = EpisodeGroup(SEED, (spec,))
    result = asyncio.run(env.run_group(group))
    assert events == [("body", group), ("closed", group, result)]


@pytest.mark.parametrize("mode", ["natural", "zero_act", "rejection", "aborted", "late_save_failure",
    "creation_failure", "header_failure"])
def test_actual_corrected_worker_selection_journal_ack_and_group_feed_same_capture(tmp_path, monkeypatch, mode):
    import gf_simulator_client.card_instance_environment as module
    from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
    from gf_common.domain import RuntimePolicyBinding
    from gf_common.group_pack import StorageSettings
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.card_instance_journal import CardInstanceJournal
    from gf_training.collection import CollectionPlan
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_capture import _new_capture
    from tests.test_group_pack_runtime_environment import Evidence, PublicClient

    label = "random-card-instance-v2"
    spec = EpisodeSpec("episode", (label, label), (1, 2), "pair", 2)
    group = EpisodeGroup(SEED, (spec,))
    plan = CollectionPlan((group,), {label: RuntimePolicyBinding("random")}, tmp_path,
        {}, float("inf"), float("inf"))
    lock, snapshot = object(), object()
    capture = _new_capture(lock, run_id="run", plan=plan, generation=0,
        snapshot_owner=snapshot, learner_seats={spec.episode_id: 2},
        learner_policy_version=label, budget=StorageBudget(StorageSettings(), time.monotonic() + 20))
    created=[]
    participant_created=capture.participant_created
    def observed_created(episode_id,game_id,viewer,bank):
        import json
        header=json.loads((tmp_path/episode_id/f'player-{viewer}.partial.jsonl').read_text().splitlines()[0])
        assert header['bank_episode']==bank.episode and not bank._events
        created.append(viewer)
        participant_created(episode_id,'foreign-game' if mode=='creation_failure' and viewer==2 else game_id,viewer,bank)
    monkeypatch.setattr(capture,'participant_created',observed_created)
    class Client(PublicClient):
        async def observe(self,game,viewer):
            if mode=='zero_act': self.terminal=True
            return await super().observe(game,viewer)
        async def act(self, game, actor, payload):
            if mode == "rejection":
                from gf_simulator_client.http import ApiError
                raise ApiError(422, {"error": {"code": "INVALID_ACTION"}})
            value = await super().act(game, actor, payload)
            if mode == "aborted":
                self.terminal = False
                value.update(status="aborted", result=None)
            return value
        async def replay(self, game, viewer):
            value = await super().replay(game, viewer)
            if mode == "aborted":
                value["episode_end"] = {"type": "aborted", "reason": "server_shutdown", "player": None,
                    "decision": {"turn": 1, "decision_index": 0, "invalid_attempt_count": 0}}
            return value
    class Server:
        def __init__(self, *args, **kwargs):
            self.process = SimpleNamespace(returncode=None)
            self.expected, self.drainers = {}, []
        async def start(self):
            return {"base_url": "http://private-script", "test_server_info": {}}
        async def stop(self, *args):
            self.process.returncode = 0
            return {"exit_code": 0, "shutdown": "stdin_eof", "log_error": None}
    monkeypatch.setattr(module, "CardInstanceRuntimeEvidence", Evidence)
    monkeypatch.setattr(module, "AttestedClient", lambda *a, **kw: Client("natural"))
    monkeypatch.setattr(module.CardInstanceSimulatorEnvironment, "_new_server", lambda *a, **kw: Server())
    settings = replace(environment(tmp_path).settings, run_id="run",
        binary=str(Path("unused").absolute()),
        actions=ActionConfigV3(action_contract=corrected_card_instance_contract(),
            max_candidates=30, max_sequence_length=1), policy_bindings=plan.bindings)
    def journal_factory(*args,**kwargs):
        if mode=='header_failure' and args[2]==2: raise OSError('actual second header failed')
        return CardInstanceJournal(*args,**kwargs)
    env = module.CardInstanceSimulatorEnvironment(settings, journal_factory,
        ledger=ReservationLedger(), runtime_evidence=Evidence("unused", tmp_path), capture=capture)
    if mode == "late_save_failure":
        original_save = module.CardInstanceSimulatorEnvironment._save_artifact
        def save(self, path, value):
            if path.name == "result.json":
                raise OSError("late result persistence failed")
            return original_save(self, path, value)
        monkeypatch.setattr(module.CardInstanceSimulatorEnvironment, "_save_artifact", save)
    result = asyncio.run(env.run_group(group))
    assert not env.ledger.workers and env.ledger.learner is None
    assert created==([1] if mode=='header_failure' else [1,2])
    if mode not in ("natural","zero_act"):
        from gf_common.domain import ContractError
        assert result.episodes[0].end_kind == {"rejection": "incomplete", "aborted": "aborted", "late_save_failure": "natural",
            "creation_failure":"incomplete","header_failure":"incomplete"}[mode]
        if mode == "rejection":
            assert result.episodes[0].failure == "rejection" and result.episodes[0].decisions == 0
        if mode == "aborted":
            assert result.episodes[0].simulator_abort_reason == "server_shutdown"
        if mode == "late_save_failure":
            assert result.episodes[0].health["save_failure"]
        with pytest.raises(ContractError):
            capture.seal(lock, plan=plan, snapshot_owner=snapshot)
        capture.close()
        return
    assert result.episodes[0].end_kind == "natural" and not result.episodes[0].health
    batch = capture.seal(lock, plan=plan, snapshot_owner=snapshot)
    assert len(batch.episodes[0].participants[0].decisions) == (0 if mode=='zero_act' else 1)
    assert batch.episodes[0].learner.viewer == 2 and not batch.episodes[0].learner.decisions
    if mode=='natural':
        recorded = batch.episodes[0].participants[0].decisions[0]
        assert recorded.command["command_id"] == recorded.acknowledgement["command_id"]
        assert recorded.command["action"] == recorded.selection.validate(recorded.table, recorded.frame)
    assert batch.episodes[0].replay_reconciled is False
    # The zero-Act learner still owns the SAME bank originally journaled when
    # its actual participant was created; raw identity is not fabricated later.
    import json

    from gf_common.structured import EventBank
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    for participant in batch.episodes[0].participants:
        header=json.loads((plan.output/spec.episode_id/f'player-{participant.viewer}.jsonl').read_text().splitlines()[0])
        initial=EventBank(participant.viewer,header['bank_episode'],layout=CORRECTED_CARD_INSTANCE_LAYOUT).view()
        assert participant.history.episode==header['bank_episode']
        prefix=participant.history.prefix(initial.boundary)
        assert prefix.boundary==initial.boundary and prefix.events()==initial.events()
    batch.close()
