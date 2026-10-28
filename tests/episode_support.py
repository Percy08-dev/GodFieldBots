"""Controlled public HTTP/worker boundaries for exercising the real episode owner.

No engine logic is reproduced: the script supplies predetermined public decisions,
acknowledgements and diagnostic replays, and checks actual transport identities.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import httpx

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, EpisodeGroup, EpisodeSpec, PolicyDecision
from gf_common.validation import action_descriptors, catalog
from gf_models.baselines import make_policy
from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment
from gf_simulator_client.http import EFFECTIVE_KEYS, AttestedClient
from gf_training.journal import ParticipantJournal, read_journal
from tests.conftest import fixture, public_options
from tests.test_http import info

ROOT = Path(__file__).parents[1]
_SCRIPT_IDS = itertools.count(1)
SEED = "00000000-0000-0000-0000-000000000001"


class PublicScript:
    def __init__(self, mode="natural", outcome="draw", actors=(2, 1, 2, 1)):
        self.instance = next(_SCRIPT_IDS)
        self.created_games = []
        self.hand_ids_by_game = {}
        self.game_id = None
        self.mode = mode
        self.outcome = outcome
        self.actors = actors
        self.revision = 0
        self.joined = 0
        self.accepted = []
        self.attempts = []
        self.requests = []
        self.event_calls = []
        self.cursor_counts = {1: 0, 2: 0}
        self.observations = []
        self.info = info()
        self.deleted = False
        self.closed = False
        self.replay_entered = asyncio.Event()
        self.replay_release = None
        self.act_entered = asyncio.Event()
        self.hold_act = asyncio.Event()
        self.terminal = False

    def hand_id(self, viewer):
        return self.hand_ids_by_game[self.game_id][viewer]

    def cursor(self, viewer, count):
        return f"{self.game_id}-viewer-{viewer}-cursor-{count}"

    def observation(self, viewer):
        obs = fixture("observation_p1_public.json")
        actor = self.actors[min(self.revision, len(self.actors) - 1)]
        own_definition = "bronze-club" if viewer == 1 else "god-sword"
        obs.update(viewer_player_id=viewer, current_player=actor, active_player=actor,
                   first_player=2, state_revision=self.revision, turn=self.revision + 1)
        obs["own"].update(player_id=viewer, hand=[{"instance_id": self.hand_id(viewer), "definition_id": own_definition}])
        obs["opponent"]["player_id"] = 3 - viewer
        if self.terminal:
            obs.update(result=self.outcome, phase="terminal", current_player=None, required_choice=None)
        elif self.mode == "unsupported":
            obs["required_choice"] = "additional_choice"
        actionable = not self.terminal and viewer == actor and self.mode != "unsupported"
        obs["actionable"] = actionable
        obs["typed_action_schema"] = action_descriptors(obs["required_choice"]) if actionable else []
        return obs

    def replay(self, viewer):
        decisions = [{"actor": actor, "visible_action": {"type": "action", "action": action}}
                     for actor, action in self.accepted if actor == viewer]
        end = {"type": "terminal", "result": self.outcome} if self.terminal else None
        if self.mode in ("turn_limit", "server_abort") and self.accepted:
            end = {"type": "aborted", "reason": "turn_limit_exceeded" if self.mode == "turn_limit" else "server_shutdown",
                   "player": None, "decision": {"turn": 1, "decision_index": None, "invalid_attempt_count": 0}}
        return {"game_id": self.game_id, "viewer_player_id": viewer, "observation_schema_version": "1",
                "episode_end": end,
                "turns": [{"turn": 1, "active_player": 2, "decisions": decisions}]}

    async def handle(self, request):
        self.requests.append((request.method, request.url.path))
        path = request.url.path
        if path == "/debug/v1/test-server/info":
            return httpx.Response(200, json=self.info, headers={"cache-control": "no-store"})
        if path == "/v1/info":
            return httpx.Response(200, json=fixture("info_public_v2.json"))
        if path == "/v1/card-catalogs/public-ja-v2":
            return httpx.Response(200, json=fixture("catalog_public_v2.json"))
        if path == "/v1/games" and request.method == "POST":
            self.game_id = f"{self.instance * 1000 + len(self.created_games) + 1:026d}"
            self.created_games.append(self.game_id)
            self.hand_ids_by_game[self.game_id] = {viewer: self.instance * 1000000 + len(self.created_games) * 1000 + viewer * 100 + 1 for viewer in (1, 2)}
            self.revision = 0
            self.joined = 0
            self.accepted = []
            self.cursor_counts = {1: 0, 2: 0}
            self.terminal = False
            created = fixture("create_public.json")
            created["game_id"] = self.game_id
            return httpx.Response(201, json=created)
        if path.startswith("/v1/games/"):
            assert path.split("/")[3] == self.game_id, "stale or cross-game request"
        if path.endswith("/players"):
            self.joined += 1
            value = fixture("join_p1_public.json")
            value.update(game_id=self.game_id, player_id=self.joined, joined_players=self.joined,
                         status="active" if self.joined == 2 else "waiting_for_players",
                         state_revision=0 if self.joined == 2 else None)
            return httpx.Response(200, json=value)
        if "/observations/" in path:
            viewer = int(path.rsplit("/", 1)[1])
            self.observations.append((viewer, self.revision))
            if self.mode == "api_aborted":
                return httpx.Response(409, json={"error": {"code": "GAME_ABORTED"}})
            return httpx.Response(200, json=self.observation(viewer))
        if "/action-options/" in path:
            actor = int(path.rsplit("/", 1)[1])
            assert actor == self.actors[min(self.revision, len(self.actors) - 1)]
            options = public_options(self.observation(actor), catalog(fixture("catalog_public_v2.json")))
            options.pop("viewer_player_id")
            return httpx.Response(200, json=options)
        if "/events/" in path:
            viewer = int(path.rsplit("/", 1)[1])
            after = request.url.params.get("after")
            count = self.cursor_counts[viewer]
            assert after == (self.cursor(viewer, count) if count else None)
            self.event_calls.append((viewer, after))
            self.cursor_counts[viewer] += 1
            event = {"type": "artifact_drawn", "player": viewer, "definition_id": "bronze-club" if viewer == 1 else "god-sword"}
            # Different public turns make repeated pages detectable without IDs.
            event_page = [{"type": "turn_started", "turn": self.revision + 1, "active_player": viewer}, event,
                          {"type": "artifact_used", "player": 3 - viewer, "definition_id": "sun-amulet"}]
            return httpx.Response(200, json={"events": event_page, "next_cursor": self.cursor(viewer, count + 1), "has_more": False})
        if path.endswith("/actions"):
            actor = int(path.split("/")[-2])
            payload = json.loads(request.content)
            self.attempts.append((actor, payload))
            assert payload["expected_revision"] == self.revision
            self.act_entered.set()
            if self.mode == "cancel_act":
                await self.hold_act.wait()
            if self.mode == "rejection":
                return httpx.Response(422, json={"error": {"code": "INVALID_ACTION", "details": {"secret_rejection_probe": "never policy input"}}})
            if self.mode == "unknown":
                # Server may have committed; there is deliberately no positive
                # acknowledgement. Replay must not become a training label.
                self.accepted.append((actor, payload["action"]))
                self.revision += 1
                self.terminal = True
                raise httpx.ReadTimeout("acknowledgement lost", request=request)
            self.accepted.append((actor, payload["action"]))
            previous = self.revision
            self.revision += 1
            self.terminal = self.mode not in ("turn_limit", "server_abort") and self.revision == len(self.actors)
            status = "aborted" if self.mode in ("turn_limit", "server_abort") else "terminal" if self.terminal else "active"
            return httpx.Response(200, json={"command_id": payload["command_id"], "previous_revision": previous,
                "state_revision": self.revision, "status": status, "result": self.outcome if self.terminal else None,
                "events": [{"forbidden_ack_events": True}], "next_observation": {"forbidden_ack_view": True},
                "action_options": {"forbidden_options": True}})
        if "/replay/" in path:
            self.replay_entered.set()
            if self.replay_release is not None:
                await self.replay_release.wait()
            return httpx.Response(200, json=self.replay(int(path.rsplit("/", 1)[1])))
        if request.method == "DELETE":
            self.deleted = True
            return httpx.Response(204)
        raise AssertionError((request.method, str(request.url)))

    def client(self):
        return AttestedClient("http://test", expected_fingerprint=self.info["seed_fingerprint"], fixed_seed_experiment=True,
                              expected_effective={key: self.info[key] for key in EFFECTIVE_KEYS}, retries=0,
                              transport=httpx.MockTransport(self.handle))


class AuditedWorkers:
    def __init__(self, monkeypatch, pause=False):
        import gf_simulator_client.environment as module
        self.instances = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        owner = self
        class Worker:
            def __init__(self, factory, model_id, sampling_seed, actions, timeout):
                self.generator = PublicCandidateGenerator(actions)
                self.inputs = []
                self.closed = False
                self.health = {}
                self.model_id = model_id
                owner.instances.append(self)
            async def select(self, value):
                self.inputs.append(value.to_dict())
                owner.entered.set()
                if pause:
                    await owner.release.wait()
                candidates = self.generator.generate(value)
                return candidates, PolicyDecision(0, self.model_id + "-v1")
            async def close(self, timeout=2):
                self.closed = True
        monkeypatch.setattr(module, "ParticipantWorker", Worker)


def environment(tmp_path, **settings):
    config = EnvironmentSettings("unused", str(ROOT / "configs/test-server.toml"), tmp_path, "test-run",
                                 ActionConfig(max_sequence_length=1, targets=("opponent",)), **settings)
    return SimulatorEnvironment(config, make_policy, ParticipantJournal)


async def play(tmp_path, definitions, script, **settings):
    client = script.client()
    await client.attest()
    spec = EpisodeSpec("s0-m0-r0-seat0", ("random", "random"), (17, 19), "pair")
    group = EpisodeGroup(SEED, (spec,))
    env = environment(tmp_path, **settings)
    try:
        result = await env._episode(client, definitions, group, spec)
    finally:
        await client.close()
    return result, env, spec


def persisted(tmp_path, spec):
    directory = tmp_path / spec.episode_id
    result = json.loads((directory / "result.json").read_text())
    journals = {}
    for viewer in (1, 2):
        final = directory / f"player-{viewer}.jsonl"
        path = final if final.exists() else directory / f"player-{viewer}.partial.jsonl"
        journals[viewer] = read_journal(path)
    return result, journals


def check_persisted_result(result, saved):
    assert saved == json.loads(json.dumps(asdict(result)))
