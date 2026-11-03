"""Shared attestation owner with explicitly selected original v2 transport."""

import asyncio
import json
import uuid

import httpx
import pytest

from gf_common.card_instance import card_instance_contract
from gf_common.domain import ContractError
from gf_simulator_client.http import AttestationError, seed_fingerprint
from tests.test_card_instance_contract import fixture, observation, options, replay
from tests.test_http import client_for, info, response


def created():
    return {
        "game_id": "01J00000000000000000000000",
        "status": "waiting_for_players",
        "required_players": 2,
        "joined_players": 0,
        "state_revision": None,
        "rule_set_version": "true-match-1v1-v2",
        "random_algorithm": "xoshiro256pp",
        "random_algorithm_version": "1",
    }


def acknowledgement(payload, **changes):
    return {
        "public_event_schema_version": "2",
        "rule_set_version": "true-match-1v1-v2",
        "card_ref_scheme": "aes256-card-ref-v1",
        "command_id": payload["command_id"],
        "previous_revision": payload["expected_revision"],
        "state_revision": payload["expected_revision"] + 1,
        "status": "active",
        "events": [{"type": "safe", "player": 1}],
        "result": None,
        "next_observation": None,
        "action_options": None,
        **changes,
    }


def test_v2_direct_routes_and_create_full_public_preconditions():
    async def run():
        management = info()
        paths = []

        def handler(request):
            paths.append(request.url.path)
            path = request.url.path
            if path == "/debug/v1/test-server/info":
                return response(management)
            if path == "/v2/info":
                return response(fixture("info"))
            if path.startswith("/v2/card-catalogs/"):
                return response(fixture("catalog"))
            if path == "/v2/games":
                assert json.loads(request.content)["rule_set_version"] == "true-match-1v1-v2"
                return response(created(), 201)
            if path.endswith("/players"):
                return response(
                    {
                        "game_id": "game",
                        "player_id": 1,
                        "status": "active",
                        "joined_players": 2,
                        "required_players": 2,
                        "state_revision": 1,
                    }
                )
            if "/observations/" in path:
                return response(observation())
            if "/action-options/" in path:
                return response({"state_revision": 7, "action_options": options()})
            if "/events/" in path:
                return response(
                    {
                        "public_event_schema_version": "2",
                        "rule_set_version": "true-match-1v1-v2",
                        "events": [{"type": "safe", "player": 1}],
                        "next_cursor": "cursor",
                        "has_more": False,
                    }
                )
            if path.endswith("/actions"):
                return response(acknowledgement(json.loads(request.content)))
            if "/replay/" in path:
                return response(replay(aborted=True))
            if request.method == "DELETE":
                return httpx.Response(204)
            raise AssertionError(path)

        client = client_for(handler, management, contract=card_instance_contract())
        try:
            assert await client.create() == created()
            assert paths == [
                "/debug/v1/test-server/info",
                "/v2/info",
                "/v2/card-catalogs/public-ja-v3",
                "/v2/games",
                "/debug/v1/test-server/info",
            ]
            await client.join("game", str(uuid.uuid4()))
            assert await client.observe("game", 1) == observation()
            assert (await client.actor_options("game", 1, observation()))[
                "action_options"
            ] == options()
            assert await client.events("game", 1, None) == (
                [{"type": "safe", "player": 1}],
                "cursor",
            )
            payload = {
                "command_id": str(uuid.uuid4()),
                "expected_revision": 7,
                "action": {"type": "pray"},
            }
            assert (await client.act("game", 1, payload))["state_revision"] == 8
            assert await client.replay("game", 1) == replay(aborted=True)
            await client.delete("game")
            assert all(path.startswith(("/v2/", "/debug/v1/")) for path in paths)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("bad_boundary", ["info", "catalog"])
def test_create_rejects_public_contract_before_game_allocation(bad_boundary):
    async def run():
        management = info()
        paths = []

        def handler(request):
            paths.append(request.url.path)
            if request.url.path.startswith("/debug/"):
                return response(management)
            if request.url.path == "/v2/info":
                value = fixture("info")
                if bad_boundary == "info":
                    value["action_schema_version"] = "2"
                return response(value)
            value = fixture("catalog")
            value["definitions"][0]["attack"] = 99
            return response(value)

        client = client_for(handler, management, contract=card_instance_contract())
        try:
            with pytest.raises(ContractError):
                await client.create()
            assert "/v2/games" not in paths
        finally:
            await client.close()

    asyncio.run(run())


def test_retry_keeps_exact_original_payload_bytes_and_order():
    async def run():
        management = info()
        bodies = []

        def handler(request):
            if request.url.path.startswith("/debug/"):
                return response(management)
            bodies.append(request.content)
            if len(bodies) == 1:
                raise httpx.ReadTimeout("uncertain", request=request)
            return response(acknowledgement(json.loads(request.content)))

        client = client_for(handler, management, contract=card_instance_contract())
        try:
            await client.attest()
            refs = [card["card_ref"] for card in observation()["own"]["cards"]]
            payload = {
                "command_id": str(uuid.uuid4()),
                "expected_revision": 7,
                "action": {"type": "play_sequence", "ordered_card_refs": refs[::-1], "target": 2},
            }
            await client.act("game", 1, payload)
            assert len(bodies) == 2 and bodies[0] == bodies[1]
            assert json.loads(bodies[1]) == payload
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("v2", [False, True])
def test_late_success_is_invalidated_by_concurrent_attestation_failure(v2):
    async def run():
        management = info()
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            if request.url.path.startswith("/debug/"):
                return response(management)
            started.set()
            await release.wait()
            return httpx.Response(204)

        client = client_for(
            handler, management, **({"contract": card_instance_contract()} if v2 else {})
        )
        try:
            await client.attest()
            pending = asyncio.create_task(client.delete("game"))
            await started.wait()
            management["seed_fingerprint"] = seed_fingerprint(
                "00000000-0000-0000-0000-000000000002"
            )
            with pytest.raises(AttestationError):
                await client.attest()
            release.set()
            with pytest.raises(AttestationError):
                await pending
            assert client.poisoned
        finally:
            release.set()
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "changes",
    [
        {"public_event_schema_version": 2},
        {"rule_set_version": "true-match-1v1-v1"},
        {"state_revision": True},
        {"events": [{"type": "safe", "player": True}]},
        {"action_options": options()},
    ],
)
def test_act_rejects_malformed_envelopes(changes):
    async def run():
        management = info()

        def handler(request):
            if request.url.path.startswith("/debug/"):
                return response(management)
            return response(acknowledgement(json.loads(request.content), **changes))

        client = client_for(handler, management, contract=card_instance_contract())
        try:
            await client.attest()
            with pytest.raises(ContractError):
                await client.act(
                    "game",
                    1,
                    {
                        "command_id": str(uuid.uuid4()),
                        "expected_revision": 7,
                        "action": {"type": "pray"},
                    },
                )
        finally:
            await client.close()

    asyncio.run(run())
