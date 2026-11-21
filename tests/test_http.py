import asyncio
import copy
import json
import uuid

import httpx
import pytest

from gf_common.domain import ContractError
from gf_simulator_client.http import (
    EFFECTIVE_KEYS,
    ApiError,
    AttestationError,
    AttestedClient,
    DeliveryUnknown,
    seed_fingerprint,
    validate_info,
)
from tests.conftest import fixture

SEED = "00000000-0000-0000-0000-000000000001"


def info():
    return {"schema_version": 1, "server_kind": "test_server", "server_instance_id": str(uuid.uuid4()),
            "seed_mode": "repeated", "seed_fingerprint": seed_fingerprint(SEED), "seed_use_limit": 2,
            "max_invalid_attempts_per_decision": 3, "waiting_for_players_timeout_secs": 60,
            "player_action_timeout_secs": 300, "max_game_turns": 2000,
            "server_idle_timeout_secs": 0, "graceful_shutdown_timeout_secs": 30}


def client_for(handler, value=None, **kwargs):
    value = value or info()
    return AttestedClient("http://test", expected_fingerprint=seed_fingerprint(SEED), fixed_seed_experiment=True,
                          expected_effective={k: value[k] for k in EFFECTIVE_KEYS},
                          transport=httpx.MockTransport(handler), **kwargs)


def response(value, status=200, **kwargs):
    return httpx.Response(status, json=value, headers={"cache-control": "no-store"}, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("seed_use_limit", 0),
    ("max_game_turns", 149), ("player_action_timeout_secs", 0),
    ("server_idle_timeout_secs", 86401), ("server_instance_id", "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"),
    ("seed_fingerprint", "v1-" + "A" * 64), ("seed_mode", "unknown"),
])
def test_info_strict_domains(field, value):
    candidate = info()
    candidate[field] = value
    with pytest.raises(AttestationError):
        validate_info(candidate)


def test_mode_nullability_and_extra_missing():
    value = info()
    assert validate_info(value) == value
    for key in value:
        bad = dict(value)
        del bad[key]
        with pytest.raises(AttestationError):
            validate_info(bad)
    value["extra"] = 1
    with pytest.raises(AttestationError):
        validate_info(value)
    value.pop("extra")
    value["seed_mode"] = "derived"
    value["seed_fingerprint"] = None
    value["seed_use_limit"] = None
    assert validate_info(value) == value
    value["seed_mode"] = "sequence"
    with pytest.raises(AttestationError):
        validate_info(value)


def test_gate_poison_is_sticky_on_http_or_identity_failure():
    async def run():
        value = info()
        calls = []
        def handler(request):
            calls.append(request.url.path)
            return response(value)
        client = client_for(handler, value)
        try:
            with pytest.raises(AttestationError):
                await client.join("game", str(uuid.uuid4()))
            assert calls == []
            await client.attest()
            value["server_instance_id"] = str(uuid.uuid4())
            with pytest.raises(AttestationError):
                await client.attest()
            assert client.poisoned
            value["server_instance_id"] = client.cached["server_instance_id"]
            for operation in (client.attest(), client.create(), client.join("game", "id")):
                with pytest.raises(AttestationError):
                    await operation
            assert len(calls) == 2
        finally:
            await client.close()
    asyncio.run(run())


def test_attestation_cancel_poisons_and_no_store_required():
    async def run():
        value = info()
        started = asyncio.Event()
        async def handler(request):
            started.set()
            await asyncio.Event().wait()
        client = client_for(handler, value)
        task = asyncio.create_task(client.attest())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.poisoned
        with pytest.raises(AttestationError):
            await client.create()
        await client.close()
        client = client_for(lambda request: httpx.Response(200, json=value), value)
        with pytest.raises(AttestationError):
            await client.attest()
        await client.close()
    asyncio.run(run())


def test_create_fenced_before_and_after_without_retry():
    async def run():
        value = info()
        paths = []
        created = fixture("create_public.json")
        def handler(request):
            paths.append(request.url.path)
            if request.url.path.endswith("info"):
                return response(value)
            return response(created, 201)
        client = client_for(handler, value)
        try:
            await client.attest()
            assert await client.create() == created
            assert paths == ["/debug/v1/test-server/info", "/debug/v1/test-server/info", "/v1/games", "/debug/v1/test-server/info"]
        finally:
            await client.close()
        paths.clear()
        def fail(request):
            paths.append(request.url.path)
            if request.url.path.endswith("info"):
                return response(value)
            raise httpx.ReadTimeout("uncertain", request=request)
        client = client_for(fail, value)
        await client.attest()
        with pytest.raises(DeliveryUnknown):
            await client.create()
        assert paths.count("/v1/games") == 1
        await client.close()
    asyncio.run(run())


def test_retry_identity_bounded_and_reattested():
    async def run():
        value = info()
        seen = []
        info_calls = 0
        def handler(request):
            nonlocal info_calls
            if request.url.path.endswith("info"):
                info_calls += 1
                return response(value)
            payload = json.loads(request.content)
            seen.append(payload)
            if len(seen) < 3:
                raise httpx.ReadTimeout("lost ack", request=request)
            return response({"command_id": payload["command_id"], "previous_revision": 7, "state_revision": 8,
                             "status": "active", "events": [{"hidden": "must be ignored"}], "result": None,
                             "next_observation": {"hidden": "must be ignored"}, "action_options": {"hidden": True}})
        client = client_for(handler, value)
        await client.attest()
        payload = {"command_id": str(uuid.uuid4()), "expected_revision": 7, "action": {"type": "pray"}}
        result = await client.act("game", 1, payload)
        assert len(seen) == 3 and seen == [payload] * 3
        assert info_calls == 3
        assert "events" not in result and "action_options" not in result
        await client.close()
    asyncio.run(run())


def test_failed_recovery_no_resend_and_rejection_no_policy_retry():
    async def run():
        value = info()
        sends = 0
        gates = 0
        def handler(request):
            nonlocal gates, sends
            if request.url.path.endswith("info"):
                gates += 1
                return response(value) if gates == 1 else httpx.Response(500)
            sends += 1
            raise httpx.ConnectError("uncertain", request=request)
        client = client_for(handler, value)
        await client.attest()
        with pytest.raises(AttestationError):
            await client.join("game", "same")
        assert sends == 1 and client.poisoned
        await client.close()
        def rejection(request):
            if request.url.path.endswith("info"):
                return response(value)
            return response({"error": {"code": "INVALID_ACTION", "details": {"hidden": True}}}, 422)
        client = client_for(rejection, value)
        await client.attest()
        with pytest.raises(ApiError) as captured:
            await client.act("game", 1, {"command_id": "id", "expected_revision": 0, "action": {"type": "pray"}})
        assert captured.value.status == 422
        assert not client.poisoned
        await client.close()
    asyncio.run(run())


def test_paged_events_only_and_cursor_progress():
    async def run():
        value = info()
        cursors = []
        def handler(request):
            if request.url.path.endswith("info"):
                return response(value)
            after = request.url.params.get("after")
            cursors.append(after)
            if after is None:
                return response({"events": [{"type": "turn_started", "turn": 1, "active_player": 1}], "has_more": True, "next_cursor": "opaque-one"})
            return response({"events": [{"type": "action_accepted", "player": 1}], "has_more": False, "next_cursor": "opaque-two"})
        client = client_for(handler, value)
        await client.attest()
        events, cursor = await client.events("game", 1, None)
        assert len(events) == 2 and cursor == "opaque-two"
        assert cursors == [None, "opaque-one"]
        await client.close()
    asyncio.run(run())


def test_startup_effective_mismatch_stops():
    async def run():
        value = info()
        startup = copy.deepcopy(value)
        startup["max_game_turns"] = 2001
        client = client_for(lambda request: response(value), value, startup_info=startup)
        with pytest.raises(AttestationError):
            await client.attest()
        await client.close()
    asyncio.run(run())


def test_exhausted_transport_requires_gate_before_replay_and_delete():
    async def run():
        value = info()
        paths = []
        def handler(request):
            paths.append(request.url.path)
            if request.url.path.endswith("info"):
                return response(value)
            if request.url.path.endswith("actions"):
                raise httpx.ReadTimeout("lost", request=request)
            if request.method == "DELETE":
                return httpx.Response(204)
            return response({"game_id": "game", "viewer_player_id": 1, "observation_schema_version": "1", "episode_end": None, "diagnostic": True})
        client = client_for(handler, value, retries=0)
        await client.attest()
        with pytest.raises(DeliveryUnknown):
            await client.act("game", 1, {"command_id": "id", "expected_revision": 0, "action": {"type": "pray"}})
        assert client.needs_reattest
        await client.replay("game", 1)
        assert paths[-2:] == ["/debug/v1/test-server/info", "/v1/games/game/replay/1"]
        client.needs_reattest = True
        await client.delete("game")
        assert paths[-2:] == ["/debug/v1/test-server/info", "/v1/games/game"]
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("game_id", "other-game"), ("game_id", None), ("viewer_player_id", 2), ("observation_schema_version", "future")])
def test_replay_rejects_wrong_or_missing_requested_identity(field, value):
    async def run():
        attestation = info()
        replay = {"game_id": "game", "viewer_player_id": 1, "observation_schema_version": "1"}
        if value is None:
            replay.pop(field)
        else:
            replay[field] = value
        client = client_for(lambda request: response(attestation if request.url.path.endswith("info") else replay), attestation)
        try:
            await client.attest()
            with pytest.raises(ContractError, match="replay game/perspective"):
                await client.replay("game", 1)
        finally:
            await client.close()
    asyncio.run(run())


def test_attestation_exhausted_deadline_sends_nothing_and_is_sticky():
    async def run():
        seen = []
        client = client_for(lambda request: seen.append(request) or response(info()))
        client.deadline = 0
        with pytest.raises(AttestationError):
            await client.attest()
        assert client.poisoned and not seen
        await client.close()
    asyncio.run(run())
