"""Attested HTTP boundary. Management data never becomes participant input."""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from typing import Any

import httpx

from gf_common import card_instance_validation as card_v2
from gf_common.card_instance import CardInstanceContract, parse_card_instance_contract
from gf_common.domain import (
    CATALOG_META,
    RULE_SET,
    ContractError,
    Json,
    UnsupportedContract,
    canonical,
    clone,
    closed,
    integer,
)
from gf_common.validation import (
    actor_action_options,
    catalog,
    observation,
    replay_closure,
    result,
    validate_events,
)

INFO_KEYS = {"schema_version", "server_kind", "server_instance_id", "seed_mode", "seed_fingerprint",
             "seed_use_limit", "max_invalid_attempts_per_decision", "waiting_for_players_timeout_secs",
             "player_action_timeout_secs", "max_game_turns", "server_idle_timeout_secs", "graceful_shutdown_timeout_secs"}
EFFECTIVE_KEYS = INFO_KEYS - {"server_kind", "server_instance_id", "seed_mode", "seed_fingerprint"}


class AttestationError(ContractError):
    pass


class ApiError(RuntimeError):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"simulator HTTP {status}")
        self.status = status
        self.body = body


class DeliveryUnknown(RuntimeError):
    pass


def seed_fingerprint(seed: str) -> str:
    return "v1-" + hashlib.sha256(b"godfield-test-server-seed-fingerprint-v1" + uuid.UUID(seed).bytes).hexdigest()


def validate_info(value: Any) -> Json:
    try:
        closed(value, INFO_KEYS)
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["server_kind"] != "test_server":
            raise ContractError("unsupported server info")
        instance = uuid.UUID(value["server_instance_id"])
        if instance.version != 4 or str(instance) != value["server_instance_id"]:
            raise ContractError("instance must be canonical UUID v4")
        mode = value["seed_mode"]
        if mode not in ("repeated", "sequence", "derived"):
            raise ContractError("unknown seed mode")
        if mode == "repeated":
            if not isinstance(value["seed_fingerprint"], str) or not re.fullmatch(r"v1-[0-9a-f]{64}", value["seed_fingerprint"]):
                raise ContractError("invalid seed fingerprint")
        elif value["seed_fingerprint"] is not None:
            raise ContractError("nonrepeated fingerprint must be null")
        if mode == "derived":
            if value["seed_use_limit"] is not None:
                raise ContractError("derived limit must be null")
        else:
            integer(value["seed_use_limit"], 1)
        integer(value["max_invalid_attempts_per_decision"], 1, 2**32 - 1)
        integer(value["max_game_turns"], 150, 2**32 - 1)
        for key in ("waiting_for_players_timeout_secs", "player_action_timeout_secs", "graceful_shutdown_timeout_secs"):
            integer(value[key], 1, 86400)
        integer(value["server_idle_timeout_secs"], 0, 86400)
        return clone(value)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise AttestationError(str(exc)) from exc


class AttestedClient:
    def __init__(self, base_url: str, *, expected_fingerprint: str, fixed_seed_experiment: bool,
                 expected_effective: Json, startup_info: Json | None = None,
                 request_seconds: float = 10, retries: int = 2,
                 transport: httpx.AsyncBaseTransport | None = None, deadline: float = float("inf"),
                 contract: CardInstanceContract | None = None) -> None:
        if not fixed_seed_experiment or not expected_fingerprint or type(retries) is not int or not 0 <= retries <= 2:
            raise AttestationError("explicit fixed-seed experiment and bounded retries required")
        if contract is not None:
            if type(contract) is not CardInstanceContract:
                raise ContractError("explicit pinned card-instance contract required")
            from dataclasses import asdict
            parse_card_instance_contract(asdict(contract))
        self.contract = contract
        self._gameplay_api = "v1" if contract is None else "v2"
        self.http = httpx.AsyncClient(base_url=base_url, timeout=request_seconds, transport=transport,
                                     follow_redirects=False, trust_env=False)
        self.expected_fingerprint = expected_fingerprint
        self.expected_effective = clone(expected_effective)
        self.startup_info = validate_info(startup_info) if startup_info is not None else None
        self.cached: Json | None = None
        self.poisoned = False
        self.needs_reattest = False
        self.retries = retries
        self.request_seconds = request_seconds
        self.deadline = deadline
        from gf_common.performance_metrics import PhaseMetrics
        from gf_simulator_client.performance_diagnostics import host_diagnostics
        self._performance = host_diagnostics()
        self.performance_metrics = (PhaseMetrics(scope=self._performance.current_scope or self._performance.scope(phase="idle"),
            start_ns=time.monotonic_ns(), counters=("event_page_successes", "event_page_failures", "event_transport_attempts"),
            histograms=("event_page",)) if self._performance is not None else None)

    async def close(self) -> None:
        await self.http.aclose()
        if self._performance is not None and self.performance_metrics is not None:
            self._performance.metrics(self.performance_metrics, role="legacy-coordinator")
            self.performance_metrics = None

    async def attest(self) -> Json:
        if self.poisoned:
            raise AttestationError("connection is poisoned")
        try:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("attestation deadline")
            async with asyncio.timeout(min(self.request_seconds, max(0.0, self.deadline - time.monotonic()))):
                response = await self.http.get("/debug/v1/test-server/info")
            if self.poisoned:
                raise AttestationError("connection was poisoned during attestation")
            directives = [s.strip().lower() for s in response.headers.get("cache-control", "").split(",")]
            if response.status_code != 200 or "no-store" not in directives:
                raise AttestationError("info requires HTTP 200 and no-store")
            value = validate_info(response.json())
            if value["seed_mode"] != "repeated" or value["seed_fingerprint"] != self.expected_fingerprint:
                raise AttestationError("repeated seed mode/fingerprint mismatch")
            if canonical({k: value[k] for k in EFFECTIVE_KEYS}) != canonical(self.expected_effective):
                raise AttestationError("effective server config mismatch")
            if self.startup_info is not None and value != self.startup_info:
                raise AttestationError("startup/HTTP info mismatch")
            if self.cached is not None and value != self.cached:
                raise AttestationError("server identity/effective config changed")
            self.cached = value
            self.needs_reattest = False
            return clone(value)
        except BaseException as exc:
            self.poisoned = True
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, AttestationError):
                raise
            raise AttestationError("attestation retrieval/validation failed") from exc

    async def _request(self, method: str, path: str, payload: Json | None = None,
                       *, retry: bool = False) -> Any:
        if self.cached is None or self.poisoned:
            raise AttestationError("request forbidden before successful attestation")
        if self.needs_reattest:
            await self.attest()
        # Serialize the retry payload once; never generate another command identity.
        body = clone(payload) if payload is not None else None
        for index in range(self.retries + 1 if retry else 1):
            if index:
                await self.attest()
            try:
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("request deadline")
                async with asyncio.timeout(min(self.request_seconds, max(0.0, self.deadline - time.monotonic()))):
                    if self.performance_metrics is not None and "/events/" in path:
                        self.performance_metrics.increment("event_transport_attempts")
                    response = await self.http.request(method, path, json=body)
            except (httpx.TransportError, TimeoutError) as exc:
                self.needs_reattest = True
                if retry and index < self.retries:
                    continue
                raise DeliveryUnknown(f"uncertain {method} delivery") from exc
            except asyncio.CancelledError:
                self.needs_reattest = True
                raise
            if self.poisoned:
                raise AttestationError("connection was poisoned during request")
            if response.status_code < 200 or response.status_code >= 300:
                try:
                    detail = response.json()
                except ValueError:
                    detail = {"body": response.text[:2048]}
                raise ApiError(response.status_code, detail)
            if response.status_code == 204:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise ContractError("invalid response JSON; no resend") from exc
        raise AssertionError("unreachable")

    async def public_catalog(self) -> Json:
        if self.contract is not None:
            card_v2.public_info(await self._request("GET", "/v2/info"))
            return card_v2.catalog(await self._request("GET", f"/v2/card-catalogs/{self.contract.catalog_version}"))
        info = await self._request("GET", "/v1/info")
        if info.get("api_version") != "v1" or info.get("rule_set_version") != RULE_SET or info.get("card_catalog") != CATALOG_META:
            raise UnsupportedContract("server/catalog version mismatch")
        return catalog(await self._request("GET", f"/v1/card-catalogs/{CATALOG_META['catalog_version']}"))

    async def create(self) -> Json:
        await self.attest()
        if self.contract is not None:
            await self.public_catalog()  # Version/catalog authority is checked before allocating a game.
        rule_set = RULE_SET if self.contract is None else self.contract.rule_set_version
        created = await self._request("POST", f"/{self._gameplay_api}/games", {"rule_set_version": rule_set,
                         "random_algorithm": "xoshiro256pp", "random_algorithm_version": "1"})
        closed(created, {"game_id", "status", "required_players", "joined_players", "state_revision", "rule_set_version", "random_algorithm", "random_algorithm_version"})
        if created["status"] != "waiting_for_players" or type(created["required_players"]) is not int or created["required_players"] != 2 or type(created["joined_players"]) is not int or created["joined_players"] != 0 or created["state_revision"] is not None or created["rule_set_version"] != rule_set or created["random_algorithm"] != "xoshiro256pp" or created["random_algorithm_version"] != "1":
            raise ContractError("unexpected create response")
        if not isinstance(created["game_id"], str) or not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", created["game_id"]):
            raise ContractError("invalid game ID")
        # Required post-Create fence, before any Join.
        await self.attest()
        return created

    async def join(self, game: str, command_id: str) -> Json:
        if self.contract is not None:
            card_v2.command_id(command_id)
        value = await self._request("POST", f"/{self._gameplay_api}/games/{game}/players", {"command_id": command_id}, retry=True)
        if self.contract is not None:
            return card_v2.join_response(value, game)
        closed(value, {"game_id", "player_id", "status", "joined_players", "required_players", "state_revision"})
        integer(value["player_id"], 1, 2)
        if value["game_id"] != game or value["required_players"] != 2:
            raise ContractError("invalid Join acknowledgement")
        return value

    async def observe(self, game: str, viewer: int) -> Json:
        if self.contract is not None:
            integer(viewer, 1, 2)
        validator = observation if self.contract is None else card_v2.observation
        return validator(await self._request("GET", f"/{self._gameplay_api}/games/{game}/observations/{viewer}"), viewer)

    async def actor_options(self, game: str, viewer: int, obs: Json) -> Json:
        integer(viewer, 1, 2)
        validator = observation if self.contract is None else card_v2.observation
        obs = validator(obs, viewer)
        if not obs["actionable"] or obs["current_player"] != viewer:
            raise ContractError("ActionOptions request requires current actor")
        value = await self._request("GET", f"/{self._gameplay_api}/games/{game}/action-options/{viewer}")
        options_validator = actor_action_options if self.contract is None else card_v2.actor_action_options
        return options_validator(value, obs, raw=True)

    async def events(self, game: str, viewer: int, after: str | None) -> tuple[list[Json], str | None]:
        from urllib.parse import urlencode
        if self.contract is not None:
            integer(viewer, 1, 2)
        accumulated: list[Json] = []
        cursor = after
        while True:
            query = urlencode({"limit": 1024, **({"after": cursor} if cursor is not None else {})})
            diagnostic_start = time.monotonic_ns() if self.performance_metrics is not None else None
            try:
                value = await self._request("GET", f"/{self._gameplay_api}/games/{game}/events/{viewer}?{query}")
                if self.contract is None:
                    closed(value, {"events", "next_cursor", "has_more"})
                    validate_events(value["events"])
                else:
                    card_v2.event_feed(value)
                if type(value["has_more"]) is not bool or value["next_cursor"] is not None and not isinstance(value["next_cursor"], str):
                    raise ContractError("invalid event pagination")
                accumulated.extend(value["events"])
                next_cursor = value["next_cursor"]
                if value["has_more"] and (next_cursor is None or next_cursor == cursor):
                    raise ContractError("event pagination made no progress")
            except BaseException:
                if self.performance_metrics is not None:
                    self.performance_metrics.increment("event_page_failures")
                raise
            else:
                if self.performance_metrics is not None and diagnostic_start is not None:
                    self.performance_metrics.increment("event_page_successes")
                    self.performance_metrics.interval("event_page", diagnostic_start, time.monotonic_ns())
            cursor = next_cursor
            if not value["has_more"]:
                return accumulated, cursor

    async def act(self, game: str, viewer: int, payload: Json) -> Json:
        if self.contract is not None:
            integer(viewer, 1, 2)
            payload = card_v2.action_command(payload)
        value = await self._request("POST", f"/{self._gameplay_api}/games/{game}/players/{viewer}/actions", payload, retry=True)
        if self.contract is not None:
            card_v2.act_response(value, payload, viewer)
            return {key: clone(value[key]) for key in ("command_id", "previous_revision", "state_revision", "status", "result")}
        closed(value, {"command_id", "previous_revision", "state_revision", "status", "events", "result", "next_observation", "action_options"})
        if value["command_id"] != payload["command_id"] or value["previous_revision"] != payload["expected_revision"]:
            raise ContractError("invalid action acknowledgement identity")
        integer(value["state_revision"], payload["expected_revision"] + 1)
        if value["status"] not in ("active", "terminal", "aborted"):
            raise ContractError("unknown action acknowledgement status")
        result(value["result"])
        # Deliberately do not read response events/options/next_observation into policy history.
        return {key: clone(value[key]) for key in ("command_id", "previous_revision", "state_revision", "status", "result")}

    async def replay(self, game: str, viewer: int) -> Json:
        if self.contract is not None:
            integer(viewer, 1, 2)
        value = await self._request("GET", f"/{self._gameplay_api}/games/{game}/replay/{viewer}")
        if self.contract is None:
            replay_closure(value, game, viewer)
        else:
            card_v2.replay(value, game, viewer)
        return value

    async def delete(self, game: str) -> None:
        await self._request("DELETE", f"/{self._gameplay_api}/games/{game}")
