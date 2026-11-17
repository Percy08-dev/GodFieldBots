"""Same-original-observation two-read overlap at the original client seam."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from gf_common.domain import ContractError


@dataclass
class _Episode:
    owner: Any
    enabled: bool
    identity: tuple[Any, str, int, Any, int] | None = None
    options: Any = None
    ready: bool = False
    tasks: list[Any] = field(default_factory=list)


class ReadOverlap:
    def __init__(self) -> None:
        self.current: ContextVar[_Episode | None] = ContextVar('collection_http_episode', default=None)
        self.pairs = 0

    @contextmanager
    def episode(self, enabled: bool) -> Any:
        state = _Episode(asyncio.current_task(), enabled)
        token = self.current.set(state)
        try:
            yield
        finally:
            state.identity = state.options = None
            state.ready = False
            self.current.reset(token)

    def state(self) -> _Episode | None:
        state = self.current.get()
        if state is not None and state.enabled and state.owner is not asyncio.current_task():
            raise ContractError('enabled HTTP overlap belongs to original episode Task')
        return state if state is not None and state.enabled else None

    def invalidate(self) -> None:
        state = self.state()
        if state is not None:
            if state.tasks:
                raise ContractError('overlap tasks still owned at invalidation')
            state.identity = state.options = None
            state.ready = False

    def observed(self, client: Any, game: str, actor: int, obs: Any) -> None:
        state = self.state()
        if state is None:
            return
        if state.ready or state.tasks:
            raise ContractError('unconsumed HTTP overlap identity')
        state.identity = None
        if (obs['actionable'] and obs['result'] is None
                and obs['viewer_player_id'] == obs['current_player'] == actor):
            state.identity = (client, game, actor, obs, obs['state_revision'])

    @staticmethod
    def matches(state: _Episode, client: Any, game: str, actor: int, obs: Any = None) -> bool:
        if state.identity is None:
            return False
        c, g, a, o, rev = state.identity
        return c is client and g == game and a == actor and (obs is None or o is obs) and o['state_revision'] == rev

    async def events(self, original_events: Any, original_options: Any, client: Any,
                     game: str, actor: int, cursor: str | None) -> Any:
        state = self.state()
        if state is None:
            return await original_events(client, game, actor, cursor)
        if not self.matches(state, client, game, actor) or state.ready or state.tasks:
            raise ContractError('HTTP overlap events identity mismatch')
        assert state.identity is not None
        obs = state.identity[3]
        def own(coroutine: Any) -> None:
            try:
                created = asyncio.create_task(coroutine)
            except BaseException:
                # A task factory may transfer this exact coroutine then raise.
                # Recover actual owners by object identity, never a task name.
                transferred = [candidate for candidate in asyncio.all_tasks()
                               if candidate.get_coro() is coroutine]
                state.tasks.extend(candidate for candidate in transferred if candidate not in state.tasks)
                if not transferred:
                    coroutine.close()
                raise
            state.tasks.append(created)

        try:
            own(original_events(client, game, actor, cursor))
            own(original_options(client, game, actor, obs))
            self.pairs += 1
            events, options = await asyncio.gather(*state.tasks)
            if not self.matches(state, client, game, actor, obs):
                raise ContractError('HTTP overlap completed identity mismatch')
            state.options, state.ready = options, True
            return events
        except BaseException:
            for owned in state.tasks:
                owned.cancel()
            joined = asyncio.gather(*state.tasks, return_exceptions=True)
            # Repeated cancellation must not detach owned original reads.
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError:
                    continue
            joined.result()
            state.identity = state.options = None
            state.ready = False
            raise
        finally:
            state.tasks.clear()

    async def options(self, original: Any, client: Any, game: str, actor: int, obs: Any) -> Any:
        state = self.state()
        if state is None:
            return await original(client, game, actor, obs)
        if not state.ready or not self.matches(state, client, game, actor, obs):
            raise ContractError('HTTP overlap options identity mismatch')
        result = state.options
        self.invalidate()
        return result


def patches() -> tuple[list[Any], ReadOverlap]:
    from gf_simulator_client.environment import SimulatorEnvironment
    from gf_simulator_client.http import AttestedClient
    overlap = ReadOverlap()
    episode = SimulatorEnvironment._episode_body
    finalize = SimulatorEnvironment._finalize_episode
    observe, events, options, act = (AttestedClient.observe, AttestedClient.events,
                                     AttestedClient.actor_options, AttestedClient.act)

    async def scoped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with overlap.episode(self.settings.actions.use_action_options):
            return await episode(self, *args, **kwargs)

    async def observed(client: Any, game: str, actor: int) -> Any:
        obs = await observe(client, game, actor)
        overlap.observed(client, game, actor, obs)
        return obs

    async def read_events(client: Any, game: str, actor: int, cursor: Any) -> Any:
        return await overlap.events(events, options, client, game, actor, cursor)

    async def read_options(client: Any, game: str, actor: int, obs: Any) -> Any:
        return await overlap.options(options, client, game, actor, obs)

    async def acted(*args: Any, **kwargs: Any) -> Any:
        overlap.invalidate()
        return await act(*args, **kwargs)

    async def finalized(*args: Any, **kwargs: Any) -> Any:
        # Finalization is an independently shielded Task in the original loop.
        token = overlap.current.set(None)
        try:
            return await finalize(*args, **kwargs)
        finally:
            overlap.current.reset(token)

    return [(SimulatorEnvironment, '_episode_body', scoped),
            (SimulatorEnvironment, '_finalize_episode', finalized),
            (AttestedClient, 'observe', observed), (AttestedClient, 'events', read_events),
            (AttestedClient, 'actor_options', read_options), (AttestedClient, 'act', acted)], overlap
