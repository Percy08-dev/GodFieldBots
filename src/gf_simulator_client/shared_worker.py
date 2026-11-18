"""Original participant validation with a CPU-only clean transport bootstrap."""
from __future__ import annotations

import sys
import time
from typing import Any

from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker


class SharedParticipantWorker(CardInstanceParticipantWorker):
    def __init__(self,*args: Any,service: Any,game: str,viewer: int,**kwargs: Any) -> None:
        self.service,self.game,self.viewer=service,game,viewer
        self.endpoint=None
        self._pending_shared=None
        super().__init__(*args,**kwargs)

    def _spawn(self,ledger: Any,residency: int,arguments: list[str],environment: dict[str,str],*,pass_fds: tuple[int,...]=()) -> None:
        self.endpoint,fd=self.service.acquire_endpoint(self)
        super()._spawn(ledger,0,[sys.executable,'-E','-m','gf_simulator_client.shared_actor_bootstrap',str(fd)],
            environment,pass_fds=(fd,))

    async def _roundtrip(self,payload: Any) -> Any:
        routing=None
        if payload.get('kind')=='select' and self.model_id=='mlp':
            deadline=min(self.service.execution_deadline,time.monotonic()+self.timeout)
            routing=self.service.authorize(self.endpoint,self,game=self.game,participant=self.viewer,
                revision=payload['frame']['revision'],checkpoint=self.initial['policy_options']['expected_manifest_digest'],deadline=deadline)
            payload={**payload,'shared_request':routing,'shared_deadline':deadline}
            self._pending_shared=routing
        completed=False
        primary=None
        try:
            result=await super()._roundtrip(payload)
            completed=True
            return result
        except BaseException as error:
            primary=error
            raise
        finally:
            if self._pending_shared is routing and routing is not None:
                try:
                    self.service.complete(self.endpoint,routing,completed=completed)
                    self._pending_shared=None
                except BaseException as cleanup:
                    if primary is None:raise
                    primary.add_note('shared request retirement unconfirmed: '+type(cleanup).__name__)

    async def _abort(self,original: BaseException) -> Any:
        if self._pending_shared is not None:
            try:
                self.service.complete(self.endpoint,self._pending_shared,completed=False)
                self._pending_shared=None
            except BaseException as cleanup:  # noqa: BLE001 - original request failure keeps precedence.
                original.add_note('shared request invalidation unconfirmed: '+type(cleanup).__name__)
        await super()._abort(original)

    async def _close_owned(self,timeout: float,started: float) -> None:
        await super()._close_owned(timeout,started)
        if self.endpoint is not None:
            self.service.release_endpoint(self.endpoint,self)
            self.endpoint=None
