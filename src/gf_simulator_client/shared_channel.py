"""Own-participant score channel. Safe to import without runtime side effects."""
from __future__ import annotations

import math
import socket
import time
from typing import Any

from gf_common.domain import ContractError
from gf_simulator_client.framing import read_frame, write_frame


class ParticipantScoreChannel:
    def __init__(self,fd: int) -> None:
        self.socket=socket.socket(fileno=fd)
        self.socket.set_inheritable(False)
        self.socket.settimeout(35.)
        self.stream=self.socket.makefile('rwb',buffering=0)
        self.routing=None
        self.deadline: float | None=None

    def bind_request(self,routing: Any,deadline: Any) -> None:
        if routing is None:
            self.routing=self.deadline=None
            return
        from gf_common.domain import closed
        from gf_common.shared_requests import REQUEST_FIELDS
        closed(routing,REQUEST_FIELDS)
        if type(deadline) not in (float,int) or not math.isfinite(deadline):
            raise ContractError('original absolute shared request deadline required')
        self.routing,self.deadline=routing,float(deadline)

    def check_deadline(self) -> None:
        if self.deadline is None or time.monotonic()>=self.deadline:
            raise TimeoutError('shared inference response expired')

    def _io_deadline(self) -> None:
        self.check_deadline()
        assert self.deadline is not None
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('shared inference response expired')
        self.socket.settimeout(remaining)

    def request(self,frame: Any,checkpoint: str,numeric: Any) -> Any:
        routing=self.routing
        if (routing is None or routing['checkpoint']!=checkpoint or routing['revision']!=frame.revision
                or routing['participant']!=frame.actor):
            raise ContractError('numeric request lacks own original Observation capability')
        self.check_deadline()
        stream=_DeadlineStream(self)
        write_frame(stream,{'routing':routing,'numeric':numeric})
        response=read_frame(stream)
        self.check_deadline()
        if response.get('routing')!=routing:
            raise ContractError('shared response discarded: original request ownership differs')
        self.routing=None
        return response


class _DeadlineStream:
    def __init__(self,owner: ParticipantScoreChannel) -> None:self.owner=owner

    def read(self,size: int) -> bytes:
        self.owner._io_deadline()
        result=self.owner.stream.read(size)
        return b'' if result is None else result

    def write(self,value: bytes) -> int | None:
        self.owner._io_deadline()
        return self.owner.stream.write(value)

    def flush(self) -> None:self.owner.stream.flush()


def policy_loader(channel: ParticipantScoreChannel,model_id: str,options: Any,actions: Any,rules: Any,load: Any) -> Any:
    if model_id not in ('mlp','gru','transformer'):return load()
    from gf_models.shared_policy import SharedParticipantPolicy
    return SharedParticipantPolicy(options,actions,rules,channel)
