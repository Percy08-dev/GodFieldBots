"""Invocation-local pending request ownership. No model or game-rule dependency."""
from __future__ import annotations

from gf_common.domain import ContractError, Json, closed, integer

REQUEST_FIELDS = {'execution','game','participant','request','revision','checkpoint'}


class RequestAuthority:
    def __init__(self, execution: str) -> None:
        self.execution=execution
        self.pending: dict[str,Json]={}
        self.serial=0
        self.accepted=self.completed=self.discarded=0

    @property
    def empty(self) -> bool:
        return not self.pending

    def begin(self, endpoint: str, *, game: str,participant: int,revision: int,checkpoint: str) -> Json:
        integer(participant,1,2)
        integer(revision,0,2**63-1)
        if endpoint in self.pending or any(row['game']==game for row in self.pending.values()):
            raise ContractError('one pending shared request per actual game required')
        self.serial+=1
        row={'execution':self.execution,'game':game,'participant':participant,'request':self.serial,
            'revision':revision,'checkpoint':checkpoint}
        self.pending[endpoint]=row
        self.accepted+=1
        return dict(row)

    def match(self, endpoint: str,value: Json) -> None:
        closed(value,REQUEST_FIELDS)
        if endpoint not in self.pending or value!=self.pending[endpoint]:
            raise ContractError('stale or mismatched shared inference response')

    def retire(self, endpoint: str, *, completed: bool = False) -> None:
        if self.pending.pop(endpoint,None) is not None:
            if completed:self.completed+=1
            else:self.discarded+=1
