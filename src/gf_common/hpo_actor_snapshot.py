"""Private inference-only actor grants; no training import or execution authority."""
from __future__ import annotations

import json
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gf_common.artifacts import ArtifactAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical

if TYPE_CHECKING:
    from gf_common.ram_bank import ReadonlyRamBankSet

_TOKEN = object()


def _fingerprint(banks: ReadonlyRamBankSet) -> tuple[Any, ...]:
    return tuple((stat.st_dev, stat.st_ino, stat.st_size, bank.descriptors)
        for bank in banks.banks for stat in (os.fstat(bank.fileno()),))


class _ActorGrantState:
    def __init__(self, budget: ArtifactAdmission, reservation: Any) -> None:
        self.budget, self.reservation = budget, reservation
        self.closed = False
        self.bindings: dict[int,Any] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True, init=False)
class OwnedActorSnapshot:
    """Exact saved origin, immutable inference grant; JSON cannot recreate it."""
    inference: ReadonlyRamBankSet
    generation: int
    identity: str
    checkpoint: str
    _metadata: bytes
    _origin: weakref.ReferenceType[Any]
    _check_origin: Callable[[Any], None]
    _backings: tuple[Any, ...]
    _state: _ActorGrantState

    def __init__(self, token: object, owner: Any, *, inference: ReadonlyRamBankSet,
                 metadata: Json, check_origin: Callable[[Any], None], budget: ArtifactAdmission,
                 reservation: Any) -> None:
        if token is not _TOKEN:
            raise ContractError('actor snapshot requires actual private snapshot issuance')
        values = {'inference': inference, 'generation': metadata['generation'],
            'identity': metadata['snapshot_identity'], 'checkpoint': metadata['inference_digest'],
            '_metadata': canonical(metadata).encode(), '_origin': weakref.ref(owner),
            '_check_origin': check_origin, '_backings': _fingerprint(inference),
            '_state': _ActorGrantState(budget, reservation)}
        for name, value in values.items():
            object.__setattr__(self, name, value)
        weakref.finalize(self, reservation.close)

    def require(self) -> Any:
        owner = self._origin()
        if self._state.closed or owner is None or getattr(owner, '_actor_capability', None) is not self:
            raise ContractError('actor snapshot is copied, retired or lacks its actual origin')
        self._check_origin(owner)
        if _fingerprint(self.inference) != self._backings:
            raise ContractError('actor snapshot inference backing changed')
        return owner

    @property
    def actor_metadata(self) -> Json:
        self.require()
        return json.loads(self._metadata)

    def bind(self, binding_owner: object) -> _ActorSnapshotBinding:
        origin = self.require()
        reservation = self._state.budget.reserve(4096)
        binding=_ActorSnapshotBinding(_TOKEN, self, binding_owner, origin, reservation)
        self._state.bindings[id(binding)]=binding
        return binding

    def _retire_origin(self, owner: object) -> None:
        if owner is not self._origin():
            raise ContractError('actor snapshot origin retirement differs')
        self._state.closed = True
        self._state.reservation.close()


class _ActorSnapshotBinding:
    def __init__(self, token: object, snapshot: OwnedActorSnapshot, binding_owner: object,
                 origin: object, reservation: Any) -> None:
        if token is not _TOKEN:
            raise ContractError('actual actor snapshot binding required')
        self._snapshot, self._binding_owner, self._origin = snapshot, binding_owner, origin
        self._reservation, self._closed = reservation, False
        for bank in snapshot.inference.banks:
            bank._pins += 1

    def require(self, binding_owner: object, snapshot: OwnedActorSnapshot) -> None:
        if (self._closed or self._binding_owner is not binding_owner or self._snapshot is not snapshot
                or snapshot._state.bindings.get(id(self)) is not self):
            raise ContractError('actor snapshot binding owner differs')
        if snapshot.require() is not self._origin:
            raise ContractError('actor snapshot permanent origin changed')

    def close(self, binding_owner: object) -> None:
        self.require(binding_owner, self._snapshot)
        for bank in self._snapshot.inference.banks:
            bank._pins -= 1
        self._closed = True
        del self._snapshot._state.bindings[id(self)]
        self._origin = None
        self._reservation.close()


def _issue_actor_snapshot(owner: Any, *, inference: ReadonlyRamBankSet, metadata: Json,
                          check_origin: Callable[[Any], None], budget: ArtifactAdmission) -> OwnedActorSnapshot:
    """Only actual local-save/Wire7 completion invokes this private mint."""
    check_origin(owner)
    reservation = budget.reserve(16 * json_bound(metadata) + 4096)
    try:
        return OwnedActorSnapshot(_TOKEN, owner, inference=inference, metadata=metadata,
            check_origin=check_origin, budget=budget, reservation=reservation)
    except BaseException:
        reservation.close()
        raise


def checked_evaluation_use(value: Json, metadata: Json, request: Json, options: Json) -> str:
    """Pure actor9 use binding; JSON supplies no execution or producer authority."""
    from pathlib import Path

    from gf_common.card_instance_collection import checked_hpo_evaluation_parent
    from gf_common.domain import closed, integer
    raw=closed(value,{'schema_version','kind','evaluation_id','finals','selected','outer_request'})
    integer(raw['schema_version'],1,1)
    if (raw['kind']!='owned-evaluation-final-policy-use-v1' or raw['evaluation_id']!=request['run_id']
            or request['command']!='BEGIN' or type(request['generation']) is not int or request['generation']!=0
            or canonical(raw['outer_request'])!=canonical(request)):
        raise ContractError('evaluation actor use differs from original outer request/control0')
    finals=raw['finals']
    if type(finals) is not list or not 1<=len(finals)<=2:
        raise ContractError('evaluation requires its fixed one/two final policies')
    for entry in finals:
        closed(entry,{'parent','snapshot_identity'})
        checked_hpo_evaluation_parent(entry['parent'])
        sha=entry['snapshot_identity']
        if type(sha) is not str or len(sha)!=64 or set(sha)-set('0123456789abcdef'):
            raise ContractError('evaluation source snapshot identity required')
    selected=raw['selected']
    if selected not in finals: raise ContractError('selected evaluation policy is outside original final set')
    parent=selected['parent']
    if (metadata['generation']!=parent['generation'] or type(metadata['generation']) is not int
            or metadata['snapshot_identity']!=selected['snapshot_identity']
            or metadata['inference_digest']!=parent['inference_sha256']
            or options['expected_manifest_digest']!=parent['inference_sha256']
            or Path(options['checkpoint'])!=Path(parent['run_path'])/'artifacts'/f"{parent['generation']:06}"):
        raise ContractError('evaluation selected source/metadata/checkpoint mismatch')
    return canonical(finals)
