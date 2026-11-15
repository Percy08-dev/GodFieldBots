"""Inference-only live artifact metadata for the explicit owned actor route."""
from __future__ import annotations

import hashlib
import time
from typing import Any

from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, clone
from gf_common.hpo_actor_snapshot import OwnedActorSnapshot
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT


class OwnedGameArtifacts:
    def __init__(self, entries: dict[str,tuple[Any,Any,Any,Json]], *, budget: Any, parent_budget: ArtifactPhaseAdmission | None = None) -> None:
        if not entries or any(type(row[0]) is not OwnedActorSnapshot for row in entries.values()):
            raise ContractError('actual common actor snapshot capabilities required')
        if type(parent_budget) is not ArtifactPhaseAdmission or parent_budget.source is not budget:
            raise ContractError('owned inference metadata requires SAME shared parent phase')
        self._reservation=parent_budget.reserve(65536+sum(32*json_bound(row[2])+32*json_bound(row[3])+65536 for row in entries.values()))
        self._entries: Json={}
        self._caps: dict[str,OwnedActorSnapshot]={}
        self._leases: list[Any]=[]
        self._released=False
        try:
            for label,(cap,binding,files,manifest) in entries.items():
                cap.require()
                metadata=cap.actor_metadata
                if budget is not cap._state.budget:
                    raise ContractError('owned inference metadata requires SAME Source account')
                if (hashlib.sha256((canonical(manifest)+'\n').encode()).hexdigest()!=cap.checkpoint
                        or canonical({name:value for name,value in manifest.items() if name!='weights'})!=canonical(metadata['inference'])):
                    raise ContractError('owned saved inference metadata differs from actual snapshot')
                if binding.policy_options['expected_manifest_digest']!=metadata['inference_digest']:
                    raise ContractError('owned policy binding differs from actual actor capability')
                self._leases.append(cap.bind(self))
                self._caps[label]=cap
                self._entries[label]={'input_contract':EXTERNAL_INPUT_CONTRACT,
                    'policy_options':clone(binding.policy_options),'inference_manifest':clone(manifest),
                    'files':clone(files)}
        except BaseException:
            self.release()
            raise

    def _live(self) -> None:
        if self._released: raise ContractError('owned inference metadata is released')
        for cap in self._caps.values(): cap.require()

    def require_options(self, model_id: str, options: Json) -> None:
        self._live()
        if not any(entry['policy_options']['model']['model_id']==model_id
                and canonical(entry['policy_options'])==canonical(options) for entry in self._entries.values()):
            raise ContractError('owned inference options differ from actual snapshot binding')

    def journal_metadata(self, label: str) -> Json:
        self._live()
        if label not in self._entries: raise ContractError('unknown actual owned policy label')
        return clone(self._entries[label])

    def finalize(self, *, deadline: float) -> Json:
        self._live()
        if time.monotonic()>=deadline: raise TimeoutError('owned inference metadata deadline')
        return {'input_contract':EXTERNAL_INPUT_CONTRACT,'policies':clone(self._entries)}

    def release(self) -> None:
        if self._released: return
        for lease in self._leases: lease.close(self)
        self._leases.clear()
        self._caps.clear()
        self._entries.clear()
        self._reservation.close()
        self._released=True
