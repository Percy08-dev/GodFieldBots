"""Private final inference ownership, separate from every execution/job scope."""
from __future__ import annotations

import hashlib
import json
import weakref
from typing import Any

from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.ram_bank import ReadonlyRamBankSet

_TOKEN=object()
_FACTS=('run_id','updates','used_decisions','collected_decisions','accepted_decisions','optimizer_steps',
    'generation_sha256','resume_identity','config_sha256')


def management_records(records: list[Json]) -> list[Json]:
    """Only record count/order and final manifest identity serve CLI consumers.

    _final_policy_reference checks count/final generation and hpo_binding reads
    the final inference hash. The ORIGINAL complete chain digest is retained
    separately; no schedule, pool RNG or numerical resume payload survives here.
    Current counters live in readonly run metadata and the inference file map
    lives in its own inference-only property.
    """
    return [{'generation':record['generation'],**({'inference_sha256':record['inference_sha256']}
        if index==len(records)-1 else {})} for index,record in enumerate(records)]


class _PreparedInference:
    def __init__(self, token: object, operations: Any) -> None:
        if token is not _TOKEN: raise ContractError('actual final inference preparation required')
        registry=operations.owned_registry
        registry.require()
        snapshot=registry.current
        snapshot.require_process(registry.process)
        state=registry.state
        if (state['used_decisions']!=state['effective_config']['hpo_spec']['decision_budget']
                or state['updates']!=snapshot.generation or registry.pending is not None):
            raise ContractError('only actual final committed D can prepare completion data')
        # Allocation-free bounds over already admitted, retained metadata.
        facts=snapshot._control.value
        charge=32*(json_bound(facts['actor_metadata'])+json_bound(facts['inference_manifest'])
            +json_bound(registry.records)+json_bound(state))+65536
        self._budget,self.parent_budget=registry.budget,registry.parent_budget
        self._reservation=self.parent_budget.reserve(charge)
        self._issuer: Any=weakref.ref(operations)
        self._writer: Any=weakref.ref(registry.writer)
        self._closed=False; self._completed=False; self._actor_capability: Any=None
        self._state=b''; self._state_sha256=''
        self._original_backings: Any=None
        banks=[]
        try:
            self._actor_metadata=canonical(snapshot.actor_metadata).encode()
            self._manifest=canonical(snapshot.inference_manifest).encode()
            self._records=canonical(management_records(registry.records)).encode()
            self._generations_sha256=digest(registry.records)
            self._initial=tuple(state[key] for key in _FACTS)
            self._identity_digest=digest(state['identity'])
            self.generation,self.identity=snapshot.generation,snapshot.identity
            self._origin_facts=(self.generation,self.identity)
            self._generation_sha256=state['generation_sha256']
            self._files=canonical({name:sha for name,sha in facts['files'].items()
                if name=='inference.json' or name.startswith('weights-')}).encode()
            for bank in snapshot.inference.banks: banks.append(bank.retain(self._budget))
            self.inference=ReadonlyRamBankSet(tuple(banks))
        except BaseException:
            for bank in banks: bank.close()
            self._reservation.close()
            raise

    def __reduce__(self) -> Any: raise TypeError('private final inference ownership cannot be copied/persisted')

    def require(self) -> None:
        if self._closed or not self._completed: raise ContractError('data preparation grants no completion')
        if (self.generation,self.identity)!=self._origin_facts: raise ContractError('final inference origin changed')
        from gf_training.hpo_owned_snapshot import _backings
        if _backings(self.inference)!=self._original_backings: raise ContractError('final inference backing changed')

    @property
    def files(self) -> Json: return json.loads(self._files)
    @property
    def state(self) -> Json:
        self.require(); return json.loads(self._state)
    @property
    def records(self) -> list[Json]:
        self.require(); return json.loads(self._records)
    @property
    def actor_metadata(self) -> Json:
        self.require(); return json.loads(self._actor_metadata)
    @property
    def inference_manifest(self) -> Json:
        self.require(); return json.loads(self._manifest)
    @property
    def actor_capability(self) -> Any:
        self.require()
        from gf_common.hpo_actor_snapshot import _issue_actor_snapshot
        if self._actor_capability is None:
            self._actor_capability=_issue_actor_snapshot(self,inference=self.inference,metadata=self.actor_metadata,
                check_origin=_check_data_origin,budget=self._budget)
        self._actor_capability.require()
        return self._actor_capability

    def close(self) -> None:
        if self._closed: return
        self.inference.close()
        if self._actor_capability is not None: self._actor_capability._retire_origin(self)
        self._reservation.close()
        self._actor_metadata=self._manifest=self._records=self._state=b''
        self._initial=()
        self._files=b'{}'
        self._issuer=lambda:None; self._writer=lambda:None
        self._closed=True


class OwnedRunCompletion(_PreparedInference):
    """Actual successful run completion; carries no open training job."""


class OwnedExternalFinalPolicy(_PreparedInference):
    """Validated external final policy; never a private run completion."""


class OwnedEvaluationPolicy(_PreparedInference):
    """Inference retained from actual completion into a new evaluation account."""


def _check_data_origin(owner: Any) -> None:
    if type(owner) not in (OwnedRunCompletion,OwnedExternalFinalPolicy,OwnedEvaluationPolicy):
        raise ContractError('actual final inference data owner required')
    owner.require()


def _prepare_completion(operations: Any) -> OwnedRunCompletion:
    return OwnedRunCompletion(_TOKEN,operations)


def digest_record(value: Any) -> str:
    return hashlib.sha256((canonical(value)+'\n').encode()).hexdigest()


def _mint_completion(operations: Any, state: Json) -> OwnedRunCompletion:
    candidate=operations._prepared_completion
    if (type(candidate) is not OwnedRunCompletion or candidate._issuer() is not operations
            or candidate._completed or candidate._closed or not operations.owned_registry._closed
            or operations.receipt is None or state['status']!='complete'
            or state['used_decisions']!=state['effective_config']['hpo_spec']['decision_budget']
            or state['updates']!=candidate.generation or state.get('quarantine') or state.get('cleanup_failure')):
        raise ContractError('completion requires actual final state, cleanup and original prepared data')
    operations.job.require_confirmed_termination(operations.receipt)
    writer=candidate._writer()
    if writer is None or operations.writer is not writer or operations.owned_registry.writer is not writer:
        raise ContractError('completion final state writer differs from original prepared witness')
    path=operations.ownership.directory/'run.json'
    # Admit final serialization, hashing and retained-byte overlap BEFORE copies.
    with candidate.parent_budget.reserve(35*(json_bound(state)+1)+65536):
        if (tuple(state[key] for key in _FACTS)!=candidate._initial
                or digest(state['identity'])!=candidate._identity_digest
                or state['generation_sha256']!=candidate._generation_sha256):
            raise ContractError('completion differs from actual committed final identity/counters')
        raw=(canonical(state)+'\n').encode()
        state_sha=hashlib.sha256(raw).hexdigest()
        receipt=writer._completed.get(path)
        from gf_training.hpo_owned_lineage import OwnedControlWriteReceipt
        if type(receipt) is not OwnedControlWriteReceipt: raise ContractError('actual completed final write receipt missing')
        receipt.require(writer,path,state_sha)
        from gf_training.hpo_owned_snapshot import _backings
        candidate._original_backings=_backings(candidate.inference)
        candidate._state=raw[:-1]
        candidate._state_sha256=state_sha
        candidate._completed=True
        candidate._initial=()
        candidate._issuer=lambda:None; candidate._writer=lambda:None
    return candidate


def _detach_external_policy(imported: Any) -> OwnedExternalFinalPolicy:
    from gf_training.hpo_owned_restore import OwnedColdImport
    if type(imported) is not OwnedColdImport: raise ContractError('actual cold import required')
    imported.require()
    if imported.state['status']!='complete': raise ContractError('external evaluation requires exact final-D parent')
    snapshot=imported.snapshots[imported.state['updates']]
    if snapshot.resume is not None: raise ContractError('external evaluation must import inference alone')
    facts=snapshot._control.value
    charge=32*(json_bound(facts['actor_metadata'])+json_bound(facts['inference_manifest'])
        +json_bound(imported.records)+json_bound(imported.state))+65536
    reservation=imported.parent_budget.reserve(charge)
    owner=OwnedExternalFinalPolicy.__new__(OwnedExternalFinalPolicy)
    owner._reservation=reservation
    owner._budget=imported.budget; owner.parent_budget=imported.parent_budget
    banks=[]
    try:
        owner._actor_metadata=canonical(snapshot.actor_metadata).encode()
        owner._manifest=canonical(snapshot.inference_manifest).encode()
        owner._records=canonical(management_records(imported.records)).encode()
        owner._generations_sha256=digest(imported.records)
        owner._generation_sha256=imported.state['generation_sha256']
        owner._state=canonical(imported.state).encode()
        owner._state_sha256=imported.scan.verified_digest(imported.scan.root/'run.json')
        owner._files=canonical({name:sha for name,sha in facts['files'].items()
            if name=='inference.json' or name.startswith('weights-')}).encode()
        owner.generation,owner.identity=snapshot.generation,snapshot.identity
        owner._origin_facts=(owner.generation,owner.identity)
        for bank in snapshot.inference.banks: banks.append(bank.retain(owner._budget))
        owner.inference=ReadonlyRamBankSet(tuple(banks))
        from gf_training.hpo_owned_snapshot import _backings
        owner._original_backings=_backings(owner.inference)
        owner._closed=False; owner._completed=True; owner._actor_capability=None
        owner._initial=(); owner._issuer=lambda:None; owner._writer=lambda:None
        return owner
    except BaseException:
        for bank in banks: bank.close()
        reservation.close()
        raise


def retain_for_evaluation(completion: OwnedRunCompletion, *, budget: Any,
                           parent_budget: ArtifactPhaseAdmission) -> OwnedEvaluationPolicy:
    """Actual completion -> independent new data scope; no training deadline/job revival."""
    if type(completion) is not OwnedRunCompletion: raise ContractError('actual private completion required')
    completion.require()
    if (type(parent_budget) is not ArtifactPhaseAdmission or parent_budget.source is not budget
            or budget is completion._budget):
        raise ContractError('evaluation requires its OWN actual admitted Source/parent scope')
    charge=32*(len(completion._actor_metadata)+len(completion._manifest)+len(completion._records)+len(completion._state))+65536
    reservation=parent_budget.reserve(charge)
    owner=OwnedEvaluationPolicy.__new__(OwnedEvaluationPolicy)
    owner._reservation=reservation
    owner._budget,owner.parent_budget=budget,parent_budget
    banks=[]
    try:
        # Immutable bytes transfer into the new charged lifetime; no old owner,
        # optimizer, resume state or closed training job is retained.
        for name in ('_actor_metadata','_manifest','_records','_state','_state_sha256','_files',
                     '_generations_sha256','_generation_sha256','generation','identity','_origin_facts'):
            setattr(owner,name,getattr(completion,name))
        for bank in completion.inference.banks: banks.append(bank.retain(budget))
        owner.inference=ReadonlyRamBankSet(tuple(banks))
        from gf_training.hpo_owned_snapshot import _backings
        owner._original_backings=_backings(owner.inference)
        owner._closed=False; owner._completed=True; owner._actor_capability=None
        owner._initial=(); owner._issuer=lambda:None; owner._writer=lambda:None
        return owner
    except BaseException:
        for bank in banks: bank.close()
        reservation.close()
        raise


def completion_artifacts(completion: Any, binding: Any, label: str) -> Any:
    from pathlib import Path

    from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
    from gf_training.structured_lineage import _stat_identity
    completion.require()
    # Provider construction and these diagnostic projections share the new data scope.
    with completion.parent_budget.reserve(32*len(completion._manifest)+32*len(completion._files)+65536):
        entries=[{'path':name,'sha256':sha,'identity':list(_stat_identity((Path(binding.policy_options['checkpoint'])/name).lstat()))}
            for name,sha in completion.files.items()]
        return OwnedGameArtifacts({label:(completion.actor_capability,binding,entries,completion.inference_manifest)},
            budget=completion._budget,parent_budget=completion.parent_budget)
