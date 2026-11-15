"""Actual fresh collection capture and writer completion, never public flag authority."""
from __future__ import annotations

import weakref
from dataclasses import asdict
from typing import Any

from gf_common.artifacts import json_bound
from gf_common.domain import ContractError, Json, canonical, clone
from gf_common.hpo import selected_locations
from gf_common.schedule import serialize_schedule
from gf_training.hpo_collection import selection_for_counts
from gf_training.hpo_owned_capture import _new_capture

_TOKEN=object()


class OwnedFinalizedCollection:
    def __init__(self, token: object, attachment: Any, publication: Any, result: Any, batch: Any) -> None:
        if token is not _TOKEN:
            raise ContractError('actual producer publication completion required')
        self._attachment,self._publication,self._result=attachment,publication,result
        self.batch,self.plan=batch,attachment.plan
        self.directory,self.report=result.directory,clone(result.report)
        self.manifest=clone(attachment.manifest)
        self.manifest_sha256=attachment.manifest_sha256
        self.locator,self.envelope_sha256,self.storage_metrics=result.locator,result.envelope_sha256,clone(result.storage_metrics)

    def __reduce__(self) -> Any:
        raise TypeError('owned collection completion cannot be persisted')

    def require(self, registry: Any, plan: Any) -> None:
        value=self._attachment
        if (registry is not value.registry or plan is not value.plan or value.result is not self
                or self._publication is not value.publication or not self._publication.committed
                or not self._publication.cleanup_complete or self._publication.result is not self._result):
            raise ContractError('owned archive differs from SAME capture/plan/completed writer transaction')
        self.batch.require_owner(registry,plan=plan,snapshot_owner=value.snapshot)

    def selection(self) -> Json:
        state=self._attachment.registry.state
        counts=tuple(len(episode.learner.decisions) for episode in self.batch.episodes)
        return selection_for_counts(counts,learning_seed=state['seed'],batch=self.batch.generation,
            decision_budget=state['effective_config']['hpo_spec']['decision_budget'],used_before=state['used_decisions'])

    def proof(self, numeric: Any) -> Json:
        from gf_common.domain import digest
        selection=numeric.metadata['selection']
        return {'schema_version':5,'origin':'producer-owned-v1','encoder':'canonical-python',
            'manifest_sha256':self.manifest_sha256,'games':len(self.batch.episodes),
            'generation':self.batch.generation,'batch_id':self.plan.linkage['batch_id'],
            'selection':selection,'selection_sha256':digest(selection),'replay_reconciled':False,
            'snapshot_identity':self._attachment.snapshot.identity,
            'expected_optimizer_step_delta':numeric.training.epochs*((numeric.used_decisions+numeric.training.minibatch_size-1)//numeric.training.minibatch_size),
            'context_costs':{'context_decisions':numeric.training.epochs*sum(position+1 for _,position in selected_locations(selection)),
                'suffix_decisions':numeric.training.epochs*sum(min(position+1,numeric.training.tbptt_length) for _,position in selected_locations(selection))}}


class _OwnedCollection:
    def __init__(self, registry: Any, plan: Any, learner_label: str) -> None:
        registry.require()
        snapshot=registry.current
        if snapshot is None or plan.linkage['collection_generation']!=snapshot.generation:
            raise ContractError('owned collection requires actual committed learner generation')
        if plan.bindings.get(learner_label)!=registry.binding(snapshot.generation)[1]:
            raise ContractError('owned learner policy differs from committed snapshot')
        metadata_charge=65536+4096*sum(len(group.episodes) for group in plan.groups)+8*json_bound(plan.linkage)
        metadata_charge+=sum(32*len(label)+8*json_bound(binding.policy_options)+4096 for label,binding in plan.bindings.items())
        reservation=registry.parent_budget.reserve(metadata_charge)
        weakref.finalize(self,reservation.close)
        self.registry,self.plan,self.snapshot=registry,plan,snapshot
        self.capture=_new_capture(registry,run_id=registry.state['run_id'],plan=plan,generation=snapshot.generation,
            snapshot_owner=snapshot,learner_seats=plan.linkage['learner_seats'],learner_policy_version=learner_label,
            budget=registry.budget)
        self.collection: Any=None
        self.writer: Any=None
        self.publication: Any=None
        self.manifest: Any=None
        self.manifest_sha256: str | None=None
        self._result_ref: Any=None
        self._plan_bytes=canonical({'groups':serialize_schedule(plan.groups,4),
            'bindings':{name:asdict(value) for name,value in plan.bindings.items()},'linkage':plan.linkage})

    @property
    def result(self) -> Any:
        return None if self._result_ref is None else self._result_ref()

    def __reduce__(self) -> Any:
        raise TypeError('owned capture/writer attachment cannot be persisted')

    def require(self, registry: Any, plan: Any) -> None:
        registry.require()
        if (registry is not self.registry or plan is not self.plan or registry.capture is not self
                or registry.current is not self.snapshot or registry.plan is not plan
                or self._plan_bytes!=canonical({'groups':serialize_schedule(plan.groups,4),
                    'bindings':{name:asdict(value) for name,value in plan.bindings.items()},'linkage':plan.linkage})):
            raise ContractError('owned collection attachment/actual plan or snapshot changed')

    def retire(self) -> None:
        self.capture.close()
        if self.collection is not None:
            if not self.collection._closed or self.writer is not None and self.writer.returncode is None:
                raise ContractError('owned archive physical cleanup remains unresolved')
            if self.collection._producer is self: self.collection._producer=None
        if self.writer is not None:
            if self.writer._producer is self: self.writer._producer=None
            if getattr(self.writer,'_producer_sealing',None) is self: self.writer._producer_sealing=None

    def bind_collection(self, collection: Any) -> None:
        from gf_training.group_pack_collection import GroupPackCollection
        self.require(self.registry,self.plan)
        if type(collection) is not GroupPackCollection or self.collection is not None or collection.writer is not None:
            raise ContractError('owned capture requires SAME fresh packed collector')
        self.collection=collection
        collection._producer=self

    def bind_writer(self, writer: Any) -> None:
        from gf_training.group_pack_writer import GroupPackWriter
        if (type(writer) is not GroupPackWriter or self.collection is None or self.writer is not None
                or writer.directory!=self.collection.directory or writer.run_id!=self.collection.run_id):
            raise ContractError('owned archive requires actual original writer')
        self.writer=writer
        writer._producer=self

    def prepare_publication(self, publication: Any, manifest: Json) -> None:
        from gf_training.group_pack_publication import GroupPackCollectionPublication
        self.require(self.registry,self.plan)
        if (type(publication) is not GroupPackCollectionPublication or publication.collection is not self.collection
                or self.publication is not None or self.writer is None
                or self.writer.returncode!=0 or not self.writer._sealed
                or manifest.get('schema_version')!=8):
            raise ContractError('owned publication requires actual completed original writer')
        self.publication=publication
        charge=64*json_bound(manifest)+65536
        reservation=self.registry.parent_budget.reserve(charge)
        weakref.finalize(self,reservation.close)
        self.manifest=clone(manifest)
        assert publication.pending is not None
        rows=[row for row in publication.pending.descriptor['controls'] if row['name']=='manifest.json']
        if len(rows)!=1: raise ContractError('owned archive writer omitted actual manifest control')
        self.manifest_sha256=rows[0]['sha256']

    def complete(self, publication: Any, result: Any) -> OwnedFinalizedCollection:
        self.require(self.registry,self.plan)
        if (publication is not self.publication or not publication.committed or not publication.cleanup_complete
                or publication.result is not result or self.result is not None):
            raise ContractError('owned archive has no actual complete publication/cleanup')
        batch=self.capture.seal(self.registry,plan=self.plan,snapshot_owner=self.snapshot)
        completed=OwnedFinalizedCollection(_TOKEN,self,publication,result,batch)
        self._result_ref=weakref.ref(completed)
        return completed
