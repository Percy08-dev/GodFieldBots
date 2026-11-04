"""CPU native Tree2 inference over deterministic admitted participant chunks."""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.action_contract import validate_game_references
from gf_common.domain import ContractError, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBankView, Selection
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.structured_binding import check_input_binding
from gf_models.structured_tree_artifacts import (
    LoadedTree,
    LoadedTreeV3,
    _bind_model_v3,
    _bind_origin,
    _check,
    _validate_booster,
    load_artifact,
    load_artifact_v3,
)
from gf_models.structured_tree_config import policy_label, resource_estimate
from gf_models.structured_tree_encoder import PreparedTreeFrame, PreparedTreeFrameV3, _owned
from gf_models.structured_tree_pairing import TREE2_RECORDS, TREE3_RECORDS, _TreeRecords


def _prepare_policy(tree: Any, frame: DecisionFrame, bank_view: EventBankView, table: CandidateTable,
                    context: _TreeRecords) -> tuple[Any, Any, float]:
    model = tree.model
    if frame.layout is not context.features.layout or table.layout is not context.features.layout:
        raise ContractError('Tree policy raw participant layout mismatch')
    integer(len(table.kinds), 1, model.inference.max_candidates)
    integer(frame.boundary.end, 0, 262144)
    interval = frame.boundary.end-frame.interval_start
    integer(interval, 0, 4096)
    view = frame.bank_view(bank_view)
    raw_owned = _owned((frame, bank_view, table, tree.origin))
    event_bytes = sum(len(value) for value in view._events[max(0, frame.boundary.end-model.encoder.aggregate_window):])
    event_bytes += sum(len(value) for value in view._events[frame.interval_start:])
    resource_estimate(model, model_bytes=tree.descriptor.model_bytes,
        extra_bytes=raw_owned+64*len(frame.payload)+32*event_bytes+2048*interval+65536)
    started = time.perf_counter()
    tokens = tokenize_frame(frame, view, aggregate_window=model.encoder.aggregate_window, layout=context.features.layout)
    candidates = tokenize_table(table, frame, model.encoder.max_positions,
        aggregate_window=model.encoder.aggregate_window, layout=context.features.layout)
    resource_estimate(model, model_bytes=tree.descriptor.model_bytes,
        extra_bytes=_owned((frame, bank_view, table, tree.origin, tokens, candidates)))
    prepared = (PreparedTreeFrameV3 if context.modern else PreparedTreeFrame).create(tokens, model,
        model_bytes=tree.descriptor.model_bytes, extra_owned_bytes=raw_owned)
    return prepared, candidates, (time.perf_counter()-started)*1000


def _score_chunks(tree: Any, prepared: Any, candidates: Any) -> Iterator[tuple[int, Any]]:
    for start, rows in prepared.iter_chunks(candidates):
        scores = np.asarray(tree.booster.predict(rows, num_threads=tree.model.inference.num_threads), dtype=np.float64)
        if scores.shape != (len(rows),) or not np.isfinite(scores).all():
            raise ContractError('Tree scores must be finite scalar native predictions')
        yield start, scores


def _select(tree: Any, prepared: Any, candidates: Any) -> int:
    best, selected = -float('inf'), -1
    for start, scores in _score_chunks(tree, prepared, candidates):
        local = int(np.argmax(scores))
        if float(scores[local]) > best:
            best, selected = float(scores[local]), start+local
    if selected < 0:
        raise ContractError('Tree cannot select an empty candidate table')
    return selected


class StructuredTreePolicy:
    def __init__(self, tree: LoadedTree, manifest_digest: str, *, deadline:float=float('inf')) -> None:
        _check(deadline)
        if type(tree) is not LoadedTree:
            raise ContractError('actual loaded Tree2 wrapper required')
        _bind_origin(tree.booster, tree.origin)
        pair=None
        if tree.origin.fit_origin=='trained':
            if tree.training_binding is None or tree.origin.provenance is None:raise ContractError('trained Tree2 policy requires complete explicit paired evidence')
            from gf_models.structured_tree_pairing import inspect_training
            pair=inspect_training(Path(tree.training_binding.directory),expected_binding=tree.training_binding,max_owned_bytes=tree.model.resources.max_owned_bytes,deadline=deadline)
            if pair.inference['tree_descriptor']!=asdict(tree.descriptor) or pair.inference['provenance']!=tree.origin.provenance.to_dict():
                raise ContractError('actual trained Tree2 wrapper differs from paired evidence')
        elif tree.training_binding is not None:raise ContractError('toy Tree2 cannot carry trained evidence')
        _, descriptor = _validate_booster(tree.booster, tree.model,deadline=deadline)
        if pair is not None:pair.finalize()
        if descriptor != tree.descriptor:
            raise ContractError('Tree2 runtime descriptor differs from actual native model')
        self.tree, self.model = tree, tree.model
        self.version = policy_label(tree.model, tree.descriptor, manifest_digest, tree.training_binding)
        self.reserved_bytes = resource_estimate(tree.model, model_bytes=descriptor.model_bytes)['total_bytes']
        self.last_timing: dict[str, float] = {}
        _check(deadline)

    @classmethod
    def from_artifact(cls, directory: Path, **options: Any) -> StructuredTreePolicy:
        return cls(load_artifact(directory, **options), options['expected_digest'],deadline=options.get('deadline',float('inf')))

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed, 0, 2**64-1)
        self.last_timing = {}

    def select_frame(self, frame: DecisionFrame, bank_view: EventBankView, table: CandidateTable) -> Selection:
        origin = self.tree.origin
        _bind_origin(self.tree.booster, origin)
        check_input_binding({'action_contract': asdict(table.action_contract), 'purpose': table.purpose,
                             'synthetic_capacity': table.synthetic_capacity}, action_contract=origin.action_contract,
                             purpose=origin.purpose, synthetic_capacity=origin.synthetic_capacity)
        if table.synthetic_capacity:
            raise ContractError('Tree2 game factory/policy rejects synthetic-capacity tables')
        validate_game_references(table.slots[a:b] for a, b in zip(table.offsets[:-1], table.offsets[1:], strict=True))
        table.validate(frame)
        self.last_timing = {}
        prepared, candidates, token_ms = _prepare_policy(self.tree, frame, bank_view, table, TREE2_RECORDS)
        selected = _select(self.tree, prepared, candidates)
        # Protocol2 already measures the entire policy call. Keep its closed
        # host timing vocabulary; no unreviewed Tree-only wire field is added.
        self.last_timing = {'tokenization_ms': token_ms}
        return Selection(table.table_hash, selected, self.version)


class ComponentStructuredTreePolicyV3:
    """Explicit component-only scoring/selection; no game registry or worker dispatch."""
    def __init__(self,tree:LoadedTreeV3,manifest_digest:str,*,deadline:float=float('inf'))->None:
        from gf_common.domain import canonical, digest
        from gf_models.structured_sealed import checksum
        from gf_models.structured_tree_pairing import inspect_training_v3
        _check(deadline);checksum(manifest_digest)
        if type(tree) is not LoadedTreeV3:
            raise ContractError('closed actual Tree3 component wrapper required')
        _bind_origin(tree.booster,tree.origin)
        _bind_model_v3(tree.booster,tree.model,tree.origin)
        TREE3_RECORDS.source(tree.origin.fields(),toy=tree.origin.fit_origin=='toy')
        pair=None
        if tree.origin.fit_origin=='trained':
            if tree.training_binding is None or tree.origin.provenance is None:
                raise ContractError('trained Tree3 component policy requires exact healthy pair')
            pair=inspect_training_v3(Path(tree.training_binding.directory),expected_binding=tree.training_binding,
                max_owned_bytes=tree.model.resources.max_owned_bytes,deadline=deadline)
            if (pair.inference['tree_descriptor']!=asdict(tree.descriptor)
                    or canonical(pair.inference['provenance'])!=canonical(tree.origin.provenance.to_dict())
                    or canonical(pair.inference['model'])!=canonical(asdict(tree.model.architecture()))
                    or pair.inference['architecture_identity']!=TREE3_RECORDS.identity(tree.model.architecture())
                    or canonical(pair.inference['feature_layout'])!=canonical(TREE3_RECORDS.layout(tree.model.encoder))):
                raise ContractError('actual trained Tree3 wrapper differs from paired evidence')
        elif tree.training_binding is not None:
            raise ContractError('Tree3 toy cannot carry a trained binding')
        _,descriptor=_validate_booster(tree.booster,tree.model,deadline=deadline)
        if descriptor!=tree.descriptor:
            raise ContractError('Tree3 actual native descriptor differs')
        if pair is not None:pair.finalize()
        self.tree,self.model=tree,tree.model
        execution={'inference':asdict(tree.model.inference),'resources':asdict(tree.model.resources),
            'tree_descriptor':asdict(tree.descriptor),'origin':tree.origin.fields(),
            'training_binding':None if tree.training_binding is None else asdict(tree.training_binding)}
        self.version='lightgbm-tree-v3-'+manifest_digest+'-'+digest(execution)
        self.reserved_bytes=resource_estimate(tree.model,model_bytes=descriptor.model_bytes)['total_bytes']
        self.last_timing:dict[str,float]={}
        _check(deadline)

    @classmethod
    def from_artifact(cls,directory:Path,**options:Any)->ComponentStructuredTreePolicyV3:
        return cls(load_artifact_v3(directory,**options),options['expected_digest'],deadline=options.get('deadline',float('inf')))

    def reset(self,sampling_seed:int)->None:
        integer(sampling_seed,0,2**64-1)
        self.last_timing={}

    def _prepare(self,frame:DecisionFrame,bank_view:EventBankView,table:CandidateTable)->tuple[Any,Any]:
        from gf_common.domain import canonical
        _bind_origin(self.tree.booster,self.tree.origin)
        _bind_model_v3(self.tree.booster,self.tree.model,self.tree.origin)
        _bind_model_v3(self.tree.booster,self.model,self.tree.origin)
        expected=TREE3_RECORDS.binding({key:self.tree.origin.fields()[key] for key in
            ('action_contract','rule_signature','purpose','synthetic_capacity')})
        actual={'action_contract':asdict(table.action_contract),'rule_signature':expected['rule_signature'],
                'purpose':table.purpose,'synthetic_capacity':table.synthetic_capacity}
        if canonical(actual)!=canonical(expected) or table.synthetic_capacity:
            raise ContractError('Tree3 component policy rejects foreign or game tables')
        validate_game_references(table.slots[a:b] for a,b in zip(table.offsets[:-1],table.offsets[1:],strict=True))
        table.validate(frame)
        self.last_timing={}
        prepared,candidates,token_ms=_prepare_policy(self.tree,frame,bank_view,table,TREE3_RECORDS)
        self.last_timing={'tokenization_ms':token_ms}
        return prepared,candidates

    def scores(self,frame:DecisionFrame,bank_view:EventBankView,table:CandidateTable)->Any:
        prepared,candidates=self._prepare(frame,bank_view,table)
        count=len(candidates.kinds)
        chunks=(count+self.model.inference.candidate_chunk-1)//self.model.inference.candidate_chunk
        # Retain chunk scores and the concatenated result together; source/raw
        # owners and model storage remain admitted until prediction returns.
        resource_estimate(self.model,model_bytes=self.tree.descriptor.model_bytes,
            extra_bytes=prepared.resident_bytes+_owned(candidates)+16*count+1024*chunks+65536)
        return np.concatenate([scores for _,scores in _score_chunks(self.tree,prepared,candidates)])

    def select_frame(self,frame:DecisionFrame,bank_view:EventBankView,table:CandidateTable)->Selection:
        prepared,candidates=self._prepare(frame,bank_view,table)
        return Selection(table.table_hash,_select(self.tree,prepared,candidates),self.version)
