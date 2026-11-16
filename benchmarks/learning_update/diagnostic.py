"""Explicit train-only NumPy adapter and pinned historical start-state import.

Neither object is an OwnedNumericBatch, Source, capability, or ordinary restore.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.learning_update.common import PRODUCER, check_files, read_json, sha256


class DiagnosticRollout:
    def __init__(self, rollout: Any) -> None:
        # Immutable numerical arrays only: no records/canonical histories or
        # Source capabilities are retained in this timed adapter.
        from gf_common.hpo_owned_tokens import CANDIDATE_ARRAY_FIELDS, FRAME_ARRAY_FIELDS
        from gf_common.structured_tokens import CandidateTokens, FrameTokens
        self.frames=[]
        self.tables=[]
        self.locations=tuple(rollout.locations)
        self.vectors=tuple(rollout.vector(i) for i in range(len(rollout)))
        self.resident_bytes=0
        def converted(tokens: Any,fields: Any,constructor: Any) -> Any:
            values=asdict(tokens)
            for name in fields:
                old=getattr(tokens,name)
                dtype=np.bool_ if name=='card_mask' else np.int64 if name in (
                    'state_categories','card_categories','opponent_learned','interval_categories',
                    'kinds','targets','offsets','slots') else np.float32
                # Empty nested interval arrays retain their declared width from
                # original tokenizer metadata; raw tensor core never consumes
                # them when interval length is zero.
                array=np.asarray(old,dtype=dtype)
                array.flags.writeable=False
                values[name]=array
                self.resident_bytes+=array.nbytes
            return constructor(**values)
        for episode in rollout.episodes:
            self.frames.append(tuple(converted(frame,FRAME_ARRAY_FIELDS,FrameTokens) for frame in episode.frames))
            self.tables.append(tuple(converted(tokens,CANDIDATE_ARRAY_FIELDS,CandidateTokens) for tokens in episode.tables))
        self.used_decisions=rollout.used_decisions
        self.collected_decisions=rollout.collected_decisions
        self.resident_bytes+=len(self.vectors)*128+sum(len(rows)*512 for rows in self.frames)

    def __len__(self) -> int:
        return len(self.locations)

    def endpoint(self,index: int) -> Any:
        episode,position=self.locations[index]
        return self.frames[episode][:position+1],self.tables[episode][position]

    def vector(self,index: int) -> Any:
        return self.vectors[index]

    @staticmethod
    def forward(network: Any,frames: Any,candidates: Any,*,suffix_length: int) -> Any:
        # The original complete readers validated every token before conversion.
        # This bypasses public tuple validators only, sharing the SAME original
        # model tensor construction, admission, frame and actor mathematics.
        return network._forward_calculate(frames,candidates,suffix_length=suffix_length,
            dense=False,chunk_size=None,mask=None,
            encode_history=lambda rows,suffix: network._encode_history_calculate(rows,suffix,network._encode_frame_calculate))


def load_input(manifest: Any,kind: str,batch: int) -> tuple[Any,Any,Any]:
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_models.hpo_config import model_config_v4
    from gf_models.structured_config import InferenceBindingV2, resource_estimate
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_rollout import read_hpo_rollout
    row=manifest['models'][kind]
    check_files(row['files'])
    source=row['batches'][batch]
    model=model_config_v4(row['model'],layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    binding=InferenceBindingV2(**row['binding'])
    actions=action_config_v3(row['actions'])
    training=hpo_training_config(row['training'])
    original=read_hpo_rollout(Path(source['numeric']).parent,source['numeric_sha256'],model,training,
        actions=actions,rules=card_instance_rules(),expected_selection=source['selection'],
        expected_snapshot_identity=source['snapshot_identity'],numeric_budget_bytes=512*1024**2,
        max_owned_bytes=2*1024**3)
    adapter=DiagnosticRollout(original)
    learner=HpoLearner(model,training,17,binding,row['resume_identity'],actions=actions,rules=card_instance_rules())
    learner._training_reservation_bytes=resource_estimate(model,binding,
        suffix_decisions=training.tbptt_length,layout=CORRECTED_CARD_INSTANCE_LAYOUT)['total_bytes']
    return learner,adapter,row['starts'][batch]


def restore_start(learner: Any,start: Any) -> None:
    from gf_models.sharded_arrays import read_bundle
    from gf_models.structured_artifacts import (
        _apply_resume_state,
        _validate_diagnostic_resume_manifest,
        _validate_resume_arrays,
        inspect_game_inference_v4,
    )
    from gf_models.torch_runtime import restore_weights
    from gf_training.hpo_artifacts import _recipe_options
    directory=Path(start['directory'])
    if sha256(directory/'resume.json')!=start['resume_sha256']:
        raise ValueError('pinned historical resume changed')
    manifest=read_json(directory/'resume.json')
    if (manifest['producer']!=start['recorded_producer']
            or manifest['producer']['source_digest']!=PRODUCER
            or manifest['optimizer_steps']!=start['steps']):
        raise ValueError('explicit historical producer/start differs')
    inference,_=inspect_game_inference_v4(directory,model=learner.model,binding=learner.binding,
        expected_digest=start['inference_sha256'],action_config_hash=learner.action_config_hash,
        action_contract=learner.action_contract,rules=learner._context.rules)
    if inference['producer']['source_digest']!=PRODUCER:
        raise ValueError('pinned inference producer differs')
    _validate_diagnostic_resume_manifest(manifest,learner.model,learner.binding,learner.resume_identity,
        recorded_source_digest=PRODUCER,context=learner._context,
        expected_inference_digest=start['inference_sha256'],**_recipe_options(learner.training))
    arrays=read_bundle(directory,manifest['state'],max_decoded_bytes=learner.model.resources.max_owned_bytes)
    _validate_resume_arrays(manifest,arrays,learner.model,learner.binding,context=learner._context)
    weights=read_bundle(directory,inference['weights'],max_decoded_bytes=learner.model.resources.max_owned_bytes)
    restore_weights(learner.network,weights)
    learner.optimizer.zero_grad(set_to_none=True)
    learner.steps=_apply_resume_state(learner.network,learner.optimizer,learner.shuffle,
        learner.model,learner.binding,manifest,arrays,context=learner._context)
    learner.synchronize()
