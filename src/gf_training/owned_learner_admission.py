"""Private normal-owned CUDA launch context and original-owner binding.

The serialized context is checked data. Only the issuing invocation and the
same actual pre-spawn learner lease authorize its use by a parent creator.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed
from gf_common.hpo_resident import resident_runtime
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT

_TOKEN = object()
_OPERATING = {'managed_ram_bytes','managed_cuda_bytes','game_concurrency',
    'max_participant_workers','parent_budget_bytes','source_budget_bytes','numeric_budget_bytes',
    'save_budget_bytes','collector_budget_bytes'}


def owned_launch_context(model: Any, training: Any, binding: Any, operating: Json, runtime: Json) -> Json:
    from gf_models.structured_runtime import hpo_resident_training_admission
    bounds = {key:operating[key] for key in sorted(_OPERATING)}
    envelope = hpo_resident_training_admission(model,binding,training.tbptt_length,bounds,runtime,
        layout=LAYOUT,owned_cuda=True)
    if 'owned_learner_admission' not in envelope:
        raise ContractError('normal owned CUDA launch cannot adopt experimental/CPU admission')
    return {'schema_version':1,'kind':'owned-cuda-learner-launch-v1','operating':bounds,
        'runtime':resident_runtime(runtime),'admission':envelope['owned_learner_admission']}


def checked_owned_launch_context(value: Json, model: Any, training: Any, binding: Any) -> Json:
    raw = closed(value,{'schema_version','kind','operating','runtime','admission'})
    if type(raw['schema_version']) is not int or raw['schema_version']!=1 or raw['kind']!='owned-cuda-learner-launch-v1':
        raise ContractError('closed owned CUDA launch version required')
    bounds = closed(raw['operating'],_OPERATING)
    expected = owned_launch_context(model,training,binding,bounds,raw['runtime'])
    if canonical(raw)!=canonical(expected):
        raise ContractError('owned CUDA launch differs from checked model/operating bounds')
    return expected


def normal_owned_launch_context(config: Any, training_execution: Any) -> Json | None:
    if (training_execution is not None or config.hpo_spec.get('schema_version')!=4
            or config.training.device!='cuda:0' or config.model.inference.device!='cuda:0'):
        return None
    from gf_models.structured_runtime import hpo_resident_training_admission
    from gf_training.hpo_config import hpo_training_config
    model, binding = config.model.architecture(layout=LAYOUT), config.model.inference
    training = hpo_training_config(config.hpo_spec['training_recipe'])
    envelope = hpo_resident_training_admission(model,binding,training.tbptt_length,
        config.hpo_spec['operating'],config.hpo_spec['runtime'],layout=LAYOUT,owned_cuda=True)
    if 'experimental_storage_bytes' in envelope or 'owned_learner_admission' not in envelope:
        return None
    return {'schema_version':1,'kind':'owned-cuda-learner-launch-v1',
        'operating':{key:config.hpo_spec['operating'][key] for key in sorted(_OPERATING)},
        'runtime':resident_runtime(config.hpo_spec['runtime']),'admission':envelope['owned_learner_admission']}


class _OwnedLearnerLaunch:
    def __init__(self, token: object, owner: Any, context: Json) -> None:
        if token is not _TOKEN:
            raise ContractError('original owned training invocation required')
        self.owner,self.invocation,self.payload = owner,owner._invocation,canonical(context)
        self.process: Any = None

    def __reduce__(self) -> Any:
        raise TypeError('owned learner launch authority cannot be copied or persisted')

    def bind(self, process: Any) -> int:
        from gf_training.hpo_owned_process import HpoOwnedLearningProcess
        self.owner.job._require_open()
        config = self.owner._owned_admission_config
        if (type(process) is not HpoOwnedLearningProcess or self.process is not None
                or self.owner._startup_child is not process or process.ledger is not self.owner.job.ledger
                or process._training_execution is not None
                or canonical(asdict(process.model))!=canonical(asdict(config.model.architecture(layout=LAYOUT)))
                or canonical(asdict(process.binding))!=canonical(asdict(config.model.inference))
                or canonical(asdict(process.hpo_training))!=canonical(config.hpo_spec['training_recipe'])):
            raise ContractError('owned learner launch differs from SAME original creator/config')
        context = checked_owned_launch_context(json.loads(self.payload),process.model,process.hpo_training,process.binding)
        if context['admission']['global_cuda_bytes']!=process.ledger.global_cuda_bytes:
            raise ContractError('owned learner launch differs from actual global CUDA ledger')
        self.process = process
        return context['admission']['device_bytes']

    def arguments(self, process: Any) -> tuple[str,...]:
        if (process is not self.process or self.owner._startup_child is not process
                or self.owner._invocation is not self.invocation):
            raise ContractError('owned learner launch lost original invocation/process')
        self.owner.job._require_open()
        process._require_active_creation_lease(process.lease)
        context = checked_owned_launch_context(json.loads(self.payload),process.model,process.hpo_training,process.binding)
        admission = context['admission']
        if process.resident_bytes!=admission['device_bytes'] or process.lease.resident_bytes!=admission['device_bytes']:
            raise ContractError('owned learner device allowance differs from SAME actual lease')
        from gf_training.hpo_learning_process import checked_execution_cap
        logical = checked_execution_cap(process.model,process.hpo_training,process.binding,admission['logical_bytes'])
        return ('--model4-execution-cap',str(logical),'--owned-learner-admission',self.payload)


def issue_owned_learner_launch(owner: Any, config: Any, state: Json) -> _OwnedLearnerLaunch | None:
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    if type(owner) is not _OwnedHpoOperations or owner._invocation is None:
        raise ContractError('actual owned training invocation required for learner launch')
    owner._invocation.require_original(owner)
    if owner._invocation.phase!='CONSTRUCTION_DISPATCHED' or owner._owned_admission_config is not config:
        raise ContractError('owned learner launch requires original construction dispatch/config')
    if owner._owned_launch_context is None:
        return None
    context = json.loads(owner._owned_launch_context)
    if canonical(state['identity'].get('owned_learner_admission'))!=canonical(context['admission']):
        raise ContractError('owned learner launch differs from persisted admission identity')
    return _OwnedLearnerLaunch(_TOKEN,owner,context)
