"""Wire6 adapter sharing the actual numerical child and reservation owner."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, integer
from gf_training.structured_learning_process import GameLearningProcess, StructuredLearningProcess


def parse_execution_cap(arguments: tuple[str,...]) -> int | None:
    if not arguments:
        return None
    if len(arguments)!=2 or arguments[0]!='--model4-execution-cap':
        raise ContractError('closed labeled Model4 execution cap required')
    try:
        value=int(arguments[1])
    except ValueError as error:
        raise ContractError('positive canonical Model4 execution cap required') from error
    integer(value,1,2**63-1)
    if str(value)!=arguments[1]:
        raise ContractError('canonical Model4 execution cap required')
    return value


def checked_execution_cap(model: Any,training: Any,binding: Any,cap: int | None) -> int:
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_models.structured_config import resource_estimate
    if cap is None:
        raise ContractError('Model4 worker requires original execution cap')
    integer(cap,1,min(model.resources.max_owned_bytes,binding.managed_budget_bytes))
    configured=resource_estimate(model,binding,suffix_decisions=training.tbptt_length,
        layout=CORRECTED_CARD_INSTANCE_LAYOUT)['total_bytes']
    if cap!=configured:
        raise ContractError('Model4 process cap differs from checked configured envelope')
    return cap


class HpoLearningProcess(GameLearningProcess):
    def __init__(self, directory: Path, max_log_bytes: int, model: Any, training: Any,
                 binding: Any, action_config_hash: str, resume_identity: str, **options: Any) -> None:
        from gf_common.hpo_progress import DiagnosticParser
        callback = options.pop('progress', None)
        enabled = options.pop('diagnostics', False)
        self.diagnostic_invocation = uuid.uuid4().hex if enabled else None
        self.diagnostic_parser = DiagnosticParser(self.diagnostic_invocation, callback) if self.diagnostic_invocation is not None else None
        actions = options.pop('actions')
        phase_limits=options.pop('phase_limits',None)
        StructuredLearningProcess.__init__(self, directory, max_log_bytes, model, binding,
            action_config_hash, resume_identity, action_contract=actions.action_contract,
            _actions=actions, _hpo_training=training, _hpo_phase_limits=phase_limits, **options)
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
        from gf_models.structured_config import resource_estimate
        self._configured_execution_cap=resource_estimate(model,binding,
            suffix_decisions=training.tbptt_length,layout=CORRECTED_CARD_INSTANCE_LAYOUT)['total_bytes']
        self.online_owner: Any = None

    def _launch_arguments(self) -> tuple[str,...]:
        lease=self.lease
        if (lease is None or lease.ledger is not self.ledger or self.ledger.learner is not lease
                or lease._cleanup_owner is not self or lease.released or not lease.creation_pending
                or lease.pid is not None or self.process is not None):
            raise ContractError('Model4 cap requires SAME active original creation lease')
        if self.binding.device=='cpu':
            if lease.resident_bytes!=0 or self.resident_bytes!=0:
                raise ContractError('CPU execution requires zero GPU lease')
            cap=self._configured_execution_cap
        else:
            if lease.resident_bytes!=self.resident_bytes:
                raise ContractError('GPU execution cap requires SAME reserved bytes')
            cap=lease.resident_bytes
        checked_execution_cap(self.model,self.hpo_training,self.binding,cap)
        return ('--model4-execution-cap',str(cap))

    def _environment(self) -> dict[str, str]:
        environment = super()._environment()
        if self.diagnostic_invocation is not None:
            environment['GFB_HPO_DIAGNOSTIC_INVOCATION'] = self.diagnostic_invocation
            environment['GFB_HPO_DIAGNOSTIC_LOG_BYTES'] = str(self.limit)
        return environment

    async def request(self, request: Any, deadline: float) -> Any:
        if self.diagnostic_parser is not None and request.get('command') != 'close':
            from gf_common.hpo_progress import observe
            self.diagnostic_parser.update = request['projection']['collection_generation']+1 if request.get('command')=='update' else 0
            observe(self.diagnostic_parser.callback, {'cuda_allocator':None, 'learner_diagnostic_status':'unavailable-until-observed'})
        return await super().request(request, deadline)
