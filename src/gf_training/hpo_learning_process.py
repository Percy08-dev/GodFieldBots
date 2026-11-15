"""Wire6 adapter sharing the actual numerical child and reservation owner."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from gf_training.structured_learning_process import GameLearningProcess, StructuredLearningProcess


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
        self.online_owner: Any = None

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
