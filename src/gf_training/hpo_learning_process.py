"""Wire6 adapter sharing the actual numerical child and reservation owner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gf_training.structured_learning_process import GameLearningProcess, StructuredLearningProcess


class HpoLearningProcess(GameLearningProcess):
    def __init__(self, directory: Path, max_log_bytes: int, model: Any, training: Any,
                 binding: Any, action_config_hash: str, resume_identity: str, **options: Any) -> None:
        actions = options.pop('actions')
        StructuredLearningProcess.__init__(self, directory, max_log_bytes, model, binding,
            action_config_hash, resume_identity, action_contract=actions.action_contract,
            _actions=actions, _hpo_training=training, **options)
        self.online_owner: Any = None
