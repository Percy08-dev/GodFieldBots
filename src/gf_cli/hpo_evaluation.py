"""Concrete HPO evaluation requests; live Source and I/O stay with their owners.

An immutable plan is a checked request, never Source or evaluation authority.
Only an actual held global job can admit its concrete collector invocation.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gf_cli.config import GroupPackExperimentConfig, LoadedConfig
from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_collection import checked_linkage, hpo_evaluation_role
from gf_common.domain import (
    ContractError,
    EpisodeGroup,
    EpisodeSpec,
    Json,
    RuntimePolicyBinding,
    canonical,
)
from gf_common.schedule import serialize_schedule
from gf_training.hpo_gpu_lease import GpuJob


@dataclass(frozen=True)
class HpoEvaluationPlan:
    groups: tuple[EpisodeGroup, ...]
    output: Path
    _bindings_json: str
    _linkage_json: str
    started_monotonic: float
    admission_deadline: float
    execution_deadline: float
    work_deadline: float
    wall_deadline: float

    def __post_init__(self) -> None:
        if type(self.groups) is not tuple or not self.groups or any(type(group) is not EpisodeGroup for group in self.groups):
            raise ContractError('exact immutable HPO evaluation groups required')
        for group in self.groups:
            if type(group.episodes) is not tuple or any(type(spec) is not EpisodeSpec
                    or type(spec.policies) is not tuple or type(spec.sampling_seeds) is not tuple for spec in group.episodes):
                raise ContractError('immutable original evaluation episode/role streams required')
        if not isinstance(self.output, Path) or not self.output.is_absolute() or '..' in self.output.parts:
            raise ContractError('absolute owned evaluation output required')
        values = (self.started_monotonic, self.admission_deadline, self.execution_deadline,
                  self.work_deadline, self.wall_deadline)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ContractError('finite fresh evaluation deadline anchors required')
        start, admission, execution, work, wall = values
        if not start < admission <= execution <= work < wall:
            raise ContractError('evaluation deadlines cannot erase execution/finalization boundaries')
        link, registry = self.linkage, json.loads(self._bindings_json)
        checked_linkage(link, role=hpo_evaluation_role(sum(len(group.episodes) for group in self.groups)),
            schedule=serialize_schedule(self.groups, 4), registry=registry)
        if canonical(link) != self._linkage_json or canonical(registry) != self._bindings_json:
            raise ContractError('canonical frozen evaluation input required')

    @classmethod
    def create(cls, groups: tuple[EpisodeGroup, ...], bindings: dict[str, RuntimePolicyBinding],
               output: Path, linkage: Json, *, started_monotonic: float, admission_deadline: float,
               execution_deadline: float, work_deadline: float, wall_deadline: float) -> HpoEvaluationPlan:
        if type(bindings) is not dict or any(type(binding) is not RuntimePolicyBinding for binding in bindings.values()):
            raise ContractError('exact frozen runtime policy bindings required')
        return cls(groups, output, canonical({name: asdict(value) for name, value in bindings.items()}),
            canonical(linkage), started_monotonic, admission_deadline, execution_deadline, work_deadline, wall_deadline)

    @property
    def kind(self) -> str:
        return 'hpo_evaluation'

    @property
    def linkage(self) -> Json:
        return json.loads(self._linkage_json)

    @property
    def bindings(self) -> dict[str, RuntimePolicyBinding]:
        return {label: RuntimePolicyBinding(**value) for label, value in json.loads(self._bindings_json).items()}

    def require_job(self, gpu_job: GpuJob) -> None:
        if type(self) is not HpoEvaluationPlan or type(gpu_job) is not GpuJob:
            raise ContractError('exact HPO evaluation plan and live GPU job required')
        gpu_job._require_open()
        link = self.linkage
        expected = {'campaign_id': link['campaign_id'], 'operation_id': link['evaluation_id'],
            'kind': 'evaluation', 'run_id': link['left_parent']['run_id'], 'evaluation_id': link['evaluation_id']}
        if gpu_job.ticket != link['gpu_ticket'] or gpu_job._admission['context'] != expected:
            raise ContractError('evaluation differs from original held GPU admission context')
        ledger = gpu_job.ledger
        if ledger.workers or ledger.learner is not None or ledger.updating or ledger.cuda_bytes:
            raise ContractError('evaluation requires the SAME idle admitted ledger before creation')
        if time.monotonic() >= self.admission_deadline:
            raise TimeoutError('HPO evaluation admission deadline')

    def require(self, loaded: LoadedConfig, *, gpu_job: GpuJob) -> None:
        self.require_job(gpu_job)
        if (type(loaded) is not LoadedConfig or not isinstance(loaded.config, GroupPackExperimentConfig)
                or type(loaded.config.actions) is not ActionConfigV3 or loaded.config.run.split != 'validation'):
            raise ContractError('actual packed corrected validation configuration required')
        config = loaded.config
        start = self.started_monotonic
        if (self.wall_deadline > start + config.run.wall_seconds
                or self.admission_deadline > start + config.run.admission_seconds
                or self.execution_deadline > start + config.run.execution_seconds
                or self.work_deadline > self.wall_deadline - config.run.report_reserve_seconds):
            raise ContractError('evaluation request extends declared operation deadlines')
