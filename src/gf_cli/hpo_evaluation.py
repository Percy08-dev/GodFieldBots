"""Concrete HPO evaluation requests; live Source and I/O stay with their owners.

An immutable plan is a checked request, never Source or evaluation authority.
Only an actual held global job can admit its concrete collector invocation.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import math
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from gf_cli.config import GroupPackExperimentConfig, LoadedConfig
from gf_cli.hpo_config import LoadedHPOConfig, require_hpo_runtime_execution
from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_collection import checked_linkage, hpo_evaluation_role
from gf_common.domain import (
    ContractError,
    EpisodeGroup,
    EpisodeSpec,
    GroupResult,
    Json,
    RuntimePolicyBinding,
    canonical,
    clone,
    digest,
)
from gf_common.hpo_progress import observe
from gf_common.schedule import serialize_schedule
from gf_common.shared_inference import evaluation_runtime
from gf_training.hpo_gpu_lease import GpuJob

if TYPE_CHECKING:
    from gf_evaluation.hpo_compare import ComparisonPanel
    from gf_training.hpo_retention import HpoReferenceReceipt

_TRAINING_RECEIPT_FIELDS = {'proposal_sha256', 'training_seed', 'run_path', 'run_id', 'config_sha256',
    'state_sha256', 'generation', 'generation_sha256', 'inference_sha256', 'used_decisions',
    'source_receipt_sha256', 'gpu_termination_sha256'}
_EVALUATION_RECEIPT_FIELDS = {'training_receipt_sha256', 'linkage_sha256', 'manifest_sha256', 'originals_digest',
    'projection', 'conditions_sha256', 'pin_acquire_sha256', 'pin_release_sha256', 'gpu_termination_sha256',
    'directory', 'linkage', 'config_sha256'}


def owned_runtime_module(name: str, required: tuple[str, ...]) -> ModuleType:
    """Only fixed caller-selected HPO modules from this exact owned build."""
    allowed = {'gf_training.hpo_retention', 'gf_training.hpo_ppo', 'gf_cli.card_instance', 'gf_cli.hpo_config'}
    if name not in allowed:
        raise ContractError('unknown HPO runtime dependency')
    expected = Path(__file__).resolve().parents[1] / (name.replace('.', '/') + '.py')
    if not expected.is_file():
        raise ContractError('required HPO runtime support is unavailable in this build')
    module = importlib.import_module(name)
    if module.__file__ is None or Path(module.__file__).resolve() != expected or any(not hasattr(module, item) for item in required):
        raise ContractError('required HPO runtime support is not the owned compatible build')
    return module


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
        if type(self) not in (HpoEvaluationPlan, ResidentHpoEvaluationPlan) or type(gpu_job) is not GpuJob:
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


@dataclass(frozen=True)
class ResidentHpoEvaluationPlan(HpoEvaluationPlan):
    _execution_runtime_json: str
    _pins: tuple[Any, ...] = field(repr=False,compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        from gf_common.hpo_resident import checked_evaluation_runtime
        raw = checked_evaluation_runtime(self.execution_runtime,self.linkage,json.loads(self._bindings_json))
        if canonical(raw) != self._execution_runtime_json or type(self._pins) is not tuple:
            raise ContractError('canonical resident evaluation runtime and original pin set required')

    @property
    def execution_runtime(self) -> Json:
        return json.loads(self._execution_runtime_json)

    @classmethod
    def create_resident(cls, *args: Any, runtime: Json, pins: tuple[Any,...], **options: Any) -> ResidentHpoEvaluationPlan:
        from gf_common.hpo_resident import resident_runtime
        base = HpoEvaluationPlan.create(*args,**options)
        link = base.linkage
        binding = {'schema_version':1,'kind':'hpo-resident-evaluation-runtime','runtime':resident_runtime(runtime),
            'evaluation_id':link['evaluation_id'],'job_ticket':link['gpu_ticket'],
            'linkage_sha256':digest(link),'policy_registry_sha256':digest(json.loads(base._bindings_json))}
        return cls(**{item.name:getattr(base,item.name) for item in fields(HpoEvaluationPlan)},
            _execution_runtime_json=canonical(binding),_pins=pins)


def execution_plan(*args: Any,runtime: Json | None,pins: tuple[Any,...],**options: Any) -> HpoEvaluationPlan:
    if runtime is None:
        return HpoEvaluationPlan.create(*args,**options)
    return ResidentHpoEvaluationPlan.create_resident(*args,runtime=runtime,pins=pins,**options)


class _HpoEvaluationCleanupObservation:
    """One invocation's existing owners; never a cleanup or data capability itself.

    Phases are advanced at the actual creation boundaries in the collector.
    Retirement always rechecks the retained concrete owners, not those phases
    alone. Partial construction without an obtainable owner remains uncertain.
    """

    def __init__(self, plan: HpoEvaluationPlan, job: GpuJob, artifacts: Any) -> None:
        from gf_models.card_instance_artifacts import RetainedGameArtifacts
        from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
        if type(plan) not in (HpoEvaluationPlan, ResidentHpoEvaluationPlan) or type(artifacts) not in (RetainedGameArtifacts,OwnedGameArtifacts):
            raise ContractError('exact evaluation cleanup inputs required')
        plan.require_job(job)
        self._identity = (plan, job, job.ledger, artifacts)
        self._core_phase = 'unentered'
        self._resident_runtime: Any = None
        self._run_phase = 'unentered'
        self._core_error: BaseException | None = None
        self._result: Any = None
        self._tasks: list[asyncio.Task[GroupResult]] | None = None
        self._task_phase = 'unentered'
        self._task_creation_failure: tuple[Any, BaseException] | None = None
        self._groups: dict[asyncio.Task[GroupResult], tuple[EpisodeGroup, str, Any, GroupResult | None]] = {}
        self._runtime_originals: dict[asyncio.Task[GroupResult], tuple[Any, Any, Any, tuple[Any, ...]]] = {}
        self._pack_phase, self._publication_phase = 'unentered', 'unentered'
        self._pack: Any = None
        self._publication: Any = None
        self._composite_phase = 'unentered'
        self._composite: Any = None
        self._verifier_phase = 'unentered'
        self._verifier: Any = None
        self._verifier_originals: tuple[Any, Any, Any] | None = None
        self._verifier_error: BaseException | None = None
        self._artifact_phase = 'unentered'

    def require_identity(self, plan: HpoEvaluationPlan, job: GpuJob, artifacts: Any) -> None:
        if (type(self) is not _HpoEvaluationCleanupObservation or
                any(left is not right for left, right in zip(self._identity,
                    (plan, job, job.ledger, artifacts), strict=True))):
            raise ContractError('foreign evaluation cleanup observation')
        job._require_owner()
        if job._receipt is None:
            job._require_open()
        plan.__post_init__()

    def _enter(self, plan: HpoEvaluationPlan, job: GpuJob, artifacts: Any) -> None:
        self.require_identity(plan, job, artifacts)
        if self._core_phase != 'unentered':
            raise ContractError('evaluation cleanup observation already invoked')
        self._core_phase = 'entered'

    def _register_task(self, group: EpisodeGroup, task: asyncio.Task[GroupResult]) -> None:
        if self._task_phase != 'creating' or self._tasks is None or self._tasks[-1] is not task:
            raise ContractError('original incrementally registered evaluation task required')
        self._groups[task] = (group, 'scheduled', None, None)

    def _creating_environment(self, group: EpisodeGroup) -> None:
        task = asyncio.current_task()
        if task not in self._groups or self._groups[task][0] is not group:
            raise ContractError('original evaluation task/group required')
        self._groups[task] = (group, 'creating', None, None)

    def _environment(self, group: EpisodeGroup, environment: Any) -> None:
        task = asyncio.current_task()
        if (task not in self._groups or self._groups[task][:2] != (group, 'creating')
                or self._groups[task][2] is not environment):
            raise ContractError('original evaluation environment creation required')
        self._groups[task] = (group, 'owned', environment, None)

    def _partial_environment(self, group: EpisodeGroup, environment: Any) -> None:
        task = asyncio.current_task()
        if task not in self._groups or self._groups[task] != (group, 'creating', None, None):
            raise ContractError('original evaluation environment construction required')
        self._groups[task] = (group, 'creating', environment, None)

    def _group_result(self, group: EpisodeGroup, result: GroupResult) -> None:
        task = asyncio.current_task()
        if task not in self._groups or self._groups[task][:2] != (group, 'owned'):
            raise ContractError('original evaluation environment/result required')
        self._groups[task] = (group, 'returned', self._groups[task][2], result)
        environment = self._groups[task][2]
        runtime = environment._storage_group_owner
        server = None if runtime is None else runtime._managed_server
        self._runtime_originals[task] = (runtime, server, None if server is None else server.process,
            () if server is None else tuple(server.drainers))

    def require_resident_resources_retired(self) -> None:
        if self._resident_runtime is None:
            return
        _,job,ledger,_=self._identity
        custody=self._resident_runtime.custody
        if job._receipt is None:
            custody.require_resource_retirement(job,ledger)
        else:
            custody.require_confirmed_resource_retirement(job,ledger,job._receipt)

    def require_core_retired(self, *, error: BaseException) -> None:
        from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
        from gf_training.card_instance_collection import CollectionPublication
        from gf_training.collection import PackedRunResult
        from gf_training.group_pack_collection import GroupPackCollection
        from gf_training.group_pack_publication import GroupPackCollectionPublication
        plan, job, _, artifacts = self._identity
        self.require_identity(plan, job, artifacts)
        if not ((self._core_phase == 'failed' and self._core_error is error) or
                (self._core_phase == 'returned' and type(self._result) is PackedRunResult)):
            raise ContractError('actual returned result or SAME original collector error required')
        if self._run_phase == 'requested' or (self._core_phase == 'returned' and self._run_phase != 'entered'):
            raise ContractError('actual core collection invocation was not observed')
        if self._task_phase != 'unentered':
            if self._tasks is None or set(self._tasks) != set(self._groups):
                raise ContractError('partial evaluation task registration')
            if self._task_phase != 'settled':
                raise ContractError('evaluation task creation not settled')
            for task, (group, phase, environment, result) in self._groups.items():
                if not task.done():
                    raise ContractError('actual evaluation task still running')
                if phase == 'scheduled':
                    # This coroutine never reached the environment creation site.
                    if not task.cancelled():
                        raise ContractError('scheduled evaluation work not positively cancelled')
                    continue
                if (phase != 'returned' or type(environment) is not CardInstanceSimulatorEnvironment
                        or type(result) is not GroupResult or task.cancelled() or task.exception() is not None
                        or task.result() is not result or not environment.storage_cleanup_complete(result)):
                    raise ContractError('actual evaluation group reap/drain remains unconfirmed')
                runtime, server, process, drainers = self._runtime_originals[task]
                if runtime is None:
                    raise ContractError('original evaluation runtime owner missing')
                if (runtime._managed_server is not server or environment._storage_group_owner is not runtime
                        or (server is not None and (server.process is not process or tuple(server.drainers) != drainers))):
                    raise ContractError('evaluation runtime owners changed after actual group settlement')
        self.require_resident_resources_retired()
        if self._pack_phase != 'unentered':
            pack = self._pack
            environments = [value[2] for value in self._groups.values() if value[2] is not None]
            if (self._pack_phase != 'owned' or type(pack) is not GroupPackCollection or not pack._closed
                    or pack._futures or pack._archival_tasks or pack._cleanup_children
                    or not pack._executor._shutdown or pack.budget.used or pack._metadata_reservations
                    or any(not group.archived for group in pack._groups)
                    or (pack._monitor is not None and not pack._monitor.done())
                    or pack.writer is None or pack.writer.returncode is None or not pack.writer._base_released
                    or len(pack._groups) != len(environments)
                    or any(sum(group.runtime_owner is environment for environment in environments) != 1 for group in pack._groups)
                    or (pack.pending is not None and (pack.pending.fd >= 0 or pack.pending.database.fd >= 0))):
                raise ContractError('actual evaluation Pack retirement remains unconfirmed')
        if self._publication_phase != 'unentered':
            publication = self._publication
            if (self._publication_phase != 'owned' or type(publication) is not GroupPackCollectionPublication
                    or publication.collection is not self._pack or not publication.cleanup_complete):
                raise ContractError('actual evaluation publication retirement remains unconfirmed')
            if self._core_phase == 'returned' and self._result is not publication._result:
                raise ContractError('returned evaluation differs from the actual publication result')
        if (self._composite_phase != 'unentered' and (self._composite_phase != 'owned'
                or type(self._composite) is not CollectionPublication or self._composite.group_pack is not self._publication
                or not self._composite.cleanup_complete)):
            raise ContractError('original composite publication retirement remains unconfirmed')

    def release_artifacts(self) -> None:
        if self._artifact_phase != 'unentered':
            return  # A failed release is uncertainty, not permission to retry it.
        self.require_resident_resources_retired()
        self._artifact_phase = 'releasing'
        self._identity[3].release()
        self._artifact_phase = 'retired'

    def require_outer_retired(self, *, error: BaseException) -> None:
        self.require_core_retired(error=error)
        artifacts = self._identity[3]
        from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
        if type(artifacts) is OwnedGameArtifacts:
            artifacts.require_retired()
        if (self._artifact_phase != 'retired' or not artifacts._released or artifacts._entries
                or type(artifacts) is not OwnedGameArtifacts and (artifacts._models or artifacts._scan.files or artifacts._scan.inventories)):
            raise ContractError('actual retained evaluation artifacts remain unconfirmed')
        if self._verifier_phase != 'unentered':
            source = self._verifier
            if self._verifier_phase != 'retired' or self._verifier_error is not None or self._verifier_originals is None:
                raise ContractError('actual evaluation verifier construction/close remains unconfirmed')
            pool, packed, locator = self._verifier_originals
            if (source.pool is not pool or source.artifacts is not packed or source.locator is not locator
                    or not source._closed or source._charge_uncertain or source._cleanup_children or source._charge_bytes
                    or not pool.closed or pool._close_holds or pool.sources or pool._cleanup_children
                    or pool._base_uncertain or pool.budget.used or not packed._closed or packed._loans
                    or locator.fd >= 0 or locator._uncertain_releases):
                raise ContractError('actual evaluation verifier originals remain owned')


def _receipt(value: Json, kind: str, fields: set[str]) -> Json:
    from gf_common.domain import closed, integer
    resident = kind == 'hpo-seed-evaluation' and value.get('schema_version') == 2
    closed(value, {'schema_version', 'kind'} | fields | ({'execution_runtime_sha256'} if resident else set()))
    integer(value['schema_version'], 2 if resident else 1, 2 if resident else 1)
    if value['kind'] != kind:
        raise ContractError('unexpected closed HPO evaluation receipt')
    return value


@dataclass(frozen=True)
class ParentInspection:
    state: Json
    generations: tuple[Json, ...]
    source: Json


def completion_inspection(completion: Any, directory: Path) -> ParentInspection:
    """CLI projection of readonly final data; training imports no CLI types."""
    import weakref

    from gf_training.hpo_owned_completion import (
        OwnedEvaluationPolicy,
        OwnedExternalFinalPolicy,
        OwnedRunCompletion,
    )
    if type(completion) not in (OwnedRunCompletion,OwnedExternalFinalPolicy,OwnedEvaluationPolicy):
        raise ContractError('actual completed/imported/retained final data required')
    completion.require()
    reservation=completion.parent_budget.reserve(32*(len(completion._state)+len(completion._records)+len(completion._files))+65536)
    try:
        state,records=completion.state,completion.records
        if directory!=Path(state['effective_config']['hpo_spec']['storage']['parent_root'])/state['run_id']:
            raise ContractError('final inference parent namespace mismatch')
        source={'schema_version':2,'kind':'hpo-owned-parent-metadata-validation','origin':'producer-owned-v1','encoder':'canonical-python',
            'run_id':state['run_id'],'run_path':str(directory),'training_seed':state['seed'],'config_sha256':state['config_sha256'],
            'state_sha256':completion._state_sha256,'generation':state['updates'],'generation_sha256':completion._generation_sha256,
            'generations_sha256':completion._generations_sha256,'inference_sha256':completion.files['inference.json'],
            'used_decisions':state['used_decisions'],'effective_settings_sha256':state['effective_config']['hpo_spec']['effective_settings_sha256']}
        result=ParentInspection(state,tuple(records),source)
        weakref.finalize(result,reservation.close)
        return result
    except BaseException:
        reservation.close()
        raise


def _checked_owned_inspection(inspection: ParentInspection, loaded: LoadedHPOConfig, settings: Any, *,
                              training_seed: int,run_id: str) -> ParentInspection:
    from gf_training.hpo_lineage import hpo_spec, hpo_state
    state=inspection.state
    spec=hpo_spec(state)
    _,model,_=hpo_state(state)
    if (state['run_id']!=run_id or state['seed']!=training_seed or state['used_decisions']!=loaded.config.decision_budget
            or spec['campaign_config_sha256']!=loaded.sha256 or spec['base_config_sha256']!=loaded.base.sha256
            or spec['study_identity']!=loaded.config.study_identity or spec['effective_settings_sha256']!=settings.identity
            or canonical(asdict(model))!=canonical(asdict(settings.model))
            or canonical(spec['training_recipe'])!=canonical(asdict(settings.training))):
        raise ContractError('owned parent metadata differs from exact campaign/proposal/final-D identity')
    return inspection


def _final_policy_reference(inspection: ParentInspection, source_sha: str) -> tuple[Json, RuntimePolicyBinding]:
    """Derive final-D identity/label from the independently inspected parent."""
    from gf_common.card_instance_collection import checked_hpo_evaluation_parent
    from gf_training.hpo_source import hpo_binding
    state, source = inspection.state, inspection.source
    generation = state['updates']
    if (len(inspection.generations) != generation + 1 or inspection.generations[-1]['generation'] != generation
            or source['run_id'] != state['run_id'] or source['training_seed'] != state['seed']
            or source['generation'] != generation
            or source['inference_sha256'] != inspection.generations[-1]['inference_sha256']):
        raise ContractError('inspected final parent/generation facts are inconsistent')
    label, binding = hpo_binding(Path(source['run_path']), state, inspection.generations[-1], generation)
    reference = {'run_id': state['run_id'], 'run_path': source['run_path'], 'training_seed': state['seed'],
        'generation': generation, 'generation_sha256': source['generation_sha256'],
        'inference_sha256': inspection.generations[-1]['inference_sha256'], 'source_receipt_sha256': source_sha,
        'policy_label': label}
    return checked_hpo_evaluation_parent(reference), binding


def _parent_conditions(inspection: ParentInspection, campaign: Json) -> None:
    conditions = campaign['conditions']
    producer = conditions.get('producer_identity')
    if type(producer) is not dict or type(conditions.get('integration_identity')) is not dict:
        raise ContractError('actual measured campaign source/Simulator conditions required')
    identity = inspection.state['identity']
    if (any(canonical(identity.get(key)) != canonical(value) for key, value in producer.items())
            or canonical(identity.get('integration')) != canonical(conditions['integration_identity'])
            or canonical(inspection.state['execution']['inference']['binding']) != canonical(conditions['inference_binding'])
            or inspection.state['execution']['inference']['cuda_visible_devices'] != conditions['cuda_visible_devices']):
        raise ContractError('original parent source/runtime/inference conditions differ from campaign')
    if 'native_identity' in conditions and canonical(identity['preprocessing'].get('binary')) != canonical(conditions['native_identity']):
        raise ContractError('original parent native preprocessing identity changed')


def prospective_evaluation_admission(loaded: LoadedHPOConfig,settings: Any=None) -> Json:
    """Full-campaign guard only; direct bounded training does not create evaluation."""
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.structured_binding import input_binding_game_v3
    from gf_models.structured_runtime import admit_hpo_game_workers
    config=loaded.config
    model=config.model if settings is None else settings.model
    actions: Any=loaded.base.config.actions
    base_model: Any=loaded.base.config.model
    options={'version':'4','checkpoint':'/prospective-final-inference','expected_manifest_digest':'0'*64,
        'model':asdict(model),'inference':asdict(config.inference),
        'cuda_visible_devices':base_model.cuda_visible_devices,
        'action_config_hash':digest(asdict(actions)),**input_binding_game_v3(actions.action_contract,card_instance_rules())}
    op=config.operating
    result=admit_hpo_game_workers(actions,(RuntimePolicyBinding(model.model_id,options),RuntimePolicyBinding('heuristic')),
        rules=card_instance_rules(),game_concurrency=op.game_concurrency,global_cuda_bytes=op.managed_cuda_bytes,
        max_workers=op.max_participant_workers)
    return result|{'evaluation_runtime':evaluation_runtime(config),'scope':'planned fresh final-policy vs baseline; no evaluation authority'}


def inspect_training_parent(directory: Path, loaded: LoadedHPOConfig, settings: object, *,
                            training_seed: int, run_id: str, deadline: float, completion: Any = None) -> ParentInspection:
    """Actually validate complete originals; Run5/JSON alone is never authority."""
    from gf_models.arrays import sha256
    from gf_training.hpo_config import EffectiveSettings
    from gf_training.hpo_lineage import hpo_spec, hpo_state
    from gf_training.hpo_source import validate_hpo_parent
    from gf_training.lineage import generation_path, run_lock

    if type(loaded) is not LoadedHPOConfig or type(settings) is not EffectiveSettings:
        raise ContractError('actual loaded campaign and normalized proposal required')
    directory = directory.resolve()
    if training_seed not in loaded.config.learning_seeds or not directory.is_relative_to(Path(loaded.config.storage.parent_root)):
        raise ContractError('parent is outside the declared campaign run/seed namespace')
    if loaded.config.schema_version in (4,5):
        from gf_training.hpo_owned_completion import OwnedRunCompletion
        if completion is not None:
            if type(completion) is not OwnedRunCompletion: raise ContractError('actual owned completion required')
            inspection=completion_inspection(completion,directory)
        else:
            from gf_common.artifacts import ArtifactPhaseAdmission
            from gf_common.group_pack import StorageSettings
            from gf_training.group_pack_resources import StorageBudget
            from gf_training.hpo_owned_restore import OwnedColdImport
            with run_lock(directory) as ownership:
                budget=StorageBudget(StorageSettings(budget_bytes=loaded.config.operating.source_budget_bytes),deadline,
                    name='source',phase='hpo-parent-inspection')
                phase=ArtifactPhaseAdmission(budget,loaded.config.operating.parent_budget_bytes,
                    name='parent',phase='hpo-parent-inspection')
                imported=OwnedColdImport(ownership,budget=budget,parent_budget=phase,deadline=deadline,require_complete=True)
                try:
                    state,records=imported.state,imported.records
                    import weakref

                    from gf_common.artifacts import json_bound
                    from gf_training.hpo_owned_completion import management_records
                    retained=phase.reserve(32*(json_bound(state)+json_bound(records))+65536)
                    source={'schema_version':2,'kind':'hpo-owned-parent-metadata-validation','origin':'producer-owned-v1','encoder':'canonical-python',
                        'run_id':state['run_id'],'run_path':str(directory),'training_seed':state['seed'],'config_sha256':state['config_sha256'],
                        'state_sha256':imported.scan.verified_digest(directory/'run.json'),'generation':state['updates'],
                        'generation_sha256':state['generation_sha256'],'generations_sha256':digest(records),
                        'inference_sha256':records[-1]['inference_sha256'],'used_decisions':state['used_decisions'],
                        'effective_settings_sha256':state['effective_config']['hpo_spec']['effective_settings_sha256']}
                    inspection=ParentInspection(clone(state),tuple(management_records(records)),source)
                    weakref.finalize(inspection,retained.close)
                finally: imported.close()
        return _checked_owned_inspection(inspection,loaded,settings,training_seed=training_seed,run_id=run_id)
    if completion is not None: raise ContractError('legacy evaluation cannot adopt owned completion')
    with run_lock(directory):
        state, records = validate_hpo_parent(directory, require_complete=True,
            max_owned_bytes=loaded.config.operating.source_budget_bytes, deadline=deadline)
        spec = hpo_spec(state)
        _, model, _ = hpo_state(state)
        if (state['run_id'] != run_id or state['seed'] != training_seed
                or state['used_decisions'] != loaded.config.decision_budget
                or spec['campaign_config_sha256'] != loaded.sha256 or spec['base_config_sha256'] != loaded.base.sha256
                or spec['study_identity'] != loaded.config.study_identity or spec['effective_settings_sha256'] != settings.identity
                or canonical(asdict(model)) != canonical(asdict(settings.model))
                or canonical(spec['training_recipe']) != canonical(asdict(settings.training))):
            raise ContractError('full original parent differs from exact campaign/proposal/final-D identity')
        generation = state['updates']
        source = {'schema_version': 1, 'kind': 'hpo-parent-source-validation', 'run_id': run_id,
            'run_path': str(directory), 'training_seed': training_seed, 'config_sha256': state['config_sha256'],
            'state_sha256': sha256(directory / 'run.json', deadline), 'generation': generation,
            'generation_sha256': sha256(generation_path(directory, generation), deadline),
            'generations_sha256': digest(records), 'inference_sha256': records[-1]['inference_sha256'],
            'used_decisions': state['used_decisions'], 'effective_settings_sha256': settings.identity}
        return ParentInspection(state, tuple(records), source)


def verify_gpu_termination(value: Json, *, campaign_id: str, operation_id: str,
                           run_id: str, evaluation_id: str | None) -> None:
    """Verify the fixed namespace's exact receipt; a copied JSON is not proof."""
    from gf_common.domain import closed
    from gf_training import hpo_gpu_lease as gpu
    ticket = value.get('ticket')
    if not isinstance(ticket, str):
        raise ContractError('actual GPU receipt ticket required')
    gpu._ticket(ticket)
    directory = gpu._GLOBAL_ROOT / 'jobs' / ticket
    for ancestor in (gpu._GLOBAL_ROOT, gpu._GLOBAL_ROOT / 'jobs', directory):
        gpu._directory(ancestor)
    admission = gpu._read(directory / 'admission.json')
    gpu._admission(admission, ticket)
    gpu._termination(value, admission)
    stored = gpu._termination(gpu._read(directory / 'termination.json'), admission)
    if canonical(asdict(stored)) != canonical(value):
        raise ContractError('campaign GPU termination differs from the actual fixed namespace receipt')
    context = closed(admission['context'], {'campaign_id', 'operation_id', 'kind', 'run_id', 'evaluation_id'})
    expected = {'campaign_id': campaign_id, 'operation_id': operation_id,
        'kind': 'training' if evaluation_id is None else 'evaluation', 'run_id': run_id, 'evaluation_id': evaluation_id}
    if context != expected:
        raise ContractError('GPU termination belongs to a different campaign/run/evaluation')


def verify_saved_evaluation(directory: Path, loaded: LoadedHPOConfig, linkage: Json, *,
                            deadline: float, extra_owned_bytes: int = 0,
                            _cleanup_observation: _HpoEvaluationCleanupObservation | None = None,
                            evaluation_runtime: Json | None = None) -> tuple[Json, Json]:
    """Reopen actual packed members and reconcile BOTH original participant views."""
    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import preparation_config
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.structured_source import _inspect_component_collection

    settings = preparation_config(asdict(loaded.base.config.preprocessing))
    root = Path(loaded.config.storage.originals_root or (
        loaded.base.config.storage.db_root if isinstance(loaded.base.config, GroupPackExperimentConfig) else '/mnt/d/godfield'))
    observation = _cleanup_observation
    owner = PackedSourceOwner.__new__(PackedSourceOwner)
    if observation is not None:
        if observation._verifier_phase != 'unentered':
            raise ContractError('evaluation verifier observation already consumed')
        observation._verifier, observation._verifier_phase = owner, 'creating'
    try:
        PackedSourceOwner.__init__(owner, directory, settings, deadline, storage_binding=storage_binding(),
            max_owned_bytes=loaded.config.operating.source_budget_bytes, extra_owned_bytes=extra_owned_bytes,
            expected_descriptor_root=root / linkage['evaluation_id'])
    except BaseException as error:
        if observation is not None:
            observation._verifier_error = error
            vars(error)['hpo_verifier_cleanup_owner'] = owner
        try:
            owner.close()
        except BaseException as cleanup:  # noqa: BLE001 -- partial constructor still retains its actual owner.
            vars(error)['hpo_verifier_cleanup_owner'] = owner
            error.add_note('partial evaluation verifier cleanup: ' + str(cleanup))
        raise
    if observation is not None:
        observation._verifier_originals = (owner.pool, owner.artifacts, owner.locator)
        observation._verifier_phase = 'owned'
    try:
        collection = _inspect_component_collection(owner, loaded.base.config.actions, 'validation', _game=True,
            _game_role=hpo_evaluation_role(2 * len(linkage['game_seeds'])))
        manifest = collection.manifest
        check_evaluation_execution(manifest,evaluation_runtime)
        if canonical(manifest['linkage']) != canonical(linkage):
            raise ContractError('actual saved evaluation differs from exact original request')
        proof = collection.proof
        if proof['games'] != 2 * len(linkage['game_seeds']) or proof['missing_replays'] or proof['closure_counts'] != {'natural': proof['games']}:
            raise ContractError('saved original evaluation is not a complete natural panel')
        owner.finalize()
    except BaseException as error:
        try:
            owner.close()
        except BaseException as cleanup:  # noqa: BLE001 -- keep primary and the SAME original Source.
            if observation is not None:
                observation._verifier_error = cleanup
            vars(error)['hpo_verifier_cleanup_owner'] = owner
            error.add_note('evaluation verifier cleanup: ' + str(cleanup))
        else:
            if observation is not None:
                observation._verifier_phase = 'retired'
        raise
    try:
        owner.close()
    except BaseException as error:
        if observation is not None:
            observation._verifier_error = error
        vars(error)['hpo_verifier_cleanup_owner'] = owner
        raise
    if observation is not None:
        observation._verifier_phase = 'retired'
    return manifest, proof


def check_evaluation_execution(manifest: Json, runtime: Json | None) -> None:
    from gf_common.hpo_resident import checked_evaluation_runtime, resident_runtime
    actual = manifest.get('execution_runtime')
    if runtime is None:
        if actual is not None or manifest['schema_version'] != 6:
            raise ContractError('saved evaluation differs from explicitly fresh runtime')
    else:
        if manifest['schema_version'] != 7 or type(actual) is not dict:
            raise ContractError('resident evaluation lacks explicit original execution runtime')
        checked_evaluation_runtime(actual,manifest['linkage'],manifest['policy_bindings'])
        if canonical(actual['runtime']) != canonical(resident_runtime(runtime)):
            raise ContractError('saved evaluation runtime differs from requested recipe')


def evaluation_receipt_runtime_fields(manifest: Json) -> Json:
    if 'execution_runtime' not in manifest:
        return {'schema_version':1}
    from gf_common.hpo_resident import checked_evaluation_runtime
    actual=checked_evaluation_runtime(manifest['execution_runtime'],manifest['linkage'],manifest['policy_bindings'])
    return {'schema_version':2,'execution_runtime_sha256':digest(actual)}


def check_evaluation_receipt_runtime(receipt: Json, manifest: Json) -> None:
    if 'execution_runtime' in manifest:
        from gf_common.hpo_resident import checked_evaluation_runtime
        actual = checked_evaluation_runtime(manifest['execution_runtime'],manifest['linkage'],manifest['policy_bindings'])
        if receipt.get('schema_version') != 2 or receipt.get('execution_runtime_sha256') != digest(actual):
            raise ContractError('resident evaluation receipt differs from independently reopened runtime')
    elif receipt.get('schema_version') != 1 or 'execution_runtime_sha256' in receipt:
        raise ContractError('fresh evaluation receipt cannot assert resident runtime')


def _projection(manifest: Json, parent: Json, source_sha: str, conditions_sha: str) -> Json:
    from gf_evaluation.hpo import (
        GameIdentity,
        GameProjection,
        RunIdentity,
        SeedEvaluationProjection,
    )
    identities = []
    games = []
    evaluation_id = manifest['linkage']['evaluation_id']
    if type(evaluation_id) is not str or not evaluation_id or manifest['run_id'] != evaluation_id:
        raise ContractError('projection requires the actual original evaluation namespace')
    for row in manifest['episodes']:
        identity = GameIdentity(row['seed'], manifest['linkage']['learner_seats'][row['episode_id']],
            evaluation_id + '/' + row['episode_id'], row['game_id'])
        identities.append(identity)
        game = GameProjection.from_result(identity, result=row['outcome'], decisions=row['decisions'])
        games.append(replace(game, elapsed_seconds=row.get('elapsed_seconds', 0.)))
    run = RunIdentity(parent['training_seed'], parent['run_id'], parent['inference_sha256'], source_sha, tuple(identities))
    return asdict(SeedEvaluationProjection(run, parent['used_decisions'], conditions_sha, tuple(games)))


def verify_seed_evaluation(store: object, loaded: LoadedHPOConfig, campaign: Json, proposal: object,
                           seed: int, *, deadline: float) -> tuple[object, str, str]:
    """Reopen every actual parent/original/cleanup link for one closed seed."""
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    from gf_evaluation.hpo import (
        GameIdentity,
        GameProjection,
        RunIdentity,
        SeedEvaluationProjection,
    )
    if type(store) is not CampaignStore or type(proposal) is not Proposal:
        raise ContractError('actual campaign owner and durable proposal required')
    if seed not in loaded.config.learning_seeds or type(seed) is not int:
        raise ContractError('seed evaluation is outside declared complete group')
    root = f'proposals/{proposal.number:06}'
    if store.exists(root + '/failure.json'):
        raise ContractError('failed proposal cannot be requalified by success-looking files')
    folder = root + f'/seeds/{seed}'
    training, train_sha = store.get(folder + '/training.json')
    _receipt(training, 'hpo-seed-training', _TRAINING_RECEIPT_FIELDS)
    if training['proposal_sha256'] != proposal.sha256 or training['training_seed'] != seed:
        raise ContractError('seed training receipt belongs to another proposal/seed')
    inspection = inspect_training_parent(Path(training['run_path']), loaded, proposal.settings,
        training_seed=seed, run_id=training['run_id'], deadline=deadline)
    _parent_conditions(inspection, campaign)
    source, source_sha = store.get(folder + '/source-validation.json')
    if canonical(source) != canonical(inspection.source) or source_sha != training['source_receipt_sha256']:
        raise ContractError('saved parent Source receipt differs from actual originals')
    for key in ('run_id', 'run_path', 'training_seed', 'config_sha256', 'state_sha256', 'generation',
                'generation_sha256', 'inference_sha256', 'used_decisions'):
        if canonical(training[key]) != canonical(source[key]):
            raise ContractError('training receipt differs from full final-D Source facts')
    termination, term_sha = store.get(folder + '/training-gpu-termination.json')
    if termination.get('schema_version')!=(2 if loaded.config.schema_version==5 else 1):
        raise ContractError('training GPU termination version differs from explicit campaign mode')
    verify_gpu_termination(termination, campaign_id=campaign['campaign_id'], operation_id=training['run_id'],
        run_id=training['run_id'], evaluation_id=None)
    if term_sha != training['gpu_termination_sha256']:
        raise ContractError('training receipt does not bind exact actual termination')
    evaluation, eval_sha = store.get(folder + '/evaluation.json')
    _receipt(evaluation, 'hpo-seed-evaluation', _EVALUATION_RECEIPT_FIELDS)
    link = evaluation['linkage']
    final_reference, final_binding = _final_policy_reference(inspection, source_sha)
    if (evaluation['training_receipt_sha256'] != train_sha or evaluation['conditions_sha256'] != campaign['conditions_sha256']
            or evaluation['linkage_sha256'] != digest(link) or link['purpose'] != 'objective'
            or link['conditions_sha256'] != campaign['conditions_sha256']
            or canonical(link['left_parent']) != canonical(final_reference)):
        raise ContractError('seed evaluation differs from exact training/conditions identity')
    manifest, proof = verify_saved_evaluation(Path(evaluation['directory']), loaded, link, deadline=deadline,
        evaluation_runtime=evaluation_runtime(loaded.config))
    check_evaluation_receipt_runtime(evaluation,manifest)
    if canonical(manifest['policy_bindings'].get(final_reference['policy_label'])) != canonical(asdict(final_binding)):
        raise ContractError('original evaluated policy differs from the inspected final-D binding')
    expected_request, _ = store.get(folder + '/evaluation.intent.json')
    if (proof['manifest_sha256'] != evaluation['manifest_sha256'] or proof['source_set_sha256'] != evaluation['originals_digest']
            or manifest['config_hash'] != evaluation['config_sha256']
            or expected_request.get('config_sha256') != evaluation['config_sha256']):
        raise ContractError('seed evaluation original bytes differ from closed receipt')
    reference = {'run_id': training['run_id'], 'token': link['gpu_ticket'], 'generation': training['generation'],
        'inference_sha256': training['inference_sha256'], 'generation_sha256': training['generation_sha256']}
    # Editable dependency paths must never supply a missing private runtime
    # implementation from another checkout.
    retention = owned_runtime_module('gf_training.hpo_retention', ('release_hpo_evaluation',))
    released = retention.release_hpo_evaluation(Path(training['run_path']), reference,
        max_owned_bytes=loaded.config.operating.source_budget_bytes, deadline=deadline)
    journal = Path(training['run_path']) / 'evaluation-reference-receipts' / link['gpu_ticket']
    from gf_models.arrays import sha256
    if (released.receipt_sha256 != evaluation['pin_release_sha256']
            or sha256(journal / 'acquire.receipt.json', deadline) != evaluation['pin_acquire_sha256']):
        raise ContractError('evaluation pin/release transaction receipt changed')
    termination, term_sha = store.get(folder + '/evaluation-gpu-termination.json')
    verify_gpu_termination(termination, campaign_id=campaign['campaign_id'], operation_id=link['evaluation_id'],
        run_id=training['run_id'], evaluation_id=link['evaluation_id'])
    if term_sha != evaluation['gpu_termination_sha256']:
        raise ContractError('evaluation does not bind actual cleanup termination')
    projected = _projection(manifest, training, source_sha, campaign['conditions_sha256'])
    if canonical(projected) != canonical(evaluation['projection']):
        raise ContractError('saved score projection differs from actual natural terminal DTOs')
    raw_run = projected['run']
    run = RunIdentity(**{**raw_run, 'games': tuple(GameIdentity(**value) for value in raw_run['games'])})
    games = tuple(GameProjection(**{**value, 'identity': GameIdentity(**value['identity'])}) for value in projected['games'])
    return SeedEvaluationProjection(run, projected['used_decisions'], projected['conditions_sha256'], games), train_sha, eval_sha


def aggregate_trial_evaluation(store: object, loaded: LoadedHPOConfig, campaign: Json, proposal: object,
                               *, deadline: float) -> Json:
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    from gf_evaluation.hpo import EvaluationPlan, SeedEvaluationProjection, aggregate_evaluation
    if type(store) is not CampaignStore or type(proposal) is not Proposal:
        raise ContractError('actual campaign owner and durable proposal required')
    projections = []
    training_hashes = []
    evaluation_hashes = []
    for seed in loaded.config.learning_seeds:
        projected, training_sha, evaluation_sha = verify_seed_evaluation(store, loaded, campaign, proposal, seed, deadline=deadline)
        if type(projected) is not SeedEvaluationProjection:
            raise ContractError('actual checked seed projection required')
        projections.append(projected)
        training_hashes.append(training_sha)
        evaluation_hashes.append(evaluation_sha)
    plan = EvaluationPlan(loaded.config.learning_seeds, loaded.config.evaluation_game_seeds,
        loaded.config.decision_budget, campaign['conditions_sha256'], tuple(value.run for value in projections))
    actual = aggregate_evaluation(plan, tuple(projections))
    return {'schema_version': 1, 'kind': 'hpo-evaluation', 'proposal_sha256': proposal.sha256,
        'training_receipts': training_hashes, 'evaluation_receipts': evaluation_hashes, 'value': actual.value,
        'projection_sha256': actual.projection_sha256, 'label': actual.label}


def verify_trial_evaluation(store: object, loaded: LoadedHPOConfig, campaign: Json, proposal: object, *,
                            deadline: float) -> tuple[Json, float, str]:
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    if type(store) is not CampaignStore or type(proposal) is not Proposal:
        raise ContractError('actual campaign owner and durable proposal required')
    aggregate, aggregate_sha = store.get(f'proposals/{proposal.number:06}/evaluation.json')
    _receipt(aggregate, 'hpo-evaluation', {'proposal_sha256', 'training_receipts', 'evaluation_receipts',
        'value', 'projection_sha256', 'label'})
    expected = aggregate_trial_evaluation(store, loaded, campaign, proposal, deadline=deadline)
    if canonical(aggregate) != canonical(expected):
        raise ContractError('aggregate differs from complete independently verified Source/originals')
    return aggregate, float(expected['value']), aggregate_sha


def persist_training_receipt(store: object, loaded: LoadedHPOConfig, campaign: Json, proposal: object, *,
                             training_seed: int, run_id: str, directory: Path, termination: object,
                             deadline: float, completion: Any = None) -> Json:
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import clone
    from gf_training.hpo_gpu_lease import SharedTerminationReceipt, TerminationReceipt

    expected_termination=SharedTerminationReceipt if loaded.config.schema_version==5 else TerminationReceipt
    if type(store) is not CampaignStore or type(proposal) is not Proposal or type(termination) is not expected_termination:
        raise ContractError('actual campaign/proposal/termination outputs required')
    if loaded.config.schema_version in (4,5):
        from gf_training.hpo_owned_completion import OwnedRunCompletion
        if type(completion) is not OwnedRunCompletion: raise ContractError('owned training receipt requires actual private completion')
    inspection = inspect_training_parent(directory, loaded, proposal.settings,
        training_seed=training_seed, run_id=run_id, deadline=deadline,completion=completion)
    _parent_conditions(inspection, campaign)
    value = clone(asdict(termination))
    verify_gpu_termination(value, campaign_id=campaign['campaign_id'], operation_id=run_id, run_id=run_id, evaluation_id=None)
    folder = f'proposals/{proposal.number:06}/seeds/{training_seed}'
    gpu_sha = store.put(folder + '/training-gpu-termination.json', value)
    source_sha = store.put(folder + '/source-validation.json', inspection.source)
    source = inspection.source
    receipt = {'schema_version': 1, 'kind': 'hpo-seed-training', 'proposal_sha256': proposal.sha256,
        **{key: source[key] for key in ('training_seed', 'run_path', 'run_id', 'config_sha256', 'state_sha256',
            'generation', 'generation_sha256', 'inference_sha256', 'used_decisions')},
        'source_receipt_sha256': source_sha, 'gpu_termination_sha256': gpu_sha}
    store.put(folder + '/training.json', receipt)
    return receipt


def verify_training_receipt(store: object, loaded: LoadedHPOConfig, campaign: Json, proposal: object,
                            training_seed: int, *, deadline: float) -> Json:
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    if type(store) is not CampaignStore or type(proposal) is not Proposal:
        raise ContractError('actual campaign/proposal required')
    folder = f'proposals/{proposal.number:06}/seeds/{training_seed}'
    value, _ = store.get(folder + '/training.json')
    _receipt(value, 'hpo-seed-training', _TRAINING_RECEIPT_FIELDS)
    if value['proposal_sha256'] != proposal.sha256 or value['training_seed'] != training_seed:
        raise ContractError('closed seed receipt differs from exact proposal')
    actual = inspect_training_parent(Path(value['run_path']), loaded, proposal.settings,
        training_seed=training_seed, run_id=value['run_id'], deadline=deadline)
    _parent_conditions(actual, campaign)
    source, source_sha = store.get(folder + '/source-validation.json')
    if canonical(source) != canonical(actual.source) or value['source_receipt_sha256'] != source_sha:
        raise ContractError('closed training receipt no longer matches full original Source')
    gpu_value, gpu_sha = store.get(folder + '/training-gpu-termination.json')
    if gpu_value.get('schema_version')!=(2 if loaded.config.schema_version==5 else 1):
        raise ContractError('training GPU termination version differs from explicit campaign mode')
    if gpu_sha != value['gpu_termination_sha256']:
        raise ContractError('training termination receipt hash differs')
    verify_gpu_termination(gpu_value, campaign_id=campaign['campaign_id'], operation_id=value['run_id'],
        run_id=value['run_id'], evaluation_id=None)
    for key in ('training_seed', 'run_path', 'run_id', 'config_sha256', 'state_sha256', 'generation',
                'generation_sha256', 'inference_sha256', 'used_decisions'):
        if value[key] != source[key]:
            raise ContractError('training receipt changed its original final-D identity')
    return value


def _release_evaluation_pins(pins: list[tuple[Path, HpoReferenceReceipt]],
                             attempts: dict[int, HpoReferenceReceipt | BaseException], *,
                             max_owned_bytes: int, deadline: float, primary: BaseException | None = None,
                             observation: _HpoEvaluationCleanupObservation | None = None) -> None:
    """Attempt every original pin once, preserving uncertain transaction owners."""
    if observation is not None and observation._resident_runtime is not None:
        try:
            observation.require_resident_resources_retired()
        except BaseException as cleanup:
            if primary is not None:
                vars(primary)['hpo_reference_cleanup'] = (pins,attempts)
                primary.add_note('resident evaluation pins retained: '+str(cleanup))
                return
            raise
    retention = owned_runtime_module('gf_training.hpo_retention', ('release_hpo_evaluation',))
    first = primary
    for directory, acquired in reversed(pins):
        if id(acquired) in attempts:
            continue
        try:
            attempts[id(acquired)] = retention.release_hpo_evaluation(directory, acquired.binding,
                max_owned_bytes=max_owned_bytes, deadline=deadline)
        except BaseException as cleanup:  # noqa: BLE001 -- comparison must attempt BOTH real parent transactions.
            attempts[id(acquired)] = cleanup
            if first is None:
                first = cleanup
            else:
                first.add_note('evaluation pin cleanup: ' + str(cleanup))
    if first is not None:
        vars(first)['hpo_reference_cleanup'] = (pins, attempts)
        if primary is None:
            raise first


def _confirmed_evaluation_cleanup(error: BaseException, observation: Any, job: GpuJob,
                                   pins: Any, releases: Any) -> Any:
    """The SAME original proof, observed while its actual job is still held."""
    from gf_training.hpo_retention import HpoReferenceReceipt
    if type(observation) is not _HpoEvaluationCleanupObservation or observation._identity[1] is not job:
        raise ContractError('positive original evaluation cleanup observation unavailable')
    observation.require_outer_retired(error=error)
    if not pins or len(releases) != len(pins):
        raise ContractError('every original acquired evaluation pin needs its actual release receipt')
    for _, acquired in pins:
        released = releases.get(id(acquired))
        if (type(acquired) is not HpoReferenceReceipt or acquired.phase != 'acquire'
                or acquired.binding['token'] != job.ticket or type(released) is not HpoReferenceReceipt
                or released.phase != 'release' or canonical(released.binding) != canonical(acquired.binding)):
            raise ContractError('evaluation pin retirement lacks exact typed release evidence')
    receipt = (job.confirm_termination() if job._receipt is None else
        job.require_confirmed_termination(job._receipt))
    return receipt


def _copy_failed_evaluation_receipt(error: BaseException, store: Any, name: str,
                                    evaluation_id: str, receipt: Any) -> None:
    """Copy already-issued actual evidence; never reopen/confirm a closed job."""
    vars(error)['hpo_gpu_termination_receipt']=receipt
    try:
        store.put(name,{'schema_version':1,'kind':'hpo-failed-evaluation-termination',
            'evaluation_id':evaluation_id,'gpu_termination':asdict(receipt),'error_type':type(error).__name__,
            'evaluation_eligible':False})
    except BaseException as copy_error:  # noqa: BLE001 - the actual global receipt survives local failure
        vars(error)['hpo_failed_termination_copy_error']=copy_error
        error.add_note('global GPU termination confirmed; failed-operation local record failed: '+str(copy_error))


def _confirm_failed_evaluation(error: BaseException, store: Any, name: str,
                               observation: _HpoEvaluationCleanupObservation | None, job: GpuJob,
                               pins: list[tuple[Path, HpoReferenceReceipt]],
                               releases: dict[int, HpoReferenceReceipt | BaseException]) -> None:
    """Operational termination only; the caller always rethrows its data failure."""
    vars(error)['hpo_evaluation_cleanup_observation']=observation
    try: receipt=_confirmed_evaluation_cleanup(error,observation,job,pins,releases)
    except BaseException as cleanup:  # noqa: BLE001 - never mask the data failure or invent proof
        vars(error)['hpo_termination_cleanup_error']=cleanup
        error.add_note('evaluation resource termination unconfirmed: '+str(cleanup))
        return
    assert observation is not None
    _copy_failed_evaluation_receipt(error,store,name,observation._identity[0].linkage['evaluation_id'],receipt)


async def evaluate_seed(store: object, loaded: LoadedHPOConfig, campaign: Json, proposal: object,
                        training: Json, *, output: Path, progress: Any = None, completion: Any = None) -> Json:
    """One actual primary evaluation; no interrupted request is silently rerun."""
    require_hpo_runtime_execution(loaded.config)
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import clone
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job

    if type(store) is not CampaignStore or type(proposal) is not Proposal:
        raise ContractError('actual campaign/proposal required')
    core = owned_runtime_module('gf_cli.card_instance', ('collect_hpo_evaluation', 'hpo_artifact_budget'))
    composition = owned_runtime_module('gf_cli.hpo_config', ('materialize_hpo_run',))
    retention = owned_runtime_module('gf_training.hpo_retention', ('pin_hpo_evaluation', 'release_hpo_evaluation'))
    seed = training['training_seed']
    folder = f'proposals/{proposal.number:06}/seeds/{seed}'
    if store.exists(folder + '/evaluation.intent.json'):
        raise ContractError('interrupted evaluation cannot be restarted or requalified')
    import uuid
    evaluation_id = str(uuid.uuid4())
    base = composition.materialize_hpo_run(loaded, proposal.settings, learning_seed=seed, run_id=training['run_id'])
    run = replace(base.config.run, split='validation', name=evaluation_id, output_dir=str(output.parent),
        seeds=loaded.config.evaluation_game_seeds, sampling_seeds=(loaded.config.sampler_seed,), pairs_per_seed=1)
    config = replace(base.config, run=run)
    raw = clone(asdict(config))
    projected = LoadedConfig(config, raw, digest(raw), base.inputs)
    started = time.monotonic()
    store.put(folder + '/evaluation.intent.json', {'schema_version': 1, 'kind': 'hpo-evaluation-intent',
        'evaluation_id': evaluation_id, 'directory': str(output), 'config_sha256': projected.sha256,
        'training_receipt_sha256': store.get(folder + '/training.json')[1],
        'started_unix': time.time(), 'started_monotonic': started})
    job = admit_gpu_job(JobContext(campaign['campaign_id'], evaluation_id, 'evaluation', training['run_id'], evaluation_id),
        global_cuda_bytes=loaded.config.operating.managed_cuda_bytes, max_workers=loaded.config.operating.max_participant_workers)
    pins: list[tuple[Path, HpoReferenceReceipt]] = []
    releases: dict[int, HpoReferenceReceipt | BaseException] = {}
    observation = None
    owner = None
    final_data = None
    drain_started=False
    termination=None
    try:
        store.put(folder + '/evaluation.admission.json', {'schema_version': 1, 'kind': 'hpo-evaluation-admission',
            'gpu_ticket': job.ticket, 'evaluation_id': evaluation_id})
        wall = started + run.wall_seconds
        work = wall - run.report_reserve_seconds
        execution = started + run.execution_seconds
        reference = retention.pin_hpo_evaluation(Path(training['run_path']), job.ticket, training['generation'], training['inference_sha256'],
            max_owned_bytes=loaded.config.operating.source_budget_bytes, deadline=execution)
        pins.append((Path(training['run_path']), reference))
        if loaded.config.schema_version in (4,5):
            from gf_common.artifacts import ArtifactPhaseAdmission
            from gf_common.group_pack import StorageSettings
            from gf_training.group_pack_resources import StorageBudget
            from gf_training.hpo_owned_completion import OwnedRunCompletion, retain_for_evaluation
            evaluation_budget=StorageBudget(StorageSettings(budget_bytes=loaded.config.operating.source_budget_bytes),work,
                name='source',phase='hpo-evaluation')
            evaluation_phase=ArtifactPhaseAdmission(evaluation_budget,loaded.config.operating.parent_budget_bytes,
                name='parent',phase='hpo-evaluation')
            if completion is not None:
                if type(completion) is not OwnedRunCompletion: raise ContractError('actual private completion required')
                final_data=retain_for_evaluation(completion,budget=evaluation_budget,parent_budget=evaluation_phase)
            else:
                from gf_training.hpo_owned_restore import import_owned_final_policy
                final_data=import_owned_final_policy(Path(training['run_path']),deadline=execution,
                    max_owned_bytes=loaded.config.operating.source_budget_bytes,parent_budget_bytes=loaded.config.operating.parent_budget_bytes,
                    budget=evaluation_budget,parent_budget=evaluation_phase)
        elif completion is not None: raise ContractError('legacy evaluation cannot adopt owned completion')
        try:
            inspected=(_checked_owned_inspection(completion_inspection(final_data,Path(training['run_path'])),loaded,proposal.settings,
                training_seed=seed,run_id=training['run_id']) if final_data is not None else
                inspect_training_parent(Path(training['run_path']),loaded,proposal.settings,training_seed=seed,run_id=training['run_id'],deadline=execution))
            parent_reference, binding = _final_policy_reference(inspected, training['source_receipt_sha256'])
            label = parent_reference['policy_label']
            bindings = {label: binding, 'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')}
            link = {'schema_version': 1, 'kind': 'hpo-evaluation-linkage', 'campaign_id': campaign['campaign_id'],
                'evaluation_id': evaluation_id, 'gpu_ticket': job.ticket, 'purpose': 'objective',
                'conditions_sha256': campaign['conditions_sha256'], 'left_parent': parent_reference,
                'right_parent': None, 'game_seeds': list(loaded.config.evaluation_game_seeds),
                'learner_seats': {}, 'schedule_digest': '0' * 64,
                'policy_registry_digest': digest({key: asdict(value) for key, value in bindings.items()})}
            groups = hpo_evaluation_groups(link)
            schedule = clone(serialize_schedule(groups, 4))
            link['learner_seats'] = {spec['episode_id']: spec['learner_viewer'] for group in schedule for spec in group['episodes']}
            link['schedule_digest'] = digest(schedule)
            resident_runtime = evaluation_runtime(loaded.config)
            plan = execution_plan(groups, bindings, output, link,runtime=resident_runtime,pins=tuple(pins), started_monotonic=started,
                admission_deadline=started + run.admission_seconds, execution_deadline=execution,
                work_deadline=work, wall_deadline=wall)
            if final_data is not None:
                from gf_training.hpo_owned_completion import completion_artifacts
                if canonical(completion_inspection(final_data,Path(training['run_path'])).source)!=canonical(inspected.source):
                    raise ContractError('owned final data differs from validated training receipt')
                owner=completion_artifacts(final_data,binding,label)
            else:
                owner = RetainedGameArtifacts.for_hpo_registry(bindings, config.actions, rules=card_instance_rules(),
                    pool_size=config.hpo_spec['operating']['opponent_pool_size'], max_owned_bytes=core.hpo_artifact_budget(projected, bindings),
                    preflight_deadline=execution, finalization_deadline=work)
            observation = _HpoEvaluationCleanupObservation(plan, job, owner)
            try:
                result = await core.collect_hpo_evaluation(projected, plan, gpu_job=job, artifact_inputs=owner,
                    _cleanup_observation=observation,**({} if progress is None else {'progress':progress}))
                if result.report.get('healthy') is not True:
                    raise ContractError('primary evaluation did not finish/save/clean up normally')
                observe(progress, {'phase':'source'})
                manifest, proof = verify_saved_evaluation(result.directory, loaded, link, deadline=work,
                    extra_owned_bytes=owner.retained_bytes if final_data is not None else owner._scan.peak, _cleanup_observation=observation,
                    evaluation_runtime=evaluation_runtime(loaded.config))
                owner.finalize(deadline=work)
            except BaseException as error:
                try:
                    observation.release_artifacts()
                except BaseException as cleanup:  # noqa: BLE001 - preserve original and reachable actual owner
                    error.__dict__['hpo_artifact_cleanup_owner'] = owner
                    error.add_note('HPO evaluation artifact cleanup: ' + type(cleanup).__name__)
                raise
            else:
                observation.release_artifacts()
        except BaseException as error:
            _release_evaluation_pins(pins, releases, max_owned_bytes=loaded.config.operating.source_budget_bytes,
                deadline=work, primary=error, observation=observation)
            raise
        else:
            _release_evaluation_pins(pins, releases, max_owned_bytes=loaded.config.operating.source_budget_bytes, deadline=work)
            released = releases[id(reference)]
            if isinstance(released, BaseException):
                raise released
        observe(progress, {'phase':'cleanup'})
        termination = _confirmed_evaluation_cleanup(ContractError('final evaluation cleanup proof'),observation,job,pins,releases)
        gpu_sha = store.put(folder + '/evaluation-gpu-termination.json', clone(asdict(termination)))
        receipt = {'schema_version': 1, 'kind': 'hpo-seed-evaluation',
            'training_receipt_sha256': store.get(folder + '/training.json')[1], 'linkage_sha256': digest(link),
            'manifest_sha256': proof['manifest_sha256'], 'originals_digest': proof['source_set_sha256'],
            'projection': _projection(manifest, training, training['source_receipt_sha256'], campaign['conditions_sha256']),
            'conditions_sha256': campaign['conditions_sha256'], 'pin_acquire_sha256': reference.receipt_sha256,
            'pin_release_sha256': released.receipt_sha256, 'gpu_termination_sha256': gpu_sha, 'directory': str(result.directory),
            'linkage': link, 'config_sha256': projected.sha256}
        receipt.update(evaluation_receipt_runtime_fields(manifest))
        check_evaluation_receipt_runtime(receipt,manifest)
        drain_started=True
        _drain_final_owners(job,() if final_data is None else (final_data,))
        store.put(folder + '/evaluation.json', receipt)
        return receipt
    except BaseException as error:
        if not drain_started:
            if owner is not None:
                try:
                    if observation is not None:
                        observation.release_artifacts()
                    else:
                        owner.release()
                except BaseException as cleanup:  # noqa: BLE001 -- actual artifact owner remains reachable.
                    vars(error)['hpo_artifact_cleanup_owner'] = owner
                    error.add_note('evaluation artifact cleanup: ' + str(cleanup))
            _release_evaluation_pins(pins, releases, max_owned_bytes=loaded.config.operating.source_budget_bytes,
                deadline=started + run.wall_seconds - run.report_reserve_seconds, primary=error, observation=observation)
            _confirm_failed_evaluation(error, store, folder + '/evaluation-failed-gpu-termination.json',
                observation, job, pins, releases)
        elif termination is not None and not getattr(error,'hpo_final_cleanup_owners',()):
            vars(error)['hpo_evaluation_cleanup_observation']=observation
            _copy_failed_evaluation_receipt(error,store,folder+'/evaluation-failed-gpu-termination.json',evaluation_id,termination)
        raise
    finally:
        if not drain_started:
            import sys
            drain_started=True
            _drain_final_owners(job,() if final_data is None else (final_data,),primary=sys.exception())


_RUNTIME_OMITTED = object()


def comparison_execution_runtime(left: Any,right: Any,*,choice: Any = _RUNTIME_OMITTED) -> Json | None:
    if choice is _RUNTIME_OMITTED:
        if getattr(left,'schema_version',None)==5 and getattr(right,'schema_version',None)==5:return None
        if hasattr(left,'runtime') or hasattr(right,'runtime'):
            raise ContractError('resident producer requires an explicit evaluation runtime table or explicit None for fresh')
        return None
    if choice is None:
        return None
    from gf_common.hpo_resident import resident_runtime
    return resident_runtime(choice)


def _drain_final_owners(job: Any, policies: Any, *, primary: BaseException | None=None) -> None:
    """Attempt every actual independent owner ONCE; never replace the primary."""
    failures=[]
    seen=set()
    owners=([('evaluation job',job)] if job is not None else [])+[("final inference",data) for data in policies]
    for role,owner in owners:
        if id(owner) in seen: continue
        seen.add(id(owner))
        try: owner.close()
        except BaseException as cleanup:  # noqa: BLE001 - every actual remaining owner must be attempted
            failures.append((role,owner,cleanup))
    if not failures: return
    first=primary if primary is not None else failures[0][2]
    previous=getattr(first,'hpo_final_cleanup_owners',())
    unresolved=list(previous)
    for role,owner,failure in failures:
        if not any(item is owner for item in unresolved): unresolved.append(owner)
        first.add_note(role+' cleanup remains owned: '+str(failure))
    vars(first)['hpo_final_cleanup_owners']=tuple(unresolved)
    vars(first)['hpo_final_cleanup_errors']=getattr(first,'hpo_final_cleanup_errors',())+tuple(failures)
    if primary is None: raise first


def _comparison_artifacts(left: Any, right: Any, sources: list[Any], bindings: Any, config: Any,
                          *, deadline: float, work: float, max_owned_bytes: int) -> tuple[Any,list[Any]]:
    """Profile dispatch, never rejection-triggered fallback or origin promotion."""
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    if left.config.schema_version not in (4,5) or right.config.schema_version not in (4,5):
        owner=RetainedGameArtifacts.for_hpo_registry(bindings,config.actions,rules=card_instance_rules(),
            pool_size=config.hpo_spec['operating']['opponent_pool_size'],max_owned_bytes=max_owned_bytes,
            preflight_deadline=deadline,finalization_deadline=work)
        return owner,[]
    from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
    from gf_common.group_pack import StorageSettings
    from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_restore import import_owned_final_policy
    from gf_training.structured_lineage import _stat_identity
    budget=StorageBudget(StorageSettings(budget_bytes=left.config.operating.source_budget_bytes),work,
        name='source',phase='hpo-comparison')
    phase=ArtifactPhaseAdmission(budget,left.config.operating.parent_budget_bytes,
        name='parent',phase='hpo-comparison')
    imported: dict[Path,Any]={}
    entries={}
    try:
        with ExitStack() as preparation:
            try:
                for directory,label,binding in sources:
                    if directory not in imported:
                        imported[directory]=import_owned_final_policy(directory,deadline=deadline,
                            max_owned_bytes=budget.limit,parent_budget_bytes=phase.limit,budget=budget,parent_budget=phase)
                    data=imported[directory]
                    if binding!=bindings[label]: continue
                    # Immutable retained-byte bounds, before getters/cap issuance,
                    # list/stat/path projections or simultaneous provider copies.
                    allowance=32*(len(data._files)+len(data._manifest)+len(data._actor_metadata))
                    allowance+=32*json_bound(binding.policy_options)+65536
                    preparation.enter_context(phase.reserve(allowance))
                    files=[{'path':name,'sha256':sha,'identity':list(_stat_identity((directory/'artifacts'/f'{data.generation:06}'/name).lstat()))}
                        for name,sha in data.files.items()]
                    entries[label]=(data.actor_capability,binding,files,data.inference_manifest)
                return OwnedGameArtifacts(entries,budget=budget,parent_budget=phase),list(imported.values())
            except BaseException:
                entries.clear()  # Drop preparation aliases before their leases.
                files=None
                raise
    except BaseException as error:
        _drain_final_owners(None,imported.values(),primary=error)
        raise


async def evaluate_comparison(store: object, left: LoadedHPOConfig, right: LoadedHPOConfig,
                              left_proposal: object, right_proposal: object, left_training: Json,
                              right_training: Json, *, comparison_id: str, conditions_sha256: str,
                              output: Path, evaluation_runtime: Any = _RUNTIME_OMITTED, progress: Any = None) -> ComparisonPanel:
    """One complete declared seed-pair matchup through the actual shared collector."""
    evaluation_runtime = comparison_execution_runtime(left.config,right.config,choice=evaluation_runtime)
    require_hpo_runtime_execution(left.config)
    require_hpo_runtime_execution(right.config)
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import clone
    from gf_evaluation.hpo import (
        GameIdentity,
        GameProjection,
        RunIdentity,
        SeedEvaluationProjection,
    )
    from gf_evaluation.hpo_compare import ComparisonPanel
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job

    if type(store) is not CampaignStore or type(left_proposal) is not Proposal or type(right_proposal) is not Proposal:
        raise ContractError('actual comparison store and closed normalized trial proposals required')
    core = owned_runtime_module('gf_cli.card_instance', ('collect_hpo_evaluation', 'hpo_artifact_budget'))
    composition = owned_runtime_module('gf_cli.hpo_config', ('materialize_hpo_run',))
    retention = owned_runtime_module('gf_training.hpo_retention', ('pin_hpo_evaluation', 'release_hpo_evaluation'))
    import uuid
    evaluation_id = str(uuid.uuid4())
    folder = f'matchups/{left_training["training_seed"]}-{right_training["training_seed"]}'
    base = composition.materialize_hpo_run(left, left_proposal.settings,
        learning_seed=left_training['training_seed'], run_id=left_training['run_id'])
    run = replace(base.config.run, split='validation', name=evaluation_id, output_dir=str(output.parent),
        seeds=left.config.evaluation_game_seeds, sampling_seeds=(left.config.sampler_seed,), pairs_per_seed=1)
    config = replace(base.config, run=run)
    raw = clone(asdict(config))
    projected = LoadedConfig(config, raw, digest(raw), base.inputs)
    started = time.monotonic()
    wall, execution = started + run.wall_seconds, started + run.execution_seconds
    work = wall - run.report_reserve_seconds
    store.put(folder + '/intent.json', {'schema_version': 1, 'kind': 'hpo-comparison-matchup-intent',
        'evaluation_id': evaluation_id, 'config_sha256': projected.sha256, 'directory': str(output),
        'left_source_receipt_sha256': left_training['source_receipt_sha256'],
        'right_source_receipt_sha256': right_training['source_receipt_sha256']})
    job = admit_gpu_job(JobContext(comparison_id, evaluation_id, 'evaluation', left_training['run_id'], evaluation_id),
        global_cuda_bytes=left.config.operating.managed_cuda_bytes, max_workers=left.config.operating.max_participant_workers)
    pins: list[tuple[Path, HpoReferenceReceipt]] = []
    releases: dict[int, HpoReferenceReceipt | BaseException] = {}
    observation = None
    owner = None
    final_inputs: list[Any]=[]
    drain_started=False
    termination=None
    try:
        store.put(folder + '/admission.json', {'schema_version': 1, 'kind': 'hpo-comparison-matchup-admission',
            'gpu_ticket': job.ticket, 'evaluation_id': evaluation_id})
        try:
            inspected = []
            parent_refs = []
            bindings = {}
            sources=[]
            for loaded, proposal, training in ((left, left_proposal, left_training), (right, right_proposal, right_training)):
                directory = Path(training['run_path']).resolve(strict=True)
                if not any(existing[0] == directory for existing in pins):
                    acquired = retention.pin_hpo_evaluation(directory, job.ticket, training['generation'], training['inference_sha256'],
                        max_owned_bytes=left.config.operating.source_budget_bytes, deadline=execution)
                    pins.append((directory, acquired))
                parent = inspect_training_parent(directory, loaded, proposal.settings, training_seed=training['training_seed'],
                    run_id=training['run_id'], deadline=execution)
                inspected.append(parent)
                parent_reference, binding = _final_policy_reference(parent, training['source_receipt_sha256'])
                label = parent_reference['policy_label']
                bindings[label] = binding
                sources.append((directory,label,binding))
                parent_refs.append(parent_reference)
            if any(parent.source['used_decisions'] != left.config.decision_budget for parent in inspected):
                raise ContractError('comparison requires both exact final-D parents')
            link = {'schema_version': 1, 'kind': 'hpo-evaluation-linkage', 'campaign_id': comparison_id,
                'evaluation_id': evaluation_id, 'gpu_ticket': job.ticket, 'purpose': 'comparison', 'conditions_sha256': conditions_sha256,
                'left_parent': parent_refs[0], 'right_parent': parent_refs[1], 'game_seeds': list(left.config.evaluation_game_seeds),
                'learner_seats': {}, 'schedule_digest': '0' * 64,
                'policy_registry_digest': digest({label: asdict(value) for label, value in bindings.items()})}
            groups = hpo_evaluation_groups(link)
            schedule = clone(serialize_schedule(groups, 4))
            link.update(learner_seats={spec['episode_id']: spec['learner_viewer'] for group in schedule for spec in group['episodes']},
                schedule_digest=digest(schedule))
            plan = execution_plan(groups, bindings, output, link,runtime=evaluation_runtime,pins=tuple(pins), started_monotonic=started,
                admission_deadline=started + run.admission_seconds, execution_deadline=execution, work_deadline=work, wall_deadline=wall)
            owner,final_inputs=_comparison_artifacts(left,right,sources,bindings,config,deadline=execution,work=work,
                max_owned_bytes=core.hpo_artifact_budget(projected,bindings))
            observation = _HpoEvaluationCleanupObservation(plan, job, owner)
            try:
                result = await core.collect_hpo_evaluation(projected, plan, gpu_job=job, artifact_inputs=owner,
                    _cleanup_observation=observation,**({} if progress is None else {'progress':progress}))
                if result.report.get('healthy') is not True:
                    raise ContractError('comparison did not finish/save/clean up normally')
                observe(progress, {'phase':'source'})
                manifest, proof = verify_saved_evaluation(result.directory, left, link, deadline=work, extra_owned_bytes=owner.retained_bytes if final_inputs else owner._scan.peak, _cleanup_observation=observation,evaluation_runtime=evaluation_runtime)
                owner.finalize(deadline=work)
            except BaseException as error:
                try:
                    observation.release_artifacts()
                except BaseException as cleanup:  # noqa: BLE001 - preserve original and actual custody owner
                    error.__dict__['hpo_artifact_cleanup_owner'] = owner
                    error.add_note('comparison artifact cleanup: ' + type(cleanup).__name__)
                raise
            else:
                observation.release_artifacts()
        except BaseException as error:
            _release_evaluation_pins(pins, releases, max_owned_bytes=left.config.operating.source_budget_bytes,
                deadline=work, primary=error, observation=observation)
            raise
        else:
            _release_evaluation_pins(pins, releases, max_owned_bytes=left.config.operating.source_budget_bytes, deadline=work)
        released = [value for value in releases.values() if not isinstance(value, BaseException)]
        observe(progress, {'phase':'cleanup'})
        termination = _confirmed_evaluation_cleanup(ContractError('final evaluation cleanup proof'),observation,job,pins,releases)
        gpu_sha = store.put(folder + '/gpu-termination.json', clone(asdict(termination)))
        projection = _projection(manifest, left_training, left_training['source_receipt_sha256'], conditions_sha256)
        run_value = projection['run']
        run_identity = RunIdentity(**{**run_value, 'games': tuple(GameIdentity(**value) for value in run_value['games'])})
        games = tuple(GameProjection(**{**value, 'identity': GameIdentity(**value['identity'])}) for value in projection['games'])
        seed_projection = SeedEvaluationProjection(run_identity, left_training['used_decisions'], conditions_sha256, games)
        panel = ComparisonPanel(left_training['training_seed'], right_training['training_seed'], right_training['run_id'],
            right_training['inference_sha256'], right_training['source_receipt_sha256'], right_training['used_decisions'], seed_projection)
        drain_started=True
        _drain_final_owners(job,final_inputs)
        store.put(folder + '/receipt.json', {**evaluation_receipt_runtime_fields(manifest), 'kind': 'hpo-comparison-matchup',
            'linkage': link, 'manifest_sha256': proof['manifest_sha256'], 'originals_digest': proof['source_set_sha256'],
            'pin_acquire_receipts': [value.receipt_sha256 for _, value in pins],
            'pin_release_receipts': [value.receipt_sha256 for value in released], 'gpu_termination_sha256': gpu_sha,
            'panel': asdict(panel), 'left_profile': left.effective, 'right_profile': right.effective})
        return panel
    except BaseException as error:
        if not drain_started:
            if owner is not None:
                try:
                    if observation is not None:
                        observation.release_artifacts()
                    else:
                        owner.release()
                except BaseException as cleanup:  # noqa: BLE001 -- actual artifact owner remains reachable.
                    vars(error)['hpo_artifact_cleanup_owner'] = owner
                    error.add_note('comparison artifact cleanup: ' + str(cleanup))
            _release_evaluation_pins(pins, releases, max_owned_bytes=left.config.operating.source_budget_bytes,
                deadline=work, primary=error, observation=observation)
            _confirm_failed_evaluation(error, store, folder + '/failed-gpu-termination.json',
                observation, job, pins, releases)
        elif termination is not None and not getattr(error,'hpo_final_cleanup_owners',()):
            vars(error)['hpo_evaluation_cleanup_observation']=observation
            _copy_failed_evaluation_receipt(error,store,folder+'/failed-gpu-termination.json',evaluation_id,termination)
        raise
    finally:
        if not drain_started:
            import sys
            drain_started=True
            _drain_final_owners(job,final_inputs,primary=sys.exception())
