"""Strict schema6 closed-seed constructors/readers at the actual CLI owner edge."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from gf_common.domain import ContractError, Json, canonical, digest, integer
from gf_common.rust_player import hex_identity
from gf_training.rust_player_evaluation import (
    _COMPLETED_AUDIT_ISSUER,
    RustCompletedParentAudit,
    validate_parent_binding,
    validate_parent_metadata,
    validate_seed_binding,
    validate_seed_training,
)

if TYPE_CHECKING:
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_config import LoadedHPOConfig
    from gf_cli.hpo_storage import CampaignStore
    from gf_training.rust_player_completion import ColdFinalizedRustParent
    from gf_training.rust_player_supervision import HostControl


def historical_record_root(bootstrap: Json, physical: Json) -> Path:
    """Pure path DATA; caller supplies the actual current authenticated bootstrap."""
    if (type(physical.get('schema_version')) is not int or physical['schema_version'] != 3
            or physical.get('kind') != 'rust-player-gpu-termination-v1'):
        raise ContractError('explicit Rust physical termination3 data required')
    hex_identity(physical['ticket'], 32)
    supervision = hex_identity(physical['exit_record']['supervision_id'], 32)
    return Path(bootstrap['campaign_directory']) / 'supervision' / supervision


def _current_context(store: Any, loaded: Any, campaign: Json, proposal: Any, seed: int, control: Any,
                     audit: RustCompletedParentAudit | None = None) -> Json:
    from gf_cli.hpo_campaign import CampaignStudy, Proposal
    from gf_cli.hpo_config import LoadedHPOConfig
    from gf_cli.hpo_storage import CampaignStore
    from gf_training.rust_player_supervision import HostControl
    if (type(store) is not CampaignStore or type(loaded) is not LoadedHPOConfig
            or type(proposal) is not Proposal or type(control) is not HostControl):
        raise ContractError('actual current campaign store/config/proposal/control required')
    store._require()
    control._require()
    if store._control is not control or loaded.config.schema_version != 6:
        raise ContractError('actual SAME authenticated campaign store and explicit schema6 required')
    bootstrap = control.campaign_bootstrap
    integer(seed, 0)
    integer(proposal.number, 0)
    if seed not in loaded.config.learning_seeds or Path(bootstrap['campaign_directory']) != store.directory:
        raise ContractError('closed Rust seed lies outside actual campaign/seed namespace')
    study = CampaignStudy(store, loaded, campaign)
    intent, _ = study._intent(proposal.number)
    if study._proposal(proposal.number, intent) != proposal:
        raise ContractError('actual durable proposal differs from current campaign settings/history')
    if audit is None:
        recorded, _ = store.get('campaign.json')
        saved, saved_sha = store.get(f'proposals/{proposal.number:06}/proposal.json')
    else:
        recorded = audit.read_data(store._path('campaign.json'), protocol=False)
        proposal_path = store._path(f'proposals/{proposal.number:06}/proposal.json')
        saved = audit.read_data(proposal_path, protocol=False)
        saved_sha = audit._header_shas[proposal_path]
    if (canonical(recorded) != canonical(campaign) or campaign['loaded_config_sha256'] != loaded.sha256
            or canonical(campaign['effective_config']) != canonical(loaded.effective)
            or saved_sha != proposal.sha256 or saved['effective_settings_sha256'] != proposal.settings.identity
            or canonical(saved['effective_model']) != canonical(asdict(proposal.settings.model))
            or canonical(saved['effective_training']) != canonical(asdict(proposal.settings.training))):
        raise ContractError('closed Rust seed differs from actual campaign/proposal materialization')
    return bootstrap


def _materialized(loaded: Any, campaign: Json, proposal: Any, seed: int, state: Json,
                  metadata: Json, directory: Path) -> Any:
    from gf_cli.hpo_config import materialize_hpo_run
    from gf_cli.hpo_evaluation import ParentInspection, _parent_conditions
    actual = materialize_hpo_run(loaded, proposal.settings, learning_seed=seed, run_id=state['run_id'])
    validate_parent_binding(metadata, state, run_path=str(directory))
    if (directory != Path(loaded.config.storage.parent_root) / state['run_id']
            or state['seed'] != seed or state['config_sha256'] != actual.sha256
            or canonical(state['effective_config']) != canonical(actual.effective)
            or state['effective_config']['hpo_spec']['effective_settings_sha256'] != proposal.settings.identity):
        raise ContractError('Rust parent differs from exact campaign/settings/seed/final-D materialization')
    _parent_conditions(ParentInspection(state, (), metadata), campaign)
    return actual


def _awaiting(audit: RustCompletedParentAudit, campaign: Json, proposal: Any, seed: int,
              operation: Json, operation_path: Path, operation_sha: str, clock_sha: str) -> Json:
    from gf_common.rust_player import safe_component, validate_campaign_awaiting
    path = audit.record_root / 'awaiting' / (safe_component(operation['operation_id']) + '.json')
    value = validate_campaign_awaiting(audit.read_data(path))
    if (value['campaign_id'] != campaign['campaign_id'] or value['proposal_no'] != proposal.number
            or value['learning_seed'] != seed or value['supervision_id'] != audit.record_root.name
            or value['operation_path'] != str(operation_path) or value['operation_sha256'] != operation_sha
            or value['session_clock_sha256'] != clock_sha
            or any(value[key] != operation[key] for key in ('run_id', 'config_path', 'config_sha256', 'runtime_binding_sha256'))
            or value['run_path'] != operation['parent_directory']):
        raise ContractError('closed seed differs from its actual original awaiting operation')
    return value


def persist_rust_training_receipt(store: Any, loaded: Any, campaign: Json, proposal: Any, *,
                                  training_seed: int, run_id: str, directory: Path, termination: Any,
                                  deadline: float, completion: Any) -> Json:
    from gf_training.group_pack_resources import remaining
    from gf_training.rust_player_completion import ColdFinalizedRustParent
    from gf_training.rust_player_supervision import RustPhysicalTermination, read_control_record
    if type(completion) is not ColdFinalizedRustParent or type(termination) is not RustPhysicalTermination:
        raise ContractError('actual qualified Rust Parent and typed physical3 required')
    completion.require()
    control = completion._invocation.cold._control
    _current_context(store, loaded, campaign, proposal, training_seed, control)
    if termination._cold is not completion._invocation.cold:
        raise ContractError('actual SAME Parent/physical original pair required')
    remaining(min(deadline, completion._invocation.pool.deadline))
    state, metadata = completion.state, validate_parent_metadata(completion.metadata)
    if state['run_id'] != run_id or directory != completion._invocation._parent_directory:
        raise ContractError('actual Rust Parent run/path differs')
    _materialized(loaded, campaign, proposal, training_seed, state, metadata, directory)
    physical = termination.global_observation()
    admission, prepared, exit_record = completion._invocation.cold.require_pair()
    if (physical['exit_record'] != exit_record or physical['context_sha256'] != digest(admission['context'])
            or admission['context']['campaign_id'] != campaign['campaign_id']
            or metadata['completion_receipt_sha256'] != completion._invocation.completion_sha256
            or metadata['prepared_sha256'] != digest(prepared) or metadata['exit_sha256'] != digest(exit_record)):
        raise ContractError('actual Parent/physical/campaign completion chain differs')
    from gf_common.rust_player import (
        safe_component,
        validate_campaign_awaiting,
        validate_campaign_clock,
    )
    clock, clock_sha = read_control_record(control.root / 'session-clock.json', control.root)
    validate_campaign_clock(clock)
    operation, operation_sha = read_control_record(control.operation_path, control.root)
    awaiting, _ = read_control_record(control.root / 'awaiting' / (safe_component(operation['operation_id']) + '.json'), control.root)
    validate_campaign_awaiting(awaiting)
    if (awaiting['campaign_id'] != campaign['campaign_id'] or awaiting['proposal_no'] != proposal.number
            or awaiting['learning_seed'] != training_seed or awaiting['run_id'] != run_id
            or awaiting['operation_path'] != str(control.operation_path) or awaiting['operation_sha256'] != operation_sha
            or awaiting['session_clock_sha256'] != clock_sha or awaiting['supervision_id'] != control.root.name
            or awaiting['run_path'] != str(directory) or awaiting['config_sha256'] != state['config_sha256']
            or awaiting['runtime_binding_sha256'] != state['runtime_binding_sha256']):
        raise ContractError('actual Parent does not match the durable proposal/seed awaiting record')
    folder = f'proposals/{proposal.number:06}/seeds/{training_seed}'
    termination_sha = store.put(folder + '/training-gpu-termination.json', physical)
    source_sha = store.put(folder + '/source-validation.json', metadata)
    receipt = {'schema_version': 2, 'kind': 'hpo-rust-seed-training', 'proposal_sha256': proposal.sha256,
        **{key: metadata[key] for key in ('training_seed', 'run_path', 'run_id', 'config_sha256', 'state_sha256',
            'generation', 'generation_sha256', 'inference_sha256', 'used_decisions', 'runtime_binding_sha256',
            'completion_receipt_sha256', 'prepared_sha256', 'exit_sha256')},
        'source_receipt_sha256': source_sha, 'gpu_termination_sha256': termination_sha}
    validate_seed_binding(receipt, metadata, source_sha256=source_sha, termination_sha256=termination_sha)
    seed_sha = store.put(folder + '/training.json', receipt)
    for name, expected, checksum in (('training-gpu-termination.json', physical, termination_sha),
            ('source-validation.json', metadata, source_sha), ('training.json', receipt, seed_sha)):
        value, actual_sha = store.get(folder + '/' + name)
        if value != expected or actual_sha != checksum:
            raise ContractError('actual closed Rust seed/metadata/physical readback differs')
    remaining(min(deadline, completion._invocation.pool.deadline))
    return receipt


def open_closed_rust_parent(store: Any, loaded: Any, campaign: Json, proposal: Any, seed: int, *,
                            control: Any, ownership: Any, pool: Any, parent_budget: Any,
                            deadline: float) -> RustCompletedParentAudit:
    """Full DATA revalidation under current actual custody, never an old Parent."""
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_config import LoadedHPOConfig
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.rust_player import (
        validate_campaign_bootstrap,
        validate_campaign_clock,
        validate_exit,
        validate_global_admission,
        validate_global_termination,
        validate_loaded_campaign,
        validate_local_admission,
        validate_operation,
        validate_prepared,
        validate_seed_settings,
    )
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.hpo_gpu_lease import _GLOBAL_ROOT
    from gf_training.lineage import require_hpo_run_owner
    from gf_training.rust_player_lineage import validate_completion, validate_run
    from gf_training.rust_player_retention import _validate_historical_audit
    from gf_training.rust_player_supervision import HostControl, runtime_producer
    if (type(store) is not CampaignStore or type(loaded) is not LoadedHPOConfig or type(proposal) is not Proposal
            or type(control) is not HostControl or type(pool) is not PackedArtifactPool
            or type(parent_budget) is not ArtifactPhaseAdmission or parent_budget.source is not pool.budget):
        raise ContractError('actual current campaign/RunOwnership/evaluation account required')
    store._require()
    control._require()
    require_hpo_run_owner(ownership.directory, ownership)
    if store._control is not control:
        raise ContractError('actual SAME current authenticated campaign store/control required')
    # Initially current campaign DATA custody only. No historical arm is opened
    # until the genuine current bootstrap and strict OLD supervision ID bind it.
    audit = RustCompletedParentAudit(_COMPLETED_AUDIT_ISSUER, control, ownership, loaded=loaded,
        pool=pool, parent_budget=parent_budget, deadline=deadline, record_root=control.root)
    try:
        bootstrap = _current_context(store, loaded, campaign, proposal, seed, control, audit)
        folder = f'proposals/{proposal.number:06}/seeds/{seed}'
        seed_path = store._path(folder + '/training.json')
        receipt = validate_seed_training(audit.read_data(seed_path, protocol=False))
        metadata_path = store._path(folder + '/source-validation.json')
        metadata = validate_parent_metadata(audit.read_data(metadata_path, receipt['source_receipt_sha256'], protocol=False))
        physical = audit.read_data(store._path(folder + '/training-gpu-termination.json'), receipt['gpu_termination_sha256'], protocol=False)
        validate_seed_binding(receipt, metadata, source_sha256=audit._header_shas[metadata_path],
            termination_sha256=receipt['gpu_termination_sha256'])
        if receipt['proposal_sha256'] != proposal.sha256 or receipt['training_seed'] != seed:
            raise ContractError('closed Rust seed differs from actual proposal/seed')
        root = historical_record_root(bootstrap, physical)
        audit.record_root = root
        old_bootstrap = validate_campaign_bootstrap(audit.read_data(root / 'bootstrap.json'))
        clock = validate_campaign_clock(audit.read_data(root / 'session-clock.json', old_bootstrap['session_clock_sha256']))
        if (Path(old_bootstrap['campaign_directory']) != store.directory
                or old_bootstrap['supervision_id'] != root.name or clock['supervision_id'] != root.name):
            raise ContractError('historical campaign/supervision namespace differs')
        historical_campaign = validate_loaded_campaign(audit.read_data(Path(old_bootstrap['loaded_campaign_path']), old_bootstrap['loaded_campaign_sha256']))
        validate_seed_settings(audit.read_data(Path(old_bootstrap['seed_settings_path']), old_bootstrap['seed_settings_sha256']))
        if historical_campaign['campaign_sha256'] != loaded.sha256 or historical_campaign['base_sha256'] != loaded.base.sha256:
            raise ContractError('historical campaign CONFIG differs; no implicit migration')
        ticket = physical['ticket']
        global_admission = validate_global_admission(audit.read_data(_GLOBAL_ROOT / 'jobs' / ticket / 'admission.json', physical['admission_sha256']), ticket)
        validate_global_termination(physical, global_admission)
        original_physical = audit.read_data(_GLOBAL_ROOT / 'jobs' / ticket / 'termination.json', digest(physical))
        if original_physical != physical:
            raise ContractError('closed seed physical3 differs from original namespace DATA')
        admission = validate_local_admission(global_admission['host_admission'])
        invocation = admission['invocation_id']
        if admission['supervision_id'] != root.name:
            raise ContractError('closed physical receipt belongs to another supervision')
        state = validate_run(audit.read_data(ownership.directory / f'state-{invocation}.json', metadata['state_sha256'], protocol=False), hpo=True)
        audit.read_data(ownership.directory / 'run.json', metadata['state_sha256'], protocol=False)
        actual_loaded = _materialized(loaded, campaign, proposal, seed, state, metadata, ownership.directory)
        audit.loaded = actual_loaded
        entry = state['invocations'][invocation]
        operation_path = Path(entry['operation_path'])
        operation = validate_operation(audit.read_data(operation_path, entry['operation_sha256']), session_finish=clock['hard_session_finish_bits'])
        original_admission = audit.read_data(Path(entry['admission_path']), entry['admission_sha256'])
        if original_admission != admission:
            raise ContractError('global/local original admission DATA differs')
        _awaiting(audit, campaign, proposal, seed, operation, operation_path, entry['operation_sha256'], old_bootstrap['session_clock_sha256'])
        if (operation['schema_version'] != 2 or operation['operation_type'] != 'hpo-train'
                or operation['parent_directory'] != str(ownership.directory)
                or operation['run_id'] != state['run_id'] or operation['config_sha256'] != actual_loaded.sha256
                or operation['runtime_binding_sha256'] != state['runtime_binding_sha256']
                or operation['expected_context'] != admission['context'] or operation['deadlines'] != admission['deadlines']
                or operation['execution_request'] != entry['execution_request']
                or admission['context']['campaign_id'] != campaign['campaign_id']
                or admission['producer'] != state['identity']['player_runtime']['producer']):
            raise ContractError('completed source original operation/config/runtime/campaign differs')
        audit.read_data(Path(operation['config_path']), actual_loaded.sha256)
        from gf_models.torch_runtime import runtime_metadata
        runtime = runtime_metadata()
        sampler = digest({'format': 'cpu-torch-generator-v1', 'runtime': runtime,
            'module_cuda': '13.2' if runtime['torch'].endswith('+cu132') else None})
        current_producer = runtime_producer(SimpleNamespace(manifest={'executable_sha256':
            hashlib.sha256(Path(f'/proc/{control._parent_pid}/exe').read_bytes()).hexdigest()}), sampling_runtime_sha256=sampler)
        if current_producer != admission['producer']:
            raise ContractError('completed source producer differs; cross-build import is not implicit')
        prepared = audit.read_data(root / f'prepared-{invocation}.json', metadata['prepared_sha256'])
        exit_record = audit.read_data(root / f'exit-{invocation}.json', metadata['exit_sha256'])
        validate_prepared(prepared, admission)
        validate_exit(exit_record, admission)
        completion = validate_completion(audit.read_data(root / f'completion-{invocation}.json', metadata['completion_receipt_sha256']))
        if (exit_record != physical['exit_record'] or not exit_record['cooperative']
                or exit_record['prepared_sha256'] != metadata['prepared_sha256']
                or completion['prepared_sha256'] != metadata['prepared_sha256']
                or completion['exit_sha256'] != metadata['exit_sha256']
                or completion['source_manifest_sha256'] != prepared['source_manifest_sha256']
                or completion['state_sha256'] != metadata['state_sha256']
                or completion['generation_sha256'] != metadata['generation_sha256']
                or completion['state'] != 'complete' or completion['eligible'] is not True
                or completion['committed_update'] != state['updates']
                or prepared['committed_update'] != state['updates'] or prepared['generation'] != state['updates']
                or prepared['state_sha256'] != metadata['state_sha256'] or prepared['state'] != 'complete'
                or completion['operation_sha256'] != entry['operation_sha256']
                or any(completion[key] != admission[key] for key in ('run_id', 'supervision_id', 'invocation_id',
                    'operation_id', 'config_sha256', 'runtime_binding_sha256'))
                or completion['audit_path'] != str(root / f'cold-audit-{invocation}.json')
                or completion['audit_sha256'] != metadata['source_validation_sha256']):
            raise ContractError('closed completion/physical/Prepared/metadata DATA chain differs')
        source_audit = audit.read_data(Path(completion['audit_path']), metadata['source_validation_sha256'])
        _validate_historical_audit(source_audit, state=state, prepared=prepared, completion=completion)
        original_metadata = validate_parent_metadata(audit.read_data(root / f'parent-metadata-{invocation}.json', digest(metadata)))
        if original_metadata != metadata:
            raise ContractError('campaign metadata3 differs from actual original Parent-issued metadata')
        audit.campaign_id = campaign['campaign_id']
        audit.audit_payload(state, metadata, prepared, session_finish=clock['hard_session_finish_bits'])
        return audit
    except BaseException as error:
        from gf_training.rust_player_completion import _capture_failure_graph
        audit._failed = True
        _capture_failure_graph(audit, error)
        raise


def _project_fresh_config(loaded: Any, settings: Any, training: Json, *, output: Path,
                           evaluation_id: str, started: float, hard_finish: float) -> Any:
    import math
    from dataclasses import replace

    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_config import materialize_hpo_run
    from gf_common.domain import clone
    if (type(started) not in (int, float) or not math.isfinite(started)
            or type(hard_finish) not in (int, float) or not math.isfinite(hard_finish)):
        raise ContractError('finite original evaluation/session deadline required')
    base = materialize_hpo_run(loaded, settings, learning_seed=training['training_seed'], run_id=training['run_id'])
    run = replace(base.config.run, split='validation', name=evaluation_id, output_dir=str(output.parent),
        seeds=loaded.config.evaluation_game_seeds, sampling_seeds=(loaded.config.sampler_seed,), pairs_per_seed=1)
    if started + run.wall_seconds > hard_finish:
        raise TimeoutError('original hard session finish cannot fit the full evaluation wall')
    config = replace(base.config, run=run)
    raw = clone(asdict(config))
    return LoadedConfig(config, raw, digest(raw), base.inputs)


def _fresh_evaluation_plan(loaded: Any, campaign: Json, parent: Json, binding: Any, projected: Any, *,
                           output: Path, evaluation_id: str, ticket: str, started: float) -> Any:
    from gf_cli.hpo_evaluation import HpoEvaluationPlan
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import RuntimePolicyBinding, clone
    from gf_common.schedule import serialize_schedule
    bindings = {parent['policy_label']: binding, 'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')}
    link = {'schema_version': 1, 'kind': 'hpo-evaluation-linkage', 'campaign_id': campaign['campaign_id'],
        'evaluation_id': evaluation_id, 'gpu_ticket': ticket, 'purpose': 'objective',
        'conditions_sha256': campaign['conditions_sha256'], 'left_parent': parent, 'right_parent': None,
        'game_seeds': list(loaded.config.evaluation_game_seeds), 'learner_seats': {}, 'schedule_digest': '0' * 64,
        'policy_registry_digest': digest({key: asdict(value) for key, value in bindings.items()})}
    groups = hpo_evaluation_groups(link)
    schedule = clone(serialize_schedule(groups, 4))
    link['learner_seats'] = {spec['episode_id']: spec['learner_viewer'] for group in schedule for spec in group['episodes']}
    link['schedule_digest'] = digest(schedule)
    run = projected.config.run
    wall = started + run.wall_seconds
    return HpoEvaluationPlan.create(groups, bindings, output, link, started_monotonic=started,
        admission_deadline=started + run.admission_seconds, execution_deadline=started + run.execution_seconds,
        work_deadline=wall - run.report_reserve_seconds, wall_deadline=wall)


def _fresh_evaluation_request(loaded: Any, settings: Any, campaign: Json, training: Json,
                              parent: Json, binding: Any, *, output: Path, evaluation_id: str,
                              ticket: str, started: float, hard_finish: float) -> tuple[Any, Any]:
    """Pure existing Fresh recipe; native training is never an evaluation selector."""
    projected = _project_fresh_config(loaded, settings, training, output=output,
        evaluation_id=evaluation_id, started=started, hard_finish=hard_finish)
    plan = _fresh_evaluation_plan(loaded, campaign, parent, binding, projected, output=output,
        evaluation_id=evaluation_id, ticket=ticket, started=started)
    return projected, plan


class _RustHistoryAudit:
    """One actual historical parent lock/account; failure keeps both reachable."""
    def __init__(self) -> None:
        from contextlib import ExitStack
        self.parent_scope = ExitStack()
        self.pool: Any = None
        self._cleanup_children: list[Any] = []
        self.closed = False

    def close(self) -> None:
        from gf_training.group_pack_resources import close_resources
        from gf_training.rust_player_completion import _clear_failure_frames, _discard_failure_graph
        if self.closed:
            return
        _clear_failure_frames(self)
        close_resources(self, [])
        if self.pool is not None:
            self.pool.close()
        self.parent_scope.close()
        self.closed = True
        _discard_failure_graph(self)


def _read_rust_history(scope: _RustHistoryAudit, store: Any, loaded: Any, campaign: Json, proposal: Any,
                       seed: int, control: Any, deadline: float, *, evaluation: bool) -> tuple[Any, BaseException | None]:
    """Return the original error so all funded producer frames stop before close."""
    from gf_cli.hpo_evaluation import verify_seed_evaluation, verify_training_receipt
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.lineage import run_lock
    from gf_training.rust_player_completion import _capture_failure_graph
    try:
        training, _ = store.get(f'proposals/{proposal.number:06}/seeds/{seed}/training.json')
        validate_seed_training(training)
        directory = Path(loaded.config.storage.parent_root) / training['run_id']
        from gf_common.rust_player import safe_component
        safe_component(training['run_id'])
        if (Path(training['run_path']) != directory or training['training_seed'] != seed
                or training['proposal_sha256'] != proposal.sha256):
            raise ContractError('historical seed differs from current exact parent/proposal namespace')
        ownership = scope.parent_scope.enter_context(run_lock(directory))
        scope.pool = PackedArtifactPool(max_owned_bytes=min(loaded.config.operating.source_budget_bytes,
            loaded.base.config.preprocessing.memory_budget_mib * 1024**2), deadline=deadline)
        phase = ArtifactPhaseAdmission(scope.pool.budget, loaded.config.operating.parent_budget_bytes,
            name='parent', phase='rust-hpo-history')
        options = {'control': control, 'ownership': ownership, 'pool': scope.pool, 'parent_budget': phase}
        value: Any
        if evaluation:
            value = verify_seed_evaluation(store, loaded, campaign, proposal, seed, deadline=deadline, **options)
        else:
            value = verify_training_receipt(store, loaded, campaign, proposal, seed, deadline=deadline, **options)
        return value, None
    except BaseException as error:  # noqa: BLE001 -- preserve cancellation/error until funded producer frames stop.
        _capture_failure_graph(scope, error)
        return None, error


def _verified_rust_history(store: Any, loaded: Any, campaign: Json, proposal: Any, seed: int, *,
                           control: Any, deadline: float, evaluation: bool) -> Any:
    from gf_training.rust_player_completion import _capture_failure_graph
    _current_context(store, loaded, campaign, proposal, seed, control)
    import math
    import time

    from gf_common.rust_player import deadline_seconds
    from gf_training.group_pack_resources import remaining
    hard_finish = deadline_seconds(_actual_campaign_clock(control)['hard_session_finish_bits'])
    if (type(deadline) not in (float, int) or not math.isfinite(deadline)
            or deadline > min(hard_finish, time.monotonic() + loaded.config.run_wall_seconds)):
        raise ContractError('historical audit cannot extend current configured operation/session wall')
    remaining(deadline)
    scope = _RustHistoryAudit()
    value, error = _read_rust_history(scope, store, loaded, campaign, proposal, seed, control, deadline, evaluation=evaluation)
    try:
        scope.close()
    except BaseException as cleanup:
        _capture_failure_graph(scope, cleanup)
        if error is None:
            raise
        error.add_note('actual historical audit cleanup remains owned: ' + type(cleanup).__name__)
    if error is not None:
        vars(error)['cleanup_owner'] = scope
        raise error
    return value


def verify_rust_training_history(store: CampaignStore, loaded: LoadedHPOConfig, campaign: Json, proposal: Proposal, seed: int, *,
                                  control: HostControl, deadline: float) -> Json:
    return _verified_rust_history(store, loaded, campaign, proposal, seed,
        control=control, deadline=deadline, evaluation=False)


def verify_rust_evaluation_history(store: CampaignStore, loaded: LoadedHPOConfig, campaign: Json, proposal: Proposal, seed: int, *,
                                    control: HostControl, deadline: float) -> tuple[object, str, str]:
    return _verified_rust_history(store, loaded, campaign, proposal, seed,
        control=control, deadline=deadline, evaluation=True)


class _RustEvaluationAttempt:
    """Actual one-attempt owners; never a job, Source or retirement capability."""
    def __init__(self) -> None:
        from contextlib import ExitStack
        self.parent_scope = ExitStack()
        self.pool: Any = None
        self.parent_budget: Any = None
        self.job: Any = None
        self.data: Any = None
        self.audit: Any = None
        self.ownership: Any = None
        self.parent: Any = None
        self.control: Any = None
        self.observation: Any = None
        self.metadata_charge: Any = None
        self.binding_charge: Any = None
        self.acquired: Any = None
        self.released: Any = None
        self.termination: Any = None
        self.admission_requested = self.collector_requested = False
        self.inputs_pending = False
        self.parent_released = False
        self._cleanup_children: list[Any] = []

    def reserve_config_metadata(self, projected: Any) -> None:
        # This distinct config is already live when the closed-parent audit starts.
        self.metadata_charge = self.parent_budget.reserve(
            32 * len(canonical(projected.effective).encode()) + 131072)

    def reserve_binding_metadata(self, binding: Any) -> None:
        from gf_common.domain import RuntimePolicyBinding
        # Increment the SAME account with only the newly known binding portion.
        # Phase coupons cannot grow; a separate reserve(delta) is nonwaiting.
        self.binding_charge = self.parent_budget.reserve(32 * sum(
            len(canonical(asdict(value)).encode()) for value in (binding, RuntimePolicyBinding('heuristic'))))

    def retire_inputs(self, error: BaseException) -> Any:
        from gf_cli.hpo_evaluation import _confirmed_evaluation_cleanup
        from gf_training.group_pack_resources import close_resources
        from gf_training.rust_player_completion import _clear_failure_frames, _discard_failure_graph
        _clear_failure_frames(self)
        if self.job is None:
            if self.admission_requested:
                raise ContractError('actual job admission effects remain uncertain')
        elif self.collector_requested:
            if self.observation is None:
                raise ContractError('actual started collector retirement observation required')
            self.observation.require_core_retired(error=error)
        else:
            if (self.inputs_pending or self.observation is not None and self.observation._core_phase != 'unentered'
                    or self.job.ledger._history):
                raise ContractError('actual collector/resource creation was attempted')
        if self.job is not None:
            self.termination = (self.job.confirm_termination() if self.job._receipt is None else
                self.job.require_confirmed_termination(self.job._receipt))
        if self.observation is not None:
            self.observation.release_artifacts()
        if self.data is not None:
            self.acquired = self.data.reference
            self.data.close(**({} if self.ownership is None else {'ownership': self.ownership}))
            self.released = self.data.release_reference
        close_resources(self, [])  # includes actual partially constructed inputs
        if self.audit is not None:
            self.audit.close()
            self.audit = None
        self.parent_scope.close()
        self.ownership = None
        if self.parent is not None and not self.parent_released:
            self.parent.close()
            self.control.release_parent()
            self.parent_released = True
        if self.observation is not None:
            # A started collector needs its original outer proof. An unentered
            # one instead has the actual empty-stage/ledger/job proof above.
            if self.collector_requested:
                _confirmed_evaluation_cleanup(error, self.observation, self.job,
                    [(self.data.directory, self.acquired)], {id(self.acquired): self.released})
            self.observation = None
        if self.binding_charge is not None:
            self.binding_charge.close()
            self.binding_charge = None
        if self.metadata_charge is not None:
            self.metadata_charge.close()
            self.metadata_charge = None
        if self.pool is not None:
            self.pool.close()
        _discard_failure_graph(self)
        return self.termination


def _actual_campaign_clock(control: HostControl) -> Json:
    from gf_common.rust_player import validate_campaign_clock
    from gf_training.rust_player_supervision import read_control_record
    bootstrap = control.campaign_bootstrap
    clock, checksum = read_control_record(Path(bootstrap['session_clock_path']), control.root)
    validate_campaign_clock(clock)
    if checksum != bootstrap['session_clock_sha256']:
        raise ContractError('actual current campaign clock differs from authenticated bootstrap')
    return clock


def _expected_fresh_linkage(loaded: LoadedHPOConfig, facts: tuple[Any, ...]) -> Json:
    """Only bounded scalar identities survive the retired plan/config graphs."""
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import clone
    from gf_common.schedule import serialize_schedule
    (campaign_id, evaluation_id, ticket, conditions_sha, parent_fields, registry_sha, schedule_sha) = facts
    names = ('run_id', 'run_path', 'training_seed', 'generation', 'generation_sha256',
        'inference_sha256', 'source_receipt_sha256', 'policy_label')
    link = {'schema_version': 1, 'kind': 'hpo-evaluation-linkage', 'campaign_id': campaign_id,
        'evaluation_id': evaluation_id, 'gpu_ticket': ticket, 'purpose': 'objective',
        'conditions_sha256': conditions_sha, 'left_parent': dict(zip(names, parent_fields, strict=True)),
        'right_parent': None, 'game_seeds': list(loaded.config.evaluation_game_seeds),
        'learner_seats': {}, 'schedule_digest': schedule_sha, 'policy_registry_digest': registry_sha}
    schedule = clone(serialize_schedule(hpo_evaluation_groups(link), 4))
    if digest(schedule) != schedule_sha:
        raise ContractError('retired request differs from original deterministic Fresh schedule')
    link['learner_seats'] = {spec['episode_id']: spec['learner_viewer'] for group in schedule for spec in group['episodes']}
    return link


async def _run_rust_evaluation(scope: _RustEvaluationAttempt, store: Any, loaded: Any, campaign: Json,
                               proposal: Any, training: Json, *, output: Path, soft_finish: float,
                               hard_finish: float, progress: Any) -> tuple[Any, BaseException | None]:
    import time
    import uuid

    from gf_cli.hpo_evaluation import (
        _check_live_evaluation_storage,
        _HpoEvaluationCleanupObservation,
    )
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.card_instance_collection import checked_hpo_evaluation_parent
    from gf_common.hpo_progress import observe
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
    from gf_training.lineage import run_lock
    from gf_training.rust_player_completion import _capture_failure_graph
    from gf_training.rust_player_evaluation import RustEvaluationData, final_policy_binding
    try:
        seed = training['training_seed']
        folder = f'proposals/{proposal.number:06}/seeds/{seed}'
        if store.exists(folder + '/evaluation.intent.json'):
            raise ContractError('interrupted evaluation cannot be restarted or requalified')
        started = time.monotonic()
        if started >= soft_finish or started + loaded.config.run_wall_seconds > hard_finish:
            raise TimeoutError('original session cannot dispatch this full Fresh evaluation')
        evaluation_id = str(uuid.uuid4())
        # Materialize ONCE; the same graph backs the original intent and plan.
        projected = _project_fresh_config(loaded, proposal.settings, training, output=output,
            evaluation_id=evaluation_id, started=started, hard_finish=hard_finish)
        run, config_sha = projected.config.run, projected.sha256
        _check_live_evaluation_storage(output, projected)
        if time.monotonic() >= soft_finish:
            raise TimeoutError('original soft dispatch finish before evaluation intent')
        store.put(folder + '/evaluation.intent.json', {'schema_version': 1, 'kind': 'hpo-evaluation-intent',
            'evaluation_id': evaluation_id, 'directory': str(output), 'config_sha256': config_sha,
            'training_receipt_sha256': store.get(folder + '/training.json')[1],
            'started_unix': time.time(), 'started_monotonic': started})
        if time.monotonic() >= soft_finish:
            raise TimeoutError('original soft dispatch finish before evaluation admission')
        scope.admission_requested = True
        scope.job = admit_gpu_job(JobContext(campaign['campaign_id'], evaluation_id, 'evaluation', training['run_id'], evaluation_id),
            global_cuda_bytes=loaded.config.operating.managed_cuda_bytes,
            max_workers=loaded.config.operating.max_participant_workers)
        scope.inputs_pending = True  # notification/ref/FD construction is not known-empty
        store.put(folder + '/evaluation.admission.json', {'schema_version': 1, 'kind': 'hpo-evaluation-admission',
            'gpu_ticket': scope.job.ticket, 'evaluation_id': evaluation_id})
        scope.pool = PackedArtifactPool(max_owned_bytes=loaded.config.operating.source_budget_bytes,
            deadline=started + run.wall_seconds - run.report_reserve_seconds)
        scope.parent_budget = ArtifactPhaseAdmission(scope.pool.budget, loaded.config.operating.parent_budget_bytes,
            name='parent', phase='rust-hpo-evaluation')
        scope.reserve_config_metadata(projected)
        if scope.parent is not None:
            scope.parent.require()
            if scope.parent._invocation.cold._control is not scope.control:
                raise ContractError('live Parent differs from SAME current campaign control')
            state, metadata = scope.parent.state, scope.parent.metadata
            _materialized(loaded, campaign, proposal, seed, state, metadata, Path(training['run_path']))
            validate_seed_binding(training, metadata, source_sha256=training['source_receipt_sha256'],
                termination_sha256=training['gpu_termination_sha256'])
            scope.ownership = scope.parent._invocation._ownership
        else:
            scope.ownership = scope.parent_scope.enter_context(run_lock(Path(training['run_path'])))
            scope.audit = open_closed_rust_parent(store, loaded, campaign, proposal, seed, control=scope.control,
                ownership=scope.ownership, pool=scope.pool, parent_budget=scope.parent_budget,
                deadline=started + run.execution_seconds)
            state, metadata = scope.audit.state, scope.audit.metadata
        label, binding = final_policy_binding(state, metadata)
        parent = checked_hpo_evaluation_parent({**{name: training[name] for name in
            ('run_id', 'run_path', 'training_seed', 'generation', 'generation_sha256', 'inference_sha256', 'source_receipt_sha256')},
            'policy_label': label})
        scope.reserve_binding_metadata(binding)
        plan = _fresh_evaluation_plan(loaded, campaign, parent, binding, projected,
            output=output, evaluation_id=evaluation_id, ticket=scope.job.ticket, started=started)
        del state, metadata
        if scope.parent is not None:
            scope.data = RustEvaluationData.from_parent(scope.parent, binding, label,
                evaluation_job=scope.job, pool=scope.pool, parent_budget=scope.parent_budget, pin=scope.job.ticket)
            scope.parent.close()
            scope.ownership = None
            scope.control.release_parent()
            scope.parent_released = True
        else:
            scope.data = RustEvaluationData.from_completed(scope.audit, binding, label,
                evaluation_job=scope.job, pool=scope.pool, parent_budget=scope.parent_budget, pin=scope.job.ticket)
            scope.audit.close()
            scope.audit = None
            scope.parent_scope.close()
        scope.ownership = None
        scope.observation = _HpoEvaluationCleanupObservation(plan, scope.job, scope.data.provider)
        scope.inputs_pending = False
        link = plan.linkage
        names = ('run_id', 'run_path', 'training_seed', 'generation', 'generation_sha256',
            'inference_sha256', 'source_receipt_sha256', 'policy_label')
        facts = (link['campaign_id'], link['evaluation_id'], link['gpu_ticket'], link['conditions_sha256'],
            tuple(link['left_parent'][name] for name in names), link['policy_registry_digest'], link['schedule_digest'])
        scope.collector_requested = True
        from gf_cli.card_instance import collect_hpo_evaluation
        result = await collect_hpo_evaluation(projected, plan, gpu_job=scope.job, artifact_inputs=scope.data.provider,
            _cleanup_observation=scope.observation, _rust_evaluation_data=scope.data, progress=progress)
        if result.report.get('healthy') is not True:
            raise ContractError('primary evaluation did not finish/save/clean up normally')
        observe(progress, {'phase': 'cleanup'})
        return (facts, str(result.directory), config_sha), None
    except BaseException as error:  # noqa: BLE001 -- preserve cancellation/error until funded producer frames stop.
        _capture_failure_graph(scope, error)
        return None, error


async def evaluate_rust_seed(store: CampaignStore, loaded: LoadedHPOConfig, campaign: Json, proposal: Proposal,
                              training: Json, *, control: HostControl, output: Path, soft_finish: float,
                              hard_finish: float, completion: ColdFinalizedRustParent | None = None,
                              progress: Any = None) -> Json:
    """One true Fresh evaluation; actual callers persist exact seed2 beforehand."""
    from gf_cli.hpo_evaluation import (
        _copy_failed_evaluation_receipt,
        _projection,
        check_evaluation_receipt_runtime,
        evaluation_receipt_runtime_fields,
        verify_saved_evaluation,
    )
    from gf_common.domain import clone
    from gf_common.hpo_progress import observe
    from gf_common.rust_player import deadline_seconds
    from gf_training.rust_player_completion import ColdFinalizedRustParent, _capture_failure_graph
    from gf_training.structured_rollout import _owned_bytes
    seed = integer(training.get('training_seed') if type(training) is dict else None, 0)
    _current_context(store, loaded, campaign, proposal, seed, control)
    validate_seed_training(training)
    clock = _actual_campaign_clock(control)
    if (type(soft_finish) not in (float, int) or type(hard_finish) not in (float, int)
            or soft_finish != deadline_seconds(clock['soft_dispatch_finish_bits'])
            or hard_finish != deadline_seconds(clock['hard_session_finish_bits'])):
        raise ContractError('evaluation requires the SAME original authenticated session cutoffs')
    folder = f"proposals/{proposal.number:06}/seeds/{training['training_seed']}"
    durable, training_sha = store.get(folder + '/training.json')
    source, source_sha = store.get(folder + '/source-validation.json')
    physical, physical_sha = store.get(folder + '/training-gpu-termination.json')
    validate_seed_binding(training, source, source_sha256=source_sha, termination_sha256=physical_sha)
    if durable != training or training['proposal_sha256'] != proposal.sha256:
        raise ContractError('evaluation requires the already durable exact seed2 training receipt')
    if completion is not None:
        if type(completion) is not ColdFinalizedRustParent:
            raise ContractError('actual qualified live Rust Parent required')
        completion.require()
        if (completion._invocation.cold._control is not control or completion.metadata != source
                or completion._invocation.cold.physical_termination.global_observation() != physical):
            raise ContractError('actual live Parent differs from durable Source/physical receipt pair')
    # The caller's immutable validated inputs remain borrowed; newly materialized
    # plan/config graphs live only in the inner attempt's receiving account.
    del durable, source, physical
    scope = _RustEvaluationAttempt()
    scope.parent, scope.control = completion, control
    outcome, original = await _run_rust_evaluation(scope, store, loaded, campaign, proposal, training,
        output=output, soft_finish=soft_finish, hard_finish=hard_finish, progress=progress)
    marker = original if original is not None else ContractError('actual Fresh core retirement')
    try:
        scope.retire_inputs(marker)
    except BaseException as cleanup:
        _capture_failure_graph(scope, cleanup)
        if original is None:
            raise
        original.add_note('actual Rust evaluation cleanup remains owned: ' + type(cleanup).__name__)
        vars(original)['cleanup_owner'] = scope
        raise original
    if original is not None:
        if scope.termination is not None:
            _copy_failed_evaluation_receipt(original, store, folder + '/evaluation-failed-gpu-termination.json',
                scope.job._admission['context']['evaluation_id'], scope.termination)
            try:
                scope.job.close()
            except BaseException as cleanup:  # noqa: BLE001 -- original collector error and uncertain close both remain owned.
                _capture_failure_graph(scope, cleanup)
                original.add_note('actual confirmed job close remains uncertain: ' + type(cleanup).__name__)
        vars(original)['cleanup_owner'] = scope
        raise original
    facts, directory_text, config_sha = outcome
    link = _expected_fresh_linkage(loaded, facts)
    try:
        observe(progress, {'phase': 'source'})
        # Independent legacy audit allowance is sequential: receiving pool is
        # positively closed and no original plan/provider/core graph survives.
        manifest, proof = verify_saved_evaluation(Path(directory_text), loaded, link, deadline=scope.pool.deadline,
            extra_owned_bytes=_owned_bytes(link), evaluation_runtime=None)
        intent, _ = store.get(folder + '/evaluation.intent.json')
        if (manifest['config_hash'] != config_sha or intent['config_sha256'] != config_sha
                or intent['training_receipt_sha256'] != training_sha
                or intent['evaluation_id'] != link['evaluation_id'] or intent['directory'] != directory_text):
            raise ContractError('saved evaluation differs from original intent/config/training bytes')
        termination = scope.job.require_confirmed_termination(scope.termination)
        gpu_sha = store.put(folder + '/evaluation-gpu-termination.json', clone(asdict(termination)))
        receipt = {'schema_version': 1, 'kind': 'hpo-seed-evaluation', 'training_receipt_sha256': training_sha,
            'linkage_sha256': digest(link), 'manifest_sha256': proof['manifest_sha256'],
            'originals_digest': proof['source_set_sha256'],
            'projection': _projection(manifest, training, training['source_receipt_sha256'], campaign['conditions_sha256']),
            'conditions_sha256': campaign['conditions_sha256'], 'pin_acquire_sha256': scope.acquired.receipt_sha256,
            'pin_release_sha256': scope.released.receipt_sha256, 'gpu_termination_sha256': gpu_sha,
            'directory': directory_text, 'linkage': link, 'config_sha256': config_sha}
        receipt.update(evaluation_receipt_runtime_fields(manifest))
        check_evaluation_receipt_runtime(receipt, manifest)
        scope.job.require_confirmed_termination(termination)
        scope.job.close()
        store.put(folder + '/evaluation.json', receipt)
        return receipt
    except BaseException as error:
        _capture_failure_graph(scope, error)
        # The job remains owned with its original receipt and all actual failed
        # verifier owners remain attached. No failed audit produces an objective.
        raise
