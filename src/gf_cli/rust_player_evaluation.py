"""Strict schema6 closed-seed constructors/readers at the actual CLI owner edge."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
