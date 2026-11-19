"""Concrete local HPO commands; optional libraries are loaded only for execution."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_cli.config import GroupPackExperimentConfig, GroupPackStorageConfig
from gf_cli.hpo_campaign import (
    _CAMPAIGN_KEYS,
    CampaignStudy,
    Proposal,
    _checked,
    canonical_control,
    encode_settings,
    import_settings,
    initialize_campaign,
)
from gf_cli.hpo_config import (
    HPOOperatingConfig,
    LoadedHPOConfig,
    joint_resource_admission,
    load_hpo_config,
    require_hpo_runtime_execution,
)
from gf_cli.hpo_evaluation import (
    aggregate_trial_evaluation,
    evaluate_seed,
    owned_runtime_module,
    persist_training_receipt,
    verify_seed_evaluation,
    verify_training_receipt,
)
from gf_cli.hpo_storage import CampaignStore, check_storage
from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_collection import HPO_EVALUATION_STREAM_VERSION
from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.hpo_progress import ProgressCallback
from gf_common.shared_inference import evaluation_runtime
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_binding import ConfiguredNeuralModelV3


def _originals(loaded: LoadedHPOConfig) -> Path:
    base = loaded.base.config
    return Path(loaded.config.storage.originals_root or (
        base.storage.db_root if isinstance(base, GroupPackExperimentConfig) else GroupPackStorageConfig().db_root))


def validate(loaded: LoadedHPOConfig) -> Json:
    control = canonical_control(loaded.config)
    from dataclasses import replace
    if loaded.config.schema_version == 6:
        from gf_cli.rust_player_entry import campaign_startup_preview
        admission = campaign_startup_preview(loaded)
        control_admission = campaign_startup_preview(loaded, settings=control)
    else:
        admission = joint_resource_admission(loaded.config, layout=LAYOUT)
        control_admission = joint_resource_admission(replace(loaded.config, model=control.model, training=control.training), layout=LAYOUT)
    storage = check_storage(loaded.config.storage, campaign_dir=Path(loaded.config.storage.parent_root) / 'campaign-validation',
        originals_root=_originals(loaded), deadline=time.monotonic() + loaded.config.run_wall_seconds)
    return {'schema_version': 1, 'kind': 'hpo-config-validation', 'config_sha256': loaded.sha256,
        'study_config_identity': loaded.config.study_identity, 'canonical_control_identity': control.identity,
        'canonical_control_in_space': encode_settings(loaded.config, control) is not None,
        'resource_admission': admission, 'canonical_control_admission': control_admission, 'storage': storage,
        'runtime_source_preflight': 'not-executed', 'qualification': 'not-performed'}


def _conditions(loaded: LoadedHPOConfig) -> Json:
    """Measure source/binary facts without Torch/device allocation or gameplay."""
    from gf_common.card_instance import reviewed_specs
    from gf_common.card_instance_rules import card_instance_rules, rule_signature
    from gf_models.arrays import sha256
    from gf_models.structured_binding import child_cuda_visibility
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_training.ppo_run import source_identity
    config = loaded.base.config
    if type(config.actions) is not ActionConfigV3 or type(config.model) is not ConfiguredNeuralModelV3:
        raise ContractError('HPO conditions require the exact corrected model3 base')
    if hasattr(loaded.config, 'training_opponents') and loaded.config.schema_version not in (4,5):
        from gf_training.native_backend import require_native_refresh_opponents
        require_native_refresh_opponents()
    deadline = time.monotonic() + loaded.config.run_wall_seconds
    if config.server.source_dir is None or config.server.external_url is not None:
        raise ContractError('HPO requires its approved locally managed Simulator source/binary')
    evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir), contract=config.actions.action_contract, deadline=deadline)
    measured = evidence.preflight()
    value: Json = {'schema_version': 1, 'kind': 'hpo-study-conditions', 'producer_identity': source_identity(deadline),
        'integration_identity': {'binary_sha256': measured['binary']['sha256'],
            'server_config_sha256': sha256(Path(config.server.config), deadline),
            'declared_simulator_revision': config.server.simulator_revision, 'binary_build_revision': None,
            'source_revision': measured['source']['revision'], 'source_spec_sha256': reviewed_specs(config.actions.action_contract)},
        'action_config_sha256': digest(asdict(config.actions)), 'action_contract': asdict(config.actions.action_contract),
        'rule_signature': rule_signature(config.actions.action_contract, card_instance_rules()),
        'input_recipe': LAYOUT.input_recipe, 'evaluation_streams': HPO_EVALUATION_STREAM_VERSION,
        'decision_budget': loaded.config.decision_budget, 'learning_seeds': list(loaded.config.learning_seeds),
        'game_seeds': list(loaded.config.evaluation_game_seeds), 'inference_binding': asdict(loaded.config.inference),
        'cuda_visible_devices': child_cuda_visibility(loaded.config.inference.device, config.model.cuda_visible_devices)}
    if measured['schema_version']==2:value['integration_identity']['source_dirty']=measured['source']['dirty']
    if loaded.config.schema_version == 6:
        value['evaluation_runtime'] = None  # Explicit existing legacy-fresh evaluation, not Rust training runtime.
    elif hasattr(loaded.config,'runtime'):
        value['evaluation_runtime'] = evaluation_runtime(loaded.config)
    if loaded.config.schema_version==5:
        from gf_common.shared_inference import shared_dependency_versions
        value['shared_inference_libraries']=shared_dependency_versions()
    if loaded.config.schema_version in (4,5):
        value.update(origin=loaded.effective['origin'],encoder=loaded.effective['encoder'])
    elif config.preprocessing.backend == 'rust':
        from gf_training.native_backend import binary_identity, hpo_reader_settings, load_native
        native = load_native()
        hpo_reader_settings(config.preprocessing, native, packed=True,resident=hasattr(loaded.config,'runtime'))
        value['native_identity'] = binary_identity(native)
    return value


def _settings_inputs(paths: tuple[Path, ...], loaded: LoadedHPOConfig) -> tuple[tuple[Any, ...], list[Json]]:
    before = [(path, path.stat()) for path in paths]
    settings = (import_settings(paths, loaded.config, study_identity='', _loaded=loaded)
        if loaded.config.schema_version == 6 else import_settings(paths, loaded.config, study_identity=''))
    records = []
    for (path, original), effective in zip(before, settings, strict=True):
        raw = path.read_bytes()
        after = path.stat()
        if any(getattr(original, name) != getattr(after, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ContractError('settings seed changed during checked import')
        origin = json.loads(raw)['origin']
        records.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'settings': asdict(effective), 'origin': origin})
    return settings, records


def next_seed_operation(store: CampaignStore, proposal: Proposal, learning_seeds: tuple[int, ...]) -> tuple[str, int] | None:
    """Scheduling only. Every closed receipt is independently revalidated by I/O."""
    root = f'proposals/{proposal.number:06}'
    if store.exists(root + '/failure.json'):
        raise ContractError('recorded proposal failure takes precedence over success-looking files')
    for seed in learning_seeds:
        folder = root + f'/seeds/{seed}'
        if not store.exists(folder + '/training.json'):
            return ('incomplete' if store.exists(folder + '/training.intent.json') else 'train', seed)
        if not store.exists(folder + '/evaluation.json'):
            return ('incomplete' if store.exists(folder + '/evaluation.intent.json') else 'evaluate', seed)
    return None


def _storage_boundary(store: CampaignStore, loaded: LoadedHPOConfig, *, operation_id: str) -> None:
    facts = check_storage(loaded.config.storage, campaign_dir=store.directory, originals_root=_originals(loaded),
        deadline=time.monotonic() + loaded.config.run_wall_seconds)
    store.put('storage-observations/' + operation_id + '.json', facts)


async def _train_seed(store: CampaignStore, loaded: LoadedHPOConfig, campaign: Json, proposal: Proposal, seed: int,
                      *, progress: ProgressCallback | None = None) -> Any:
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
    composition = owned_runtime_module('gf_cli.hpo_config', ('materialize_hpo_run',))
    core = owned_runtime_module('gf_cli.card_instance', ('collect_hpo_ppo',))
    runtime = owned_runtime_module('gf_training.hpo_ppo', ('train_hpo_ppo', 'HpoRunResult'))
    if loaded.config.schema_version==5:
        from gf_cli.hpo_evaluation import prospective_evaluation_admission
        prospective_evaluation_admission(loaded,proposal.settings)
    run_id = str(uuid.uuid4())
    directory = Path(loaded.config.storage.parent_root) / run_id
    materialized = composition.materialize_hpo_run(loaded, proposal.settings, learning_seed=seed, run_id=run_id)
    folder = f'proposals/{proposal.number:06}/seeds/{seed}'
    store.put(folder + '/training.intent.json', {'schema_version': 1, 'kind': 'hpo-training-intent',
        'proposal_sha256': proposal.sha256, 'training_seed': seed, 'run_id': run_id, 'run_path': str(directory),
        'config_sha256': materialized.sha256})
    from gf_common.training_opponents import training_opponents
    if training_opponents(materialized.effective) is not None and materialized.config.hpo_spec['schema_version'] not in (4,5):
        from gf_training.native_backend import require_native_refresh_opponents
        require_native_refresh_opponents()
    with admit_gpu_job(JobContext(campaign['campaign_id'], run_id, 'training', run_id, None),
            global_cuda_bytes=loaded.config.operating.managed_cuda_bytes,
            max_workers=loaded.config.operating.max_participant_workers) as job:
        store.put(folder + '/training.admission.json', {'schema_version': 1, 'kind': 'hpo-training-admission',
            'gpu_ticket': job.ticket, 'run_id': run_id})
        async def collector(plan: Any, ledger: Any) -> Any:
            if ledger is not job.ledger:
                raise ContractError('HPO collector requires the exact caller job ledger')
            return await core.collect_hpo_ppo(materialized, plan, ledger, **({} if progress is None else {'progress':progress}))
        result = await runtime.train_hpo_ppo(materialized, collector, directory, gpu_job=job, mode='fresh',
            **({'_reuse_precollection':True} if materialized.config.hpo_spec['schema_version']==3 else {}),
            **({} if progress is None else {'progress':progress}))
        if type(result) is not runtime.HpoRunResult or result.report.get('status') != 'complete' or result.report.get('used_decisions') != loaded.config.decision_budget:
            raise ContractError('training stopped incomplete; no objective and no budget/deadline extension')
        completion=getattr(result,'completion',None)
        try:
            persist_training_receipt(store, loaded, campaign, proposal, training_seed=seed, run_id=run_id,
                directory=directory, termination=result.gpu_termination, deadline=time.monotonic() + loaded.config.run_wall_seconds,
                **({'completion':completion} if materialized.config.hpo_spec['schema_version'] in (4,5) else {}))
            return completion
        except BaseException:
            if completion is not None: completion.close()
            raise


async def run_campaign(loaded: LoadedHPOConfig, directory: Path, *, mode: str,
                        seed_paths: tuple[Path, ...] = (), progress: str = 'auto') -> Json:
    """Concrete sequential operations; session time never interrupts an operation."""
    require_hpo_runtime_execution(loaded.config)
    if mode not in ('fresh', 'resume') or mode == 'resume' and seed_paths:
        raise ContractError('explicit fresh/resume required; stage settings are fresh-only')
    started, started_unix = time.monotonic(), time.time()
    # Do not let an older editable checkout supply a future runtime.
    owned_runtime_module('gf_cli.hpo_config', ('materialize_hpo_run',))
    owned_runtime_module('gf_training.hpo_ppo', ('train_hpo_ppo',))
    validate(loaded)
    if loaded.config.schema_version==5:
        from gf_cli.hpo_evaluation import prospective_evaluation_admission
        prospective_evaluation_admission(loaded)
    check_storage(loaded.config.storage, campaign_dir=directory, originals_root=_originals(loaded),
        deadline=time.monotonic() + loaded.config.run_wall_seconds)
    conditions = _conditions(loaded)
    imports, sources = _settings_inputs(seed_paths, loaded) if mode == 'fresh' else ((), [])
    if mode == 'fresh':
        conditions['initial_settings'] = sources
    with CampaignStore(directory, mode=mode, max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store, ExitStack() as completion_scope:
        live_completions: dict[int,Any]={}
        from gf_cli.hpo_progress import ProgressOwner
        owner = ProgressOwner(store,mode=progress)
        def run_progress(facts: Json) -> None:
            value = dict(facts)
            if 'status' in value:
                value['run_status'] = value.pop('status')
            owner.emit(value)
        async with owner.live():
            if mode == 'fresh':
                if 'producer_identity' in conditions:
                    # An explicitly chosen non-ignored output directory can make
                    # Git dirty without changing any source module bytes. Capture
                    # the actual post-namespace observation before binding study.
                    from gf_training.ppo_run import source_identity
                    conditions['producer_identity'] = source_identity(time.monotonic() + loaded.config.run_wall_seconds)
                campaign = initialize_campaign(store, loaded, conditions)
                if any(row['origin']['study_identity'] == campaign['study_identity'] for row in sources):
                    raise ContractError('settings imports require a different fresh study')
            else:
                campaign, _ = store.get('campaign.json')
                _checked(campaign, 'hpo-campaign', _CAMPAIGN_KEYS)
                preserved_imports = campaign['conditions'].get('initial_settings', [])
                conditions['initial_settings'] = preserved_imports
                if canonical(conditions) != canonical(campaign['conditions']):
                    raise ContractError('actual source/Simulator/inference conditions changed; use a new study')
                from gf_models.hpo_config import model_config_v4
                from gf_training.hpo_config import effective_settings, hpo_training_config
                imports = tuple(effective_settings(model_config_v4(row['settings']['model'], layout=LAYOUT),
                    hpo_training_config(row['settings']['training']), {}, layout=LAYOUT) for row in preserved_imports)
            study = CampaignStudy(store, loaded, campaign)
            control = canonical_control(loaded.config)
            in_space = encode_settings(loaded.config, control) is not None
            requests: list[tuple[int | None, str, Any]] = []
            if not in_space:
                requests.append((None, 'outside-control', control))
            for slot in range(loaded.config.trial_count):
                if in_space and slot == 0:
                    requests.append((slot, 'control', control))
                else:
                    imported_index = slot - (1 if in_space else 0)
                    requests.append((slot, 'seed', imports[imported_index]) if imported_index < len(imports) else (slot, 'sample', None))
            if len(imports) > loaded.config.trial_count - (1 if in_space else 0):
                raise ContractError('stage imports exceed declared attempted in-space slots')
            stop = 'complete'
            completed = failed = 0
            boundary: Json = {'phase': 'before-proposal'}
            try:
                for number, (attempt_slot, kind, requested) in enumerate(requests):
                    if time.monotonic() - started >= loaded.config.session_seconds:
                        stop = 'session-boundary'
                        break
                    owner.emit({'proposal_no':number,'proposal_kind':kind,'attempted_trial_count':sum(row[0] is not None for row in requests[:number+1]),
                        'outside_control_count':sum(row[0] is None for row in requests[:number+1]),'declared_trial_count':loaded.config.trial_count})
                    proposal = study.propose(number, slot=attempt_slot, kind=kind, requested=requested)
                    if proposal is None:
                        failed += 1
                        continue
                    root = f'proposals/{number:06}'
                    if store.exists(root + '/failure.json'):
                        failure, _ = store.get(root + '/failure.json')
                        study.fail(proposal, failure['reason'])
                        failed += 1
                        continue
                    if store.exists(root + '/tell.receipt.json'):
                        tell, _ = store.get(root + '/tell.receipt.json')
                        if tell.get('state') == 'COMPLETE':
                            study.complete(proposal, deadline=time.monotonic() + loaded.config.run_wall_seconds)
                            completed += 1
                        elif tell.get('state') == 'FAIL':
                            study.fail(proposal, tell['reason'])
                            failed += 1
                        else:
                            raise ContractError('unknown terminal proposal receipt')
                        continue
                    try:
                        validated_seeds: set[int] = set()
                        while True:
                            for reusable_seed in loaded.config.learning_seeds:
                                if (reusable_seed not in validated_seeds
                                        and store.exists(root + f'/seeds/{reusable_seed}/evaluation.json')):
                                    verify_seed_evaluation(store, loaded, campaign, proposal, reusable_seed,
                                        deadline=time.monotonic() + loaded.config.run_wall_seconds)
                                    validated_seeds.add(reusable_seed)
                            operation = next_seed_operation(store, proposal, loaded.config.learning_seeds)
                            if operation is None:
                                break
                            phase, seed = operation
                            boundary = {'proposal_no': number, 'phase': phase, 'training_seed': seed}
                            if phase == 'incomplete':
                                raise ContractError('interrupted operation has no closed receipt; original failure remains ineligible')
                            _storage_boundary(store, loaded, operation_id=str(uuid.uuid4()))
                            owner.emit({'phase':'startup' if phase=='train' else 'evaluate','model':proposal.settings.model.model_id,
                                'seed':seed,'status':'running',
                                **({'committed_used':None,'committed_updates':None,'latest_metrics':None} if phase=='train' else {}),
                                'decision_budget':loaded.config.decision_budget,'eta_seconds':None,'decisions_per_second':None,
                                'transient_games':None,'transient_learner_decisions':None,'transient_total_game_actions':None,'cuda_allocator':None})
                            if phase == 'train':
                                completion=await _train_seed(store, loaded, campaign, proposal, seed, progress=run_progress)
                                if completion is not None:
                                    live_completions[seed]=completion
                                    completion_scope.callback(completion.close)
                            else:
                                training = verify_training_receipt(store, loaded, campaign, proposal, seed,
                                    deadline=time.monotonic() + loaded.config.run_wall_seconds)
                                owner.emit({'committed_used':training.get('used_decisions'),'committed_updates':training.get('generation')})
                                await evaluate_seed(store, loaded, campaign, proposal, training,
                                    output=directory / 'evaluations' / str(uuid.uuid4()),progress=owner.emit,
                                    **({'completion':live_completions[seed]} if seed in live_completions else {}))
                                if seed in live_completions:
                                    live_completions.pop(seed).close()
                            # Only receipt-complete whole operations are session boundaries.
                            _storage_boundary(store, loaded, operation_id=str(uuid.uuid4()))
                            if time.monotonic() - started >= loaded.config.session_seconds:
                                stop = 'session-boundary'
                                break
                        if stop == 'session-boundary':
                            break
                        # Revalidate every reusable closed receipt, even across sessions.
                        for seed in loaded.config.learning_seeds:
                            verify_seed_evaluation(store, loaded, campaign, proposal, seed,
                                deadline=time.monotonic() + loaded.config.run_wall_seconds)
                        aggregate = aggregate_trial_evaluation(store, loaded, campaign, proposal,
                            deadline=time.monotonic() + loaded.config.run_wall_seconds)
                        store.put(root + '/evaluation.json', aggregate)
                        study.complete(proposal, deadline=time.monotonic() + loaded.config.run_wall_seconds)
                        completed += 1
                    except Exception as error:  # noqa: BLE001 - all operation failures produce no objective
                        reason = type(error).__name__ + ':' + str(error)
                        store.put(root + '/failure.json', {'schema_version': 1, 'kind': 'hpo-proposal-failure',
                            'proposal_sha256': proposal.sha256, 'reason': reason, 'boundary': boundary})
                        study.fail(proposal, reason)
                        failed += 1
                        owner.emit({'run_status':'failed'})
                        from gf_training.hpo_gpu_lease import AdmissionBusy, RecoveryRequired
                        if isinstance(error, (AdmissionBusy, RecoveryRequired)):
                            stop = 'resource-blocked'
                            break
            except BaseException:
                stop = 'error'
                owner.emit({'status':'failed'})
                raise
            finally:
                store.put('sessions/' + str(uuid.uuid4()) + '.json', {'schema_version': 1, 'kind': 'hpo-session',
                    'started_unix': started_unix, 'elapsed_seconds': time.monotonic() - started,
                    'session_budget_seconds': loaded.config.session_seconds, 'stop_reason': stop, 'boundary': boundary})
            owner.emit({'status':'session-boundary' if stop=='session-boundary' else 'resource-blocked' if stop=='resource-blocked' else 'failed' if failed else 'complete', 'phase':'cleanup'})
            return {'schema_version': 1, 'kind': 'hpo-campaign-result', 'campaign_dir': str(directory),
                'campaign_id': campaign['campaign_id'], 'study_identity': campaign['study_identity'],
                'status': 'paused' if stop == 'session-boundary' else 'blocked' if stop == 'resource-blocked' else 'complete',
                'completed_in_session': completed, 'failed_in_session': failed, 'qualification': 'exploratory-only'}


def inspect(directory: Path, *, max_owned_bytes: int) -> Json:
    with CampaignStore(directory, mode='resume', max_owned_bytes=max_owned_bytes) as store:
        campaign, _ = store.get('campaign.json')
        _checked(campaign, 'hpo-campaign', _CAMPAIGN_KEYS)
        proposals = []
        for name in store.records():
            if name.startswith('proposals/') and name.endswith('/proposal.json'):
                value, sha = store.get(name)
                root = name.rsplit('/', 1)[0]
                terminal = store.get(root + '/tell.receipt.json')[0] if store.exists(root + '/tell.receipt.json') else None
                proposals.append({'path': root, 'proposal_sha256': sha, 'effective_settings_sha256': value['effective_settings_sha256'],
                    'recorded_state': None if terminal is None else terminal['state'],
                    'recorded_value': None if terminal is None else terminal['value']})
        return {'schema_version': 1, 'kind': 'hpo-campaign-inspection', 'inspection_scope': 'structural-only',
            'qualification': 'not-performed', 'campaign_id': campaign['campaign_id'],
            'study_identity': campaign['study_identity'], 'proposals': proposals}


def _closed_trial(path: Path, stack: ExitStack, stores: dict[Path, CampaignStore]) -> tuple[CampaignStore, LoadedHPOConfig, Json, Proposal]:
    from gf_cli.hpo_evaluation import verify_trial_evaluation
    if path.parent.name != 'proposals' or not path.name.isdecimal():
        raise ContractError('comparison input must be an explicit closed proposal directory')
    directory = path.parent.parent
    if directory not in stores:
        stores[directory] = stack.enter_context(CampaignStore(directory, mode='resume',
            max_owned_bytes=HPOOperatingConfig().parent_budget_bytes))
    store = stores[directory]
    campaign, _ = store.get('campaign.json')
    _checked(campaign, 'hpo-campaign', _CAMPAIGN_KEYS)
    inputs = [row for row in campaign['config_inputs'] if row.get('kind') == 'ppo-hpo']
    if not inputs:
        raise ContractError('closed comparison trial requires its original explicit HPO configuration input')
    source = inputs[-1]
    config_path = Path(source['path'])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != source['sha256']:
        raise ContractError('original comparison campaign configuration changed')
    loaded = load_hpo_config(config_path)
    study = CampaignStudy(store, loaded, campaign)
    number = int(path.name)
    intent, _ = study._intent(number)
    proposal = study._proposal(number, intent)
    tell, _ = store.get(f'proposals/{number:06}/tell.receipt.json')
    _, score, evaluation_sha = verify_trial_evaluation(store, loaded, campaign, proposal,
        deadline=time.monotonic() + loaded.config.run_wall_seconds)
    if tell.get('state') != 'COMPLETE' or tell.get('value') != score or tell.get('evaluation_receipt_sha256') != evaluation_sha:
        raise ContractError('comparison requires a complete independently verified closed trial')
    if proposal.trial_number is not None:
        if not Path(loaded.config.storage.optuna_path).is_file():
            raise ContractError('closed comparison trial has lost its actual Optuna storage')
        with study.study(number) as actual:
            study._known_trials(actual)
            rows = actual.get_trials()
            if proposal.trial_number >= len(rows) or rows[proposal.trial_number].state.name != 'COMPLETE':
                raise ContractError('comparison trial is not actually complete in its bound study')
    return store, loaded, campaign, proposal


async def compare_trials(common: LoadedHPOConfig, left_path: Path, right_path: Path, output: Path, *, progress: str = 'auto') -> Json:
    require_hpo_runtime_execution(common.config)
    from gf_cli.hpo_evaluation import evaluate_comparison
    from gf_evaluation.hpo_compare import comparison_table
    validate(common)
    check_storage(common.config.storage, campaign_dir=output, originals_root=_originals(common),
        deadline=time.monotonic() + common.config.run_wall_seconds)
    conditions = _conditions(common)
    condition_sha = digest(conditions)
    with ExitStack() as stack:
        stores: dict[Path, CampaignStore] = {}
        left_store, left, left_campaign, left_proposal = _closed_trial(left_path, stack, stores)
        right_store, right, right_campaign, right_proposal = _closed_trial(right_path, stack, stores)
        if (common.config.decision_budget != left.config.decision_budget or common.config.decision_budget != right.config.decision_budget
                or common.config.evaluation_game_seeds != left.config.evaluation_game_seeds
                or common.config.evaluation_game_seeds != right.config.evaluation_game_seeds
                or canonical(asdict(common.config.operating)) != canonical(asdict(left.config.operating))
                or canonical(asdict(common.config.storage)) != canonical(asdict(left.config.storage))
                or canonical(asdict(common.config.inference)) != canonical(asdict(left.config.inference))
                or common.config.run_wall_seconds != left.config.run_wall_seconds):
            raise ContractError('initial comparison requires the left operating/storage/inference profile and common exact D/panel')
        for source in (left_campaign, right_campaign):
            for field in ('producer_identity', 'integration_identity', 'action_config_sha256', 'action_contract',
                          'rule_signature', 'input_recipe', 'inference_binding', 'cuda_visible_devices', 'native_identity'):
                if canonical(source['conditions'].get(field)) != canonical(conditions.get(field)):
                    raise ContractError('comparison parent public/runtime/inference conditions are incompatible')
        comparison_id = str(uuid.uuid4())
        with CampaignStore(output, mode='fresh', max_owned_bytes=common.config.operating.parent_budget_bytes) as store:
            from gf_cli.hpo_progress import ProgressOwner
            owner = ProgressOwner(store,mode=progress)
            async with owner.live():
                try:
                    header = {'schema_version': 1, 'kind': 'hpo-comparison-request', 'comparison_id': comparison_id,
                        'config_sha256': common.sha256, 'conditions_sha256': condition_sha, 'conditions': conditions,
                        'left_campaign_id': left_campaign['campaign_id'], 'right_campaign_id': right_campaign['campaign_id'],
                        'left_proposal_sha256': left_proposal.sha256, 'right_proposal_sha256': right_proposal.sha256,
                        'left_profile': left.effective, 'right_profile': right.effective,
                        'left_training_seeds': list(left.config.learning_seeds), 'right_training_seeds': list(right.config.learning_seeds)}
                    store.put('comparison.intent.json', header)
                    panels = []
                    matchup_receipts = {}
                    for left_seed in left.config.learning_seeds:
                        left_training = verify_training_receipt(left_store, left, left_campaign, left_proposal, left_seed,
                            deadline=time.monotonic() + common.config.run_wall_seconds)
                        for right_seed in right.config.learning_seeds:
                            right_training = verify_training_receipt(right_store, right, right_campaign, right_proposal, right_seed,
                                deadline=time.monotonic() + common.config.run_wall_seconds)
                            owner.emit({'phase':'evaluate','left_seed':left_seed,'right_seed':right_seed,
                                'left_model':left_proposal.settings.model.model_id,'right_model':right_proposal.settings.model.model_id})
                            _storage_boundary(store, common, operation_id=str(uuid.uuid4()))
                            panel = await evaluate_comparison(store, left, right, left_proposal, right_proposal,
                                left_training, right_training, comparison_id=comparison_id, conditions_sha256=condition_sha,
                                output=output / 'evaluations' / str(uuid.uuid4()),
                                evaluation_runtime=evaluation_runtime(common.config),progress=owner.emit)
                            panels.append(panel)
                            if hasattr(common.config,'runtime'):
                                matchup_receipts[f'{left_seed}-{right_seed}'] = store.get(f'matchups/{left_seed}-{right_seed}/receipt.json')[1]
                    table = comparison_table(left.config.learning_seeds, right.config.learning_seeds,
                        common.config.evaluation_game_seeds, common.config.decision_budget, condition_sha, tuple(panels))
                    if hasattr(common.config,'runtime'):
                        table.update(schema_version=2,matchup_receipts=matchup_receipts)
                    table.update(left_profile=left.effective, right_profile=right.effective, request_sha256=store.get('comparison.intent.json')[1])
                    store.put('comparison.json', table)
                    owner.emit({'phase':'cleanup','status':'complete'})
                    return table
                except BaseException:
                    owner.emit({'status':'failed'})
                    raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='gf-bots hpo')
    commands = parser.add_subparsers(dest='command', required=True)
    validation = commands.add_parser('validate')
    validation.add_argument('--config', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--mode', choices=('fresh', 'resume'), required=True)
    run.add_argument('--campaign-dir', type=Path, required=True)
    run.add_argument('--seed-settings', type=Path, nargs='+', default=[])
    inspection = commands.add_parser('inspect')
    inspection.add_argument('--campaign-dir', type=Path, required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--config', type=Path, required=True)
    comparison.add_argument('--left', type=Path, required=True)
    comparison.add_argument('--right', type=Path, required=True)
    comparison.add_argument('--output', type=Path, required=True)
    recovery = commands.add_parser('recover', description='Resolve admission through a confirmed restart or an explicit '
        'operator release. Release accepts possible residual resource use and does not certify termination.')
    recovery.add_argument('--ticket', required=True)
    recovery.add_argument('--operator', required=True)
    resolution = recovery.add_mutually_exclusive_group(required=True)
    resolution.add_argument('--confirm-environment-restarted', action='store_true')
    resolution.add_argument('--release-unconfirmed', action='store_true',
        help='authorize new admission without confirming termination of the previous job')
    recovery.add_argument('--reason', help='mandatory reason for --release-unconfirmed; rejected for restart')
    for command in (run, comparison):
        command.add_argument('--progress', choices=('auto','plain','off'), default='auto')
    watcher = commands.add_parser('watch', description='Read-only diagnostic observer; no execution locks')
    watcher.add_argument('--path',type=Path,required=True)
    watcher.add_argument('--interval',type=float,default=1.)
    watcher.add_argument('--once',action='store_true')
    watcher.add_argument('--json',action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'recover':
        if args.release_unconfirmed and args.reason is None:
            recovery.error('--release-unconfirmed requires --reason')
        if args.confirm_environment_restarted and args.reason is not None:
            recovery.error('--reason is only valid with --release-unconfirmed')
    try:
        if args.command == 'watch':
            from gf_cli.hpo_progress import watch
            try:
                return asyncio.run(watch(args.path,interval=args.interval,once=args.once,as_json=args.json))
            except KeyboardInterrupt:
                return 0
        if args.command == 'inspect':
            result = inspect(args.campaign_dir.resolve(), max_owned_bytes=HPOOperatingConfig().parent_budget_bytes)
        elif args.command == 'recover':
            from gf_training.hpo_gpu_lease import (
                RELEASE_CONFIRMATION,
                RESTART_CONFIRMATION,
                confirm_environment_restart,
                release_unconfirmed_admission,
            )
            if args.release_unconfirmed:
                result = asdict(release_unconfirmed_admission(expected_ticket=args.ticket, operator=args.operator,
                    reason=args.reason, confirmation=RELEASE_CONFIRMATION))
            else:
                result = asdict(confirm_environment_restart(expected_ticket=args.ticket, operator=args.operator,
                    confirmation=RESTART_CONFIRMATION))
        else:
            loaded = load_hpo_config(args.config)
            if args.command == 'validate':
                result = validate(loaded)
            elif args.command == 'run':
                result = asyncio.run(run_campaign(loaded, args.campaign_dir.resolve(), mode=args.mode,
                    seed_paths=tuple(path.resolve() for path in args.seed_settings),progress=args.progress))
            else:
                result = asyncio.run(compare_trials(loaded, args.left.resolve(), args.right.resolve(), args.output.resolve(),progress=args.progress))
        print(canonical(result))
        return 2 if result.get('status') == 'blocked' else 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(canonical({'schema_version': 1, 'kind': 'hpo-command-failure', 'error': type(error).__name__, 'message': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
