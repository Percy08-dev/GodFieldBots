"""Fixed supervised campaign application and canonical bootstrap factories.

Decoded historical records are DATA; actual current owners perform every audit.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gf_cli.hpo_config import LoadedHPOConfig
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer


class _ScheduledSeedDoesNotFit(TimeoutError):
    """A fresh seed missed the soft budget before publishing anything."""


def campaign_snapshot(loaded: LoadedHPOConfig) -> Json:
    from gf_common.rust_player import validate_loaded_campaign
    if (type(loaded) is not LoadedHPOConfig or loaded.config.schema_version != 6
            or canonical(asdict(loaded.config)) != canonical(loaded.effective)
            or canonical(asdict(loaded.base.config)) != canonical(loaded.base.effective)):
        raise ContractError('actual native6 loaded campaign CONFIG required')
    row = {'schema_version': 1, 'kind': 'rust-player-loaded-campaign-v1',
        'campaign_effective': loaded.effective, 'campaign_sha256': loaded.sha256,
        'base_effective': loaded.base.effective, 'base_sha256': loaded.base.sha256,
        'inputs': list(loaded.inputs)}
    validate_loaded_campaign(row)
    return clone(row)


def _raw_space(value: Json) -> Json:
    """Adapt only normalized CONFIG shape; the existing search parser owns rules."""
    from gf_training.hpo_config import CoupledParameter, SearchParameter, SearchSpace
    closed(value, set(SearchSpace.__dataclass_fields__))
    parameters = []
    for row in value['parameters']:
        closed(row, set(SearchParameter.__dataclass_fields__))
        raw = {'path': row['path'], 'kind': row['kind'], 'when': dict(row['when'])}
        if row['kind'] == 'categorical': raw['choices'] = row['choices']
        elif row['kind'] == 'fixed': raw['value'] = row['value']
        else:
            raw.update(low=row['low'], high=row['high'])
            if row['step'] is not None: raw['step'] = row['step']
        parameters.append(raw)
    coupled = []
    for row in value['coupled']:
        closed(row, set(CoupledParameter.__dataclass_fields__))
        coupled.append({**row, 'when': dict(row['when'])})
    return {'parameters': parameters, 'coupled': coupled}


def load_campaign_snapshot(value: Any) -> LoadedHPOConfig:
    """Reconstruct only typed CONFIG from pinned bytes; never reopen TOML paths."""
    from gf_cli.config import _normalized_config
    from gf_cli.hpo_config import campaign_config
    from gf_common.rust_player import hex_identity, text, validate_loaded_campaign
    try:
        row = validate_loaded_campaign(value)
        inputs = []
        for item in row['inputs']:
            closed(item, {'path', 'sha256'}, {'kind'})
            text(item['path']); hex_identity(item['sha256'])
            if 'kind' in item: text(item['kind'])
            inputs.append(dict(item))
        # Snapshot encodes the original combined sequence. It does not encode a
        # recoverable base-only subset; both CONFIG containers retain that DATA.
        base = _normalized_config(row['base_effective'], data=row['base_effective'], inputs=tuple(inputs))
        raw = clone(row['campaign_effective'])
        if raw['schema_version'] != 6:
            raise ContractError('explicit native6 campaign snapshot required')
        raw.pop('base_config_sha256')
        raw['space'] = _raw_space(raw['space'])
        loaded = campaign_config(raw, origin=Path('/'), _pinned_base=base)
        if (canonical(asdict(base.config)) != canonical(row['base_effective']) or base.sha256 != row['base_sha256']
                or canonical(asdict(loaded.config)) != canonical(row['campaign_effective'])
                or loaded.sha256 != row['campaign_sha256'] or canonical(loaded.inputs) != canonical(row['inputs'])):
            raise ContractError('canonical campaign CONFIG/default/derived/hash differs')
        return loaded
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError('complete canonical campaign CONFIG required') from error


def campaign_session_clock(loaded: LoadedHPOConfig, *, supervision_id: str, clock_binding: Json) -> Json:
    from gf_common.rust_player import deadline_bits, deadline_seconds, validate_campaign_clock
    campaign_snapshot(loaded)
    closed(clock_binding, {'boot_id', 'monotonic_before_bits', 'monotonic_after_bits', 'unix_anchor_bits'})
    anchor = deadline_seconds(clock_binding['monotonic_after_bits'])
    soft = anchor + loaded.config.session_seconds
    # Both current train and evaluation materializations retain this TOTAL wall.
    wall = loaded.config.run_wall_seconds
    hard = soft + wall
    if not anchor < soft < hard:
        raise ContractError('original campaign session ordering is not representable')
    row = {'schema_version': 1, 'kind': 'rust-player-campaign-clock-v1', 'supervision_id': supervision_id,
        **clock_binding, 'soft_dispatch_finish_bits': deadline_bits(soft), 'hard_session_finish_bits': deadline_bits(hard),
        'max_operation_wall_seconds': wall, 'mapping_recipe': 'soft-dispatch-plus-max-operation-wall-v1'}
    return validate_campaign_clock(row)


def initial_campaign_operation(session_clock: Json, *, operation_id: str, bootstrap_path: Path,
                               bootstrap_sha256: str, runtime_binding_sha256: str) -> Json:
    from gf_common.rust_player import validate_campaign_clock, validate_operation
    validate_campaign_clock(session_clock)
    deadlines = {**dict.fromkeys(('admission_bits', 'execution_bits', 'term_bits', 'kill_bits'),
        session_clock['soft_dispatch_finish_bits']), 'finish_bits': session_clock['hard_session_finish_bits']}
    return validate_operation({'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': operation_id,
        'run_id': None, 'mode': 'hpo', 'operation_type': 'campaign-continue', 'config_path': str(bootstrap_path),
        'config_sha256': bootstrap_sha256, 'runtime_binding_sha256': runtime_binding_sha256,
        'deadlines': deadlines, 'parent_directory': None, 'expected_context': None, 'execution_request': None},
        session_finish=session_clock['hard_session_finish_bits'])


def load_seed_snapshot(value: Any, config: Any, *, study_identity: str) -> tuple[Any, ...]:
    """Existing imported settings CONFIG only, with no late path/model-state read."""
    from gf_cli.hpo_campaign import encode_settings
    from gf_common.rust_player import validate_seed_settings
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.hpo_config import model_config_v4
    from gf_training.hpo_config import effective_settings, hpo_training_config
    row = validate_seed_settings(value)
    result, identities = [], set()
    for source in row['sources']:
        settings = source['settings']
        model = model_config_v4(settings['model'], layout=layout)
        parsed = effective_settings(model, hpo_training_config(settings['training']), {}, layout=layout)
        if (canonical(asdict(parsed)) != canonical(settings)
                or canonical(asdict(model.resources)) != canonical(asdict(config.model.resources))
                or parsed.identity in identities or source['origin']['study_identity'] == study_identity
                or encode_settings(config, parsed) is None):
            raise ContractError('pinned seed settings/config/resources/fresh-study identity differs')
        identities.add(parsed.identity)
        result.append(parsed)
    return tuple(result)


def _store_mapping(store: Any) -> Json:
    from gf_cli.hpo_storage import CampaignStore
    if type(store) is not CampaignStore:
        raise ContractError('actual same-thread campaign store required')
    store._require()
    if store._lock is None:
        raise ContractError('actual original campaign descriptor required')
    fd = store._lock.fileno()
    actual = os.fstat(fd)
    return {'lock_kind': 'campaign', 'fd': fd, 'device': actual.st_dev, 'inode': actual.st_ino}


def _seed_candidate(store: Any, campaign: Json, number: int, seed: int, *, awaiting: Json | None = None) -> Json:
    """Structural receipt DATA only; never substitutes for the full owned audit."""
    from gf_training.rust_player_evaluation import validate_seed_training
    try:
        intent, intent_sha = store.get(f'proposals/{number:06}/intent.json')
        proposal, proposal_sha = store.get(f'proposals/{number:06}/proposal.json')
        receipt, _ = store.get(f'proposals/{number:06}/seeds/{seed}/training.json')
    except FileNotFoundError as error:
        raise ContractError('historical awaiting lacks a complete closed seed candidate') from error
    validate_seed_training(receipt)
    if (intent['campaign_id'] != campaign['campaign_id'] or intent['proposal_no'] != number
            or proposal['intent_sha256'] != intent_sha or receipt['proposal_sha256'] != proposal_sha
            or receipt['training_seed'] != seed or awaiting is not None
            and any(receipt[key] != awaiting[key] for key in ('run_path', 'run_id', 'config_sha256', 'runtime_binding_sha256'))):
        raise ContractError('structural closed seed/awaiting/proposal candidate differs')
    return receipt


def _resume_history(store: Any, loaded: LoadedHPOConfig, conditions: Json) -> tuple[Json, list[Json], str | None]:
    from gf_cli.hpo_campaign import CampaignStudy
    from gf_common.rust_player import (
        validate_campaign_awaiting,
        validate_campaign_bootstrap,
        validate_campaign_clock,
        validate_campaign_cursor,
    )
    campaign, _ = store.get('campaign.json')
    CampaignStudy(store, loaded, campaign)  # Existing closed CONFIG/study binding, no Optuna operation.
    sources = campaign['conditions'].get('initial_settings', [])
    if canonical({**conditions, 'initial_settings': sources}) != canonical(campaign['conditions']):
        raise ContractError('original campaign conditions differ; a fresh study is required')
    load_seed_snapshot({'schema_version': 1, 'kind': 'rust-player-seed-settings-v1', 'sources': sources},
        loaded.config, study_identity=campaign['study_identity'])
    cursor, cursor_sha = (store.get('supervision/current.json') if store.exists('supervision/current.json') else (None, None))
    if cursor is not None:
        validate_campaign_cursor(cursor)
        relative = 'supervision/' + cursor['supervision_id'] + '/'
        clock, clock_sha = store.get(relative + 'session-clock.json')
        bootstrap, _ = store.get(relative + 'bootstrap.json')
        validate_campaign_clock(clock); validate_campaign_bootstrap(bootstrap)
        if (cursor['campaign_id'] != campaign['campaign_id'] or cursor['session_clock_sha256'] != clock_sha
                or bootstrap['campaign_directory'] != str(store.directory)
                or bootstrap['supervision_id'] != cursor['supervision_id'] or bootstrap['session_clock_sha256'] != clock_sha):
            raise ContractError('original campaign cursor/session differs')
        snapshot, snapshot_sha = store.get(relative + 'loaded-campaign.json')
        seed, seed_sha = store.get(relative + 'seed-settings.json')
        if (snapshot_sha != bootstrap['loaded_campaign_sha256'] or seed_sha != bootstrap['seed_settings_sha256']
                or load_campaign_snapshot(snapshot).sha256 != loaded.sha256):
            raise ContractError('original immutable campaign CONFIG differs')
        load_seed_snapshot(seed, loaded.config, study_identity=campaign['study_identity'])
    # Structural candidates only. Actual old parent/Source/namespace verification
    # belongs to the authenticated new CPU controller, before its cursor CAS.
    candidates: dict[str, str] = {}
    paused: set[tuple[int, int]] = set()
    terminal: set[tuple[int, int]] = set()
    completed: set[tuple[int, int]] = set()
    for name in store.records():
        parts = name.split('/')
        if len(parts) == 5 and parts[0] == 'proposals' and parts[2] == 'seeds' and parts[4] == 'training.json':
            subject = int(parts[1]), int(parts[3])
            _seed_candidate(store, campaign, *subject)
            completed.add(subject)
        if name.endswith(('/training.intent.json', '/evaluation.intent.json')):
            receipt = name.removesuffix('.intent.json') + '.json'
            diagnostic = _checkpoint_intent_path(parts)
            if (not store.exists(receipt)
                    and (not diagnostic or not _checkpoint_failed_candidate(store, name))):
                raise ContractError('interrupted seed intent has no closed receipt candidate')
        if len(parts) != 4 or parts[0] != 'supervision' or parts[2] != 'awaiting':
            continue
        awaiting, awaiting_sha = store.get(name)
        validate_campaign_awaiting(awaiting)
        if awaiting['campaign_id'] != campaign['campaign_id'] or awaiting['supervision_id'] != parts[1]:
            raise ContractError('foreign structural awaiting campaign/session')
        subject = awaiting['proposal_no'], awaiting['learning_seed']
        disposition_path = '/'.join((parts[0], parts[1], 'terminal', parts[3]))
        pause_path = '/'.join((parts[0], parts[1], 'checkpoint-pauses', parts[3]))
        if store.exists(disposition_path):
            from gf_common.rust_player import (
                validate_campaign_disposition,
                validate_campaign_disposition_ack,
            )
            disposition, disposition_sha = store.get(disposition_path)
            validate_campaign_disposition(disposition)
            ack, _ = store.get('/'.join((parts[0], parts[1], 'disposition-ack-' + disposition['invocation_id'] + '.json')))
            validate_campaign_disposition_ack(ack)
            if (disposition['awaiting_path'] != str(store.directory / name)
                    or disposition['awaiting_sha256'] != awaiting_sha or ack['disposition_sha256'] != disposition_sha
                    or ack['operation_sha256'] != awaiting['operation_sha256']):
                raise ContractError('structural terminal ACK candidate differs')
            terminal.add(subject)
        elif store.exists(pause_path):
            pause, _ = store.get(pause_path)
            _checkpoint_pause_candidate(loaded, awaiting_sha, pause)
            paused.add(subject)
            failure_path = f"proposals/{subject[0]:06}/failure.json"
            if store.exists(failure_path):
                failure, _ = store.get(failure_path)
                boundary = failure.get('boundary')
                if (type(boundary) is dict and boundary.get('phase') == 'checkpoint-time-cap'
                        and boundary.get('awaiting_sha256') == awaiting_sha):
                    terminal.add(subject)
        else:
            _seed_candidate(store, campaign, awaiting['proposal_no'], awaiting['learning_seed'], awaiting=awaiting)
        candidates[str(store.directory / name)] = awaiting_sha
    if paused - completed - terminal:
        raise ContractError('interrupted checkpoint training cannot resume under another supervisor root')
    if (cursor is not None and cursor['awaiting_path'] is not None
            and candidates.get(cursor['awaiting_path']) != cursor['awaiting_sha256']):
        raise ContractError('current awaiting is unresolved/mismatched')
    return campaign, sources, cursor_sha


def _checkpoint_intent_path(parts: list[str]) -> bool:
    return (len(parts) == 7 and parts[0] == 'proposals' and parts[2] == 'seeds'
        and parts[4] == 'checkpoint-evaluations' and parts[5].isascii() and parts[5].isdecimal()
        and (parts[5] == '0' or not parts[5].startswith('0')) and parts[6] == 'evaluation.intent.json')


def _checkpoint_failed_candidate(store: Any, name: str, *, campaign: Json | None = None) -> bool:
    from gf_cli.hpo_campaign import _checked

    directory = name.removesuffix('/evaluation.intent.json')
    if not store.exists(directory + '/status.json'):
        return False
    status, _ = store.get(directory + '/status.json')
    _checked(status, 'hpo-checkpoint-evaluation-status',
        {'generation', 'status', 'reason', 'elapsed_seconds', 'gpu_termination_sha256'})
    intent, _ = store.get(name)
    copied, checksum = store.get(directory + '/evaluation-failed-gpu-termination.json')
    _checked(copied, 'hpo-failed-evaluation-termination',
        {'evaluation_id', 'gpu_termination', 'error_type', 'evaluation_eligible'})
    valid = (status['status'] == 'failed' and str(status['generation']) == directory.rsplit('/', 1)[-1]
        and status['gpu_termination_sha256'] == checksum and copied['evaluation_id'] == intent['evaluation_id']
        and copied['evaluation_eligible'] is False and copied['error_type'] == status['reason'])
    if valid and campaign is not None:
        from gf_cli.hpo_evaluation import verify_gpu_termination

        checkpoint, checkpoint_sha = store.get(directory + '/checkpoint.json')
        if checkpoint_sha != intent['checkpoint_receipt_sha256']:
            raise ContractError('diagnostic failure intent differs from its checkpoint')
        verify_gpu_termination(copied['gpu_termination'], campaign_id=campaign['campaign_id'],
            operation_id=intent['evaluation_id'], run_id=checkpoint['run_id'], evaluation_id=intent['evaluation_id'])
    return valid


@dataclass(frozen=True, eq=False)
class PreparedCampaign:
    store: Any
    loaded: LoadedHPOConfig
    operation: Json
    operation_path: Path
    bootstrap: Json
    bootstrap_sha256: str
    session_clock: Json
    launch: Json
    companion: Any
    _launch_json: str | None = field(default=None, init=False, repr=False)
    _factory_companion: Any = field(default=None, init=False, repr=False)
    _companion_json: str | None = field(default=None, init=False, repr=False)
    _resume_cursor_sha: str | None = field(default=None, init=False, repr=False)

    def require(self) -> None:
        from gf_cli.rust_player_entry import _boot_id
        from gf_common.rust_player import (
            deadline_seconds,
            validate_campaign_bootstrap,
            validate_campaign_clock,
            validate_launch,
        )
        if (self._launch_json is None or self._factory_companion is None
                or self.companion is not self._factory_companion
                or canonical(self.companion.manifest) != self._companion_json
                or canonical(self.launch) != self._launch_json):
            raise ContractError('actual fixed campaign factory preparation required')
        if self.launch['locks'] != [_store_mapping(self.store)]:
            raise ContractError('campaign launch lost its SAME original descriptor')
        campaign_snapshot(self.loaded)
        validate_launch(self.launch)
        root = self.store.directory / 'supervision' / self.launch['supervision_id']
        relative = str(root.relative_to(self.store.directory)) + '/'
        bootstrap, bootstrap_sha = self.store.get(relative + 'bootstrap.json')
        session, session_sha = self.store.get(relative + 'session-clock.json')
        operation, operation_sha = self.store.get(str(self.operation_path.relative_to(self.store.directory)))
        validate_campaign_bootstrap(bootstrap); validate_campaign_clock(session)
        if (bootstrap != self.bootstrap or bootstrap_sha != self.bootstrap_sha256
                or session != self.session_clock or session_sha != bootstrap['session_clock_sha256']
                or operation != self.operation or operation_sha != self.launch['operation_sha256']
                or self.operation_path != root / 'operation-initial.json'
                or self.launch['record_directory'] != str(root)
                or operation['config_path'] != str(root / 'bootstrap.json')
                or operation['config_sha256'] != bootstrap_sha
                or session['boot_id'] != _boot_id()):
            raise ContractError('actual immutable campaign bootstrap/session/operation differs')
        snapshot, snapshot_sha = self.store.get(relative + 'loaded-campaign.json')
        seeds, seed_sha = self.store.get(relative + 'seed-settings.json')
        if (snapshot_sha != bootstrap['loaded_campaign_sha256'] or seed_sha != bootstrap['seed_settings_sha256']
                or load_campaign_snapshot(snapshot).sha256 != self.loaded.sha256):
            raise ContractError('pinned campaign CONFIG changed before exec')
        campaign, _ = self.store.get('campaign.json')
        if self.bootstrap['mode'] == 'resume':
            actual_sha = self.store.get('supervision/current.json')[1] if self.store.exists('supervision/current.json') else None
            if actual_sha != self._resume_cursor_sha:
                raise ContractError('original resume cursor changed before exec')
        load_seed_snapshot(seeds, self.loaded.config, study_identity=campaign['study_identity'])
        if time.monotonic() >= deadline_seconds(session['hard_session_finish_bits']):
            raise ContractError('original campaign hard deadline expired before exec')

    def execute(self) -> None:
        from gf_simulator_client.rust_player_companion import exec_supervisor
        self.require()
        exec_supervisor(self._factory_companion, self.launch)
        raise ContractError('fixed campaign supervisor exec unexpectedly returned')

    def __reduce__(self) -> Any:
        raise TypeError('actual campaign factory owner cannot be copied/serialized')


@contextmanager
def prepare_campaign(loaded: LoadedHPOConfig, directory: Path, *, mode: str,
                     seed_paths: tuple[Path, ...] = (), progress: str = 'auto') -> Iterator[PreparedCampaign]:
    from gf_cli import hpo
    from gf_cli.hpo_campaign import initialize_campaign
    from gf_cli.hpo_config import RustPlayerHPOCampaignConfig
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_entry import (
        _capture_clock,
        check_native_collection_storage,
        require_native_hpo_storage,
    )
    from gf_common.card_instance import DREAM_COMPAT_PROFILE, ActionConfigV3
    from gf_common.rust_player import BUILD_KEYS, runtime_binding, validate_campaign_bootstrap
    from gf_models.rust_player import sampling_runtime_identity
    from gf_simulator_client.rust_player_companion import discover_companion
    from gf_training.native_backend import binary_identity, require_native_concealed_draw
    from gf_training.rust_player_supervision import runtime_producer
    if mode not in ('fresh', 'resume') or mode == 'resume' and seed_paths or progress not in ('auto', 'plain', 'off'):
        raise ContractError('explicit campaign mode/progress; seed settings are fresh-only')
    snapshot = campaign_snapshot(loaded)
    load_campaign_snapshot(snapshot)
    config = loaded.config
    if not isinstance(config, RustPlayerHPOCampaignConfig):
        raise ContractError('actual native6 campaign configuration required')
    directory = hpo.campaign_directory(loaded, directory)
    check_native_collection_storage(getattr(loaded.base.config, 'storage', None), directory,
                                    catalog_root=hpo._originals(loaded))
    hpo.validate(loaded, storage_probe=False)
    hpo.check_storage(config.storage, campaign_dir=directory, originals_root=hpo._originals(loaded),
                      deadline=time.monotonic() + config.run_wall_seconds)
    conditions = hpo._conditions(loaded)
    _settings, sources = hpo._settings_inputs(seed_paths, loaded) if mode == 'fresh' else ((), [])
    native = binary_identity()
    actions = loaded.base.config.actions
    if isinstance(actions, ActionConfigV3) and actions.action_contract.profile_id == DREAM_COMPAT_PROFILE:
        require_native_concealed_draw(native['build'])
    require_native_hpo_storage(config.storage, native)
    if (config.checkpoint_evaluation is not None
            and native['build'].get('card_instance_hpo_checkpoint_evaluation') != 'hpo-checkpoint-evaluation-v1'):
        raise ContractError('scheduled Rust HPO requires a checkpoint-evaluation runtime wheel')
    companion = discover_companion(native)
    bound = runtime_binding(config.runtime, runtime_producer(companion,
        sampling_runtime_sha256=sampling_runtime_identity()))
    supervision_id = uuid.uuid4().hex
    binding = _capture_clock()  # Original SESSION anchor, before actual store ownership/publication.
    session = campaign_session_clock(loaded, supervision_id=supervision_id, clock_binding=binding)
    with CampaignStore(Path(directory).absolute(), mode=mode,
            max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        if mode == 'fresh':
            conditions['initial_settings'] = sources
            if 'producer_identity' in conditions:
                from gf_common.rust_player import deadline_seconds
                from gf_training.ppo_run import source_identity
                conditions['producer_identity'] = source_identity(deadline_seconds(session['hard_session_finish_bits']))
            campaign = initialize_campaign(store, loaded, conditions)
            previous_cursor_sha = None
        else:
            campaign, sources, previous_cursor_sha = _resume_history(store, loaded, conditions)
        seed_snapshot = {'schema_version': 1, 'kind': 'rust-player-seed-settings-v1', 'sources': sources}
        load_seed_snapshot(seed_snapshot, loaded.config, study_identity=campaign['study_identity'])
        relative = 'supervision/' + supervision_id + '/'
        root = store.directory / relative
        loaded_sha = store.put(relative + 'loaded-campaign.json', snapshot)
        seed_sha = store.put(relative + 'seed-settings.json', seed_snapshot)
        clock_sha = store.put(relative + 'session-clock.json', session)
        bootstrap = {'schema_version': 1, 'kind': 'rust-player-campaign-bootstrap-v1',
            'supervision_id': supervision_id, 'campaign_directory': str(store.directory), 'mode': mode,
            'loaded_campaign_path': str(root / 'loaded-campaign.json'), 'loaded_campaign_sha256': loaded_sha,
            'seed_settings_path': str(root / 'seed-settings.json'), 'seed_settings_sha256': seed_sha,
            'progress': progress, 'session_clock_path': str(root / 'session-clock.json'), 'session_clock_sha256': clock_sha,
            'cursor_path': str(store.directory / 'supervision/current.json')}
        validate_campaign_bootstrap(bootstrap)
        bootstrap_sha = store.put(relative + 'bootstrap.json', bootstrap)
        operation = initial_campaign_operation(session, operation_id=uuid.uuid4().hex,
            bootstrap_path=root / 'bootstrap.json', bootstrap_sha256=bootstrap_sha, runtime_binding_sha256=digest(bound))
        operation_path = root / 'operation-initial.json'
        operation_sha = store.put(relative + 'operation-initial.json', operation)
        if mode == 'fresh':
            store.update_supervision_cursor({'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
                'campaign_id': campaign['campaign_id'], 'supervision_id': supervision_id, 'session_clock_sha256': clock_sha,
                'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None, 'phase': 'control'},
                expected_sha256=previous_cursor_sha)
        launch = {'schema_version': 1, 'kind': 'rust-player-launch-v1', 'supervision_id': supervision_id,
            'python_executable': str(Path(sys.executable).absolute()), 'operation_path': str(operation_path),
            'operation_sha256': operation_sha, 'record_directory': str(root), 'mode': 'hpo',
            'session_deadlines': {'finish_bits': session['hard_session_finish_bits']}, 'clock_binding': binding,
            'binding': {'python_abi': native['python_abi'], 'extension_sha256': native['extension_sha256'],
                'build': {key: native['build'][key] for key in BUILD_KEYS}}, 'locks': [_store_mapping(store)]}
        prepared = PreparedCampaign(store, loaded, operation, operation_path, bootstrap, bootstrap_sha, session, launch, companion)
        object.__setattr__(prepared, '_launch_json', canonical(launch))
        object.__setattr__(prepared, '_factory_companion', companion)
        object.__setattr__(prepared, '_companion_json', canonical(companion.manifest))
        object.__setattr__(prepared, '_resume_cursor_sha', previous_cursor_sha)
        prepared.require()
        yield prepared


def exec_campaign(loaded: LoadedHPOConfig, directory: Path, *, mode: str,
                  seed_paths: tuple[Path, ...] = (), progress: str = 'auto') -> None:
    with prepare_campaign(loaded, directory, mode=mode, seed_paths=seed_paths, progress=progress) as prepared:
        prepared.execute()


def _publish_campaign_training(store: Any, owner: Any, *, materialized: Any, context: Json,
                                proposal_no: int, learning_seed: int, session: Json,
                                clock_binding: Json, runtime_binding_sha256: str,
                                pause_after_update: int | None = None,
                                original_deadlines: Json | None = None) -> Json:
    """Publish CONFIG DATA under actual caller custody; this grants no admission."""
    from gf_cli.rust_player_entry import _inputs, _snapshot
    from gf_common.rust_player import (
        deadline_seconds,
        safe_component,
        validate_campaign_awaiting,
        validate_campaign_clock,
        validate_campaign_cursor,
        validate_context,
        validate_operation,
    )
    from gf_common.rust_player_clock import clock_deadlines, operation_clock
    from gf_training.lineage import require_hpo_run_owner
    store._require()
    require_hpo_run_owner(owner.directory, owner)
    validate_context(context); validate_campaign_clock(session)
    operation_id = safe_component(context['operation_id'])
    parent = Path(materialized.config.run.output_dir) / materialized.config.run.name
    if owner.directory != parent or context['run_id'] != materialized.config.run.name:
        raise ContractError('actual training parent/materialized CONFIG differs')
    relative = 'supervision/' + session['supervision_id'] + '/'
    root = store.directory / relative
    actual_session, session_sha = store.get(relative + 'session-clock.json')
    cursor, cursor_sha = store.get('supervision/current.json')
    validate_campaign_cursor(cursor)
    if (actual_session != session or cursor['campaign_id'] != context['campaign_id']
            or cursor['supervision_id'] != session['supervision_id'] or cursor['session_clock_sha256'] != session_sha
            or cursor['phase'] != 'control' or cursor['awaiting_path'] is not None or cursor['current_invocation_id'] is not None):
        raise ContractError('exact current parent-free campaign cursor/session required')
    fresh = original_deadlines is None
    deadlines = original_deadlines if not fresh else clock_deadlines(clock_binding, inputs=_inputs(materialized.config),
        post_host_seconds=materialized.config.run.post_host_seconds)
    assert deadlines is not None
    operation = {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': operation_id,
        'run_id': context['run_id'], 'mode': 'hpo', 'operation_type': 'hpo-train',
        'config_path': str(parent / '.rust-player/effective-config.json'), 'config_sha256': materialized.sha256,
        'runtime_binding_sha256': runtime_binding_sha256, 'deadlines': deadlines,
        'parent_directory': str(parent), 'expected_context': context,
        'execution_request': {'mode': 'fresh' if fresh else 'resume', 'pause_after_update': pause_after_update}}
    validate_operation(operation, session_finish=session['hard_session_finish_bits'])
    snapshot = _snapshot(materialized, owner, fresh=fresh)
    snapshot_config: Any = snapshot.config
    if not fresh and time.monotonic() >= deadline_seconds(deadlines['admission_bits']):
        raise TimeoutError('original checkpoint admission expired before operation publication')
    operation_path = root / f'operation-{operation_id}.json'
    operation_sha = store.put(relative + operation_path.name, operation)
    clock = operation_clock(operation, clock_binding=clock_binding, inputs=_inputs(snapshot_config),
        post_host_seconds=snapshot_config.run.post_host_seconds)
    store.put(relative + f'clock-{operation_id}.json', clock)
    awaiting = {'schema_version': 1, 'kind': 'rust-player-training-awaiting-v1',
        'campaign_id': context['campaign_id'], 'proposal_no': proposal_no, 'learning_seed': learning_seed,
        'run_id': context['run_id'], 'run_path': str(parent), 'config_path': operation['config_path'],
        'config_sha256': snapshot.sha256, 'runtime_binding_sha256': runtime_binding_sha256,
        'operation_path': str(operation_path), 'operation_sha256': operation_sha,
        'supervision_id': session['supervision_id'], 'session_clock_sha256': session_sha, 'created_unix': time.time()}
    validate_campaign_awaiting(awaiting)
    awaiting_path = root / 'awaiting' / f'{operation_id}.json'
    if not fresh and time.monotonic() >= deadline_seconds(deadlines['admission_bits']):
        raise TimeoutError('original checkpoint admission expired before awaiting publication')
    awaiting_sha = store.put(relative + f'awaiting/{operation_id}.json', awaiting)
    awaiting_cursor = {**cursor, 'phase': 'awaiting-host', 'awaiting_path': str(awaiting_path),
        'awaiting_sha256': awaiting_sha, 'current_invocation_id': None}
    awaiting_cursor_sha = store.update_supervision_cursor(awaiting_cursor, expected_sha256=cursor_sha)
    return {'loaded': snapshot, 'operation': operation, 'operation_path': operation_path, 'operation_sha256': operation_sha,
        'clock': clock, 'awaiting': awaiting, 'awaiting_path': awaiting_path, 'awaiting_sha256': awaiting_sha,
        'cursor': awaiting_cursor, 'cursor_sha256': awaiting_cursor_sha}


def _campaign_context(control: Any, store: Any, study: Any) -> tuple[Json, Json, Json]:
    from gf_cli.hpo_campaign import CampaignStudy
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.rust_player import validate_campaign_clock
    from gf_training.rust_player_supervision import HostControl
    if type(control) is not HostControl or type(store) is not CampaignStore or type(study) is not CampaignStudy:
        raise ContractError('actual campaign HostControl/borrowed Store/study required')
    control._require(); store._require()
    bootstrap = control.campaign_bootstrap
    if (store._control is not control or store.directory != Path(bootstrap['campaign_directory'])
            or study.store is not store):
        raise ContractError('actual SAME borrowed campaign owner/study required')
    relative = 'supervision/' + control.launch['supervision_id'] + '/'
    session, session_sha = store.get(relative + 'session-clock.json')
    validate_campaign_clock(session)
    campaign, campaign_sha = store.get('campaign.json')
    if (session_sha != bootstrap['session_clock_sha256'] or session['supervision_id'] != control.launch['supervision_id']
            or session['hard_session_finish_bits'] != control.launch['session_deadlines']['finish_bits']
            or campaign_sha != study.manifest_sha or canonical(campaign) != canonical(study.manifest)):
        raise ContractError('actual current campaign/session/proposal namespace differs')
    return bootstrap, session, campaign


def _durable_proposal(study: Any, proposal: Any) -> Any:
    from gf_cli.hpo_campaign import Proposal
    if type(proposal) is not Proposal:
        raise ContractError('exact durable campaign proposal required')
    intent, _ = study._intent(proposal.number)
    actual = study._proposal(proposal.number, intent)
    if actual != proposal:
        raise ContractError('proposal differs from SAME durable configuration/resources')
    return actual


def _admission_cursor(control: Any, store: Any, packet: Json, admission: Any) -> None:
    from gf_training.rust_player_supervision import HostAdmission
    if type(admission) is not HostAdmission:
        raise ContractError('actual current HostAdmission ACK required')
    admission.require()
    observation = admission.observation
    actual, checksum = store.get('supervision/current.json')
    if (admission._control is not control or control._admission is not admission
            or actual != packet['cursor'] or checksum != packet['cursor_sha256']
            or any(observation[key] != packet['operation'][key] for key in
                ('operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256'))
            or observation['context'] != packet['operation']['expected_context']):
        raise ContractError('actual admission differs from exact awaiting/current cursor')
    store.update_supervision_cursor({**actual, 'phase': 'awaiting-cold',
        'current_invocation_id': observation['invocation_id']}, expected_sha256=checksum)


def dispatch_campaign_training(control: Any, store: Any, study: Any, proposal: Any,
                               *, learning_seed: int, materialized: Any, resume: bool = False) -> Json:
    """One real bind→OFD transfer→admit→cursor CAS→existing training, no controller."""
    import hashlib
    from types import SimpleNamespace

    from gf_cli.hpo import _originals
    from gf_cli.hpo_config import HPODiagnosticStorageConfig, materialize_hpo_run
    from gf_cli.hpo_storage import check_run_destination
    from gf_cli.rust_player_entry import (
        _capture_clock,
        _paused_run,
        _resume_origin,
        check_native_collection_storage,
    )
    from gf_common.rust_player import deadline_seconds, runtime_binding
    from gf_models.rust_player import sampling_runtime_identity
    from gf_training.hpo_gpu_lease import require_resolved_namespace
    from gf_training.lineage import run_lock
    from gf_training.rust_player_learning import run_hpo
    from gf_training.rust_player_supervision import runtime_producer
    _bootstrap, session, campaign = _campaign_context(control, store, study)
    actual = _durable_proposal(study, proposal)
    if control._phase != 'session' or 'parent' in control._borrowed or control._parent_borrowers:
        raise ContractError('actual parent-free campaign Session required before bind')
    expected = materialize_hpo_run(study.loaded, actual.settings, learning_seed=learning_seed,
        run_id=materialized.config.run.name)
    if expected.sha256 != materialized.sha256 or canonical(expected.effective) != canonical(materialized.effective):
        raise ContractError('actual proposal/materialized training CONFIG differs')
    require_resolved_namespace()
    soft = deadline_seconds(session['soft_dispatch_finish_bits'])
    if time.monotonic() >= soft:
        raise TimeoutError('original soft campaign dispatch boundary reached')
    checkpoint = study.loaded.config.checkpoint_evaluation
    if resume and (checkpoint is None or not checkpoint.enabled):
        raise ContractError('campaign seed resume requires its scheduled checkpoint program')
    companion = SimpleNamespace(manifest={'executable_sha256': hashlib.sha256(
        Path(f'/proc/{control._parent_pid}/exe').read_bytes()).hexdigest()})
    bound = runtime_binding(materialized.config.hpo_spec['runtime'], runtime_producer(companion,
        sampling_runtime_sha256=sampling_runtime_identity()))
    binding = _capture_clock()
    if (not resume and checkpoint is not None and checkpoint.enabled
            and deadline_seconds(binding['monotonic_after_bits']) + materialized.config.run.wall_seconds >= soft):
        raise _ScheduledSeedDoesNotFit('original scheduled seed wall must fit before the soft dispatch boundary')
    parent = Path(materialized.config.run.output_dir) / materialized.config.run.name
    check_native_collection_storage(materialized.config.storage, parent)
    storage = study.loaded.config.storage
    if isinstance(storage, HPODiagnosticStorageConfig):
        check_run_destination(storage, campaign_dir=store.directory,
                              originals_root=_originals(study.loaded), directory=parent)
    scope = ExitStack()
    try:
        owner = scope.enter_context(run_lock(parent, create=not resume))
        context = {'campaign_id': campaign['campaign_id'], 'operation_id': uuid.uuid4().hex,
            'kind': 'training', 'run_id': materialized.config.run.name, 'evaluation_id': None}
        current = -1
        original_deadlines = None
        if resume:
            paused = _paused_run(owner, hpo=True)
            origin, clock = _resume_origin(materialized, owner, bound, hpo_context=context, state=paused,
                campaign_record_root=control.root)
            original_deadlines, binding = origin['deadlines'], clock['clock_binding']
            current = paused['updates']
        milestone = (next((g for g in checkpoint.updates if g > current), None)
            if checkpoint is not None and checkpoint.enabled else None)
        packet = _publish_campaign_training(store, owner, materialized=materialized, context=context,
            proposal_no=actual.number, learning_seed=learning_seed, session=session,
            clock_binding=binding, runtime_binding_sha256=digest(bound), pause_after_update=milestone,
            original_deadlines=original_deadlines)
        if time.monotonic() >= deadline_seconds(session['soft_dispatch_finish_bits']):
            raise TimeoutError('original soft boundary crossed before actual bind; awaiting remains interrupted')
        control.bind_operation(packet['operation_path'])  # MUST precede SAME-OFD transfer inside run_hpo.
        if control.operation_sha256 != packet['operation_sha256']:
            raise ContractError('actual bound operation ACK differs from published training')
        return run_hpo(control, control.operation, loaded=packet['loaded'],
            _campaign_parent_scope=scope, _campaign_owner=owner,
            _campaign_admitted=lambda admission: _admission_cursor(control, store, packet, admission))
    except BaseException as error:
        try:
            scope.close()
        except BaseException as cleanup:  # noqa: BLE001 -- preserve original error and actual entered scope
            vars(error)['rust_player_dispatch_scope_cleanup_error'] = cleanup
        raise
    finally:
        scope.close()


def _terminal_records(store: Any, study: Any, proposal: Any, *, owner: Any,
                      awaiting: Json, awaiting_path: Path, awaiting_sha256: str,
                      operation_id: str, invocation_id: str, classification: str,
                      evidence: Json, reason: str) -> Path:
    """Existing FAIL/disposition DATA only; Native ACK remains separate authority."""
    from gf_cli.hpo_campaign import _TELL_KEYS, _checked
    from gf_common.rust_player import (
        safe_component,
        validate_campaign_awaiting,
        validate_campaign_disposition,
    )
    from gf_training.lineage import require_hpo_run_owner
    store._require()
    require_hpo_run_owner(Path(awaiting['run_path']), owner)
    validate_campaign_awaiting(awaiting)
    actual = _durable_proposal(study, proposal)
    operation_id = safe_component(operation_id)
    root = store.directory / 'supervision' / awaiting['supervision_id']
    relative = str(awaiting_path.relative_to(store.directory))
    row, checksum = store.get(relative)
    if (row != awaiting or checksum != awaiting_sha256 or awaiting_path != root / 'awaiting' / f'{operation_id}.json'
            or awaiting['proposal_no'] != actual.number or awaiting['campaign_id'] != study.manifest['campaign_id']):
        raise ContractError('actual terminal awaiting/proposal/parent differs')
    proposal_root = f'proposals/{actual.number:06}'
    failure_path = store.directory / proposal_root / 'failure.json'
    failure_sha = store.put(proposal_root + '/failure.json', {'schema_version': 1, 'kind': 'hpo-proposal-failure',
        'proposal_sha256': actual.sha256, 'reason': reason, 'boundary': 'training'})
    study.fail(actual, reason)  # Existing exact durable Optuna FAIL/value=None transaction.
    intent, intent_sha = store.get(proposal_root + '/tell.intent.json')
    receipt, receipt_sha = store.get(proposal_root + '/tell.receipt.json')
    for value in (intent, receipt):
        _checked(value, 'hpo-tell', _TELL_KEYS)
        if (value['proposal_sha256'] != actual.sha256 or value['state'] != 'FAIL' or value['value'] is not None
                or value['evaluation_receipt_sha256'] is not None or value['reason'] != reason):
            raise ContractError('actual terminal FAIL tell differs')
    if intent != receipt:
        raise ContractError('actual durable FAIL tell intent/receipt differs')
    disposition = {'schema_version': 1, 'kind': 'rust-player-campaign-terminal-disposition-v1',
        'campaign_id': awaiting['campaign_id'], 'proposal_no': actual.number, 'proposal_sha256': actual.sha256,
        'learning_seed': awaiting['learning_seed'], 'supervision_id': awaiting['supervision_id'],
        'session_clock_sha256': awaiting['session_clock_sha256'], 'awaiting_path': str(awaiting_path),
        'awaiting_sha256': awaiting_sha256, 'operation_path': awaiting['operation_path'],
        'operation_sha256': awaiting['operation_sha256'], 'invocation_id': invocation_id,
        'classification': classification, 'evidence': evidence, 'proposal_failure_path': str(failure_path),
        'proposal_failure_sha256': failure_sha, 'tell_intent_path': str(store.directory / proposal_root / 'tell.intent.json'),
        'tell_intent_sha256': intent_sha, 'tell_receipt_path': str(store.directory / proposal_root / 'tell.receipt.json'),
        'tell_receipt_sha256': receipt_sha}
    validate_campaign_disposition(disposition)
    path = root / 'terminal' / f'{operation_id}.json'
    store.put(str(path.relative_to(store.directory)), disposition)
    return path


def _finish_terminal(control: Any, store: Any, *, disposition_path: Path, retire_owner: Any,
                      cursor: Json, cursor_sha256: str, after_ack: Any = None) -> Json:
    """Caller has actual authority; retirement/release never follows an unknown ACK."""
    try:
        current, current_sha = store.get('supervision/current.json')
        if current != cursor or current_sha != cursor_sha256:
            raise ContractError('terminal cursor changed before actual native disposition')
        ack = control.campaign_disposition(disposition_path)
        if after_ack is not None:
            after_ack()
        retire_owner.close()
        if after_ack is None or 'parent' in control._borrowed:
            control.release_parent()
        current, current_sha = store.get('supervision/current.json')
        if current != cursor or current_sha != cursor_sha256:
            raise ContractError('terminal cursor changed before acknowledged release')
        updated = {**current, 'phase': 'control', 'awaiting_path': None,
            'awaiting_sha256': None, 'current_invocation_id': None}
        store.update_supervision_cursor(updated, expected_sha256=current_sha)
        return ack
    except BaseException as error:
        vars(error)['rust_player_terminal_owner'] = retire_owner
        raise


def _current_awaiting(control: Any, store: Any, study: Any,
                      invocation_id: str) -> tuple[Any, Json, Path, str, Json, str]:
    from gf_common.rust_player import validate_campaign_awaiting, validate_campaign_cursor
    _bootstrap, session, campaign = _campaign_context(control, store, study)
    operation = control.operation
    root = store.directory / 'supervision' / session['supervision_id']
    path = root / 'awaiting' / (operation['operation_id'] + '.json')
    awaiting, awaiting_sha = store.get(str(path.relative_to(store.directory)))
    validate_campaign_awaiting(awaiting)
    cursor, cursor_sha = store.get('supervision/current.json')
    validate_campaign_cursor(cursor)
    if (cursor['phase'] != 'awaiting-cold' or cursor['awaiting_path'] != str(path)
            or cursor['awaiting_sha256'] != awaiting_sha or cursor['current_invocation_id'] != invocation_id
            or cursor['campaign_id'] != campaign['campaign_id'] or cursor['supervision_id'] != session['supervision_id']
            or cursor['session_clock_sha256'] != awaiting['session_clock_sha256']
            or awaiting['session_clock_sha256'] != _bootstrap['session_clock_sha256']
            or awaiting['campaign_id'] != campaign['campaign_id'] or awaiting['supervision_id'] != session['supervision_id']
            or awaiting['operation_sha256'] != control.operation_sha256 or awaiting['operation_path'] != str(control.operation_path)
            or awaiting['run_path'] != operation['parent_directory']
            or any(awaiting[key] != operation[key] for key in ('run_id', 'config_path', 'config_sha256', 'runtime_binding_sha256'))
            or operation['expected_context']['campaign_id'] != campaign['campaign_id']):
        raise ContractError('actual current terminal awaiting/operation/session/parent differs')
    intent, _ = study._intent(awaiting['proposal_no'])
    proposal = study._proposal(awaiting['proposal_no'], intent)
    if awaiting['learning_seed'] not in study.loaded.config.learning_seeds:
        raise ContractError('actual terminal learning seed differs')
    return proposal, awaiting, path, awaiting_sha, cursor, cursor_sha


def settle_failed_campaign_training(control: Any, store: Any, study: Any) -> Json:
    """Only genuine Native campaign-return; report contents grant no retirement."""
    from gf_training.hpo_gpu_lease import require_resolved_namespace
    from gf_training.lineage import borrow_run_lock
    from gf_training.rust_player_supervision import read_control_record
    _campaign_context(control, store, study)
    payload = control.campaign_return  # Actual current failure/Exit/physical pair and OLD cutoff.
    proposal, awaiting, path, checksum, cursor, cursor_sha = _current_awaiting(
        control, store, study, payload['admission']['invocation_id'])
    require_resolved_namespace()
    scope = ExitStack()
    try:
        owner = scope.enter_context(borrow_run_lock(Path(awaiting['run_path']), control))
        failure, failure_sha = read_control_record(Path(payload['failure_path']), control.root)
        if failure_sha != payload['failure_sha256']:
            raise ContractError('actual failed return diagnostic original differs')
        evidence = {key + suffix: payload[key + suffix]
            for key in ('failure', 'exit', 'physical') for suffix in ('_path', '_sha256')}
        disposition = _terminal_records(store, study, proposal, owner=owner, awaiting=awaiting,
            awaiting_path=path, awaiting_sha256=checksum, operation_id=control.operation['operation_id'],
            invocation_id=payload['admission']['invocation_id'], classification='training_failed',
            evidence=evidence, reason=failure['reason'])
        return _finish_terminal(control, store, disposition_path=disposition, retire_owner=scope,
            cursor=cursor, cursor_sha256=cursor_sha)
    except BaseException as error:
        # Keep the original borrower and all scopes reachable; no unknown ACK retry.
        vars(error)['rust_player_terminal_scope'] = scope
        raise


def _incomplete_evidence(control: Any, result: Any) -> tuple[Json, Json]:
    from gf_training.hpo_gpu_lease import _GLOBAL_ROOT
    from gf_training.rust_player_completion import ColdFinalizedRustInvocation
    from gf_training.rust_player_supervision import read_control_record
    if type(result) is not ColdFinalizedRustInvocation:
        raise ContractError('actual generic finalized incomplete invocation required')
    result._require_parent()
    if (result.cold is not control.cold or result.cold._finalized_sha256 != result.completion_sha256
            or result.completion['state'] not in ('paused', 'incomplete') or result.completion['eligible'] is not False):
        raise ContractError('actual acknowledged ineligible incomplete Cold required')
    admission, _prepared, _exit = result.cold.require_pair()
    physical = result.cold.physical_termination.global_observation()
    physical_path = _GLOBAL_ROOT / 'jobs' / admission['global_ticket'] / 'termination.json'
    actual, physical_sha = read_control_record(physical_path, _GLOBAL_ROOT)
    if actual != physical:
        raise ContractError('actual incomplete original physical record differs')
    # Compact protocol DATA is copied; no validated-state/Source graph escapes
    # into result.close after the ACK.
    return clone(admission), {'completion_path': str(result.completion_path),
        'completion_sha256': str(result.completion_sha256), 'physical_path': str(physical_path),
        'physical_sha256': physical_sha}


def settle_incomplete_campaign_training(control: Any, store: Any, study: Any, result: Any, *,
                                        after_ack: Any = None) -> Json:
    from gf_training.hpo_gpu_lease import require_resolved_namespace
    from gf_training.lineage import require_hpo_run_owner
    try:
        _campaign_context(control, store, study)
        admission, evidence = _incomplete_evidence(control, result)
        proposal, awaiting, path, checksum, cursor, cursor_sha = _current_awaiting(
            control, store, study, admission['invocation_id'])
        require_resolved_namespace()
        owner = require_hpo_run_owner(Path(awaiting['run_path']), result._ownership)
        disposition = _terminal_records(store, study, proposal, owner=owner, awaiting=awaiting,
            awaiting_path=path, awaiting_sha256=checksum, operation_id=control.operation['operation_id'],
            invocation_id=admission['invocation_id'], classification='healthy_incomplete', evidence=evidence,
            reason='Training stopped incomplete under original limits')
        return _finish_terminal(control, store, disposition_path=disposition, retire_owner=result,
            cursor=cursor, cursor_sha256=cursor_sha, after_ack=after_ack)
    except BaseException as error:
        vars(error)['rust_player_terminal_result_owner'] = result
        raise


def _record_checkpoint_pause(control: Any, store: Any, *, result: Any,
                             awaiting_sha256: str, generation: int) -> None:
    admission, evidence = _incomplete_evidence(control, result)
    completion = result.completion
    if (completion['state'] != 'paused' or completion['committed_update'] != generation
            or control.operation['execution_request']['pause_after_update'] != generation):
        raise ContractError('actual paused invocation differs from the scheduled checkpoint')
    store.put(f"supervision/{admission['supervision_id']}/checkpoint-pauses/{admission['operation_id']}.json", {
        'schema_version': 1, 'kind': 'rust-player-campaign-checkpoint-pause-v1',
        'awaiting_sha256': awaiting_sha256, 'invocation_id': admission['invocation_id'],
        'generation': generation, 'completion_sha256': evidence['completion_sha256'],
        'physical_sha256': evidence['physical_sha256']})


def _checkpoint_pause_candidate(loaded: LoadedHPOConfig, awaiting_sha256: str, receipt: Json) -> Json:
    from gf_common.rust_player import hex_identity

    row = closed(receipt, {'schema_version', 'kind', 'awaiting_sha256', 'invocation_id',
        'generation', 'completion_sha256', 'physical_sha256'})
    integer(row['schema_version'], 1, 1)
    generation = integer(row['generation'], 0)
    hex_identity(row['invocation_id'], 32)
    for name in ('completion_sha256', 'physical_sha256'):
        hex_identity(row[name])
    checkpoint = loaded.config.checkpoint_evaluation
    if (row['kind'] != 'rust-player-campaign-checkpoint-pause-v1'
            or row['awaiting_sha256'] != awaiting_sha256
            or checkpoint is None or not checkpoint.enabled or generation not in checkpoint.updates):
        raise ContractError('historical pause differs from the exact scheduled awaiting operation')
    return row


def _verify_checkpoint_pause(store: Any, loaded: LoadedHPOConfig, awaiting: Json,
                             awaiting_sha256: str, receipt: Json) -> Json:
    from gf_common.rust_player import (
        hex_identity,
        initial_hpo_pause,
        validate_exit,
        validate_operation,
        validate_prepared,
    )
    from gf_training.hpo_gpu_lease import _GLOBAL_ROOT
    from gf_training.rust_player_lineage import validate_completion
    from gf_training.rust_player_supervision import read_control_record

    row = _checkpoint_pause_candidate(loaded, awaiting_sha256, receipt)
    generation, invocation = row['generation'], row['invocation_id']
    root = store.directory / 'supervision' / awaiting['supervision_id']
    path = Path(awaiting['operation_path'])
    operation, operation_sha = read_control_record(path, root)
    validate_operation(operation)
    admission, admission_sha = read_control_record(root / f'admission-{invocation}.json', root)
    prepared, prepared_sha = read_control_record(root / f'prepared-{invocation}.json', root)
    exit_record, exit_sha = read_control_record(root / f'exit-{invocation}.json', root)
    completion, completion_sha = read_control_record(root / f'completion-{invocation}.json', root)
    validate_prepared(prepared, admission, operation=operation)
    validate_exit(exit_record, admission)
    validate_completion(completion)
    ticket = hex_identity(admission['global_ticket'], 32)
    physical, physical_sha = read_control_record(_GLOBAL_ROOT / 'jobs' / ticket / 'termination.json', _GLOBAL_ROOT)
    if (path != root / f"operation-{operation['operation_id']}.json"
            or operation_sha != awaiting['operation_sha256']
            or operation['execution_request']['pause_after_update'] != generation
            or operation['expected_context']['campaign_id'] != awaiting['campaign_id']
            or admission['context'] != operation['expected_context']
            or any(admission[key] != awaiting[key] for key in
                ('run_id', 'config_sha256', 'runtime_binding_sha256', 'supervision_id'))
            or not exit_record['cooperative'] or not exit_record['descendants_reaped']
            or exit_record['signals'] or exit_record['wait']['exit_code'] != 0
            or exit_record['admission_sha256'] != admission_sha
            or exit_record['prepared_sha256'] != prepared_sha
            or completion['state'] != prepared['state'] or completion['state'] != 'paused'
            or completion['eligible'] or completion['committed_update'] != generation
            or completion['committed_update'] != prepared['committed_update']
            or completion['operation_sha256'] != operation_sha
            or completion['prepared_sha256'] != prepared_sha or completion['exit_sha256'] != exit_sha
            or any(completion[key] != prepared[key] for key in
                ('state_sha256', 'generation_sha256', 'source_manifest_sha256'))
            or initial_hpo_pause(operation) != (completion['schema_version'] == 2)
            or completion_sha != row['completion_sha256'] or physical_sha != row['physical_sha256']
            or physical['exit_record'] != exit_record
            or physical['context_sha256'] != digest(admission['context'])):
        raise ContractError('historical checkpoint pause lost its original completion or process retirement')
    return operation


def _checkpoint_time_cap_boundary(awaiting_sha256: str, operation: Json, now: float) -> Json:
    from gf_common.rust_player import deadline_bits, deadline_seconds

    admission = operation['deadlines']['admission_bits']
    if now < deadline_seconds(admission):
        raise ContractError('checkpoint time cap requires actual exhausted original admission')
    return {'phase': 'checkpoint-time-cap', 'awaiting_sha256': awaiting_sha256,
        'admission_bits': admission, 'observed_bits': deadline_bits(now)}


def complete_campaign_control(control: Any, store: Any, study: Any, *, status: str) -> str:
    from gf_common.rust_player import validate_campaign_cursor, validate_control_completion
    bootstrap, _session, campaign = _campaign_context(control, store, study)
    cursor, _sha = store.get('supervision/current.json')
    validate_campaign_cursor(cursor)
    if (cursor['campaign_id'] != campaign['campaign_id'] or cursor['phase'] != 'control'
            or cursor['awaiting_path'] is not None or cursor['current_invocation_id'] is not None
            or cursor['supervision_id'] != control.launch['supervision_id']
            or cursor['session_clock_sha256'] != bootstrap['session_clock_sha256']):
        raise ContractError('actual parent-free settled campaign cursor required')
    body = {'schema_version': 1, 'kind': 'rust-player-control-completion-v1',
        'supervision_id': control.launch['supervision_id'], 'operation_sha256': control.operation_sha256,
        'session_clock_sha256': bootstrap['session_clock_sha256'], 'status': status}
    validate_control_completion(body)
    path = control.root / f'control-completion-{control._envelope["invocation_id"]}.json'
    store.put(str(path.relative_to(store.directory)), body)
    return control.control_complete(path)  # Existing actual permitted protocol arms only.


def _audit_deadline(loaded: LoadedHPOConfig, hard_finish: float) -> float:
    return min(time.monotonic() + loaded.config.run_wall_seconds, hard_finish)


def _audit_terminal_history(store: Any, name: str, awaiting: Json, *, parent_root: Path) -> Json:
    from gf_training.lineage import run_lock
    parent = parent_root / awaiting['run_id']
    if Path(awaiting['run_path']) != parent:
        raise ContractError('historical terminal parent lies outside actual configured namespace')
    with run_lock(parent):
        disposition, _ = store.read_acknowledged_supervision_disposition(name)
        return disposition


def _verify_closed_tells(control: Any, store: Any, study: Any, *, failed_proposals: set[int], hard_finish: float) -> None:
    from gf_cli.hpo_campaign import _TELL_KEYS, _checked
    from gf_cli.hpo_evaluation import verify_trial_evaluation
    for name in store.records():
        parts = name.split('/')
        if len(parts) != 3 or parts[0] != 'proposals' or parts[2] not in ('tell.intent.json', 'tell.receipt.json'):
            continue
        root = '/'.join(parts[:2])
        if not store.exists(root + '/tell.receipt.json') or not store.exists(root + '/tell.intent.json'):
            raise ContractError('partial historical tell cannot be reconciled as closed')
        if parts[2] == 'tell.receipt.json':
            continue  # The same exact pair is verified once through its intent.
        number = int(parts[1])
        intent, _ = study._intent(number)
        proposal = study._proposal(number, intent)
        tell, _ = store.get(name)
        receipt, _ = store.get(root + '/tell.receipt.json')
        _checked(tell, 'hpo-tell', _TELL_KEYS)
        if (receipt != tell or tell['proposal_sha256'] != proposal.sha256 or tell['trial_number'] != proposal.trial_number
                or canonical(tell['params']) != canonical(proposal.params)
                or canonical(tell['distributions']) != canonical(proposal.distributions)):
            raise ContractError('historical closed tell differs from exact actual proposal')
        if tell['state'] == 'COMPLETE':
            _, score, checksum = verify_trial_evaluation(store, study.loaded, study.manifest, proposal,
                deadline=_audit_deadline(study.loaded, hard_finish), control=control)
            if tell['value'] != score or tell['evaluation_receipt_sha256'] != checksum or tell['reason'] is not None:
                raise ContractError('historical closed objective differs from independent actual evaluation')
        elif tell['state'] == 'FAIL':
            if tell['value'] is not None or tell['evaluation_receipt_sha256'] is not None:
                raise ContractError('failed historical tell cannot contain an objective')
            if number not in failed_proposals:
                duplicate, _ = store.get(root + '/duplicate.json') if store.exists(root + '/duplicate.json') else ({}, '')
                _checked(duplicate, 'hpo-effective-duplicate', {'proposal_sha256', 'original_proposal_sha256'})
                if duplicate.get('proposal_sha256') != proposal.sha256 or tell['reason'] != 'effective-duplicate':
                    raise ContractError('failed historical proposal lacks acknowledged terminal or exact duplicate')
                originals = []
                for other in store.records():
                    if other.startswith('proposals/') and other.endswith('/proposal.json') and other < root + '/proposal.json':
                        previous, previous_sha = store.get(other)
                        if previous_sha == duplicate['original_proposal_sha256']:
                            originals.append(previous)
                if len(originals) != 1 or originals[0]['effective_settings_sha256'] != proposal.settings.identity:
                    raise ContractError('failed duplicate lacks the original identical effective settings')
        else:
            raise ContractError('unknown historical tell state')


def _verify_campaign_history(control: Any, store: Any, study: Any, *, hard_finish: float) -> None:
    """Full current-owner audits before cursor replacement or another Optuna ask."""
    from gf_cli.rust_player_evaluation import (
        verify_rust_evaluation_history,
        verify_rust_training_history,
    )
    from gf_common.rust_player import validate_campaign_awaiting
    from gf_training.hpo_gpu_lease import _GLOBAL_ROOT, require_resolved_namespace
    from gf_training.rust_player_supervision import read_control_record
    _campaign_context(control, store, study)
    loaded, campaign = study.loaded, study.manifest
    checked: set[tuple[int, int]] = set()
    paused_seeds: set[tuple[int, int]] = set()
    terminal_seeds: set[tuple[int, int]] = set()
    failed_proposals: set[int] = set()
    closed: dict[str, str] = {}
    for name in store.records():
        parts = name.split('/')
        if len(parts) != 4 or parts[0] != 'supervision' or parts[2] != 'awaiting':
            continue
        awaiting, checksum = store.get(name)
        validate_campaign_awaiting(awaiting)
        if awaiting['campaign_id'] != campaign['campaign_id'] or awaiting['supervision_id'] != parts[1]:
            raise ContractError('historical awaiting belongs to another campaign/session')
        terminal = '/'.join((parts[0], parts[1], 'terminal', parts[3]))
        pause = '/'.join((parts[0], parts[1], 'checkpoint-pauses', parts[3]))
        seed_key = (awaiting['proposal_no'], awaiting['learning_seed'])
        if store.exists(terminal):
            disposition = _audit_terminal_history(store, terminal, awaiting,
                parent_root=Path(loaded.config.storage.parent_root))
            if (disposition['awaiting_path'] != str(store.directory / name)
                    or disposition['awaiting_sha256'] != checksum):
                raise ContractError('acknowledged terminal does not close the exact awaiting')
            failed_proposals.add(awaiting['proposal_no'])
            terminal_seeds.add(seed_key)
            checkpoint = loaded.config.checkpoint_evaluation
            folder = f'proposals/{seed_key[0]:06}/seeds/{seed_key[1]}'
            if (checkpoint is not None and checkpoint.enabled
                    and disposition['classification'] == 'healthy_incomplete'
                    and not store.exists(folder + '/best-checkpoint.json')):
                from gf_cli.hpo_evaluation import persist_best_checkpoint
                from gf_training.rust_player_lineage import validate_completion

                path = Path(disposition['evidence']['completion_path'])
                completion, completion_sha = read_control_record(path, store.directory / 'supervision' / parts[1])
                validate_completion(completion)
                if completion_sha != disposition['evidence']['completion_sha256']:
                    raise ContractError('terminal checkpoint index requires its exact acknowledged completion')
                intent, _ = study._intent(seed_key[0])
                proposal = study._proposal(seed_key[0], intent)
                persist_best_checkpoint(store, loaded, campaign, proposal, seed_key[1],
                    terminal_generation=completion['committed_update'], deadline=hard_finish)
        elif store.exists(pause):
            pause_receipt, _ = store.get(pause)
            operation = _verify_checkpoint_pause(store, loaded, awaiting, checksum, pause_receipt)
            paused_seeds.add(seed_key)
            folder = f"proposals/{awaiting['proposal_no']:06}"
            if store.exists(folder + '/failure.json'):
                failure, _ = store.get(folder + '/failure.json')
                boundary = failure.get('boundary')
                if (type(boundary) is dict and boundary.get('phase') == 'checkpoint-time-cap'
                        and boundary.get('awaiting_sha256') == checksum):
                    from gf_common.rust_player import deadline_seconds

                    expected = _checkpoint_time_cap_boundary(checksum, operation,
                        deadline_seconds(boundary['observed_bits']))
                    if boundary != expected or failure['reason'] != 'Original admission expired at checkpoint':
                        raise ContractError('checkpoint time cap differs from the original pause and clock')
                    terminal_seeds.add(seed_key)
                    failed_proposals.add(awaiting['proposal_no'])
        else:
            number, seed = awaiting['proposal_no'], awaiting['learning_seed']
            intent, _ = study._intent(number)
            proposal = study._proposal(number, intent)
            candidate = _seed_candidate(store, campaign, number, seed, awaiting=awaiting)
            training = verify_rust_training_history(store, loaded, campaign, proposal, seed, control=control,
                deadline=_audit_deadline(loaded, hard_finish))
            if training != candidate:
                raise ContractError('full historical seed differs from structural candidate')
            # The full reader binds physical/admission/Source; additionally ensure
            # THIS awaiting is its actual operation, not another same-run record.
            physical, _ = store.get(f'proposals/{number:06}/seeds/{seed}/training-gpu-termination.json')
            admission, _ = read_control_record(_GLOBAL_ROOT / 'jobs' / physical['ticket'] / 'admission.json', _GLOBAL_ROOT)
            if (admission['host_admission']['operation_id'] + '.json' != parts[3]
                    or physical['exit_record']['supervision_id'] != parts[1]):
                raise ContractError('historical successful awaiting differs from actual admitted operation')
            checked.add((number, seed))
        closed[str(store.directory / name)] = checksum
    if paused_seeds - checked - terminal_seeds:
        raise ContractError('interrupted checkpoint training cannot resume under another supervisor root')
    cursor = store.get('supervision/current.json')[0] if store.exists('supervision/current.json') else None
    if cursor is not None and cursor['awaiting_path'] is not None and closed.get(cursor['awaiting_path']) != cursor['awaiting_sha256']:
        raise ContractError('historical current cursor remains unresolved')
    for name in store.records():
        parts = name.split('/')
        if (_checkpoint_intent_path(parts) and not store.exists(name.removesuffix('.intent.json') + '.json')
                and not _checkpoint_failed_candidate(store, name, campaign=campaign)):
            raise ContractError('interrupted checkpoint panel has no verified diagnostic closure')
        if len(parts) != 5 or parts[0] != 'proposals' or parts[2] != 'seeds':
            continue
        number, seed = int(parts[1]), int(parts[3])
        folder = '/'.join(parts[:4])
        if parts[4] in ('training.intent.json', 'evaluation.intent.json'):
            if not store.exists(folder + '/' + parts[4].removesuffix('.intent.json') + '.json'):
                raise ContractError('interrupted historical operation cannot be retried')
        elif parts[4] in ('training.json', 'evaluation.json'):
            intent, _ = study._intent(number)
            proposal = study._proposal(number, intent)
            if (number, seed) not in checked:
                # A successful seed without an actual awaiting cannot be imported.
                raise ContractError('closed seed lacks a fully verified original awaiting')
            if parts[4] == 'evaluation.json':
                verify_rust_evaluation_history(store, loaded, campaign, proposal, seed, control=control,
                    deadline=_audit_deadline(loaded, hard_finish))
                checkpoint = loaded.config.checkpoint_evaluation
                if (checkpoint is not None and checkpoint.enabled
                        and not store.exists(folder + '/best-checkpoint.json')):
                    _index_final_checkpoint(store, loaded, campaign, proposal,
                        store.get(folder + '/training.json')[0], store.get(name)[0], hard_finish=hard_finish)
        elif parts[4] in ('source-validation.json', 'training-gpu-termination.json') and not store.exists(folder + '/training.json'):
            raise ContractError('partial closed training cannot be retried')
    _verify_closed_tells(control, store, study, failed_proposals=failed_proposals, hard_finish=hard_finish)
    if store.exists('study.intent.json'):
        with study.study(0) as optuna_study:
            study._history(optuna_study)
    require_resolved_namespace()  # Existing exclusive lock/scan; no dummy job.


def _control_cursor(store: Any, campaign: Json, session: Json, bootstrap: Json) -> Json:
    return {'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
        'campaign_id': campaign['campaign_id'], 'supervision_id': session['supervision_id'],
        'session_clock_sha256': bootstrap['session_clock_sha256'], 'phase': 'control',
        'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None}


def _adopt_campaign_cursor(control: Any, store: Any, study: Any, *, session: Json, bootstrap: Json,
                           hard_finish: float) -> None:
    old_sha = store.get('supervision/current.json')[1] if store.exists('supervision/current.json') else None
    _verify_campaign_history(control, store, study, hard_finish=hard_finish)
    store.update_supervision_cursor(_control_cursor(store, study.manifest, session, bootstrap), expected_sha256=old_sha)


def _clear_success_cursor(control: Any, store: Any, study: Any, expected: Json, checksum: str) -> None:
    bootstrap, session, campaign = _campaign_context(control, store, study)
    if 'parent' in control._borrowed or control._parent_borrowers:
        raise ContractError('actual positive parent release required before successful cursor advance')
    actual, actual_sha = store.get('supervision/current.json')
    if actual != expected or actual_sha != checksum:
        raise ContractError('successful current cursor changed before release/CAS')
    store.update_supervision_cursor(_control_cursor(store, campaign, session, bootstrap), expected_sha256=checksum)


def _finish_campaign_result(control: Any, store: Any, study: Any, *, status: str) -> str:
    # A finalized cold Exit already has its genuine protocol completion. Initial
    # and failed-return arms require the existing explicit control-complete ACK.
    if control._cold is None or control._cold._finalized_sha256 is None:
        complete_campaign_control(control, store, study, status=status)
    return canonical({'schema_version': 1, 'kind': 'hpo-campaign-result', 'campaign_dir': str(store.directory),
        'campaign_id': study.manifest['campaign_id'], 'study_identity': study.manifest['study_identity'],
        'status': 'paused' if status == 'session_boundary' else 'complete', 'qualification': 'exploratory-only'})


def _drive_campaign(control: Any, store: Any, study: Any, imports: tuple[Any, ...], *,
                    soft_finish: float, hard_finish: float) -> str | None:
    import asyncio

    from gf_cli.hpo import _storage_boundary, next_seed_operation
    from gf_cli.hpo_campaign import campaign_requests
    from gf_cli.hpo_config import materialize_hpo_run
    from gf_cli.hpo_evaluation import aggregate_trial_evaluation
    from gf_cli.rust_player_evaluation import evaluate_rust_seed, verify_rust_training_history
    loaded, campaign = study.loaded, study.manifest
    for number, (slot, kind, requested) in enumerate(campaign_requests(loaded.config, imports)):
        if time.monotonic() >= soft_finish:
            return _finish_campaign_result(control, store, study, status='session_boundary')
        proposal = study.propose(number, slot=slot, kind=kind, requested=requested)
        if proposal is None:
            continue
        root = f'proposals/{number:06}'
        if store.exists(root + '/failure.json'):
            failure, _ = store.get(root + '/failure.json')
            study.fail(proposal, failure['reason'])
            continue
        if store.exists(root + '/tell.receipt.json'):
            tell, _ = store.get(root + '/tell.receipt.json')
            if tell['state'] == 'COMPLETE':
                study.complete(proposal, deadline=_audit_deadline(loaded, hard_finish))
            elif tell['state'] == 'FAIL':
                study.fail(proposal, tell['reason'])
            else:
                raise ContractError('unknown actual terminal proposal receipt')
            continue
        while (next_operation := next_seed_operation(store, proposal, loaded.config.learning_seeds)) is not None:
            phase, seed = next_operation
            if phase == 'incomplete':
                raise ContractError('interrupted campaign seed has no closed receipt; retry forbidden')
            if time.monotonic() >= soft_finish:
                return _finish_campaign_result(control, store, study, status='session_boundary')
            _storage_boundary(store, loaded, operation_id=uuid.uuid4().hex, _deadline=hard_finish)
            if phase == 'train':
                checkpoint = loaded.config.checkpoint_evaluation
                if (checkpoint is not None and checkpoint.enabled
                        and time.monotonic() + loaded.config.run_wall_seconds >= soft_finish):
                    return _finish_campaign_result(control, store, study, status='session_boundary')
                materialized = materialize_hpo_run(loaded, proposal.settings, learning_seed=seed, run_id=str(uuid.uuid4()))
                try:
                    dispatch_campaign_training(control, store, study, proposal,
                        learning_seed=seed, materialized=materialized)
                except _ScheduledSeedDoesNotFit:
                    return _finish_campaign_result(control, store, study, status='session_boundary')
                return None  # Prepared: only Native reap followed by a NEW cold host may continue.
            training = verify_rust_training_history(store, loaded, campaign, proposal, seed, control=control,
                deadline=_audit_deadline(loaded, hard_finish))
            if time.monotonic() >= soft_finish:
                return _finish_campaign_result(control, store, study, status='session_boundary')
            evaluation = asyncio.run(evaluate_rust_seed(store, loaded, campaign, proposal, training, control=control,
                output=store.directory / 'evaluations' / str(uuid.uuid4()), soft_finish=soft_finish, hard_finish=hard_finish))
            _index_final_checkpoint(store, loaded, campaign, proposal, training, evaluation, hard_finish=hard_finish)
        aggregate = aggregate_trial_evaluation(store, loaded, campaign, proposal,
            deadline=_audit_deadline(loaded, hard_finish), control=control)
        store.put(root + '/evaluation.json', aggregate)
        study.complete(proposal, deadline=_audit_deadline(loaded, hard_finish))
    return _finish_campaign_result(control, store, study, status='complete')


def _index_final_checkpoint(store: Any, loaded: LoadedHPOConfig, campaign: Json, proposal: Any,
                            training: Json, evaluation: Json, *, hard_finish: float) -> None:
    checkpoint = loaded.config.checkpoint_evaluation
    if checkpoint is not None and checkpoint.enabled:
        from gf_cli.hpo_evaluation import persist_best_checkpoint

        persist_best_checkpoint(store, loaded, campaign, proposal, training['training_seed'],
            terminal_generation=training['generation'], final_training=training, final_evaluation=evaluation,
            deadline=hard_finish)


def _evaluate_closed_checkpoint(control: Any, store: Any, study: Any, proposal: Any,
                                result: Any, awaiting: Json, *, hard_finish: float) -> None:
    import asyncio

    from gf_cli.hpo import _record_checkpoint_evaluation_error
    from gf_cli.hpo_evaluation import evaluate_checkpoint
    from gf_common.rust_player import deadline_seconds

    generation = result.completion['committed_update']
    paused = result.completion['state'] == 'paused'
    deadline = min(hard_finish, deadline_seconds(control.operation['deadlines'][
        'admission_bits' if paused else 'finish_bits']))
    seed = awaiting['learning_seed']
    generation_folder = f'proposals/{proposal.number:06}/seeds/{seed}/checkpoint-evaluations/{generation}'
    start = time.monotonic()
    try:
        asyncio.run(evaluate_checkpoint(store, study.loaded, study.manifest, proposal,
            directory=Path(awaiting['run_path']), generation=generation,
            output=store.directory / 'evaluations' / str(uuid.uuid4()), deadline=deadline,
            rust_invocation=result, control=control))
    except Exception as error:  # noqa: BLE001 - only a verified diagnostic failure can continue
        _record_checkpoint_evaluation_error(store, study.manifest, path=generation_folder,
            run_id=awaiting['run_id'], generation=generation, started=start, error=error)


def _continue_cold_campaign(control: Any, store: Any, study: Any, *, soft_finish: float, hard_finish: float) -> bool:
    import asyncio

    from gf_cli.config import load_rust_player_effective
    from gf_cli.rust_player_evaluation import evaluate_rust_seed, persist_rust_training_receipt
    from gf_common.rust_player import deadline_seconds
    from gf_training.rust_player_completion import (
        ColdFinalizedRustInvocation,
        ColdFinalizedRustParent,
    )
    from gf_training.rust_player_learning import finalize_cold
    admission, _prepared, _exit = control.cold.require_pair()
    proposal, awaiting, _path, checksum, cursor, cursor_sha = _current_awaiting(
        control, store, study, admission['invocation_id'])
    loaded = load_rust_player_effective(Path(control.operation['config_path']), expected_sha256=control.operation['config_sha256'])
    result = finalize_cold(control, control.cold, loaded=loaded)
    try:
        if type(result) is ColdFinalizedRustInvocation:
            checkpoint = study.loaded.config.checkpoint_evaluation
            generation = result.completion['committed_update']
            if checkpoint is not None and checkpoint.enabled and result.completion['state'] == 'paused':
                operation = control.operation
                _record_checkpoint_pause(control, store, result=result,
                    awaiting_sha256=checksum, generation=generation)
                _evaluate_closed_checkpoint(control, store, study, proposal, result, awaiting, hard_finish=hard_finish)
                result.close()
                if 'parent' in control._borrowed:
                    control.release_parent()
                _clear_success_cursor(control, store, study, cursor, cursor_sha)
                now = time.monotonic()
                cutoff = deadline_seconds(operation['deadlines']['admission_bits'])
                if now < cutoff:
                    try:
                        dispatch_campaign_training(control, store, study, proposal,
                            learning_seed=awaiting['learning_seed'], materialized=loaded, resume=True)
                    except TimeoutError:
                        now = time.monotonic()
                        if now < cutoff or control._phase != 'session' or store.get('supervision/current.json')[0]['phase'] != 'control':
                            raise
                    else:
                        return True
                from gf_cli.hpo_evaluation import persist_best_checkpoint

                persist_best_checkpoint(store, study.loaded, study.manifest, proposal,
                    awaiting['learning_seed'], terminal_generation=generation, deadline=hard_finish)
                reason = 'Original admission expired at checkpoint'
                store.put(f'proposals/{proposal.number:06}/failure.json', {'schema_version': 1,
                    'kind': 'hpo-proposal-failure', 'proposal_sha256': proposal.sha256, 'reason': reason,
                    'boundary': _checkpoint_time_cap_boundary(checksum, operation, now)})
                study.fail(proposal, reason)
                return False
            after_ack = None
            if checkpoint is not None and checkpoint.enabled:
                from gf_cli.hpo_evaluation import persist_best_checkpoint

                def report_cap() -> None:
                    if (generation in checkpoint.updates
                            and result._validated_state is not None
                            and result._validated_state['stop_reason'] == 'update_cap'):
                        _evaluate_closed_checkpoint(control, store, study, proposal, result, awaiting, hard_finish=hard_finish)
                    persist_best_checkpoint(store, study.loaded, study.manifest, proposal,
                        awaiting['learning_seed'], terminal_generation=generation, deadline=hard_finish)

                after_ack = report_cap
            if after_ack is None:
                settle_incomplete_campaign_training(control, store, study, result)
            else:
                settle_incomplete_campaign_training(control, store, study, result, after_ack=after_ack)
            return False
        if type(result) is not ColdFinalizedRustParent:
            raise ContractError('actual exact-D cold Parent or ineligible invocation required')
        result.require()
        training = persist_rust_training_receipt(store, study.loaded, study.manifest, proposal,
            training_seed=awaiting['learning_seed'], run_id=awaiting['run_id'], directory=Path(awaiting['run_path']),
            termination=control.cold.physical_termination, deadline=_audit_deadline(study.loaded, hard_finish), completion=result)
        if time.monotonic() < soft_finish:
            evaluation = asyncio.run(evaluate_rust_seed(store, study.loaded, study.manifest, proposal, training, control=control,
                output=store.directory / 'evaluations' / str(uuid.uuid4()), soft_finish=soft_finish,
                hard_finish=hard_finish, completion=result))
            _index_final_checkpoint(store, study.loaded, study.manifest, proposal, training, evaluation, hard_finish=hard_finish)
        else:
            result.close()
            control.release_parent()
        _clear_success_cursor(control, store, study, cursor, cursor_sha)
        return False
    except BaseException as error:
        # The evaluator may already own a nested pin/job; retain that original
        # cleanup owner and the actual Parent without pretending cleanup succeeded.
        vars(error)['rust_player_campaign_parent'] = result
        raise


def run_campaign(control: Any, operation: Json) -> str | None:
    """Authenticated fixed CPU controller; one actual operation per host boundary."""
    from gf_cli.hpo_campaign import CampaignStudy
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.rust_player import deadline_seconds
    from gf_training.rust_player_supervision import HostControl, read_control_record
    if type(control) is not HostControl:
        raise ContractError('actual authenticated campaign HostControl required')
    control._require()
    if (control.initial_operation['operation_type'] != 'campaign-continue'
            or canonical(operation) != canonical(control.operation)):
        raise ContractError('authenticated initial campaign purpose/current bound operation required')
    bootstrap = control.campaign_bootstrap
    snapshot, snapshot_sha = read_control_record(Path(bootstrap['loaded_campaign_path']), control.root)
    if snapshot_sha != bootstrap['loaded_campaign_sha256']:
        raise ContractError('actual immutable campaign CONFIG snapshot differs')
    loaded = load_campaign_snapshot(snapshot)
    store = CampaignStore.borrow(control, max_owned_bytes=loaded.config.operating.parent_budget_bytes)
    try:
        campaign, _ = store.get('campaign.json')
        study = CampaignStudy(store, loaded, campaign, _control=control)
        _bootstrap, session, _campaign = _campaign_context(control, store, study)
        seeds, seeds_sha = store.get(str(Path(bootstrap['seed_settings_path']).relative_to(store.directory)))
        if seeds_sha != bootstrap['seed_settings_sha256']:
            raise ContractError('actual immutable imported settings snapshot differs')
        imports = load_seed_snapshot(seeds, loaded.config, study_identity=campaign['study_identity'])
        soft, hard = (deadline_seconds(session[key]) for key in ('soft_dispatch_finish_bits', 'hard_session_finish_bits'))
        dispatched = False
        if control._phase == 'cold':
            dispatched = _continue_cold_campaign(control, store, study, soft_finish=soft, hard_finish=hard)
        elif control._phase == 'campaign-return':
            settle_failed_campaign_training(control, store, study)
        elif control._phase == 'session' and control.operation['operation_type'] == 'campaign-continue':
            _adopt_campaign_cursor(control, store, study, session=session, bootstrap=bootstrap, hard_finish=hard)
        else:
            raise ContractError('unknown authenticated campaign controller phase')
        # Every imported closed operation is reverified before reuse/new ask.
        output = None
        if not dispatched:
            if control.operation['operation_type'] != 'campaign-continue':
                _verify_campaign_history(control, store, study, hard_finish=hard)
            output = _drive_campaign(control, store, study, imports, soft_finish=soft, hard_finish=hard)
    except BaseException as error:
        try:
            store.close()
        except BaseException as cleanup:  # noqa: BLE001 -- retain original error and actual borrower on uncertain close
            vars(error)['rust_player_campaign_store_cleanup_error'] = cleanup
        vars(error)['rust_player_campaign_store'] = store
        raise
    try:
        store.close()  # Positive borrower close before host control close/stdout.
    except BaseException as error:
        vars(error)['rust_player_campaign_store'] = store
        raise
    return output
