"""Canonical campaign CONFIG/bootstrap factories, without a campaign controller.

Public HPO execution remains guarded until the real continuation/evaluation
application is integrated. Decoded snapshots never reconstruct live owners.
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
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest


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


def _resume_history(store: Any, loaded: LoadedHPOConfig, conditions: Json,
                    parent_scope: ExitStack) -> tuple[Json, list[Json], str | None]:
    from gf_cli.hpo_campaign import CampaignStudy
    from gf_common.rust_player import (
        validate_campaign_awaiting,
        validate_campaign_bootstrap,
        validate_campaign_clock,
        validate_campaign_cursor,
    )
    from gf_training.lineage import require_hpo_run_owner, run_lock
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
    settled: dict[str, str] = {}
    parent_owners: dict[Path, Any] = {}
    for name in store.records():
        parts = name.split('/')
        if name.startswith('proposals/') and name.endswith(('/training.json', '/evaluation.json')):
            raise ContractError('successful history requires the C2b full Source/seed application; foundation is unsupported')
        if name.endswith(('/training.intent.json', '/evaluation.intent.json')):
            raise ContractError('legacy or unclosed seed intent cannot be resumed by the campaign foundation')
        if len(parts) != 4 or parts[0] != 'supervision' or parts[2] != 'awaiting':
            continue
        awaiting, awaiting_sha = store.get(name)
        validate_campaign_awaiting(awaiting)
        # The actual verifier rederives the ID from the original operation. Here
        # the immutable awaiting basename is the canonical lookup, never closure.
        operation_id = Path(parts[3]).stem
        terminal = '/'.join((parts[0], parts[1], 'terminal', operation_id + '.json'))
        if not store.exists(terminal):
            raise ContractError('unclosed/orphan awaiting lacks already-native-ACK terminal evidence')
        parent = Path(awaiting['run_path'])
        if parent != Path(loaded.config.storage.parent_root) / awaiting['run_id']:
            raise ContractError('historical awaiting parent leaves actual configured root')
        if parent not in parent_owners:
            parent_owners[parent] = parent_scope.enter_context(run_lock(parent))
        owner = parent_owners[parent]
        require_hpo_run_owner(parent, owner)
        disposition, _ack = store.read_acknowledged_supervision_disposition(terminal)
        if (disposition['campaign_id'] != campaign['campaign_id']
                or disposition['awaiting_path'] != str(store.directory / name)
                or disposition['awaiting_sha256'] != awaiting_sha):
            raise ContractError('already acknowledged failure/awaiting binding differs')
        settled[str(store.directory / name)] = awaiting_sha
    if (cursor is not None and cursor['awaiting_path'] is not None
            and settled.get(cursor['awaiting_path']) != cursor['awaiting_sha256']):
        raise ContractError('current awaiting is unresolved/mismatched')
    return campaign, sources, cursor_sha


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
    from gf_cli.rust_player_entry import _capture_clock
    from gf_common.rust_player import BUILD_KEYS, runtime_binding, validate_campaign_bootstrap
    from gf_models.rust_player import sampling_runtime_identity
    from gf_simulator_client.rust_player_companion import discover_companion
    from gf_training.native_backend import binary_identity
    from gf_training.rust_player_supervision import runtime_producer
    if mode not in ('fresh', 'resume') or mode == 'resume' and seed_paths or progress not in ('auto', 'plain', 'off'):
        raise ContractError('explicit campaign mode/progress; seed settings are fresh-only')
    snapshot = campaign_snapshot(loaded)
    load_campaign_snapshot(snapshot)
    config = loaded.config
    if not isinstance(config, RustPlayerHPOCampaignConfig):
        raise ContractError('actual native6 campaign configuration required')
    hpo.validate(loaded)
    conditions = hpo._conditions(loaded)
    _settings, sources = hpo._settings_inputs(seed_paths, loaded) if mode == 'fresh' else ((), [])
    native = binary_identity()
    companion = discover_companion(native)
    bound = runtime_binding(config.runtime, runtime_producer(companion,
        sampling_runtime_sha256=sampling_runtime_identity()))
    supervision_id = uuid.uuid4().hex
    binding = _capture_clock()  # Original SESSION anchor, before actual store ownership/publication.
    session = campaign_session_clock(loaded, supervision_id=supervision_id, clock_binding=binding)
    with CampaignStore(Path(directory).absolute(), mode=mode,
            max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store, ExitStack() as history_parents:
        if mode == 'fresh':
            conditions['initial_settings'] = sources
            if 'producer_identity' in conditions:
                from gf_common.rust_player import deadline_seconds
                from gf_training.ppo_run import source_identity
                conditions['producer_identity'] = source_identity(deadline_seconds(session['hard_session_finish_bits']))
            campaign = initialize_campaign(store, loaded, conditions)
            previous_cursor_sha = None
        else:
            campaign, sources, previous_cursor_sha = _resume_history(store, loaded, conditions, history_parents)
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
        store.update_supervision_cursor({'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
            'campaign_id': campaign['campaign_id'], 'supervision_id': supervision_id, 'session_clock_sha256': clock_sha,
            'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None, 'phase': 'control'},
            expected_sha256=previous_cursor_sha)
        # Historical parent custody spans its last consumer, the cursor CAS.
        # No parent descriptor is inherited by this initial CPU-control launch.
        history_parents.close()
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
        prepared.require()
        yield prepared


def exec_campaign(loaded: LoadedHPOConfig, directory: Path, *, mode: str,
                  seed_paths: tuple[Path, ...] = (), progress: str = 'auto') -> None:
    with prepare_campaign(loaded, directory, mode=mode, seed_paths=seed_paths, progress=progress) as prepared:
        prepared.execute()
