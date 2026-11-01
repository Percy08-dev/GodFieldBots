"""Original collection3/4 + journal5 structural components, never Source authority.

No active Torch/runtime/hardware/producer validation happens here. Whole-parent
Generation2 authorization remains a separate required stage for owned ingestion.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.action_contract import require_qualified_contract
from gf_common.candidate_columns import validate_descriptor
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    ContractError,
    Json,
    canonical,
    closed,
    digest,
    group_health_failures,
    integer,
)
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.structured import ActionConfigV2, action_config_v2
from gf_common.validation import projected_abort_kind, reconcile_replay, replay_closure
from gf_models.sharded_arrays import _pairs
from gf_models.structured_binding import (
    check_input_binding,
    input_binding,
    policy_label,
    safe_options_v2,
)
from gf_models.structured_config import InferenceBindingV2, model_config_v2
from gf_training.candidate_storage import (
    _bounded_hash,
    _identity,
    _open_regular,
    candidate_basename,
)
from gf_training.preparation import ResourceLimit
from gf_training.structured_journal import STRUCTURED_FEATURE_VERSION, read_structured_journal
from gf_training.structured_rollout import EpisodeRecords


@dataclass(frozen=True)
class ComponentEpisode:
    episode_id: str
    viewer: int
    model_id: str
    model_version: str
    records: EpisodeRecords


@dataclass(frozen=True)
class ComponentCollection:
    directory: Path
    episodes: tuple[ComponentEpisode, ...]
    _manifest: bytes
    _proof: bytes

    @property
    def manifest(self) -> Json:
        return json.loads(self._manifest)

    @property
    def proof(self) -> Json:
        return json.loads(self._proof)


def _json(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ContractError('nonfinite source JSON')))


def _name(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}', value) is None or value in ('.', '..'):
        raise ContractError('unsafe original episode path')
    return value


class _SourceOwner:
    def __init__(self, directory: Path, settings: PreparationConfig, deadline: float) -> None:
        self.root, self.settings, self.deadline = directory.absolute(), settings, deadline
        self.used = 0
        self.files: Json = {}
        self.inventories: list[tuple[str, tuple[str, ...]]] = []
        self._safe(self.root)

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError('structured source deadline')

    def charge(self, size: int) -> None:
        self.check()
        integer(size, 0)
        if self.used + size > self.settings.memory_budget_mib * 1024**2:
            raise ResourceLimit('structured original-source reservation exhausted before decoding')
        self.used += size

    def _safe(self, path: Path) -> None:
        if not path.is_relative_to(self.root):
            raise ContractError('foreign source path')
        for parent in (path, *path.parents):
            value = parent.lstat()
            if stat.S_ISLNK(value.st_mode):
                raise ContractError('symlink in original source')
            if parent == self.root:
                break

    def read(self, path: Path) -> bytes:
        self.check()
        self._safe(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > self.settings.max_source_file_mib * 1024**2:
            raise ContractError('bounded regular original source required')
        self.charge(32 * before.st_size + 65536)
        with _open_regular(path, before.st_size) as source:
            actual = source_stat = os.fstat(source.fileno())
            if _identity(actual) != _identity(before):
                raise ContractError('source replaced after admission')
            raw = source.read(before.st_size + 1)
            after = os.fstat(source.fileno())
            if _identity(actual) != _identity(after) or len(raw) != before.st_size:
                raise ContractError('source changed during read')
        self._safe(path)
        if _identity(path.lstat()) != _identity(source_stat):
            raise ContractError('source path changed during read')
        self.remember(path, _identity(source_stat), hashlib.sha256(raw).hexdigest())
        return raw

    def remember(self, path: Path, identity: tuple[int, ...], checksum: str) -> None:
        name = path.relative_to(self.root).as_posix()
        entry = {'sha256': checksum, 'bytes': identity[2], 'identity': list(identity)}
        if name in self.files and self.files[name] != entry:
            raise ContractError('original source identity changed')
        self.files[name] = entry

    def object(self, path: Path) -> Json:
        value = _json(self.read(path))
        if not isinstance(value, dict):
            raise ContractError('original source object required')
        return value

    def _temporary(self, amount: int) -> None:
        self.check()
        if self.used + amount > self.settings.memory_budget_mib * 1024**2:
            raise ResourceLimit('structured inventory comparison reservation exhausted')

    def _inventory_names(self, pattern: str, *, retain: bool) -> tuple[str, ...]:
        self.check()
        if pattern not in ('*/player-*.jsonl', '*/player-*.partial.jsonl', '*/*.columns', '*/replay-player-*.json', '*/result.json'):
            raise ContractError('unsupported private source inventory pattern')
        storage = 1024
        self._temporary(storage)
        seen: set[str] = set()
        leaf = pattern.split('/')[1]
        with os.scandir(self.root) as directories:
            while True:
                self.check()
                self._temporary(storage + 1024)
                try:
                    directory = next(directories)
                except StopIteration:
                    break
                if directory.is_symlink():
                    raise ContractError('symlink in original source inventory')
                if not directory.is_dir(follow_symlinks=False):
                    continue
                self._temporary(storage + 1024 + 4 * len(directory.name))
                with os.scandir(directory.path) as entries:
                    while True:
                        self.check()
                        self._temporary(storage + 1024 + 4 * len(directory.name))
                        try:
                            entry = next(entries)
                        except StopIteration:
                            break
                        if not fnmatch.fnmatchcase(entry.name, leaf):
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            raise ContractError('nonregular original inventory entry')
                        cost = 512 + 4 * (len(directory.name) + len(entry.name) + 1)
                        if retain:
                            self.charge(cost)
                        else:
                            storage += cost
                            self._temporary(storage)
                        name = directory.name + '/' + entry.name
                        if name in seen:
                            raise ContractError('duplicate original inventory entry')
                        seen.add(name)
        self._temporary(storage + 512 * len(seen))
        names = tuple(sorted(seen))
        self.check()
        return names

    def inventory(self, pattern: str) -> tuple[str, ...]:
        self.charge(512)
        names = self._inventory_names(pattern, retain=True)
        self.inventories.append((pattern, names))
        return names

    def finalize(self) -> None:
        for pattern, names in self.inventories:
            self.check()
            if self._inventory_names(pattern, retain=False) != names:
                raise ContractError('original source inventory changed')
        for name, entry in self.files.items():
            self.check()
            path = self.root / name
            self._safe(path)
            with _open_regular(path, entry['bytes']) as source:
                before = os.fstat(source.fileno())
                if list(_identity(before)) != entry['identity'] or _bounded_hash(source, entry['bytes']) != entry['sha256']:
                    raise ContractError('original source bytes/identity changed')
                after = os.fstat(source.fileno())
                if _identity(before) != _identity(after) or _identity(path.lstat()) != _identity(before):
                    raise ContractError('original source changed during final recheck')
        # A long hash sweep must not publish the earlier inventory/stat snapshot
        # as current evidence after a late addition or an earlier file mutation.
        for pattern, names in self.inventories:
            self.check()
            if self._inventory_names(pattern, retain=False) != names:
                raise ContractError('original source inventory changed after hash sweep')
        for name, entry in self.files.items():
            self.check()
            path = self.root / name
            self._safe(path)
            if list(_identity(path.lstat())) != entry['identity']:
                raise ContractError('original source identity changed after hash sweep')


def reject_engineering_source(manifest: Json) -> None:
    """Qualified paths inspect original Config first, never a claimed purpose."""
    config = manifest.get('effective_config', {})
    actions = config.get('actions', {}) if isinstance(config, dict) else {}
    if (manifest.get('journal_schema_version') == 5 or manifest.get('input_recipe') == STRUCTURED_FEATURE_VERSION
            or isinstance(actions, dict) and actions.get('version') == '2'):
        actual = action_config_v2(actions)
        require_qualified_contract(actual.action_contract)


def _manifest(manifest: Json, actions: ActionConfigV2, expected_split: str | None) -> tuple[Json, list[tuple[Json, Json]]]:
    check_input_binding({key: manifest.get(key) for key in ('action_contract', 'purpose', 'synthetic_capacity')},
                        action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] not in (3, 4)
            or manifest.get('status') not in ('complete', 'incomplete', 'interrupted')):
        raise ContractError('finalized original collection3/4 required')
    for key, value in {'journal_schema_version': 5, 'feature_version': STRUCTURED_FEATURE_VERSION,
                       'input_recipe': STRUCTURED_FEATURE_VERSION, 'reward_policy_version': 'natural-terminal-v1',
                       'end_projection_version': END_PROJECTION_VERSION, 'policy_input_version': POLICY_INPUT_VERSION,
                       'catalog': CATALOG_META, 'simulator_contract_revision': SIMULATOR_REVISION,
                       'reviewed_simulator_spec_sha256': REVIEWED_SPEC_HASHES, 'observation_schema_version': '1',
                       'action_schema_version': 1}.items():
        if canonical(manifest.get(key)) != canonical(value):
            raise ContractError('original collection semantic declaration mismatch')
    config = manifest.get('effective_config', {})
    if not isinstance(config, dict):
        raise ContractError('original Config object required')
    if digest(config) != manifest.get('config_sha256') or canonical(config.get('actions')) != canonical(asdict(actions)):
        raise ContractError('original collection Config/actions mismatch')
    split = manifest.get('split')
    if split not in ('train', 'validation', 'test') or config.get('run', {}).get('split') != split or expected_split is not None and split != expected_split:
        raise ContractError('original source split mismatch')
    elapsed, limit = manifest.get('elapsed_seconds'), config.get('run', {}).get('wall_seconds')
    if elapsed is None or limit is None:
        raise ContractError('original collection timing provenance missing')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (elapsed, limit)) or not 0 <= elapsed <= limit:
        raise ContractError('original collection timing provenance invalid')
    specs = [(group, spec) for group in manifest.get('schedule', []) for spec in group.get('episodes', [])]
    names = [_name(spec.get('episode_id')) for _, spec in specs]
    for _, spec in specs:
        seeds = spec.get('sampling_seeds')
        if type(seeds) is not list or len(seeds) != 2:
            raise ContractError('original schedule requires two uint64 participant seeds')
        for seed in seeds:
            integer(seed, 0, 2**64 - 1)
    episodes = manifest.get('episodes')
    if not names or len(set(names)) != len(names) or not isinstance(episodes, list) or len(episodes) != len(names) or {item.get('episode_id') for item in episodes} != set(names):
        raise ContractError('entire original cohort required')
    game_ids = [item.get('game_id') for item in episodes]
    if any(type(game) is not str or not game for game in game_ids) or len(set(game_ids)) != len(game_ids):
        raise ContractError('duplicate or invalid original game/participant identity')
    if manifest['schema_version'] == 4:
        from gf_training.lineage import validate_collection_linkage
        validate_collection_linkage(manifest)  # Structural linkage only, never parent qualification.
        seats = {spec['episode_id']: integer(spec.get('learner_viewer'), 1, 2) for _, spec in specs}
        link = manifest.get('ppo', {})
        if manifest.get('collection_kind') not in ('ppo_batch', 'ppo_validation') or link.get('learner_seats') != seats:
            raise ContractError('original explicit learner role linkage mismatch')
    return {item['episode_id']: item for item in episodes}, specs


def _header_identity(header: Json, manifest: Json, episode: Json, spec: Json, actions: ActionConfigV2, viewer: int) -> None:
    name, policies = spec['episode_id'], spec['policies']
    if (header['run_id'] != manifest.get('run_id') or header['episode_id'] != name or header['viewer'] != viewer
            or header['game_id'] != episode.get('game_id') or header['policy'] != policies[viewer - 1]
            or header['action_config_hash'] != digest(asdict(actions))):
        raise ContractError('original journal/manifest actor identity mismatch')
    if header.get('sampling_seed') != spec['sampling_seeds'][viewer - 1] or type(header.get('sampling_seed')) is not int:
        raise ContractError('original journal sampling stream differs from scheduled participant')
    expected_caps = {'max_interval_events': 4096, 'max_episode_events': 262144, 'max_decisions': None,
                     'max_positions': actions.max_sequence_length, 'max_candidates': actions.max_candidates}
    binding = manifest.get('policy_bindings', {}).get(header['policy'], {})
    closed(binding, {'model_id', 'policy_options'})
    if binding.get('model_id') != header['model_id']:
        raise ContractError('original policy binding model mismatch')
    if header['model_version'] == '1':
        if binding.get('policy_options') is not None or header['policy'] != header['model_id'] + '-structured-v1':
            raise ContractError('original baseline execution identity mismatch')
    else:
        options = safe_options_v2(binding.get('policy_options'), action_contract=actions.action_contract,
                                  purpose='component', synthetic_capacity=False)
        model = model_config_v2(options['model'])
        inference = InferenceBindingV2(**options['inference'])
        if model.max_positions < actions.max_sequence_length or inference.max_candidates < actions.max_candidates:
            raise ContractError('saved policy capacity does not cover original actions')
        expected_caps.update(max_interval_events=model.event.max_interval, max_episode_events=model.event.max_episode,
                             max_decisions=model.temporal.max_decisions)
        if (header['model_id'] != model.model_id or header['aggregate_window'] != model.event.aggregate_window
                or header['policy'] != policy_label(model, inference, options['expected_manifest_digest'],
                                                    cuda_visible_devices=options['cuda_visible_devices'])):
            raise ContractError('original neural execution identity/window mismatch')
    if canonical(header.get('caps')) != canonical(expected_caps):
        raise ContractError('original journal caps differ from declared actions/policy')


def inspect_component_collection(directory: Path, *, actions: ActionConfigV2, settings: PreparationConfig,
                                 expected_split: str | None = None, deadline: float = float('inf')) -> ComponentCollection:
    """Read structural components only; no Generation2/whole-parent qualification."""
    settings = preparation_config(asdict(settings))
    if settings.backend != 'python' or settings.reader != 'structured-v1':
        raise ContractError('explicit Python structured source reader required; no native fallback')
    deadline = min(deadline, time.monotonic() + settings.wall_seconds - settings.cleanup_seconds)
    owner = _SourceOwner(directory, settings, deadline)
    manifest = owner.object(owner.root / 'manifest.json')
    episodes, specs = _manifest(manifest, actions, expected_split)
    report = owner.object(owner.root / 'report.json')
    if (type(report.get('schema_version')) is not int or report['schema_version'] != 1
            or type(report.get('planned_games')) is not int or type(report.get('natural_denominator')) is not int
            or any(not isinstance(report.get(key), dict) for key in ('failures', 'health_failures', 'group_health_failures'))):
        raise ContractError('original report health/count ledger required')
    if owner.files['report.json']['sha256'] != manifest.get('report_sha256') or report.get('planned_games') != len(specs):
        raise ContractError('original linked report hash/cohort mismatch')
    natural = sum(item.get('end_kind') == 'natural' for item in episodes.values())
    if report.get('natural_denominator') != natural:
        raise ContractError('original report natural count mismatch')
    if (manifest['status'] == 'complete' or report.get('comparison_valid') is True) and natural != len(specs):
        raise ContractError('complete/successful original collection must be all natural')
    if (manifest['status'] == 'complete' or natural == len(specs)) and (report.get('comparison_valid') is not True or any(report.get(key) for key in ('failures', 'health_failures', 'group_health_failures'))):
        raise ContractError('complete/all-natural original collection requires successful report')
    expected_journals = {f'{name}/player-{viewer}.jsonl' for name in episodes for viewer in (1, 2)}
    if set(owner.inventory('*/player-*.jsonl')) != expected_journals or owner.inventory('*/player-*.partial.jsonl'):
        raise ContractError('missing/extra original participant journals')
    if set(owner.inventory('*/result.json')) != {f'{name}/result.json' for name in episodes}:
        raise ContractError('missing/extra original episode results')
    all_columns = set(owner.inventory('*/*.columns'))
    replays = set(owner.inventory('*/replay-player-*.json'))
    if not replays.issubset({f'{name}/replay-player-{viewer}.json' for name in episodes for viewer in (1, 2)}):
        raise ContractError('extra original participant Replay')
    referenced: set[str] = set()
    outputs = []
    closure_counts: Json = {}
    accepted_count = 0
    missing_replays = []
    for group, spec in specs:
        name, policies = spec['episode_id'], spec.get('policies')
        episode = episodes[name]
        if (not isinstance(policies, list) or len(policies) != 2 or episode.get('policies') != policies
                or episode.get('seed') != group.get('seed') or episode.get('pair_id') != spec.get('pair_id')):
            raise ContractError('original schedule/result identity mismatch')
        if episode.get('end_kind') not in ('natural', 'truncated', 'aborted', 'cancelled', 'incomplete'):
            raise ContractError('unknown original episode closure')
        result = owner.object(owner.root / name / 'result.json')
        if any(canonical(result.get(key)) != canonical(value) for key, value in episode.items()):
            raise ContractError('original saved result differs from collection manifest')
        groups = [item for item in manifest.get('groups', []) if name in item.get('episode_ids', [])]
        if len(groups) != 1:
            raise ContractError('original episode group health missing/duplicated')
        if episode['end_kind'] == 'natural' and (episode.get('failure') or any(episode.get('health', {}).values()) or group_health_failures(groups[0])):
            raise ContractError('natural original episode has failure evidence')
        closures = []
        for viewer in (1, 2):
            path = owner.root / name / f'player-{viewer}.jsonl'
            raw = owner.read(path)
            lines = raw.splitlines()
            if any(len(line) > settings.max_record_mib * 1024**2 for line in lines):
                raise ResourceLimit('original journal record bound exceeded before parse')
            rows = [_json(line) for line in lines]
            if any(not isinstance(row, dict) for row in rows):
                raise ContractError('original journal objects required')
            if not rows or rows[0].get('kind') != 'header':
                raise ContractError('original journal header required')
            _header_identity(rows[0], manifest, episode, spec, actions, viewer)
            for row in rows:
                if row.get('kind') == 'decision':
                    descriptor = row.get('table', {})
                    _, size = validate_descriptor(descriptor, expected_file=candidate_basename(viewer, row['decision']),
                        expected_frame_hash=row.get('frame', {}).get('frame_hash'), action_contract=actions.action_contract,
                        purpose='component', synthetic_capacity=False)
                    owner.charge(32 * size + 4096 * descriptor['count'] + 128 * descriptor['elements'] + 65536)
            data = read_structured_journal(path, records=rows, deadline=deadline,
                max_jsonl_bytes=settings.max_source_file_mib * 1024**2, max_record_bytes=settings.max_record_mib * 1024**2,
                action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
            header = data.header
            if data.end['end_kind'] != episode['end_kind'] or data.end['outcome'] != episode.get('outcome') or data.end['simulator_abort_reason'] != episode.get('simulator_abort_reason'):
                raise ContractError('original closure/result mismatch')
            for item in data.sidecars:
                target = Path(item['path'])
                owner._safe(target)
                relative = target.relative_to(owner.root).as_posix()
                if relative in referenced:
                    raise ContractError('candidate sidecar referenced more than once')
                referenced.add(relative)
                owner.remember(target, tuple(item['identity']), item['sha256'])
            decisions = [data.decisions[index] for index in data.accepted]
            replay_path = owner.root / name / f'replay-player-{viewer}.json'
            if replay_path.exists() or replay_path.is_symlink():
                replay = owner.object(replay_path)
                reconcile_replay(replay, viewer, [item.command['action'] for item in decisions], episode.get('outcome'))
                closure = replay_closure(replay, header['game_id'], viewer)
                if closure is not None and closure['type'] == 'terminal' and episode.get('outcome') is None:
                    raise ContractError('Replay cannot invent unknown natural outcome')
                closures.append((True, closure))
            elif episode['end_kind'] == 'natural' or not episode.get('health', {}).get('replay_or_finalization_failure'):
                raise ContractError('missing original participant Replay')
            else:
                closures.append((False, None))
                missing_replays.append(f'{name}/replay-player-{viewer}.json')
            reason = None if episode['end_kind'] == 'natural' else episode.get('simulator_abort_reason') or episode.get('failure') or episode['end_kind']
            records = EpisodeRecords(data.bank, tuple(item.frame for item in decisions), tuple(item.generated.table for item in decisions),
                tuple(item.selection for item in decisions), episode['end_kind'], episode.get('outcome'), reason,
                action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
            outputs.append(ComponentEpisode(name, viewer, header['model_id'], header['model_version'], records))
            accepted_count += len(decisions)
        present = [closure for exists, closure in closures if exists]
        if len(present) == 2 and present[0] != present[1]:
            raise ContractError('participant Replay closure conflict')
        surviving = present[0] if present else None
        if surviving is not None and surviving['type'] == 'aborted' and episode.get('simulator_abort_reason') is None:
            raise ContractError('original abort Replay lacks projected diagnostic cause')
        if episode.get('simulator_abort_reason') is not None and present and (surviving is None or surviving.get('reason') != episode['simulator_abort_reason'] or projected_abort_kind(surviving['reason']) != episode['end_kind']):
            raise ContractError('original abort cause/Replay projection conflict')
        closure_counts[episode['end_kind']] = closure_counts.get(episode['end_kind'], 0) + 1
    if referenced != all_columns:
        raise ContractError('missing/extra original candidate sidecars')
    owner.finalize()
    proof = {'schema_version': 1, 'kind': 'structured-component-source-proof',
             **input_binding(actions.action_contract, 'component', False), 'qualified': False,
             'parent_qualification': 'not_performed', 'manifest_sha256': owner.files['manifest.json']['sha256'],
             'source_set_sha256': digest(owner.files), 'files': owner.files, 'games': len(specs),
             'accepted_decisions': accepted_count, 'closure_counts': closure_counts, 'missing_replays': missing_replays}
    return ComponentCollection(owner.root, tuple(outputs), canonical(manifest).encode(), canonical(proof).encode())


def component_episode_records(collection: ComponentCollection, *, learner_seats: dict[str, int]) -> tuple[EpisodeRecords, ...]:
    """Select explicit original learner roles; this supplies no parent authority."""
    manifest = collection.manifest
    if manifest['schema_version'] != 4 or learner_seats != manifest['ppo']['learner_seats']:
        raise ContractError('component PPO projection requires exact original learner roles')
    selected = tuple(item for item in collection.episodes if learner_seats[item.episode_id] == item.viewer)
    if len(selected) != len(learner_seats) or any(item.model_version != '2' for item in selected):
        raise ContractError('component PPO learner must be a neural model2 role')
    if any(selection.logprob is None or selection.value is None for item in selected for selection in item.records.selections):
        raise ContractError('component PPO rows require finite saved logprob/value')
    return tuple(item.records for item in selected)
