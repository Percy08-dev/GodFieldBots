"""Snapshot-owned declared Tree fixtures, distinct from executed learning Sources."""
from __future__ import annotations

import hashlib
import math
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

from gf_common import card_instance_fixture as fixture
from gf_common.candidate_columns import decode_candidate_columns_v2, validate_descriptor_v2
from gf_common.card_instance import ActionConfigV3, action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import (
    CandidateTokens,
    FrameTokens,
    signature,
    tokenize_frame,
    tokenize_table,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from gf_models.structured_config import InferenceBindingV2, model_config_v3
from gf_models.structured_sealed import SealedScan, _inference_metadata, _SealedContextV3, basename
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV3,
    TreeEncoderConfigV3,
    TreeResourcesV2,
    resource_estimate,
)
from gf_models.structured_tree_encoder import _owned, feature_layout_v3

MANIFEST_FIELDS = {'schema_version', 'kind', 'declaration', 'origin', 'policies', 'cohorts',
                   'files', 'construction_cost', 'content_sha256'}


def _recorded_component_inference(directory: Path, expected: str, frozen: Json,
                                  scan: SealedScan) -> Json:
    """One bounded private coupling to existing sealed kernels, without active queries.

    Revisit/promote the helper if a second consumer needs inference-only component
    inspection. This deliberately never invents an optimizer/resume declaration.
    """
    actions = action_config_v3(frozen['actions'])
    context = _SealedContextV3(LAYOUT, fixture.CONTRACT, card_instance_rules(),
                              'component', False, digest(asdict(actions)))
    inference, model, _ = _inference_metadata(directory, expected, fixture.CONTRACT,
                                             'component', False, scan, context)
    if canonical(asdict(model)) != canonical(frozen['neural_model']):
        raise ContractError('fixture policy differs from homogeneous declared MLP')
    scan.bundle(directory, inference['weights'])
    return {'inference.json': expected, **inference['weights']['files']}


def _ancestors(directory: Path) -> dict[Path, tuple[int, int]]:
    result = {}
    for path in (directory, *directory.parents):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ContractError('fixture original ancestors must be nonsymlink directories')
        result[path] = (info.st_dev, info.st_ino)
    return result


@dataclass(frozen=True)
class FixtureSelectedInput:
    tokens: FrameTokens
    candidates: CandidateTokens
    selected_index: int
    label: float
    population_id: str
    cohort_id: str
    episode: str
    split: str
    viewer: int
    decision: int
    policy_id: str

    def selected_tokens(self) -> CandidateTokens:
        i = integer(self.selected_index, 0, len(self.candidates.kinds)-1)
        first, end = self.candidates.offsets[i:i+2]
        return replace(self.candidates, kinds=(self.candidates.kinds[i],), targets=(self.candidates.targets[i],),
            numerics=(self.candidates.numerics[i],), present=(self.candidates.present[i],),
            offsets=(0, end-first), slots=self.candidates.slots[first:end])


class TreeFixtureSource:
    """Concrete original-file owner. A summary/flag/sample cannot issue this lifetime."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith('_') and getattr(self, '_active', False):
            raise AttributeError('active fixture original configuration is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, path: Path, expected_digest: str, settings: PreparationConfig, *,
                 max_owned_bytes: int, deadline: float, encoder: TreeEncoderConfigV3 | None = None,
                 extra_owned_bytes: int = 0) -> None:
        self.path = Path(path).absolute()
        if self.path.name != 'manifest.json':
            raise ContractError('fixture originals require manifest.json')
        self.expected_digest = fixture.sha256(expected_digest)
        self.settings = preparation_config(asdict(settings))
        if self.settings.backend != 'python' or self.settings.reader != 'participant-structured-v2':
            raise ContractError('fixture Source requires the explicit serial corrected Python reader')
        integer(max_owned_bytes, 1, 12*1024**3); integer(extra_owned_bytes, 0, 2**63-1)
        if type(deadline) not in (float, int) or not math.isfinite(deadline) or deadline <= 0:
            raise ContractError('fixture original finite deadline required')
        self.encoder = TreeEncoderConfigV3() if encoder is None else encoder
        if type(self.encoder) is not TreeEncoderConfigV3 or self.encoder.max_positions != 18:
            raise ContractError('fixture Tree3 requires closed real18 projection')
        self.limit = min(max_owned_bytes, self.settings.memory_budget_mib*1024**2)
        self.deadline = min(deadline, time.monotonic()+self.settings.wall_seconds-self.settings.cleanup_seconds)
        self._scan = SealedScan(max_owned_bytes=self.limit, deadline=self.deadline)
        self._scan.charge(extra_owned_bytes)
        self._extra = extra_owned_bytes
        self._manifest: Json = {}; self._frozen: Json = {}; self._policies: dict[tuple[str, int], str] = {}
        self._roots: list[Path] = []; self._ancestors: dict[Path, tuple[int, int]] = {}
        self._active = self._closed = self._released = self._iterating = self._complete = False
        self._failed = False; self._sink = 0; self._current_bytes = 0
        self._counts: Json = {}; self._receipts: list[Json] = []
        self._diagnostic_borrow: _FixtureEpisodeBorrow | None = None

    def _live(self) -> None:
        if not self._active or self._closed or self._released or self._failed:
            raise ContractError('active concrete TreeFixtureSource required')
        self._scan.check()

    @property
    def manifest(self) -> Json:
        self._live()
        return clone(self._manifest)

    @property
    def original_roots(self) -> tuple[Path, ...]:
        self._live()
        return tuple(self._roots)

    @property
    def retained_bytes(self) -> int:
        self._live()
        return self._scan.retained+self._current_bytes

    def _file(self, path: Path, expected: str, size: int) -> None:
        fixture.sha256(expected)
        integer(size, 1, min(64, self.settings.max_source_file_mib)*1024**2)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ContractError('fixture original file size/type mismatch')
        with self._scan.file(path, expected):
            pass

    def _object(self, path: Path, expected: str, *, record: bool = False) -> Json:
        fixture.sha256(expected)
        cap = min(64, self.settings.max_record_mib if record else self.settings.max_source_file_mib)*1024**2
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= cap:
            raise ContractError('fixture JSON record/file limit or unsafe file')
        return self._scan.object(path, expected)

    def __enter__(self) -> Self:
        if self._active or self._closed:
            raise ContractError('fixture Source lifetime cannot be reopened')
        try:
            self._ancestors.update(_ancestors(self.path.parent)); self._roots.append(self.path.parent)
            self._manifest = self._object(self.path, self.expected_digest)
            raw = closed(self._manifest, MANIFEST_FIELDS)
            if (type(raw['schema_version']) is not int or raw['schema_version'] != 1
                    or raw['kind'] != 'card-instance-tree-fixture-originals'
                    or canonical(raw['origin']) != canonical(fixture.origin())
                    or digest({k: v for k, v in raw.items() if k != 'content_sha256'}) != raw['content_sha256']):
                raise ContractError('fixture originals manifest origin/content mismatch')
            declaration = closed(raw['declaration'], {'file', 'sha256'})
            if declaration['file'] != 'declaration.json':
                raise ContractError('fixed preconstruction fixture declaration required')
            self._frozen = fixture.declaration(self._object(self.path.parent/'declaration.json', declaration['sha256']))
            model = model_config_v3(self._frozen['neural_model'], layout=LAYOUT)
            binding = InferenceBindingV2(**closed(self._frozen['execution'], set(InferenceBindingV2.__dataclass_fields__)))
            if model.max_positions != 18 or binding.device != 'cpu':
                raise ContractError('fixture requires real18 homogeneous CPU component inference')
            cost = closed(raw['construction_cost'], {'seconds', 'scope'})
            if (cost['scope'] != 'fixture-construction-only' or type(cost['seconds']) not in (int, float)
                    or not math.isfinite(cost['seconds']) or cost['seconds'] < 0):
                raise ContractError('actual finite fixture-construction-only cost required')
            files = raw['files']
            if type(files) is not dict or 'declaration.json' not in files or self.path.name in files:
                raise ContractError('complete original fixture file inventory required')
            self._scan.inventory(self.path.parent, {self.path.name, *files})
            for file, item in files.items():
                basename(file); closed(item, {'sha256', 'bytes'})
                self._file(self.path.parent/file, item['sha256'], item['bytes'])
            if files['declaration.json']['sha256'] != declaration['sha256']:
                raise ContractError('fixture declaration inventory mismatch')
            if type(raw['policies']) is not list or len(raw['policies']) != 9:
                raise ContractError('all nine original declared component policies required')
            for i, policy in enumerate(raw['policies']):
                closed(policy, {'population_id', 'generation', 'directory', 'inference_sha256', 'policy_id', 'files'})
                pop = self._frozen['populations'][i//3]['population_id']
                if policy['population_id'] != pop or type(policy['generation']) is not int or policy['generation'] != i % 3:
                    raise ContractError('fixture original policy population/generation mismatch')
                if policy['policy_id'] != fixture.policy_id(self._frozen['fixture_id'], pop, i % 3, policy['inference_sha256']):
                    raise ContractError('fixture policy identity does not bind original inference')
                directory = Path(policy['directory'])
                if not directory.is_absolute() or directory in self._roots:
                    raise ContractError('distinct absolute original policy directories required')
                self._ancestors.update(_ancestors(directory)); self._roots.append(directory)
                if type(policy['files']) is not dict:
                    raise ContractError('original policy file inventory required')
                self._scan.inventory(directory, set(policy['files']))
                for file, item in policy['files'].items():
                    basename(file); closed(item, {'sha256', 'bytes'})
                    self._file(directory/file, item['sha256'], item['bytes'])
                actual = _recorded_component_inference(directory, policy['inference_sha256'], self._frozen, self._scan)
                if actual != {key: value['sha256'] for key, value in policy['files'].items()}:
                    raise ContractError('fixture requires exact inference-only original policy files')
                self._policies[pop, i % 3] = policy['policy_id']
            if type(raw['cohorts']) is not list or len(raw['cohorts']) != 9:
                raise ContractError('fixture complete cohort receipts required')
            expected_files = {'declaration.json'}
            for cohort, receipt in zip(self._frozen['cohorts'], raw['cohorts'], strict=True):
                closed(receipt, {'cohort_id', 'episodes'})
                if receipt['cohort_id'] != cohort['cohort_id'] or type(receipt['episodes']) is not list or len(receipt['episodes']) != len(cohort['episodes']):
                    raise ContractError('fixture original cohort closure mismatch')
                for spec, item in zip(cohort['episodes'], receipt['episodes'], strict=True):
                    closed(item, {'episode', 'file', 'columns'})
                    if item['episode'] != spec['episode'] or item['file'] != spec['episode']+'.json':
                        raise ContractError('fixture original episode membership/order mismatch')
                    if type(item['columns']) is not list or len(item['columns']) != spec['decision_count']:
                        raise ContractError('fixture original decision column membership mismatch')
                    expected_files.add(item['file'])
                    for decision, file in enumerate(item['columns']):
                        if file != spec['episode']+f'-d{decision:04}.bin':
                            raise ContractError('fixture original actor/decision column name mismatch')
                        expected_files.add(file)
            if set(files) != expected_files:
                raise ContractError('fixture missing/added/foreign original files')
            self._active = True
            self.guard()
            return self
        except BaseException:
            self._failed = True; self._closed = True
            self.release()
            raise

    def guard(self) -> None:
        self._live()
        for path, identity in self._ancestors.items():
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
                raise ContractError('fixture original directory identity changed')
        self._scan.finalize()

    def reserve_sink(self, total_bytes: int) -> None:
        self._live(); integer(total_bytes, self._sink, 2**63-1)
        self._admit(self._current_bytes + total_bytes-self._sink)
        self._scan.charge(total_bytes-self._sink)
        self._sink = total_bytes

    def retained_without_sink(self) -> int:
        self._live()
        return self._scan.retained+self._current_bytes-self._sink

    def _admit(self, size: int) -> None:
        self._scan.charge(size, temporary=True)
        model = ConfiguredTreeModelV3(encoder=self.encoder, resources=TreeResourcesV2(max_owned_bytes=self.limit))
        resource_estimate(model, extra_bytes=self._scan.retained + size)

    def _read_episode(self, cohort: Json, spec: Json, item: Json) -> tuple[Any, Any, Any, Json]:
        self._live()
        expected = self._manifest['files'][item['file']]
        held = self._current_bytes if self._diagnostic_borrow is not None else 0
        if self._diagnostic_borrow is not None:
            # SealedScan owns the JSON admission, but cannot see the live
            # diagnostic caller/borrow reservation. Include both before parse.
            self._admit(held+32*expected['bytes']+65536)
        raw = self._object(self.path.parent/item['file'], expected['sha256'], record=True)
        self._admit(held+64*expected['bytes']+65536)
        raw, frames, bank = fixture.envelope(raw, frozen=self._frozen,
            declaration_sha256=self._manifest['declaration']['sha256'], cohort=cohort, spec=spec,
            policies=self._policies, check=self._scan.check)
        model = self._frozen['neural_model']
        if len(frames) > model['temporal']['max_decisions'] or bank.boundary.end > model['event']['max_episode']:
            raise ContractError('fixture original exceeds declared neural episode capacity')
        return raw, frames, bank, expected

    def _column(self, frame: Any, row: Json, file: str) -> tuple[Json, int, Json]:
        if frame.boundary.end-frame.interval_start > self._frozen['neural_model']['event']['max_interval']:
            raise ContractError('fixture original exceeds declared neural interval capacity')
        descriptor = row['columns']
        _, size = validate_descriptor_v2(descriptor, layout=LAYOUT, action_contract=fixture.CONTRACT,
            purpose='component', synthetic_capacity=False, expected_file=file, expected_frame_hash=frame.frame_hash,
            max_bytes=min(64, self.settings.max_source_file_mib)*1024**2)
        declared = self._manifest['files'][file]
        if descriptor['count'] > min(self._frozen['actions']['max_candidates'], self._frozen['execution']['max_candidates']):
            raise ContractError('fixture table exceeds declared action/inference candidate capacity')
        if descriptor['sha256'] != declared['sha256'] or size != declared['bytes']:
            raise ContractError('fixture numeric inventory/descriptor mismatch')
        return descriptor, size, declared

    @property
    def diagnostic_episode_count(self) -> int:
        self._live()
        return sum(len(cohort['episodes']) for cohort in self._frozen['cohorts'])

    @contextmanager
    def borrow_diagnostic_episode(self, ordinal: int, *, extra_owned_bytes: int = 0) -> Iterator[_FixtureEpisodeBorrow]:
        self._live()
        integer(ordinal, 0, self.diagnostic_episode_count-1)
        integer(extra_owned_bytes, 0, self.limit)
        if self._iterating or self._diagnostic_borrow is not None:
            raise ContractError('fixture diagnostic borrow cannot overlap another worklist')
        borrow = _FixtureEpisodeBorrow(self, extra_owned_bytes)
        self._diagnostic_borrow = borrow
        try:
            self.guard()
            position = 0
            for cohort, receipt in zip(self._frozen['cohorts'], self._manifest['cohorts'], strict=True):
                if ordinal < position+len(cohort['episodes']):
                    index = ordinal-position
                    borrow._load(cohort, cohort['episodes'][index], receipt['episodes'][index])
                    break
                position += len(cohort['episodes'])
            borrow.guard()
            yield borrow
            if borrow._iterating:
                raise ContractError('diagnostic reference iterator must retire before borrow closes')
            borrow.guard()
        except BaseException as error:
            try:
                self.guard()
            except BaseException as cleanup:  # noqa: BLE001 - initiating cancellation/failure survives
                error.add_note('fixture diagnostic final guard: '+str(cleanup))
            self._failed = True
            raise
        finally:
            borrow._invalidate()
            self._diagnostic_borrow = None
            self._current_bytes = 0

    def selected_inputs(self) -> Iterator[FixtureSelectedInput]:
        self._live()
        if self._iterating or self._complete or self._diagnostic_borrow is not None:
            raise ContractError('one complete fixture original worklist required')
        self._iterating = True
        counts = {split: {'games': 0, 'participants': 0, 'selected_decisions': 0, 'zero_endpoint_games': 0}
                  for split in ('train', 'validation')}
        try:
            for cohort, receipt in zip(self._frozen['cohorts'], self._manifest['cohorts'], strict=True):
                current = {'cohort_id': cohort['cohort_id'], 'games': 0, 'selected_decisions': 0, 'zero_endpoint_games': 0}
                for spec, item in zip(cohort['episodes'], receipt['episodes'], strict=True):
                    self._live()
                    raw, frames, bank, expected = self._read_episode(cohort, spec, item)
                    for decision, (row, frame, file) in enumerate(zip(raw['decisions'], frames, item['columns'], strict=True)):
                        self._live()
                        descriptor, size, declared = self._column(frame, row, file)
                        self._admit(128*size + 64*expected['bytes'] + 65536)
                        with self._scan.file(self.path.parent/file, declared['sha256']) as (source, info):
                            if info.st_size != size:
                                raise ContractError('fixture numeric original size changed')
                            payload = source.read(size+1)
                            table = decode_candidate_columns_v2(payload, descriptor, layout=LAYOUT, action_contract=fixture.CONTRACT,
                                purpose='component', synthetic_capacity=False, expected_file=file, expected_frame_hash=frame.frame_hash,
                                max_positions=18)
                            table.validate(frame)
                            index = integer(row['selected_index'], 0, len(table.kinds)-1)
                            tokens = tokenize_frame(frame, bank.view(), aggregate_window=self.encoder.aggregate_window, layout=LAYOUT)
                            candidates = tokenize_table(table, frame, 18, aggregate_window=self.encoder.aggregate_window, layout=LAYOUT)
                            validate_frame_tokens(tokens, layout=LAYOUT)
                            validate_candidate_tokens(candidates, tokens, 18, layout=LAYOUT)
                            sample = FixtureSelectedInput(tokens, candidates, index, fixture.label(raw['outcome'], spec['viewer']),
                                cohort['population_id'], cohort['cohort_id'], spec['episode'], cohort['split'], spec['viewer'], decision,
                                raw['selected_policy_id'])
                            self._current_bytes = _owned((raw, frames, bank.view(), sample))+size
                            self._admit(self._current_bytes)
                        # No file handle survives a yielded immutable raw-token sample.
                        self._live()
                        yield sample
                        self._live()
                        self._current_bytes = 0
                        del sample, candidates, tokens, table, payload
                        current['selected_decisions'] += 1
                    current['games'] += 1
                    current['zero_endpoint_games'] += not frames
                    del raw, frames, bank
                split = counts[cohort['split']]
                split['games'] += current['games']; split['participants'] += 2*current['games']
                split['selected_decisions'] += current['selected_decisions']
                split['zero_endpoint_games'] += current['zero_endpoint_games']
                self._receipts.append(current)
            self._counts = counts; self._complete = True
            self.guard()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._current_bytes = 0
            self._iterating = False

    def verify_complete(self) -> Json:
        self._live()
        if not self._complete or self._iterating:
            raise ContractError('all declared fixture episodes/decisions must be consumed')
        self.guard()
        value = {'source_kind': 'card-instance-tree-fixture-originals', 'source_directory': str(self.path.parent),
            'manifest_sha256': self.expected_digest, 'content_sha256': self._manifest['content_sha256'],
            'declaration_sha256': self._manifest['declaration']['sha256'], 'origin': fixture.origin(),
            'label_recipe': fixture.LABEL_RECIPE, 'input_binding': self._frozen['input_binding'],
            'layout_signature': self._frozen['layout_signature'], 'tree_projection': feature_layout_v3(self.encoder),
            'neural_model': self._frozen['neural_model'], 'actions': self._frozen['actions'],
            'execution': self._frozen['execution'],
            'populations': self._frozen['populations'], 'policies': self._manifest['policies'],
            'cohorts': self._manifest['cohorts'], 'counts': self._counts, 'cohort_counts': self._receipts,
            'construction_cost': self._manifest['construction_cost']}
        return clone(value)

    def finalize(self) -> Json:
        return self.verify_complete()

    def close(self) -> None:
        if self._closed:
            return
        if self._diagnostic_borrow is not None:
            raise ContractError('fixture diagnostic borrow must close before Source close')
        # Even failed/partially consumed Sources retain their original final guard.
        self._scan.check()
        for path, identity in self._ancestors.items():
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
                raise ContractError('fixture original directory identity changed at close')
        self._scan.finalize()
        self._closed = True; self._active = False

    def release(self) -> None:
        if not self._closed:
            raise ContractError('fixture reportable close must precede memory-only release')
        self._manifest.clear(); self._frozen.clear(); self._policies.clear(); self._counts.clear()
        self._receipts.clear(); self._scan.files.clear(); self._scan.inventories.clear()
        self._roots.clear(); self._ancestors.clear()
        self._released = True

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        failure = value if isinstance(value, BaseException) else None
        try:
            if failure is None and not self._closed:
                self.finalize()
        except BaseException as error:  # noqa: BLE001 - preserve initiating failure through original guards
            failure = error
        try:
            self.close()
        except BaseException as error:  # noqa: BLE001 - preserve initiating failure through original close
            if failure is None:
                failure = error
            else:
                failure.add_note('fixture original close: '+str(error))
        if self._closed:
            self.release()
        elif failure is not None:
            vars(failure)['_fixture_cleanup_owner'] = self
        if value is None and failure is not None:
            raise failure


class FixtureDiagnosticDecision:
    """Guarded immutable facts from one Source-owned diagnostic borrow."""
    __slots__ = ('_borrow', '_facts')

    def __init__(self, borrow: _FixtureEpisodeBorrow, frame: Any, bank_view: Any,
                 table: Any, sample: FixtureSelectedInput) -> None:
        self._borrow, self._facts = borrow, (frame, bank_view, table, sample)

    @property
    def frame(self) -> Any:
        self._borrow._live(); return self._facts[0]

    @property
    def bank_view(self) -> Any:
        self._borrow._live(); return self._facts[1]

    @property
    def table(self) -> Any:
        self._borrow._live(); return self._facts[2]

    @property
    def sample(self) -> FixtureSelectedInput:
        self._borrow._live(); return self._facts[3]


class _FixtureEpisodeBorrow:
    """Bounded original bytes/reference stream; never completes a Source."""
    __slots__ = (
        '_actions',
        '_active',
        '_bank',
        '_columns',
        '_envelope',
        '_expected',
        '_external',
        '_extra',
        '_frames',
        '_iterating',
        '_raw',
        '_reference',
        '_source',
        '_started',
    )

    def __init__(self, source: TreeFixtureSource, extra_owned_bytes: int) -> None:
        self._source, self._extra = source, extra_owned_bytes
        self._active, self._started, self._iterating = True, False, False
        self._raw: Json = {}; self._frames: tuple[Any, ...] = (); self._bank: Any = None
        self._envelope = self._expected = b''
        self._columns: tuple[bytes, ...] = ()
        self._actions: ActionConfigV3 | None = None
        self._external = 0
        self._reference: Any = None

    def _live(self) -> None:
        self._source._live()
        if not self._active or self._source._diagnostic_borrow is not self:
            raise ContractError('active SAME fixture diagnostic borrow required')

    def _refresh(self) -> None:
        self._source._current_bytes = self._extra+self._external+_owned((self._raw, self._frames,
            None if self._bank is None else self._bank.view(), self._envelope, self._columns, self._expected, self._actions, self._reference))
        self._source._admit(self._source._current_bytes)

    def _admit_temporary(self, size: int) -> None:
        self._source._admit(self._source._current_bytes+size)

    def _load(self, cohort: Json, spec: Json, item: Json) -> None:
        self._live()
        source = self._source
        self._refresh()
        raw, frames, bank, expected = source._read_episode(cohort, spec, item)
        self._raw, self._frames, self._bank = raw, frames, bank
        self._actions = action_config_v3(source._frozen['actions'])
        self._refresh()
        self._admit_temporary(expected['bytes']+65536)
        with source._scan.file(source.path.parent/item['file'], expected['sha256']) as (stream, info):
            if info.st_size != expected['bytes']:
                raise ContractError('fixture original envelope size changed')
            self._envelope = stream.read(info.st_size+1)
            if len(self._envelope) != info.st_size:
                raise ContractError('fixture original envelope length changed')
        # Immutable original bytes remain owned during every column decode.
        self._refresh()
        payloads: list[bytes] = []
        for row, frame, file in zip(raw['decisions'], frames, item['columns'], strict=True):
            descriptor, size, declared = source._column(frame, row, file)
            self._admit_temporary(_owned(payloads)+128*size+65536)
            with source._scan.file(source.path.parent/file, declared['sha256']) as (stream, info):
                if info.st_size != size:
                    raise ContractError('fixture numeric original size changed')
                payload = stream.read(size+1)
                table = decode_candidate_columns_v2(payload, descriptor, layout=LAYOUT, action_contract=fixture.CONTRACT,
                    purpose='component', synthetic_capacity=False, expected_file=file, expected_frame_hash=frame.frame_hash,
                    max_positions=18)
                table.validate(frame)
                integer(row['selected_index'], 0, len(table.kinds)-1)
            payloads.append(payload)
            del table
        self._columns = tuple(payloads)
        self._refresh()
        self._admit_temporary(65536+16384*len(frames))
        metadata = {key: raw[key] for key in ('fixture_id', 'declaration_sha256', 'population_id', 'cohort_id',
            'episode', 'viewer', 'split', 'generation', 'roles', 'selected_policy_id', 'outcome', 'outcome_origin')}
        decisions = []
        for i, (row, frame) in enumerate(zip(raw['decisions'], frames, strict=True)):
            decisions.append({'decision': i, 'revision': frame.revision, 'interval_start': frame.interval_start,
                'interval_start_hash': frame.interval_start_hash, 'boundary': asdict(frame.boundary),
                'frame_hash': frame.frame_hash, 'table_hash': row['columns']['table_hash'],
                'columns_file': row['columns']['file'], 'columns_sha256': row['columns']['sha256'],
                'selected_index': row['selected_index']})
        metadata.update(envelope_sha256=hashlib.sha256(self._envelope).hexdigest(),
            label_recipe=fixture.LABEL_RECIPE, label=fixture.label(raw['outcome'], raw['viewer']),
            input_binding=fixture.input_binding(), layout_signature=self._source._frozen['layout_signature'],
            aggregate_window=source.encoder.aggregate_window,
            token_signature=signature(source.encoder.aggregate_window, layout=LAYOUT),
            boundary=asdict(bank.boundary), decisions=decisions)
        self._expected = canonical(metadata).encode()
        self._refresh()

    @property
    def envelope(self) -> bytes:
        self._live(); return self._envelope

    @property
    def columns(self) -> tuple[bytes, ...]:
        self._live(); return self._columns

    @property
    def actions(self) -> ActionConfigV3:
        self._live()
        assert self._actions is not None
        return self._actions

    @property
    def expected_metadata(self) -> bytes:
        self._live(); return self._expected

    @property
    def retained_bytes(self) -> int:
        self._live(); return self._source.retained_bytes

    def guard(self) -> None:
        self._live()
        try:
            self._source.guard()
        except BaseException:
            self._source._failed = True
            raise

    def admit_external(self, total_bytes: int) -> None:
        self._live(); integer(total_bytes, 0, self._source.limit)
        previous = self._external
        self._external = total_bytes
        try:
            self._refresh()
        except BaseException:
            self._external = previous
            self._source._failed = True
            raise

    def reference_decisions(self) -> Iterator[FixtureDiagnosticDecision]:
        self._live()
        if self._iterating:
            raise ContractError('diagnostic reference worklists cannot overlap')
        self._started = self._iterating = True
        try:
            for i, (row, frame, payload) in enumerate(zip(self._raw['decisions'], self._frames, self._columns, strict=True)):
                self.guard()
                self._admit_temporary(128*len(payload)+65536)
                table = decode_candidate_columns_v2(payload, row['columns'], layout=LAYOUT,
                    action_contract=fixture.CONTRACT, purpose='component', synthetic_capacity=False,
                    expected_file=row['columns']['file'], expected_frame_hash=frame.frame_hash, max_positions=18)
                table.validate(frame)
                tokens = tokenize_frame(frame, self._bank.view(), aggregate_window=self._source.encoder.aggregate_window, layout=LAYOUT)
                candidates = tokenize_table(table, frame, 18, aggregate_window=self._source.encoder.aggregate_window, layout=LAYOUT)
                validate_frame_tokens(tokens, layout=LAYOUT)
                validate_candidate_tokens(candidates, tokens, 18, layout=LAYOUT)
                sample = FixtureSelectedInput(tokens, candidates, row['selected_index'],
                    fixture.label(self._raw['outcome'], self._raw['viewer']), self._raw['population_id'],
                    self._raw['cohort_id'], self._raw['episode'], self._raw['split'], self._raw['viewer'], i,
                    self._raw['selected_policy_id'])
                self._reference = (table, sample)
                self._refresh()
                yield FixtureDiagnosticDecision(self, frame, frame.bank_view(self._bank.view()), table, sample)
                self.guard()
                del sample, candidates, tokens, table
                self._reference = None
                self._refresh()
            self.guard()
        except BaseException:
            self._source._failed = True
            raise
        finally:
            self._iterating = False
            self._reference = None
            if self._active and not self._source._failed:
                self._refresh()

    def _invalidate(self) -> None:
        self._active = False
        self._raw = {}; self._frames = (); self._bank = None
        self._envelope = self._expected = b''; self._columns = ()
        self._reference = None; self._external = 0; self._actions = None
