"""Real Dataset3 originals/accounting; no array or summary fitting permission."""
from __future__ import annotations

import math
import stat
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from gf_common import card_instance_game_tree as game
from gf_common.card_instance_game_tree import MAX_OWNED_BYTES, managed_memory
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import (
    CandidateTokens,
    FrameTokens,
    signature,
    tokenize_frame,
    tokenize_table,
)
from gf_models.structured_sealed import SEMANTIC_KEYS_V3, SealedScan
from gf_models.structured_tree_config import TreeArchitectureV3, TreeEncoderConfigV3
from gf_models.structured_tree_encoder import feature_layout_v3
from gf_training.card_instance_source_locks import GameSourceCustody, ValidationManifestOwnership
from gf_training.structured_tree_data import _check_output_ancestors, _outside_output

if TYPE_CHECKING:
    from gf_training.card_instance_final_validation import GameFinalValidationSource
    from gf_training.card_instance_formal import FinalSourceBudgetLoan, FormalGameParent
    from gf_training.card_instance_source import OriginalGameDecision


class GameTreeBudget:
    """ONE closed three-parent Dataset/controller/full-child resource governor.

    Intrinsic amounts exclude each owner's external base/local claims. This
    resource bookkeeping never qualifies a Source, Dataset or fit.
    """
    def __init__(self, total_bytes: int, *, dataset_bytes: int = 0, child_limit_bytes: int = 0) -> None:
        integer(total_bytes, 1, MAX_OWNED_BYTES)
        self.limit = total_bytes
        self._parents = [0, 0, 0]
        self._sources = [0, 0, 0]
        self._dataset = integer(dataset_bytes, 0, total_bytes)
        self._child = integer(child_limit_bytes, 0, total_bytes)
        self._active = self._sink = self._publisher = 0
        self._peak = 0
        self._failure: BaseException | None = None
        self._gate = threading.Lock()
        self._check()

    def _sum(self) -> int:
        return self._dataset + sum(self._parents) + sum(self._sources) + self._active + self._sink + self._publisher + self._child

    def _check(self) -> None:
        if self._failure is not None:
            raise ContractError('game Dataset budget permanently failed') from self._failure
        size = self._sum()
        if size > self.limit:
            error = ContractError('ALL retained game originals/controller/full child exceed joint8GiB')
            self._failure = error
            raise error
        self._peak = max(self._peak, size)

    def _set(self, kind: str, amount: int, index: int | None = None, *, retired: bool = False) -> None:
        integer(amount, 0, self.limit)
        with self._gate:
            self._check()
            if kind in ('parent', 'source'):
                assert index is not None
                integer(index, 0, 2)
                values = self._parents if kind == 'parent' else self._sources
                old = values[index]
                if amount < old and not retired:
                    raise ContractError('actual owner retirement required before lowering budget')
                values[index] = amount
            else:
                attribute = {'dataset': '_dataset', 'active': '_active', 'sink': '_sink',
                    'publisher': '_publisher', 'child': '_child'}[kind]
                old = getattr(self, attribute)
                if amount < old and not retired:
                    raise ContractError('actual buffers must retire before lowering budget')
                setattr(self, attribute, amount)
            self._check()

    def parent_intrinsic(self, index: int, amount: int, *, retired: bool = False) -> None:
        self._set('parent', amount, index, retired=retired)

    def source_intrinsic(self, index: int, amount: int, *, retired: bool = False) -> None:
        self._set('source', amount, index, retired=retired)

    def reserve_dataset(self, amount: int, *, retired: bool = False) -> None:
        self._set('dataset', amount, retired=retired)

    def reserve_active(self, amount: int, *, retired: bool = False) -> None:
        self._set('active', amount, retired=retired)

    def reserve_sink(self, amount: int, *, retired: bool = False) -> None:
        self._set('sink', amount, retired=retired)

    def reserve_publisher(self, amount: int, *, retired: bool = False) -> None:
        self._set('publisher', amount, retired=retired)

    def reserve_child(self, configured_limit: int) -> None:
        # Never min/downscale the configured cap to remaining room.
        self._set('child', configured_limit)

    def external_base(self, parent_index: int, *, source_local_bytes: int = 0) -> int:
        integer(parent_index, 0, 2)
        integer(source_local_bytes, 0, self.limit)
        with self._gate:
            self._check()
            # Parent native/session intrinsic already contains any native raw
            # export. Source residuals and local copies must not include it again.
            return self._sum() - self._parents[parent_index] - self._sources[parent_index] + source_local_bytes

    @property
    def retained_bytes(self) -> int:
        with self._gate:
            self._check()
            return self._sum()

    @property
    def peak_bytes(self) -> int:
        return self._peak

    def fit_memory(self) -> Json:
        with self._gate:
            self._check()
            if not self._child:
                raise ContractError('full configured child reservation required for fit')
            return managed_memory(self.limit, self._sum() - self._child, self._child)


def _note(primary: BaseException, label: str, secondary: BaseException) -> None:
    try:
        primary.add_note(label + str(secondary))
    except BaseException:  # noqa: BLE001, S110 - diagnostic allocation cannot replace initiating failure
        pass


def _encoder(value: TreeEncoderConfigV3) -> TreeEncoderConfigV3:
    if type(value) is not TreeEncoderConfigV3:
        raise ContractError('closed real18 Tree encoder required')
    TreeArchitectureV3(encoder=value)
    return value


def _settings(value: PreparationConfig, total: int, deadline: float) -> PreparationConfig:
    integer(total, 1, MAX_OWNED_BYTES)
    if type(value) is not PreparationConfig:
        raise ContractError('closed recorded preparation settings required')
    result = preparation_config(asdict(value))
    if (result.reader != 'participant-structured-v2' or result.memory_budget_mib*1024**2 < total
            or type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic()):
        raise ContractError('explicit corrected reader, unchanged joint allowance and live deadline required')
    return result


def _reader_preflight(settings: PreparationConfig, *, inspect_recorded: bool = False) -> None:
    """Structural runtime gate only, never a Source from a capability marker."""
    if settings.backend == 'python':
        return
    from gf_training.native_backend import original_history_reader_settings
    if inspect_recorded:
        # Each protected original parent will negotiate against its recorded build.
        original_history_reader_settings(settings, _defer_draw_capability=True)
    else:
        original_history_reader_settings(settings)


def _tree_tokens(decision: OriginalGameDecision, encoder: TreeEncoderConfigV3) -> tuple[FrameTokens,CandidateTokens]:
    """Project original own history at the Tree window; no Source permission."""
    tokens = tokenize_frame(decision.frame,decision.bank,aggregate_window=encoder.aggregate_window,layout=LAYOUT)
    candidates = tokenize_table(decision.table,decision.frame,18,aggregate_window=encoder.aggregate_window,layout=LAYOUT)
    decision.table.validate(decision.frame)
    return tokens,candidates


def _relative(directory: Path, value: Any) -> Path:
    if type(value) is not str or not value:
        raise ContractError('original management reference must be a safe relative path')
    path = Path(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ContractError('foreign original management reference')
    result = directory/path
    for parent in (result.parent, *result.parent.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ContractError('original management ancestors must be nonsymlink directories')
        if parent == directory:
            return result
    raise ContractError('original management reference escaped retained root')


def _management_source_set(directory: Path, entries: Any, controls: dict[Path,Any]) -> str:
    """Original-proof metadata/stat recipe ONLY; supplies no Source permission.

    Caller must pre-admit metadata and bracket with its actual Source guards.
    This leaf never reads J6/codec/Replay, outcomes or selections.
    """
    if type(entries) is not list:
        raise ContractError('complete canonical final inventory required')
    files: Json = {}
    for entry in entries:
        closed(entry, {'path','sha256','bytes'})
        path = _relative(directory,entry['path']); name = path.relative_to(directory).as_posix()
        if name in files or name in ('manifest.json','report.json'):
            raise ContractError('duplicate/foreign original inventory control')
        sha = game.sha256(entry['sha256']); size = integer(entry['bytes'],0,64*1024**2)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size!=size:
            raise ContractError('original final inventory path/type/size changed')
        files[name] = {'sha256':sha,'bytes':size,
            'identity':[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]}
    for name in ('manifest.json','report.json'):
        path = directory/name
        identity,sha = controls[path]
        game.sha256(sha)
        info = path.lstat()
        actual = (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        if not stat.S_ISREG(info.st_mode) or tuple(identity)!=actual or not 0<info.st_size<=64*1024**2:
            raise ContractError('original final management control identity changed')
        files[name] = {'sha256':sha,'bytes':identity[2],'identity':list(identity)}
    return digest(files)


def _verified_fit_rows(metadata: Json, actual: Json) -> Json:
    """Count correspondence only; caller must supply SAME actual owner's facts."""
    closed(actual,{'train','validation'})
    for count in actual.values():
        integer(count,0,2**63-1)
    if any(metadata['counts'][split]['selected_decisions']!=actual[split] for split in actual):
        raise ContractError('Dataset index selected counts differ from actual original owners before private rows')
    return dict(actual)


def _cohort_comparison(state: Json, neural: Json) -> str:
    """Memory-only equality over protected originals; never Source permission."""
    identity = closed(state['identity'],{'code','module_sha256','cuda_distributions','runtime','numerics',
        'package','integration','configuration_inputs_sha256','preprocessing','process_envelope_recipe','formal_parent'})
    hardware = closed(state['hardware'],{'device','name','cuda_runtime','device_count','cuda_driver_api_version','exact_driver_build'})
    # These two per-run references are independently retained/validated in each
    # parent row. Common literal recipe/algorithm bindings remain exact in neural.
    common = {key:value for key,value in identity.items() if key not in ('configuration_inputs_sha256','formal_parent')}
    return canonical({'config':game.normalized_formal_config(state['effective_config']),
        'neural':neural,'hardware':hardware,'identity':common})


@dataclass(frozen=True)
class GameSelectedInput:
    """One actual borrowed own selected endpoint; diagnostics never become features."""
    tokens: FrameTokens
    candidates: CandidateTokens
    selected_index: int
    original_selection: Any
    label: float
    parent_index: int
    cohort_index: int
    episode_id: str
    viewer: int
    decision: int
    policy_label: str

    def selected_tokens(self) -> CandidateTokens:
        index = integer(self.selected_index, 0, len(self.candidates.kinds)-1)
        first, end = self.candidates.offsets[index:index+2]
        return replace(self.candidates, kinds=(self.candidates.kinds[index],),
            targets=(self.candidates.targets[index],), numerics=(self.candidates.numerics[index],),
            present=(self.candidates.present[index],), offsets=(0, end-first), slots=self.candidates.slots[first:end])


class _GameTreeOriginals:
    """One concrete three-parent/raw-source lifetime, never constructed from arrays."""
    _packed = False

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith('_') and getattr(self,'_entered',False):
            raise AttributeError('original game Source configuration is immutable after entry')
        object.__setattr__(self,name,value)

    def __init__(self, parents: tuple[Path, ...], validations: tuple[Path, ...], settings: PreparationConfig, *,
                 encoder: TreeEncoderConfigV3, max_owned_bytes: int, deadline: float,
                 extra_owned_bytes: int = 0, child_limit_bytes: int = 0,
                 ownership: GameSourceCustody | None = None, packed: bool = False,
                 _inspect_recorded: bool = False) -> None:
        self._packed = packed
        self.settings = _settings(settings, max_owned_bytes, deadline)
        _reader_preflight(self.settings, inspect_recorded=_inspect_recorded)
        self.encoder = _encoder(encoder)
        self.limit, self.deadline = max_owned_bytes, deadline
        self._custody = GameSourceCustody(parents, validations, deadline=deadline, ownership=ownership, packed=packed)
        self.original_roots = (*self._custody.parents, *self._custody.validations)
        self._budget = GameTreeBudget(max_owned_bytes, dataset_bytes=extra_owned_bytes+131072,
            child_limit_bytes=child_limit_bytes)
        self._extra = extra_owned_bytes
        self._scan = SealedScan(max_owned_bytes=max_owned_bytes, deadline=deadline)
        self._logical_controls: dict[Path, str] = {}
        self._parents: list[FormalGameParent] = []
        self._sources: list[GameFinalValidationSource] = []
        self._source_scopes: list[Any] = []
        self._loans: list[FinalSourceBudgetLoan] = []
        self._states: list[Json] = []
        self._records: list[list[Json]] = []
        self._parent_rows: list[Json] = []
        self._collections: list[Json] = []
        self._controls = 131072
        self._persistent = 131072
        self._read_peak = 0
        self._sink = 0
        self._manifest: Json = {}
        self._neural: Json = {}
        self._comparison: str | None = None
        self._active = self._entered = self._closed = self._iterating = self._complete = False
        self._failure: BaseException | None = None
        self._current_feature_bytes = 0
        self._retired_sources: set[int] = set()
        self._retired_parents: set[int] = set()

    def _live(self) -> None:
        if self._failure is not None or not self._active or self._closed:
            raise ContractError('SAME active qualified-game originals owner required') from self._failure
        self._custody.require()
        self._scan.check()

    def _sync(self, *, retired: bool = False) -> None:
        # Getters are exclusive intrinsic counters, not external logical bases.
        for index, parent in enumerate(self._parents):
            self._budget.parent_intrinsic(index, parent.resident_bytes, retired=retired)
        for index, source in enumerate(self._sources):
            self._budget.source_intrinsic(index, source.resident_bytes, retired=retired)
        for index, loan in enumerate(self._loans):
            loan.update_base(self._budget.external_base(index), custodian=self._custody)

    def _reserve_controls(self, amount: int) -> None:
        self._controls = max(self._controls, integer(amount, 0, self.limit))
        self._budget.reserve_dataset(self._extra+self._controls)
        self._sync()

    def _object(self, path: Path, expected: str | None = None) -> Json:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= min(64, self.settings.max_source_file_mib)*1024**2:
            raise ContractError('game management original must be regular and bounded64MiB')
        # Source/parent native construction and all prior parsed controls remain
        # charged BEFORE stat-sized JSON read/decode/copy, never after allocation.
        self._read_peak = max(self._read_peak, 32*info.st_size+65536)
        self._reserve_controls(self._persistent+self._read_peak+self._scan.retained+4096+4*len(str(path)))
        reader = SealedScan(max_owned_bytes=32*info.st_size+65536+4096+4*len(str(path)), deadline=self.deadline)
        value = reader.object(path, expected)
        snapshot = reader.files[path]
        if path in self._scan.files and self._scan.files[path] != snapshot:
            raise ContractError('original game management snapshot changed')
        if path not in self._scan.files:
            self._scan.charge(4096+4*len(str(path)))
            self._scan.files[path] = snapshot
        reader.finalize()
        return value

    def _logical_object(self, owner: Any, path: Path, expected: str | None = None) -> Json:
        from gf_training.group_pack_game_source import PackedSourceOwner
        if not self._packed or type(owner) is not PackedSourceOwner:
            raise ContractError('exact retained Dataset4 original wrapper required')
        name = path.relative_to(owner.root).as_posix()
        size = owner._entries[name].get('length', owner._entries[name].get('bytes'))
        self._read_peak = max(self._read_peak, 32*size+65536)
        self._reserve_controls(self._persistent+self._read_peak+self._scan.retained+4096+4*len(str(path)))
        value = owner.object(path)
        sha = owner.files[name]['sha256']
        if expected is not None and sha != game.sha256(expected):
            raise ContractError('packed original Dataset control hash differs')
        previous = self._logical_controls.get(path)
        if previous is not None and previous != sha:
            raise ContractError('packed Dataset original control changed')
        self._logical_controls[path] = sha
        owner.finalize()
        return value

    def _control_digest(self, path: Path) -> str:
        return self._logical_controls[path] if path in self._logical_controls else self._scan.files[path][1]

    def __enter__(self) -> Self:
        if self._entered or self._closed:
            raise ContractError('game original custody has exactly one lifetime')
        self._entered = True
        try:
            self._custody.__enter__()
            identities = [(path.stat().st_dev, path.stat().st_ino) for path in self.original_roots]
            if len(set(identities)) != 6:
                raise ContractError('three distinct parent/final-validation populations required')
            # These are approved concrete seams, supplied by NativeActor. There
            # is no alternate parser/backend/summary constructor if absent.
            from gf_training.card_instance_final_validation import open_parent_game_final_validation
            from gf_training.card_instance_formal import FinalSourceBudgetLoan, FormalGameParent
            for index, directory in enumerate(self._custody.parents):
                parent = FormalGameParent(directory, max_owned_bytes=self.limit, deadline=self.deadline,
                    ownership=self._custody.ownership_for(directory), extra_owned_bytes=self._budget.retained_bytes)
                self._parents.append(parent)  # failed opening owner remains reachable
                parent.__enter__()
                self._sync()
                loan = FinalSourceBudgetLoan(parent, custodian=self._custody,
                    base_bytes=self._budget.external_base(index), deadline=self.deadline)
                self._loans.append(loan)
                size = sum(parent.canonical_control_sizes)
                self._persistent += 32*size+65536
                self._reserve_controls(self._persistent+self._read_peak+self._scan.retained)
                state, records = parent.state, parent.records
                if state is None or records is None:
                    raise ContractError('SAME admitted parent must retain complete original state/generations')
                from gf_common.card_instance_formal import PACKED_RUN_VARIANT, RUN_VARIANT
                if (type(state.get('schema_version')) is not int or state['schema_version'] != (4 if self._packed else 3)
                        or state.get('run_variant') != (PACKED_RUN_VARIANT if self._packed else RUN_VARIANT)):
                    raise ContractError('Dataset requires three homogeneous explicit formal storage variants')
                if state['seed'] != game.SEEDS[index]:
                    raise ContractError('actual original formal parents must be ordered17/29/43')
                if state['effective_config']['preprocessing']['backend'] != self.settings.backend:
                    raise ContractError('Dataset reader backend differs from actual original parent; no fallback')
                self._states.append(state); self._records.append(records)
                self._parent_rows.append(self._parent_row(index))
                token = self._custody.ownership_for(self._custody.validations[index])
                from gf_training.card_instance_source_locks import ValidationLocatorOwnership
                if type(token) is not (ValidationLocatorOwnership if self._packed else ValidationManifestOwnership):
                    raise ContractError('SAME canonical-manifest custody token required')
                scope = open_parent_game_final_validation(parent, self._custody.validations[index],
                    manifest_ownership=token, budget_loan=loan, max_owned_bytes=self.limit, deadline=self.deadline)
                source = scope.__enter__()
                self._source_scopes.append(scope)
                self._sources.append(source)
                self._sync()
            self._active = True
            self.recheck()
            return self
        except BaseException as error:
            self._failure = error
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain uncertain actual owners/custody
                vars(error)['_game_tree_dataset_cleanup_owner'] = self
                _note(error, 'game originals opening cleanup: ', cleanup)
            raise

    def _parent_row(self, index: int) -> Json:
        from gf_training.lineage import checkpoint_path, generation_path
        state, records, directory = self._states[index], self._records[index], self._custody.parents[index]
        run = self._object(directory/'run.json')
        if canonical(run) != canonical(state):
            raise ContractError('actual protected formal state differs from canonical run')
        initial = self._object(checkpoint_path(directory, 0)/'inference.json', records[0]['inference_sha256'])
        neural = {'model': initial['model'], 'actions_sha256': initial['action_config_hash'],
            'execution': state['execution'], 'semantics': {key: initial[key] for key in SEMANTIC_KEYS_V3},
            **{key: state['formal_declaration'][key] for key in ('recipe_id','recipe_sha256','algorithm_recipe_sha256')},
            'code': state['identity']['code']}
        comparison = _cohort_comparison(state,neural)
        if self._comparison is None:
            self._comparison = comparison; self._neural = neural
        elif self._comparison != comparison:
            raise ContractError('all three full formal variants/code/recipe/execution must be homogeneous')
        generations = []
        for number, record in enumerate(records):
            self._object(generation_path(directory, number))
            generations.append({'generation': number,
                'generation_sha256': self._scan.files[generation_path(directory, number)][1],
                **{key: record[key] for key in ('update_sha256', 'inference_sha256', 'resume_sha256')},
                'artifact_files_sha256': digest(record['artifact_files'])})
        self._object(directory/'artifact-registry.json')
        return {'parent_index': index, 'directory': str(directory), 'run_id': state['run_id'], 'seed': state['seed'],
            'run_sha256': self._scan.files[directory/'run.json'][1],
            'formal_declaration_sha256': state['formal_declaration']['sha256'], 'config_sha256': state['config_sha256'],
            'config_inputs_sha256': digest(state['config_inputs']), 'resume_identity': state['resume_identity'],
            'registry_sha256': self._scan.files[directory/'artifact-registry.json'][1],
            'final_generation': state['updates'], 'final_generation_sha256': state['generation_sha256'],
            'final_inference_sha256': records[-1]['inference_sha256'], 'generations': generations,
            'elapsed_seconds': state['elapsed_seconds'],
            **({'storage_binding': state['storage_binding']} if self._packed else {})}

    def recheck(self) -> None:
        self._live()
        try:
            for parent in self._parents:
                parent.recheck()
            for source in self._sources:
                source.recheck()
            self._scan.finalize(); self._custody.require()
        except BaseException as error:
            self._failure = error
            raise

    @property
    def retained_bytes(self) -> int:
        self._live(); self._sync()
        return self._budget.retained_bytes

    @property
    def current_feature_bytes(self) -> int:
        self._live()
        return self._current_feature_bytes

    def reserve_sink(self, amount: int) -> None:
        self._live()
        self._sink = max(self._sink, integer(amount, 0, self.limit))
        self._budget.reserve_sink(self._sink); self._sync()

    def _retire_sink(self) -> None:
        # Called only after _PreparedTreeFit has cleared ALL row/label/packing
        # references. No iterator end or Source close can retire those arrays.
        if not self._entered or self._closed:
            raise ContractError('actual entered game controller resource owner required')
        self._sink = 0
        if self._failure is None and self._budget._failure is None:
            self._budget.reserve_sink(0, retired=True); self._sync()
        # A failed owner keeps its old conservative external counter. Clearing
        # actual buffers is cleanup-only; this path never revives permission.

    def retained_without_sink(self) -> int:
        return self.retained_bytes-self._sink-self._budget._child-self._budget._publisher

    def reserve_fit(self, amount: int) -> None:
        """Controller workspace envelope separate from FULL child and row sink."""
        self._live(); self._budget.reserve_publisher(max(amount,self._budget._publisher)); self._sync()

    def fit_memory(self) -> Json:
        self._live(); self._sync()
        return self._budget.fit_memory()

    def verified_row_counts(self) -> Json:
        self._live()
        counts = {'train':sum(state['accepted_decisions'] for state in self._states),'validation':0}
        for index in range(len(self._sources)):
            facts = self._facts(index)
            counts['validation'] += integer(facts['selected_decisions'],0,2**63-1)
            del facts
            self._sync()
        return counts

    def _facts(self, index: int) -> Json:
        self._live(); self._sync()
        state_bytes,_ = self._parents[index].canonical_control_sizes
        # Caller owns the returned clone, separately from Source's internal
        # temporary/counter. Original run_id is bounded by protected state bytes;
        # JSON path escaping by6*UTF8 bytes. The closed14 fields, projection4
        # hashes/literal and invocation6 finite numbers fit a16KiB header even
        # for309-digit finite integral clocks. No observed-run maxima or raw
        # episode/NN buffers are used to admit this management-only clone.
        encoded = state_bytes+6*len(str(self._custody.validations[index]).encode())+16384
        if self._packed:
            encoded += len(self._sources[index]._facts)
        self._reserve_controls(self._persistent+self._read_peak+self._scan.retained+32*encoded+65536)
        return self._sources[index].facts

    @property
    def cleanup_complete(self) -> bool:
        return (self._closed and self._custody.cleanup_complete
            and all(source.cleanup_complete for source in self._sources)
            and all(parent.cleanup_complete for parent in self._parents))

    def _final_source_set(self, index: int, directory: Path, manifest: Json) -> str:
        """SAME original-proof file-map recipe, management/stat data ONLY.

        The actual Source brackets this derivation with whole-original guards.
        No referenced J6/codec/Replay bytes are parsed, decoded or used as facts.
        """
        source = self._sources[index]
        source.recheck()
        entries = manifest['files']
        if type(entries) is not list:
            raise ContractError('complete canonical final inventory required')
        self._reserve_controls(self._persistent+self._read_peak+self._scan.retained+4096*(len(entries)+2))
        self._object(directory/'report.json')
        result = _management_source_set(directory,entries,self._scan.files)
        source.recheck()
        return result

    def _collection(self, index: int, generation: int, *, validation: bool) -> tuple[Json, Json, Json | None]:
        directory = self._custody.parents[index]
        update: Json | None = None
        if validation:
            directory = self._custody.validations[index]
            facts = self._facts(index)
            if self._packed:
                slot = self._parents[index]._packed_final_originals
                if (slot is None or slot.owner is None or facts.get('schema_version') != 2
                        or facts.get('kind') != 'card-instance-packed-final-validation-facts'
                        or self._sources[index]._pending is not None):
                    raise ContractError('Dataset4 requires SAME PUBLIC packed final64 Source')
                packed_owner = slot.owner
                manifest = self._logical_object(packed_owner, directory/'manifest.json', facts['manifest_sha256'])
                source_set = facts['source_set_sha256']
            else:
                manifest = self._object(directory/'manifest.json', facts['manifest_sha256'])
                source_set = self._final_source_set(index, directory, manifest)
            numeric = None
            elapsed = manifest['elapsed_seconds']
        else:
            record = self._records[index][generation+1]
            update = self._object(directory/'updates'/f'{generation+1:06}.json', record['update_sha256'])
            collection_directory = _relative(directory, update['collection'])
            if self._packed:
                packed_owner = self._parents[index].scan.packed_owner(collection_directory)
            manifest = (self._logical_object(packed_owner,
                collection_directory/'manifest.json', update['collection_manifest_sha256']) if self._packed else
                self._object(collection_directory/'manifest.json', update['collection_manifest_sha256']))
            numeric_directory = _relative(directory, update['numeric'])
            numeric_metadata = self._object(numeric_directory/'rollout.json', update['numeric_sha256'])
            if (numeric_metadata['schema_version'] != 4 or numeric_metadata['kind'] != 'structured-rollout'
                    or numeric_metadata['decision_count'] != update['accepted_decisions']):
                raise ContractError('actual complete original Num4 control reference differs')
            numeric = {'directory': str(numeric_directory), 'sha256': update['numeric_sha256'], 'files': update['numeric_files']}
            source_set = update['source_proof']['source_set_sha256']
            elapsed = update['collection_seconds']
            directory = collection_directory
        link = manifest['linkage']
        if (manifest['collection_role'] != ('ppo-final-validation' if validation else 'ppo-batch')
                or link['parent_run_id'] != self._states[index]['run_id']
                or Path(link['parent_run_path']) != self._custody.parents[index]
                or link['final_generation' if validation else 'collection_generation'] != generation):
            raise ContractError('original C5 must belong to SAME actual parent/generation/role')
        row = {'parent_index': index, 'directory': str(directory), 'collection_role': manifest['collection_role'],
            'split': 'validation' if validation else 'train', 'generation': generation,
            'manifest_sha256': self._control_digest(directory/'manifest.json'), 'source_set_sha256': source_set,
            'schedule_sha256': digest(manifest['schedule']), 'bindings_sha256': digest(manifest['policy_bindings']),
            'original_correspondence': None, 'numeric': numeric, 'selected_roles': [], 'counts': {},
            'elapsed_seconds': elapsed}
        if self._packed:
            if validation:
                proof = facts
            else:
                assert update is not None
                proof = update['source_proof']
            from gf_common.group_pack_game import same_storage_custody
            if not same_storage_custody(proof['storage'], proof['custody'], packed_owner.storage, packed_owner.custody):
                raise ContractError('Dataset4 immutable storage evidence differs from actual owner')
            row.update(storage=packed_owner.storage, custody=packed_owner.custody)
        return row, manifest, update

    def selected_inputs(self) -> Iterator[tuple[str, GameSelectedInput]]:
        from gf_training.journal import _reward
        from gf_training.structured_rollout import (
            _episode_correspondence_digest,
            _owned_bytes,
            _source_correspondence,
        )
        self._live()
        if self._iterating:
            raise ContractError('game original worklist has exactly one pass')
        self._iterating = True
        try:
            for parent_index, parent in enumerate(self._parents):
                final_generation = self._states[parent_index]['updates']
                loan = self._loans[parent_index]
                for generation in range(final_generation+1):
                    validation = generation == final_generation
                    row, manifest, update = self._collection(parent_index, generation, validation=validation)
                    specs = [spec for group in manifest['schedule'] for spec in group['episodes']]
                    expected_games = 64 if validation else 16
                    if len(specs) != expected_games:
                        raise ContractError('complete original16/64 worklist required')
                    digests: list[str] = []
                    for ordinal, spec in enumerate(specs):
                        self.recheck(); self._sync()
                        borrow_scope = (self._sources[parent_index].borrow_original_selected_episode(ordinal, budget_loan=loan)
                            if validation else parent.borrow_original_training_episode(generation, ordinal, budget_loan=loan))
                        with borrow_scope as borrow:
                            self._sync()
                            viewer = manifest['linkage']['learner_seats'][spec['episode_id']]
                            policy = spec['policies'][viewer-1]
                            if (borrow.episode_id != spec['episode_id'] or borrow.viewer != viewer
                                    or borrow.end_kind != 'natural' or borrow.outcome is None):
                                raise ContractError('actual original own role/natural endpoint differs from C5 membership')
                            observed = borrow.records
                            episode_digest = _episode_correspondence_digest(observed)
                            del observed
                            if episode_digest != borrow.episode_digest:
                                raise ContractError('actual borrowed original record correspondence differs')
                            digests.append(episode_digest)
                            accepted = 0
                            for decision in borrow.iter_decisions():
                                borrow.recheck(); self.recheck()
                                # Conservative token/copy bound uses this ACTUAL
                                # original graph, never successful-run maxima or
                                # cached NN-window arrays. Native packet bytes
                                # themselves remain intrinsic P, counted once.
                                token_bound = 128*(_owned_bytes(decision)+65536)
                                self._budget.reserve_active(token_bound)
                                self._sync()
                                tokens,candidates = _tree_tokens(decision,self.encoder)
                                self._current_feature_bytes = _owned_bytes((tokens,candidates))
                                if self._current_feature_bytes > token_bound:
                                    raise ContractError('original Tree token graph exceeded pre-admitted copy bound')
                                sample = GameSelectedInput(tokens,candidates,decision.selection.index,decision.selection,
                                    float(_reward(borrow.outcome,viewer)),parent_index,generation,borrow.episode_id,
                                    viewer,accepted,policy)
                                try:
                                    yield row['split'],sample
                                finally:
                                    # Consumer must discard its row view before
                                    # closing/advancing this concrete raw borrow.
                                    del sample,tokens,candidates,decision
                                    self._current_feature_bytes = 0
                                    self._budget.reserve_active(0,retired=True)
                                    self._sync()
                                accepted += 1
                            if accepted != borrow.decision_count:
                                raise ContractError('actual original decision worklist omitted/duplicated an endpoint')
                            role = {'episode_id': borrow.episode_id, 'viewer': viewer, 'policy_label': policy,
                                'end_kind': borrow.end_kind, 'outcome': borrow.outcome, 'accepted_count': accepted}
                            # The outcome is taken from actual original custody,
                            # not authorized by metadata; C5 is a crosscheck only.
                            endpoint = next(value for value in manifest['episodes'] if value['episode_id'] == borrow.episode_id)
                            if endpoint['end_kind'] != role['end_kind'] or canonical(endpoint['outcome']) != canonical(role['outcome']):
                                raise ContractError('C5 endpoint differs from actual observed original closure')
                            row['selected_roles'].append(role)
                            borrow.recheck()
                            del borrow
                        self._sync(retired=True)
                    if validation:
                        row['original_correspondence'] = game.final_correspondence(row['manifest_sha256'],digests,packed=self._packed)
                    else:
                        assert update is not None
                        # Exact existing original Source control recipe. The
                        # result remains a correspondence FACT, no authority.
                        fingerprint_fields = self._training_fingerprint_fields(parent_index, generation, row)
                        fingerprint = digest(fingerprint_fields)
                        row['original_correspondence'] = _source_correspondence(fingerprint,self._neural['semantics'],tuple(digests),game=True,storage_binding=self._states[parent_index].get('storage_binding'))
                    roles = row['selected_roles']
                    row['counts'] = {'games':expected_games,'participants':2*expected_games,
                        'selected_decisions':sum(role['accepted_count'] for role in roles),
                        'zero_endpoint_games':sum(role['accepted_count']==0 for role in roles)}
                    self._persistent += 8*_owned_bytes(row)+65536
                    self._reserve_controls(self._persistent+self._read_peak+self._scan.retained)
                    self._collections.append(row)
                    del manifest,row,update
            self._manifest = self._content()
            self._complete = True
            self.recheck()
        except BaseException as error:
            self._failure = error
            raise

    def _training_fingerprint_fields(self, index: int, generation: int, row: Json) -> Json:
        from gf_training.card_instance_source import recorded_plan
        from gf_training.lineage import checkpoint_path, generation_path
        plan, _, _ = recorded_plan(self._custody.parents[index], self._states[index],
            self._records[index], generation, self.deadline)
        root, state = self._custody.parents[index], self._states[index]
        fields = {'run_id': state['run_id'], 'config_sha256': state['config_sha256'],
            'identity': state['identity'], 'execution': state['execution'], 'plan': plan.linkage,
            'files': {str(generation_path(root, generation).relative_to(root)):
                self._parent_rows[index]['generations'][generation]['generation_sha256'],
                str((checkpoint_path(root, generation)/'inference.json').relative_to(root)):
                self._records[index][generation]['inference_sha256']},
            'source_set_sha256': row['source_set_sha256']}
        if self._packed:
            fields.update(recipe='card-instance-packed-source-controls-v1', storage=row['storage'], custody=row['custody'])
        return fields

    def historical_correspondences(self, saved: Json) -> tuple[Json, ...]:
        """Comparison facts only, recomputed from original controls and actual episodes."""
        from gf_common.group_pack_game import same_storage_custody
        from gf_training.structured_rollout import _source_correspondence
        values = []
        if len(saved['collections']) != len(self._collections):
            raise ContractError('saved Dataset4 complete membership differs')
        for expected, actual in zip(saved['collections'], self._collections, strict=True):
            if not same_storage_custody(expected['storage'], expected['custody'], actual['storage'], actual['custody']):
                raise ContractError('saved Dataset4 original custody differs')
            if actual['split'] == 'validation':
                values.append(actual['original_correspondence'])
            else:
                index, generation = actual['parent_index'], actual['generation']
                # The saved fingerprint binds its own validated custody recipe.
                # No historical receipt is emitted; fresh rows always retain custody2.
                fields = self._training_fingerprint_fields(index, generation, actual)
                fields.update(storage=expected['storage'], custody=expected['custody'])
                values.append(_source_correspondence(digest(fields), self._neural['semantics'],
                    tuple(actual['original_correspondence']['episode_digests']), game=True,
                    storage_binding=self._states[index]['storage_binding']))
        return tuple(values)

    def _content(self) -> Json:
        totals = {split:{key:sum(row['counts'][key] for row in self._collections if row['split']==split)
            for key in game.COUNT_FIELDS} for split in ('train','validation')}
        layout = feature_layout_v3(self.encoder)
        value = {'schema_version':4 if self._packed else 3,'kind':game.KIND,'selection':game.SELECTION,'label_recipe':game.LABEL,
            'source_kind':game.PACKED_SOURCE_KIND if self._packed else game.SOURCE_KIND,
            'source_origin':game.packed_origin() if self._packed else game.origin(),'input_binding':game.input_binding(),
            'input_recipe':LAYOUT.input_recipe,'tree_projection':{'encoder':asdict(self.encoder),'feature_layout':layout,
                'feature_signature':layout['feature_signature'],'token_signature':signature(self.encoder.aggregate_window,layout=LAYOUT)},
            'neural_cohort':self._neural,'parents':self._parent_rows,'collections':self._collections,'counts':totals,
            'source_set_sha256':game.source_identity(self._parent_rows,self._collections),
            'shared_generation':game.shared_generation(self._parent_rows,self._collections)}
        if self._packed:
            value['storage_binding'] = self._states[0]['storage_binding']
        return (game.packed_manifest if self._packed else game.manifest)({**value,'content_sha256':digest(value)})

    def verify_complete(self) -> Json:
        self.recheck()
        if not self._complete:
            raise ContractError('complete actual original game worklist required')
        return self._content()

    def finalize(self) -> None:
        self.verify_complete()

    def close(self) -> None:
        if self._closed:
            return
        failure = self._failure
        if self._sink or self._current_feature_bytes:
            error = ContractError('actual controller rows/current tokens must retire BEFORE original Source close')
            self._failure = self._failure or error
            vars(self._failure)['_game_tree_dataset_cleanup_owner'] = self
            raise self._failure
        if self._active and failure is None:
            try:
                self.recheck()
            except BaseException as error:  # noqa: BLE001 - terminal guards fail closure
                failure = error
        self._active = False
        for index in reversed(range(len(self._source_scopes))):
            if index in self._retired_sources:
                continue
            self._retired_sources.add(index)  # never repeat a consumed context exit
            try:
                self._source_scopes[index].__exit__(type(failure) if failure else None,failure,None)
            except BaseException as error:  # noqa: BLE001 - preserve primary and actual uncertain owner
                if failure is None: failure = error
                else: _note(failure,'original final Source close: ',error)
        if any(not source.cleanup_complete for source in self._sources):
            failure = failure or ContractError('actual final Source retirement unconfirmed')
            self._failure = failure
            vars(failure)['_game_tree_dataset_cleanup_owner'] = self
            raise failure
        # Actual Sources must retire first. A loan created before Source entry
        # can still be unbound: the actual Parent owns that opening retirement.
        # Its exit retires the loan before native inputs; do not erase B via0.
        for index in reversed(range(len(self._parents))):
            if index in self._retired_parents:
                continue
            self._retired_parents.add(index)
            try:
                self._parents[index].__exit__(type(failure) if failure else None,failure,None)
            except BaseException as error:  # noqa: BLE001 - never infer close success from empty retry
                if failure is None: failure = error
                else: _note(failure,'original formal parent close: ',error)
        if any(not parent.cleanup_complete for parent in self._parents) or any(not loan.cleanup_complete for loan in self._loans):
            failure = failure or ContractError('actual formal parent/loan retirement unconfirmed')
            self._failure = failure
            vars(failure)['_game_tree_dataset_cleanup_owner'] = self
            raise failure
        try:
            self._scan.finalize()
        except BaseException as error:  # noqa: BLE001 - a guard failure does not consume lock cleanup
            if failure is None: failure = error
            else: _note(failure,'game original metadata final guard: ',error)
        try:
            self._custody.__exit__(type(failure) if failure else None,failure,None)
        except BaseException as error:  # noqa: BLE001 - all locks/reportable guards precede healthy publication
            if failure is None: failure = error
            else: _note(failure,'game original custody close: ',error)
        self._closed = self._custody.cleanup_complete
        self._failure = failure
        if failure is not None:
            if not self.cleanup_complete:
                vars(failure)['_game_tree_dataset_cleanup_owner'] = self
            raise failure

    def release(self) -> None:
        if not self.cleanup_complete:
            raise ContractError('confirmed reportable game-original retirement required before memory release')
        self._states.clear(); self._records.clear(); self._parent_rows.clear(); self._collections.clear()
        self._manifest.clear(); self._neural.clear(); self._scan.files.clear(); self._scan.inventories.clear()

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        original = value if isinstance(value,BaseException) else None
        self._failure = self._failure or original
        try:
            self.close()
        except BaseException as error:
            if original is None:
                raise
            vars(original)['_game_tree_dataset_cleanup_owner'] = self
            if error is not original:
                _note(original,'game originals close: ',error)


class VerifiedGameTreeDataset3:
    """Concrete real Dataset3/index plus SAME three formal and final Source owners."""
    _packed = False

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith('_') and getattr(self,'_entered',False):
            raise AttributeError('verified game Dataset configuration is immutable after entry')
        object.__setattr__(self,name,value)

    def __init__(self, path: Path, expected_digest: str, settings: PreparationConfig, *,
                 max_owned_bytes: int = MAX_OWNED_BYTES, deadline: float,
                 child_limit_bytes: int = 0, ownership: GameSourceCustody | None = None,
                 _inspect_recorded: bool = False) -> None:
        self._packed = type(self) is VerifiedGameTreeDataset4
        self.path = Path(path).absolute(); self.expected_digest = game.sha256(expected_digest)
        self.settings = _settings(settings,max_owned_bytes,deadline)
        _reader_preflight(self.settings, inspect_recorded=_inspect_recorded)
        self._inspect_recorded = _inspect_recorded
        self.limit = max_owned_bytes
        self.deadline = min(deadline,time.monotonic()+self.settings.wall_seconds-self.settings.cleanup_seconds)
        self._child = integer(child_limit_bytes,0,max_owned_bytes)
        self._ownership = ownership
        self._index = SealedScan(max_owned_bytes=max_owned_bytes-self._child,deadline=self.deadline)
        self._metadata: Json = {}
        self._verified_rows: Json = {}
        self._source: _GameTreeOriginals | None = None
        self._ancestors: dict[Path,tuple[int,int]] = {}
        self._active = self._entered = self._closed = self._complete = self._finalized = False
        self._failure: BaseException | None = None

    def _live(self) -> _GameTreeOriginals:
        if self._failure is not None or not self._active or self._closed or self._source is None:
            raise ContractError('exact active VerifiedGameTreeDataset3 required') from self._failure
        return self._source

    @property
    def manifest(self) -> Json:
        self._live()
        return clone(self._metadata)

    @property
    def source(self) -> _GameTreeOriginals:
        return self._live()

    def __enter__(self) -> Self:
        from gf_training.card_instance_tree_fixture import _ancestors
        if self._entered or self._closed:
            raise ContractError('verified game Dataset has exactly one lifetime')
        self._entered = True
        try:
            if self.path.name != 'manifest.json':
                raise ContractError('game Dataset index requires canonical manifest.json')
            self._ancestors = _ancestors(self.path.parent)
            self._index.inventory(self.path.parent,{'manifest.json','preparation.json'})
            for path,expected in ((self.path,self.expected_digest),(self.path.parent/'preparation.json',None)):
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= min(64,self.settings.max_source_file_mib)*1024**2:
                    raise ContractError('game Dataset management index bounded64MiB regular file required')
                # Includes parsed index, pure-validator clone and external full
                # child BEFORE discovering paths or constructing original owners.
                self._index.charge(32*info.st_size+65536)
                value = self._index.object(path,expected)
                if path == self.path:
                    self._metadata = (game.packed_manifest if self._packed else game.manifest)(value)
                else:
                    closed(value,{'schema_version','kind','dataset_content_sha256','seconds','scope'})
                    if (type(value['schema_version']) is not int or value['schema_version']!=1
                            or value['kind']!='card-instance-game-dataset-preparation'
                            or value['dataset_content_sha256']!=self._metadata['content_sha256']
                            or value['scope']!='actual original inspection and selected Tree projection; excludes shared PPO/final-validation generation and fitting'):
                        raise ContractError('game Dataset original preparation receipt differs')
                    game.seconds(value['seconds'])
            raw = self._metadata['tree_projection']['encoder']
            encoder = TreeEncoderConfigV3(**closed(raw,set(TreeEncoderConfigV3.__dataclass_fields__)))
            projection = self._metadata['tree_projection']; layout = feature_layout_v3(_encoder(encoder))
            if (canonical(projection['feature_layout'])!=canonical(layout)
                    or projection['feature_signature']!=layout['feature_signature']
                    or projection['token_signature']!=signature(encoder.aggregate_window,layout=LAYOUT)):
                raise ContractError('exact original Tree window/feature projection required')
            parents = tuple(Path(row['directory']) for row in self._metadata['parents'])
            validations = tuple(Path(row['directory']) for row in self._metadata['collections'] if row['split']=='validation')
            self._source = _GameTreeOriginals(parents,validations,self.settings,encoder=encoder,
                max_owned_bytes=self.limit,deadline=self.deadline,extra_owned_bytes=self._index.retained,
                child_limit_bytes=self._child,ownership=self._ownership,packed=self._packed,
                _inspect_recorded=self._inspect_recorded)
            self._source.__enter__()
            self._active = True
            if canonical(self._source._parent_rows)!=canonical(self._metadata['parents']):
                raise ContractError('exact SAME three actual formal parent/generation memberships differ from index')
            self._verified_rows = _verified_fit_rows(self._metadata,self._source.verified_row_counts())
            self._index.finalize(); _check_output_ancestors(self._ancestors)
            return self
        except BaseException as error:
            self._failure = error
            if self._active:
                try:
                    self.close()
                except BaseException as cleanup:  # noqa: BLE001 - preserve primary and actual original owner
                    _note(error,'verified game index opening retirement: ',cleanup)
            if self._source is not None and not self._source.cleanup_complete:
                vars(error)['_game_tree_dataset_cleanup_owner'] = self
            raise

    def selected_inputs(self) -> Iterator[tuple[str,GameSelectedInput]]:
        source = self._live()
        if self._complete:
            raise ContractError('verified original game worklist has exactly one pass')
        yield from source.selected_inputs()
        self._complete = True
        self.verify_complete()

    def reserve_sink(self, amount: int) -> None:
        self._live().reserve_sink(amount)

    def _retire_fit_sink(self, prepared: Any) -> None:
        from gf_models.structured_tree_pairing import TREE3_GAME_RECORDS, TREE3_PACKED_GAME_RECORDS
        from gf_training.structured_tree_fit import _PreparedTreeFit
        if (type(prepared) is not _PreparedTreeFit or prepared.dataset is not self
                or prepared.context is not (TREE3_PACKED_GAME_RECORDS if self._packed else TREE3_GAME_RECORDS) or self._source is None
                or any(prepared.blocks.values()) or any(prepared.labels.values()) or prepared._packing_arrays
                or any(reference() is not None for reference in prepared._row_refs)):
            raise ContractError('actual same prepared row owner and physical retirement required before sink release')
        self._source._retire_sink()

    def reserve_fit(self, amount: int) -> None:
        self._live().reserve_fit(amount)

    def fit_memory(self) -> Json:
        return self._live().fit_memory()

    def retained_without_sink(self) -> int:
        return self._live().retained_without_sink()

    def verify_complete(self) -> None:
        source = self._live()
        from gf_common.group_pack_game import same_packed_dataset
        actual = source.verify_complete()
        if not self._complete or not (same_packed_dataset(self._metadata, actual,
                historical_correspondences=source.historical_correspondences(self._metadata)) if self._packed
                else canonical(actual) == canonical(self._metadata)):
            raise ContractError('actual original game Dataset membership/roles/labels/costs differ from index')
        self._index.finalize(); _check_output_ancestors(self._ancestors)

    def finalize(self) -> None:
        self.verify_complete()
        self._finalized = True

    @property
    def cleanup_complete(self) -> bool:
        return self._closed and (self._source is None or self._source.cleanup_complete)

    def close(self) -> None:
        if self._closed:
            return
        failure = self._failure
        if failure is None and self._active:
            try:
                self.finalize()
            except BaseException as error:  # noqa: BLE001 - source mutation prevents healthy closure
                failure = error
        if self._source is not None:
            try:
                self._source.__exit__(type(failure) if failure else None,failure,None)
            except BaseException as error:  # noqa: BLE001 - preserve actual remaining owners
                if failure is None: failure = error
                else: _note(failure,'verified game originals close: ',error)
        try:
            self._index.finalize(); _check_output_ancestors(self._ancestors)
        except BaseException as error:  # noqa: BLE001 - reportable index/ancestor guards before publisher
            if failure is None: failure = error
            else: _note(failure,'verified game index close: ',error)
        self._closed = self._source is None or self._source.cleanup_complete
        if self._closed:
            self._active = False
        self._failure = failure
        if failure is not None:
            if not self._closed:
                vars(failure)['_game_tree_dataset_cleanup_owner'] = self
            raise failure

    def release(self) -> None:
        if not self._closed:
            raise ContractError('actual original game Dataset reportable close must precede memory release')
        if self._source is not None:
            self._source.release()
        self._metadata.clear(); self._verified_rows.clear(); self._index.files.clear(); self._index.inventories.clear()

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        original = value if isinstance(value,BaseException) else None
        self._failure = self._failure or original
        try:
            self.close()
        except BaseException as error:
            if original is None:
                raise
            vars(original)['_game_tree_dataset_cleanup_owner'] = self
            if error is not original:
                _note(original,'verified game Dataset close: ',error)
        if self._closed:
            self.release()



class VerifiedGameTreeDataset4(VerifiedGameTreeDataset3):
    """Exact Dataset4 subtype retaining complete packed formal/public originals."""


def verified_game_dataset_type(path: Path, expected_digest: str, settings: PreparationConfig, *, max_owned_bytes: int,
                               deadline: float, _inspect_recorded: bool = False) -> type[VerifiedGameTreeDataset3]:
    """Select a closed schema only. The returned class grants no Source rights."""
    entry = time.monotonic()
    _settings(settings,max_owned_bytes,deadline)
    deadline = min(deadline,entry+settings.wall_seconds-settings.cleanup_seconds)
    _reader_preflight(settings, inspect_recorded=_inspect_recorded)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= min(64,settings.max_source_file_mib)*1024**2:
        raise ContractError('game Dataset schema metadata requires the original bounded regular source file')
    scan = SealedScan(max_owned_bytes=max_owned_bytes, deadline=deadline)
    raw = scan.object(path, game.sha256(expected_digest))
    schema = raw.get('schema_version')
    if type(schema) is not int or schema not in (3, 4):
        raise ContractError('explicit game Dataset3 or Dataset4 required')
    scan.finalize()
    return VerifiedGameTreeDataset4 if schema == 4 else VerifiedGameTreeDataset3


def _packed_parent_variant(parents: tuple[Path, ...], *, max_owned_bytes: int, deadline: float) -> bool:
    from gf_common.card_instance_formal import PACKED_RUN_VARIANT, RUN_VARIANT
    from gf_common.group_pack_game import packed_state
    if len(parents) != 3:
        raise ContractError('exact three formal parents required for schema selection')
    scan = SealedScan(max_owned_bytes=max_owned_bytes, deadline=deadline)
    modes = set()
    for directory in parents:
        raw = scan.object(directory/'run.json')
        mode = packed_state(raw)
        if type(raw.get('schema_version')) is not int or raw['schema_version'] != (4 if mode else 3) or raw.get('run_variant') != (PACKED_RUN_VARIANT if mode else RUN_VARIANT):
            raise ContractError('ordinary/fixture/short parent has no original formal declaration')
        modes.add(mode)
    if len(modes) != 1:
        raise ContractError('Dataset mixed storage parents are unsupported')
    scan.finalize()
    return modes.pop()


def build_game_tree_dataset(parents: tuple[Path,...], validations: tuple[Path,...], output: Path,
                            settings: PreparationConfig, *, encoder: TreeEncoderConfigV3,
                            max_owned_bytes: int = MAX_OWNED_BYTES, deadline: float,
                            ownership: GameSourceCustody | None = None) -> Json:
    """Build a recipe index by consuming SAME actual originals; no arrays cached."""
    from gf_training.structured_lineage import _ChainScan, _write
    started = time.monotonic()
    _settings(settings,max_owned_bytes,deadline)
    deadline = min(deadline,started+settings.wall_seconds-settings.cleanup_seconds)
    _reader_preflight(settings)
    output,ancestors = _outside_output(output,[*parents,*validations])
    packed = _packed_parent_variant(parents,max_owned_bytes=max_owned_bytes,deadline=deadline)
    with _GameTreeOriginals(parents,validations,settings,encoder=encoder,max_owned_bytes=max_owned_bytes,
                           deadline=deadline,ownership=ownership,packed=packed) as source:
        for _,sample in source.selected_inputs():
            del sample
        manifest = source.verify_complete()
        receipt = {'schema_version':1,'kind':'card-instance-game-dataset-preparation',
            'dataset_content_sha256':manifest['content_sha256'],'seconds':time.monotonic()-started,
            'scope':'actual original inspection and selected Tree projection; excludes shared PPO/final-validation generation and fitting'}
        payload = len(canonical(manifest).encode())+len(canonical(receipt).encode())+2
        if payload>settings.scratch_budget_mib*1024**2 or any(len(canonical(value).encode())+1>min(64,settings.max_source_file_mib)*1024**2 for value in (manifest,receipt)):
            raise ContractError('whole original game index publication scratch/file allowance exceeded')
        source.reserve_fit(32*payload+131072)
        source.finalize(); source.close()
        # Complete actual Source/native/parent/lock retirement precedes ANY
        # published index. The index itself supplies no fitting permission.
        _check_output_ancestors(ancestors)
        output.mkdir(mode=0o700,exist_ok=False)
        publisher = _ChainScan(directory=output,max_owned_bytes=max_owned_bytes,deadline=deadline)
        _write(output/'preparation.json',receipt,scan=publisher,immutable=True)
        _write(output/'manifest.json',manifest,scan=publisher,immutable=True)
        publisher.finalize(); _check_output_ancestors(ancestors)
        source.release()
        return manifest


def inspect_game_tree_dataset(path: Path, *, expected_digest: str, settings: PreparationConfig,
                              max_owned_bytes: int = MAX_OWNED_BYTES, deadline: float,
                              ownership: GameSourceCustody | None = None) -> Json:
    entry = time.monotonic()
    _settings(settings,max_owned_bytes,deadline)
    deadline = min(deadline,entry+settings.wall_seconds-settings.cleanup_seconds)
    dataset_type = verified_game_dataset_type(path, expected_digest, settings,
        max_owned_bytes=max_owned_bytes, deadline=deadline, _inspect_recorded=True)
    with dataset_type(path,expected_digest,settings,max_owned_bytes=max_owned_bytes,
                                 deadline=deadline,ownership=ownership,_inspect_recorded=True) as dataset:
        for _,sample in dataset.selected_inputs():
            del sample
        return dataset.manifest
