"""SAME fixture-owner native token/persistence parity, never fitting authority.

Returned evidence remains guarded by the live Source. The acceptance caller owns
ordinary-worklist completion and every reportable close before a healthy report.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import signal
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import numpy as np

from gf_common.card_instance_fixture import origin
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.structured import event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import (
    CandidateTokens,
    FrameTokens,
    signature,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from gf_models.card_instance_tree_scoring import COMPONENT, ScoringResult
from gf_models.sharded_arrays import _pairs, read_bundle, validate_bundle_metadata, write_bundle
from gf_models.structured_sealed import SealedScan, checksum
from gf_models.structured_tree_artifacts import (
    LoadedTreeV3,
    _bind_model_v3,
    _bind_origin,
    _validate_booster,
)
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV3,
    resource_estimate,
)
from gf_models.structured_tree_encoder import PreparedTreeFrameV3, _owned
from gf_models.structured_tree_pairing import TREE3_RECORDS, PairedTraining
from gf_models.structured_tree_policy import _score_chunks
from gf_simulator_client.card_instance_tree_scoring_worker import CardInstanceTreeScoringWorker
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training._raw_token_arrays import raw_token_tuple
from gf_training.card_instance_tree_fixture import (
    FixtureSelectedInput,
    TreeFixtureSource,
    _FixtureEpisodeBorrow,
)
from gf_training.native_backend import binary_identity, load_native, require_native_defense_priority
from gf_training.native_execution import NativeExecution
from gf_training.structured_rollout import CANDIDATE_FIELDS, EVENT_FIELDS, FRAME_FIELDS
from gf_training.structured_tree_data import _check_output_ancestors, _outside_output

LIMIT = 64*1024**2
FIXTURE_CONTRACT = 'component-envelope1;participant-structured-v2;codec2'
METADATA = {'schema_version','kind','envelope_sha256','fixture_id','declaration_sha256','population_id','cohort_id',
    'episode','viewer','split','generation','roles','selected_policy_id','outcome','outcome_origin','label_recipe','label',
    'input_binding','layout_signature','aggregate_window','token_signature','boundary','decisions','arrays'}
DECISION = {'decision','revision','interval_start','interval_start_hash','boundary','frame_hash','table_hash',
    'columns_file','columns_sha256','selected_index','token_prefix','event_token_prefix'}
CONTAINER = {'schema_version','kind','source_manifest_sha256','native_binary','metadata','bundle'}


def _json(raw: bytes) -> Json:
    def nonfinite(_: str) -> Any:
        raise ContractError('nonfinite diagnostic JSON')
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=nonfinite)
    if type(value) is not dict:
        raise ContractError('closed diagnostic object required')
    return value


def _preflight(module: Any, *, diagnostic: bool) -> Json:
    info = dict(module.build_info())
    marker = info.get('card_instance_fixture_tokens')
    if marker is not None and marker != FIXTURE_CONTRACT:
        raise ContractError('native fixture-token capability profile differs')
    if not diagnostic and marker != FIXTURE_CONTRACT:
        raise ContractError('native binary lacks accepted component fixture-token capability')
    if info.get('card_instance_signature') != signature(128, layout=LAYOUT) or info.get('abi') != 'cp314-gil':
        raise ContractError('native fixture signature/ABI differs')
    require_native_defense_priority(module)
    checksum(info.get('native_source_sha256'))
    cls = getattr(module, 'CardInstanceFixtureTokens', None)
    if (not callable(cls) or any(not callable(getattr(cls, key, None)) for key in
            ('export_tokens','used_bytes','peak_bytes','control_reserved_bytes','cancel_handle','close'))
            or not callable(getattr(module, 'CancelHandle', None))
            or not callable(getattr(module.CancelHandle, 'cancel', None))):
        raise ContractError('native fixture class/cancel ABI incomplete')
    identity = binary_identity(module)
    if identity['build'] != info or identity['gil_disabled'] is not False:
        raise ContractError('native fixture loaded binary identity differs')
    return identity


def _paired_predictor_binding(tree: LoadedTreeV3, pair: PairedTraining | None) -> None:
    """Pure SAME-pair identity checks; runtime execution overrides stay separate."""
    if type(tree) is not LoadedTreeV3 or type(pair) is not PairedTraining or tree.training_binding != pair.binding:
        raise ContractError('actual paired component Tree3 CPU predictor required')
    _bind_origin(tree.booster, tree.origin)
    _bind_model_v3(tree.booster, tree.model, tree.origin)
    fields = tree.origin.fields()
    TREE3_RECORDS.source(fields, toy=False)
    layout = TREE3_RECORDS.layout(tree.model.encoder)
    expected = {**fields, 'tree_descriptor':asdict(tree.descriptor),
        'model':asdict(tree.model.architecture()),
        'architecture_identity':TREE3_RECORDS.identity(tree.model.architecture()), 'feature_layout':layout,
        'feature_recipe':layout['recipe'], **{key:layout[key] for key in
            ('feature_signature','feature_names','categorical_indices','token_signature','input_recipe','rule_signature','layout_signature')}}
    if tree.origin.fit_origin != 'trained' or tree.origin.provenance is None or any(
            canonical(pair.inference.get(key)) != canonical(value) for key,value in expected.items()):
        raise ContractError('actual Tree3 predictor origin/provenance/architecture/features differ from SAME paired evidence')


def _semantics(metadata: Json) -> bytes:
    closed(metadata, METADATA)
    if type(metadata['schema_version']) is not int or metadata['schema_version'] != 1 or metadata['kind'] != 'card-instance-component-token-payload':
        raise ContractError('native fixture token payload kind/schema differs')
    if type(metadata['decisions']) is not list or type(metadata['arrays']) is not list:
        raise ContractError('ordered fixture decisions/arrays required')
    value = {key:item for key,item in metadata.items() if key not in ('schema_version','kind','arrays','decisions')}
    value['decisions'] = []
    for i, frame in enumerate(metadata['decisions']):
        closed(frame, DECISION)
        if type(frame['decision']) is not int or frame['decision'] != i:
            raise ContractError('ordered diagnostic decision identity differs')
        if frame['token_prefix'] != f'e.0.f.{i}' or frame['event_token_prefix'] != f'e.0.event-tokens.{i}':
            raise ContractError('diagnostic storage prefix differs')
        value['decisions'].append({key:item for key,item in frame.items() if key not in ('token_prefix','event_token_prefix')})
    return canonical(value).encode()


def _arrays(metadata: Json, banks: tuple[Any, Any, Any]) -> dict[str, Any]:
    groups = dict(zip(('uint8','int64','float32'), banks, strict=True))
    for dtype, values in groups.items():
        if type(values) is not tuple:
            raise ContractError('closed readonly diagnostic bank tuple required')
        for value in values:
            if (type(value) is not np.ndarray or str(value.dtype) != dtype or value.ndim != 1 or value.flags.writeable
                    or not value.flags.c_contiguous or value.nbytes > LIMIT):
                raise ContractError('diagnostic bank dtype/readonly/size differs')
    expected: set[str] = set()
    for frame in metadata['decisions']:
        expected.update(frame['token_prefix']+'.'+name for name in FRAME_FIELDS)
        expected.update(frame['event_token_prefix']+'.'+name for name in EVENT_FIELDS)
        expected.update(frame['token_prefix']+'.candidate.'+name for name in CANDIDATE_FIELDS)
    arrays: dict[str, Any] = {}; ranges: dict[tuple[str,int], list[tuple[int,int]]] = {}
    for item in metadata['arrays']:
        closed(item, {'key','dtype','shape','bank_index','element_offset','element_count'})
        key, dtype, shape = item['key'], item['dtype'], item['shape']
        if type(key) is not str or key not in expected or key in arrays or dtype not in ('uint8','int64'):
            raise ContractError('diagnostic unique typed token key required')
        expected_dtype = 'uint8' if 'present' in key or 'mask' in key else 'int64'
        if dtype != expected_dtype or type(shape) is not list or not 1 <= len(shape) <= 2:
            raise ContractError('diagnostic token dtype/shape differs')
        for size in shape:
            integer(size, 0, LIMIT)
        bank_index = integer(item['bank_index'], 0, len(groups[dtype])-1)
        bank = groups[dtype][bank_index]
        start = integer(item['element_offset'], 0, len(bank))
        count = integer(item['element_count'], 0, len(bank)-start)
        if count != math.prod(shape) or count*np.dtype(dtype).itemsize > LIMIT:
            raise ContractError('diagnostic token array bounds differ')
        ranges.setdefault((dtype, bank_index), []).append((start,start+count))
        arrays[key] = bank[start:start+count].reshape(shape)
    if set(arrays) != expected:
        raise ContractError('diagnostic complete raw token arrays required')
    for dtype, values in groups.items():
        for index, value in enumerate(values):
            if (dtype,index) not in ranges:
                raise ContractError('diagnostic bank has no descriptor binding')
            end = 0
            for first, stop in sorted(ranges.get((dtype,index), [])):
                if first != end:
                    raise ContractError('diagnostic bank gaps/overlap differ')
                end = stop
            if end != len(value):
                raise ContractError('diagnostic bank contains unbound bytes')
    return arrays


def _tokens(metadata: Json, frame: Json, arrays: dict[str, Any]) -> tuple[FrameTokens, CandidateTokens]:
    prefix, event = frame['token_prefix'], frame['event_token_prefix']
    fields = {name:raw_token_tuple(arrays[prefix+'.'+name], boolean='present' in name or 'mask' in name) for name in FRAME_FIELDS}
    fields.update({name:raw_token_tuple(arrays[event+'.'+name], boolean='present' in name) for name in EVENT_FIELDS})
    tokens = FrameTokens(frame['frame_hash'], metadata['token_signature'], metadata['aggregate_window'], metadata['viewer'],
        metadata['episode'], frame['revision'], frame['interval_start'], frame['boundary']['end'], **fields)
    candidates = CandidateTokens(frame['table_hash'], frame['frame_hash'], metadata['token_signature'],
        **{name:raw_token_tuple(arrays[prefix+'.candidate.'+name], boolean=name == 'present') for name in CANDIDATE_FIELDS})
    validate_frame_tokens(tokens, layout=LAYOUT)
    validate_candidate_tokens(candidates, tokens, 18, layout=LAYOUT)
    return tokens, candidates


class _NativeEpisode:
    """Only native work ownership; original/fit authority stays with Source."""
    def __init__(self, module: Any, deadline: float) -> None:
        self.cancel = module.CancelHandle(deadline)
        self.executor = NativeExecution(self.cancel, thread_name_prefix='gf-fixture-tokens')
        self.owner: Any = None

    def construct(self, module: Any, borrow: _FixtureEpisodeBorrow, settings: bytes, deadline: float, window: int) -> None:
        envelope, columns = borrow.envelope, borrow.columns
        def create() -> None:
            try:
                self.owner = module.CardInstanceFixtureTokens(envelope, columns, settings, deadline, self.cancel,
                    aggregate_window=window)
            except BaseException as error:
                self.owner = getattr(error, '_native_fixture_cleanup_owner', None)
                raise
        self.executor.call(create)

    def close(self) -> None:
        if self.owner is not None:
            self.executor.call(self.owner.close)
            self.owner = None
        self.executor.close()


def _cleanup(native: _NativeEpisode, error: BaseException) -> None:
    try:
        native.close()
    except BaseException as cleanup:  # noqa: BLE001 - retain initiating cancellation and cleanup owner
        error.add_note('fixture native cleanup: '+str(cleanup))
        vars(error)['_native_fixture_adapter_cleanup_owner'] = native


class FixtureTokenDiagnostics:
    """Guarded structural results only; outer caller owns acceptance commit."""
    def __init__(self, source: TreeFixtureSource, binary: SealedScan, persisted: SealedScan,
                 ancestors: dict[Path, tuple[int,int]], report: Json, pair: PairedTraining | None) -> None:
        self._source, self._binary, self._persisted, self._ancestors = source, binary, persisted, ancestors
        self._report, self._pair = canonical(report).encode(), pair
        self._output = Path(report['output'])
        self._membership = tuple(f'episode-{i:06}' for i in range(report['episodes']))
        self._closed = self._failed = False

    def guard(self) -> None:
        if self._closed or self._failed:
            raise ContractError('active fixture token diagnostic result required')
        try:
            self._source.guard(); self._binary.finalize(); self._persisted.finalize()
            _check_output_ancestors(self._ancestors)
            if tuple(sorted(path.name for path in self._output.iterdir())) != self._membership:
                raise ContractError('diagnostic output root membership differs')
            if self._pair is not None:
                self._pair.finalize()
        except BaseException:
            self._failed = True
            raise

    @property
    def report(self) -> Json:
        self.guard(); return _json(self._report)

    def close(self) -> None:
        if self._closed:
            return
        self.guard()
        self._closed = True
        self._binary.files.clear(); self._binary.inventories.clear()
        self._persisted.files.clear(); self._persisted.inventories.clear()


def _reference_arrays(sample: FixtureSelectedInput, frame: Json) -> dict[str, tuple[Any, ...]]:
    result = {frame['token_prefix']+'.'+name:getattr(sample.tokens, name) for name in FRAME_FIELDS}
    result.update({frame['event_token_prefix']+'.'+name:getattr(sample.tokens, name) for name in EVENT_FIELDS})
    result.update({frame['token_prefix']+'.candidate.'+name:getattr(sample.candidates, name) for name in CANDIDATE_FIELDS})
    return result


class _FacadeLoop:
    """One owned event loop; all framing/process ownership stays in the facade."""
    def __init__(self, source: TreeFixtureSource | None = None, reserved_bytes: int = 0) -> None:
        self.loop = asyncio.new_event_loop()
        # The Source-lifetime sink survives failed borrow exit. An uncertain
        # worker/reap retains this SAME Source and its unchanged full allowance.
        self.source, self.reserved_bytes = source, reserved_bytes
        self.worker: CardInstanceTreeScoringWorker | None = None
        self.workers_closed = 0
        self.closed_health: list[Json] = []

    def run(self, operation: Any) -> Any:
        task = self.loop.create_task(operation)
        try:
            return self.loop.run_until_complete(task)
        except BaseException as original:
            if not task.done():
                task.cancel()
                try:
                    self.loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                except BaseException as cleanup:  # noqa: BLE001 - join before preserving initiating failure
                    original.add_note('fixture scorer operation join: '+str(cleanup))
            raise

    def initialize(self, pair: PairedTraining, model: ConfiguredTreeModelV3,
                   borrow: _FixtureEpisodeBorrow, metadata: Json, ledger: ReservationLedger,
                   deadline: float) -> None:
        if self.worker is not None:
            raise ContractError('one active component scorer per episode required')
        try:
            self.worker = CardInstanceTreeScoringWorker(Path(pair.binding.directory), model, borrow.actions,
                0, metadata['viewer'], metadata['episode'], input_scope=COMPONENT, ledger=ledger,
                deadline=deadline, training_binding=pair.binding)
            self.run(self.worker.initialize())
        except BaseException as error:
            self.worker = getattr(error, '_tree_scoring_startup_owner', self.worker)
            raise

    def score(self, reference: Any) -> ScoringResult:
        if self.worker is None:
            raise ContractError('initialized component scorer required')
        frame = reference.frame
        delta = event_delta(frame.actor, frame.episode, frame.interval_start, frame.interval_start_hash,
            reference.bank_view.events(frame.interval_start), layout=LAYOUT)
        result = self.run(self.worker.score_original(frame, delta, reference.table))
        if (type(result) is not ScoringResult or result.frame_hash != frame.frame_hash
                or result.table_hash != reference.table.table_hash or result.generated is not None
                or type(result.scores) is not tuple or len(result.scores) != len(reference.table.kinds)
                or any(type(value) is not float or not math.isfinite(value) for value in result.scores)):
            raise ContractError('component facade result crossed full original table/frame/order')
        return result

    def close_worker(self) -> None:
        if self.worker is not None:
            self.run(self.worker.close())
            health = self.worker.health
            closed(health, {'returncode'}, {'exit_code','stderr_bytes'})
            allowed = (0, -signal.SIGTERM)
            if type(health['returncode']) is not int or health['returncode'] not in allowed:
                raise ContractError('component scorer close has no confirmed healthy returncode')
            if 'exit_code' in health and (type(health['exit_code']) is not int or health['exit_code'] not in allowed
                    or health['exit_code'] != health['returncode']):
                raise ContractError('component scorer reported unhealthy exit code')
            if 'stderr_bytes' in health:
                integer(health['stderr_bytes'],0)
            self.closed_health.append(dict(health))
            self.worker = None
            self.workers_closed += 1

    def close(self) -> None:
        if self.loop.is_closed():
            return
        self.close_worker()
        self.run(self.loop.shutdown_asyncgens())
        self.run(self.loop.shutdown_default_executor())
        self.loop.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if isinstance(value, BaseException):
                value.add_note('fixture scorer close/reap: '+str(cleanup))
                vars(value)['_fixture_scoring_cleanup_owner'] = self
                return
            vars(cleanup)['_fixture_scoring_cleanup_owner'] = self
            raise


def _reserve_facade_sink(source: TreeFixtureSource, model: ConfiguredTreeModelV3, model_bytes: int,
                         joint_limit: int, extra_owned_bytes: int, fixed: int) -> Json:
    """One persistent full controller+child envelope, never a reduced runtime."""
    resource_estimate(model, model_bytes=model_bytes)
    controller = child = model.resources.max_owned_bytes
    reserved = controller+child
    existing = source.retained_bytes-source.retained_without_sink()
    if source.retained_bytes+extra_owned_bytes+fixed+16*1024**2+reserved > joint_limit:
        raise ContractError('full facade controller/child joint admission exceeded before construction')
    source.reserve_sink(existing+reserved)
    return {'existing_source_sink_bytes':existing, 'controller_bytes':controller, 'child_bytes':child,
        'added_source_sink_bytes':reserved, 'source_sink_high_water_bytes':existing+reserved,
        'lifetime':'Source lifetime, including healthy close; conservative through native/reload and ordinary completion'}


def _admit_feature_work(model: ConfiguredTreeModelV3, model_bytes: int, local_bytes: int,
                        retained: int, admit: Callable[[int], None]) -> None:
    # Only this predictor's relevant inputs/caches/chunks/scores enter its local
    # runtime limit. SAME allocations additionally enter the global Source check.
    resource_estimate(model, model_bytes=model_bytes, extra_bytes=local_bytes)
    admit(retained+local_bytes+2*LIMIT)


def _admit_joint_work(borrow: _FixtureEpisodeBorrow, fixed: int, limit: int, external: int) -> None:
    borrow.admit_external(external)
    if borrow.retained_bytes+fixed > limit:
        raise ContractError('joint diagnostic caller/Source/native/feature budget exceeded')


def _feature_parity(borrow: _FixtureEpisodeBorrow, metadata: Json, arrays: dict[str, Any], model: ConfiguredTreeModelV3,
                    tree: LoadedTreeV3 | None, retained: int, admit: Callable[[int], None],
                    facade: _FacadeLoop | None = None) -> tuple[int,int,int]:
    rows = scores = worker_rows = 0
    model_bytes = 0 if tree is None else tree.descriptor.model_bytes
    for reference, frame in zip(borrow.reference_decisions(), metadata['decisions'], strict=True):
        sample = reference.sample
        _admit_feature_work(model,model_bytes,_owned((sample.tokens,sample.candidates))
            +128*sum(value.nbytes for value in arrays.values())+65536,retained,admit)
        tokens, candidates = _tokens(metadata, frame, arrays)
        if tokens != sample.tokens or candidates != sample.candidates or frame['selected_index'] != sample.selected_index:
            raise ContractError('reloaded immutable tokens/selected identity differ from original Python reference')
        python = PreparedTreeFrameV3.create(sample.tokens, model, model_bytes=model_bytes,
            extra_owned_bytes=_owned((sample.candidates,tokens,candidates)))
        _admit_feature_work(model,model_bytes,_owned((sample.tokens,sample.candidates,tokens,candidates,python))
            +2*65536,retained,admit)
        native = PreparedTreeFrameV3.create(tokens, model, model_bytes=model_bytes,
            extra_owned_bytes=_owned((sample.tokens,sample.candidates,python,candidates)))
        # Two prepared caches, concurrent full-row chunks, selected tokens and
        # returned full scores overlap. The shared runtime estimate already
        # reserves chunk tensors/working memory; add retained objects and reply.
        reply_bound = 0 if facade is None else 64*len(candidates.kinds)+65536
        _admit_feature_work(model,model_bytes,_owned((sample.tokens,sample.candidates,tokens,candidates,python,native))
            +2*65536+reply_bound,retained,admit)
        for (left_start,left), (right_start,right) in zip(python.iter_chunks(sample.candidates), native.iter_chunks(candidates), strict=True):
            if left_start != right_start or left.shape[1:] != (2810,) or left.dtype != right.dtype or not np.array_equal(left,right):
                raise ContractError('native/reloaded full2810 Tree3 feature rows differ from Python reference')
            rows += len(left)
        selected = FixtureSelectedInput(tokens,candidates,sample.selected_index,sample.label,sample.population_id,
            sample.cohort_id,sample.episode,sample.split,sample.viewer,sample.decision,sample.policy_id).selected_tokens()
        _, left = next(python.iter_chunks(sample.selected_tokens()))
        _, right = next(native.iter_chunks(selected))
        if left.shape != (1,2810) or not np.array_equal(left,right):
            raise ContractError('native/reloaded selected2810 Tree3 feature differs')
        if tree is not None:
            reply = None if facade is None else facade.score(reference)
            for (a,x),(b,y) in zip(_score_chunks(tree,python,sample.candidates),_score_chunks(tree,native,candidates),strict=True):
                if a != b or not np.array_equal(x,y):
                    raise ContractError('native CPU full-table predictions differ after raw persistence')
                if reply is not None and tuple(float(value) for value in x) != reply.scores[a:a+len(x)]:
                    raise ContractError('component facade full-table predictions differ from paired native/reloaded CPU')
                scores += len(x)
            left_score = next(_score_chunks(tree,python,sample.selected_tokens()))[1]
            right_score = next(_score_chunks(tree,native,selected))[1]
            if reply is not None:
                if float(left_score[0]) != reply.scores[sample.selected_index]:
                    raise ContractError('component facade selected original-row prediction differs')
                worker_rows += len(reply.scores)
            if not np.array_equal(left_score,right_score):
                raise ContractError('native CPU selected-row prediction differs after raw persistence')
        del native, python, tokens, candidates, selected
    return rows,scores,worker_rows


def native_fixture_diagnostics(source: TreeFixtureSource, output: Path, *, max_owned_bytes: int,
        deadline: float, extra_owned_bytes: int = 0, max_scratch_bytes: int | None = None,
        diagnostic_native: Any = None, tree: LoadedTreeV3 | None = None,
        pair: PairedTraining | None = None, scoring_ledger: ReservationLedger | None = None) -> FixtureTokenDiagnostics:
    """Verify all Source-owned episodes; never consume its ordinary worklist."""
    if type(source) is not TreeFixtureSource:
        raise ContractError('SAME concrete TreeFixtureSource required')
    module = load_native() if diagnostic_native is None else diagnostic_native
    identity = _preflight(module, diagnostic=diagnostic_native is not None)
    if scoring_ledger is not None:
        if type(scoring_ledger) is not ReservationLedger or tree is None or pair is None:
            raise ContractError('component facade requires actual paired Tree3 and caller ledger')
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ContractError('synchronous fixture adapter cannot own a loop inside an active caller loop')
    integer(max_owned_bytes, 1, 12*1024**3); integer(extra_owned_bytes, 0, max_owned_bytes)
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline <= 0:
        raise ContractError('finite diagnostic caller deadline required')
    deadline = min(deadline, source.deadline)
    limit = min(max_owned_bytes, source.limit)
    scratch_limit = source.settings.scratch_budget_mib*1024**2
    if max_scratch_bytes is not None:
        scratch_limit = min(scratch_limit, integer(max_scratch_bytes, 1))
    if tree is not None:
        _paired_predictor_binding(tree,pair)
    elif pair is not None:
        raise ContractError('paired owner requires its actual loaded Tree3 predictor')
    source.guard()
    model = ConfiguredTreeModelV3(encoder=source.encoder) if tree is None else tree.model
    model_bytes = 0 if tree is None else tree.descriptor.model_bytes
    fixed = resource_estimate(model, model_bytes=model_bytes)['total_bytes']
    if source.retained_bytes+extra_owned_bytes+fixed+16*1024**2 > limit:
        raise ContractError('joint Source/caller/model admission exhausted before diagnostic allocation')
    facade_reservation = None if scoring_ledger is None else _reserve_facade_sink(
        source,model,model_bytes,limit,extra_owned_bytes,fixed)
    if tree is not None:
        if type(tree) is not LoadedTreeV3 or type(pair) is not PairedTraining or tree.training_binding != pair.binding:
            raise ContractError('actual paired component Tree3 CPU predictor required')
        if (tree.origin.source_kind != 'card-instance-tree-fixture-originals'
                or canonical(tree.origin.fields()['source_origin']) != canonical(origin())
                or canonical(asdict(model.encoder)) != canonical(asdict(source.encoder))
                or pair.inference['tree_descriptor'] != asdict(tree.descriptor)):
            raise ContractError('paired component Tree3 predictor/input identity differs')
        pair.finalize()
        _, descriptor = _validate_booster(tree.booster, model, deadline=deadline)
        if descriptor != tree.descriptor:
            raise ContractError('paired native CPU predictor changed')
    elif pair is not None:
        raise ContractError('paired owner requires its actual loaded Tree3 predictor')
    binary = SealedScan(max_owned_bytes=limit-source.retained_bytes-extra_owned_bytes, deadline=deadline)
    with binary.file(Path(identity['extension_path']), identity['extension_sha256']):
        pass
    binary.finalize()
    persisted = SealedScan(max_owned_bytes=limit, deadline=deadline)
    roots = list(source.original_roots)
    if pair is not None:
        roots.append(Path(pair.binding.directory))
    output, ancestors = _outside_output(output, roots)
    source.guard(); binary.finalize()
    output.mkdir(mode=0o700)
    info = output.lstat()
    ancestors[output] = (info.st_dev,info.st_ino)
    receipts: list[Json] = []
    counts: dict[str, Json] = {}
    disk_bytes = feature_rows = score_rows = facade_rows = 0
    initial_workers = set() if scoring_ledger is None else set(scoring_ledger.workers)
    with _FacadeLoop(source,2*model.resources.max_owned_bytes) if scoring_ledger is not None else nullcontext(None) as facade:
        for ordinal in range(source.diagnostic_episode_count):
            caller = extra_owned_bytes+binary.peak+persisted.retained+_owned(receipts)+16*1024**2+16*model_bytes
            with source.borrow_diagnostic_episode(ordinal, extra_owned_bytes=caller) as borrow:
                def admit(external: int) -> None:
                    _admit_joint_work(borrow,fixed,limit,external)
                def guard() -> None:
                    borrow.guard(); binary.finalize(); _check_output_ancestors(ancestors)
                    if pair is not None:
                        pair.finalize()
                guard()
                available = limit-borrow.retained_bytes-fixed
                if available <= 0:
                    raise ContractError('no remaining owned storage for native fixture tokens')
                limits = canonical({'memory_bytes': available, 'scratch_bytes': scratch_limit-disk_bytes,
                    'max_source_file_bytes': min(64,source.settings.max_source_file_mib)*1024**2,
                    'max_record_bytes': min(64,source.settings.max_record_mib)*1024**2,
                    'chunk_target_bytes': min(64,source.settings.chunk_target_mib)*1024**2}).encode()
                native = _NativeEpisode(module, deadline)
                try:
                    native.construct(module, borrow, limits, deadline, source.encoder.aggregate_window)
                    guard()
                    packet = native.executor.call(native.owner.export_tokens)
                    if type(packet) is not tuple or len(packet) != 4 or type(packet[0]) is not bytes or len(packet[0]) > LIMIT:
                        raise ContractError('closed fixture4tuple payload required')
                    admit(native.owner.used_bytes()+64*len(packet[0])+65536)
                    metadata = _json(packet[0])
                    if _semantics(metadata) != borrow.expected_metadata:
                        raise ContractError('native semantic facts differ from SAME Source originals')
                    admit(native.owner.used_bytes()+_owned(metadata)+65536+1024*len(metadata['arrays']))
                    arrays = _arrays(metadata, packet[1:])
                    raw_bytes = sum(value.nbytes for value in arrays.values())
                    retained = native.owner.used_bytes()+_owned(metadata)
                    admit(retained+128*raw_bytes+65536)
                    for reference, frame in zip(borrow.reference_decisions(), metadata['decisions'], strict=True):
                        expected = _reference_arrays(reference.sample, frame)
                        for key, values in expected.items():
                            array = arrays[key]
                            dtype = np.uint8 if 'present' in key or 'mask' in key else np.int64
                            actual = np.asarray(values, dtype=dtype)
                            if array.shape != actual.shape or array.dtype != actual.dtype or not np.array_equal(array,actual):
                                raise ContractError('native raw tokens/order differ from Source-owned Python reference')
                            del actual
                        del expected
                    if metadata['decisions']:
                        del reference, array
                    guard()
                    # The writer uses shared deterministic bundle1, never Dataset or
                    # Source manifests. Reserve shard encoding before any publication.
                    writer_peak = retained+2*LIMIT+raw_bytes+65536
                    admit(writer_peak)
                    def io_check() -> None:
                        admit(writer_peak)  # noqa: B023 - synchronous per-episode callback; refreshed before reload
                        _check_output_ancestors(ancestors)
                    if disk_bytes+4*raw_bytes+len(packet[0])+1024**2 > scratch_limit:
                        raise ContractError('cumulative fixture diagnostic scratch exhausted before persistence')
                    episode = output/f'episode-{ordinal:06}'
                    episode.mkdir(mode=0o700)
                    info = episode.lstat()
                    ancestors[episode] = (info.st_dev,info.st_ino)
                    bundle = None if not arrays else write_bundle(episode, 'raw', arrays,
                        shard_bytes=min(64,source.settings.chunk_target_mib)*1024**2, _check=io_check)
                    container = {'schema_version': 1, 'kind': 'card-instance-fixture-token-diagnostic',
                        'source_manifest_sha256': source.expected_digest, 'native_binary': identity,
                        'metadata': metadata, 'bundle': bundle}
                    encoded = (canonical(container)+'\n').encode()
                    if len(encoded) > LIMIT or len(encoded) > min(64,source.settings.max_record_mib)*1024**2:
                        raise ContractError('diagnostic container record exceeds configured bound')
                    with (episode/'diagnostic.json').open('xb') as stream:
                        stream.write(encoded)
                    file_names = {'diagnostic.json'} | (set() if bundle is None else set(bundle['files']))
                    persisted.inventory(episode, file_names)
                    for name in file_names:
                        with persisted.file(episode/name):
                            pass
                        disk_bytes += (episode/name).stat().st_size
                    if disk_bytes > scratch_limit:
                        raise ContractError('cumulative persisted diagnostic bytes exceed scratch')
                    container_sha = hashlib.sha256(encoded).hexdigest()
                    # No bank/view/native ownership is retained during reload.
                    del arrays, packet, encoded
                    native_owner = native.owner
                    native.close()
                    if native_owner.used_bytes() != 0:
                        raise ContractError('exported native views remain before reload')
                    del native_owner
                    writer_peak = _owned((metadata,bundle,container))+persisted.retained+2*LIMIT
                    admit(writer_peak)
                    guard()
                    with persisted.file(episode/'diagnostic.json', container_sha) as (reader, info):
                        loaded = _json(reader.read(info.st_size+1))
                    closed(loaded, CONTAINER)
                    if (type(loaded['schema_version']) is not int or loaded['schema_version'] != 1
                            or loaded['kind'] != 'card-instance-fixture-token-diagnostic'
                            or loaded['source_manifest_sha256'] != source.expected_digest
                            or canonical(loaded['native_binary']) != canonical(identity)
                            or _semantics(loaded['metadata']) != borrow.expected_metadata
                            or canonical(loaded['metadata']) != canonical(metadata)
                            or canonical(loaded['bundle']) != canonical(bundle)):
                        raise ContractError('persisted fixture container crossed original/native identity')
                    if loaded['bundle'] is None:
                        if loaded['metadata']['arrays'] or loaded['metadata']['decisions']:
                            raise ContractError('bundle:null requires the real empty diagnostic mapping')
                        reloaded: dict[str, Any] = {}
                    else:
                        validate_bundle_metadata(loaded['bundle'])
                        reloaded = read_bundle(episode, loaded['bundle'], max_decoded_bytes=limit-borrow.retained_bytes-fixed, _check=io_check)
                        if set(reloaded) != {item['key'] for item in metadata['arrays']}:
                            raise ContractError('reloaded raw token keyset differs')
                        for item in metadata['arrays']:
                            array = reloaded[item['key']]
                            if str(array.dtype) != item['dtype'] or list(array.shape) != item['shape'] or array.nbytes > LIMIT:
                                raise ContractError('reloaded raw token dtype/shape differs')
                            array.flags.writeable = False
                    guard()
                    retained = _owned((loaded,reloaded,metadata,bundle,container))+persisted.retained
                    admit(retained+128*raw_bytes+2*LIMIT)
                    worker_reserved = worker_limit = 0
                    if facade is not None and metadata['decisions']:
                        assert pair is not None and scoring_ledger is not None
                        # Full controller+child ownership is ALREADY charged in
                        # Source's retained sink, even after a failed borrow.
                        worker_limit = model.resources.max_owned_bytes
                        worker_reserved = 2*worker_limit
                        admit(retained+128*raw_bytes+2*LIMIT)
                        guard()
                        facade.initialize(pair,model,borrow,metadata,scoring_ledger,deadline)
                        guard()
                    try:
                        frows, srows, wrows = _feature_parity(borrow,metadata,reloaded,model,tree,
                            retained,admit,facade if worker_reserved else None)
                    except BaseException as error:
                        if facade is not None:
                            try:
                                facade.close_worker()
                            except BaseException as cleanup:  # noqa: BLE001 - initiating comparison/cancellation wins
                                error.add_note('fixture scorer close: '+str(cleanup))
                                vars(error)['_fixture_scoring_cleanup_owner'] = facade
                        raise
                    else:
                        if facade is not None:
                            facade.close_worker()
                    feature_rows += frows; score_rows += srows; facade_rows += wrows
                    guard(); persisted.finalize()
                    cohort = counts.setdefault(metadata['cohort_id'], {'episodes':0,'decisions':0,'zero_endpoint_episodes':0})
                    cohort['episodes'] += 1; cohort['decisions'] += len(metadata['decisions'])
                    cohort['zero_endpoint_episodes'] += not metadata['decisions']
                    receipts.append({'ordinal':ordinal,'cohort_id':metadata['cohort_id'],'episode':metadata['episode'],
                        'viewer':metadata['viewer'],'label':metadata['label'],'decisions':len(metadata['decisions']),
                        'container_sha256':container_sha,'directory':str(episode),
                        'scorer_managed_bytes':worker_limit,'scorer_controller_and_child_reserved_bytes':worker_reserved})
                    del reloaded, loaded, container, metadata, bundle
                    admit(0)
                except BaseException as error:
                    _cleanup(native,error)
                    raise
        report = {'kind':'card-instance-fixture-token-comparisons','source_manifest_sha256':source.expected_digest,
            'native_binary':identity,'episodes':len(receipts),'cohorts':counts,'receipts':receipts,
            'feature_columns':2810,'full_feature_rows':feature_rows,'native_cpu_score_rows':score_rows,
            'joint_limit_bytes':limit,'scoring_scope':None if facade is None else COMPONENT,
            'facade_score_rows':facade_rows,'scoring_workers_closed':0 if facade is None else facade.workers_closed,
            'scoring_closed_health':[] if facade is None else facade.closed_health,
            'scoring_reservation':facade_reservation,
            'persisted_bytes':disk_bytes,'output':str(output)}
        retained_result = 64*len(canonical(report).encode())+_owned((ancestors,receipts))+binary.retained+persisted.retained+extra_owned_bytes+16*model_bytes+65536
        existing_sink = source.retained_bytes-source.retained_without_sink()
        source.reserve_sink(existing_sink+retained_result)
        source.guard(); binary.finalize(); persisted.finalize(); _check_output_ancestors(ancestors)
        if scoring_ledger is not None and set(scoring_ledger.workers) != initial_workers:
            raise ContractError('component scorer reservations remain before diagnostic result')
        return FixtureTokenDiagnostics(source,binary,persisted,ancestors,report,pair)
