"""Scores-only component/actual-input facade over the existing private process owner."""
from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn, cast

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBankView, EventDelta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import delta_dict, frame_dict, parse_game_generated_v3
from gf_models.card_instance_tree_scoring import (
    COMPONENT,
    DIAGNOSTIC,
    ScoringResult,
    binding_for,
    check_binding,
    scores,
    scoring_settings,
    search_state,
    stage_bank,
    timings,
)
from gf_models.structured_tree_artifacts import inspect_artifact_v3
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV3,
    TreeDescriptor,
    resource_estimate,
)
from gf_models.structured_tree_encoder import _owned
from gf_models.structured_tree_pairing import TrainingBinding, inspect_training_v3
from gf_simulator_client.framing import MAX_FRAME_BYTES, frame_decode_bytes, read_frame_admitted
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker, _failed_spawn


class _ScoringTransport(StructuredParticipantWorker):
    """Only a private composition detail; reuse exchange, abort, close and reap."""
    def __init__(self, initial: Json, limit: int, retained: int, reply_limit: int, ledger: ReservationLedger, deadline: float) -> None:
        self.initial = initial
        self.timeout, self.closed, self.initialized = 10., False, False
        self.health = {}
        self.io_thread = self._close_task = None
        self._spawn_started = time.monotonic()
        self.process = cast('subprocess.Popen[bytes]', None)
        self.limit, self.retained, self.reply_limit = limit, retained, reply_limit
        self.session_deadline = deadline
        self.lease = ledger.reserve_worker(0)
        self.lease._cleanup_owner = self
        process = None
        try:
            process = subprocess.Popen([sys.executable, '-E', '-m',
                'gf_simulator_client.card_instance_tree_scoring_bootstrap', str(limit)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                env={'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
                     'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
            self.process = process
            self.lease.spawned(process)
        except BaseException as error:
            _failed_spawn(process, self.lease, error)
            if not self.lease.released:
                vars(error)['_structured_startup_owner'] = self
            raise

    def _read_reply(self, source: Any) -> Json:
        return read_frame_admitted(source, max_owned_bytes=self.limit,
                                   retained_bytes=self.retained, max_frame_bytes=self.reply_limit)

    def begin_close(self) -> None:
        if self._close_task is None:
            self.closed = True
            self._close_task = asyncio.create_task(self._close_owned(min(2., max(0., self.session_deadline-time.monotonic())), time.monotonic()))

    async def _abort(self, original: BaseException) -> NoReturn:
        self.begin_close()
        await super()._abort(original)


class CardInstanceTreeScoringWorker:
    """No participant inheritance, Selection/Action API, or game factory registration."""
    def __init__(self, training_directory: Path, model: ConfiguredTreeModelV3, actions: ActionConfigV3,
                 sampling_seed: int, actor: int, episode: str, *, input_scope: str,
                 ledger: ReservationLedger, deadline: float, training_binding: TrainingBinding | None = None) -> None:
        if type(model) is not ConfiguredTreeModelV3 or type(actions) is not ActionConfigV3:
            raise ContractError('explicit typed Tree3 scoring model/actions required')
        integer(sampling_seed)
        if type(deadline) not in (float, int) or not math.isfinite(deadline):
            raise ContractError('original finite absolute scoring session deadline required')
        self._deadline = float(deadline)
        self._active_deadline = self._deadline
        self._check_deadline()
        self._pair = inspect_training_v3(training_directory, expected_binding=training_binding,
                                         max_owned_bytes=model.resources.max_owned_bytes, deadline=self._deadline)
        self._model, self._actions, self._scope = model, actions, input_scope
        self._rules = card_instance_rules()
        self._closed_requested = self._failed = self._initialized = False
        self._inflight: asyncio.Task[Any] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._view: EventBankView | None = None
        self._last: DecisionFrame | None = None
        self._count = 0
        self._rng = search_state(sampling_seed)
        self._last_timings: Json = {}
        self._startup_ms: float | None = None
        artifact = self._pair.inference
        source_dir = Path(training_directory).absolute()/'artifact'
        sha = self._pair.completion['outputs']['inference_sha256']
        # Active choices can differ only in execution; immutable actual pair metadata stays unchanged.
        _, text = inspect_artifact_v3(source_dir, expected_digest=sha, model=model,
            action_config_hash=artifact['action_config_hash'], action_contract=actions.action_contract,
            purpose='component', synthetic_capacity=False, source_kind=artifact['source_kind'], source_origin=artifact['source_origin'],
            deadline=time.monotonic()+10, _scan=self._pair.scan)
        metadata_size = self._pair.scan.files[source_dir/'inference.json'][0][2]
        text_size = artifact['model_file']['bytes']
        actual_text_size = len(text.encode())
        if metadata_size > MAX_FRAME_BYTES or text_size > MAX_FRAME_BYTES or actual_text_size > MAX_FRAME_BYTES:
            raise ContractError('scoring metadata/text exceeds64MiB before copying')
        if actual_text_size != text_size:
            raise ContractError('scoring inspected text size differs from exact artifact')
        self._binding = binding_for(model, actions, input_scope, artifact, sha, actor, episode)
        self._caps, self._costs = scoring_settings(model, actions, input_scope,
            artifact['action_config_hash'], TreeDescriptor(**artifact['tree_descriptor']))
        controls = {'protocol_version': 3, 'kind': 'tree-scoring-init', 'binding': self._binding,
            'model': {**asdict(model), 'checkpoint': None}, 'actions': asdict(actions), 'rules': asdict(self._rules),
            'caps': asdict(self._caps), 'numeric_costs': asdict(self._costs), 'sampling_seed': sampling_seed,
            'actor': actor, 'episode': episode}
        # UTF-8/JSON escaping can expand strings sixfold; admit encoding + both
        # outer and inner decoded ownership before constructing the full packet.
        controls_bytes = len(canonical(controls).encode())
        upper = 6*(metadata_size+text_size+controls_bytes)+65536
        resource_estimate(model, model_bytes=text_size, extra_bytes=self._pair.scan.retained
            + _owned(controls)+frame_decode_bytes(min(upper, MAX_FRAME_BYTES)) + 34*(metadata_size+text_size) + upper*3)
        with self._pair.scan.file(source_dir/'inference.json', sha) as (stream, info):
            metadata = stream.read(info.st_size+1)
            if len(metadata) != info.st_size:
                raise ContractError('scoring inference original length changed')
        initial = {**controls, 'inference_metadata': metadata.decode(), 'model_text': text}
        encoded = canonical(initial).encode()
        if not 0 < len(encoded) <= MAX_FRAME_BYTES:
            raise ContractError('actual encoded Tree scoring initialization exceeds64MiB before spawn')
        self._retained = self._pair.scan.retained+_owned(initial)
        self._reply_limit = min(MAX_FRAME_BYTES, 65536+actions.max_candidates*(256+18*4))
        resource_estimate(model, model_bytes=text_size, extra_bytes=self._retained
            + frame_decode_bytes(len(encoded))+frame_decode_bytes(self._reply_limit))
        self._pair.finalize()
        self._check_deadline()
        self._transport: Any = None
        try:
            self._transport = _ScoringTransport(initial, model.resources.max_owned_bytes,
                resource_estimate(model, model_bytes=text_size, extra_bytes=self._retained)['total_bytes'], self._reply_limit, ledger, self._deadline)
        except BaseException as error:
            self._failed = True
            self._transport = getattr(error, '_structured_startup_owner', None)
            vars(error)['_tree_scoring_startup_owner'] = self
            raise

    def _check_deadline(self) -> None:
        if time.monotonic() >= min(self._deadline, self._active_deadline):
            raise TimeoutError('original Tree scoring session/operation deadline exhausted')

    def _exchange_budget(self) -> None:
        self._check_deadline()
        self._transport.timeout = min(10., self._active_deadline-time.monotonic())
        if self._transport.timeout <= 0:
            raise TimeoutError('Tree scoring operation exhausted before exchange')

    @property
    def health(self) -> Json:
        result = dict(self._transport.health) if self._transport is not None else {'startup_failure': True}
        process = None if self._transport is None else self._transport.process
        result['returncode'] = None if process is None else process.poll()
        return result

    @property
    def last_timings(self) -> Json:
        return dict(self._last_timings)

    @property
    def startup_elapsed_ms(self) -> float | None:
        return self._startup_ms

    @property
    def closed(self) -> bool:
        return self._closed_requested

    def _admit_operation(self) -> None:
        if self._inflight is not None:
            raise ContractError('one in-flight Tree scoring operation required')
        if self._closed_requested or self._failed:
            raise ContractError('Tree scoring worker is cleanup-only')
        self._check_deadline()
        self._active_deadline = min(self._deadline, time.monotonic()+10)
        self._pair.scan.deadline = self._active_deadline
        self._inflight = asyncio.current_task()

    async def _initialize_owned(self) -> None:
        if self._initialized:
            return
        self._pair.finalize()
        self._exchange_budget()
        received = await self._transport._roundtrip(self._transport.initial)
        closed(received, {'protocol_version', 'kind', 'binding', 'model_load_ms'})
        self._check(received, 'tree-scoring-ready')
        timings({'model_load_ms': received['model_load_ms']}, ready=True)
        self._check_deadline()
        if self._closed_requested:
            raise ContractError('Tree scorer closed before ready could commit')
        self._startup_ms = (time.monotonic()-self._transport._spawn_started)*1000
        self._initialized = True
        # Model/init wire buffers are no longer needed by the controller.
        self._transport.initial = {}

    def _check(self, received: Json, kind: str) -> None:
        if type(received['protocol_version']) is not int or received['protocol_version'] != 3 or received['kind'] != kind:
            raise ContractError('Tree scoring response discriminator mismatch')
        check_binding(received['binding'], self._binding)

    async def initialize(self) -> None:
        self._admit_operation()
        try:
            await self._initialize_owned()
        except BaseException as error:
            self._failed = True
            vars(error)['_tree_scoring_cleanup_owner'] = self
            await self._transport._abort(error)
            raise
        finally:
            self._inflight = None
            self._active_deadline = self._deadline

    async def score_original(self, frame: DecisionFrame, delta: EventDelta, table: CandidateTable) -> ScoringResult:
        return await self._score(frame, delta, table)

    async def generate_and_score(self, frame: DecisionFrame, delta: EventDelta) -> ScoringResult:
        return await self._score(frame, delta, None)

    async def _score(self, frame: DecisionFrame, delta: EventDelta, table: CandidateTable | None) -> ScoringResult:
        self._admit_operation()
        try:
            generated = table is None
            if generated and self._scope != DIAGNOSTIC:
                raise ContractError('only actual-own-input diagnostics may generate')
            await self._initialize_owned()
            self._pair.finalize()
            before = _owned((frame, delta, table, self._view, self._last))
            event_bytes = sum(len(event) for event in delta._events)
            if self._view is not None:
                event_bytes += sum(len(event) for event in self._view._events)
            resource_estimate(self._model, model_bytes=self._binding['tree_descriptor']['model_bytes'],
                extra_bytes=self._retained+4*before+64*len(frame.payload)+32*event_bytes+65536)
            bank = stage_bank(self._view, self._last, frame, delta, self._binding, self._caps)
            staged_view = bank.view()
            if table is not None:
                expected_purpose = 'component' if self._scope == COMPONENT else 'qualified_game'
                if table.purpose != expected_purpose or table.synthetic_capacity:
                    raise ContractError('Tree score-original scope/table purpose mismatch')
                table.validate(frame)
                integer(len(table.kinds), 1, self._caps.max_candidates)
            request = {'protocol_version': 3, 'kind': 'generate-and-score' if generated else 'score-original',
                'binding': self._binding, 'request_id': self._count+1, 'frame': frame_dict(frame, layout=LAYOUT),
                'delta': delta_dict(delta, layout=LAYOUT),
                **({'rng_before': self._rng} if table is None else {'table': asdict(table)})}
            encoded_size = len(canonical(request).encode())
            if encoded_size > MAX_FRAME_BYTES:
                raise ContractError('Tree scoring request exceeds64MiB')
            extra = self._retained+before+_owned(request)+_owned(staged_view)
            resource_estimate(self._model, model_bytes=self._binding['tree_descriptor']['model_bytes'],
                extra_bytes=extra+frame_decode_bytes(encoded_size)+frame_decode_bytes(self._reply_limit))
            self._transport.retained = resource_estimate(self._model,
                model_bytes=self._binding['tree_descriptor']['model_bytes'], extra_bytes=extra)['total_bytes']
            lease_key = self._transport.lease.reserve_decision(0)
            self._exchange_budget()
            received = await self._transport._roundtrip(request)
            closed(received, {'protocol_version', 'kind', 'binding', 'request_id', 'frame_hash', 'table_hash',
                              'row_count', 'scores', 'timings'} | ({'generated'} if generated else set()))
            self._check(received, 'tree-generated-scores' if generated else 'tree-scores')
            if received['request_id'] != self._count+1 or type(received['request_id']) is not int or received['frame_hash'] != frame.frame_hash:
                raise ContractError('Tree scores original request/frame identity mismatch')
            raw_generated = None
            next_rng = self._rng
            if generated:
                proposals = parse_game_generated_v3(received['generated'], action_contract=self._actions.action_contract, rules=self._rules)
                proposals.table.validate(frame)
                report = proposals.report
                if (report.config_hash != digest(asdict(self._actions)) or report.numeric_costs_hash != digest(asdict(self._costs))
                        or report.rng_before != self._rng or report.attempts > self._actions.max_expansions
                        or report.discovered > self._actions.max_temporary_candidates
                        or report.precount.count_states > self._actions.count_states):
                    raise ContractError('Tree generated config/cost/RNG/bounds mismatch')
                table = proposals.table
                next_rng = report.rng_after
                raw_generated = canonical(received['generated']).encode()
            assert table is not None
            if received['table_hash'] != table.table_hash or type(received['row_count']) is not int or received['row_count'] != len(table.kinds):
                raise ContractError('Tree scores original full table/order/count mismatch')
            if len(table.kinds) > self._caps.max_candidates or any(b-a > 18 for a,b in zip(table.offsets[:-1],table.offsets[1:],strict=True)):
                raise ContractError('Tree generated table exceeds exact caller bounds')
            result_scores = scores(received['scores'], len(table.kinds))
            timings(received['timings'], generated=generated)
            self._pair.finalize()
            self._check_deadline()
            if self._closed_requested:
                raise ContractError('Tree scorer closed before reply could commit')
            result = ScoringResult(frame.frame_hash, table.table_hash, result_scores,
                                   canonical(received['timings']).encode(), raw_generated)
            next_timings = dict(received['timings'])
            self._check_deadline()
            self._transport.lease.reply(lease_key)
            self._view, self._last, self._count, self._rng = staged_view, frame, self._count+1, next_rng
            self._last_timings = next_timings
            return result
        except BaseException as error:
            self._failed = True
            vars(error)['_tree_scoring_cleanup_owner'] = self
            await self._transport._abort(error)
            raise
        finally:
            self._inflight = None
            self._active_deadline = self._deadline

    async def close(self) -> None:
        if self._close_task is None:
            self._closed_requested = True
            self._pair.scan.deadline = self._deadline
            if self._transport is not None:
                self._transport.begin_close()
            pending = self._inflight
            if pending is not None and pending is not asyncio.current_task():
                pending.cancel()
            self._close_task = asyncio.create_task(self._close_owned(pending))
        cancellation: asyncio.CancelledError | None = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException:  # noqa: BLE001 - task.result preserves the original cleanup failure below.
                break
        try:
            self._close_task.result()
        except BaseException as error:
            vars(error)['_tree_scoring_cleanup_owner'] = self
            if cancellation is not None:
                vars(cancellation)['_tree_scoring_cleanup_owner'] = self
                raise cancellation from error
            raise
        if cancellation is not None:
            raise cancellation

    async def _close_owned(self, pending: asyncio.Task[Any] | None) -> None:
        failure = None
        try:
            if self._transport is not None:
                await self._transport.close()
        except BaseException as error:  # noqa: BLE001 - retain cleanup failure and same process owner.
            failure = error
        if pending is not None and pending is not asyncio.current_task():
            try:
                await pending
            except asyncio.CancelledError:
                pass
            except BaseException as error:  # noqa: BLE001 - operation failure is independently retained in health.
                self._transport.health['pending_error'] = type(error).__name__
        try:
            self._pair.finalize()
        except BaseException as error:  # noqa: BLE001 - original process cleanup failure has priority.
            failure = failure or error
        if failure is not None:
            vars(failure)['_tree_scoring_cleanup_owner'] = self
            raise failure
        self._pair.scan.files.clear()
        self._pair.scan.inventories.clear()
