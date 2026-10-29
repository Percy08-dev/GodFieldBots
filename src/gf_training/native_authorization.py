"""Once-per-operation schedule authorization and lifetime ownership for Rust I/O."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple, Self

from gf_common.domain import ContractError, Json, canonical
from gf_common.preprocessing import PreparationConfig
from gf_common.schedule import serialize_schedule
from gf_training.collection import OnlinePPOContext, training_groups, validation_groups
from gf_training.lineage import _OWNED_LOCKS, run_lock
from gf_training.native_backend import binary_identity, load_native
from gf_training.native_execution import NativeExecution
from gf_training.preparation import ResourceLimit


def _invoke(function: Any, *arguments: Any) -> Any:
    try:
        return function(*arguments)
    except MemoryError as exc:
        raise ResourceLimit(str(exc)) from exc
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


class NativeBatch(NamedTuple):
    features: Any
    state_features: Any
    episode_offsets: Any
    candidate_offsets: Any
    candidate_counts: Any
    selected: Any
    old_logprob: Any
    old_value: Any
    old_logprob_present: Any
    old_value_present: Any
    terminal_rewards: Any
    winner: Any
    learner_viewer: Any
    split: Any
    feature_layout: str
    metadata: bytes
    index_columns: Any = None


def _expected_collection_items(summary: Json, check: Callable[[], None] | None = None) -> Iterator[Json]:
    """Reconstruct policy decisions from compact immutable controls, never rows."""
    import numpy as np

    for collection in summary['collections']:
        if check is not None:
            check()
        parent = summary['parents'][collection['parent_index']]
        generations = parent['generations']
        generation = collection['collection_generation']
        record = generations[generation]
        controls = record['parent_controls']
        registry: Json = {'random': {'model_id': 'random', 'policy_options': None},
                          'heuristic': {'model_id': 'heuristic', 'policy_options': None}}

        def binding(number: int, generations: Any = generations, registry: Json = registry,
                    parent: Json = parent) -> str:
            checksum = generations[number]['inference_sha256']
            name = 'mlp-' + checksum
            registry[name] = {'model_id': 'mlp', 'policy_options': {
                'checkpoint': str(Path(parent['directory']) / 'artifacts' / f'{number:06}'),
                'expected_manifest_digest': checksum}}
            return name

        learner = binding(generation)
        if collection['collection_kind'] == 'ppo_batch':
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = controls['schedule_opponent_rng']
            snapshots = tuple(binding(number) for number in controls['snapshot_pool'])
            expected = training_groups(parent['state']['seed'], generation, learner, snapshots, rng)
        elif collection['collection_kind'] == 'ppo_validation':
            expected = validation_groups(parent['state']['seed'], learner)
        else:
            raise ContractError('unsupported native collection authorization kind')
        yield {'directory': collection['directory'],
               'schedule': serialize_schedule(expected, 4), 'bindings': registry}

def _expected_collections(summary: Json) -> list[Json]:
    return list(_expected_collection_items(summary))


class NativePreparation:
    """Main-thread locks/authorization with one joined native execution thread.

    Offline owners survive learner reap. Online owners finalize after numeric
    staging and before optimizer requests. The cancel handle exists before even
    session construction, and cleanup always joins native work before locks end.
    """

    def _setup(self, config: PreparationConfig, deadline: float, wall_deadline: float | None) -> None:
        self.config = config
        self.deadline = deadline
        self.wall_deadline = deadline if wall_deadline is None else wall_deadline
        self.started = time.monotonic()
        self._locks = ExitStack()
        self._tokens: dict[Path, str] = {}
        self._closed = False
        self.exhausted = False
        self._proof: Json | None = None
        self._session: Any = None
        self._execution: NativeExecution | None = None
        native = load_native()
        self.identity = binary_identity(native)
        self._native = native
        self._cancel = _invoke(native.CancelHandle, deadline)
        self._execution = NativeExecution(self._cancel)
        self.diagnostics: Json = {'schema_version': 1, 'backend': 'rust', 'settings': asdict(config),
                                  'native_identity': self.identity, 'cleanup': 'pending',
                                  'accounting_policy': 'native-capacity-leases-and-owned-numpy-v1',
                                  'control_admission_policy': 'python-controls-64x-plus32mib-v1',
                                  'control_handoff': 'Python control results become caller-owned after operation retirement'}

    def _construct(self, plan: bytes, config: bytes, deadline: float) -> None:
        # This worker operation captures ONLY the private native handle, so a
        # constructor success/cancel race still has an explicit cleanup owner.
        # No locks, authorization or publication occur until the joined await.
        self._session = self._native.NativeSession(plan, config, deadline, self.wall_deadline, self._cancel)

    def __init__(self, plan: Json, config: PreparationConfig, *, deadline: float,
                 online_context: OnlinePPOContext | None = None, wall_deadline: float | None = None) -> None:
        self._setup(config, deadline, wall_deadline)
        try:
            self._check_mode(plan, online_context)
            _invoke(self._executor().call, self._construct, canonical(plan).encode(),
                    canonical(asdict(config)).encode(), deadline)
            self.summary = json.loads(_invoke(self._executor().call, self._session.control_summary))
            authorization = self._authorization(online_context)
            _invoke(self._executor().call, self._session.authorize, self._authorization_bytes(authorization))
        except BaseException as error:
            try:
                self.abort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve failure after mandatory cleanup
                error.add_note(f'native preparation cleanup failed: {cleanup_error}')
            raise

    @classmethod
    async def create_async(cls, plan: Json, config: PreparationConfig, *, deadline: float,
                           online_context: OnlinePPOContext | None = None, wall_deadline: float | None = None) -> Self:
        self = cls.__new__(cls)
        self._setup(config, deadline, wall_deadline)
        try:
            self._check_mode(plan, online_context)
            await self._acall(self._construct, canonical(plan).encode(),
                              canonical(asdict(config)).encode(), deadline)
            self.summary = json.loads(await self._acall(self._session.control_summary))
            authorization = await self._aauthorization(online_context)
            await self._acall(self._session.authorize, self._authorization_bytes(authorization))
        except BaseException as error:
            try:
                await self.aabort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve failure after mandatory cleanup
                error.add_note(f'native preparation cleanup failed: {cleanup_error}')
            raise
        return self

    @staticmethod
    def _check_mode(plan: Json, online_context: OnlinePPOContext | None) -> None:
        if (plan.get('kind') == 'online') != (online_context is not None):
            raise ContractError('native online preparation requires its live ownership context')

    def _executor(self) -> NativeExecution:
        if self._execution is None:
            raise ContractError('native execution has not initialized')
        return self._execution

    async def _acall(self, function: Any, *arguments: Any) -> Any:
        try:
            return await self._executor().acall(function, *arguments)
        except MemoryError as exc:
            raise ResourceLimit(str(exc)) from exc
        except ValueError as exc:
            raise ContractError(str(exc)) from exc

    def _ownership(self, online_context: OnlinePPOContext | None) -> None:
        # All lock and ownership operations run on the caller's main thread.
        for parent in sorted(self.summary['parents'], key=lambda item: item['directory']):
            path = Path(parent['directory']).resolve()
            token = _OWNED_LOCKS.get(path)
            if online_context is None:
                if parent['state']['status'] != 'complete':
                    raise ContractError('offline native preparation requires whole complete parents')
                if token is None:
                    token = self._locks.enter_context(run_lock(path)).token
            elif (path != online_context.parent_directory.resolve()
                  or token != online_context.ownership_token
                  or parent['state']['run_id'] != online_context.parent_run_id
                  or parent['state']['status'] != 'running'
                  or parent['state']['updates'] != online_context.generation):
                raise ContractError('live ownership does not authorize this native parent')
            if token is None:
                raise ContractError('native source parent is not owned')
            self._tokens[path] = token
        if online_context is not None:
            collections = self.summary['collections']
            if (len(self.summary['parents']) != 1 or len(collections) != 1
                    or collections[0]['batch_id'] != online_context.batch_id
                    or collections[0]['collection_generation'] != online_context.generation
                    or collections[0]['parent_run_id'] != online_context.parent_run_id
                    or collections[0]['collection_kind'] != 'ppo_batch'):
                raise ContractError('online ownership does not authorize this exact native batch')
        self._check_owned()

    def _authorization(self, online_context: OnlinePPOContext | None) -> Json:
        self._ownership(online_context)
        return {'control_fingerprint': self.summary['control_fingerprint'],
                'collections': list(_expected_collection_items(self.summary, self.check))}

    def _authorization_bytes(self, authorization: Json) -> bytes:
        # Native admitted summary and the full escaped recipe bound BEFORE this
        # tree/string/bytes allocation. Never widen an underestimated bound.
        bound = self.summary['authorization_wire_bound_bytes']
        if type(bound) is not int or bound < 0:
            raise ContractError('invalid native authorization bound')
        encoded = canonical(authorization).encode()
        if len(encoded) > bound:
            raise ResourceLimit('serialized authorization exceeds pre-admitted bound')
        return encoded

    async def _aauthorization(self, online_context: OnlinePPOContext | None) -> Json:
        self._ownership(online_context)
        collections = []
        for collection in _expected_collection_items(self.summary, self.check):
            collections.append(collection)
            await asyncio.sleep(0)
        return {'control_fingerprint': self.summary['control_fingerprint'], 'collections': collections}

    def _check_owned(self) -> None:
        if self._closed:
            raise ContractError('native preparation ownership is closed')
        if any(_OWNED_LOCKS.get(path) != token for path, token in self._tokens.items()):
            raise ContractError('native source ownership expired')

    def check(self, *, cleanup: bool = False) -> None:
        self._check_owned()
        if time.monotonic() >= (self.wall_deadline if cleanup else self.deadline):
            raise TimeoutError('native preparation deadline')

    def hash_file(self, path: Path) -> str:
        self.check()
        value: str = _invoke(self._executor().call, self._session.source_hash, str(path.resolve()))
        return value

    async def ahash_file(self, path: Path) -> str:
        self.check()
        value: str = await self._acall(self._session.source_hash, str(path.resolve()))
        return value

    async def areserve_staging(self, memory_bytes: int, scratch_bytes: int) -> Any:
        self.check()
        return await self._acall(self._session.reserve_staging, memory_bytes, scratch_bytes)

    def reserve_staging(self, memory_bytes: int, scratch_bytes: int) -> Any:
        self.check()
        return _invoke(self._executor().call, self._session.reserve_staging, memory_bytes, scratch_bytes)

    def export_staging(self, path: str, offset: int, rows: int, columns: int) -> Any:
        self.check()
        return _invoke(self._executor().call, self._session.export_staging, path, offset, rows, columns)

    def _batch(self, batch: Any) -> Any:
        self.exhausted = batch is None
        if batch is None:
            return None
        *arrays, proof = batch
        if proof is not None:
            self._proof = json.loads(proof)
        return NativeBatch(*arrays)

    def next_batch(self, retained_python_bytes: int, consumer_layout: tuple[int, int, int, int]) -> Any:
        self.check()
        return self._batch(_invoke(self._executor().call, self._session.next_batch,
                                  retained_python_bytes, consumer_layout))

    async def anext_batch(self, retained_python_bytes: int, consumer_layout: tuple[int, int, int, int]) -> Any:
        self.check()
        return self._batch(await self._acall(self._session.next_batch, retained_python_bytes,
                                            consumer_layout))

    def proof(self) -> Json:
        self.check()
        if self._proof is None:
            raise ContractError('native sources have not passed global validation')
        return self._proof

    def cancel(self) -> None:
        self._executor().cancel()

    def _retire(self) -> None:
        try:
            if self._session is not None and callable(getattr(self._session, 'release_controls', None)):
                _invoke(self._executor().call, self._session.release_controls)
        finally:
            try:
                self._executor().close()
            finally:
                self._closed = True
                self._locks.close()

    async def _aretire(self) -> None:
        try:
            if self._session is not None and callable(getattr(self._session, 'release_controls', None)):
                await self._acall(self._session.release_controls)
        finally:
            try:
                await self._executor().aclose()
            finally:
                self._closed = True
                self._locks.close()

    def _metrics(self, status: str) -> None:
        self.diagnostics.update(cleanup=status, elapsed_seconds=time.monotonic() - self.started)
        if self._session is not None:
            self.diagnostics['managed_peak_bytes'] = _invoke(self._executor().call, self._session.peak_bytes)
            if callable(getattr(self._session, 'telemetry', None)):
                self.diagnostics.update(json.loads(_invoke(self._executor().call, self._session.telemetry)))

    async def _ametrics(self, status: str) -> None:
        self.diagnostics.update(cleanup=status, elapsed_seconds=time.monotonic() - self.started)
        if self._session is not None:
            self.diagnostics['managed_peak_bytes'] = await self._acall(self._session.peak_bytes)
            if callable(getattr(self._session, 'telemetry', None)):
                self.diagnostics.update(json.loads(await self._acall(self._session.telemetry)))

    def finalize(self) -> Json:
        self._check_owned()
        original: BaseException | None = None
        try:
            self.check(cleanup=True)
            result: Json = json.loads(_invoke(self._executor().call, self._session.finalize))
            if self._proof is not None and result != self._proof:
                raise ContractError('native final proof differs from admitted source proof')
            self._metrics('complete')
            return result
        except BaseException as error:
            original = error
            try:
                _invoke(self._executor().call, self._session.close)
                self._metrics('aborted')
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve failure after mandatory cleanup
                error.add_note(f'native finalization cleanup failed: {cleanup_error}')
            raise
        finally:
            try:
                self._retire()
            except BaseException as cleanup_error:
                if original is None:
                    raise
                original.add_note(f'native executor cleanup failed: {cleanup_error}')

    async def afinalize(self) -> Json:
        self._check_owned()
        original: BaseException | None = None
        try:
            self.check(cleanup=True)
            result: Json = json.loads(await self._acall(self._session.finalize))
            if self._proof is not None and result != self._proof:
                raise ContractError('native final proof differs from admitted source proof')
            await self._ametrics('complete')
            return result
        except BaseException as error:
            original = error
            try:
                await self._acall(self._session.close)
                await self._ametrics('aborted')
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve failure after mandatory cleanup
                error.add_note(f'native finalization cleanup failed: {cleanup_error}')
            raise
        finally:
            try:
                await self._aretire()
            except BaseException as cleanup_error:
                if original is None:
                    raise
                original.add_note(f'native executor cleanup failed: {cleanup_error}')

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self.cancel()
            if self._session is not None:
                _invoke(self._executor().call, self._session.close)
            self._metrics('aborted')
        finally:
            self._retire()

    async def aabort(self) -> None:
        if self._closed:
            return
        try:
            self.cancel()
            if self._session is not None:
                await self._acall(self._session.close)
            await self._ametrics('aborted')
        finally:
            await self._aretire()

    def __enter__(self) -> Self:
        self._check_owned()
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        if not self._closed:
            try:
                self.finalize() if kind is None else self.abort()
            except BaseException as cleanup_error:
                if not isinstance(value, BaseException):
                    raise
                value.add_note(f'native preparation cleanup failed: {cleanup_error}')

    async def __aenter__(self) -> Self:
        self._check_owned()
        return self

    async def __aexit__(self, kind: object, value: object, traceback: object) -> None:
        if not self._closed:
            try:
                if kind is None:
                    await self.afinalize()
                else:
                    await self.aabort()
            except BaseException as cleanup_error:
                if not isinstance(value, BaseException):
                    raise
                value.add_note(f'native preparation cleanup failed: {cleanup_error}')
