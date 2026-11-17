"""Parent-only bounded PPO orchestration, commit ordering and pause/resume."""
from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import resource
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import (
    REVIEWED_SPEC_HASHES,
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    digest,
)
from gf_models.arrays import sha256
from gf_models.config import (
    GRUModelConfig,
    MLPModelConfig,
    PPOTrainingConfig,
    SequencePPOTrainingConfig,
    TransformerModelConfig,
)
from gf_models.torch_runtime import NUMERICS, runtime_metadata
from gf_training.collection import (
    CollectionPlan,
    FinalizedCollection,
    OnlinePPOContext,
    training_groups,
)
from gf_training.journal import atomic_json
from gf_training.lineage import (
    checkpoint_path,
    commit_generation,
    owned_binding,
    read_json,
    retain,
    run_lock,
    validate_chain,
)
from gf_training.native_ppo import preprocessing_identity, stage_rollout
from gf_training.ppo import ingest_collection, write_rollout

Collector = Callable[[CollectionPlan], Awaitable[FinalizedCollection]]


def source_identity(deadline: float = float("inf")) -> Json:
    if time.monotonic() >= deadline:
        raise TimeoutError('PPO source identity deadline')
    package_names = ('gf_common', 'gf_models', 'gf_training', 'gf_cli',
                     'gf_evaluation', 'gf_simulator_client', 'godfieldbots')
    package_roots = {}
    for name in package_names:
        if time.monotonic() >= deadline:
            raise TimeoutError('PPO source discovery deadline')
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).name != '__init__.py':
            raise ContractError('owned Bots package source is unavailable')
        package_roots[name] = Path(spec.origin).resolve().parent
    module_path = Path(__file__).resolve()
    root = module_path.parent.parent.parent
    checkout = ((root / '.git').exists()
                and (root / 'pyproject.toml').is_file()
                and module_path == (root / 'src/gf_training/ppo_run.py').resolve()
                and all(path == (root / 'src' / name).resolve() for name, path in package_roots.items()))
    revision, dirty = None, None
    if checkout:
        try:
            revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=max(.001, min(5., deadline - time.monotonic())), check=True).stdout.strip()
            if time.monotonic() >= deadline:
                raise TimeoutError('source Git provenance deadline')
            dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, timeout=max(.001, min(5., deadline - time.monotonic())), check=True).stdout)
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
    hashes = {}
    for name, package_root in sorted(package_roots.items()):
        for path in sorted(package_root.rglob('*.py')):
            if time.monotonic() >= deadline:
                raise TimeoutError('PPO source hashing deadline')
            hashes[f'{name}/{path.relative_to(package_root).as_posix()}'] = sha256(path, deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError('PPO source identity deadline')
    native_distributions = {}
    for distribution in importlib.metadata.distributions():
        if time.monotonic() >= deadline:
            raise TimeoutError('native distribution identity deadline')
        name = str(distribution.metadata.get('Name', '')).lower().replace('_', '-')
        if name.startswith('nvidia-') or name in {'cuda-bindings', 'cuda-toolkit', 'triton'}:
            native_distributions[name] = distribution.version
    return {'code': {'revision': revision, 'dirty': dirty, 'installed_build_revision': None}, 'module_sha256': hashes,
            'cuda_distributions': dict(sorted(native_distributions.items())),
            'runtime': runtime_metadata(), 'numerics': NUMERICS,
            'package': importlib.metadata.version('godfieldbots')}


def integration_identity(config: Any, deadline: float) -> Json:
    result: Json = {'binary_sha256': sha256(Path(config.server.binary), deadline),
                    'server_config_sha256': sha256(Path(config.server.config), deadline),
                    'declared_simulator_revision': config.server.simulator_revision,
                    'binary_build_revision': None, 'source_revision': None, 'source_spec_sha256': None}
    if config.server.source_dir is not None:
        source = Path(config.server.source_dir)
        if time.monotonic() >= deadline:
            raise TimeoutError('Simulator source provenance deadline')
        actual = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                                text=True, timeout=max(.001, min(5., deadline - time.monotonic()))).stdout.strip()
        if actual != config.server.simulator_revision:
            raise ContractError('Simulator checkout revision does not match accepted pin')
        hashes = {name: sha256(source / name, deadline) for name in REVIEWED_SPEC_HASHES}
        if hashes != REVIEWED_SPEC_HASHES:
            raise ContractError('Simulator normative source hashes differ from accepted contract')
        result.update(source_revision=actual, source_spec_sha256=hashes)
    return result


def owned_bytes(directory: Path, deadline: float = float("inf"), *,
                external_roots: tuple[Path, ...] = ()) -> int:
    count = 0
    seen: set[tuple[int, int]] = set()
    pending = [directory, *external_roots]
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError('owned storage measurement deadline')
        path = pending.pop()
        if not path.is_absolute():
            path = path.absolute()
        # Do not follow symlinks in external roots or their ancestry.
        for ancestor in (path, *path.parents):
            if ancestor.is_symlink():
                raise ContractError('symlink in owned PPO storage')
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)
        elif stat.S_ISREG(info.st_mode):
            count += info.st_size
        else:
            raise ContractError('nonregular owned PPO storage entry')
    return count


class LearningProcess:
    def __init__(self, directory: Path, max_log_bytes: int) -> None:
        self.directory, self.limit = directory, max_log_bytes
        self.process: Any = None
        self.drainer: asyncio.Task[None] | None = None
        self.health: Json = {}
        self.close_task: asyncio.Task[None] | None = None
        self.creation_task: asyncio.Task[Any] | None = None
        self._creation_no_child = False
        self._process_notification_attempted = False
        self._drainer_creation_attempted = False
        self._creation_observer_failure: BaseException | None = None

    def _environment(self) -> dict[str, str]:
        return {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}

    def _process_created(self) -> None:
        pass

    def _creation_task_created(self, creation: asyncio.Task[Any]) -> None:
        """Optional observation of the original local creation task."""

    def _creation_failed(self, error: Exception) -> None:
        """Optional observation of the default creator's actual exception."""

    def _creation_task_failed(self, error: BaseException) -> None:
        """Optional observation after closing an actually unstarted coroutine."""

    def _close_request(self) -> Json:
        return {'command': 'close'}

    def _validate_close(self, result: Json) -> None:
        if result != {'closed': True}:
            raise ContractError('invalid learner close acknowledgement')

    def _launch_arguments(self) -> tuple[str,...]:
        """Private launch extension; frozen/non-HPO creators retain exact argv."""
        return ()

    async def _create_process(self) -> Any:
        arguments=self._launch_arguments()  # Validation has no stdlib no-child proof.
        environment = self._environment()
        try:
            return await asyncio.create_subprocess_exec(sys.executable, '-E', '-m', 'gf_training.ppo_worker', *arguments,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=environment, limit=1024 * 1024)
        except Exception as error:
            # The default stdlib creation operation failed without returning a
            # process. Alternate launchers must supply their own settlement
            # evidence; cancellation alone never establishes no child.
            self._creation_no_child = True
            try:
                self._creation_failed(error)
            except BaseException as observer_error:  # noqa: BLE001 - retain the actual creation failure
                self._creation_observer_failure = observer_error
                error.add_note('creation failure observation: ' + type(observer_error).__name__)
            raise

    def _adopt_process(self, process: Any) -> None:
        if process is None:
            raise ContractError('actual numerical process required for adoption')
        if self.process is None:
            self.process = process
        elif self.process is not process:
            raise ContractError('cannot replace an owned numerical process')
        error: BaseException | None = None
        if not self._process_notification_attempted:
            self._process_notification_attempted = True
            try:
                self._process_created()
            except BaseException as caught:  # noqa: BLE001 - retain process through partial registration
                error = caught
        if not self._drainer_creation_attempted:
            self._drainer_creation_attempted = True
            coroutine = None
            try:
                coroutine = self._drain()
                self.drainer = asyncio.create_task(coroutine)
            except BaseException as caught:  # noqa: BLE001 - no unowned coroutine on failed task creation
                if coroutine is not None:
                    coroutine.close()
                if error is None:
                    error = caught
                else:
                    error.add_note('learner drainer creation: ' + type(caught).__name__)
        if error is not None:
            raise error

    async def start(self) -> None:
        coroutine = self._create_process()
        try:
            creation = asyncio.create_task(coroutine)
        except BaseException as error:
            unstarted = inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
            coroutine.close()
            self._creation_no_child = True
            if unstarted:
                try:
                    self._creation_task_failed(error)
                except BaseException as callback_error:  # noqa: BLE001 - retain the actual task failure
                    self._creation_observer_failure = callback_error
                    error.add_note('creation task failure observation: ' + type(callback_error).__name__)
            raise
        self.creation_task = creation
        observer_error: BaseException | None = None
        try:
            self._creation_task_created(creation)
        except BaseException as error:  # noqa: BLE001 - defer until LOCAL creation settles/adopts
            self._creation_observer_failure = error
            observer_error = error
        cancelled = None
        while not creation.done():
            try:
                await asyncio.shield(creation)
            except asyncio.CancelledError as error:
                cancelled = cancelled or error
            except BaseException as error:
                if observer_error is not None:
                    error.add_note('creation task observation: ' + type(observer_error).__name__)
                raise
        try:
            self._adopt_process(creation.result())
        except BaseException as error:
            if observer_error is not None:
                error.add_note('creation task observation: ' + type(observer_error).__name__)
            raise
        if cancelled:
            if observer_error is not None:
                cancelled.add_note('creation task observation: ' + type(observer_error).__name__)
            raise cancelled
        if observer_error is not None:
            raise observer_error

    async def _drain(self) -> None:
        count = 0
        with (self.directory / f'learner-{uuid.uuid4().hex[:12]}.log').open('x') as output:
            while chunk := await self.process.stderr.read(65536):
                output.write(chunk[:max(0, self.limit - count)].decode('utf-8', errors='replace'))
                count += len(chunk)
                parser = getattr(self, 'diagnostic_parser', None)
                if parser is not None:
                    try:
                        parser.feed(chunk)
                    except Exception:  # noqa: BLE001, S110 - keep original raw drain/limits intact
                        pass
                if count > self.limit:
                    self.health['log_limit_exceeded'] = True
        self.health['stderr_bytes'] = count

    async def request(self, request: Json, deadline: float) -> Json:
        if time.monotonic() >= deadline or self.health.get('log_limit_exceeded'):
            raise TimeoutError('PPO learner request deadline/log limit')
        async with asyncio.timeout_at(deadline):
            self.process.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
        if not line or self.health.get('log_limit_exceeded'):
            raise RuntimeError('PPO numerical child failed; inspect bounded learner log')
        result = json.loads(line)
        if not isinstance(result, dict):
            raise ContractError('malformed numerical acknowledgement')
        return result

    async def _close(self, deadline: float) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.returncode is None:
                result = await self.request(self._close_request(), deadline)
                self._validate_close(result)
            async with asyncio.timeout_at(deadline):
                await process.wait()
                if self.drainer:
                    await self.drainer
            if process.returncode != 0 or self.health.get('log_limit_exceeded'):
                raise RuntimeError('learner close/health failure')
        except BaseException as error:
            self.health['cleanup_failure'] = str(error)
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), .5)
                except TimeoutError:
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), 1.)
                    except TimeoutError:
                        # Record bounded reap failure; ownership survives delayed OS exit.
                        self.health['reap_failure'] = 'mandatory final join after kill grace'
                        await process.wait()
            if self.drainer:
                try:
                    await asyncio.wait_for(asyncio.shield(self.drainer), .5)
                except TimeoutError:
                    self.health['drain_failure'] = 'pipe EOF grace expired'
                    self.drainer.cancel()
                    await asyncio.gather(self.drainer, return_exceptions=True)
            raise
        finally:
            if process.stdin is not None:
                process.stdin.close()
                try:
                    await asyncio.wait_for(process.stdin.wait_closed(), .5)
                except (TimeoutError, BrokenPipeError, ConnectionResetError):
                    self.health['stdin_close_failure'] = True
            self.health['exit_code'] = process.returncode
            if time.monotonic() >= deadline:
                self.health['cleanup_overrun'] = True

    async def close(self, deadline: float) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close(deadline))
        cancellation = None
        while not self.close_task.done():
            try:
                await asyncio.shield(self.close_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException:  # noqa: BLE001 - task.result preserves cleanup error
                break
        self.close_task.result()
        if cancellation:
            raise cancellation


_CLOCK_BRACKET_SECONDS = .025
_CLOCK_SAMPLE_ATTEMPTS = 3
_CLOCK_ROLLBACK_SECONDS = .1
type _ClockBracket = tuple[float, float, float]


def _precise_clock_bracket(before: float, wall: float, after: float) -> bool:
    return (all(type(value) in (int, float) and math.isfinite(value) for value in (before, wall, after))
        and 0 <= after - before <= _CLOCK_BRACKET_SECONDS)


def _validate_clock_anchor(before: float, wall: float, after: float) -> None:
    if not _precise_clock_bracket(before, wall, after):
        raise ContractError('PPO initial clock observation uncertain')


def _checked_wall_clock(anchor: _ClockBracket, *, scope: str) -> float:
    """Bound net wall-offset loss using observations, not scheduler read gaps."""
    _validate_clock_anchor(*anchor)  # The captured original anchor is never retried/replaced.
    m0, w0, m1 = anchor
    frontier = m1
    for _attempt in range(_CLOCK_SAMPLE_ATTEMPTS):
        before = time.monotonic()
        wall = time.time()
        after = time.monotonic()
        previous_frontier = frontier
        # Invalid brackets still observed time: a retry cannot precede either
        # finite monotonic read, including one from a decreasing bracket.
        for observed in (before, after):
            if type(observed) in (int, float) and math.isfinite(observed):
                frontier = max(frontier, observed)
        if not _precise_clock_bracket(before, wall, after) or before < previous_frontier:
            continue
        rollback_min = (w0 - m1) - (wall - before)
        rollback_max = (w0 - m0) - (wall - after)
        if not math.isfinite(rollback_min) or not math.isfinite(rollback_max):
            continue
        if rollback_min > _CLOCK_ROLLBACK_SECONDS:
            raise ContractError('wall clock rollback during PPO ' + scope)
        if rollback_max <= _CLOCK_ROLLBACK_SECONDS:
            return wall
    raise ContractError('PPO clock observation uncertain during ' + scope)


def _controls(rng: Any, pool: list[int], collection_position: int) -> Json:
    return {'schedule_opponent_rng': rng.bit_generator.state, 'snapshot_pool': pool, 'collection_position': collection_position}


async def train_ppo(loaded: Any, collector: Collector, directory: Path, *, mode: str = 'fresh', pause_after_update: int | None = None) -> Json:
    from gf_common.action_contract import require_qualified_contract
    from gf_common.structured import ActionConfigV2
    if isinstance(loaded.config.actions, ActionConfigV2):
        require_qualified_contract(loaded.config.actions.action_contract)
    return await _run_ppo(loaded, collector, directory, mode=mode, pause_after_update=pause_after_update)


async def _run_ppo(loaded: Any, collector: Any, directory: Path, *, mode: str = 'fresh',
                   pause_after_update: int | None = None, _structured_component: bool = False,
                   _structured_ops: Any = None, _game_ops: Any = None) -> Json:
    operation_started = time.monotonic()
    operation_unix = time.time()
    operation_monotonic_after = time.monotonic()
    clock_anchor = (operation_started, operation_unix, operation_monotonic_after)
    config = loaded.config
    ops: Any = None
    game = _game_ops is not None
    hpo = False
    owned = False
    if game:
        from gf_training import structured_ppo
        from gf_training.card_instance_ppo import _GameOperations
        from gf_training.hpo_owned_run import _OwnedHpoOperations
        from gf_training.hpo_ppo import _HpoOperations
        owned=type(_game_ops) is _OwnedHpoOperations
        hpo=type(_game_ops) in (_HpoOperations,_OwnedHpoOperations)
        if type(_game_ops) not in (_GameOperations,_HpoOperations,_OwnedHpoOperations) or _structured_component or _structured_ops is not None:
            raise ContractError("closed corrected game operations required")
        ops = _game_ops
    if _structured_ops is not None and not _structured_component:
        raise ContractError('private native operations require structured component composition')
    if _structured_component:
        from gf_training import structured_ppo
        if _structured_ops is not None:
            if type(_structured_ops) is not structured_ppo._NativeComponentOps:
                raise ContractError('concrete native structured operations required')
            if config.preprocessing.backend != 'rust' or config.preprocessing.reader != 'structured-v1':
                raise ContractError('native component operations require explicit rust/structured-v1')
        if _structured_ops is None and config.preprocessing.backend == 'rust':
            structured_ppo.validate_config(config)
            from gf_training.native_backend import load_native, reader_settings
            selected_native = load_native()
            reader_settings(config.preprocessing, selected_native)  # Real accepted marker/signature only.
            ops = structured_ppo._NativeComponentOps(selected_native)
            ops._implementation = 'native-structured-v2'
        else:
            ops = _structured_ops if _structured_ops is not None else structured_ppo
    with (ops.owned_invocation() if owned else nullcontext()):
        def measure_owned(deadline: float) -> int:
            if game and (state['schema_version'] == 4 or hpo and 'storage_binding' in state):
                return owned_bytes(directory, deadline, external_roots=ops.external_roots(state))
            return owned_bytes(directory, deadline)

        formal = ops.formal if game else None
        if formal is not None:
            formal.preflight(loaded)
        model, training = ops.validate_config(config) if ops else config.model, config.training
        if not ops and (not isinstance(model, (MLPModelConfig, GRUModelConfig, TransformerModelConfig)) or not isinstance(training, PPOTrainingConfig) or model.checkpoint is not None or config.run.split != 'train'):
            raise ContractError('PPO requires fresh supported model recipe and train split')
        if not ops and isinstance(model, (GRUModelConfig, TransformerModelConfig)) != isinstance(training, SequencePPOTrainingConfig):
            raise ContractError("PPO model/sequence training mismatch")
        if config.server.external_url is not None:
            raise ContractError('PPO requires owned distinct-seed Simulator groups')
        if not hpo and not (config.run.wall_seconds <= 7200 and config.run.admission_seconds <= 5400 and config.run.execution_seconds <= 7140 and config.run.finalization_seconds >= 60):
            raise ContractError('PPO original deadline caps exceeded')
        if mode not in ('fresh', 'resume'):
            raise ContractError('unsupported PPO mode')
        if mode == 'fresh' and pause_after_update is not None and (type(pause_after_update) is not int or not 0 < pause_after_update <= training.max_updates):
            raise ContractError('invalid absolute pause target')
        preprocessing_options: Json = {'model_id': model.model_id} if isinstance(model, (GRUModelConfig, TransformerModelConfig)) else {}
        preprocessing = ((ops.preprocessing_identity(config, deadline=operation_started + config.run.execution_seconds) if ops else preprocessing_identity(config.preprocessing, deadline=operation_started + config.run.execution_seconds, **preprocessing_options))
                         if mode == 'fresh' else None)
        directory = directory.resolve()
        if game and mode == "fresh":
            ops.preflight(config, deadline=operation_started + config.run.execution_seconds)
        with run_lock(directory, create=mode == 'fresh') as ownership, ExitStack() as cold_scope:
            initial_scan = None
            if owned and mode == 'resume':
                initial_state, cold_records=ops.read_cold(ownership,config,deadline=operation_started+config.run.execution_seconds)
                cold_scope.push(ops.cold_import._scope_exit)
                identity_deadline=operation_started+ops.saved_deadline(initial_state)-operation_unix
            elif ops and mode == 'resume':
                initial_state, initial_scan = ops.read_control(directory / 'run.json', config,
                    deadline=operation_started + config.run.execution_seconds)
                identity_deadline = operation_started + ops.saved_deadline(initial_state) - operation_unix
                del initial_state
            else:
                identity_deadline = operation_started + config.run.execution_seconds if mode == 'fresh' else operation_started + read_json(directory / 'run.json')['execution_deadline_unix'] - operation_unix
            if preprocessing is None:
                preprocessing = (ops.preprocessing_identity(config, deadline=identity_deadline) if ops else preprocessing_identity(config.preprocessing, deadline=identity_deadline, **preprocessing_options))
            identity = source_identity(identity_deadline)
            identity["integration"] = ops.integration_identity(config, identity_deadline) if game else integration_identity(config, identity_deadline)
            identity["configuration_inputs_sha256"] = digest(loaded.inputs)
            identity["preprocessing"] = preprocessing
            if ops:
                ops.decorate_identity(identity)
            if formal is not None:
                formal.attach_identity(directory, identity, ops.execution(config), mode=mode, deadline=identity_deadline, ownership=ownership)
            invocation_started = operation_started
            now = operation_unix
            if mode == 'fresh':
                for name in ('generations', 'updates', 'artifacts', 'collections', 'numeric'):
                    (directory / name).mkdir()
                state = {'schema_version': 1, 'algorithm': 'ppo', 'run_id': config.run.name if hpo else str(uuid.uuid4()), 'status': 'running',
                         'config_sha256': loaded.sha256, 'effective_config': loaded.effective, 'config_inputs': list(loaded.inputs), 'identity': identity,
                         'seed': training.seeds[0], 'device': training.device, 'created_unix': now, 'last_unix': now,
                         'wall_deadline_unix': now + config.run.wall_seconds,
                         'admission_deadline_unix': now + config.run.admission_seconds,
                         'execution_deadline_unix': now + config.run.execution_seconds,
                         'updates': 0, 'accepted_decisions': 0, 'optimizer_steps': 0, 'generation_sha256': None,
                         'collection_seconds': 0., 'update_seconds': 0., 'invocations': [], 'completed_updates': None}
                if ops:
                    ops.initialize_state(state, config)
                if ops:
                    ops.persist_control(directory / 'artifact-registry.json', ({'schema_version':ops.registry_schema,'generations':{},'evaluation_references':{}}
                        | ({'cache_references':{}} if ops.registry_schema in (5,6) else {}) if hpo else
                        {'schema_version': 3 if game else 2, 'generations': {}}), config, deadline=identity_deadline)
                    ops.persist_control(directory / 'run.json', state, config, deadline=identity_deadline)
                else:
                    atomic_json(directory / 'artifact-registry.json', {})
                    atomic_json(directory / 'run.json', state)
                rng = np.random.Generator(np.random.PCG64(training.seeds[0]))
                pool = [0]
            else:
                state, records = ((initial_state,cold_records) if owned else ops.validate_parent(directory, config, identity_deadline,
                    extra_owned_bytes=initial_scan.retained if initial_scan is not None else 0)
                    if ops else validate_chain(directory, require_complete=False, deadline=identity_deadline))
                if initial_scan is not None:
                    initial_scan.finalize()
                if state['status'] != 'paused' or state['updates'] == 0:
                    raise ContractError('resume requires an explicitly paused committed PPO run')
                if state['config_sha256'] != loaded.sha256 or state['identity'] != identity or state['device'] != training.device:
                    raise ContractError('resume code/runtime/config/device mismatch')
                if ops and canonical(state['execution']) != canonical(ops.execution(config)):
                    raise ContractError('resume execution binding differs from current Config')
                if now < state['last_unix'] or now >= state['wall_deadline_unix'] or now >= state['execution_deadline_unix']:
                    raise ContractError('resume clock rollback or original deadline expired')
                controls = records[-1]['parent_controls']
                rng = np.random.Generator(np.random.PCG64())
                rng.bit_generator.state = controls['schedule_opponent_rng']
                pool = controls['snapshot_pool']
                if controls['collection_position'] != state['updates'] or not hpo and pool != [number for number in range(state['updates'] + 1) if number % 10 == 0][-5:]:
                    raise ContractError('resume collection position/pool mismatch')
            if pause_after_update is not None and (type(pause_after_update) is not int or pause_after_update <= state['updates'] or pause_after_update > training.max_updates):
                raise ContractError('pause target must exceed current absolute committed-update count')
            remaining = state['wall_deadline_unix'] - now
            wall = ((invocation_started + state['wall_deadline_unix']) - now) if formal is not None else invocation_started + remaining
            execution = invocation_started + state['execution_deadline_unix'] - now
            admission = invocation_started + state['admission_deadline_unix'] - now
            # Even initialization and source-identity work consume the original clock.
            state['status'] = 'running'
            if hpo: state['stop_reason']=None
            if ops:
                ops.persist_control(directory / 'run.json', state, config, deadline=execution, original=initial_scan)
                initial_scan = None
            else:
                atomic_json(directory / 'run.json', state)
            ledger = None
            if game:
                ops.bind_run(ownership, state)
            if ops:
                if owned:
                    try:
                        ops.consume_prechild()
                        child, ledger = ops.make_child(directory,config,state,deadline=execution,cleanup_deadline=wall)
                    except BaseException as startup_error:
                        await ops.abort_startup(startup_error,state,config,deadline=min(wall,time.monotonic()+training.cleanup_seconds))
                        raise
                elif hpo and config.hpo_spec['schema_version'] in (3,4):
                    child, ledger = ops.make_child(directory, config, state, deadline=execution,cleanup_deadline=wall)
                else:
                    child, ledger = ops.make_child(directory, config, state, deadline=execution)
            else:
                child = LearningProcess(directory, training.max_log_bytes)
            error: BaseException | None = None
            final_status = 'complete'
            def check_clock(*, scope: str = 'invocation') -> float:
                checked = _checked_wall_clock(clock_anchor, scope=scope)
                state['last_unix'] = max(state['last_unix'], checked)
                return state['last_unix']
            try:
                _validate_clock_anchor(*clock_anchor)
                if owned: await ops.start_owned(child)
                else: await child.start()
                request = {'command': 'initialize', 'model': asdict(model), 'seed': training.seeds[0], 'device': training.device,
                           'output': str(checkpoint_path(directory, 0)), 'action_hash': digest(asdict(config.actions))}
                if isinstance(training, SequencePPOTrainingConfig):
                    request['sequence_length'] = training.sequence_length
                if mode == 'resume':
                    request['restore'] = str(checkpoint_path(directory, state['updates']))
                initial = (await ops.initialize_owned(child,training.seeds[0],checkpoint_path(directory,0),deadline=execution) if owned else
                    await child.initialize(training.seeds[0], checkpoint_path(directory, 0),
                    restore=checkpoint_path(directory, state['updates']) if mode == 'resume' else None, deadline=execution)
                    if ops else await child.request(request, execution))
                if mode == 'fresh':
                    state['hardware'] = initial['hardware']
                    if ops:
                        ops.commit(directory, state, config, generation=0, update=None, controls=ops.controls(rng,pool,0) if hpo else _controls(rng, pool, 0), acknowledgement=initial, deadline=execution)
                    else:
                        commit_generation(directory, state, generation=0, update=None, controls=_controls(rng, pool, 0), deadline=execution)
                elif initial['hardware'] != state['hardware']:
                    raise ContractError('resume hardware identity mismatch')
                while True:
                    check_clock()
                    usage = measure_owned(execution)
                    if hpo:
                        if ops.stop(state,config,usage,time.monotonic(),execution,admission):
                            final_status='complete' if state['used_decisions']==config.hpo_spec['decision_budget'] else 'incomplete'
                            break
                    elif formal is not None:
                        observed = time.monotonic()
                        if formal.observe_stop(state, operation_unix, invocation_started, observed, usage):
                            break
                    else:
                        if usage > training.max_owned_bytes or time.monotonic() >= execution:
                            raise RuntimeError('PPO actual execution/storage limit exceeded')
                        if state['updates'] >= training.max_updates or time.monotonic() >= admission or usage + training.admission_reservation_bytes > training.max_owned_bytes:
                            if not state['updates']:
                                raise ContractError('zero-update PPO run ineligible')
                            break
                    with ExitStack() as generation_scope:
                        generation = state['updates']
                        registry = ({'random-card-instance-v2': RuntimePolicyBinding('random'), 'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')} if game else
                                    {'random-structured-v1': RuntimePolicyBinding('random'), 'heuristic-structured-v1': RuntimePolicyBinding('heuristic')}
                                    if ops else {'random': RuntimePolicyBinding('random'), 'heuristic': RuntimePolicyBinding('heuristic')})
                        learner_label, learner_binding = ops.binding(directory, generation, config, execution) if ops else owned_binding(directory, generation)
                        registry[learner_label] = learner_binding
                        mixed = None
                        if game:
                            from gf_common.training_opponents import (
                                refresh_opponent,
                                training_opponents,
                            )
                            mixed = training_opponents(loaded.effective)
                            if mixed is not None:
                                refresh_label, refresh_binding = refresh_opponent(config.actions, mixed)
                                registry[refresh_label] = refresh_binding
                        snapshots = []
                        for snapshot in pool:
                            label, binding = ops.binding(directory, snapshot, config, execution) if ops else owned_binding(directory, snapshot)
                            registry[label] = binding
                            snapshots.append(label)
                        if game and not hpo and mixed is not None:
                            from gf_training.collection import training_groups_refresh
                            groups = training_groups_refresh(training.seeds[0], generation, learner_label,
                                tuple(snapshots), rng, actions=config.actions, training_opponents=mixed)
                        else:
                            groups = (ops.groups(config,generation,learner_label,tuple(snapshots),rng) if hpo else
                                      training_groups(training.seeds[0], generation, learner_label, tuple(snapshots), rng, actions_version=3 if game else 2)
                                      if ops else training_groups(training.seeds[0], generation, learner_label, tuple(snapshots), rng))
                        batch_id = f'batch-{generation:06}'
                        linkage = {'parent_run_id': state['run_id'], 'parent_run_path': str(directory), 'batch_id': batch_id,
                                   'collection_generation': generation,
                                   'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes}}
                        if game:
                            linkage = ops.linkage(state, directory, generation, groups, registry)
                        if hpo: ops.phase('collect', transient_games=0, transient_learner_decisions=None, transient_total_game_actions=0)
                        collection_started = time.monotonic()
                        plan = CollectionPlan(groups, registry, directory / 'collections' / batch_id, linkage, execution, wall)
                        if ops:
                            assert ledger is not None
                            with ops.precollection_scope(plan, ledger, config=config, deadline=execution) if hpo else nullcontext():
                                ops.validate_owned_collection(plan, ledger, config=config, deadline=execution)
                                collection = await collector(plan, ledger)
                            if hpo and getattr(ops, 'resident_runtime', None) is not None:
                                ops.resident_runtime.pool.authority.require(ledger)
                            elif ledger.workers or ledger.updating:
                                raise ContractError('structured collection returned unreaped participants or active update')
                        else:
                            collection = await collector(plan)
                        if collection.directory.resolve() != (directory / 'collections' / batch_id).resolve() or collection.report.get('comparison_valid') is not True:
                            raise ContractError('collection callback did not finalize requested original batch')
                        if game:
                            ops.accept_collection(collection, config)
                        check_clock()
                        from gf_common.schedule import serialize_schedule
                        collection_scan = None
                        if owned:
                            collection_manifest=collection.manifest
                        elif ops:
                            collection_manifest, collection_scan = ops.read_control(collection.directory / 'manifest.json', config, deadline=execution)
                            if game:
                                generation_scope.enter_context(collection_scan)
                        else:
                            collection_manifest = read_json(collection.directory / 'manifest.json')
                        expected_bindings = {key: asdict(value) for key, value in registry.items()}
                        bindings_match = (canonical(collection_manifest.get('policy_bindings')) == canonical(expected_bindings)
                                          if ops else collection_manifest.get('policy_bindings') == expected_bindings)
                        if collection_manifest.get('schedule') != serialize_schedule(groups, 4) or not bindings_match:
                            raise ContractError('collection callback changed frozen schedule/policies')
                        collection_seconds = time.monotonic() - collection_started
                        if hpo: ops.phase('source')
                        numeric = directory / 'numeric' / batch_id
                        numeric_files = None
                        with ExitStack() as source_scope:
                            if owned:
                                decision_count,proof,numeric_files,issued_projection=generation_scope.enter_context(
                                    ops.project_owned(collection,plan,numeric,config,execution))
                                if decision_count==0:
                                    ops.stop_reason='empty_collection'
                                    final_status='incomplete'
                                    break
                            elif ops:
                                assert collection_scan is not None
                                projection_options: dict[str, Any] = {
                                    'expected_manifest_sha256': collection_scan.verified_digest(collection.directory / 'manifest.json'),
                                    'extra_owned_bytes': collection_scan.retained,
                                }
                                if type(ops) is structured_ppo._NativeComponentOps:
                                    # Actual remaining owned storage bounds the private
                                    # spool. The collection-admission reserve is separate.
                                    projection_options['scratch_allowance_bytes'] = max(
                                        0, training.max_owned_bytes - measure_owned(execution))
                                projected = (generation_scope if game else source_scope).enter_context(
                                    ops.project(plan, numeric, config, learner_label, generation, execution,
                                        **projection_options))
                                if game:
                                    decision_count, proof, numeric_files, issued_projection = projected
                                    if hpo and decision_count==0:
                                        ops.stop_reason='empty_collection'
                                        final_status='incomplete'
                                        break
                                else:
                                    decision_count, proof, numeric_files = projected
                            else:
                                online_context = OnlinePPOContext(directory, state["run_id"], batch_id, generation, ownership.token)
                                if config.preprocessing.backend == 'rust':
                                    decision_count, proof = await stage_rollout(collection.directory, numeric, model, linkage,
                                                                         config.preprocessing, deadline=execution,
                                                                         online_context=online_context,
                                                                         expected_identity=preprocessing, wall_deadline=wall)
                                else:
                                    decisions, proof = ingest_collection(collection.directory, model, linkage, deadline=execution,
                                                                         online_context=online_context)
                                    if isinstance(model, (GRUModelConfig, TransformerModelConfig)):
                                        from gf_training.sequence_rollout import (
                                            write_rollout as write_sequence,
                                        )
                                        write_sequence(numeric, decisions, model, deadline=execution)
                                    else:
                                        write_rollout(numeric, decisions, deadline=execution)
                                    decision_count = len(decisions)
                                    del decisions
                            if owned: assert numeric_files is not None
                            numeric_hash = numeric_files['rollout.json'] if owned and numeric_files is not None else sha256(numeric / 'rollout.json', execution)
                            if measure_owned(execution) > training.max_owned_bytes or time.monotonic() >= execution:
                                raise RuntimeError('PPO collection/encoding exceeded execution/storage limits')
                            next_generation = generation + 1
                            await asyncio.sleep(0)  # Serve pending cancellation before optimizer side effects.
                            if hpo: ops.phase('update', transient_learner_decisions=decision_count)
                            update_started = time.monotonic()
                            if ops:
                                acknowledgement = (await ops.update_owned(issued_projection,checkpoint_path(directory,next_generation),deadline=execution) if owned else
                                    await child.update(issued_projection, checkpoint_path(directory, next_generation), deadline=execution) if game else
                                    await child.update(numeric, numeric_hash, learner_label,
                                    checkpoint_path(directory, next_generation), deadline=execution))
                                metrics = {**acknowledgement['metrics'], **acknowledgement['timings']}
                            else:
                                metrics = await child.request({'command': 'update', 'batch': str(numeric), 'batch_hash': numeric_hash,
                                                               'output': str(checkpoint_path(directory, next_generation)), 'action_hash': digest(asdict(config.actions))}, execution)
                            check_clock()
                            update_seconds = time.monotonic() - update_started
                            if time.monotonic() >= execution or measure_owned(execution) > training.max_owned_bytes:
                                raise RuntimeError('PPO update/save execution/storage overrun')
                        if collection_scan is not None:
                            collection_scan.finalize()
                        state['accepted_decisions'] += decision_count
                        if hpo:
                            previous_used=state['used_decisions']
                            state['used_decisions']+=decision_count
                            state['collected_decisions']+=proof['selection']['collected_decisions']
                        state['optimizer_steps'] = metrics['optimizer_steps']
                        state['collection_seconds'] += collection_seconds
                        state['update_seconds'] += update_seconds
                        if hpo:
                            operating=config.hpo_spec['operating']
                            interval=operating['pool_every_decisions']
                            if state['used_decisions']//interval>previous_used//interval:
                                pool=(pool+[next_generation])[-operating['opponent_pool_size']:]
                        elif next_generation % 10 == 0:
                            pool = (pool + [next_generation])[-5:]
                        update = {'schema_version': 1, 'status': 'complete', 'run_id': state['run_id'], 'generation': next_generation,
                                  'parent_sha256': state['generation_sha256'], 'batch_id': batch_id,
                                  'collection': str(collection.directory.relative_to(directory)), 'collection_manifest_sha256': proof['manifest_sha256'],
                                  'numeric': str(numeric.relative_to(directory)), 'numeric_sha256': numeric_hash,
                                  'accepted_decisions': decision_count, 'source_proof': proof, 'metrics': metrics,
                                  'planned_games': config.hpo_spec['training_recipe']['collection_games'] if hpo else 16, 'actual_games': proof['games'], 'health': {},
                                  'policy_bindings': {key: asdict(value) for key, value in registry.items()},
                                  'learner_seats': linkage['learner_seats'], 'collection_seconds': collection_seconds, 'update_seconds': update_seconds}
                        if ops:
                            if game:
                                ops.update_record(update, state, numeric_files)
                            else:
                                update.update(schema_version=2, model_id=state['model_id'], model_version='2', input_recipe=state['input_recipe'],
                                              action_contract=state['action_contract'], purpose='component', synthetic_capacity=False,
                                              numeric_files=numeric_files)
                            committed_record = ops.commit(directory, state, config, generation=next_generation, update=update,
                                controls=ops.controls(rng,pool,next_generation) if hpo else _controls(rng, pool, next_generation), acknowledgement=acknowledgement, deadline=execution)
                            if hpo and getattr(ops, 'resident_runtime', None) is not None:
                                generation_scope.close()  # Retire original Source/control/registry snapshots BEFORE pins.
                                ops.phase('sync')
                                await ops.resident_runtime.after_commit(next_generation, committed_record)
                            ops.retain(directory, next_generation, config, execution)
                        else:
                            commit_generation(directory, state, generation=next_generation, update=update, controls=_controls(rng, pool, next_generation), deadline=execution)
                            retain(directory, next_generation, deadline=execution)
                    check_clock()
                    if time.monotonic() >= execution or measure_owned(execution) > training.max_owned_bytes:
                        raise RuntimeError('PPO commit/retention execution/storage overrun')
                    if pause_after_update == next_generation:
                        final_status = 'paused'
                        break
            except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
                error = failure
            finally:
                if hpo: ops.phase('cleanup')
                try:
                    if hpo:
                        ops.close_precollection()
                    if hpo and getattr(ops, 'resident_runtime', None) is not None:
                        await ops.resident_runtime.close()
                except BaseException as failure:  # noqa: BLE001 - actual pool cleanup precedes learner termination
                    error = error or failure
                    state['cleanup_failure'] = str(failure)
                try:
                    if owned: await ops.close_owned_child(child,min(wall,time.monotonic()+training.cleanup_seconds))
                    else: await child.close(min(wall, time.monotonic() + training.cleanup_seconds))
                except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
                    error = error or failure
                    state['cleanup_failure'] = str(failure)
            if game:
                try:
                    ops.close_batch()
                except BaseException as cleanup_error:  # noqa: BLE001 -- persist cleanup failure before release.
                    error = error or cleanup_error
                    state['cleanup_failure'] = str(cleanup_error)
            if set(child.health) - {'exit_code', 'stderr_bytes'} or child.health.get('exit_code') != 0:
                error = error or RuntimeError('PPO learner cleanup health failed')
            try:
                check_clock(scope='cleanup')
            except BaseException as clock_failure:  # noqa: BLE001 -- finish owned bookkeeping and preserve the initiating error.
                if error is None:
                    error = clock_failure
                else:
                    error.add_note('PPO cleanup clock observation: ' + str(clock_failure))
            state['invocations'].append({'mode': mode, 'elapsed_seconds': time.monotonic() - invocation_started, 'child_health': child.health})
            state['elapsed_seconds'] = state['last_unix'] - state['created_unix']
            try:
                storage_bytes = measure_owned(wall)
            except TimeoutError as failure:
                storage_bytes = None
                error = error or failure
            state['artifact_bytes'] = storage_bytes
            state['peak_coordinator_memory_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            state['coordinator_memory_scope'] = 'process lifetime peak RSS, distinct from learner child metrics'
            state['timing_scope'] = 'pre-final-write snapshot; original wall includes paused time'
            if owned and error is None and final_status=='complete':
                try: ops.prepare_completion(state)
                except BaseException as preparation:  # noqa: BLE001 - original preparation failure joins actual cleanup
                    error=preparation
            if hpo:
                try:
                    ops.finish_cleanup(error)
                except BaseException as cleanup:  # noqa: BLE001 -- preserve original failure after actual cleanup.
                    if error is not None: error.add_note('HPO final cleanup: '+str(cleanup))
                    error=error or cleanup
                    state['cleanup_failure']=str(cleanup)
            if error is not None or time.monotonic() >= wall or storage_bytes is None or storage_bytes > training.max_owned_bytes:
                state.update(status='failed', failure=type(error).__name__ if error else 'LimitOverrun', quarantine=True)
            else:
                state['status'] = final_status
                state['completed_updates'] = state['updates'] if final_status == 'complete' else None
                if hpo: state['stop_reason']='pause' if final_status=='paused' else ops.stop_reason
            try:
                if ops:
                    ops.persist_control(directory / 'run.json', state, config, deadline=wall)
                else:
                    atomic_json(directory / 'run.json', state)
                if time.monotonic() >= wall or measure_owned(wall) > training.max_owned_bytes:
                    if formal is not None:
                        raise RuntimeError('formal post-persistence wall/storage overrun')
                    state.update(status='failed', failure='FinalPersistenceOverrun', quarantine=True)
                    if ops:
                        ops.persist_control(directory / 'run.json', state, config, deadline=wall)
                    else:
                        atomic_json(directory / 'run.json', state)
            except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
                state.update(status='failed', failure=type(failure).__name__, quarantine=True)
                if ops:
                    try:
                        ops.persist_control(directory / 'run.json', state, config, deadline=wall)
                    except BaseException as correction:  # noqa: BLE001 - initiating failure survives best-effort correction
                        failure.add_note('structured state correction failed: ' + type(correction).__name__)
                else:
                    try:
                        atomic_json(directory / 'run.json', state)
                    except OSError:
                        pass
                error = error or failure
            if owned:
                try: ops.finalize_completion(state,error)
                except BaseException as completion_error:  # noqa: BLE001 - preserve failure and revoke final eligibility
                    state.update(status='failed',failure=type(completion_error).__name__,quarantine=True)
                    try: ops.persist_control(directory/'run.json',state,config,deadline=wall)
                    except BaseException as correction: completion_error.add_note('completion correction: '+str(correction))  # noqa: BLE001
                    if error is None: error=completion_error
                    else: error.add_note('completion retirement: '+str(completion_error))
            if error is not None:
                raise error
            if state['status'] == 'failed':
                raise RuntimeError('PPO run failed original limits')
            return {'run_dir': str(directory), 'status': state['status'], 'updates': state['updates'], 'accepted_decisions': state['accepted_decisions'],
                    'elapsed_seconds': time.time() - state['created_unix'], 'generation_sha256': state['generation_sha256'],
                    **({'used_decisions':state['used_decisions'],'collected_decisions':state['collected_decisions'],
                        'decision_budget':config.hpo_spec['decision_budget'],'stop_reason':state['stop_reason']} if hpo else {})}
