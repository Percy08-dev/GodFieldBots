"""CLI-owned bounded latest progress and lock-free, read-only observation."""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import stat
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from gf_common.domain import ContractError, Json
from gf_common.hpo_progress import PHASES, ActiveRate, ProgressCallback, finite_number

if TYPE_CHECKING:
    from gf_training.lineage import RunOwnership
    from gf_training.rust_player_supervision import HostControl

MAX_BYTES = 65536


def _checked(value: Any) -> Json:
    if (type(value) is not dict or type(value.get('schema_version')) is not int or value.get('schema_version') != 1 or value.get('kind') != 'hpo-progress'
            or not finite_number(value.get('updated_unix')) or value.get('phase') not in PHASES
            or value.get('status') not in ('running','complete','paused','incomplete','failed','session-boundary','resource-blocked')):
        raise ValueError('unsupported or malformed progress snapshot')
    # JSON accepts NaN/Infinity and overflowing float literals by default.
    # Validate every nested field before admitting a snapshot to either renderer.
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) in (list,tuple):
            pending.extend(item)
        elif type(item) in (int,float) and not finite_number(item):
            raise ValueError('nonfinite progress value')
    if value.get('standalone') is True:
        for name in ('seed', 'committed_updates', 'committed_used', 'committed_collected',
                     'decision_budget', 'target_updates', 'batch_games_planned', 'batch_games_ended',
                     'transient_learner_decisions'):
            field = value.get(name)
            if field is not None and (type(field) is not int or field < 0):
                raise ValueError('invalid standalone progress counter')
        if (value.get('committed_updates') is None) != (value.get('committed_used') is None):
            raise ValueError('standalone committed counters require both values')
        for name in ('last_collection_games_per_second', 'last_collection_actions_per_second', 'invocation_elapsed_seconds',
                     'eta_seconds'):
            field = value.get(name)
            if field is not None and (not finite_number(field) or field < 0):
                raise ValueError('invalid standalone progress time or rate')
        for name in ('latest_metrics', 'latest_durations'):
            fields = value.get(name)
            if fields is not None and (type(fields) is not dict or
                    any(item is not None and not finite_number(item) for item in fields.values())):
                raise ValueError('invalid standalone progress scalars')
    return value


def read_progress(root: Path, *, now: float | None = None) -> Json:
    """Never opens a campaign, takes a lock, or writes anything."""
    try:
        path = root/'telemetry'/'progress.json'
        for parent in (root, root/'telemetry'):
            if parent.is_symlink():
                raise ValueError('linked telemetry namespace')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                raise ValueError('progress is not a bounded regular file')
            data = source.read(MAX_BYTES+1)
        if len(data) > MAX_BYTES:
            raise ValueError('oversized progress')
        value = _checked(json.loads(data))
        age = max(0., (time.time() if now is None else now)-value['updated_unix'])
        stale = age > 5
        if value.get('standalone') is True and (
                stale or value['status'] != 'running') and 'eta_seconds' in value:
            value['eta_seconds'] = None
        return {'availability':'stale' if stale else 'available', 'age_seconds':age, 'snapshot':value,
            'message':'update pending; status is unconfirmed' if stale else 'diagnostic observation only'}
    except FileNotFoundError:
        return {'availability':'missing','snapshot':None,'message':'progress unavailable; no snapshot'}
    except (OSError,ValueError,RecursionError,OverflowError):
        return {'availability':'invalid','snapshot':None,'message':'progress unavailable; invalid snapshot'}


def render(value: Json) -> str:
    snapshot = value.get('snapshot')
    durable = value.get('durable')
    if durable is not None or snapshot is not None and snapshot.get('standalone') is True:
        return _render_standalone(value, snapshot, durable)
    if snapshot is None:
        return value['message']
    used, budget = snapshot.get('committed_used'), snapshot.get('decision_budget')
    committed = f' committed={used}/{budget} updates={snapshot.get("committed_updates")}' if used is not None else ''
    pending = ' evaluation/cleanup pending' if used is not None and used == budget and snapshot['status']=='running' else ''
    eta = snapshot.get('eta_seconds')
    rate = snapshot.get('decisions_per_second')
    speed = f' rate={rate:.1f}/s eta={eta:.0f}s' if finite_number(rate) and finite_number(eta) else ''
    transient = f' collected-games={snapshot["transient_games"]}' if snapshot.get('transient_games') is not None else ''
    metrics = snapshot.get('latest_metrics')
    loss = f' loss={metrics["total_loss"]:.4g} entropy={metrics["entropy"]:.4g} KL={metrics["approx_kl"]:.4g}' if isinstance(metrics,dict) and all(finite_number(metrics.get(key)) for key in ('total_loss','entropy','approx_kl')) else ''
    model = snapshot.get('model') or ('/'.join(str(snapshot[key]) for key in ('left_model','right_model')) if snapshot.get('left_model') else 'model unavailable')
    return (f'{model} {snapshot.get("proposal_kind") or "comparison"} '
        f'seed={snapshot.get("seed") if snapshot.get("seed") is not None else snapshot.get("left_seed")} '
        f'{snapshot["phase"]} {snapshot["status"]}{committed}{transient}{speed}{loss}{pending} '
        f'[{value["message"]}]')


def _standalone_running(live: Json | None, durable: Json | None, availability: str) -> bool:
    return (live is not None and live.get('standalone') is True
            and availability == 'available' and live.get('status') == 'running'
            and (durable is None or durable['status'] == 'running'
                and durable['availability'] == 'available'))


def _render_standalone(value: Json, live: Json | None, durable: Json | None) -> str:
    current = live or {}
    committed = durable or current
    live_availability = value.get('availability', 'available')
    running = _standalone_running(live, durable, live_availability)
    parts = [f'{committed.get("model", "mlp")} standalone HPO seed={committed.get("seed")}']
    if live is not None and live_availability == 'available':
        parts.append(f'phase={current["phase"]}')
        ended, planned = current.get('batch_games_ended'), current.get('batch_games_planned')
        if type(ended) is int and type(planned) is int:
            parts.append(f'batch games ended={ended}/{planned}')
        pending = current.get('transient_learner_decisions')
        if type(pending) is int:
            parts.append(f'batch selected learner decisions={pending}')
    status = committed.get('status')
    if status is not None:
        parts.append(f'{"run record" if durable is not None else "live"} status={status}')
    if committed.get('host_result') == 'awaiting-supervised-finalization':
        parts.append('run record written before supervised finalization')
    if committed.get('committed_updates') is not None:
        target = current.get('target_updates') if running else committed.get('target_updates')
        parts.append(f'updates={committed["committed_updates"]}/{target}')
        parts.append(f'committed learner decisions={committed["committed_used"]}/{committed.get("decision_budget")}')
        if committed.get('committed_collected') is not None:
            parts.append(f'collected learner decisions={committed["committed_collected"]}')
    metrics = committed.get('latest_metrics') or {}
    for key, label in (('total_loss', 'loss'), ('entropy', 'entropy'), ('approx_kl', 'KL')):
        if finite_number(metrics.get(key)):
            parts.append(f'{label}={metrics[key]:.4g}')
    durations = committed.get('latest_durations') or {}
    for key, label in (('collection_seconds', 'last collect'), ('update_seconds', 'last update/save')):
        if finite_number(durations.get(key)):
            parts.append(f'{label}={durations[key]:.2f}s')
    rates = []
    for unit in ('games', 'actions'):
        rate = committed.get(f'last_collection_{unit}_per_second')
        if finite_number(rate):
            rates.append(f'{rate:.2f} {unit}/s')
    if rates:
        parts.append('last collection=' + ', '.join(rates))
    elapsed = current.get('invocation_elapsed_seconds') if running else committed.get('elapsed_seconds')
    if finite_number(elapsed):
        parts.append(f'{"invocation elapsed" if running else "elapsed"}={elapsed:.2f}s')
    eta = current.get('eta_seconds') if running else None
    if finite_number(eta):
        parts.append(f'ETA≈{eta:.0f}s')
    if durable is not None and durable['availability'] == 'stale':
        parts.append('last verified run records')
    elif durable is not None and durable['availability'] == 'partial':
        parts.append('committed records unavailable')
    message = value['message']
    if durable is not None and live_availability != 'available':
        live_state = 'stale' if live_availability == 'stale' else 'unavailable'
        message = f'run records; live snapshot {live_state}'
    parts.append(f'[{message}]')
    return ' '.join(parts)


class ProgressOwner:
    """Sole execution owner composes callbacks; telemetry failures disable only I/O."""
    def __init__(self, store: Any, *, mode: str = 'auto', stream: TextIO | None = None) -> None:
        self.store, self.mode = store, mode
        self.stream = sys.stderr if stream is None else stream
        self.disabled = False
        self._heartbeat_warned = False
        self.value: Json = {'schema_version':1,'kind':'hpo-progress','updated_unix':time.time(),
            'phase':'startup','status':'running','model':None,'proposal_kind':None,'seed':None,
            'committed_used':None,'decision_budget':None,'committed_updates':None,'latest_metrics':None,
            'transient_games':None,'transient_learner_decisions':None,'transient_total_game_actions':None,
            'cuda_allocator':None,'decisions_per_second':None,'eta_seconds':None,
            'whole_device_sampled_high_water_bytes':None,
            'whole_device_memory_status':'unavailable; external sampling required',
            'timing_scope':'active campaign/comparison invocation monotonic; ETA is current training run only'}
        self._phase_started = time.monotonic()
        self._last_published = float('-inf')
        self._durations: dict[str,float] = {}

    def emit(self, facts: Json) -> None:
        try:
            now = time.monotonic()
            phase = self.value['phase']
            changed = facts.get('phase',phase)!=phase or facts.get('status',self.value['status'])!=self.value['status']
            self._durations[phase] = self._durations.get(phase,0.) + now-self._phase_started
            self._phase_started = now
            self.value.update(facts)
            self.value.update(updated_unix=time.time(),phase_seconds=dict(self._durations))
            if not changed and now-self._last_published < 1.:
                return
            self._last_published = now
            if not self.disabled:
                try:
                    self._write()
                except Exception:  # noqa: BLE001 - never poison the campaign or alter learning
                    self.disabled = True
                    with contextlib.suppress(Exception):
                        print('HPO progress persistence disabled; training continues.',file=self.stream,flush=True)
            if self.mode == 'plain' or self.mode == 'auto' and self.stream.isatty():
                print(render({'snapshot':self.value,'message':'diagnostic observation only'}),file=self.stream,flush=True)
        except Exception:  # noqa: BLE001, S110 - rendering is also optional
            pass

    def _write(self) -> None:
        self.store._require()
        data = (json.dumps(_checked(self.value), separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(data)>MAX_BYTES:
            raise ValueError('progress exceeds 64KiB')
        folder = self.store.directory/'telemetry'
        folder.mkdir(mode=0o700,exist_ok=True)
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('telemetry must be private and owned')
        temporary = folder/('progress.tmp-'+uuid.uuid4().hex)
        try:
            fd = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as output:
                output.write(data)
            os.replace(temporary,folder/'progress.json')
        finally:
            temporary.unlink(missing_ok=True)

    async def _retire_heartbeat(self, coroutine: Any, returned: Any = None) -> None:
        # A task factory can transfer the ORIGINAL coroutine and then raise.
        # Discover only actual Tasks with that exact original identity; never
        # substitute a Future or cancel a foreign factory-returned task.
        tasks = {task for task in asyncio.all_tasks() if task.get_coro() is coroutine}
        if isinstance(returned,asyncio.Task) and returned.get_coro() is coroutine:
            tasks.add(returned)
        for task in tasks:
            task.cancel()
        cancelled = None
        for task in tasks:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        cancelled = cancelled or error
                except Exception:  # noqa: BLE001 - optional heartbeat has finished with an error
                    break
            with contextlib.suppress(asyncio.CancelledError,Exception):
                task.result()
        if inspect.getcoroutinestate(coroutine)!=inspect.CORO_CLOSED:
            coroutine.close()
        if cancelled is not None:
            raise cancelled

    @contextlib.asynccontextmanager
    async def live(self) -> AsyncIterator[ProgressOwner]:
        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(1)
                self.emit({})
        self.emit({})
        coroutine = heartbeat()
        task = None
        try:
            task = asyncio.create_task(coroutine)
            if not isinstance(task,asyncio.Task) or task.get_coro() is not coroutine:
                raise RuntimeError('optional heartbeat requires its original returned Task')
        except BaseException as setup_error:
            try:
                await self._retire_heartbeat(coroutine,task)
            except asyncio.CancelledError:
                if isinstance(setup_error,asyncio.CancelledError):
                    raise setup_error
                raise
            if not isinstance(setup_error,Exception):
                raise
            if not self._heartbeat_warned:
                self._heartbeat_warned = True
                with contextlib.suppress(Exception):
                    print('HPO progress heartbeat unavailable; phase updates continue.',file=self.stream,flush=True)
            self.emit({'heartbeat_status':'unavailable'})
            task = None
        primary: BaseException | None = None
        try:
            yield self
        except BaseException as error:
            primary = error
            self.emit({'status':'failed'})
            raise
        finally:
            try:
                await self._retire_heartbeat(coroutine,task)
            except asyncio.CancelledError:
                if not isinstance(primary,asyncio.CancelledError):
                    raise


class _StandaloneProgressStore:
    """Diagnostic-only view of an existing supervised parent lock."""

    def __init__(self, control: HostControl, ownership: RunOwnership) -> None:
        from gf_cli.hpo_progress_records import checked_namespace
        from gf_models.structured_sealed import MAX_SEALED_SCAN_BYTES, SealedScan

        self.control, self.ownership = control, ownership
        self.parent = ownership.directory
        scan = SealedScan(max_owned_bytes=MAX_SEALED_SCAN_BYTES, deadline=time.monotonic() + 1.)
        self.directory, self.identity, self.lock = checked_namespace(self.parent, control.operation, scan)

    def _require(self) -> None:
        from gf_cli.hpo_progress_records import namespace_identity
        from gf_training.lineage import _HELD_RUN_OWNERS, require_hpo_run_owner

        require_hpo_run_owner(self.parent, self.ownership)
        if (_HELD_RUN_OWNERS[self.parent].control is not self.control
                or self.control.operation['parent_directory'] != str(self.parent)):
            raise ContractError('standalone progress requires the same supervised parent owner')
        mapping = self.control._borrowed['campaign']
        fd = mapping['fd']
        lock = os.fstat(fd)
        if (self.lock != (mapping['device'], mapping['inode'])
                or self.lock != (lock.st_dev, lock.st_ino)
                or namespace_identity(self.directory) != (self.identity, self.lock)):
            raise ContractError('standalone progress original control Store identity changed')
        self.control._require_campaign_ofd(fd)


class _StandaloneTiming:
    """An active invocation's optional elapsed time and bounded commit history."""

    def __init__(self, emit: ProgressCallback, *, target: int, budget: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.emit, self.target, self.budget, self.clock = emit, target, budget, clock
        self.started = clock()
        self.updates = ActiveRate()
        self.decisions = ActiveRate()
        self.estimated_end: float | None = None

    def __call__(self, facts: Json) -> None:
        from gf_cli.hpo_progress_records import collection_rate

        now = self.clock()
        snapshot = {**facts, 'invocation_elapsed_seconds': now - self.started}
        actions = snapshot.pop('latest_actual_game_actions', None)
        generation, used = facts.get('committed_updates'), facts.get('committed_used')
        if generation is not None and used is not None:
            update_eta = self.updates.commit(generation, now=now, budget=self.target)['eta_seconds']
            decision_eta = self.decisions.commit(used, now=now, budget=self.budget)['eta_seconds']
            estimates = [eta for eta in (update_eta, decision_eta) if eta is not None]
            self.estimated_end = now + min(estimates) if estimates else None
            durations = facts.get('latest_durations') or {}
            seconds = durations.get('collection_seconds')
            snapshot['last_collection_games_per_second'] = collection_rate(facts.get('latest_actual_games'), seconds)
            snapshot['last_collection_actions_per_second'] = collection_rate(actions, seconds)
        snapshot['eta_seconds'] = None if self.estimated_end is None else max(0., self.estimated_end - now)
        self.emit(snapshot)


def standalone_publisher(control: HostControl, config: Any, ownership: RunOwnership) -> ProgressCallback:
    """Activated by the training host only while the original parent is held."""
    owner = ProgressOwner(_StandaloneProgressStore(control, ownership))
    limit = config.training.max_updates
    pause = control.operation['execution_request']['pause_after_update']
    target = min(limit, pause) if type(pause) is int and pause > 0 else limit
    budget = config.hpo_spec['decision_budget']
    timing = _StandaloneTiming(owner.emit, target=target, budget=budget)
    timing({'standalone': True, 'phase': 'startup', 'model': config.model.model_id,
        'seed': config.training.seeds[0], 'decision_budget': budget, 'target_updates': target,
        'timing_scope': 'active standalone training invocation monotonic'})
    return timing


_WATCH_TERM: str | None = None


def _terminal_screen(output: TextIO) -> tuple[str, str, str] | None:
    global _WATCH_TERM
    term = os.environ.get('TERM', '').strip()
    if term.lower() in ('', 'dumb') or not output.isatty():
        return None
    # Python's curses keeps the first successful setupterm for the process.
    if _WATCH_TERM is not None and term != _WATCH_TERM:
        return None
    try:
        import curses
    except ImportError:
        return None
    try:
        if _WATCH_TERM is None:
            curses.setupterm(term=term, fd=output.fileno())
            _WATCH_TERM = term
        enter, leave, clear = (curses.tigetstr(name) for name in ('smcup', 'rmcup', 'clear'))
        if not enter or not leave or not clear or any(b'$<' in control for control in (enter, leave, clear)):
            return None
        return enter.decode('ascii'), leave.decode('ascii'), clear.decode('ascii')
    except (curses.error, OSError, ValueError):
        return None


async def watch(root: Path, *, interval: float = 1., once: bool = False,
                as_json: bool = False, output: TextIO | None = None) -> int:
    if not finite_number(interval) or interval <= 0:
        raise ValueError('watch interval must be finite and positive')
    output = sys.stdout if output is None else output
    from gf_cli.hpo_progress_records import StandaloneProgressRecords

    records = StandaloneProgressRecords(root)
    screen = None if once or as_json else _terminal_screen(output)
    enter, leave, prefix = screen or ('', '', '')
    try:
        if enter:
            print(enter, end='', file=output, flush=True)
        while True:
            value = read_progress(root)
            snapshot = value['snapshot']
            if value['availability'] != 'available' or snapshot.get('standalone') is True:
                durable = records.read()
                if records.live_root is not None:
                    value = read_progress(records.live_root)
                if durable is not None:
                    value['durable'] = durable
                live = value['snapshot']
                if (live is not None and live.get('standalone') is True
                        and live.get('eta_seconds') is not None
                        and not _standalone_running(live, durable, value['availability'])):
                    live['eta_seconds'] = None
            frame = json.dumps(value, allow_nan=False) if as_json else render(value)
            print(prefix + frame, file=output, flush=True)
            if once:
                return 0
            await asyncio.sleep(interval)
    finally:
        if leave:
            # Attempt restoration even after a partial write or failed frame.
            with contextlib.suppress(OSError, ValueError):
                print(leave, end='', file=output, flush=True)
