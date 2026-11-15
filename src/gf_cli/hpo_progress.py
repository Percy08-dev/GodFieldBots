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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, TextIO

from gf_common.domain import Json
from gf_common.hpo_progress import PHASES, finite_number

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
        return {'availability':'stale' if stale else 'available', 'age_seconds':age, 'snapshot':value,
            'message':'update pending; status is unconfirmed' if stale else 'diagnostic observation only'}
    except FileNotFoundError:
        return {'availability':'missing','snapshot':None,'message':'progress unavailable; no snapshot'}
    except (OSError,ValueError,RecursionError,OverflowError):
        return {'availability':'invalid','snapshot':None,'message':'progress unavailable; invalid snapshot'}


def render(value: Json) -> str:
    snapshot = value.get('snapshot')
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


async def watch(root: Path, *, interval: float = 1., once: bool = False,
                as_json: bool = False, output: TextIO | None = None) -> int:
    if not finite_number(interval) or interval <= 0:
        raise ValueError('watch interval must be finite and positive')
    output = sys.stdout if output is None else output
    while True:
        value = read_progress(root)
        print(json.dumps(value,allow_nan=False) if as_json else render(value),file=output,flush=True)
        if once:
            return 0
        await asyncio.sleep(interval)
