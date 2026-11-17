"""Bounded passive counts/CPU scopes; no argument/result contents are retained."""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import threading
import time
import weakref
from pathlib import Path
from typing import Any

from benchmarks.collection_cpu.common import immutable_json, sha256


class Counters:
    def __init__(self) -> None:
        self.rows: dict[str,Any] = {}
        self.lock = threading.Lock()
        self.dropped = 0
        self.wall_started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.thread_started = time.thread_time()

    def wrap(self, name: str, original: Any) -> Any:
        def begin() -> tuple[str,float,float]:
            caller = inspect.currentframe()
            try:
                caller = None if caller is None else caller.f_back
                caller = None if caller is None else caller.f_back
                location = 'unknown' if caller is None else f'{Path(caller.f_code.co_filename).name}:{caller.f_code.co_name}:{caller.f_lineno}'
            finally:
                del caller
            return name+' <- '+location,time.perf_counter(),time.thread_time()

        def end(start: tuple[str,float,float], failed: bool) -> None:
            key,wall,cpu = start
            with self.lock:
                if key not in self.rows and len(self.rows) >= 2048:
                    self.dropped += 1
                    return
                row = self.rows.setdefault(key,{'count':0,'failures':0,'inclusive_wall_seconds':0.,'thread_cpu_seconds':0.})
                row['count'] += 1
                row['failures'] += int(failed)
                row['inclusive_wall_seconds'] += time.perf_counter()-wall
                row['thread_cpu_seconds'] += time.thread_time()-cpu

        if inspect.iscoroutinefunction(original):
            @functools.wraps(original)
            async def async_call(*args: Any, **kwargs: Any) -> Any:
                started,failed = begin(),True
                try:
                    result = await original(*args,**kwargs)
                    failed = False
                    return result
                finally:
                    end(started,failed)
            return async_call
        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            started,failed = begin(),True
            try:
                result = original(*args,**kwargs)
                failed = False
                return result
            finally:
                end(started,failed)
        return call

    def payload(self) -> dict[str, Any]:
        return {'rows':self.rows,'dropped':self.dropped,
                'process_wall_seconds':time.perf_counter()-self.wall_started,
                'process_cpu_seconds':time.process_time()-self.cpu_started,
                'main_thread_cpu_seconds':time.thread_time()-self.thread_started,
                'scope':'nested/concurrent inclusive wall and owner-thread CPU; never summed into parent wall',
                'cpu_limits':'async CPU spans include unrelated callbacks on same thread; runtime/driver spin is not useful-work proof',
                'policy_visibility':'no arguments, results, IDs, private observations or management facts retained as policy features'}


_ACTIVE: list[tuple[Path,Counters]] = []


def patches(output: Path) -> list[Any]:
    import httpx

    from gf_common import domain
    from gf_common.structured import CandidateTable, DecisionFrame
    from gf_simulator_client.http import AttestedClient
    from gf_simulator_client.resident_pool import ResidentPhysicalWorker, _ExchangeSetup
    from gf_training.card_instance_journal import CardInstanceJournal
    from gf_training.group_pack_memory import MemoryCardInstanceJournal
    from gf_training.hpo_owned_capture import _CaptureContext
    counters = Counters()
    _ACTIVE.append((output/'parent-counters.json',counters))
    targets: list[tuple[Any,str]] = [(CandidateTable,'validate'),(CandidateTable,'to_action'),
               (CardInstanceJournal,'begin_frame'),(MemoryCardInstanceJournal,'begin_frame'),
               (_CaptureContext,'begin_decision'),(domain,'canonical'),(domain,'digest'),
               (json,'loads'),(json,'dumps'),(asyncio.Handle,'_run'),
               (ResidentPhysicalWorker,'exchange'),(ResidentPhysicalWorker,'_read_reply'),(_ExchangeSetup,'__init__'),
               (AttestedClient,'_request'),(AttestedClient,'observe'),
               (AttestedClient,'events'),(AttestedClient,'actor_options'),(AttestedClient,'act'),
               (httpx.AsyncClient,'send'),(httpx.Response,'json'),(httpx.Response,'aread')]
    result = [(owner,name,counters.wrap(owner.__name__+'.'+name,getattr(owner,name))) for owner,name in targets]
    completed: Any = weakref.WeakKeyDictionary()
    exchange_original, wait_original = _ExchangeSetup._exchange, _ExchangeSetup._wait_original

    def exchange(self: Any) -> Any:
        try:
            return exchange_original(self)
        finally:
            completed[self] = time.perf_counter()

    async def resumed(self: Any) -> Any:
        result = await wait_original(self)
        finished = completed.pop(self,None)
        if finished is not None:
            key = 'IPC completion-to-original-wait-resume'
            with counters.lock:
                row = counters.rows.setdefault(key,{'count':0,'inclusive_wall_seconds':0.})
                row['count'] += 1
                row['inclusive_wall_seconds'] += time.perf_counter()-finished
        return result

    result.extend([(_ExchangeSetup,'_exchange',counters.wrap('IPC original blocking framing',exchange)),
                   (_ExchangeSetup,'_wait_original',counters.wrap('IPC original wait',resumed))])
    for name in ('observation','raw','references'):
        prop = getattr(DecisionFrame,name)
        result.append((DecisionFrame,name,property(counters.wrap('DecisionFrame.'+name,prop.fget))))
    return result


def actor_patches(network: Any, counters: Counters) -> None:
    for name in ('_tensor','_encode_frame_calculate','_candidate_features','_references',
                 '_candidate_chunk_calculate','step','distribution','value'):
        setattr(network,name,counters.wrap('network.'+name,getattr(network,name)))


def actor_install(counters: Counters) -> None:
    import json

    from gf_common.diverse_actions import PublicCandidateGeneratorV3
    from gf_models import structured_policy
    from gf_simulator_client import card_instance_episode
    from gf_simulator_client.card_instance_episode import CardInstanceEpisode
    for owner,name in ((PublicCandidateGeneratorV3,'generate_game'),(CardInstanceEpisode,'select'),
                       (structured_policy,'tokenize_frame'),(structured_policy,'tokenize_table'),
                       (card_instance_episode,'game_generated_dict_v3'),(json,'loads'),(json,'dumps')):
        setattr(owner,name,counters.wrap(owner.__name__+'.'+name,getattr(owner,name)))


def flush() -> list[dict[str,Any]]:
    records = []
    for path,counters in _ACTIVE:
        immutable_json(path,counters.payload())
        records.append({'path':str(path),'sha256':sha256(path),'dropped':counters.dropped})
    _ACTIVE.clear()
    return records
