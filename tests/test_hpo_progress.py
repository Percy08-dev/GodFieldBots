"""Bounded diagnostic observation; no execution authority or ML imports."""
import asyncio
import io
import json
import subprocess
import sys

import pytest

from gf_cli.hpo_storage import CampaignStore


def test_live_watch_is_optional_free_read_only_even_with_lock(tmp_path):
    from gf_cli.hpo_progress import ProgressOwner
    with CampaignStore(tmp_path/'campaign', mode='fresh', max_owned_bytes=8*1024**2) as store:
        owner = ProgressOwner(store, mode='off', stream=io.StringIO())
        owner.emit({'phase':'collect', 'model':'gru', 'seed':17})
        before = {str(p):p.stat().st_mtime_ns for p in store.directory.rglob('*')}
        source = """
import sys
from gf_cli.hpo import main
assert main(['watch','--path',sys.argv[1],'--once','--json']) == 0
assert 'torch' not in sys.modules and 'optuna' not in sys.modules
"""
        result = subprocess.run([sys.executable,'-E','-c',source,str(store.directory)],check=False,capture_output=True,text=True,timeout=10)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['snapshot']['phase'] == 'collect'
        assert {str(p):p.stat().st_mtime_ns for p in store.directory.rglob('*')} == before
        assert store.records() == ()
        with pytest.raises(ValueError): store.put('telemetry/other.json', {})


def test_snapshot_missing_bad_stale_and_bounds(tmp_path):
    from gf_cli.hpo_progress import read_progress
    assert read_progress(tmp_path, now=10)['availability'] == 'missing'
    folder = tmp_path/'telemetry'
    folder.mkdir()
    path = folder/'progress.json'
    for data in (b'{}', b'x'*65537, b'{"schema_version":0}'):
        path.write_bytes(data)
        assert read_progress(tmp_path, now=10)['availability'] == 'invalid'
    path.write_text(json.dumps({'schema_version':1,'kind':'hpo-progress','updated_unix':1,'phase':'update','status':'running'}))
    value = read_progress(tmp_path, now=10)
    assert value['availability'] == 'stale' and value['snapshot']['status'] == 'running'
    assert 'update pending' in value['message']


def test_writer_bounded_atomic_warn_once_and_training_independent(tmp_path, monkeypatch):
    from gf_cli.hpo_progress import ProgressOwner, read_progress
    with CampaignStore(tmp_path/'campaign', mode='fresh', max_owned_bytes=8*1024**2) as store:
        stderr = io.StringIO()
        owner = ProgressOwner(store, mode='plain', stream=stderr)
        owner.emit({'phase':'update','committed_used':0})
        assert read_progress(store.directory)['snapshot']['committed_used'] == 0
        before = (store.directory/'telemetry/progress.json').read_bytes()
        owner.emit({'extra':'x'*65536})
        owner.emit({'phase':'cleanup'})
        assert (store.directory/'telemetry/progress.json').read_bytes() == before
        assert stderr.getvalue().count('persistence disabled') == 1
        assert not list((store.directory/'telemetry').glob('*.tmp*'))


def test_render_auto_redirect_silent_plain_stderr_and_observer_cancel(tmp_path):
    from gf_cli.hpo_progress import ProgressOwner, watch
    with CampaignStore(tmp_path/'campaign', mode='fresh', max_owned_bytes=8*1024**2) as store:
        output = io.StringIO()
        owner = ProgressOwner(store, mode='auto', stream=output)
        owner.emit({'phase':'cleanup','committed_used':10,'decision_budget':10,'status':'running'})
        assert output.getvalue() == ''
        plain = io.StringIO()
        owner = ProgressOwner(store, mode='plain', stream=plain)
        owner.emit({'phase':'cleanup','committed_used':10,'decision_budget':10})
        assert 'cleanup' in plain.getvalue() and 'pending' in plain.getvalue()
        before = (store.directory/'telemetry/progress.json').read_bytes()
        async def cancel():
            task = asyncio.create_task(watch(store.directory, interval=.01, output=io.StringIO()))
            await asyncio.sleep(.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        asyncio.run(cancel())
        assert (store.directory/'telemetry/progress.json').read_bytes() == before
        store.put('still-owned.json', {'ok':True})


def test_active_rate_needs_two_intervals_and_new_invocation_excludes_pause():
    from gf_common.hpo_progress import ActiveRate
    rate = ActiveRate()
    assert rate.commit(0, now=0, budget=100)['eta_seconds'] is None
    assert rate.commit(10, now=1, budget=100)['eta_seconds'] is None
    assert rate.commit(20, now=2, budget=100)['eta_seconds'] == 8
    for used, now in ((30,3),(40,4),(50,5)):
        rate.commit(used, now=now, budget=100)
    assert len(rate.intervals) == 3
    rate = ActiveRate()
    assert rate.commit(50, now=1000, budget=100)['eta_seconds'] is None
    assert rate.commit(60, now=1001, budget=100)['eta_seconds'] is None
    assert rate.commit(70, now=1002, budget=100)['eta_seconds'] == 3


def test_tagged_diagnostics_bounded_quota_stale_and_callback_failure():
    from gf_common.hpo_progress import DiagnosticEmitter, DiagnosticParser
    output = io.StringIO()
    emitter = DiagnosticEmitter('invocation', 2048, output)
    values = []
    parser = DiagnosticParser('invocation', values.append)
    emitter.emit(0, {'phase':'save'})
    raw = output.getvalue().encode()
    parser.feed(raw[:7]); parser.feed(raw[7:])
    assert values[0]['phase'] == 'save'
    parser.feed(raw)  # repeated sequence cannot rewrite live progress
    parser.feed(b'x'*10000+b'\n')
    parser.feed(raw)
    assert len(values) == 1 and len(parser.buffer) <= 4096
    for _ in range(100): emitter.emit(1, {'phase':'save'})
    assert len(output.getvalue().encode()) <= 2048//4
    assert emitter.disabled
    parser.callback = lambda value: (_ for _ in ()).throw(RuntimeError('display only'))
    second = io.StringIO()
    DiagnosticEmitter('invocation',2048,second).emit(2,{'phase':'save'})
    parser.feed(second.getvalue().encode())


def test_existing_stderr_drain_keeps_raw_bytes_and_limits_after_bad_callback(tmp_path):
    from types import SimpleNamespace

    from gf_common.hpo_progress import DiagnosticEmitter, DiagnosticParser
    from gf_training.ppo_run import LearningProcess
    diagnostic = io.StringIO()
    DiagnosticEmitter('invocation',4096,diagnostic).emit(0,{'phase':'save'})
    raw = diagnostic.getvalue().encode()+b'actual numerical failure\n'+b'x'*200
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(raw); reader.feed_eof()
        child = LearningProcess(tmp_path,256)
        def broken(value): raise RuntimeError('optional display')
        child.diagnostic_parser = DiagnosticParser('invocation',broken)
        child.process = SimpleNamespace(stderr=reader)
        await child._drain()
        return child
    child = asyncio.run(run())
    assert child.health == {'stderr_bytes':len(raw),'log_limit_exceeded':True}
    assert child.diagnostic_parser.sequence == 0
    assert next(tmp_path.glob('learner-*.log')).read_bytes() == raw[:256]


def test_cli_progress_preserves_final_json_and_flags(tmp_path,monkeypatch,capsys):
    import gf_cli.hpo as cli
    monkeypatch.setattr(cli,'load_hpo_config',lambda path: object())
    async def run(loaded,path,**options):
        assert options['progress'] == 'plain'
        return {'status':'complete','exact':'original result'}
    monkeypatch.setattr(cli,'run_campaign',run)
    assert cli.main(['run','--config','unused','--campaign-dir',str(tmp_path),'--mode','fresh','--progress','plain']) == 0
    assert json.loads(capsys.readouterr().out) == {'status':'complete','exact':'original result'}


def test_tty_render_and_os_replace_failure_disable_only_telemetry(tmp_path,monkeypatch):
    import gf_cli.hpo_progress as progress
    class TTY(io.StringIO):
        def isatty(self): return True
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        stderr=TTY()
        owner=progress.ProgressOwner(store,mode='auto',stream=stderr)
        owner.emit({'phase':'collect'})
        assert 'collect' in stderr.getvalue()
        def failed(*args): raise OSError('unwritable diagnostic namespace')
        monkeypatch.setattr(progress.os,'replace',failed)
        owner.emit({'phase':'source'}); owner.emit({'phase':'cleanup'})
        assert stderr.getvalue().count('persistence disabled')==1
        assert not list((store.directory/'telemetry').glob('*.tmp*'))
        assert store.records()==()
        # Immutable records use a separate transaction and remain writable.
        store.put('survives.json',{'ok':True})


def test_collection_counts_do_not_infer_learner_from_equal_policy_labels():
    from gf_common.hpo_progress import CollectionProgress
    values=[]
    counter=CollectionProgress(values.append)
    counter.completed(games=2,total_game_actions=7)
    counter.completed(games=1,total_game_actions=4)
    assert values[-1]=={'transient_games':3,'transient_total_game_actions':11,'transient_learner_decisions':None}


def test_diagnostic_parser_rejects_foreign_future_and_committed_counter_spoofs():
    from gf_common.hpo_progress import TAG, DiagnosticParser
    values=[]
    parser=DiagnosticParser('owned',values.append)
    for changed in ({'invocation':'foreign'},{'update':1},{'facts':{'committed_used':100}},
                    {'schema_version':True},{'sequence':-1},{'facts':{'cuda_allocator':{'reserved_bytes':999}}}):
        value={'schema_version':1,'invocation':'owned','update':0,'sequence':0,'facts':{'phase':'save'},**changed}
        parser.feed((TAG+json.dumps(value)+'\n').encode())
    assert values==[] and parser.sequence==-1


def test_heartbeat_cancellation_does_not_change_owner_or_make_success(tmp_path):
    from gf_cli.hpo_progress import ProgressOwner, read_progress
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        owner=ProgressOwner(store,mode='off',stream=io.StringIO())
        async def run():
            async with owner.live():
                before=owner.value['updated_unix']
                await asyncio.sleep(1.05)
                assert owner.value['updated_unix']>before
        asyncio.run(run())
        assert read_progress(store.directory)['snapshot']['status']=='running'
        store.put('after.json',{'owned':True})


def test_reserved_subtree_only_is_excluded_before_traversal(tmp_path):
    import os
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        (store.directory/'telemetry').mkdir()
        os.mkfifo(store.directory/'telemetry'/'not-an-immutable-record')
        assert store.records()==()
        os.mkfifo(store.directory/'unexpected')
        with pytest.raises(ValueError,match='namespace entry'): store.records()


def test_watch_keyboard_interrupt_stops_observer_only(tmp_path,monkeypatch):
    import gf_cli.hpo as cli
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        def interrupted(coroutine):
            coroutine.close()
            raise KeyboardInterrupt
        monkeypatch.setattr(cli.asyncio,'run',interrupted)
        assert cli.main(['watch','--path',str(store.directory)])==0
        store.put('after-watch.json',{'owned':True})


def test_allocator_sampler_does_not_import_or_synchronize_cuda():
    from types import SimpleNamespace

    from gf_common.hpo_progress import allocator_facts
    class Allocator:
        def memory_allocated(self,device): return 11
        def memory_reserved(self,device): return 22
        def max_memory_allocated(self,device): return 33
        def max_memory_reserved(self,device): return 44
        def synchronize(self,*args): raise AssertionError('display must not synchronize')
    learner=SimpleNamespace(binding=SimpleNamespace(device='cuda:0'),network=SimpleNamespace(device='cuda:0'),
        torch=SimpleNamespace(cuda=Allocator()))
    assert allocator_facts(learner)['cuda_allocator']=={'scope':'process-lifetime','reset':'never',
        'allocated_bytes':11,'reserved_bytes':22,'max_allocated_bytes':33,'max_reserved_bytes':44}
    assert allocator_facts(SimpleNamespace(binding=SimpleNamespace(device='cpu')))=={'cuda_allocator':None}


def test_callback_bursts_remain_in_memory_until_heartbeat_or_phase_change(tmp_path,monkeypatch):
    import gf_cli.hpo_progress as progress
    replacements=[]
    original=progress.os.replace
    def replaced(source,destination):
        replacements.append(destination)
        return original(source,destination)
    monkeypatch.setattr(progress.os,'replace',replaced)
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        owner=progress.ProgressOwner(store,mode='off')
        owner.emit({'phase':'collect'})
        for count in range(1,20): owner.emit({'transient_games':count})
        assert len(replacements)==1
        assert owner.value['transient_games']==19
        owner.emit({'phase':'source'})
        assert len(replacements)==2
        assert progress.read_progress(store.directory)['snapshot']['transient_games']==19


def test_late_save_diagnostic_cannot_rewind_sync_or_cleanup_phase(tmp_path,monkeypatch):
    import gf_training.hpo_gpu_lease as gpu
    from gf_training.hpo_ppo import _HpoOperations
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'gpu')
    values=[]
    with gpu.admit_gpu_job(gpu.JobContext('fixture','phase','training','run',None),global_cuda_bytes=1024,max_workers=4) as job:
        operations=_HpoOperations(job,values.append)
        for phase in ('sync','cleanup'):
            operations.phase(phase)
            operations.learner_progress({'phase':'save','cuda_allocator':None})
            assert 'phase' not in values[-1] and values[-1]['cuda_allocator'] is None
        job.confirm_termination()


@pytest.mark.parametrize('nested',[
    '"latest_metrics":{"loss":NaN}',
    '"latest_metrics":{"entropy":Infinity}',
    '"pool_workers":[{"cuda_allocator":{"allocated_bytes":-Infinity}}]',
    '"phase_seconds":{"samples":[1e10000]}',
])
def test_nonfinite_snapshot_is_invalid_through_actual_json_watch(tmp_path,nested):
    from gf_cli.hpo_progress import read_progress
    telemetry=tmp_path/'telemetry'
    telemetry.mkdir()
    (telemetry/'progress.json').write_text('{"schema_version":1,"kind":"hpo-progress",'
        '"updated_unix":1,"phase":"update","status":"running",'+nested+'}')
    assert read_progress(tmp_path)['availability']=='invalid'
    result=subprocess.run([sys.executable,'-E','-m','gf_cli.hpo','watch','--path',str(tmp_path),
        '--once','--json'],check=False,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['availability']=='invalid' and value['snapshot'] is None
    assert result.stderr==''


@pytest.mark.parametrize('transfer',['before','after','eager'])
def test_optional_heartbeat_factory_failure_retires_originals_and_keeps_body(tmp_path,transfer):
    import inspect

    from gf_cli.hpo_progress import ProgressOwner, read_progress
    coroutines=[]
    tasks=[]
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        stream=io.StringIO()
        owner=ProgressOwner(store,mode='off',stream=stream)
        async def run():
            loop=asyncio.get_running_loop()
            original=loop.get_task_factory()
            def factory(loop,coroutine,**options):
                if coroutine.cr_code.co_name=='heartbeat':
                    coroutines.append(coroutine)
                    if transfer!='before':
                        tasks.append(asyncio.Task(coroutine,loop=loop,eager_start=transfer=='eager'))
                    raise RuntimeError('optional heartbeat task factory failure')
                return asyncio.Task(coroutine,loop=loop,**options)
            loop.set_task_factory(factory)
            entered=0
            try:
                for phase in ('collect','source'):
                    async with owner.live():
                        entered+=1
                        owner.emit({'phase':phase})
                        await asyncio.sleep(0)
            finally:
                loop.set_task_factory(original)
            assert entered==2
        asyncio.run(run())
        assert all(task.done() for task in tasks)
        assert all(inspect.getcoroutinestate(coroutine)==inspect.CORO_CLOSED for coroutine in coroutines)
        assert stream.getvalue().count('heartbeat unavailable')==1
        assert read_progress(store.directory)['snapshot']['phase']=='source'
        store.put('authorized-body.json',{'entered':True})


@pytest.mark.parametrize('creation_failure',[False,True])
def test_external_cancel_during_optional_heartbeat_retirement_is_preserved(tmp_path,creation_failure):
    import inspect

    from gf_cli.hpo_progress import ProgressOwner
    coroutines=[]
    tasks=[]
    with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        owner=ProgressOwner(store,mode='off',stream=io.StringIO())
        async def run():
            loop=asyncio.get_running_loop()
            original=loop.get_task_factory()
            def factory(loop,coroutine,**options):
                task=asyncio.Task(coroutine,loop=loop,**options)
                if coroutine.cr_code.co_name=='heartbeat':
                    coroutines.append(coroutine)
                    tasks.append(task)
                    asyncio.current_task().cancel('caller cancelled')
                    if creation_failure:
                        raise RuntimeError('factory transferred heartbeat then failed while caller cancelled')
                return task
            loop.set_task_factory(factory)
            try:
                with pytest.raises(asyncio.CancelledError,match='caller cancelled'):
                    async with owner.live():
                        await asyncio.sleep(0)
            finally:
                loop.set_task_factory(original)
            assert all(task.done() for task in tasks)
        asyncio.run(run())
        assert all(inspect.getcoroutinestate(coroutine)==inspect.CORO_CLOSED for coroutine in coroutines)


@pytest.mark.parametrize('route',['baseline','metrics','packed','microbatch','shared-prefix'])
def test_optional_execution_fact_is_closed_and_non_authoritative(route):
    from gf_common.hpo_progress import DiagnosticEmitter, DiagnosticParser, learner_facts
    assert learner_facts({'diagnostic_boundary':'update-synchronized','update_execution':route})
    output=io.StringIO()
    DiagnosticEmitter('route',4096,output).emit(2,{'diagnostic_boundary':'update-synchronized','update_execution':route})
    values=[]
    parser=DiagnosticParser('route',values.append)
    parser.update=2
    parser.feed(output.getvalue().encode())
    assert values[0]['update_execution']==route and values[0]['diagnostic_update']==2
    assert not learner_facts({'update_execution':'future-unknown'})
    assert not learner_facts({'update_execution':None})


@pytest.mark.parametrize('log_bytes',[None,'invalid','', '0','-1'])
def test_optional_emitter_setup_disables_invalid_or_missing_budget(log_bytes):
    from gf_common.hpo_progress import optional_diagnostic_emitter
    environment={'GFB_HPO_DIAGNOSTIC_INVOCATION':'setup-test'}
    if log_bytes is not None:
        environment['GFB_HPO_DIAGNOSTIC_LOG_BYTES']=log_bytes
    assert optional_diagnostic_emitter(environment,io.StringIO()) is None


def test_optional_emitter_setup_constructor_failure_and_control_exceptions(monkeypatch):
    from gf_common import hpo_progress
    environment={'GFB_HPO_DIAGNOSTIC_INVOCATION':'setup-test','GFB_HPO_DIAGNOSTIC_LOG_BYTES':'4096'}
    for error in (RuntimeError('display setup'),KeyboardInterrupt(),SystemExit(7)):
        def fail(*args,_error=error,**kwargs):
            raise _error
        monkeypatch.setattr(hpo_progress,'DiagnosticEmitter',fail)
        if isinstance(error,Exception):
            assert hpo_progress.optional_diagnostic_emitter(environment,io.StringIO()) is None
        else:
            with pytest.raises(type(error)) as caught:
                hpo_progress.optional_diagnostic_emitter(environment,io.StringIO())
            assert caught.value is error


@pytest.mark.parametrize('wire',['owned','legacy'])
def test_worker_constructor_display_failure_preserves_original_update_close(tmp_path,monkeypatch,wire):
    from gf_common import hpo_progress
    def fail(*args,**kwargs):
        raise RuntimeError('optional emitter constructor')
    monkeypatch.setattr(hpo_progress,'DiagnosticEmitter',fail)
    monkeypatch.setenv('GFB_HPO_DIAGNOSTIC_INVOCATION','setup-test')
    monkeypatch.setenv('GFB_HPO_DIAGNOSTIC_LOG_BYTES','4096')
    if wire=='owned':
        from tests.test_learning_update_telemetry import (
            test_worker_off_has_no_observer_markers_and_original_phase_order,
        )
        test_worker_off_has_no_observer_markers_and_original_phase_order(tmp_path,monkeypatch,False,None)
    else:
        # Same shared factory as Wire6; its constructor is display-only.
        assert hpo_progress.optional_diagnostic_emitter(dict(__import__('os').environ),io.StringIO()) is None
