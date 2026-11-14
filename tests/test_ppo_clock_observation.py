"""Bracketed observations and owned CPU failure paths; no system clock writes."""
import math
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError


def scripted_clock(monkeypatch, brackets):
    import gf_training.ppo_run as coordinator
    mono=iter(value for before,_wall,after in brackets for value in (before,after))
    wall=iter(value for _before,value,_after in brackets)
    reads=[]
    def monotonic():
        reads.append('m')
        return next(mono)
    def unix():
        reads.append('w')
        return next(wall)
    monkeypatch.setattr(coordinator,'time',SimpleNamespace(monotonic=monotonic,time=unix))
    return reads


@pytest.mark.parametrize('anchor,current',[
    ((0.,100.,.02),(10.,110.01,10.02)),
    ((0.,100.02,.02),(10.,110.,10.02)),
    ((0.,100.,0.),(1.,102.,1.)),  # Persistent forward adjustment.
    ((0.,0.,0.),(0.,-.1,0.)),  # Exact zero-width tolerance boundary.
    ((0.,100.,.025),(.025,100.025,.05)),
])
def test_precise_constant_offset_forward_and_exact_boundary_accept(monkeypatch,anchor,current):
    from gf_training.ppo_run import _checked_wall_clock, _validate_clock_anchor
    _validate_clock_anchor(*anchor)
    reads=scripted_clock(monkeypatch,[current])
    assert _checked_wall_clock(anchor,scope='invocation')==current[1]
    assert reads==['m','w','m']


@pytest.mark.parametrize('anchor',[(0.,100.,.026),(1.,100.,0.),(math.nan,100.,0.),(0.,math.inf,0.),(0.,100.,math.inf)])
def test_original_anchor_imprecision_is_uncertainty_without_resampling(monkeypatch,anchor):
    from gf_training.ppo_run import _checked_wall_clock, _validate_clock_anchor
    reads=scripted_clock(monkeypatch,[])
    with pytest.raises(ContractError,match='uncertain'): _validate_clock_anchor(*anchor)
    with pytest.raises(ContractError,match='uncertain'): _checked_wall_clock(anchor,scope='cleanup')
    assert reads==[]


def test_old_read_order_false_positive_is_not_evidence_of_a_clock_step():
    from gf_training.ppo_run import _validate_clock_anchor
    # Constant offset100: the initial wall read occurs .2 after m0, and the
    # old current monotonic read occurs .2 after its wall read. No wall step.
    m0,w0,m1=0.,100.2,.2
    current_wall,current_mono=110.,10.2
    assert current_wall < w0+(current_mono-m0)-.1
    with pytest.raises(ContractError,match='uncertain') as raised:
        _validate_clock_anchor(m0,w0,m1)
    assert 'rollback' not in str(raised.value)


@pytest.mark.parametrize('first',[(10.,110.,10.2),(10.,math.nan,10.),(10.,110.,9.),(math.inf,110.,math.inf)])
def test_current_imprecision_retries_without_reanchor(monkeypatch,first):
    from gf_training.ppo_run import _checked_wall_clock
    reads=scripted_clock(monkeypatch,[first,(11.,111.,11.)])
    assert _checked_wall_clock((0.,100.,0.),scope='invocation')==111.
    assert reads==['m','w','m']*2


def test_samples_before_original_anchor_are_uncertain_not_wall_rollback(monkeypatch):
    from gf_training.ppo_run import _checked_wall_clock
    reads=scripted_clock(monkeypatch,[(50.,150.,50.)]*3)
    with pytest.raises(ContractError,match='uncertain') as raised:
        _checked_wall_clock((100.,200.,100.),scope='invocation')
    assert 'rollback' not in str(raised.value)
    assert reads==['m','w','m']*3


@pytest.mark.parametrize('first,frontier',[
    ((10.,110.,10.2),10.2),  # Imprecise width still observes the later after.
    ((12.,math.nan,11.),12.),  # A decreasing bracket still observes its before.
    ((math.nan,110.,12.),12.),
    ((12.,110.,math.nan),12.),
])
def test_retries_follow_all_finite_monotonic_observations(monkeypatch,first,frontier):
    from gf_training.ppo_run import _checked_wall_clock
    stale=frontier-.1
    later=frontier+1.
    reads=scripted_clock(monkeypatch,[first,(stale,100.+stale,stale),(later,100.+later,later)])
    assert _checked_wall_clock((0.,100.,0.),scope='invocation')==100.+later
    assert reads==['m','w','m']*3


def test_invalid_retry_advances_frontier_and_cannot_enable_earlier_third_sample(monkeypatch):
    from gf_training.ppo_run import _checked_wall_clock
    reads=scripted_clock(monkeypatch,[(10.,110.,10.2),(10.1,110.1,11.),(10.9,110.9,10.9)])
    with pytest.raises(ContractError,match='uncertain') as raised:
        _checked_wall_clock((0.,100.,0.),scope='cleanup')
    assert 'rollback' not in str(raised.value)
    assert reads==['m','w','m']*3


def test_persistent_backward_step_is_proven_and_stops_first_sample(monkeypatch):
    from gf_training.ppo_run import _checked_wall_clock
    reads=scripted_clock(monkeypatch,[(10.,109.8,10.01)])
    with pytest.raises(ContractError,match='rollback'):
        _checked_wall_clock((0.,100.,.01),scope='invocation')
    assert reads==['m','w','m']


@pytest.mark.parametrize('bracket',[(10.,110.,10.2),(10.,109.9,10.02)])
def test_exhausted_imprecision_or_boundary_ambiguity_is_uncertainty(monkeypatch,bracket):
    from gf_training.ppo_run import _checked_wall_clock
    # Each retry is chronologically later, independently testing width/ambiguity.
    reads=scripted_clock(monkeypatch,[tuple(value+step for value in bracket) for step in range(3)])
    with pytest.raises(ContractError,match='uncertain'):
        _checked_wall_clock((0.,100.,.01),scope='cleanup')
    assert reads==['m','w','m']*3


def test_ambiguous_step_can_resolve_but_cannot_hide_persistent_loss(monkeypatch):
    from gf_training.ppo_run import _checked_wall_clock
    # True loss beyond .1 must never be accepted; a narrower second bracket
    # proves the loss instead of treating the first overlap as a valid clock.
    reads=scripted_clock(monkeypatch,[(10.,109.9,10.02),(11.,110.88,11.)])
    with pytest.raises(ContractError,match='rollback'):
        _checked_wall_clock((0.,100.,.01),scope='invocation')
    assert reads==['m','w','m']*2


@pytest.mark.parametrize('mode',['anchor','invocation','cleanup','primary-plus-cleanup'])
def test_actual_cpu_clock_failures_keep_owned_close_and_primary(tmp_path,monkeypatch,mode):
    pytest.importorskip('torch')
    import gf_training.ppo_run as coordinator
    original_check=coordinator._checked_wall_clock
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_learning_process import HpoLearningProcess
    from gf_training.hpo_ppo import _HpoOperations
    from gf_training.lineage import _OWNED_LOCKS
    from tests.test_hpo_coordinator import collector_for, controlled_run, invoke
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=1)
    original_make=_HpoOperations.make_child
    children=[]
    def make(self,*args,**kwargs):
        child,ledger=original_make(self,*args,**kwargs)
        children.append(child)
        return child,ledger
    monkeypatch.setattr(_HpoOperations,'make_child',make)
    actual_close=HpoLearningProcess.close
    closed=[]
    async def close(self,deadline):
        await actual_close(self,deadline)
        closed.append(self)
    monkeypatch.setattr(HpoLearningProcess,'close',close)
    clock_error=ContractError('clock observation uncertain: controlled CPU failure')
    primary=RuntimeError('original controlled collector failure')
    checked=[]
    if mode=='anchor':
        def invalid(*anchor):
            assert children and tmp_path/'parent' in _OWNED_LOCKS
            assert children[0].process is None
            raise clock_error
        monkeypatch.setattr(coordinator,'_validate_clock_anchor',invalid)
    else:
        def check(anchor,*,scope):
            if (mode=='invocation' and scope=='invocation'
                    or mode in ('cleanup','primary-plus-cleanup') and scope=='cleanup'):
                raise clock_error
            result=original_check(anchor,scope=scope)
            checked.append(result)
            return result
        monkeypatch.setattr(coordinator,'_checked_wall_clock',check)
    collector=collector_for(loaded,model,recipe,binding)
    if mode=='primary-plus-cleanup':
        async def collector(*args): raise primary
    expected=primary if mode=='primary-plus-cleanup' else clock_error
    with pytest.raises(type(expected)) as raised:
        invoke(loaded,collector,tmp_path/'parent',run_id)
    assert raised.value is expected
    assert closed==children and len(children)==1
    child=children[0]
    assert child._terminal_closed and child.ledger.learner is None and not child.ledger.workers
    if mode=='anchor': assert child.process is None and child.creation_task is None
    else: assert child.process.returncode==0 and child.lease.released
    state=read_json(tmp_path/'parent'/'run.json')
    assert state['status']=='failed' and state['quarantine'] is True and len(state['invocations'])==1
    if mode in ('cleanup','primary-plus-cleanup'):
        assert state['last_unix']==max(state['created_unix'],*checked)
    if mode=='primary-plus-cleanup':
        assert any('clock' in note for note in primary.__notes__)


def test_actual_cpu_checked_wall_highwater_is_not_unchecked_or_lower_cleanup(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    import gf_training.ppo_run as coordinator
    original_check=coordinator._checked_wall_clock
    from gf_models.sharded_arrays import read_json
    from tests.test_hpo_coordinator import collector_for, controlled_run, invoke
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=1)
    values=[]
    def check(anchor,*,scope):
        if scope=='cleanup': return max(values)-.05
        result=original_check(anchor,scope=scope)
        values.append(result)
        return result
    monkeypatch.setattr(coordinator,'_checked_wall_clock',check)
    result=invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    state=read_json(tmp_path/'parent'/'run.json')
    assert result.report['status']=='complete' and state['device']=='cpu'
    assert state['last_unix']==max(values)
    assert state['elapsed_seconds']==state['last_unix']-state['created_unix']
