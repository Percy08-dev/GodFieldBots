"""External audit capacity is independent of the recorded live receiver phase."""
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, clone
from gf_training.group_pack_resources import ByteBudget

MIB=1024**2
GIB=1024**3
OBSERVED_C16_DECODE=583949647


@pytest.fixture
def adapter_case(tmp_path,monkeypatch):
    from gf_training import card_instance_source, structured_source
    from gf_training.hpo_owned_restore import _audit_owned_cohort
    # Unit adapter boundary: original parsing/correspondence is owned by those
    # readers and is separately exercised with real Numeric6 below.
    record=SimpleNamespace(frames=(1,2,3))
    episodes=[SimpleNamespace(records=record,episode_id=f'e{i}',viewer=1) for i in range(8)]
    collection=SimpleNamespace(episodes=episodes,manifest={'original':'collection'})
    selection={'episode_counts':[3]*8}
    update={'collection_manifest_sha256':'a'*64,'numeric':'numeric/batch-000000','numeric_sha256':'b'*64,
        'source_proof':{'selection':selection}}
    state={'effective_config':{'hpo_spec':{'storage':{},'operating':{'numeric_budget_bytes':512*MIB}}}}
    pool=ByteBudget(2*GIB,base=256*MIB)
    class Original:
        used=128*MIB
        finalized=False
        checks=0
        def __init__(self):self.files={'manifest.json':{'sha256':'a'*64}}
        def charge(self,size):
            pool.acquire(size,time.monotonic()+10)
            self.used+=size
        def check(self):self.checks+=1
        def finalize(self):self.finalized=True
    original=Original()
    original.pool=SimpleNamespace(budget=pool)
    scan=SimpleNamespace(root=tmp_path,retained=64*MIB,limit=2*GIB,object=lambda *args:update)
    plan=SimpleNamespace(linkage={'learner_seats':{row.episode_id:1 for row in episodes}})
    model=SimpleNamespace(resources=SimpleNamespace(max_owned_bytes=3*GIB))
    recipe=SimpleNamespace(collection_games=8)
    anchored=[]
    anchor=SimpleNamespace(collection=anchored.append,rollout=anchored.append)
    monkeypatch.setattr(structured_source,'_inspect_component_collection',lambda *args,**kwargs:collection)
    monkeypatch.setattr(card_instance_source,'_check_collection_parent',lambda *args:None)
    def invoke():return _audit_owned_cohort(scan,state,[{'snapshot_identity':'c'*64},
        {'update_sha256':'d'*64}],0,original,plan,'frozen-policy',model,recipe,object(),anchor)
    return SimpleNamespace(invoke=invoke,scan=scan,original=original,pool=pool,state=state,
        model=model,recipe=recipe,anchored=anchored,record=record,selection=selection)


def test_audit_uses_remaining_caller_capacity_without_changing_live512_or_peak_account(adapter_case,monkeypatch):
    from gf_training import hpo_rollout
    case=adapter_case
    before=clone(case.state)
    pool_before=case.pool.used
    extra=case.scan.retained+case.original.used
    capacity=min(case.scan.limit-extra,case.pool.limit-case.pool.used)
    checked=[]
    class Result:
        policy_version='frozen-policy'
        def __len__(self):return 24
        def require_learning(self):checked.append('original numeric result')
    def decoded(directory,sha,model,recipe,**options):
        assert options['numeric_budget_bytes']==capacity>512*MIB
        assert options['max_owned_bytes']==case.scan.limit and options['extra_owned_bytes']==extra
        assert options['expected_records']==(case.record,)*8
        assert options['expected_selection'] is case.selection and options['expected_snapshot_identity']=='c'*64
        # Exercise the actual previously failing C16 reservation without allocating
        # synthetic hundreds of MB; repeated absolute reserves must be idempotent.
        options['reserve'](OBSERVED_C16_DECODE)
        options['reserve'](OBSERVED_C16_DECODE)
        return Result()
    monkeypatch.setattr(hpo_rollout,'read_hpo_rollout',decoded)
    assert case.invoke()==(8,24)
    assert case.state==before and case.state['effective_config']['hpo_spec']['operating']['numeric_budget_bytes']==512*MIB
    assert case.pool.used==pool_before+OBSERVED_C16_DECODE
    assert checked and case.original.finalized


@pytest.mark.parametrize('exhausted',['joint','pool','model'])
def test_exhausted_audit_ownership_refuses_before_decoder_entry(adapter_case,monkeypatch,exhausted):
    from gf_training import hpo_rollout
    case=adapter_case
    if exhausted=='joint':case.original.used=case.scan.limit-case.scan.retained
    elif exhausted=='pool':case.pool.acquire(case.pool.limit-case.pool.used,time.monotonic()+10)
    else:case.model.resources.max_owned_bytes=case.scan.retained+case.original.used
    def forbidden(*args,**kwargs):pytest.fail('audit entered decoder without remaining caller capacity')
    monkeypatch.setattr(hpo_rollout,'read_hpo_rollout',forbidden)
    before=case.pool.used
    with pytest.raises(ContractError,match='audit.*capacity'):case.invoke()
    assert case.pool.used==before


@pytest.fixture
def real_numeric6(tmp_path):
    pytest.importorskip('torch')
    from gf_models.arrays import sha256
    from gf_models.sharded_arrays import read_json, validate_bundle_metadata
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.structured_rollout import _read_reservation
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import captured_record
    learner=learner_for()
    batch,owner,plan,snapshot,account,record=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    from tests.test_card_instance_learning import ACTIONS, RULES
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    try:
        numeric.persist(tmp_path/'numeric')
        manifest=read_json(tmp_path/'numeric/rollout.json')
        base=_read_reservation(manifest,validate_bundle_metadata(manifest['data']))
        sizes={name:(tmp_path/'numeric'/name).stat().st_size for name in manifest['data']['files']}
        required=base+32*sum(sizes.values())+65536
        case=SimpleNamespace(root=tmp_path/'numeric',sha=sha256(tmp_path/'numeric/rollout.json'),
            learner=learner,record=record,selection=selection,snapshot=snapshot,required=required,actions=ACTIONS,rules=RULES)
        yield case
    finally:
        numeric.close();batch.close();snapshot.close()
        case.record=None
        record=None
        import gc
        gc.collect()  # Exported fixture frames retain the actual capture account until dropped.
    assert account.used==0


@pytest.mark.parametrize('fits',[True,False])
def test_numeric6_strict_decoder_preserves_correspondence_and_refuses_before_arrays(real_numeric6,monkeypatch,fits):
    from gf_models import arrays
    from gf_training.hpo_owned_restore import _AuditNumericPeak
    from gf_training.hpo_rollout import read_hpo_rollout
    case=real_numeric6
    pool=ByteBudget(512*MIB)
    class Original:
        def charge(self,size):pool.acquire(size,time.monotonic()+10)
        def check(self):pass
    peak=_AuditNumericPeak(Original())
    def forbidden(*args,**kwargs):pytest.fail('insufficient audit decode opened arrays')
    if not fits:monkeypatch.setattr(arrays,'load_arrays',forbidden)
    def read():return read_hpo_rollout(case.root,case.sha,case.learner.model,case.learner.training,
        actions=case.actions,rules=case.rules,expected_selection=case.selection,
        expected_records=(case.record,),expected_snapshot_identity=case.snapshot.identity,
        max_owned_bytes=512*MIB,numeric_budget_bytes=case.required if fits else case.required-1,reserve=peak)
    if not fits:
        with pytest.raises(ContractError,match='before arrays'):read()
        assert peak.peak==pool.used==0
    else:
        result=read()
        assert len(result)==3 and result.episodes[0].records is case.record
        result.require_learning()
        assert peak.peak==pool.used==case.required
