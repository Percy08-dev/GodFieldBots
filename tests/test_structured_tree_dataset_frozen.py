"""Bounded read-only checks on the preserved actual9-cohort component originals."""
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.domain import canonical
from gf_common.preprocessing import PreparationConfig
from gf_training.native_tree_dataset import NativeTreeDataset
from gf_training.structured_tree_data import build_component_dataset, inspect_component_dataset
from tests.test_structured_tree_dataset import dataset_plan

native=pytest.importorskip('gf_native')
pytestmark=pytest.mark.skipif(not hasattr(native,'TreeDatasetSession'),reason='Dataset2 candidate required')


@pytest.fixture(scope='module')
def frozen():
    directory=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not directory:pytest.skip('explicit actual9-cohort original fixture required')
    evidence=json.loads((Path(directory)/'fixture-evidence.json').read_bytes())
    parents=[Path(p) for p in evidence['parents']];validations=[Path(p) for p in evidence['validations']]
    settings=PreparationConfig(backend='rust',reader='structured-v1',memory_budget_mib=2048,cache_budget_mib=512,chunk_target_mib=16)
    return parents,validations,settings


def test_readonly_all9_originals_manifest_build_and_inspect_after_origin_fix(tmp_path,frozen):
    parents,validations,settings=frozen
    plan=dataset_plan(parents,validations)
    value=build_component_dataset(plan,tmp_path/'dataset',settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+300,native=native)
    assert len(value['collections'])==9 and value['counts']['validation']['games']==192
    checked=inspect_component_dataset(tmp_path/'dataset/manifest.json',expected_digest=hashlib.sha256((tmp_path/'dataset/manifest.json').read_bytes()).hexdigest(),
        settings=settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+300,native=native)
    assert canonical(checked)==canonical(value)


def test_all_selected_native_facts_rows_and_natural_labels_match_independent_python_reader(frozen):
    import numpy as np

    from gf_common.action_contract import engineering_contract
    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    from gf_models.structured_tree_config import ConfiguredTreeModelV2, TreeEncoderConfigV2
    from gf_models.structured_tree_encoder import PreparedTreeFrame
    from gf_training.structured_journal import read_structured_journal
    parents,validations,settings=frozen;plan=dataset_plan(parents,validations)
    rows=0;history_diff=0;nonzero=0;label_values=set()
    with NativeTreeDataset(plan,settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+300,native=native) as owner:
        while owner.stage_next():
            work=owner.summary['worklist'][owner._position]
            directory=Path(work['directory']);manifest=json.loads((directory/'manifest.json').read_bytes())
            # Independent Python Source reference is TEST ONLY, never a native
            # authorization implementation. Admit it before constructing it.
            estimate=65536+32*sum(p.stat().st_size for p in directory.rglob('*') if p.is_file() and p.suffix in ('.jsonl','.columns'))
            owner.reserve_sink(estimate)
            references={}
            for name,viewer in manifest['ppo']['learner_seats'].items():
                raw=read_structured_journal(directory/name/f'player-{viewer}.jsonl',action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
                for index in raw.accepted:
                    decision=raw.decisions[index]
                    references[(decision.frame.episode,decision.frame.revision)]=(raw,decision)
            for sample in owner.selected_inputs():
                raw,decision=references[(sample.frame.episode,sample.frame.revision)]
                frame=tokenize_frame(decision.frame,raw.bank,aggregate_window=1)
                full=tokenize_table(decision.generated.table,decision.frame,18,aggregate_window=1)
                selected=decision.selection.index;first,end=full.offsets[selected:selected+2]
                candidate=replace(full,kinds=(full.kinds[selected],),targets=(full.targets[selected],),
                    numerics=(full.numerics[selected],),present=(full.present[selected],),offsets=(0,end-first),slots=full.slots[first:end])
                assert sample.frame==frame and sample.candidate==candidate and sample.original_selection==decision.selection
                outcome=raw.end['outcome'];expected=0 if outcome=='draw' else 1 if outcome['win']['winner']==frame.actor else -1
                assert sample.label==expected;label_values.add(expected)
                model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
                actual=next(PreparedTreeFrame.create(sample.frame,model).iter_chunks(sample.candidate))[1]
                reference=next(PreparedTreeFrame.create(frame,model).iter_chunks(candidate))[1]
                np.testing.assert_array_equal(actual,reference)
                history_diff+=frame.history!=tokenize_frame(decision.frame,raw.bank,aggregate_window=128).history
                nonzero+=bool(selected);rows+=1
                del sample,frame,candidate,actual,reference
            references.clear();del references,raw,decision,manifest
            owner.reserve_sink(0);owner.retire_current()
        assert rows==512 and history_diff>0 and nonzero>0 and label_values=={-1,0,1}
        owner.verify_complete()


@pytest.mark.parametrize('mutation',['omit_parent','parent_order','caller_subset'])
def test_native_dataset_derives_complete_worklist_not_caller_subset(frozen,mutation):
    parents,validations,_settings=frozen;plan=dataset_plan(parents,validations)
    if mutation=='omit_parent':plan['parents'].pop()
    elif mutation=='parent_order':plan['parents'][0],plan['parents'][1]=plan['parents'][1],plan['parents'][0]
    else:plan['worklist']=[]
    encoded=canonical({'memory_bytes':512*1024**2,'scratch_bytes':1024**3,'max_source_file_bytes':64*1024**2,'max_record_bytes':8*1024**2,'chunk_target_bytes':16*1024**2}).encode()
    with pytest.raises(ValueError):
        native.TreeDatasetSession(canonical(plan).encode(),encoded,time.monotonic()+120)


def test_current_partial_payload_no_advance_or_finalize_and_cancel_reaps_owner(frozen):
    from gf_common.domain import ContractError
    parents,validations,settings=frozen;plan=dataset_plan(parents,validations)
    owner=NativeTreeDataset(plan,settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+300,native=native)
    try:
        owner.__enter__()
        with pytest.raises((ValueError,RuntimeError)):owner._session.finalize(owner._held())
        assert owner.stage_next()
        with pytest.raises(ContractError):owner.stage_next()
        values=owner.selected_inputs();sample=next(values)
        assert sample.candidate.offsets[0]==0 and len(sample.candidate.kinds)==1
        with pytest.raises(ContractError):owner.retire_current()
        values.close();del sample,values
        with pytest.raises(ContractError):owner.retire_current()  # a smaller consumed prefix cannot advance
        owner._session.cancel_handle().cancel()
        with pytest.raises(RuntimeError,match="cancelled"):owner._session.recheck(owner._held())
    finally:owner.close()
    assert owner._closed and owner._executor._closed and owner._executor._active is None
    assert owner._session.used_bytes()==0


def test_late_source_mutation_after_retirement_invalidates_whole_pending_manifest(tmp_path,frozen,monkeypatch):
    import shutil

    import gf_training.structured_tree_data as module
    parents,validations,settings=frozen
    disposable=tmp_path/'disposable-derived-validation-not-original'
    shutil.copytree(validations[0],disposable)
    plan=dataset_plan(parents,[disposable,*validations[1:]])
    original=NativeTreeDataset.retire_current
    changed=[]
    def retire_then_mutate(self):
        original(self)
        if self._position==7:
            # This is a disposable COPY, never a mutated/repaired healthy
            # original. All six original root files remain byte/stat unchanged.
            path=disposable/'report.json';path.write_bytes(path.read_bytes()+b'\n')
            changed.append(path)
    monkeypatch.setattr(NativeTreeDataset,'retire_current',retire_then_mutate)
    with pytest.raises(ValueError,match='changed|snapshot|original'):
        module.build_component_dataset(plan,tmp_path/'must-not-publish',settings,max_owned_bytes=2*1024**3,
                                       deadline=time.monotonic()+300,native=native)
    assert changed and not (tmp_path/'must-not-publish').exists()


def test_old62_metadata_is_not_active_dataset_source(frozen):
    parents,validations,_settings=frozen
    # Original historical bits are read-only; no .lock creation or repair in
    # the golden fixture. Its missing raw Source is never called a positive.
    old=Path(__file__).parent/'fixtures/structured_history62_parent'
    plan=dataset_plan([old.resolve(),*parents[1:]],validations)
    encoded=canonical({'memory_bytes':512*1024**2,'scratch_bytes':1024**3,'max_source_file_bytes':64*1024**2,'max_record_bytes':8*1024**2,'chunk_target_bytes':16*1024**2}).encode()
    with pytest.raises(ValueError,match='aggregate_recipe|old62|current376'):
        native.TreeDatasetSession(canonical(plan).encode(),encoded,time.monotonic()+120)
