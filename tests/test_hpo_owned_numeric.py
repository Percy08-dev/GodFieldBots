"""Producer-owned numerical components; fixtures grant no Source authority."""
from __future__ import annotations

import time

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.group_pack import StorageSettings
from gf_training.group_pack_resources import StorageBudget


def budget():
    return StorageBudget(StorageSettings(), time.monotonic() + 60)


def test_many_array_bankset_keeps_names_order_and_retained_ownership():
    from gf_common.ram_bank import ReadonlyRamBankSet
    account = budget()
    arrays = {f'a.{index:04}': np.asarray([index], dtype=np.int64) for index in reversed(range(600))}
    banks = ReadonlyRamBankSet.build(arrays, account, max_bytes=1024**2)
    assert len(banks.banks) == 3
    assert banks.names == tuple(sorted(arrays))
    assert all(len(bank.descriptors) <= 256 for bank in banks.banks)
    value = banks.array('a.0500')
    np.testing.assert_array_equal(value, [500])
    assert not value.flags.writeable
    with pytest.raises(BufferError):
        banks.close()
    assert account.used > 0
    del value
    banks.close()
    assert account.used == 0


def captured_record(learner, *, count=3):
    """Drive the actual capture protocol with the numerical fixture's originals."""
    import uuid
    from pathlib import Path

    from gf_common.domain import EpisodeGroup, EpisodeResult, EpisodeSpec, GroupResult
    from gf_common.structured import event_delta
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_owned_capture import _new_capture
    from tests.test_card_instance_learning import records_for
    from tests.test_hpo_owned_capture import terminal
    record = records_for(learner, count=count)
    owner = object()
    group = EpisodeGroup('seed', (EpisodeSpec('first', ('same', 'same'), (1, 2), 'pair', 1),))
    plan = CollectionPlan((group,), {}, Path('unused'), {}, float('inf'), float('inf'))
    account = budget()
    snapshot = learner.snapshot_owned(budget=account)
    capture = _new_capture(owner, run_id='run', plan=plan, generation=0,
        snapshot_owner=snapshot, learner_seats={'first': 1},
        learner_policy_version=record.selections[0].policy_version, budget=account)
    capture.open_episode('first', 'game')
    for index, (frame, table, selection) in enumerate(zip(record.frames, record.tables, record.selections, strict=True)):
        delta = event_delta(1, frame.episode, frame.interval_start, frame.interval_start_hash,
            record.bank.events(frame.interval_start, frame.boundary.end), layout=frame.layout)
        command = {'command_id': str(uuid.uuid4()), 'expected_revision': frame.revision,
            'action': selection.validate(table, frame)}
        capture.begin_decision('first', 1, frame, delta, table, selection, command)
        next_revision = record.frames[index+1].revision if index+1 < len(record.frames) else frame.revision+1
        capture.accept_decision('first', 1, command['command_id'], {'command_id': command['command_id'],
            'previous_revision': frame.revision, 'state_revision': next_revision,
            'status': 'active' if index+1 < len(record.frames) else 'terminal',
            'result': None if index+1 < len(record.frames) else record.outcome})
    for viewer in (1, 2):
        observation = terminal(viewer, record.frames[-1].revision + 1)
        observation['result'] = record.outcome
        capture.terminal_observation('first', viewer, observation)
    result = EpisodeResult('first', 'pair', 'seed', ('same', 'same'), game_id='game',
        outcome=record.outcome, end_kind='natural', decisions=count)
    capture.close_episode('first', result)
    capture.close_group(group, GroupResult([result]))
    return capture.seal(owner, plan=plan, snapshot_owner=snapshot), owner, plan, snapshot, account, record


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_owned_capture_packs_same_arrays_and_updates_as_strict_numeric_reader(tmp_path, monkeypatch, kind):
    from dataclasses import replace

    import torch

    from gf_models.arrays import sha256
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.hpo_rollout import HpoRollout, read_hpo_rollout, write_hpo_rollout
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_structured_learning import assert_same_learners
    recipe = replace(HPOTrainingConfig(), epochs=2, minibatch_size=2, tbptt_length=1)
    learner, reference = learner_for(kind, recipe=recipe), learner_for(kind, recipe=recipe)
    batch, owner, plan, snapshot, account, record = captured_record(learner)
    selection = selection_for_counts((3,), learning_seed=17, batch=0, decision_budget=2, used_before=0)
    locations = tuple((0, index) for index in selection["indices"])
    strict = HpoRollout((record,), learner.model, recipe, actions=ACTIONS, rules=RULES,
        selected_locations=locations)
    strict_manifest=write_hpo_rollout(tmp_path/'strict', strict, selection=selection)
    from gf_models.sharded_arrays import read_bundle
    reference_arrays=read_bundle(tmp_path/'strict',strict_manifest['data'])
    strict = read_hpo_rollout(tmp_path/'strict', sha256(tmp_path/'strict'/'rollout.json'),
        learner.model, recipe, actions=ACTIONS, rules=RULES, expected_selection=selection,
        expected_records=(record,))
    owned = encode_owned_batch(batch, owner=owner, plan=plan, snapshot_owner=snapshot,
        model=learner.model, training=recipe, actions=ACTIONS, rules=RULES,
        selected_locations=locations, selection=selection, budget=account)
    assert set(owned.banks.names)==set(reference_arrays)
    for name in reference_arrays:
        np.testing.assert_array_equal(owned.banks.array(name),reference_arrays[name])
    with pytest.raises(ContractError, match='persist'):
        learner.update_owned(owned)
    def forbidden(*args, **kw):
        pytest.fail('owned route called strict archive/numeric reader or parity')
    monkeypatch.setattr('gf_models.sharded_arrays.inspect_bundle',forbidden)
    monkeypatch.setattr('gf_models.sharded_arrays.read_bundle',forbidden)
    monkeypatch.setattr('gf_training.hpo_rollout.read_hpo_rollout',forbidden)
    monkeypatch.setattr('gf_training.structured_rollout._read_rollout',forbidden)
    owned.persist(tmp_path/'owned')
    for index in range(len(strict)):
        assert strict.vector(index) == owned.vector(index)
        frames, candidates = owned.endpoint(index)
        old_frames, old_candidates = strict.endpoint(index)
        for current, old in zip(frames, old_frames, strict=True):
            for field in current.array_fields:
                np.testing.assert_array_equal(getattr(current, field), getattr(old, field))
        for field in candidates.array_fields:
            np.testing.assert_array_equal(getattr(candidates, field), getattr(old_candidates, field))
        with pytest.raises(ContractError):
            learner.network(frames,candidates,suffix_length=1)
        with torch.no_grad():
            expected = reference.network(old_frames, old_candidates, suffix_length=1)
            actual = learner.network._forward_owned(frames, candidates, suffix_length=1)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        reference.network.zero_grad(set_to_none=True)
        learner.network.zero_grad(set_to_none=True)
        strict_logs,strict_value,strict_entropy=reference.network(old_frames,old_candidates,suffix_length=1)
        owned_logs,owned_value,owned_entropy=learner.network._forward_owned(frames,candidates,suffix_length=1)
        (strict_logs.sum()+strict_value+strict_entropy).backward()
        (owned_logs.sum()+owned_value+owned_entropy).backward()
        for left_parameter,right_parameter in zip(reference.network.parameters(),learner.network.parameters(),strict=True):
            if left_parameter.grad is None:
                assert right_parameter.grad is None
            else:
                torch.testing.assert_close(left_parameter.grad,right_parameter.grad,rtol=0,atol=0)
    del frames, candidates, current
    before = torch.get_rng_state().clone()
    expected = reference.update(strict)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    monkeypatch.setattr(learner, 'parity', lambda *_: pytest.fail('owned route called parity'))
    actual = learner.update_owned(owned)
    assert actual['parity'] == {'status': 'not_run'}
    for key in ('policy_loss', 'value_mse', 'entropy', 'approx_kl', 'total_loss', 'optimizer_steps'):
        assert expected[key] == actual[key]
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert_same_learners(reference, learner)
    assert reference.shuffle.bit_generator.state == learner.shuffle.bit_generator.state
    with pytest.raises(ContractError, match='single use'):
        learner.update_owned(owned)
    owned.close()
    batch.close()
    snapshot.close()


@pytest.mark.parametrize('value,valid', [(0.125,True),(0,True),(-1,False),(True,False),(float('nan'),False)])
def test_optional_legacy_parity_duration_is_diagnostic_only(value,valid):
    from gf_common.hpo_progress import learner_facts
    assert learner_facts({'parity_seconds':value}) is valid


def test_owned_wire_layout_rejects_wrong_shape_before_torch_or_mutation(tmp_path):
    from gf_common.domain import ContractError
    from gf_common.ram_bank import ReadonlyRamBankSet
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import _validate_owned_layout, encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    arrays={name:numeric.banks.array(name).copy() for name in numeric.banks.names}
    arrays['e.0.f.0.state_numerics']=np.zeros(1,np.int64)
    broken=ReadonlyRamBankSet.build(arrays,account)
    with pytest.raises(ContractError,match='shape'):
        _validate_owned_layout(numeric.metadata,broken,model=learner.model,actions=ACTIONS)
    assert learner.steps==0
    broken.close()
    numeric.close()
    batch.close()
    snapshot.close()


def test_owned_batch_close_refuses_dispatched_until_actual_completion(tmp_path):
    from gf_common.domain import ContractError
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    numeric.persist(tmp_path/'numeric')
    numeric._begin_update()
    with pytest.raises(ContractError,match='completion'):
        numeric.close()
    # This focused state test has no child or submitted work; original owner
    # explicitly retires its unsubmitted private transaction after refusal.
    numeric._state='FAILED'
    numeric.close()
    batch.close()
    snapshot.close()


def test_post_adam_failure_terminal_refuses_resend(tmp_path,monkeypatch):
    import torch

    from gf_common.domain import ContractError
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    numeric.persist(tmp_path/'numeric')
    before={name:value.clone() for name,value in learner.network.state_dict().items()}
    real=learner.optimizer.step
    def fail_after_actual_adam():
        real()
        raise OSError('after Adam')
    monkeypatch.setattr(learner.optimizer,'step',fail_after_actual_adam)
    with pytest.raises(OSError,match='after Adam'):
        learner.update_owned(numeric)
    assert any(not torch.equal(value,learner.network.state_dict()[name]) for name,value in before.items())
    with pytest.raises(ContractError,match='terminal'):
        learner.update_owned(numeric)
    with pytest.raises(ContractError):
        learner.snapshot_owned(budget=account)
    numeric.close()
    batch.close()
    snapshot.close()


def test_direct_endpoints_do_not_redecode_control_and_exported_views_keep_metadata_charge(tmp_path,monkeypatch):
    import gc

    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner,count=15)
    before=account.used
    selection=selection_for_counts((15,),learning_seed=17,batch=0,decision_budget=15,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=tuple((0,i) for i in range(15)),selection=selection,budget=account)
    charge=numeric.resident_bytes
    before=account.used-charge
    assert len(numeric.banks.banks)>1
    class NoLinearLookup(tuple):
        def __contains__(self,item): pytest.fail("linear full-bank name lookup")
    numeric.banks.names=NoLinearLookup(numeric.banks.names)
    def forbidden(*args,**kw): pytest.fail('direct endpoint decoded control again')
    with monkeypatch.context() as patch:
        patch.setattr('gf_training.hpo_owned_rollout.json.loads',forbidden)
        frames,candidates=numeric.endpoint(14)
        assert len(frames)==15 and numeric.vector(14)[0]>=0
        assert numeric.endpoint(0)[0][0] is frames[0]
        for frame in frames: frame._require_live()
    numeric.close()
    assert account.used==before+charge
    with pytest.raises(ContractError): numeric.require_retired()
    leaf=frames[0].state_numerics[:]
    del frames,candidates,frame
    gc.collect()
    assert account.used==before+charge
    assert leaf.size>0
    del leaf
    gc.collect()
    assert account.used==before
    numeric.require_retired()
    batch.close()
    snapshot.close()


@pytest.mark.parametrize('field',['state_numerics','actor','recipe_signature','_banks','_keys','_scalars','_retention','array_fields'])
def test_owned_tokens_refuse_ordinary_shadow_and_provenance_assignment(field):
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    frames,candidates=numeric.endpoint(0)
    tokens=frames[0]
    with pytest.raises((AttributeError,TypeError)):
        setattr(tokens,field,np.asarray([999],np.int64))
    with pytest.raises(TypeError): tokens._keys['state_numerics']='e.0.target'
    with pytest.raises(TypeError): tokens._scalars['actor']=2
    with pytest.raises((AttributeError,TypeError)): candidates.kinds=np.asarray([999],np.int64)
    numeric.close()
    del frames,candidates,tokens
    numeric.require_retired()
    batch.close()
    snapshot.close()


@pytest.mark.parametrize('alias_kind',['memoryview','direct_array'])
def test_live_numeric_alias_defers_and_retries_positive_retirement(alias_kind):
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    baseline=account.used-numeric.resident_bytes
    alias=(numeric.endpoint(0)[0][0].state_numerics.base.base if alias_kind=='memoryview'
        else numeric.banks.array('e.0.f.0.state_numerics')[1:])
    numeric.close()
    with pytest.raises(ContractError): numeric.require_retired()
    numeric.close()
    assert account.used>baseline
    if alias_kind=='memoryview': alias.release()
    del alias
    numeric.close()
    numeric.require_retired()
    numeric.close()
    numeric.require_retired()
    assert account.used==baseline
    batch.close()
    snapshot.close()


def test_owned_encoding_and_cached_export_require_same_actual_source_budget():
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,_account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    foreign=budget()
    with pytest.raises(ContractError,match='Source'):
        encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
            model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
            selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=foreign)
    with pytest.raises(ContractError,match='Source'): learner.snapshot_owned(budget=foreign)
    assert foreign.used==0
    batch.close()
    snapshot.close()


@pytest.mark.parametrize('legacy',['update','restore'])
def test_legacy_mutation_cannot_be_promoted_to_owned_generation(tmp_path,legacy):
    from tests.test_hpo_numeric import learner_for, rollout_for
    learner=learner_for()
    snapshot=learner.snapshot_owned(budget=budget())
    if legacy=='update': learner.update(rollout_for(learner))
    else:
        learner.save(tmp_path/'old')
        learner.restore(tmp_path/'old')
    with pytest.raises(ContractError,match='legacy'):
        learner.snapshot_owned(budget=snapshot._budget)
    snapshot.close()


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_mixed_zero_population_seat2_loss_draw_win_matches_strict_reader(tmp_path,kind):
    import uuid
    from dataclasses import replace
    from pathlib import Path

    import torch

    from gf_common.domain import EpisodeGroup, EpisodeResult, EpisodeSpec, GroupResult
    from gf_common.structured import (
        CandidateTable,
        DecisionFrame,
        EventBank,
        Selection,
        event_delta,
    )
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    from gf_models.arrays import sha256
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_owned_capture import _new_capture
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.hpo_rollout import HpoRollout, read_hpo_rollout, write_hpo_rollout
    from gf_training.structured_rollout import game_episode_records
    from tests.card_instance_action_support import inputs
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_capture import terminal
    from tests.test_structured_learning import assert_same_learners
    recipe=replace(HPOTrainingConfig(),epochs=1,tbptt_length=1,minibatch_size=2)
    learner,reference=learner_for(kind,recipe=recipe),learner_for(kind,recipe=recipe)
    account=budget()
    snapshot=learner.snapshot_owned(budget=account)
    cases=(('zero',1,0,'draw'),('loss',2,2,{'win':{'winner':1}}),
        ('draw',1,1,'draw'),('win',2,2,{'win':{'winner':2}}))
    group=EpisodeGroup('seed',tuple(EpisodeSpec(name,('same','same'),(1,2),'pair',seat) for name,seat,_,_ in cases))
    plan=CollectionPlan((group,),{},Path('unused'),{},float('inf'),float('inf'))
    owner=object()
    capture=_new_capture(owner,run_id='run',plan=plan,generation=0,snapshot_owner=snapshot,
        learner_seats={name:seat for name,seat,_,_ in cases},learner_policy_version='mixed-fixture',budget=account)
    records=[]
    results=[]
    for name,seat,count,outcome in cases:
        capture.open_episode(name,'game-'+name)
        bank=EventBank(seat,name,layout=layout)
        frames,tables,selections=[],[],[]
        for position in range(count):
            value=inputs([],can_pray=True)
            value.observation.update(viewer_player_id=seat,active_player=seat,current_player=seat,state_revision=position)
            value.observation['own']['player_id']=seat
            value.observation['opponent']['player_id']=3-seat
            value.action_options.update(viewer_player_id=seat,state_revision=position)
            value.action_options['action_options'].update(actor=seat,state_revision=position)
            before=bank.boundary
            delta=event_delta(seat,name,before.end,before.prefix_hash,
                [{'type':'resource_changed','player':seat,'hp_delta':-position,'mp_delta':0,'money_delta':0}],layout=layout)
            bank.append(delta)
            value=replace(value,events=bank.view().events())
            frame=DecisionFrame.from_input(value,bank,before.end,layout=layout)
            table=CandidateTable.from_actions(frame,[{'type':'pray'}],action_contract=ACTIONS.action_contract,
                purpose='qualified_game',synthetic_capacity=False)
            frames.append(frame)
            tokens=tuple(tokenize_frame(row,bank.view(),aggregate_window=learner.model.event.aggregate_window,
                aggregate_recipe=learner.model.event.aggregate_recipe,layout=layout) for row in frames)
            candidates=tokenize_table(table,frame,18,aggregate_window=learner.model.event.aggregate_window,
                aggregate_recipe=learner.model.event.aggregate_recipe,layout=layout)
            with torch.no_grad(): logs,critic,_=learner.network(tokens,candidates)
            selection=Selection(table.table_hash,0,'mixed-fixture',float(logs[0]),float(critic))
            tables.append(table)
            selections.append(selection)
            command={'command_id':str(uuid.uuid4()),'expected_revision':position,'action':{'type':'pray'}}
            capture.begin_decision(name,seat,frame,delta,table,selection,command)
            last=position+1==count
            capture.accept_decision(name,seat,command['command_id'],{'command_id':command['command_id'],
                'previous_revision':position,'state_revision':position+1,'status':'terminal' if last else 'active',
                'result':outcome if last else None})
        for viewer in (1,2):
            observation=terminal(viewer,count)
            observation['result']=outcome
            capture.terminal_observation(name,viewer,observation)
        result=EpisodeResult(name,'pair','seed',('same','same'),game_id='game-'+name,
            outcome=outcome,end_kind='natural',decisions=count)
        capture.close_episode(name,result)
        results.append(result)
        records.append(game_episode_records(bank.view(),tuple(frames),tuple(tables),tuple(selections),
            'natural',outcome,actions=ACTIONS,rules=RULES))
    capture.close_group(group,GroupResult(results))
    batch=capture.seal(owner,plan=plan,snapshot_owner=snapshot)
    selection=selection_for_counts((0,2,1,2),learning_seed=17,batch=0,decision_budget=5,used_before=0)
    from gf_common.hpo import selected_locations
    locations=selected_locations(selection)
    strict=HpoRollout(tuple(records),learner.model,recipe,actions=ACTIONS,rules=RULES,selected_locations=locations)
    write_hpo_rollout(tmp_path/'strict',strict,selection=selection)
    strict=read_hpo_rollout(tmp_path/'strict',sha256(tmp_path/'strict'/'rollout.json'),learner.model,recipe,
        actions=ACTIONS,rules=RULES,expected_selection=selection,expected_records=tuple(records))
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,model=learner.model,
        training=recipe,actions=ACTIONS,rules=RULES,selected_locations=locations,selection=selection,budget=account)
    numeric.persist(tmp_path/'owned')
    assert [len(row['frames']) for row in numeric.metadata['episodes']]==[0,2,1,2]
    assert [row['actor'] for row in numeric.metadata['episodes']]==[1,2,1,2]
    for index in range(len(strict)): assert numeric.vector(index)==strict.vector(index)
    for episode_index in (1,2,3):
        np.testing.assert_array_equal(numeric.banks.array(f'e.{episode_index}.target'),strict.episodes[episode_index].target)
    rng=torch.get_rng_state().clone()
    expected=reference.update(strict)
    expected_rng=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual=learner.update_owned(numeric)
    for key in ('total_loss','policy_loss','value_mse','entropy','approx_kl','optimizer_steps'):
        assert actual[key]==expected[key]
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert_same_learners(reference,learner)
    numeric.close()
    numeric.require_retired()
    batch.close()
    snapshot.close()


def test_uncertain_numeric_close_stays_terminal_and_never_retries_mapping():
    import errno

    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    backing=numeric.banks.banks[0]
    class Uncertain:
        def __init__(self,real): self.real,self.calls=real,0
        def close(self):
            self.calls+=1
            raise OSError(errno.EIO,'uncertain numeric close')
    mapping=Uncertain(backing._mapping)
    backing._mapping=mapping
    charge=account.used
    with pytest.raises(ContractError,match='quarantined'): numeric.close()
    assert account.used==charge
    with pytest.raises(ContractError): numeric.close()
    with pytest.raises(ContractError): numeric.require_retired()
    assert mapping.calls==1
    # Actual owner remains in the production quarantine until this bounded
    # test process is reaped; no forced retry/refund of historical resources.
    batch.close()
    snapshot.close()
