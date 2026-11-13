"""Wire6 numerical component execution; fixtures grant no online Source authority."""
import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, canonical, digest
from gf_models.arrays import sha256
from gf_training.hpo_collection import selection_for_counts
from gf_training.hpo_numeric_worker import digest_projection, validate_projection, wire_identity
from gf_training.hpo_rollout import HpoRollout, write_hpo_rollout
from gf_training.structured_rollout import (
    _episode_correspondence_digest,
    _hpo_semantics,
    _source_correspondence,
)
from tests.test_card_instance_learning import ACTIONS, RULES, records_for
from tests.test_hpo_numeric import learner_for


def numerical_request(tmp_path):
    learner = learner_for('gru')
    records = (records_for(learner, count=3),)
    selection = selection_for_counts((3,), learning_seed=17, batch=0, decision_budget=1, used_before=0)
    # The component wire still requires the declared complete collection size.
    from dataclasses import replace

    from gf_training.hpo_config import HPOTrainingConfig
    recipe = replace(HPOTrainingConfig(), collection_games=8)
    learner.training = recipe
    from gf_common.structured import EventBank
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_training.structured_rollout import game_episode_records
    records = (*records, *(game_episode_records(EventBank(1, f'empty-{index}',
        layout=CORRECTED_CARD_INSTANCE_LAYOUT).view(), (), (), (), 'natural', 'draw',
        actions=ACTIONS, rules=RULES) for index in range(7)))
    selection = selection_for_counts(tuple(len(record.frames) for record in records),
        learning_seed=17, batch=0, decision_budget=1, used_before=0)
    rollout = HpoRollout(records, learner.model, recipe, actions=ACTIONS, rules=RULES,
        selected_locations=tuple((0, index) for index in selection['indices']))
    manifest = write_hpo_rollout(tmp_path / 'numeric', rollout, selection=selection)
    files = {'rollout.json': sha256(tmp_path / 'numeric/rollout.json'), **manifest['data']['files']}
    correspondence = _source_correspondence('b'*64, _hpo_semantics(learner.model, rollout._context),
        tuple(_episode_correspondence_digest(record) for record in records), game=True, hpo=True)
    projection = {'recipe': 'card-instance-hpo-source-update-v1', 'parent_run_id': 'unqualified-component',
        'batch_id': 'batch-000000', 'collection_generation': 0, 'expected_optimizer_steps': 0,
        'expected_policy_version': rollout.policy_version, 'collection_manifest_sha256': 'c'*64,
        'numeric_manifest_sha256': files['rollout.json'], 'numeric_files_sha256': digest(dict(sorted(files.items()))),
        'source_correspondence': correspondence, 'training_recipe': asdict(recipe), 'selection': selection,
        'collected_decisions': 3, 'used_decisions': 1, 'expected_optimizer_step_delta': 4}
    identity = wire_identity(learner.model, recipe, learner.binding, learner.action_config_hash,
        learner.resume_identity, actions=ACTIONS, cuda_visible_devices='')
    common = {'protocol_version': 6, 'identity': identity}
    requests = [{**common, 'command': 'initialize', 'model': asdict(learner.model), 'seed': 17,
        'output': str(tmp_path / 'g0'), 'restore': None},
        {**common, 'command': 'update', 'batch': str(tmp_path / 'numeric'), 'batch_digest': files['rollout.json'],
         'expected_policy_version': rollout.policy_version, 'output': str(tmp_path / 'g1'),
         'transaction_id': digest_projection(projection), 'projection': projection},
        {**common, 'command': 'close'}]
    return learner, rollout, requests


def test_actual_cpu_wire6_initialize_selected_update_full_ack_and_close(tmp_path):
    _, _, requests = numerical_request(tmp_path)
    result = subprocess.run([sys.executable, '-E', '-m', 'gf_training.ppo_worker'],
        input=''.join(canonical(request)+'\n' for request in requests), text=True,
        capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [reply['command'] for reply in replies] == ['initialize', 'update', 'close']
    assert replies[1]['artifact']['optimizer_steps'] == 4
    assert replies[1]['metrics']['collected_decisions'] == 3
    assert replies[1]['metrics']['used_decisions'] == 1
    restore = dict(requests[0], restore=str(tmp_path/'g1'), output=str(tmp_path/'unused'))
    restored = subprocess.run([sys.executable, '-E', '-m', 'gf_training.ppo_worker'],
        input=canonical(restore)+'\n'+canonical(requests[-1])+'\n', text=True,
        capture_output=True, timeout=60, check=False)
    assert restored.returncode == 0, restored.stderr
    assert json.loads(restored.stdout.splitlines()[0])['artifact']['optimizer_steps'] == 4
    assert not (tmp_path/'unused').exists()


def test_wire6_wrong_recipe_and_counts_reject_before_update(tmp_path):
    learner, _, requests = numerical_request(tmp_path)
    projection = requests[1]['projection']
    projection['used_decisions'] += 1
    with pytest.raises(ContractError):
        validate_projection(projection, expected_steps=0, policy=requests[1]['expected_policy_version'],
            batch_digest=requests[1]['batch_digest'], training=learner.training)


@pytest.mark.parametrize('numeric_limit',[1,64*1024**2])
def test_actual_cpu_serve_admits_first_manifest_and_releases_preliminary_copies(tmp_path,numeric_limit):
    _, _, requests=numerical_request(tmp_path)
    for request in requests:
        request['identity']['phase_limits']['numeric_budget_bytes']=numeric_limit
    # A clean process preserves actual visibility selection, CPU learner/save,
    # request validation, file custody and decoder/optimizer behavior.
    probe=r'''
import hashlib,inspect,json,sys,weakref
from pathlib import Path
from gf_common.domain import ContractError
from gf_models import sharded_arrays
from gf_training import hpo_rollout,structured_numeric_worker
from gf_training.hpo_numeric import HpoLearner
requests=json.loads(sys.stdin.read())
batch_digest=requests[1]['batch_digest']
report={'parsed':0,'optimizer_calls':0,'decoder_calls':0,'budgets':[],'descriptor_limits':[],'released':False,'error':None}
refs=[]
original_json=json.loads
def observed_json(raw,*args,**kwargs):
    data=raw.encode() if isinstance(raw,str) else raw
    if hashlib.sha256(data).hexdigest()==batch_digest:
        report['parsed']+=1
    return original_json(raw,*args,**kwargs)
json.loads=observed_json
original_read=sharded_arrays.read_json
class Manifest(dict):
    pass
def observed_read(path,*args,**kwargs):
    if Path(path).name!='rollout.json':
        return original_read(path,*args,**kwargs)
    report['budgets'].append(kwargs.get('max_owned_bytes'))
    value=Manifest(original_read(path,*args,**kwargs))
    refs.append(weakref.ref(value))
    return value
sharded_arrays.read_json=observed_read
original_descriptors=sharded_arrays.validate_bundle_metadata
def observed_descriptors(*args,**kwargs):
    if report['budgets']:
        report['descriptor_limits'].append(kwargs.get('max_decoded_bytes'))
    return original_descriptors(*args,**kwargs)
sharded_arrays.validate_bundle_metadata=observed_descriptors
original_decode=hpo_rollout.read_hpo_rollout
def observed_decode(*args,**kwargs):
    report['decoder_calls']+=1
    caller=inspect.currentframe().f_back
    report['released']=refs[0]() is None and all(name not in caller.f_locals for name in ('manifest','files','descriptors'))
    return original_decode(*args,**kwargs)
hpo_rollout.read_hpo_rollout=observed_decode
original_update=HpoLearner.update
def observed_update(self,*args,**kwargs):
    report['optimizer_calls']+=1
    return original_update(self,*args,**kwargs)
HpoLearner.update=observed_update
try:
    structured_numeric_worker.serve(requests[0],(json.dumps(row) for row in requests[1:]))
except ContractError as error:
    report['error']=str(error)
print('PROBE:'+json.dumps(report),flush=True)
'''
    result=subprocess.run([sys.executable,'-E','-c',probe],input=json.dumps(requests),
        text=True,capture_output=True,timeout=60,check=False)
    assert result.returncode==0,result.stderr
    lines=result.stdout.splitlines()
    report=json.loads(next(line.removeprefix('PROBE:') for line in lines if line.startswith('PROBE:')))
    replies=[json.loads(line) for line in lines if not line.startswith('PROBE:')]
    assert report['budgets'][0]==numeric_limit
    if numeric_limit==1:
        assert report['error'] and 'before parse' in report['error']
        assert report['parsed']==report['decoder_calls']==report['optimizer_calls']==0
        assert report['descriptor_limits']==[]
        assert [reply['command'] for reply in replies]==['initialize']
        assert not (tmp_path/'g1').exists()
    else:
        assert report['error'] is None and report['released']
        assert report['budgets']==[numeric_limit,numeric_limit]
        assert report['descriptor_limits'][0]==numeric_limit
        assert report['parsed']==2 and report['optimizer_calls']==1
        assert [reply['command'] for reply in replies]==['initialize','update','close']
        assert replies[1]['artifact']['optimizer_steps']==4


def test_owned_parent_initialize_close_reaps_and_rejects_raw_update(tmp_path):
    import asyncio
    import time

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_learning_process import HpoLearningProcess
    learner = learner_for()
    ledger = ReservationLedger()
    async def run():
        deadline = time.monotonic()+60
        child = HpoLearningProcess(tmp_path, 1024**2, learner.model, learner.training, learner.binding,
            learner.action_config_hash, learner.resume_identity, actions=ACTIONS, ledger=ledger,
            cuda_visible_devices='', cleanup_deadline=deadline)
        await child.start()
        try:
            reply = await child.initialize(17, tmp_path/'initial', restore=None, deadline=deadline)
            assert reply['protocol_version'] == 6 and child.steps == 0
            with pytest.raises(ContractError): await child.update({}, tmp_path/'forbidden', deadline=deadline)
        finally: await child.close(deadline)
        assert child.process.returncode == 0 and child.lease.released and ledger.learner is None
    asyncio.run(run())


@pytest.mark.parametrize('field', ['identity', 'protocol', 'steps', 'inventory'])
def test_owned_parent_rejects_bad_saved_ack_and_still_reaps(tmp_path, monkeypatch, field):
    import asyncio
    import time

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_learning_process import HpoLearningProcess
    from gf_training.ppo_run import LearningProcess
    learner = learner_for()
    original = LearningProcess.request
    async def corrupted(self, request, deadline):
        reply = await original(self, request, deadline)
        if request['command'] == 'initialize':
            if field == 'identity': reply['identity']['resume_identity'] = 'a'*64
            elif field == 'protocol': reply['protocol_version'] = 5
            elif field == 'steps': reply['artifact']['optimizer_steps'] = 1
            else: reply['artifact']['files']['foreign.bin'] = 'a'*64
        return reply
    monkeypatch.setattr(LearningProcess, 'request', corrupted)
    async def run():
        ledger = ReservationLedger()
        deadline = time.monotonic()+60
        child = HpoLearningProcess(tmp_path, 1024**2, learner.model, learner.training, learner.binding,
            learner.action_config_hash, learner.resume_identity, actions=ACTIONS, ledger=ledger,
            cuda_visible_devices='', cleanup_deadline=deadline)
        await child.start()
        try:
            with pytest.raises(ContractError):
                await child.initialize(17, tmp_path/'initial', restore=None, deadline=deadline)
            assert child.steps is None and not child.lease.released
        finally: await child.close(deadline)
        assert child.process.returncode == 0 and child.lease.released and ledger.learner is None
    asyncio.run(run())
