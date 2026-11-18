"""Late/mismatched replies never reach participant sampling or create an action."""
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_simulator_client.framing import read_frame, write_frame
from gf_simulator_client.shared_channel import ParticipantScoreChannel


@pytest.mark.parametrize('field,value',[('execution','b'*32),('game','other-game'),('participant',2),
    ('request',2),('revision',8),('checkpoint','c'*64)])
def test_private_score_channel_discards_every_mismatched_response(field,value):
    left,right=socket.socketpair()
    channel=ParticipantScoreChannel(os.dup(left.fileno()))
    routing={'execution':'a'*32,'game':'actual-game','participant':1,'request':1,'revision':7,'checkpoint':'b'*64}
    channel.routing=dict(routing)
    channel.deadline=time.monotonic()+5
    def respond():
        with right.makefile('rwb',buffering=0) as stream:
            read_frame(stream)
            write_frame(stream,{'routing':routing|{field:value},'logs':[0.],'probabilities':[1.],'value':0.})
    worker=threading.Thread(target=respond)
    worker.start()
    try:
        with pytest.raises(ContractError,match='discarded'):
            channel.request(SimpleNamespace(revision=7,actor=1),'b'*64,{})
    finally:
        worker.join(5)
        assert not worker.is_alive()
        channel.stream.close();channel.socket.close();left.close();right.close()


def test_retired_channel_rejects_late_use_before_any_write_or_sampler():
    left,right=socket.socketpair()
    channel=ParticipantScoreChannel(os.dup(left.fileno()))
    try:
        assert channel.routing is None
        with pytest.raises(ContractError,match='capability'):
            channel.request(SimpleNamespace(revision=7,actor=1),'b'*64,{})
        right.setblocking(False)
        with pytest.raises(BlockingIOError):right.recv(1)
    finally:
        channel.stream.close();channel.socket.close();left.close();right.close()


def test_expired_request_is_discarded_before_send():
    left,right=socket.socketpair()
    channel=ParticipantScoreChannel(os.dup(left.fileno()))
    channel.routing={'execution':'a'*32,'game':'g','participant':1,'request':1,'revision':7,'checkpoint':'b'*64}
    channel.deadline=time.monotonic()-1
    try:
        with pytest.raises(TimeoutError,match='expired'):
            channel.request(SimpleNamespace(revision=7,actor=1),'b'*64,{})
        right.setblocking(False)
        with pytest.raises(BlockingIOError):right.recv(1)
    finally:
        channel.stream.close();channel.socket.close();left.close();right.close()


def test_reply_that_expires_during_decode_never_returns_to_sampler():
    left,right=socket.socketpair()
    channel=ParticipantScoreChannel(os.dup(left.fileno()))
    routing={'execution':'a'*32,'game':'g','participant':1,'request':1,'revision':7,'checkpoint':'b'*64}
    channel.routing=routing
    channel.deadline=time.monotonic()+5
    def respond():
        with right.makefile('rwb',buffering=0) as stream:
            read_frame(stream)
            channel.deadline=time.monotonic()-1
            write_frame(stream,{'routing':routing,'logs':[0.],'probabilities':[1.],'value':0.})
    worker=threading.Thread(target=respond)
    worker.start()
    try:
        with pytest.raises(TimeoutError,match='expired'):
            channel.request(SimpleNamespace(revision=7,actor=1),'b'*64,{})
    finally:
        worker.join(5)
        channel.stream.close();channel.socket.close();left.close();right.close()


def test_expired_scores_do_not_advance_policy_rng_position_or_create_selection():
    import math
    import random
    from dataclasses import asdict

    import torch

    from gf_common.card_instance import ActionConfigV3, shared_card_instance_contract
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
    from gf_common.domain import digest
    from gf_models.shared_policy import SharedParticipantPolicy
    from gf_models.structured_binding import input_binding_game_v3
    from gf_models.structured_config import InferenceBindingV2
    from tests.model3_support import model3
    from tests.test_card_instance_baseline_runtime import game
    actions=ActionConfigV3(action_contract=shared_card_instance_contract(),max_candidates=64)
    rules=card_instance_rules()
    frame,bank,_delta,_=game()
    generated=PublicCandidateGeneratorV3(actions,rules=rules,clock=lambda:0.).generate_game(frame,bank.view(),random.Random(17),NumericCosts())
    class ExpiredChannel:
        def request(self,*args):
            count=len(generated.table.kinds)
            return {'routing':{},'logs':[-math.log(count)]*count,'probabilities':[1/count]*count,'value':0.}
        def check_deadline(self):raise TimeoutError('expired before sampling')
    options={'version':'3','model':asdict(model3()),'inference':asdict(InferenceBindingV2(device='cuda:0',max_candidates=64)),
        'expected_manifest_digest':'b'*64,'action_config_hash':digest(asdict(actions)),'cuda_visible_devices':None,
        **input_binding_game_v3(actions.action_contract,rules)}
    policy=SharedParticipantPolicy(options,actions,rules,ExpiredChannel())
    policy.reset(17)
    before=policy.rng.get_state().clone()
    with pytest.raises(TimeoutError,match='sampling'):
        policy.select_frame(frame,bank.view(),generated)
    assert torch.equal(before,policy.rng.get_state())
    assert policy.position==0 and policy.previous is None


def test_abort_retires_original_request_before_entering_actor_join(monkeypatch):
    import asyncio

    from gf_simulator_client.shared_worker import SharedParticipantWorker
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    events=[]
    class Service:
        def complete(self,index,routing,*,completed):
            assert completed is False
            events.append('retire')
    async def original_join(self,error):
        events.append('join')
        raise error
    monkeypatch.setattr(StructuredParticipantWorker,'_abort',original_join)
    worker=SharedParticipantWorker.__new__(SharedParticipantWorker)
    worker.service=Service()
    worker.endpoint=0
    worker._pending_shared={'request':1}
    with pytest.raises(RuntimeError,match='primary'):
        asyncio.run(worker._abort(RuntimeError('primary')))
    assert events==['retire','join']


def test_failed_request_invalidation_cannot_replace_primary_roundtrip_failure(monkeypatch):
    import asyncio

    from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
    from gf_simulator_client.shared_worker import SharedParticipantWorker
    route={'request':1}
    class Service:
        execution_deadline=time.monotonic()+30
        def authorize(self,*args,**kwargs):return route
        def complete(self,*args,**kwargs):raise RuntimeError('cleanup failure')
    async def failure(self,payload):raise ValueError('primary request failure')
    monkeypatch.setattr(CardInstanceParticipantWorker,'_roundtrip',failure)
    worker=SharedParticipantWorker.__new__(SharedParticipantWorker)
    worker.service=Service();worker.endpoint=0;worker.game='g';worker.viewer=1;worker.model_id='mlp';worker.timeout=5
    worker.initial={'policy_options':{'expected_manifest_digest':'b'*64}}
    worker._pending_shared=None
    with pytest.raises(ValueError,match='primary request'):
        asyncio.run(worker._roundtrip({'kind':'select','frame':{'revision':0}}))
