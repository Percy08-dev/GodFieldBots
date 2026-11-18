"""Service authority cannot be supplied by response metadata or participant slots."""
import pytest

from gf_common.domain import ContractError
from gf_simulator_client.structured_resources import ReservationLedger


def test_service_residency_is_distinct_from_participant_slots_and_survives_update():
    ledger = ReservationLedger(1000,2)
    owner = object()
    service = ledger.reserve_service(200,owner)
    a,b = ledger.reserve_worker(0),ledger.reserve_worker(0)
    assert len(ledger.workers) == 2 and ledger.cuda_bytes == 200
    with pytest.raises(ContractError): ledger.reserve_worker(0)
    a.cancel_before_spawn(); b.cancel_before_spawn()
    with pytest.raises(ContractError): ledger.begin_update(100)
    token = service.quiescent(owner)
    ledger.begin_shared_update(100,service=service,quiescence=token)
    assert ledger.cuda_bytes == 300
    ledger.finish_update()
    assert ledger.cuda_bytes == 200 and not service.released
    service.invalidate(owner)
    with pytest.raises(ContractError): ledger.begin_shared_update(100,service=service,quiescence=token)


def test_request_matching_rejects_cross_game_revision_checkpoint_and_late_responses():
    from gf_common.shared_requests import RequestAuthority
    authority = RequestAuthority('a'*32)
    a = authority.begin('endpoint-a',game='same',participant=1,revision=3,checkpoint='b'*64)
    with pytest.raises(ContractError):
        authority.begin('endpoint-b',game='same',participant=2,revision=3,checkpoint='b'*64)
    b = authority.begin('endpoint-b',game='other',participant=1,revision=3,checkpoint='b'*64)
    assert a != b
    for key,value in (('execution','c'*32),('game','other'),('participant',2),('request',99),('revision',4),('checkpoint','c'*64)):
        with pytest.raises(ContractError): authority.match('endpoint-a',a | {key:value})
    authority.match('endpoint-a',a)
    authority.retire('endpoint-a')
    with pytest.raises(ContractError): authority.match('endpoint-a',a)
    authority.retire('endpoint-b')
    assert authority.empty
