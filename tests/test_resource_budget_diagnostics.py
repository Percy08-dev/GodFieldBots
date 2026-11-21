"""Refusal messages survive plain-string campaign persistence with owner context."""

import time

import pytest

from gf_common.artifacts import ArtifactPhaseAdmission
from gf_common.group_pack import StorageSettings
from gf_training.group_pack_resources import BorrowedBudget, ByteBudget, StorageBudget


@pytest.mark.parametrize('size,reason', [(11, 'reservation exceeds capacity'),
                                        (4, 'storage budget exhausted')])
def test_byte_refusal_reports_current_locked_account_without_reserving(size, reason):
    budget = ByteBudget(10, name='source', phase='evaluation')
    deadline = time.monotonic() + 10
    budget.acquire(7, deadline)
    with pytest.raises(ValueError, match=reason) as caught:
        budget.acquire(size, deadline)
    for token in ('budget=source', 'phase=evaluation', 'used=7', f'request={size}', 'limit=10'):
        assert token in str(caught.value)
    assert budget.used == budget.peak == 7
    budget.release(7)


@pytest.mark.parametrize('local_limit,size,reason', [(10, 11, 'exceeds borrowed capacity'),
    (10, 4, 'borrowed storage budget exhausted'), (None, 4, 'storage budget exhausted')])
def test_borrower_refusal_identifies_the_actual_rejecting_owner(local_limit, size, reason):
    parent = StorageBudget(StorageSettings(), time.monotonic() + 10,
                           name='source', phase='evaluation')
    budget = BorrowedBudget(parent, limit=local_limit, name='borrowed', phase='decode')
    budget.acquire(7, parent.deadline)
    if local_limit is None:
        parent.acquire(parent.limit - 10, parent.deadline)
    with pytest.raises(ValueError, match=reason) as caught:
        budget.acquire(size, parent.deadline)
    if local_limit is None:
        expected = ('budget=source', 'phase=evaluation', f'used={parent.limit - 3}',
                    'request=4', f'limit={parent.limit}')
    else:
        expected = ('budget=borrowed', 'phase=decode', 'used=7', f'request={size}', 'limit=10')
    assert all(token in str(caught.value) for token in expected)
    assert budget.used == budget.peak == 7
    budget.release(7)
    if local_limit is None:
        parent.release(parent.limit - 10)
    assert parent.used == 0


def test_measured_ram_overrun_keeps_measured_usage_and_stops_admission():
    budget = StorageBudget(StorageSettings(), time.monotonic() + 10,
                           name='source', phase='evaluation')
    with budget.reserve(7):
        with pytest.raises(ValueError, match='storage budget exceeded by RAM files') as caught:
            budget.measure_ram(budget.limit)
        assert all(token in str(caught.value) for token in (
            'budget=source', 'phase=evaluation', f'used={budget.limit + 7}',
            f'request={budget.limit}', f'limit={budget.limit}'))
        with pytest.raises(RuntimeError, match='storage admission stopped'):
            budget.reserve(0)
    assert budget.used == budget.ram_bytes == budget.limit


@pytest.mark.parametrize('wait,size,reason', [(False, 4, 'shared artifact phase budget exhausted'),
    (True, 1, 'artifact phase admission does not wait')])
def test_phase_refusal_reports_its_own_account_and_releases_original_source(wait, size, reason):
    source = StorageBudget(StorageSettings(), time.monotonic() + 10)
    phase = ArtifactPhaseAdmission(source, 10, name='parent', phase='evaluation')
    with phase.reserve(7):
        with pytest.raises(ValueError, match=reason) as caught:
            phase.reserve(size, wait=wait)
        assert all(token in str(caught.value) for token in (
            'budget=parent', 'phase=evaluation', 'used=7', f'request={size}', 'limit=10'))
        assert phase.used == source.used == 7
    assert phase.used == source.used == 0


@pytest.mark.parametrize('wait', [False, True])
def test_phase_invalid_request_does_not_render_arbitrary_input(wait):
    class InvalidRequest:
        def __str__(self):
            raise AssertionError('diagnostics must contain scalar requests only')

    source = StorageBudget(StorageSettings(), time.monotonic() + 10)
    phase = ArtifactPhaseAdmission(source, 10)
    reason = 'does not wait' if wait else 'budget exhausted'
    with pytest.raises(ValueError, match=reason):
        phase.reserve(InvalidRequest(), wait=wait)
    assert phase.used == source.used == 0


def test_actual_evaluation_refusal_survives_campaign_failure_persistence(tmp_path, monkeypatch):
    """Real completion retention refuses; controlled training/conditions cannot qualify data."""
    import asyncio

    from gf_cli.hpo_evaluation import evaluate_seed
    from gf_cli.hpo_storage import CampaignStore
    from tests.hpo_owned_support import owned_runtime, run_owned
    from tests.test_hpo_campaign import controller_fixture

    cli, loaded, _, _ = controller_fixture(tmp_path, monkeypatch, owned=True)
    runtime, run_id, _, _ = owned_runtime(tmp_path, monkeypatch)
    parent = tmp_path / 'parents' / run_id
    _, operations = asyncio.run(run_owned(runtime, parent))
    completion = operations.completion
    retained = []
    scopes = []
    initialize = ArtifactPhaseAdmission.__init__

    def occupied(self, source, limit, **kwargs):
        initialize(self, source, limit, **kwargs)
        # Occupy the genuine account, then let retain_for_evaluation make its
        # original request. No fabricated exception or replacement evaluator.
        retained.append(self.reserve(limit - 1))
        scopes.append(self)

    monkeypatch.setattr(ArtifactPhaseAdmission, '__init__', occupied)

    async def trained(store, loaded, campaign, proposal, seed, **kwargs):
        folder = f'proposals/{proposal.number:06}/seeds/{seed}'
        store.put(folder + '/training.intent.json', {'scope': 'unqualified controller fixture'})
        state = completion.state
        store.put(folder + '/training.json', {'training_seed': seed, 'run_id': run_id,
            'run_path': str(parent), 'generation': state['updates'],
            'used_decisions': state['used_decisions'], 'inference_sha256': completion.records[-1]['inference_sha256']})
        return completion

    monkeypatch.setattr(cli, '_train_seed', trained)
    monkeypatch.setattr(cli, 'evaluate_seed', evaluate_seed)
    directory = tmp_path / 'campaign'
    try:
        result = asyncio.run(cli.run_campaign(loaded, directory, mode='fresh', progress='off'))
        assert result['failed_in_session'] >= 1
        with CampaignStore(directory, mode='resume', max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
            failure = store.get('proposals/000000/failure.json')[0]
            reason = failure['reason']
            assert failure['boundary']['phase'] == 'evaluate'
            assert reason.startswith('ValueError:shared artifact phase budget exhausted')
            assert all(token in reason for token in ('budget=parent', 'phase=hpo-evaluation',
                f'used={loaded.config.operating.parent_budget_bytes - 1}', 'request=',
                f'limit={loaded.config.operating.parent_budget_bytes}'))
            assert store.get('proposals/000000/tell.receipt.json')[0]['reason'] == reason
            assert not store.exists('proposals/000000/seeds/17/evaluation.json')
            assert not store.exists('proposals/000000/evaluation.json')
    finally:
        for reservation in retained:
            reservation.close()
        completion.close()
    assert scopes and all(scope.used == scope.source.used == 0 for scope in scopes)
    assert operations.budget.used == 0
