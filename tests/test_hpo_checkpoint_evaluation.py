"""Scheduled evaluation keeps checkpoint evidence out of final-D authority."""
import asyncio
import time
from dataclasses import asdict, replace

import pytest

from gf_cli import hpo_evaluation
from gf_common.card_instance_collection import (
    checked_hpo_evaluation_parent,
    hpo_evaluation_groups,
)
from gf_common.domain import ContractError

GAME_SEED = '00000000-0000-4000-8000-000000000000'


def parent():
    return {'run_id': 'training', 'run_path': '/parents/training', 'training_seed': 17,
        'generation': 0, 'generation_sha256': 'a' * 64, 'inference_sha256': 'b' * 64,
        'checkpoint_receipt_sha256': 'c' * 64, 'policy_label': 'model'}


def checkpoint_linkage():
    return {'schema_version': 2, 'kind': 'hpo-evaluation-linkage', 'campaign_id': 'campaign',
        'evaluation_id': 'evaluation', 'gpu_ticket': '0' * 32, 'purpose': 'checkpoint',
        'conditions_sha256': 'd' * 64, 'left_parent': parent(), 'right_parent': None,
        'game_seeds': [GAME_SEED], 'learner_seats': {'eval-p000000-seat0': 1, 'eval-p000000-seat1': 2},
        'schedule_digest': '0' * 64, 'policy_registry_digest': '0' * 64}


def test_checkpoint_panel_keeps_legacy_paired_streams_and_cannot_bind_objective():
    link = checkpoint_linkage()
    objective_parent = parent()
    objective_parent['source_receipt_sha256'] = objective_parent.pop('checkpoint_receipt_sha256')
    objective = {**link, 'schema_version': 1, 'purpose': 'objective', 'left_parent': objective_parent}
    groups, = hpo_evaluation_groups(link)
    assert groups == hpo_evaluation_groups(objective)[0]
    assert groups.seed == GAME_SEED
    assert [row.policies for row in groups.episodes] == [
        ('model', 'heuristic-card-instance-v2'), ('heuristic-card-instance-v2', 'model')]
    with pytest.raises(ContractError):
        hpo_evaluation_groups({**link, 'schema_version': 1, 'purpose': 'objective'})
    with pytest.raises(ContractError):
        hpo_evaluation_groups({**link, 'left_parent': objective_parent})
    with pytest.raises(ContractError):
        checked_hpo_evaluation_parent(parent())


def saved_panel():
    link = checkpoint_linkage()
    return {'run_id': 'evaluation', 'linkage': link, 'episodes': [
        {'episode_id': 'eval-p000000-seat1', 'seed': GAME_SEED, 'game_id': 'game-2',
            'end_kind': 'natural', 'outcome': 'draw', 'decisions': 30},
        {'episode_id': 'eval-p000000-seat0', 'seed': GAME_SEED, 'game_id': 'game-1',
            'end_kind': 'natural', 'outcome': {'win': {'winner': 1}}, 'decisions': 20}]}


def test_checkpoint_projection_preserves_pair_and_counts_draw_in_score_only():
    panel = hpo_evaluation.checkpoint_panel_projection(saved_panel())
    assert (panel['wins'], panel['draws'], panel['losses'], panel['games']) == (1, 1, 0, 2)
    assert (panel['win_rate'], panel['score']) == (0.5, 0.75)
    pair, = panel['pairs']
    assert pair['game_seed'] == GAME_SEED
    assert [(row['learner_seat'], row['game_id'], row['outcome']) for row in pair['games']] == [
        (1, 'game-1', {'win': {'winner': 1}}), (2, 'game-2', 'draw')]


@pytest.mark.parametrize('change', ['partial', 'duplicate', 'truncated', 'no-outcome'])
def test_checkpoint_projection_requires_full_natural_panel(change):
    manifest = saved_panel()
    rows = manifest['episodes']
    if change == 'partial':
        rows.pop()
    elif change == 'duplicate':
        rows[-1]['game_id'] = rows[0]['game_id']
    elif change == 'truncated':
        rows[0]['end_kind'] = 'truncated'
    else:
        rows[0]['outcome'] = None
    with pytest.raises(ContractError):
        hpo_evaluation.checkpoint_panel_projection(manifest)


def test_checkpoint_evaluation_cannot_start_with_only_checkpoint_data(tmp_path):
    with pytest.raises(ContractError, match='actual closed training'):
        asyncio.run(hpo_evaluation.evaluate_checkpoint(None, None, {}, None, directory=tmp_path,
            generation=0, output=tmp_path / 'panel', deadline=0, training_result=parent()))


def test_best_checkpoint_uses_score_and_breaks_ties_by_earlier_generation():
    rows = ({'generation': 10, 'wins': 2, 'draws': 0, 'games': 4},
        {'generation': 0, 'wins': 1, 'draws': 1, 'games': 4},
        {'generation': 5, 'wins': 1, 'draws': 2, 'games': 4})
    assert hpo_evaluation._best_checkpoint(rows) == rows[2]
    assert hpo_evaluation._best_checkpoint(()) is None


def test_checkpoint_update_cap_is_the_original_operating_limit(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    from gf_cli.hpo_config import materialize_hpo_run
    from tests.test_rust_player_campaign_evaluation import campaign_inputs

    loaded, settings, training, _, _ = campaign_inputs(tmp_path)
    cap = loaded.config.operating.max_updates
    run_id = training['run_id']
    directory = Path(loaded.config.storage.parent_root) / run_id
    base = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=run_id)
    proposal = SimpleNamespace(settings=settings)
    state = {'seed': 17, 'run_id': run_id, 'updates': cap, 'used_decisions': 1,
        'status': 'incomplete', 'stop_reason': 'update_cap', 'config_sha256': base.sha256,
        'effective_config': base.effective, 'identity': {'integration': {}},
        'execution': {'inference': {'binding': {}, 'cuda_visible_devices': ''}}}
    campaign = {'conditions': {'producer_identity': {}, 'integration_identity': {},
        'inference_binding': {}, 'cuda_visible_devices': ''}}
    assert hpo_evaluation._checkpoint_state_facts(state, loaded, campaign, proposal,
        directory=directory, generation=cap).sha256 == base.sha256
    with pytest.raises(ContractError, match='healthy committed training pause or update cap'):
        hpo_evaluation._checkpoint_state_facts({**state, 'updates': cap - 1}, loaded, campaign,
            proposal, directory=directory, generation=cap - 1)


def test_rust_checkpoint_uses_bounded_fresh_panel_and_original_finalization_reserve(tmp_path):
    from gf_cli.hpo_config import materialize_hpo_run
    from gf_cli.rust_player_evaluation import _fresh_evaluation_plan
    from tests.test_rust_player_campaign_evaluation import campaign_inputs

    loaded, settings, training, original_parent, binding = campaign_inputs(tmp_path)
    original_parent['checkpoint_receipt_sha256'] = original_parent.pop('source_receipt_sha256')
    base = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=training['run_id'])
    output = tmp_path / 'panel'
    reserve = base.config.run.finalization_seconds
    projected = hpo_evaluation._checkpoint_projection_config(base, loaded, output=output,
        evaluation_id='checkpoint', wall_seconds=reserve + 60)
    plan = _fresh_evaluation_plan(loaded, {'campaign_id': 'campaign', 'conditions_sha256': 'c' * 64},
        original_parent, binding, projected, output=output, evaluation_id='checkpoint', ticket='1' * 32, started=100.)
    assert plan.linkage['purpose'] == 'checkpoint' and plan.linkage['schema_version'] == 2
    assert plan.wall_deadline - plan.execution_deadline == reserve
    assert plan.execution_deadline == 160.
    assert tuple(plan.linkage['game_seeds']) == loaded.config.evaluation_game_seeds


def test_owned_initial_checkpoint_publishes_one_verified_two_seat_panel(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    import gf_cli.card_instance as core
    import gf_cli.hpo_config as composition
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import digest
    from gf_models.arrays import sha256
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_ppo import train_hpo_ppo
    from gf_training.lineage import read_json
    from tests.hpo_owned_support import auditable_runtime, owned_runtime

    runtime, run_id, profile, settings = owned_runtime(tmp_path, monkeypatch, operating={
        'checkpoint_evaluation': {'enabled': True, 'updates': [0], 'wall_seconds': 120}})
    profile = replace(profile, base=replace(profile.base,
        config=replace(profile.base.config, actions=runtime.config.actions)))
    directory = tmp_path / 'parents' / run_id
    auditable_runtime(monkeypatch)
    # The bounded fixture uses the real CPU learner/actor and controlled own-view Simulator endpoints.
    monkeypatch.setattr(composition, 'materialize_hpo_run', lambda *args, **kwargs: runtime)
    monkeypatch.setattr(core, '_require_hpo_evaluation_policy', lambda binding: None)

    async def unexpected_collection(*args):
        raise AssertionError('G0 must close without collecting training games')

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture', 'initial', 'training', run_id, None),
                global_cuda_bytes=runtime.config.run.global_cuda_budget_bytes, max_workers=4) as job:
            training = await train_hpo_ppo(runtime, unexpected_collection, directory, gpu_job=job, pause_after_update=0)
        state = read_json(directory / 'run.json')
        provenance = ('code', 'module_sha256', 'cuda_distributions', 'runtime', 'numerics', 'package')
        conditions = {'producer_identity': {key: state['identity'][key] for key in provenance},
            'integration_identity': state['identity']['integration'],
            'inference_binding': state['execution']['inference']['binding'],
            'cuda_visible_devices': state['execution']['inference']['cuda_visible_devices']}
        campaign = {'campaign_id': 'fixture', 'loaded_config_sha256': profile.sha256,
            'conditions': conditions, 'conditions_sha256': digest(conditions)}
        with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=64 * 1024**2) as store:
            store.put('campaign.json', campaign)
            checksum = store.put('proposals/000000/proposal.json', {'fixture': True})
            proposal = Proposal(0, None, None, checksum, settings, '{}', '{}')
            slice_folder = 'proposals/000000/seeds/17/training-slices/000000'
            scope = {'proposal_sha256': checksum, 'training_seed': 17, 'run_id': run_id,
                'operation_id': 'initial', 'slice_no': 0}
            intent_sha = store.put(slice_folder + '/intent.json', {'schema_version': 1,
                'kind': 'hpo-training-slice-intent', **scope, 'run_path': str(directory),
                'config_sha256': runtime.sha256, 'mode': 'fresh', 'pause_after_update': 0,
                'previous_closure_sha256': None, 'previous_state_sha256': None, 'previous_generation_sha256': None})
            store.put(slice_folder + '/closure.json', {'schema_version': 1, 'kind': 'hpo-training-slice-closure',
                **scope, 'intent_sha256': intent_sha, 'state_sha256': sha256(directory / 'run.json'),
                'generation': 0, 'generation_sha256': state['generation_sha256'], 'used_decisions': 0,
                'status': 'paused', 'stop_reason': 'pause', 'termination_sha256': digest(asdict(training.gpu_termination))})
            output = tmp_path / 'panel'
            deadline = time.monotonic() + 180
            with gpu.admit_gpu_job(gpu.JobContext('fixture', 'foreign-slice', 'training', run_id, None),
                    global_cuda_bytes=runtime.config.run.global_cuda_budget_bytes, max_workers=4) as job:
                other_retirement = job.confirm_termination()
            with pytest.raises(ContractError, match='training slice'):
                await hpo_evaluation.evaluate_checkpoint(store, profile, campaign, proposal,
                    directory=directory, generation=0, output=output, deadline=deadline,
                    training_result=replace(training, gpu_termination=other_retirement))
            assert not output.exists()
            receipt = await hpo_evaluation.evaluate_checkpoint(store, profile, campaign, proposal,
                directory=directory, generation=0, output=output, deadline=deadline, training_result=training)
            assert receipt['used_decisions'] == receipt['training_games'] == 0
            assert receipt['panel']['games'] == 2 and receipt['status'] == 'complete'
            assert receipt['linkage']['purpose'] == 'checkpoint'
            panel, receipt_sha = hpo_evaluation.verify_checkpoint_evaluation_receipt(store, profile, campaign,
                proposal, 17, 0, deadline=deadline)
            assert panel == receipt
            index = hpo_evaluation.persist_best_checkpoint(store, profile, campaign, proposal, 17,
                terminal_generation=0, deadline=deadline)
            assert index['best']['generation'] == 0 and index['best']['evaluation_receipt_sha256'] == receipt_sha
            originals = (directory / 'evaluation-reference-receipts' / receipt['linkage']['gpu_ticket'] / 'release.receipt.json',
                gpu._GLOBAL_ROOT / 'jobs' / training.gpu_termination.ticket / 'termination.json')
            for original in originals:
                moved = original.with_name(original.name + '.removed')
                original.rename(moved)
                try:
                    with pytest.raises(FileNotFoundError):
                        hpo_evaluation.verify_checkpoint_evaluation_receipt(store, profile, campaign,
                            proposal, 17, 0, deadline=deadline)
                finally:
                    moved.rename(original)
            with pytest.raises(ContractError, match='already completed or started'):
                await hpo_evaluation.evaluate_checkpoint(store, profile, campaign, proposal,
                    directory=directory, generation=0, output=tmp_path / 'duplicate', deadline=deadline,
                    training_result=training)
            assert not (tmp_path / 'duplicate').exists()

    asyncio.run(run())
