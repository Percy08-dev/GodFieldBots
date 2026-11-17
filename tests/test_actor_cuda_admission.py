"""Resident device enforcement changes while legacy host/learner/cost contracts remain."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli.hpo_config import campaign_config, joint_resource_admission
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_runtime import hpo_training_admission
from tests.test_hpo_config import campaign_raw

RUNTIME = {'mode': 'resident-worker-v1', 'cache_entries': 6, 'cache_weight_bytes': 134217728}


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_resident_four_games_uses_actor_device_charge_and_keeps_full_legacy_host(kind, tmp_path):
    raw = campaign_raw()
    raw.update(schema_version=3, model_id=kind, runtime=RUNTIME,
        base_experiment=str(Path(__file__).parents[1] / f'configs/experiments/card-instance-{kind}-ppo-acceptance.toml'),
        operating={'game_concurrency': 4, 'max_participant_workers': 8})
    loaded = campaign_config(raw, origin=tmp_path)
    config = loaded.config
    operating = asdict(config.operating)
    with pytest.raises(ValueError, match='joint HPO CUDA demand'):
        hpo_training_admission(config.model, config.inference, config.training.tbptt_length,
                               operating, layout=LAYOUT)
    admitted = joint_resource_admission(config, layout=LAYOUT)
    base = hpo_training_admission(config.model, config.inference, config.training.tbptt_length,
        {**operating, 'managed_cuda_bytes': 128 * 1024**3}, layout=LAYOUT)
    weights, positions = admitted['network_weight_bytes'], admitted['network_position_bytes']
    extra_host = 8 * (5 * weights + 6 * positions)
    assert admitted['worker_host_bytes'] == base['worker_bytes'] + extra_host
    assert admitted['worker_bytes'] == 8 * admitted['actor_cuda_estimate']['cuda_bytes'] + 8 * 5 * (weights + positions)
    assert admitted['learner_bytes'] == base['learner_bytes']
    assert admitted['worker_bytes'] < admitted['worker_host_bytes']
    assert admitted['ram_phases']['save'] == base['ram_phases']['save'] + admitted['worker_host_bytes']
    assert admitted['cuda_phases']['collection'] == base['learner_bytes'] + admitted['worker_bytes'] + 8 * 134217728
    assert admitted['peak_cuda_bytes'] <= config.operating.managed_cuda_bytes


def test_fresh_training_keeps_original_device_gate_and_derived_identity(tmp_path):
    loaded = campaign_config({**campaign_raw(), 'operating': {'game_concurrency': 4,
        'max_participant_workers': 8}}, origin=tmp_path)
    with pytest.raises(ValueError, match='joint HPO CUDA demand'):
        joint_resource_admission(loaded.config, layout=LAYOUT)
    safe = replace(loaded.config, operating=replace(loaded.config.operating, managed_cuda_bytes=128 * 1024**3))
    admitted = joint_resource_admission(safe, layout=LAYOUT)
    assert admitted['recipe'] == 'hpo-joint-process-envelope-v1'
    assert 'actor_cuda_estimate' not in admitted and 'worker_host_bytes' not in admitted


def test_resident_eight_games_still_enforces_unchanged_host_ram(tmp_path):
    loaded = campaign_config({**campaign_raw(), 'schema_version': 3, 'runtime': RUNTIME,
        'operating': {'game_concurrency': 8, 'max_participant_workers': 16}}, origin=tmp_path)
    with pytest.raises(ValueError, match='resource=ram') as caught:
        joint_resource_admission(loaded.config, layout=LAYOUT)
    assert 'phase=collection' in str(caught.value)


def test_resident_role_checks_use_actual_pool_lease_including_explicit_extras():
    from gf_common.domain import ContractError, RuntimePolicyBinding, digest
    from gf_models.structured_binding import input_binding_game_v3
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_runtime import admit_hpo_resident_game_workers
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_network import small_model4

    model = small_model4()
    binding = InferenceBindingV2(max_candidates=64, candidate_chunk=2, device='cuda:0')
    options = {'version': '4', 'checkpoint': '/tmp/actor-admission-fixture',
        'expected_manifest_digest': 'a' * 64, 'model': asdict(model), 'inference': asdict(binding),
        'cuda_visible_devices': '0', 'action_config_hash': digest(asdict(ACTIONS)),
        **input_binding_game_v3(ACTIONS.action_contract, RULES)}
    roles = (RuntimePolicyBinding('mlp', options), RuntimePolicyBinding('random'))
    per_owner = 12345 + 96 * 1024**2
    arguments = {'rules': RULES, 'model': model, 'inference': binding,
        'worker_cuda_bytes': per_owner, 'begin_import_bytes': 128 * 1024**2,
        'game_concurrency': 4, 'global_cuda_bytes': 12 * 1024**3,
        'max_workers': 8, 'learner_cuda_bytes': 4000}
    admitted = admit_hpo_resident_game_workers(ACTIONS, roles, **arguments)
    assert admitted['worker_resident_bytes'] == 8 * per_owner
    assert admitted['begin_import_bytes'] == 8 * 128 * 1024**2
    assert admitted['peak_cuda_bytes'] == 4000 + 8 * (per_owner + 128 * 1024**2)
    changed = replace(model, temporal=replace(model.temporal, num_layers=2))
    foreign = RuntimePolicyBinding('mlp', {**options, 'model': asdict(changed)})
    with pytest.raises(ContractError, match='homogeneous'):
        admit_hpo_resident_game_workers(ACTIONS, (foreign, roles[1]), **arguments)


def test_normal_resident_preflight_uses_final_device_gate_without_allocating(tmp_path, monkeypatch):
    import time

    from gf_cli.hpo_config import materialize_hpo_run
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_ppo import _HpoOperations

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    loaded = campaign_config({**campaign_raw(), 'schema_version': 3, 'runtime': RUNTIME,
        'operating': {'game_concurrency': 4, 'max_participant_workers': 8}}, origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run_id = '00000000-0000-0000-0000-000000000017'
    runtime = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=run_id)
    monkeypatch.setattr('gf_simulator_client.card_instance_runtime.CardInstanceRuntimeEvidence.preflight', lambda self: {})
    with gpu.admit_gpu_job(gpu.JobContext('fixture', 'preflight', 'training', run_id, None),
                          global_cuda_bytes=12 * 1024**3, max_workers=8) as job:
        _HpoOperations(job).preflight(runtime.config, deadline=time.monotonic() + 10)
        assert not job.ledger._history and job.ledger.cuda_bytes == 0
        job.confirm_termination()


def test_actual_owned_collector_reports_the_same_live_pool_amounts(tmp_path, monkeypatch):
    """Controlled public endpoints; real collector/Source/FDs/actors, no GPU or qualification."""
    import asyncio

    import gf_cli.card_instance as core
    from tests.hpo_owned_support import auditable_runtime, owned_runtime, run_owned

    runtime, run_id, _, _ = owned_runtime(tmp_path, monkeypatch)
    auditable_runtime(monkeypatch)
    original = core._run_corrected
    rows = []

    def forbidden(*args, **kwargs):
        raise AssertionError('resident collector reapplied the obsolete per-role CUDA estimate')

    monkeypatch.setattr('gf_models.structured_runtime.admit_hpo_game_workers', forbidden)

    async def observe(*args, **kwargs):
        row, pool = kwargs['resource_admission'], kwargs['resident_pool']
        workers = 2 * runtime.config.run.concurrency
        assert row['workers'] == workers
        assert row['worker_resident_bytes'] == workers * pool.resident_bytes
        assert row['begin_import_bytes'] == workers * pool.begin_import_bytes
        assert row['peak_cuda_bytes'] == row['learner_cuda_bytes'] + workers * (pool.resident_bytes + pool.begin_import_bytes)
        rows.append(row)
        return await original(*args, **kwargs)

    monkeypatch.setattr(core, '_run_corrected', observe)

    async def run():
        _, operations = await run_owned(runtime, tmp_path / 'parents' / run_id, production=True)
        operations.completion.close()
        assert len(rows) == 2
        assert not operations.job.ledger._resident_begin_imports
        assert operations.job.ledger.cuda_bytes == 0

    asyncio.run(run())
