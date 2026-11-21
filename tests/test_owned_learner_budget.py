"""B2b dual-budget behavior; baseline source before implementation: 9e6cb5d."""

from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_config import InferenceBindingV2


def test_group_device_allowance_is_independent_of_host_rollout_and_logical_cap():
    from gf_training.hpo_update_planning import execution_plan, joint_estimate, plan_groups
    from tests.test_hpo_numeric import learner_for, rollout_for

    learner = learner_for()
    endpoint = rollout_for(learner).endpoint(0)
    position_bytes = learner.network.positions.numel() * learner.network.positions.element_size()
    single = execution_plan([endpoint], learner.model, learner.binding, 32, layout=LAYOUT,
        reservation_bytes=learner.binding.managed_budget_bytes, resident_bytes=0)
    two = joint_estimate([endpoint, endpoint], learner.model, learner.binding, 32, layout=LAYOUT)
    # Choose a physical allowance admitting exactly a singleton, while retaining
    # enough independent logical allowance for real host rollout ownership.
    device = single.estimate['total_bytes'] + position_bytes
    host_rollout = 8 * 1024**2
    logical = two['total_bytes'] + host_rollout

    class Input:
        resident_bytes = host_rollout

        def endpoint(self, index):
            return endpoint

    groups = plan_groups(Input(), [0, 1], learner.model, learner.binding, 32, variant='microbatch',
        layout=LAYOUT, reservation_bytes=logical, device_bytes=device)
    assert [group.indices for group in groups] == [(0,), (1,)]
    for group in groups:
        assert group.plan.estimate['total_bytes'] + position_bytes <= device
    control = plan_groups(Input(), [0, 1], learner.model, learner.binding, 32, variant='microbatch',
        layout=LAYOUT, reservation_bytes=logical, device_bytes=logical)
    assert [group.indices for group in control] == [(0, 1)]


def test_non_group_minima_refuse_before_model_allocation_and_pool_is_not_free():
    from gf_models.learner_cuda import learner_cuda_minima, owned_learner_cuda_admission
    from tests.test_hpo_network import small_model4

    model = small_model4()
    binding = InferenceBindingV2(device='cuda:0', max_candidates=64, candidate_chunk=2)
    minima = learner_cuda_minima(model, binding, layout=LAYOUT)
    required = max(minima['phases'].values())
    arguments = {'logical_bytes': required * 2, 'global_cuda_bytes': required + 200,
        'pool_cuda_bytes': 100, 'overlap_scratch_bytes': 100, 'layout': LAYOUT}
    admitted = owned_learner_cuda_admission(model, binding, **arguments)
    assert admitted['device_bytes'] == required
    assert admitted['logical_bytes'] == 2 * required
    with pytest.raises(ContractError, match='phase='):
        owned_learner_cuda_admission(model, binding,
            **{**arguments, 'global_cuda_bytes': required + 199})
    with pytest.raises(ContractError, match='pool'):
        owned_learner_cuda_admission(model, binding,
            **{**arguments, 'global_cuda_bytes': 199})


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_owned_allowance_preserves_original_logical_cap_and_future_pool(kind, tmp_path):
    from pathlib import Path

    from gf_cli.hpo_config import campaign_config, joint_resource_admission
    from tests.test_hpo_config import campaign_raw

    raw = campaign_raw()
    raw.update(schema_version=4, origin='producer-owned-v1', encoder='canonical-python',
        model_id=kind, base_experiment=str(Path(__file__).parents[1] / f'configs/experiments/card-instance-{kind}-ppo-acceptance.toml'),
        runtime={'mode': 'resident-worker-v1', 'cache_entries': 6, 'cache_weight_bytes': 134217728},
        operating={'game_concurrency': 5, 'max_participant_workers': 10})
    config = campaign_config(raw, origin=tmp_path).config
    admission = joint_resource_admission(config, layout=LAYOUT)
    allowance = admission['owned_learner_admission']
    assert allowance['device_bytes'] == min(allowance['logical_bytes'], config.operating.managed_cuda_bytes
        - admission['worker_bytes'] - max(admission['begin_staging_bytes'], admission['staging_bytes']))
    assert admission['learner_bytes'] == allowance['logical_bytes']
    assert admission['cuda_phases']['collection'] == allowance['device_bytes'] + admission['worker_bytes'] + admission['begin_staging_bytes']
    assert allowance['logical_bytes'] >= allowance['device_bytes'] > 0
    # A separate fresh/resident spec3 keeps its original learner semantics.
    from gf_models.structured_runtime import hpo_resident_training_admission
    original = hpo_resident_training_admission(config.model, config.inference, config.training.tbptt_length,
        {**asdict(config.operating), 'managed_cuda_bytes': 128 * 1024**3}, config.runtime, layout=LAYOUT)
    assert 'owned_learner_admission' not in original
    assert original['learner_bytes'] == allowance['logical_bytes']


def test_launch_context_recomputes_from_checked_model_and_closed_operating_bounds():
    import copy

    from gf_cli.hpo_config import HPOOperatingConfig
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.owned_learner_admission import (
        checked_owned_launch_context,
        owned_launch_context,
    )
    from tests.test_hpo_network import small_model4

    model = small_model4()
    binding = InferenceBindingV2(device='cuda:0', max_candidates=64, candidate_chunk=2)
    training = HPOTrainingConfig()
    context = owned_launch_context(model, training, binding, asdict(HPOOperatingConfig()),
        {'mode': 'resident-worker-v1', 'cache_entries': 6, 'cache_weight_bytes': 134217728})
    assert checked_owned_launch_context(context, model, training, binding) == context
    for mutation in ('field', 'device', 'operating', 'pool'):
        changed = copy.deepcopy(context)
        if mutation == 'field': changed['extra'] = 1
        elif mutation == 'device': changed['admission']['device_bytes'] -= 1
        elif mutation == 'operating': changed['operating']['managed_cuda_bytes'] += 1
        else: changed['admission']['pool_cuda_bytes'] -= 1
        with pytest.raises(ContractError):
            checked_owned_launch_context(changed, model, training, binding)


def test_original_owned_creator_binds_distinct_device_lease_and_logical_launch_without_gpu(tmp_path, monkeypatch):
    import asyncio
    import json
    import sys
    import time
    from pathlib import Path

    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.group_pack import StorageSettings
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.owned_learner_admission import (
        checked_owned_launch_context,
        issue_owned_learner_launch,
    )
    from tests.test_hpo_config import campaign_raw

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    raw = campaign_raw()
    raw.update(schema_version=4, origin='producer-owned-v1', encoder='canonical-python', model_id='transformer',
        base_experiment=str(Path(__file__).parents[1] / 'configs/experiments/card-instance-transformer-ppo-acceptance.toml'),
        runtime={'mode': 'resident-worker-v1', 'cache_entries': 6, 'cache_weight_bytes': 134217728},
        operating={'game_concurrency': 5, 'max_participant_workers': 10})
    campaign = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=LAYOUT)
    run_id = '00000000-0000-0000-0000-000000000017'
    config = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=run_id).config
    observed = []
    arguments = HpoOwnedLearningProcess._launch_arguments

    def inspect_arguments(self):
        result = arguments(self)
        context = checked_owned_launch_context(json.loads(result[3]), self.model, self.hpo_training, self.binding)
        observed.append((self.lease.resident_bytes, int(result[1]), context['admission']))
        assert self.lease.resident_bytes == context['admission']['device_bytes'] < int(result[1])
        return result

    monkeypatch.setattr(HpoOwnedLearningProcess, '_launch_arguments', inspect_arguments)

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture', 'launch', 'training', run_id, None),
                              global_cuda_bytes=config.run.global_cuda_budget_bytes, max_workers=10) as job:
            operations = _OwnedHpoOperations(job)
            with operations.owned_invocation():
                model = operations.validate_config(config)
                identity = {}
                operations.decorate_identity(identity)
                operations.consume_prechild()
                process = HpoOwnedLearningProcess.__new__(HpoOwnedLearningProcess)
                operations._startup_child = process
                launch = issue_owned_learner_launch(operations, config, {'identity': identity})
                account = StorageBudget(StorageSettings(), time.monotonic() + 30)
                HpoOwnedLearningProcess.__init__(process, tmp_path, 1024**2, model, settings.training,
                    config.model.inference, digest(asdict(config.actions)), 'b' * 64, actions=config.actions, ledger=job.ledger,
                    cuda_visible_devices='0', cleanup_deadline=time.monotonic() + 30,
                    budget=account, _owned_learner_launch=launch)
                # A genuine OS ENOENT exercises the untouched creator/no-child
                # settlement. No interpreter or CUDA learner is ever started.
                with monkeypatch.context() as patch:
                    patch.setattr(sys, 'executable', str(tmp_path / 'absent-interpreter'))
                    with pytest.raises(FileNotFoundError):
                        await process.start()
                assert process.lease.released and process.lease._proof.outcome == 'never_started'
                assert job.ledger.cuda_bytes == 0
                process.channel.close()
                process._inherited.close()
            assert observed and observed[0][2] == identity['owned_learner_admission']
            job.confirm_termination()

    asyncio.run(run())


def test_owned_singleton_refusal_precedes_shuffle_parameters_and_batch_lifecycle(tmp_path):
    import copy

    import torch

    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import captured_record

    learner = learner_for()
    batch, owner, plan, snapshot, account, _ = captured_record(learner)
    selection = selection_for_counts((3,), learning_seed=17, batch=0, decision_budget=3, used_before=0)
    numeric = encode_owned_batch(batch, owner=owner, plan=plan, snapshot_owner=snapshot,
        model=learner.model, training=learner.training, actions=ACTIONS, rules=RULES,
        selected_locations=((0, 0), (0, 1), (0, 2)), selection=selection, budget=account)
    numeric.persist(tmp_path / 'numeric')
    initial = {key:value.clone() for key,value in learner.network.state_dict().items()}
    rng, shuffle = torch.get_rng_state().clone(), copy.deepcopy(learner.shuffle.bit_generator.state)
    learner._training_device_bytes = 1  # Controlled pure-planner boundary on CPU, no GPU authority.
    with pytest.raises(ContractError, match='device allowance') as caught:
        learner.update_owned(numeric)
    caught.value.__traceback__ = None
    del caught
    assert numeric._state == 'PERSISTED' and not learner._owned_failed
    assert learner.steps == 0 and not learner.optimizer.state
    assert learner.shuffle.bit_generator.state == shuffle and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(initial[key], value) for key, value in learner.network.state_dict().items())
    assert all(parameter.grad is None for parameter in learner.network.parameters())
    numeric.close()
    batch.close()
    snapshot.close()


def test_cached_plan_is_rechecked_before_any_final_tensor_packing(monkeypatch):
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    from gf_training.hpo_update_execution import group_forward
    from gf_training.hpo_update_planning import UpdateGroup, execution_plan
    from tests.test_hpo_numeric import learner_for, rollout_for

    learner = learner_for()
    rollout = rollout_for(learner)
    plan = execution_plan([rollout.endpoint(0)], learner.model, learner.binding, 32, layout=LAYOUT,
        reservation_bytes=learner._training_reservation_bytes, resident_bytes=rollout.resident_bytes)
    learner._training_device_bytes = 1

    def forbidden(*args, **kwargs):
        raise AssertionError('packing preceded final independent device admission')

    monkeypatch.setattr('gf_training.hpo_update_execution.pack_window', forbidden)
    monkeypatch.setattr('gf_training.hpo_update_execution.pack_candidates', forbidden)
    with pytest.raises(ContractError, match='device allowance'):
        group_forward(learner, rollout, UpdateGroup((0,), plan=plan),
            TrainingExecutionSpec(variant='microbatch'), learner.network)


def test_two_device_budgets_refine_groups_but_keep_logical_optimizer_steps_and_values():
    import torch

    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_update_planning import execution_plan
    from tests.test_hpo_numeric import learner_for, rollout_for

    recipe = replace(HPOTrainingConfig(), epochs=1, minibatch_size=4)
    control, candidate = learner_for(recipe=recipe), learner_for(recipe=recipe)
    rollout = rollout_for(control, count=4)
    position_bytes = candidate.network.positions.numel() * candidate.network.positions.element_size()
    candidate._training_device_bytes = max(execution_plan([rollout.endpoint(index)], candidate.model,
        candidate.binding, 32, layout=LAYOUT, reservation_bytes=candidate._training_reservation_bytes,
        resident_bytes=rollout.resident_bytes).estimate['total_bytes'] for index in range(4)) + position_bytes
    expected = control._update_numeric(rollout, {}, 0., control.network)
    actual = candidate._update_numeric(rollout, {}, 0., candidate.network)
    assert expected['optimizer_step_delta'] == actual['optimizer_step_delta'] == 1
    assert expected['endpoint_evaluations'] == actual['endpoint_evaluations'] == 4
    assert candidate._training_update_diagnostics['microbatches'] > control._training_update_diagnostics['microbatches']
    for left, right in zip(control.network.parameters(), candidate.network.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    for name in ('total_loss', 'value_mse', 'policy_loss', 'entropy', 'approx_kl'):
        assert actual[name] == pytest.approx(expected[name], abs=1e-5, rel=1e-5)


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_phase_minima_count_actual_parameters_buffers_adam_and_cpu_steps(kind):
    from gf_models.learner_cuda import learner_cuda_minima
    from tests.test_hpo_numeric import learner_for

    learner = learner_for(kind)
    sum(parameter.sum() for parameter in learner.network.parameters()).backward()
    learner.optimizer.step()
    parameters = list(learner.network.parameters())
    weights = sum(value.numel() * value.element_size() for value in parameters)
    buffers = sum(value.numel() * value.element_size() for value in learner.network.buffers())
    largest = max(value.numel() * value.element_size() for value in parameters)
    moments = sum(state[key].numel() * state[key].element_size()
        for state in learner.optimizer.state.values() for key in ('exp_avg', 'exp_avg_sq'))
    steps = [state['step'] for state in learner.optimizer.state.values()]
    assert all(value.device.type == 'cpu' for value in steps)
    minima = learner_cuda_minima(learner.model, replace(learner.binding, device='cuda:0'), layout=LAYOUT)
    assert moments == 2 * weights
    assert minima['parameter_bytes'] == weights and minima['position_bytes'] == buffers
    assert minima['optimizer_step_cpu_bytes'] == sum(value.numel() * value.element_size() for value in steps)
    assert minima['optimizer_step_cuda_bytes'] == minima['export_rng_cuda_bytes'] == 0
    fixed = learner.model.resources.cuda_context_bytes + learner.model.resources.workspace_bytes
    scalars, finite = 40 + 2 * len(parameters) * 4 + 16, largest + largest // 4
    assert minima['phases'] == {
        'initialize': fixed + weights + buffers,
        'restore': fixed + 4 * weights + 2 * buffers + finite + scalars,
        'optimizer': fixed + 4 * weights + buffers + max(weights, 2 * largest) + scalars,
        'export': fixed + 4 * weights + buffers + finite + scalars,
        'nonempty_update_floor': fixed + 6 * weights + len(parameters) * 8 + buffers + 65536 + 512 + 1024,
    }


def test_constructor_checks_phase_allowance_before_any_cuda_runtime(monkeypatch):
    from gf_models.learner_cuda import learner_cuda_minima, owned_learner_cuda_admission
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_numeric import HpoLearner
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_network import small_model4

    model, binding = small_model4(), InferenceBindingV2(device='cuda:0', max_candidates=64, candidate_chunk=2)
    required = max(learner_cuda_minima(model, binding, layout=LAYOUT)['phases'].values())
    admitted = owned_learner_cuda_admission(model, binding, logical_bytes=required * 2,
        global_cuda_bytes=required + 200, pool_cuda_bytes=100, overlap_scratch_bytes=100, layout=LAYOUT)
    admitted['global_cuda_bytes'] -= 1

    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA runtime allocated before non-group minima check')

    monkeypatch.setattr('gf_training.hpo_numeric.torch_runtime', forbidden)
    with pytest.raises(ContractError, match='phase='):
        HpoLearner(model, HPOTrainingConfig(), 17, binding, 'a' * 64,
            actions=ACTIONS, rules=RULES, _owned_device_admission=admitted)


def test_real_width192_six_game_allowance_rejects_a_proven_impossible_nonempty_update():
    import tomllib
    from pathlib import Path

    from gf_cli.hpo_config import campaign_config, joint_resource_admission

    path = Path(__file__).parents[1] / 'configs/hpo/budget-probe-mlp-d10000.toml'
    config = campaign_config(tomllib.loads(path.read_text()), origin=path.parent).config
    config = replace(config, operating=replace(config.operating, game_concurrency=6, max_participant_workers=12))
    with pytest.raises(ContractError, match='phase=nonempty_update_floor') as caught:
        joint_resource_admission(config, layout=LAYOUT)
    assert 'required=847918096' in str(caught.value)
    assert 'limit=846503928' in str(caught.value)
