"""Controlled HPO numeric fixtures; no qualified original Source."""
import copy
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gf_common.domain import ContractError, canonical
from gf_models.structured_config import InferenceBindingV2, model_config_v3
from gf_training.card_instance_formal import load_recipe, verify_kernel_ast
from gf_training.hpo_config import HPOTrainingConfig
from gf_training.hpo_numeric import HpoLearner, hpo_losses
from gf_training.hpo_rollout import HpoRollout
from gf_training.ppo import gae, normalize_advantages
from gf_training.ppo_math import losses
from gf_training.structured_numeric import GameLearner
from gf_training.structured_rollout import game_rollout
from tests.model3_support import LAYOUT
from tests.test_card_instance_learning import ACTIONS, RESUME, RULES, records_for
from tests.test_hpo_network import small_model4
from tests.test_structured_learning import assert_same_learners


def learner_for(kind="mlp", *, recipe=None, device="cpu", activation="relu"):
    recipe = HPOTrainingConfig() if recipe is None else recipe
    return HpoLearner(small_model4(kind, activation=activation), recipe, 17,
        InferenceBindingV2(device=device, max_candidates=64, candidate_chunk=2),
        RESUME, actions=ACTIONS, rules=RULES)


def rollout_for(learner, *, selected=None, count=3):
    records = (records_for(learner, count=count),)
    if selected is None:
        selected = tuple((0, index) for index in range(count))
    return HpoRollout(records, learner.model, learner.training, actions=ACTIONS,
        rules=RULES, selected_locations=selected)


def test_all_eight_frozen_kernels_remain_exact():
    verify_kernel_ast(load_recipe())


def test_configurable_loss_preserves_default_and_varies_coefficients():
    logs = torch.tensor([[-1.5, -.5]], requires_grad=True)
    values = torch.tensor([.2], requires_grad=True)
    args = (logs, values, torch.ones(1, 2, dtype=torch.bool), torch.tensor([0]),
            torch.tensor([-2.]), torch.tensor([.4]), torch.tensor([1.]))
    expected, em = losses(*args)
    actual, am = hpo_losses(*args, training=HPOTrainingConfig())
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert em == am
    recipe = replace(HPOTrainingConfig(), clip_range=.1, value_coefficient=.2,
                     entropy_coefficient=.3)
    other, measured = hpo_losses(*args, training=recipe)
    ratio = (logs[0, 0] + 2).exp()
    entropy = -(logs.exp() * logs).sum()
    formula = -torch.minimum(ratio * .4, ratio.clamp(.9, 1.1) * .4) + .2 * .8**2 - .3 * entropy
    torch.testing.assert_close(other, formula)
    assert measured["total_loss"] != em["total_loss"]


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_default_same_input_matches_old_learner_all_endpoints(kind):
    learner = learner_for(kind)
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    learner._training_execution=TrainingExecutionSpec(variant='baseline')
    raw = asdict(learner.model)
    raw.update(version="3", initialization="orthogonal-structured-v1")
    raw["temporal"]["sequence_length"] = 32
    old = GameLearner(model_config_v3(raw, layout=LAYOUT), 17, learner.binding,
        RESUME, actions=ACTIONS, rules=RULES)
    records = (records_for(learner),)
    old_rollout = game_rollout(records, old.model, actions=ACTIONS, rules=RULES)
    new_rollout = HpoRollout(records, learner.model, learner.training, actions=ACTIONS,
        rules=RULES, selected_locations=((0, 0), (0, 1), (0, 2)))
    for index in range(3):
        assert old_rollout.vector(index) == new_rollout.vector(index)
    old_metrics = old.update(old_rollout)
    new_metrics = learner.update(new_rollout)
    for name in ("policy_loss", "value_mse", "entropy", "approx_kl", "total_loss", "optimizer_steps"):
        assert new_metrics[name] == old_metrics[name]
    assert_same_learners(old, learner)


def test_selected_only_losses_complete_gae_normalization_and_step_counts(monkeypatch):
    recipe = replace(HPOTrainingConfig(), epochs=3, minibatch_size=2, gae_lambda=.6, tbptt_length=1)
    learner = learner_for("gru", recipe=recipe)
    rollout = rollout_for(learner, selected=((0, 0), (0, 2), (0, 4)), count=5)
    episode = rollout.episodes[0]
    raw, target = gae([selection.value for selection in episode.records.selections], 1, lam=.6)
    np.testing.assert_array_equal(episode.target, target.astype(np.float32))
    np.testing.assert_array_equal(episode.advantage[[0, 2, 4]], normalize_advantages(raw[[0, 2, 4]]))
    assert len(episode.frames) == rollout.collected_decisions == 5 and len(rollout) == 3
    seen = []
    vector = rollout.vector
    monkeypatch.setattr(rollout, "vector", lambda index: (seen.append(index), vector(index))[1])
    rng = torch.get_rng_state().clone()
    parity = learner.parity(rollout)
    assert torch.equal(rng, torch.get_rng_state())
    assert parity["decisions"] == 5
    result = learner.update(rollout)
    assert result["collected_decisions"] == 5 and result["used_decisions"] == 3
    assert result["optimizer_step_delta"] == recipe.epochs * 2 == learner.steps
    assert result["endpoint_evaluations"] == len(seen) == 9
    assert sorted(seen) == [0] * 3 + [1] * 3 + [2] * 3


def test_failed_unselected_frozen_parity_mutates_nothing():
    learner = learner_for()
    rollout = rollout_for(learner, selected=((0, 0),))
    episode = rollout.episodes[0]
    selection = episode.records.selections[2]
    object.__setattr__(episode.records, "selections", (*episode.records.selections[:2],
        replace(selection, value=selection.value + .5)))
    parameters = {name: value.clone() for name, value in learner.network.state_dict().items()}
    shuffle = copy.deepcopy(learner.shuffle.bit_generator.state)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ContractError, match="parity"):
        learner.update(rollout)
    assert learner.steps == 0 and not learner.optimizer.state
    assert canonical(shuffle) == canonical(learner.shuffle.bit_generator.state)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, learner.network.state_dict()[name]) for name, value in parameters.items())


@pytest.mark.parametrize("locations", [(), ((0, 1), (0, 0)), ((0, 0), (0, 0)), ((0, True),), ((0, 3),), ((1, 0),)])
def test_selection_locations_are_closed_sorted_unique_nonempty(locations):
    learner = learner_for()
    with pytest.raises(ContractError):
        rollout_for(learner, selected=locations)


def test_wrong_recipe_rejected_before_parity_or_optimizer(monkeypatch):
    learner = learner_for()
    rollout = rollout_for(learner)
    rollout.training = replace(learner.training, epochs=1)
    monkeypatch.setattr(learner.network, "forward", lambda *a, **kw: pytest.fail("recipe reached forward"))
    with pytest.raises(ContractError):
        learner.update(rollout)


@pytest.mark.parametrize("outcome,reward", [({"win": {"winner": 1}}, 1), ({"win": {"winner": 2}}, -1), ("draw", 0)])
def test_full_episode_reward_perspective_single_selected_and_interruption(outcome, reward):
    learner = learner_for(recipe=replace(HPOTrainingConfig(), gae_lambda=0.4))
    record = records_for(learner, outcome=outcome)
    rollout = HpoRollout((record,), learner.model, learner.training, actions=ACTIONS,
        rules=RULES, selected_locations=((0, 1),))
    _, target = gae([selection.value for selection in record.selections], reward, lam=.4)
    np.testing.assert_array_equal(rollout.episodes[0].target, target.astype(np.float32))
    assert rollout.vector(0)[3] == 0 and np.isfinite(rollout.vector(0)[4])
    interrupted = records_for(learner, kind="aborted")
    with pytest.raises(ContractError, match="natural"):
        HpoRollout((interrupted,), learner.model, learner.training, actions=ACTIONS,
            rules=RULES, selected_locations=((0, 0),))


def test_explicit_gradient_clip_and_integral_numeric_recipe_normalization(monkeypatch):
    recipe = replace(HPOTrainingConfig(), epochs=1, minibatch_size=2,
        max_grad_norm=.125, adam_learning_rate=1)
    learner = learner_for(recipe=recipe)
    assert type(learner.training.adam_learning_rate) is float
    clip = torch.nn.utils.clip_grad_norm_
    calls = []
    def measured(parameters, max_norm, **options):
        calls.append(max_norm)
        return clip(parameters, max_norm, **options)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", measured)
    learner.update(rollout_for(learner))
    assert calls == [.125, .125]
