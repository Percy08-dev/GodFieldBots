"""HPO configuration admission is pure and preserves historical recipes."""
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV3,
    architecture_identity,
    parameter_shapes,
    resource_estimate,
)

ROOT = Path(__file__).parents[1]


def test_model4_default_shapes_order_match_model3_without_training_fields():
    from gf_models.hpo_config import model_config_v4, model_defaults_v4
    for kind in ("mlp", "gru", "transformer"):
        model = model_defaults_v4(kind)
        old = replace(ModelConfigV3(), model_id=kind,
                      temporal=replace(ModelConfigV3().temporal, kind=kind,
                                       num_layers=2 if kind == "transformer" else 1))
        assert list(parameter_shapes(model, layout=LAYOUT).items()) == list(parameter_shapes(old, layout=LAYOUT).items())
        assert "sequence_length" not in asdict(model)["temporal"]
        assert model_config_v4(asdict(model), layout=LAYOUT) == model
        assert architecture_identity(model, layout=LAYOUT) != architecture_identity(old, layout=LAYOUT)


def test_model4_depth_activation_and_inactive_normalization():
    from gf_models.hpo_config import activation_gain, model_config_v4, model_defaults_v4
    raw = asdict(model_defaults_v4())
    raw["temporal"].update(num_layers=3, activation="tanh", num_heads=7, feedforward_size=99)
    raw["state"].update(kind="identity", hidden_sizes=[19], output_size=17, activation="tanh")
    model = model_config_v4(raw, layout=LAYOUT)
    shapes = parameter_shapes(model, layout=LAYOUT)
    assert shapes["temporal_dense.0.weight"] == (128, 128)
    assert shapes["temporal_dense.2.weight"] == (128, 128)
    assert model.temporal.num_heads == 4 and model.temporal.feedforward_size == 256
    assert model.state.hidden_sizes == (128,) and model.state.activation == "relu"
    assert activation_gain("tanh") == 5 / 3
    assert activation_gain("relu") ** 2 == pytest.approx(2)
    other = asdict(model)
    other["state"].update(output_size=99, hidden_sizes=[91])
    assert architecture_identity(model_config_v4(other, layout=LAYOUT), layout=LAYOUT) == architecture_identity(model, layout=LAYOUT)


@pytest.mark.parametrize("path,value", [
    ("version", "3"), ("initialization", "orthogonal-structured-v1"), ("dropout", False),
    ("temporal.num_layers", True), ("temporal.num_layers", 5),
    ("temporal.max_decisions", 32), ("temporal.sequence_length", 32),
    ("card.output_size", float("nan")), ("hand.attention_heads", 3),
    ("heads.activation", "gelu"), ("state.hidden_sizes", [True]),
    ("temporal.kind", "gru"), ("unknown", 1),
])
def test_model4_rejects_closed_types_shapes_and_fixed_contract(path, value):
    from gf_models.hpo_config import model_config_v4, model_defaults_v4
    raw = asdict(model_defaults_v4())
    parts = path.split(".")
    target = raw if len(parts) == 1 else raw[parts[0]]
    target[parts[-1]] = value
    with pytest.raises(ContractError):
        model_config_v4(raw, layout=LAYOUT)


def test_model4_resource_default_uses_inference_suffix_and_training_explicit_tbptt():
    from gf_models.hpo_config import model_defaults_v4
    model = model_defaults_v4()
    binding = InferenceBindingV2(device="cuda:0")
    assert resource_estimate(model, binding, layout=LAYOUT) == resource_estimate(model, binding, layout=LAYOUT, suffix_decisions=1)
    assert resource_estimate(model, binding, layout=LAYOUT, suffix_decisions=32)["total_bytes"] > resource_estimate(model, binding, layout=LAYOUT)["total_bytes"]


def test_hpo_recipe_keeps_tbptt_out_of_inference_identity():
    from gf_models.hpo_config import model_defaults_v4
    from gf_training.hpo_config import HPOTrainingConfig, effective_settings, hpo_training_config
    model = model_defaults_v4()
    recipe = HPOTrainingConfig()
    assert hpo_training_config(asdict(recipe)) == recipe
    first = effective_settings(model, recipe, {}, layout=LAYOUT)
    second = effective_settings(model, recipe, {"training.tbptt_length": 64, "training.adam_learning_rate": 0.001}, layout=LAYOUT)
    assert first.model == second.model and first.identity != second.identity
    assert architecture_identity(first.model, layout=LAYOUT) == architecture_identity(second.model, layout=LAYOUT)


@pytest.mark.parametrize("key,value", [
    ("collection_games", 7), ("collection_games", True), ("epochs", 0),
    ("adam_beta1", 1.0), ("adam_beta2", -1.0), ("adam_epsilon", float("inf")),
    ("adam_learning_rate", True), ("gamma", .99), ("optimizer", "sgd"),
    ("reward", "shaped"), ("gae_lambda", 1.01), ("clip_range", 0),
    ("value_coefficient", -1), ("tbptt_length", 1025), ("unknown", 1),
])
def test_hpo_recipe_closed_finite_and_fixed_rules(key, value):
    from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config
    with pytest.raises(ContractError):
        hpo_training_config({**asdict(HPOTrainingConfig()), key: value})


def space_raw():
    return {"parameters": [
        {"path": "model.hand.kind", "kind": "categorical", "choices": ["direct", "attention"]},
        {"path": "model.heads.actor_hidden_sizes", "kind": "categorical", "choices": [[32], [64, 32]]},
        {"path": "training.adam_learning_rate", "kind": "log", "low": 0.0001, "high": 0.001},
        {"path": "training.epochs", "kind": "int", "low": 1, "high": 4, "step": 1},
        {"path": "training.entropy_coefficient", "kind": "float", "low": 0., "high": .1},
        {"path": "training.tbptt_length", "kind": "fixed", "value": 32},
    ], "coupled": [{"name": "hand_attention", "paths": ["model.card.output_size", "model.hand.attention_heads"],
        "choices": [[32, 2], [64, 4]], "when": {"model.hand.kind": ["attention"]}}]}


def test_search_domains_conditions_coupling_and_effective_deduplication():
    from gf_models.hpo_config import model_defaults_v4
    from gf_training.hpo_config import (
        HPOTrainingConfig,
        effective_settings,
        search_space,
        valid_attention_choices,
    )
    space = search_space(space_raw(), model_id="mlp")
    assert valid_attention_choices([32, 33], [2, 3]) == ((32, 2), (33, 3))
    assert space.parameters[1].choices == ((32,), (64, 32))
    chosen = space.resolve({"model.hand.kind": 1, "model.heads.actor_hidden_sizes": 0,
        "training.adam_learning_rate": .0003, "training.epochs": 2,
        "training.entropy_coefficient": .01, "hand_attention": 0})
    assert chosen["model.card.output_size"] == 32 and chosen["training.tbptt_length"] == 32
    settings = effective_settings(model_defaults_v4(), HPOTrainingConfig(), chosen, layout=LAYOUT)
    assert settings.model.hand.attention_heads == 2
    direct = {**chosen, "model.hand.kind": "direct"}
    left = effective_settings(model_defaults_v4(), HPOTrainingConfig(), direct, layout=LAYOUT)
    right = effective_settings(model_defaults_v4(), HPOTrainingConfig(), {**direct, "model.hand.attention_heads": 4}, layout=LAYOUT)
    assert left.identity == right.identity
    with pytest.raises(ContractError):
        space.resolve({"model.hand.kind": 1})


@pytest.mark.parametrize("mutate", [
    lambda r: r["parameters"].append(r["parameters"][0]),
    lambda r: r["parameters"][0].update(choices=["attention", "attention"]),
    lambda r: r["parameters"][2].update(low=0),
    lambda r: r["parameters"][3].update(step=True),
    lambda r: r["parameters"][0].update(path="operating.game_concurrency"),
    lambda r: r["coupled"][0].update(choices=[[33, 2]]),
    lambda r: r["coupled"][0].update(when={"unknown": [1]}),
])
def test_search_rejects_invalid_and_operating_domains(mutate):
    from gf_training.hpo_config import search_space
    raw = space_raw()
    mutate(raw)
    with pytest.raises(ContractError):
        search_space(raw, model_id="mlp")


def campaign_raw():
    return {"schema_version": 1, "kind": "ppo-hpo", "base_experiment": str(ROOT / "configs/experiments/card-instance-mlp-ppo-acceptance.toml"),
        "model_id": "mlp", "stage": "explore", "trial_count": 2, "decision_budget": 100,
        "learning_seeds": [17, 29], "evaluation_game_seeds": ["00000000-0000-0000-0000-000000000001"],
        "sampler_seed": 19, "startup_trials": 1, "session_seconds": 3600, "run_wall_seconds": 7200,
        "space": space_raw()}


def test_campaign_effective_config_paths_base_identity_and_joint_admission(tmp_path):
    from gf_cli.config import load_config
    from gf_cli.hpo_config import campaign_config, joint_resource_admission
    raw = campaign_raw()
    loaded = campaign_config(raw, origin=tmp_path)
    assert loaded.base.sha256 == load_config(raw["base_experiment"]).sha256
    assert loaded.config.operating.managed_ram_bytes == 24 * 1024**3
    assert loaded.config.operating.managed_cuda_bytes == 12 * 1024**3
    assert loaded.config.model.version == "4" and loaded.config.inference.device == "cuda:0"
    admission = joint_resource_admission(loaded.config, layout=LAYOUT)
    assert admission["workers"] == 4
    assert admission["peak_ram_bytes"] >= admission["learner_bytes"] + admission["worker_bytes"]
    assert set(admission["ram_phases"]) == {"collection", "update", "save"}
    changed = campaign_config({**raw, "operating": {"managed_ram_bytes": 1}}, origin=tmp_path)
    with pytest.raises(ContractError, match="RAM"):
        joint_resource_admission(changed.config, layout=LAYOUT)
    changed = campaign_config({**raw, "operating": {"managed_cuda_bytes": 1}}, origin=tmp_path)
    with pytest.raises(ContractError, match="CUDA"):
        joint_resource_admission(changed.config, layout=LAYOUT)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("space"), lambda r: r.pop("decision_budget"), lambda r: r.pop("trial_count"),
    lambda r: r.pop("learning_seeds"), lambda r: r.pop("evaluation_game_seeds"),
    lambda r: r.update(learning_seeds=[17, 17]), lambda r: r.update(evaluation_game_seeds=["bad"]),
    lambda r: r.update(startup_trials=True), lambda r: r.update(model_id="gru"),
    lambda r: r.update(session_seconds=float("nan")), lambda r: r.update(unknown=1),
    lambda r: r.update(operating={"trial_concurrency": 2}),
])
def test_campaign_rejects_missing_required_and_bad_conditions_before_io(tmp_path, mutate):
    from gf_cli.hpo_config import campaign_config
    raw = campaign_raw()
    mutate(raw)
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)
    assert not list(tmp_path.iterdir())


def test_hpo_loader_is_toml_explicit_and_imports_no_optional_learning_stack(tmp_path):
    from gf_cli.hpo_config import load_hpo_config
    path = tmp_path / "campaign.toml"
    path.write_text('schema_version=1\nkind="ppo-hpo"\n')
    with pytest.raises(ContractError):
        load_hpo_config(path)
    subprocess.run([sys.executable, "-c", "import sys; import gf_cli.hpo_config; assert not ({'torch','optuna'} & sys.modules.keys())"], check=True)


def test_historical_model_and_eight_frozen_kernels_remain_exact():
    from gf_models.structured_config import model_config_v3
    from gf_training.card_instance_formal import load_recipe, verify_kernel_ast
    verify_kernel_ast(load_recipe())
    raw = asdict(ModelConfigV3())
    raw["heads"]["activation"] = "tanh"
    with pytest.raises(ContractError):
        model_config_v3(raw, layout=LAYOUT)


def test_configured_model4_dispatch_preserves_cuda_binding_and_no_checkpoint_relabel():
    from gf_models.structured_binding import configured_model_v4, model_defaults_v4
    configured = model_defaults_v4("gru")
    parsed = configured_model_v4(asdict(configured), layout=LAYOUT)
    assert parsed == configured and parsed.inference.device == "cuda:0"
    assert parsed.architecture(layout=LAYOUT).temporal.kind == "gru"
    assert "sequence_length" not in asdict(parsed.architecture(layout=LAYOUT))["temporal"]
    raw = asdict(configured)
    raw["checkpoint"] = True
    with pytest.raises(ContractError):
        configured_model_v4(raw, layout=LAYOUT)


def test_search_transformer_coupling_accepts_three_heads_and_checks_conditions():
    from gf_training.hpo_config import search_space
    raw = {"parameters": [{"path": "training.epochs", "kind": "fixed", "value": 4}],
           "coupled": [{"name": "attention", "paths": ["model.temporal.hidden_size", "model.temporal.num_heads"],
                         "choices": [[33, 3], [32, 4]]}]}
    space = search_space(raw, model_id="transformer")
    assert space.resolve({"attention": 0})["model.temporal.hidden_size"] == 33
    raw["coupled"][0]["when"] = {"training.epochs": [True]}
    with pytest.raises(ContractError):
        search_space(raw, model_id="transformer")


def test_search_rejects_unsafe_marginals_and_off_grid_proposals():
    from gf_training.hpo_config import search_space
    raw = {"parameters": [
        {"path": "model.card.output_size", "kind": "categorical", "choices": [32, 33]},
        {"path": "model.hand.attention_heads", "kind": "categorical", "choices": [2, 3]},
    ], "coupled": []}
    with pytest.raises(ContractError):
        search_space(raw, model_id="mlp")
    raw = {"parameters": [{"path": "training.epochs", "kind": "int", "low": 1, "high": 5, "step": 2}], "coupled": []}
    space = search_space(raw, model_id="mlp")
    for proposal in ({"training.epochs": 2}, {"training.epochs": True}, {"training.epochs": 7}, {"unknown": 1}):
        with pytest.raises(ContractError):
            space.resolve(proposal)


def test_search_conditions_drop_inactive_fields_and_fixed_dimensions_are_implicit():
    from gf_training.hpo_config import search_space
    raw = space_raw()
    space = search_space(raw, model_id="mlp")
    proposal = {"model.hand.kind": 0, "model.heads.actor_hidden_sizes": 1,
                "training.adam_learning_rate": .0003, "training.epochs": 2,
                "training.entropy_coefficient": .01}
    resolved = space.resolve(proposal)
    assert "model.hand.attention_heads" not in resolved
    assert resolved["training.tbptt_length"] == 32
    for extra in ({"hand_attention": 0}, {"training.tbptt_length": 32}):
        with pytest.raises(ContractError):
            space.resolve({**proposal, **extra})


def test_campaign_study_identity_binds_experiment_and_ignores_operating_settings(tmp_path):
    from gf_cli.hpo_config import campaign_config
    raw = campaign_raw()
    first = campaign_config(raw, origin=tmp_path)
    other = campaign_config({**raw, "operating": {"game_concurrency": 1}, "session_seconds": 100}, origin=tmp_path)
    assert first.config.study_identity == other.config.study_identity
    assert first.sha256 != other.sha256
    changed = campaign_config({**raw, "decision_budget": 101}, origin=tmp_path)
    assert first.config.study_identity != changed.config.study_identity
    base = tmp_path / "different-base.toml"
    base.write_text(Path(raw["base_experiment"]).read_text().replace('"../', '"' + str(ROOT / "configs") + '/').replace('name = "card-instance-mlp-ppo-acceptance"', 'name = "different-base"'))
    modified = campaign_config({**raw, "base_experiment": str(base)}, origin=tmp_path)
    assert first.config.study_identity != modified.config.study_identity


def test_valid_campaign_toml_load_and_origin_relative_storage(tmp_path):
    from gf_cli.hpo_config import load_hpo_config
    path = tmp_path / "campaign.toml"
    path.write_text('schema_version=1\nkind="ppo-hpo"\nbase_experiment="' + str(ROOT / "configs/experiments/card-instance-mlp-ppo-acceptance.toml") + '"\n'
        'model_id="mlp"\nstage="explore"\ntrial_count=2\ndecision_budget=100\nlearning_seeds=[17]\n'
        'evaluation_game_seeds=["00000000-0000-0000-0000-000000000001"]\nsampler_seed=17\nstartup_trials=1\n'
        'session_seconds=3600\nrun_wall_seconds=7200\n[space]\ncoupled=[]\n'
        '[[space.parameters]]\npath="training.epochs"\nkind="int"\nlow=1\nhigh=4\n'
        '[storage]\nparent_root="parents"\nnumeric_root="numeric"\noptuna_path="search.sqlite3"\noriginals_root="/mnt/d/hpo"\n')
    loaded = load_hpo_config(path)
    assert loaded.config.storage.parent_root == str(tmp_path / "parents")
    assert loaded.config.storage.originals_root == "/mnt/d/hpo"
    assert loaded.inputs[-1]["kind"] == "ppo-hpo"
    assert not (tmp_path / "parents").exists()


def test_large_nonfinite_numeric_values_are_contract_errors():
    from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config
    with pytest.raises(ContractError):
        hpo_training_config({**asdict(HPOTrainingConfig()), "adam_learning_rate": 10**1000})


@pytest.mark.parametrize("path,value", [
    ("event.max_episode", "bad"), ("event.max_episode", True),
    ("event.max_interval", {}), ("event.aggregate_window", False),
    ("resources.max_owned_bytes", "bad"),
])
def test_model4_malformed_component_numbers_raise_contract_error(path, value):
    from gf_models.hpo_config import model_config_v4, model_defaults_v4
    raw = asdict(model_defaults_v4())
    component, key = path.split(".")
    raw[component][key] = value
    with pytest.raises(ContractError):
        model_config_v4(raw, layout=LAYOUT)


def test_search_conditions_can_refer_to_fixed_base_fields_and_event_cap_coupling():
    from gf_training.hpo_config import search_space
    raw = {"parameters": [{"path": "model.hand.latent_size", "kind": "categorical", "choices": [32, 64],
                            "when": {"model.hand.kind": ["attention"]}}],
           "coupled": [{"name": "events", "paths": ["model.event.max_interval", "model.event.max_episode", "model.event.aggregate_window"],
                        "choices": [[64, 64, 32]]}]}
    space = search_space(raw, model_id="mlp")
    chosen = space.resolve({"model.hand.latent_size": 1, "events": 0})
    assert chosen["model.hand.latent_size"] == 64 and chosen["model.event.max_episode"] == 64


def test_search_joint_choices_and_integer_field_domains_are_prevalidated():
    from gf_training.hpo_config import search_space
    invalid = [
        {"parameters": [], "coupled": [{"name": "events", "paths": ["model.event.max_interval", "model.event.max_episode"], "choices": [[128, 64]]}]},
        {"parameters": [{"path": "model.state.output_size", "kind": "float", "low": 32, "high": 64}], "coupled": []},
        {"parameters": [{"path": "model.event.max_episode", "kind": "int", "low": 32, "high": 64}], "coupled": []},
    ]
    for raw in invalid:
        with pytest.raises(ContractError):
            search_space(raw, model_id="mlp")


def test_model4_pure_dimensions_reject_historical_card_instance_layout():
    from gf_common.structured_layout import CARD_INSTANCE_LAYOUT
    from gf_models.hpo_config import model_defaults_v4
    from gf_models.structured_config import dimensions
    with pytest.raises(ContractError):
        dimensions(model_defaults_v4(), layout=CARD_INSTANCE_LAYOUT)


@pytest.mark.parametrize("field,value", [("pool_every_decisions", 5000), ("opponent_pool_size", 3)])
def test_study_identity_binds_fixed_opponent_curriculum(tmp_path, field, value):
    from gf_cli.hpo_config import campaign_config
    raw = campaign_raw()
    original = campaign_config(raw, origin=tmp_path)
    changed = campaign_config({**raw, "operating": {field: value}}, origin=tmp_path)
    assert original.config.study_identity != changed.config.study_identity


def test_campaign_search_uses_actual_attention_control_and_does_not_mutate_input(tmp_path):
    from copy import deepcopy

    from gf_cli.hpo_config import campaign_config
    raw = campaign_raw()
    raw["model"] = {"card": {"output_size": 33}, "hand": {"attention_heads": 3}}
    raw["space"] = {"parameters": [{"path": "model.hand.attention_heads", "kind": "categorical", "choices": [2, 4]}], "coupled": []}
    before = deepcopy(raw)
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)
    assert raw == before


def test_conditional_width_is_valid_only_in_reachable_direct_hand_branch(tmp_path):
    from copy import deepcopy

    from gf_cli.hpo_config import campaign_config
    from gf_training.hpo_config import effective_settings
    raw = campaign_raw()
    raw["space"] = {"parameters": [
        {"path": "model.hand.kind", "kind": "categorical", "choices": ["attention", "direct"]},
        {"path": "model.card.output_size", "kind": "fixed", "value": 33, "when": {"model.hand.kind": ["direct"]}},
    ], "coupled": []}
    before = deepcopy(raw)
    campaign = campaign_config(raw, origin=tmp_path).config
    for index, expected_width in ((0, 64), (1, 33)):
        values = campaign.space.resolve({"model.hand.kind": index})
        effective = effective_settings(campaign.model, campaign.training, values, layout=LAYOUT)
        assert effective.model.card.output_size == expected_width
    assert raw == before


def test_bound_control_drives_resolve_conditions_and_rejects_rebinding(tmp_path):
    import json

    from gf_cli.hpo_config import campaign_config
    from gf_common.domain import canonical
    from gf_training.hpo_config import effective_settings
    raw = campaign_raw()
    raw["training"] = {"epochs": 3}
    raw["space"] = {"parameters": [
        {"path": "model.hand.kind", "kind": "fixed", "value": "direct", "when": {"training.epochs": [3]}},
        {"path": "model.card.output_size", "kind": "fixed", "value": 33, "when": {"model.hand.kind": ["direct"]}},
    ], "coupled": []}
    campaign = campaign_config(raw, origin=tmp_path).config
    assert dict(campaign.space.control_values)["training.epochs"] == 3
    values = campaign.space.resolve({})
    assert values["model.card.output_size"] == 33
    assert effective_settings(campaign.model, campaign.training, values, layout=LAYOUT).model.hand.kind == "direct"
    assert canonical(asdict(campaign.space)) == canonical(json.loads(canonical(asdict(campaign.space))))
    with pytest.raises(ContractError, match="validated search control"):
        campaign.space.resolve({}, base_values={"training.epochs": 4})


def test_actual_control_with_partial_coupled_dimensions_is_checked(tmp_path):
    from gf_cli.hpo_config import campaign_config
    from gf_training.hpo_config import effective_settings
    raw = campaign_raw()
    raw["model"] = {"card": {"output_size": 33}, "hand": {"attention_heads": 3}}
    raw["space"] = {"parameters": [], "coupled": [{"name": "heads_and_epochs", "paths": ["model.hand.attention_heads", "training.epochs"], "choices": [[3, 2], [11, 4]]}]}
    campaign = campaign_config(raw, origin=tmp_path).config
    for index in (0, 1):
        values = campaign.space.resolve({"heads_and_epochs": index})
        effective = effective_settings(campaign.model, campaign.training, values, layout=LAYOUT)
        assert effective.model.card.output_size == 33
        assert effective.model.hand.attention_heads == (3 if index == 0 else 11)
    raw["space"]["coupled"][0]["choices"].append([4, 4])
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)


def test_conditional_coupled_attention_rows_skip_inactive_direct_branch(tmp_path):
    from gf_cli.hpo_config import campaign_config
    from gf_training.hpo_config import effective_settings
    raw = campaign_raw()
    raw["space"] = {"parameters": [
        {"path": "model.hand.kind", "kind": "categorical", "choices": ["attention", "direct"]},
    ], "coupled": [{"name": "direct", "paths": ["model.card.output_size", "model.hand.attention_heads"],
                    "choices": [[33, 2], [35, 3]], "when": {"model.hand.kind": ["direct"]}}]}
    campaign = campaign_config(raw, origin=tmp_path).config
    for proposal in ({"model.hand.kind": 0}, {"model.hand.kind": 1, "direct": 0}, {"model.hand.kind": 1, "direct": 1}):
        effective_settings(campaign.model, campaign.training, campaign.space.resolve(proposal), layout=LAYOUT)
    raw["space"]["coupled"][0]["when"] = {"model.hand.kind": ["attention"]}
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)


def test_numeric_condition_complements_and_transitive_dependencies_are_checked():
    from gf_training.hpo_config import search_space
    raw = {"parameters": [
        {"path": "training.epochs", "kind": "int", "low": 1, "high": 1000},
        {"path": "model.hand.kind", "kind": "fixed", "value": "direct", "when": {"training.epochs": [1, 1000]}},
        {"path": "model.card.output_size", "kind": "fixed", "value": 33},
    ], "coupled": []}
    # Endpoint-only validation would miss epochs2..999's active attention.
    with pytest.raises(ContractError):
        search_space(raw, model_id="mlp")
    raw["parameters"][2]["when"] = {"model.hand.kind": ["direct"]}
    space = search_space(raw, model_id="mlp")
    assert space.resolve({"training.epochs": 500}) == {"training.epochs": 500}
    assert space.resolve({"training.epochs": 1000})["model.card.output_size"] == 33


def test_actual_event_control_and_reachable_joint_cap_branches(tmp_path):
    from gf_cli.hpo_config import campaign_config
    from gf_training.hpo_config import effective_settings
    raw = campaign_raw()
    raw["model"] = {"event": {"max_interval": 64, "max_episode": 64, "aggregate_window": 32}}
    raw["space"] = {"parameters": [
        {"path": "model.hand.kind", "kind": "categorical", "choices": ["attention", "direct"]},
    ], "coupled": [{"name": "events", "paths": ["model.event.max_interval", "model.event.max_episode", "model.event.aggregate_window"],
                    "choices": [[8, 8, 8], [16, 32, 16]], "when": {"model.hand.kind": ["direct"]}}]}
    campaign = campaign_config(raw, origin=tmp_path).config
    for proposal in ({"model.hand.kind": 0}, {"model.hand.kind": 1, "events": 0}, {"model.hand.kind": 1, "events": 1}):
        effective_settings(campaign.model, campaign.training, campaign.space.resolve(proposal), layout=LAYOUT)
    raw["space"] = {"parameters": [{"path": "model.event.max_episode", "kind": "int", "low": 32, "high": 1000}], "coupled": []}
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)
    raw["space"]["parameters"][0]["low"] = 64
    campaign = campaign_config(raw, origin=tmp_path).config
    assert campaign.space.resolve({"model.event.max_episode": 65})["model.event.max_episode"] == 65


def test_search_refuses_excessive_conditional_validation_states():
    from gf_training.hpo_config import search_space
    choices = {
        "training.collection_games": [8, 16, 24, 32],
        "training.adam_learning_rate": [.0001, .0002, .0003, .0004],
        "training.adam_beta1": [.1, .2, .3, .4],
        "training.adam_beta2": [.1, .2, .3, .4],
        "training.adam_epsilon": [.0001, .0002, .0003, .0004],
        "training.epochs": [1, 2, 3, 4],
        "training.minibatch_size": [1, 2, 3, 4],
        "training.clip_range": [.1, .2, .3, .4],
        "training.value_coefficient": [.1, .2, .3, .4],
    }
    parameters = [{"path": path, "kind": "categorical", "choices": values} for path, values in choices.items()]
    parameters.append({"path": "model.card.output_size", "kind": "fixed", "value": 64,
                       "when": {path: [values[0]] for path, values in choices.items()}})
    with pytest.raises(ContractError, match="65536.*simplify"):
        search_space({"parameters": parameters, "coupled": []}, model_id="mlp")
