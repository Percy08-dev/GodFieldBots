"""Version-first application configuration and closed own-execution bindings."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli.config import ExperimentConfig, RunConfigV2, load_config
from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, digest
from gf_common.structured import ActionConfigV2, Selection
from gf_models.config import PPOTrainingConfig, SequencePPOTrainingConfig, model_config
from gf_models.structured_binding import (
    ConfiguredNeuralModelV2,
    child_cuda_visibility,
    configured_model_v2,
    execution_digest,
    input_binding,
    model_defaults_v2,
    policy_label,
    safe_options_v2,
    validate_cuda_visibility,
)
from gf_models.structured_config import ModelConfigV2

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_TOML = "[actions.action_contract]\n" + "\n".join(f'{key}="{value}"' for key, value in asdict(engineering_contract()).items()) + "\n"
HEADER = ('schema_version=1\nkind="experiment"\n[server]\nfixed_seed_experiment=true\n'
          '[actions]\nversion="2"\n' + CONTRACT_TOML + '[preprocessing]\nbackend="python"\nreader="structured-v1"\n')


def configuration(tmp_path, model="mlp", extra=""):
    path = tmp_path / "experiment.toml"
    path.write_text(HEADER + f'[model]\nmodel_id="{model}"\nversion="2"\n'
                    '[training]\nalgorithm="ppo"\ndevice="cpu"\n' + extra)
    return path


def test_original_default_dataclass_hash_is_unchanged():
    # Captured from committed cfcb068 before the additive application dispatch.
    assert digest(asdict(ExperimentConfig())) == "bdb2b3663397693baf9bc059e9831c17ea91cbc85a3184e30d3d2f1aa6f0fa33"
    assert "global_cuda_budget_bytes" not in asdict(ExperimentConfig().run)
    assert "version" not in asdict(ExperimentConfig().actions)


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_model2_partial_dispatch_defaults_exact_base_conversion_and_one_window(tmp_path, kind):
    loaded = load_config(configuration(tmp_path, kind), {"model": {"state": {"output_size": 32},
                         "temporal": {"sequence_length": 7}, "inference": {"candidate_chunk": 64}}})
    model = loaded.config.model
    assert type(model) is ConfiguredNeuralModelV2
    assert type(model.architecture()) is ModelConfigV2
    assert model.state.output_size == 32 and model.temporal.sequence_length == 7
    assert model.inference.candidate_chunk == 64
    assert type(loaded.config.training) is PPOTrainingConfig
    assert not isinstance(loaded.config.training, SequencePPOTrainingConfig)
    assert "sequence_length" not in loaded.effective["training"]
    assert type(loaded.config.actions) is ActionConfigV2 and type(loaded.config.run) is RunConfigV2
    assert model.architecture().model_id == kind and model.architecture().temporal.kind == kind
    assert model_config(loaded.effective["model"]) == model
    assert configured_model_v2(json.loads(json.dumps(asdict(model)))) == model


def test_reference_origin_and_partial_override_inherit_version_before_validation(tmp_path):
    model = tmp_path / "model.toml"
    model.write_text('schema_version=1\nkind="model"\n[model]\nmodel_id="gru"\nversion="2"\ncheckpoint="weights"\n')
    experiment = tmp_path / "nested" / "experiment.toml"
    experiment.parent.mkdir()
    experiment.write_text(HEADER + '[references]\nmodel="../model.toml"\n[model.hand]\npooling="max"\n'
                          '[training]\nalgorithm="ppo"\ndevice="cpu"\n')
    loaded = load_config(experiment, {"model": {"hand": {"latent_size": 64}}})
    assert loaded.config.model.checkpoint == str(tmp_path / "weights")
    assert loaded.config.model.version == "2" and loaded.config.model.hand.pooling == "max"
    assert loaded.config.model.hand.latent_size == 64
    assert len(loaded.inputs) == 2


@pytest.mark.parametrize("override", [
    {"model": {"encoder": {"hash_buckets": 32}}},
    {"model": {"network": {"hidden_sizes": [128, 128]}}},
    {"training": {"sequence_length": 7}},
    {"run": {"concurrency": 3}}, {"run": {"max_participant_workers": 3}},
    {"run": {"global_cuda_budget_bytes": True}},
    {"preprocessing": {"reader": "legacy"}},
    {"opponent": {"model_id": "mlp"}},
    {"model": {"model_id": "lightgbm"}},
    {"model": {"version": "3"}},
])
def test_unknown_mixed_or_capacity_settings_reject_before_io(tmp_path, override):
    with pytest.raises(ContractError):
        load_config(configuration(tmp_path), override)


def test_actions2_baselines_keep_model1_and_require_explicit_reader(tmp_path):
    path = tmp_path / "baseline.toml"
    path.write_text(HEADER + '[model]\nmodel_id="random"\n[opponent]\nmodel_id="heuristic"\n')
    loaded = load_config(path)
    assert loaded.config.model.version == loaded.config.opponent.version == "1"
    assert loaded.effective["preprocessing"]["reader"] == "structured-v1"
    with pytest.raises(ContractError):
        load_config(ROOT / "configs/experiments/smoke.toml", {"preprocessing": {"reader": "structured-v1"}})


@pytest.mark.parametrize("value", [None, "", "0", "0,1", "GPU-12345678-1234-abcd-1234-123456789abc"])
def test_safe_visibility_is_explicit_not_inherited(value):
    assert validate_cuda_visibility(value) == value
    assert child_cuda_visibility("cpu", value) == ""
    assert child_cuda_visibility("cuda:0", value) == value


@pytest.mark.parametrize("value", [True, "-1", "0,0", "00", "GPU-short", "0\nTOKEN=secret", "secret", "0;evil"])
def test_unsafe_visibility_rejects(value):
    with pytest.raises(ContractError):
        validate_cuda_visibility(value)


def test_execution_identity_includes_actual_resources_device_and_visibility():
    model = model_defaults_v2().architecture()
    binding = model_defaults_v2().inference
    identities = {execution_digest(model, binding, cuda_visible_devices=visibility) for visibility in (None, "", "0")}
    identities.add(execution_digest(replace(model, resources=replace(model.resources, max_owned_bytes=2 * 1024**3)), binding))
    identities.add(execution_digest(model, replace(binding, candidate_chunk=1)))
    assert len(identities) == 5
    label = policy_label(model, binding, "a" * 64, cuda_visible_devices="")
    assert "a" * 64 in label and execution_digest(model, binding, cuda_visible_devices="") in label
    own = {"version": "2", "checkpoint": "/own/checkpoint", "expected_manifest_digest": "a" * 64,
           "model": asdict(model), "inference": asdict(binding), "cuda_visible_devices": "",
           **input_binding(engineering_contract(), "component", False)}
    assert safe_options_v2(own, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False) == json.loads(json.dumps(own))
    with pytest.raises(ContractError):
        safe_options_v2({**own, "game_seed": "forbidden", **input_binding(engineering_contract(), "component", False)}, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def test_baseline_null_selection_roundtrip_and_neural_finite_rules():
    baseline = Selection("a" * 64, 0, "random-structured-v1")
    assert Selection(**asdict(baseline)) == baseline
    assert baseline.logprob is baseline.value is None
    with pytest.raises(ContractError):
        Selection("a" * 64, 0, "neural", float("nan"), 0.)
    with pytest.raises(ContractError):
        Selection("a" * 64, 0, "neural", .1, 0.)


def test_config_and_binding_modules_remain_optional_runtime_free():
    code = ('import sys; import gf_cli.config; import gf_models.structured_binding; '
            'assert "torch" not in sys.modules')
    subprocess.run([sys.executable, "-c", code], check=True)
    binding_only = ('import sys; import gf_models.structured_binding; '
                    'assert "torch" not in sys.modules and "numpy" not in sys.modules')
    subprocess.run([sys.executable, "-c", binding_only], check=True)


def test_ppodata_rejects_baseline_null_before_tokenization(make_input, monkeypatch):
    import gf_training.structured_rollout as module
    from tests.test_structured_learning import ACTION_HASH, learner_for, records_for
    learner = learner_for()
    record = records_for(make_input, learner, count=1)
    selection = replace(record.selections[0], logprob=None, value=None)
    baseline = module.EpisodeRecords(record.bank, record.frames, record.tables, (selection,), record.end_kind, record.outcome, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    def forbidden(*args, **kwargs):
        pytest.fail("PPO null-value row reached deterministic token allocation")
    monkeypatch.setattr(module, "tokenize_frame", forbidden)
    with pytest.raises(ContractError, match="finite logprob and value"):
        module.StructuredRollout((baseline,), learner.model, ACTION_HASH, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)


@pytest.mark.parametrize("seed", [0, 2**63, 2**64 - 1])
def test_structured_sampling_preserves_existing_uint64_domain(seed):
    from gf_models.structured_policy import StructuredPolicy
    from tests.test_structured_learning import learner_for
    learner = learner_for()
    policy = StructuredPolicy(learner.network, learner.model, learner.binding, "a" * 64, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    probabilities = learner.torch.tensor([.2, .3, .5])
    policy.reset(seed)
    expected = learner.torch.multinomial(probabilities, 12, replacement=True, generator=policy.rng)
    policy.reset(seed)
    actual = learner.torch.multinomial(probabilities, 12, replacement=True, generator=policy.rng)
    assert learner.torch.equal(expected, actual)


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.])
def test_structured_sampling_rejects_invalid_uint64_values(seed):
    from gf_models.structured_policy import StructuredPolicy
    from tests.test_structured_learning import learner_for
    learner = learner_for()
    policy = StructuredPolicy(learner.network, learner.model, learner.binding, "a" * 64, action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    with pytest.raises(ContractError):
        policy.reset(seed)
