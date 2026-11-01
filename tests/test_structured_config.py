"""Closed/inactive settings and preallocation parameter/resource proofs."""
import json
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError
from gf_common.structured import ActionConfigV2, action_config_v2
from gf_models.structured_config import (
    CardConfigV2,
    EventConfigV2,
    HandConfigV2,
    InferenceBindingV2,
    ModelConfigV2,
    ResourceConfigV2,
    StateConfigV2,
    TemporalConfigV2,
    architecture_identity,
    model_config_v2,
    resource_estimate,
    validate_config,
)


def test_closed_v2_config_roundtrip_and_no_optional_imports():
    assert model_config_v2(json.loads(json.dumps(asdict(ModelConfigV2())))) == ModelConfigV2()
    assert action_config_v2(json.loads(json.dumps(asdict(ActionConfigV2())))) == ActionConfigV2()
    for constructor, parser in ((ModelConfigV2, model_config_v2), (ActionConfigV2, action_config_v2)):
        with pytest.raises(ContractError):
            parser({**asdict(constructor()), "unknown": 2})
    code = ('import sys; import gf_common.structured; import gf_common.structured_tokens; '
            'import gf_models.structured_config; import gf_models.structured_network; '
            'assert "torch" not in sys.modules and "numpy" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize("model", [
    replace(ModelConfigV2(), version="1"),
    replace(ModelConfigV2(), max_positions=True),
    replace(ModelConfigV2(), max_positions=129),
    replace(ModelConfigV2(), dropout=False),
    replace(ModelConfigV2(), card=CardConfigV2(output_size=63)),
    replace(ModelConfigV2(), state=StateConfigV2(kind="identity", output_size=65)),
    replace(ModelConfigV2(), hand=HandConfigV2(kind="direct", pooling="max")),
    replace(ModelConfigV2(), hand=HandConfigV2(kind="deepsets", attention_layers=3)),
    replace(ModelConfigV2(), temporal=TemporalConfigV2(num_heads=2)),
    replace(ModelConfigV2(), temporal=TemporalConfigV2(sequence_length=1025)),
    replace(ModelConfigV2(), resources=ResourceConfigV2(max_parameters=10)),
    replace(ModelConfigV2(), resources=ResourceConfigV2(max_tensor_bytes=8)),
    replace(ModelConfigV2(), event=EventConfigV2(max_interval=65537)),
])
def test_invalid_config_rejects_before_torch_or_allocation(model):
    with pytest.raises(ContractError):
        validate_config(model)


def test_admission_and_runtime_binding_are_separate():
    model = ModelConfigV2()
    identity = architecture_identity(model)
    assert identity == architecture_identity(replace(model, resources=ResourceConfigV2(max_owned_bytes=2**30)))
    estimate = resource_estimate(model, InferenceBindingV2())
    assert estimate["total_bytes"] < 12 * 1024**3
    assert estimate["parameters"] > 0
    with pytest.raises(ContractError):
        resource_estimate(model, InferenceBindingV2(managed_budget_bytes=1))
    # Flexible event caps are accepted until the actual required tensor envelope
    # exceeds the shard/tensor guard; no according-to-cap allocation is made.
    larger = replace(model, event=EventConfigV2(max_interval=8192, max_episode=1_000_000))
    assert resource_estimate(larger, InferenceBindingV2())["total_bytes"] > estimate["total_bytes"]
    with pytest.raises(ContractError, match="tensor"):
        resource_estimate(replace(model, event=EventConfigV2(max_interval=65536)), InferenceBindingV2())


@pytest.mark.parametrize("settings", [{"max_candidates": True}, {"max_sequence_length": 129},
                                      {"channel_weights": (1,) * 6}, {"channel_weights": (0,) * 7},
                                      {"targets": ("self", "self")}, {"targets": ()},
                                      {"use_action_options": False}])
def test_action_config_strict(settings):
    with pytest.raises(ContractError):
        ActionConfigV2(**settings)
