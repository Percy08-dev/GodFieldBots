"""Metadata-only Config3 dispatch and independently recovered old Config hashes.

Old identities recovered from complete committed0f87 package by root; receipt
runs/assessments/card-instance-corrected-adoption-20261104/pre-model3-configs.json
SHA44674705ab4a29a36d6169890d89dc3bf9358facc90d735207339546fb169e9e.
"""

from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.config import ExperimentConfig, load_config
from gf_common.card_instance import card_instance_contract
from gf_common.domain import ContractError, digest
from gf_common.structured_layout import layout_for_contract
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_models.structured_config import dimensions
from tests.test_card_instance_contract import relocate_effective_paths

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["mlp", "gru", "transformer"])
def test_model3_presets_select_corrected_layout_and_explicit_capacity(name):
    loaded = load_config(ROOT / f"configs/experiments/card-instance-{name}-contract.toml")
    assert type(loaded.config.model) is ConfiguredNeuralModelV3
    assert loaded.config.model.model_id == name and loaded.config.model.version == "3"
    layout = layout_for_contract(loaded.config.actions.action_contract)
    model = loaded.config.model.architecture(layout=layout)
    assert dimensions(model, layout=layout)["card_actor"] == 71
    assert loaded.config.actions.max_sequence_length <= model.max_positions
    assert loaded.config.actions.max_candidates <= loaded.config.model.inference.max_candidates
    altered = load_config(
        ROOT / f"configs/experiments/card-instance-{name}-contract.toml",
        {"model": {"heads": {"critic_hidden_sizes": [32]}, "hand": {"kind": "deepsets"}}},
    )
    assert (
        altered.config.model.heads.critic_hidden_sizes == (32,)
        and altered.config.model.hand.kind == "deepsets"
    )
    assert altered.sha256 != loaded.sha256


@pytest.mark.parametrize(
    "override",
    [
        {"model": {"version": "2"}},
        {"model": {"version": "1"}},
        {"model": {"model_id": "lightgbm"}},
        {"model": {"unknown": 1}},
        {"model": {"event": {"aggregate_recipe": "public-exact-v2"}}},
        {"model": {"max_positions": 1}},
        {"model": {"inference": {"max_candidates": 1}}},
    ],
)
def test_model3_config_mixed_versions_tree_unknown_or_insufficient_capacity_reject(override):
    with pytest.raises(ContractError):
        load_config(ROOT / "configs/experiments/card-instance-mlp-contract.toml", override)


def test_model3_active_config_requires_corrected_profile_and_old_defaults_stay_exact():
    with pytest.raises(ContractError):
        load_config(
            ROOT / "configs/experiments/card-instance-v2-contract.toml",
            {"model": {"version": "3", "model_id": "mlp"}},
        )
    with pytest.raises(ContractError):
        load_config(
            ROOT / "configs/experiments/card-instance-mlp-contract.toml",
            {"actions": {"action_contract": asdict(card_instance_contract())}},
        )
    assert BEFORE["base_revision"] == "0f87d785c0b98d106f97df5670ef338b8322842b"
    assert len(BEFORE["experiments"]) == 14
    assert digest(asdict(ExperimentConfig())) == BEFORE["default_config_sha256"]
    for name, expected in BEFORE["experiments"].items():
        current = load_config(ROOT / name)
        restored_origin = relocate_effective_paths(
            current.effective, source_parent=ROOT.parent, target_parent=Path("/home/percy/develop")
        )
        assert digest(restored_origin) == expected, name


def test_model3_reference_partial_override_and_run_guard_stays_closed(tmp_path):
    import asyncio

    import gf_cli

    candidate = tmp_path / "experiment.toml"
    candidate.write_text(
        'schema_version=1\nkind="experiment"\n[references]\nactions="'
        + str(ROOT / "configs/actions/card-instance-v2-corrected.toml")
        + '"\nmodel="'
        + str(ROOT / "configs/models/card-instance-gru.toml")
        + '"\n[server]\ngameplay_api="v2"\nsimulator_revision="91d5926a7572db327eaea3751e7c24f428120996"\nfixed_seed_experiment=true\n'
        + "[model.heads]\nactor_hidden_sizes=[32,16]\n"
    )
    loaded = load_config(candidate, {"run": {"output_dir": str(tmp_path / "forbidden")}})
    assert (
        loaded.config.model.model_id == "gru"
        and loaded.config.model.heads.actor_hidden_sizes == (32, 16)
    )
    with pytest.raises(ContractError, match="explicit game checkpoint"):
        asyncio.run(gf_cli.run_experiment(loaded, "collect"))
    assert not (tmp_path / "forbidden").exists()


BEFORE = {
    "base_revision": "0f87d785c0b98d106f97df5670ef338b8322842b",
    "default_config_sha256": "bdb2b3663397693baf9bc059e9831c17ea91cbc85a3184e30d3d2f1aa6f0fa33",
    "experiments": {
        "configs/experiments/baseline.toml": "71e189b3dca2cbfa16560655354950a4cf3250c40202430909467b5281ddb70b",
        "configs/experiments/card-instance-v2-contract.toml": "2705f95c8b470b8e4464f76569353d51c2adcdf006e90c380795c5de40979e3c",
        "configs/experiments/card-instance-v2-corrected-contract.toml": "898a059574c6bec0b5b52089c83e88e079898b8292191ee7218343fbe978bc9f",
        "configs/experiments/gru-ppo-cpu.toml": "9c7fee094e3ea58aabe9d03910845950e9cb93516aa43f96aa12644f4901a830",
        "configs/experiments/gru-ppo.toml": "3cae8d42b7f983e215634ad9214f2693dd00bfbce9d2273bd1d1ac5c17ab80a8",
        "configs/experiments/lightgbm-fixed-history.toml": "aaec03ce9e1d9905f85f49198089b1ec2a7cf968036365cbc68199fc3826fa2a",
        "configs/experiments/lightgbm-fixed-observation.toml": "467c973d1b8f4132409639fccd2d68b4e9ce51258ca953da6ef2bc5a9d52c5a6",
        "configs/experiments/lightgbm.toml": "397d5a92dada69d867a7b1841055948b996cbfc61da306c13bb8ab76a1becd8a",
        "configs/experiments/pilot.toml": "1fb48a5493f0d183e326c7184018cbcb0a51cccd1e718ba9a61bf75131f32385",
        "configs/experiments/ppo-cpu.toml": "07207648d5936705e1f0c7c8efd60ef45d0494c2ad7c69c60779a88457f676ef",
        "configs/experiments/ppo.toml": "73a4d5326378043cf0f7421fbe06674bd7f14c8503d6f9eae39be15697ca8456",
        "configs/experiments/smoke.toml": "5834ff038d225c98c2a0ac6ef1e219a52122e61c7fdd59b7adde13d3ec4b9eb9",
        "configs/experiments/transformer-ppo-cpu.toml": "0e32e428cd91bb3b6c04dd90f1e7f92fdef8245ae4c39c91bcd1d1fbaa73265f",
        "configs/experiments/transformer-ppo.toml": "60df8fdb3eba6c266a99b194b7f8a4a18e2e919b6cf9b56e0f9393bf0acefc34",
    },
}
