"""Explicit storage never changes old hashes or silently enters later consumers."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.config import load_config
from gf_common.domain import ContractError

ROOT = Path(__file__).parents[1]
BASE = ROOT / "configs/experiments/card-instance-v2-corrected-contract.toml"


def packed_config(tmp_path, storage=""):
    text = BASE.read_text()
    # Preserve the source config's path origins while placing the explicit opt-in locally.
    for part in ("../actions/", "../models/"):
        text = text.replace('"' + part, '"' + str(BASE.parent / part) + "/")
    path = tmp_path / "packed.toml"
    path.write_text(text + "\n[storage]\n" + storage)
    return path


def test_explicit_storage_subtype_defaults_and_old_serialization(tmp_path):
    from gf_cli import config
    old = load_config(BASE)
    assert type(old.config) is config.ExperimentConfig
    assert "storage" not in asdict(old.config)
    loaded = load_config(packed_config(tmp_path, 'db_root = "db"\nram_root = "ram"\n'))
    assert type(loaded.config) is config.GroupPackExperimentConfig
    assert loaded.config.storage.format == "sqlite-group-pack-v1"
    assert loaded.config.storage.db_root == str(tmp_path / "db")
    assert loaded.config.storage.ram_root == str(tmp_path / "ram")
    assert loaded.config.storage.budget_mib == 2048
    assert loaded.sha256 != old.sha256
    assert load_config(BASE).sha256 == old.sha256
    with pytest.raises(ContractError, match="explicit"):
        load_config(BASE, {"storage": {}})


@pytest.mark.parametrize("storage", ['format="other"', 'budget_mib=true', 'queue_mib=0',
    'budget_mib=128', 'queue_mib=2049', 'unknown=1', 'max_pack_mib=0'])
def test_invalid_storage_rejected_without_output(tmp_path, storage):
    with pytest.raises((ContractError, ValueError)):
        load_config(packed_config(tmp_path, storage))
    assert not (tmp_path / "db").exists()


@pytest.mark.parametrize("command", ["benchmark", "train", "train-formal", "validate-policy",
    "validate-formal-policy", "game-dataset", "fit-game-tree"])
def test_storage_command_guard_precedes_runtime_allocation(tmp_path, command):
    from gf_cli.config import require_storage_command
    loaded = load_config(packed_config(tmp_path))
    with pytest.raises(ContractError, match="storage"):
        require_storage_command(loaded.config, command)


def test_current_v1_pack_bound_rejects_config_and_direct_settings_above256(tmp_path):
    from gf_common.group_pack import MIB, StorageSettings
    with pytest.raises(ValueError, match="256"):
        StorageSettings(max_pack_bytes=257 * MIB)
    with pytest.raises(ValueError, match="256"):
        load_config(packed_config(tmp_path, 'max_pack_mib=257'))


@pytest.mark.parametrize("entry", ["train", "train-formal", "formal-preflight", "final-caller", "final-backend"])
def test_invalid_formal_source_commands_refuse_before_runtime_or_source_work(tmp_path, monkeypatch, entry):
    import asyncio
    from types import SimpleNamespace

    from gf_training import card_instance_final_validation as final
    from gf_training import card_instance_formal as formal
    from gf_training import card_instance_ppo as ppo

    loaded = load_config(packed_config(tmp_path))
    def later(*args, **kwargs):
        pytest.fail("invalid full formal Config reached Source or collection work")
    monkeypatch.setattr(final, "_admit_source", later)
    output = tmp_path / "never-created"
    with pytest.raises(ContractError):
        if entry == "train":
            asyncio.run(ppo.train_card_instance_ppo(loaded, later, output))
        elif entry == "train-formal":
            asyncio.run(ppo.train_formal_card_instance_ppo(loaded, later, output))
        elif entry == "formal-preflight":
            formal.formal_preflight(loaded)
        elif entry == "final-caller":
            final._caller_contract(SimpleNamespace(config=loaded.config))
        else:
            # This fixture retains its old scalar reader. Final64 requires the
            # explicit participant reader even after storage support is added.
            final.require_actor_backend(loaded)
    assert not output.exists()
