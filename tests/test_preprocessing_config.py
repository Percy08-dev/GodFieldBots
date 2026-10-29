from dataclasses import asdict

import pytest

from gf_cli.config import load_config
from gf_common.domain import ContractError
from gf_common.preprocessing import (
    PreparationConfig,
    load_preprocessing_config,
    preparation_config,
)


@pytest.mark.parametrize("values", [
    {"workers": 2}, {"max_in_flight": 2}, {"backend": "unknown"},
    {"backend": "rust", "workers": 3},
    {"backend": "rust", "native_json_depth": True},
    {"backend": "rust", "native_json_depth": 991},
    {"backend": "rust", "native_json_depth": 0},
    {"native_json_depth": 16},
    {"memory_budget_mib": True}, {"cache_budget_mib": -1},
    {"chunk_target_mib": 0}, {"chunk_target_mib": 8193},
    {"cache_budget_mib": 8193}, {"max_record_mib": 65},
    {"cleanup_seconds": 7200}, {"wall_seconds": float("inf")},
    {"wall_seconds": True}, {"extra": 1},
])
def test_invalid_preparation_limits(values):
    with pytest.raises(ContractError):
        preparation_config(values)


def test_standalone_and_experiment_preset(tmp_path):
    preset = tmp_path / "prepare.toml"
    preset.write_text('schema_version=1\nkind="preprocessing"\n'
                      '[preprocessing]\nmemory_budget_mib=2048\ncache_budget_mib=0\n')
    config = load_preprocessing_config(preset)
    assert config.memory_budget_mib == 2048
    assert config.cache_budget_mib == 0
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('schema_version=1\nkind="experiment"\n'
                          '[references]\npreprocessing="prepare.toml"\n'
                          '[preprocessing]\nchunk_target_mib=16\n'
                          '[server]\nfixed_seed_experiment=true\n')
    loaded = load_config(experiment, {"preprocessing": {"memory_budget_mib": 4096}})
    assert loaded.config.preprocessing.memory_budget_mib == 4096
    assert loaded.config.preprocessing.cache_budget_mib == 0
    assert loaded.config.preprocessing.chunk_target_mib == 16
    assert loaded.effective["preprocessing"] == asdict(loaded.config.preprocessing)
    assert len(loaded.inputs) == 2
    assert load_preprocessing_config(None) == PreparationConfig()


@pytest.mark.parametrize("text", [
    'schema_version=true\nkind="preprocessing"\n[preprocessing]\n',
    'schema_version=1\nkind="training"\n[preprocessing]\n',
    'schema_version=1\nkind="preprocessing"\n[server]\n[preprocessing]\n',
])
def test_standalone_rejects_unrelated_tables_and_versions(tmp_path, text):
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ContractError):
        load_preprocessing_config(path)


def test_referenced_preprocessing_preset_requires_table(tmp_path):
    (tmp_path / "prepare.toml").write_text('schema_version=1\nkind="preprocessing"\n')
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('schema_version=1\nkind="experiment"\n'
                          '[references]\npreprocessing="prepare.toml"\n')
    with pytest.raises(ContractError, match="settings table"):
        load_config(experiment)


def test_native_shared_budget_and_explicit_worker_counts():
    for workers in (1, 2, 4, 8):
        config = preparation_config({'backend': 'rust', 'workers': workers,
                                     'max_in_flight': workers, 'memory_budget_mib': 2048})
        assert config.memory_budget_mib == 2048
        assert config.workers == workers and config.max_in_flight == workers
    assert PreparationConfig().backend == 'python'  # Unspecified/reference settings remain Python.
    assert 'native_json_depth' not in asdict(PreparationConfig())


@pytest.mark.parametrize('backend', ['python', 'rust'])
def test_removed_native_depth_setting_is_unknown(backend):
    with pytest.raises(ContractError, match='invalid fields'):
        preparation_config({'backend': backend, 'native_json_depth': 990})


@pytest.mark.parametrize('name', ['ppo', 'ppo-cpu', 'lightgbm-fixed-observation', 'lightgbm-fixed-history'])
def test_rollout_presets_resolve_native_defaults_and_explicit_overrides(name):
    path = f'configs/experiments/{name}.toml'
    native = load_config(path).config.preprocessing
    assert (native.backend, native.memory_budget_mib, native.chunk_target_mib,
            native.workers, native.max_in_flight) == ('rust', 8192, 64, 1, 1)

    reference = load_config(path, {'preprocessing': {'backend': 'python'}}).config.preprocessing
    assert reference.backend == 'python'
    assert (reference.memory_budget_mib, reference.chunk_target_mib, reference.workers) == (8192, 64, 1)

    low_memory = load_preprocessing_config('configs/preprocessing/native-low-memory.toml')
    assert (low_memory.backend, low_memory.memory_budget_mib, low_memory.chunk_target_mib,
            low_memory.workers, low_memory.max_in_flight) == ('rust', 2048, 16, 1, 1)
    resolved = load_config(path, {'preprocessing': asdict(low_memory)}).config.preprocessing
    assert resolved == low_memory
