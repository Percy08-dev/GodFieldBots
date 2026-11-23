"""HPO destination configuration, historical hashes, and pre-write admission."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli import hpo, hpo_config, hpo_storage
from gf_cli.hpo_campaign import canonical_control
from gf_common import hpo_storage as storage_facts
from gf_common.domain import ContractError, canonical, digest
from tests.test_hpo_storage import admitted_storage
from tests.test_rust_player_config import BASE, ROOT, RUN_ID

PROBE = ROOT / 'configs/hpo/mlp-256-update-budget-probe.toml'
TEMPLATE = PROBE.with_name('mlp-256-update-budget-probe-template.toml')
PROBE_ROOT = Path('/mnt/gf_data/ppo-hpo/mlp-256-probe')


def _write_legacy_campaign(path: Path, *, parents: Path) -> Path:
    path.write_text(f'''\
schema_version = 6
kind = "ppo-hpo"
origin = "producer-owned-v1"
encoder = "canonical-python"
base_experiment = {json.dumps(str(BASE))}
model_id = "mlp"
stage = "legacy-storage-test"
trial_count = 1
decision_budget = 64
learning_seeds = [17]
evaluation_game_seeds = ["00000000-0000-0000-0000-000000000001"]
sampler_seed = 17
startup_trials = 1
session_seconds = 1200
run_wall_seconds = 600

[runtime]
mode = "rust-player-runtime-v1"
scheduler = "tokio"
game_concurrency = 2
player_worker_threads = 7
max_batch_size = 3
formation_wait_ms = 0

[operating]
game_concurrency = 2
post_host_seconds = 30.0

[space]
coupled = []
parameters = []

[storage]
parent_root = {json.dumps(str(parents))}
numeric_root = {json.dumps(str(parents))}
optuna_path = "study.sqlite3"
''')
    return path


@pytest.mark.parametrize('config_path', [PROBE, TEMPLATE], ids=['probe', 'template'])
def test_probe_materializes_tmpfs_and_dedicated_hdd_storage(config_path):
    loaded = hpo_config.load_hpo_config(config_path)
    storage = loaded.config.storage
    parent = str(PROBE_ROOT / 'parents')
    originals = str(PROBE_ROOT / 'originals')
    assert storage.parent_root == storage.numeric_root == parent
    assert storage.originals_root == originals
    assert storage.optuna_path == str(PROBE_ROOT / 'study.sqlite3')
    assert hpo.campaign_directory(loaded, PROBE_ROOT / 'parents' / 'campaign') == PROBE_ROOT / 'parents' / 'campaign'
    assert asdict(storage.hpo_diagnostics) == {
        'http_log_ram_root': '/dev/shm/godfieldbots/http-logs',
        'http_log_archive_root': str(PROBE_ROOT / 'http-logs'),
        'http_log_retention': 'errors',
        'http_log_compression': 'zstd',
        'update_compression': 'zstd',
    }
    assert loaded.base.effective['run']['output_dir'] == parent
    assert loaded.base.effective['storage']['db_root'] == originals
    run = hpo_config.materialize_hpo_run(loaded, canonical_control(loaded.config), learning_seed=17, run_id=RUN_ID)
    assert run.effective['run']['output_dir'] == parent
    assert run.effective['storage']['db_root'] == originals
    assert run.effective['storage']['collection_work_root'] == '/dev/shm/godfieldbots/collections'
    assert run.effective['storage']['ram_root'] == '/dev/shm/godfieldbots/pack-work'


def test_template_copy_validates_without_storage_writes(tmp_path, capsys):
    config_dir = tmp_path / 'configs'
    config_path = config_dir / 'hpo' / 'experiment.toml'
    config_path.parent.mkdir(parents=True)
    (config_dir / 'experiments').symlink_to(ROOT / 'configs/experiments', target_is_directory=True)
    config_path.write_bytes(TEMPLATE.read_bytes())
    loaded = hpo_config.load_hpo_config(config_path)
    parent = PROBE_ROOT / 'parents'
    assert loaded.config.storage.parent_root == loaded.config.storage.numeric_root == str(parent)
    assert hpo.campaign_directory(loaded, None) == parent / 'campaign'
    assert hpo.campaign_directory(loaded, parent / 'campaign') == parent / 'campaign'
    with pytest.raises(ContractError, match='campaign_dir'):
        hpo.campaign_directory(loaded, parent / 'different')
    assert hpo.main(['validate', '--config', str(config_path)]) == 0
    assert str(parent / 'campaign') in capsys.readouterr().out


def test_legacy_toml_resolves_defaults_or_exact_old_hash(tmp_path):
    from gf_cli.rust_player_campaign import campaign_snapshot, load_campaign_snapshot

    config_path = _write_legacy_campaign(tmp_path / 'historical.toml', parents=tmp_path / 'parents')
    current = hpo_config.load_hpo_config(config_path)
    old_effective = dict(current.effective)
    old_effective['storage'] = dict(old_effective['storage'])
    diagnostics = old_effective['storage'].pop('hpo_diagnostics')
    assert diagnostics == {
        'http_log_ram_root': '/dev/shm/godfieldbots/http-logs',
        'http_log_archive_root': str(Path(current.config.storage.parent_root) / 'http-logs'),
        'http_log_retention': 'errors',
        'http_log_compression': 'zstd',
        'update_compression': 'zstd',
    }
    old_sha = digest({'campaign': old_effective, 'base': current.base.sha256})
    old = hpo_config.load_hpo_config(config_path, expected_sha256=old_sha)
    assert old.effective == old_effective
    assert canonical(asdict(old.config)) == canonical(old_effective)
    assert old.sha256 == old_sha and current.sha256 != old.sha256
    assert old.config.study_identity == current.config.study_identity
    assert load_campaign_snapshot(campaign_snapshot(old)).sha256 == old_sha
    with pytest.raises(ContractError, match='configuration'):
        hpo_config.load_hpo_config(config_path, expected_sha256='0' * 64)
    directory = Path(old.config.storage.parent_root) / 'cli-campaign'
    assert hpo.campaign_directory(old, directory) == directory
    with pytest.raises(ContractError, match='campaign-dir'):
        hpo.campaign_directory(old, None)


def test_read_only_and_mount_fallback_fail_before_creation(tmp_path, monkeypatch):
    limits, controls, _, originals = admitted_storage(tmp_path, monkeypatch)
    mounts = hpo_storage._mounts()
    campaign = controls / 'campaign'
    monkeypatch.setattr(hpo_storage, '_mounts', lambda: (replace(mounts[0], options=('ro', 'rw')), *mounts[1:]))
    with pytest.raises(ContractError, match='read-only'):
        hpo_storage.check_storage(limits, campaign_dir=campaign, originals_root=originals, deadline=float('inf'))
    assert not campaign.exists()
    fake = tmp_path / 'dedicated'
    fake.mkdir()
    monkeypatch.setattr(storage_facts, 'GF_DATA_ROOT', fake)
    monkeypatch.setattr(hpo_storage, '_mounts', lambda: mounts)
    storage = replace(limits, parent_root=str(fake), numeric_root=str(fake), optuna_path=str(fake / 'study.db'))
    with pytest.raises(ContractError, match='dedicated'):
        hpo_storage.check_storage(storage, campaign_dir=fake / 'campaign', originals_root=originals, deadline=float('inf'))
    assert not (fake / 'campaign').exists()


def test_http_destinations_use_tmpfs_and_admit_without_scanning_archive(tmp_path, monkeypatch):
    from gf_common.hpo_storage import HPODiagnosticsConfig

    limits, controls, _, originals = admitted_storage(tmp_path, monkeypatch)
    ram = tmp_path / 'tmpfs'
    ram.mkdir()
    archive = controls / 'http-logs'
    archive.mkdir()
    (archive / 'retained.jsonl').write_bytes(b'0' * 2048)
    diagnostics = HPODiagnosticsConfig(http_log_ram_root=str(ram / 'pairs'), http_log_archive_root=str(archive))
    extended = hpo_config.HPODiagnosticStorageConfig(**asdict(limits), hpo_diagnostics=diagnostics)
    mounts = hpo_storage._mounts()
    tmpfs = storage_facts.Mount(28, '0:3', '/', str(ram), 'tmpfs', 'tmpfs', ('rw',))
    campaign = controls / 'campaign'
    with pytest.raises(ContractError, match='tmpfs'):
        hpo_storage.check_storage(extended, campaign_dir=campaign, originals_root=originals, deadline=float('inf'))
    monkeypatch.setattr(hpo_storage, '_mounts', lambda: (*mounts, tmpfs))
    receipt = hpo_storage.check_storage(extended, campaign_dir=campaign, originals_root=originals, deadline=float('inf'))
    assert receipt['http_diagnostics_destinations'][str(ram / 'pairs')]['mount']['fs_type'] == 'tmpfs'
    assert receipt['destinations'][0]['used_bytes'] == 0
    assert not campaign.exists()


def test_cli_run_resolves_toml_and_checks_writability_before_opening(tmp_path, capsys, monkeypatch):
    from gf_cli import rust_player_campaign
    from tests.test_rust_player_config import campaign_raw

    raw = campaign_raw()
    limits, controls, _, originals = admitted_storage(tmp_path, monkeypatch)
    campaign = controls / 'declared'
    raw['storage'] = asdict(limits) | {'campaign_dir': str(campaign)}
    raw['operating']['post_host_seconds'] = 30.
    loaded = hpo_config.campaign_config(raw, origin=ROOT)
    mounts = hpo_storage._mounts()
    monkeypatch.setattr(hpo_storage, '_mounts', lambda: (replace(mounts[0], options=('ro',)), *mounts[1:]))
    monkeypatch.setattr(hpo, 'load_hpo_config', lambda *args, **kwargs: loaded)
    selected = []
    exec_campaign = rust_player_campaign.exec_campaign

    def observe(*args, **kwargs):
        selected.append(args[1])
        exec_campaign(*args, **kwargs)

    monkeypatch.setattr(rust_player_campaign, 'exec_campaign', observe)
    argv = ['run', '--config', 'campaign.toml', '--mode', 'fresh']
    assert hpo.main([*argv, '--campaign-dir', str(controls / 'other')]) == 2
    assert 'same path' in capsys.readouterr().out
    assert not selected
    assert hpo.main(argv) == 2
    assert 'read-only' in capsys.readouterr().out
    assert selected == [campaign]
    assert not campaign.exists() and not any(originals.iterdir())


def test_cli_resume_accepts_an_exact_historical_config_with_old_path_alias(tmp_path, monkeypatch):
    from gf_cli import rust_player_campaign

    controls = tmp_path / 'controls'
    controls.mkdir()
    alias = tmp_path / 'old-alias'
    alias.symlink_to(controls, target_is_directory=True)
    source = _write_legacy_campaign(tmp_path / 'campaign.toml', parents=alias / 'parents')
    old = hpo_config.load_hpo_config(source, _new_storage_defaults=False)
    directory = controls / 'parents' / 'campaign'
    with hpo_storage.CampaignStore(directory, mode='fresh', max_owned_bytes=old.config.operating.parent_budget_bytes) as store:
        store.put('campaign.json', {'kind': 'hpo-campaign', 'loaded_config_sha256': old.sha256})
    with pytest.raises(ContractError, match='symlink'):
        hpo_config.load_hpo_config(source)

    received = []

    def resume(loaded, path, **kwargs):
        received.append((loaded.sha256, path))
        raise ContractError('stop before live work')

    monkeypatch.setattr(rust_player_campaign, 'exec_campaign', resume)
    original_cli_path = alias / 'parents' / 'campaign'
    assert hpo.main(['run', '--config', str(source), '--mode', 'resume', '--campaign-dir', str(original_cli_path)]) == 2
    assert received == [(old.sha256, directory)]


def test_evaluation_admits_actual_output_mount(tmp_path, monkeypatch):
    _, controls, _, _ = admitted_storage(tmp_path, monkeypatch)
    evaluations = controls / 'campaign' / 'evaluations'
    evaluations.mkdir(parents=True)
    mounts = hpo_storage._mounts()
    readonly = storage_facts.Mount(34, '8:8', '/', str(evaluations), 'ext4', '/dev/sdb', ('ro',))
    monkeypatch.setattr(hpo_storage, '_mounts', lambda: (*mounts, readonly))
    output = evaluations / 'panel'
    with pytest.raises(ContractError, match='read-only'):
        hpo_storage.check_evaluation_destination(output)
    assert not output.exists()


def test_original_writer_rechecks_nested_mount_and_existing_catalog_file(tmp_path, monkeypatch):
    _, controls, _, originals = admitted_storage(tmp_path, monkeypatch)
    collections = originals / 'parent' / 'collections'
    collections.mkdir(parents=True)
    mounts = hpo_storage._mounts()
    foreign = storage_facts.Mount(34, '8:8', '/', str(collections), 'ext4', '/dev/sdb', ('rw',))
    monkeypatch.setattr(storage_facts, 'WINDOWS_D_ROOT', originals)
    monkeypatch.setattr(storage_facts, 'read_mounts', lambda: (*mounts, foreign))
    with pytest.raises(ContractError, match='actual D mount'):
        storage_facts.check_collection_destinations(controls / 'collect', collections / 'batch',
                                                   catalog_root=originals)
    assert not (collections / 'batch').exists()

    catalog = originals / 'catalog.sqlite3'
    catalog.write_bytes(b'')
    readonly = replace(mounts[-1], mount_id=35, mount_point=str(catalog), options=('ro',))
    monkeypatch.setattr(storage_facts, 'read_mounts', lambda: (*mounts, readonly))
    with pytest.raises(ContractError, match='read-only'):
        storage_facts.check_collection_destinations(controls / 'collect', collections / 'batch',
                                                   catalog_root=originals)
