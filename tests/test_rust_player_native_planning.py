"""Native6 planning and actual campaign CONFIG metadata, never training/admission."""
from dataclasses import asdict

import pytest

from gf_cli.hpo_config import campaign_config
from gf_common.domain import ContractError, canonical
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
from gf_training.hpo_config import effective_settings
from tests.test_rust_player_config import ROOT, campaign_raw


def loaded_campaign(tmp_path):
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    raw['space'] = {'parameters': [], 'coupled': []}
    raw['storage'].update(parent_root=str(tmp_path), numeric_root=str(tmp_path), optuna_path=str(tmp_path / 'study.sqlite3'))
    return campaign_config(raw, origin=ROOT)


def test_native_campaign_preview_binds_actual_campaign_and_selected_settings_without_torch(tmp_path):
    import subprocess
    import sys
    code = '''
import sys
from pathlib import Path
from gf_cli.rust_player_entry import campaign_startup_preview
from tests.test_rust_player_native_planning import loaded_campaign
from gf_cli.hpo_campaign import canonical_control
c=loaded_campaign(Path(sys.argv[1]))
for settings in [None,canonical_control(c.config)]:
    row=campaign_startup_preview(c,settings=settings)
    assert row['config_sha256']==c.sha256
    assert row['host_admission'] is False and row['future_phase_checks_required']
    startup=row['startup']
    assert startup['startup_cuda_bytes']==startup['host_fixed_bytes']+2*(startup['parameter_bytes']+startup['position_bytes'])
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)


def test_all_required_native6_callers_use_startup_data_without_legacy_admission(tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    from gf_cli import hpo, hpo_campaign, hpo_config
    from gf_cli.hpo_storage import CampaignStore
    loaded = loaded_campaign(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('native6 called legacy joint admission')
    monkeypatch.setattr(hpo, 'joint_resource_admission', forbidden)
    monkeypatch.setattr(hpo_campaign, 'joint_resource_admission', forbidden)
    monkeypatch.setattr(hpo, 'check_storage', lambda *args, **kwargs: {'scope': 'isolated CONFIG storage I/O'})
    materialized = []
    actual_materialize = hpo_config.materialize_hpo_run
    def materialize(*args, **kwargs):
        value = actual_materialize(*args, **kwargs)
        materialized.append((args[0].sha256, args[1].identity))
        return value
    monkeypatch.setattr(hpo_config, 'materialize_hpo_run', materialize)
    validation = hpo.validate(loaded)
    for key in ('resource_admission', 'canonical_control_admission'):
        assert validation[key]['config_sha256'] == loaded.sha256
        assert validation[key]['host_admission'] is False
    selected = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    seed = tmp_path / 'settings-seed.json'
    seed.write_text(canonical({'schema_version': 1, 'kind': 'hpo-settings-seed',
        'model': {key: value for key, value in asdict(selected.model).items() if key != 'resources'},
        'training': asdict(selected.training), 'effective_settings_sha256': selected.identity,
        'origin': {'campaign_id': 'prior', 'study_identity': 'different-prior-study', 'proposal_sha256': 'a' * 64}}) + '\n')
    imported = hpo_campaign.import_settings((seed,), loaded.config, study_identity='fresh-study', _loaded=loaded)
    assert imported[0].identity == selected.identity
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        manifest = hpo_campaign.initialize_campaign(store, loaded, {'scope': 'CONFIG-only no qualification'})
        study = hpo_campaign.CampaignStudy(store, loaded, manifest)
        proposal = study.propose(0, slot=None, kind='outside-control', requested=hpo_campaign.canonical_control(loaded.config))
        assert proposal is not None
        row, _sha = store.get('proposals/000000/proposal.json')
        assert row['resource_admission']['config_sha256'] == loaded.sha256
        assert row['resource_admission']['effective_settings_sha256'] == proposal.settings.identity
        assert row['resource_admission']['host_admission'] is False
    assert len(materialized) == 6  # validate2/import1/initialize2/selected proposal1, all REAL materializer.
    assert all(config_sha == loaded.sha256 for config_sha, _settings in materialized)


def test_public_native_hpo_run_stays_closed_without_a_campaign_handler_stub(tmp_path, monkeypatch, capsys):
    from gf_cli import hpo, rust_player_campaign
    loaded = loaded_campaign(tmp_path)
    monkeypatch.setattr(hpo, 'load_hpo_config', lambda path: loaded)
    def forbidden(*args, **kwargs):
        raise AssertionError('native6 public run reached bootstrap before application is integrated')
    monkeypatch.setattr(rust_player_campaign, 'prepare_campaign', forbidden)
    assert not hasattr(rust_player_campaign, 'run_campaign')
    assert hpo.main(['run', '--config', str(tmp_path / 'unused.toml'), '--mode', 'fresh',
        '--campaign-dir', str(tmp_path / 'campaign')]) == 2
    assert not (tmp_path / 'campaign').exists()
    import json
    assert json.loads(capsys.readouterr().out)['kind'] == 'hpo-command-failure'


@pytest.mark.parametrize('post', [60., 59.])
@pytest.mark.parametrize('path', ['trial-and-control-validation', 'selected-control'])
def test_r1_actual_trial_and_control_materialize_before_external_storage(tmp_path, monkeypatch, post, path):
    from gf_cli import hpo, hpo_config
    from gf_cli.hpo_campaign import canonical_control
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = post
    loaded = campaign_config(raw, origin=ROOT)
    assert loaded.base.config.run.finalization_seconds == 60.
    materialized, storage, attempted = [], [], []
    actual = hpo_config.materialize_hpo_run
    def materialize(*args, **kwargs):
        attempted.append(args[1].identity)
        result = actual(*args, **kwargs)
        materialized.append(result.config.hpo_spec['effective_settings_sha256'])
        return result
    def external(*args, **kwargs):
        storage.append(True)
        if post == 60.: raise AssertionError('invalid P reached external storage preflight')
        return {'scope': 'external storage I/O isolated'}
    monkeypatch.setattr(hpo_config, 'materialize_hpo_run', materialize)
    monkeypatch.setattr(hpo, 'check_storage', external)
    if path == 'selected-control':
        from gf_cli.rust_player_entry import campaign_startup_preview
        control = canonical_control(loaded.config)
        if post == 60.:
            with pytest.raises(ContractError): campaign_startup_preview(loaded, settings=control)
        else:
            row = campaign_startup_preview(loaded, settings=control)
            assert row['effective_settings_sha256'] == control.identity
        assert attempted == [control.identity] and not storage
        return
    if post == 60.:
        with pytest.raises(ContractError):
            hpo.validate(loaded)
        assert not storage
    else:
        result = hpo.validate(loaded)
        expected = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
        assert materialized == [expected.identity, canonical_control(loaded.config).identity]
        assert result['resource_admission']['config_sha256'] == loaded.sha256
        assert storage == [True]


def test_r2_in_space_control_preview_uses_actual_published_resources_not_scientific_identity(tmp_path):
    pytest.importorskip('optuna')
    from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign
    from gf_cli.hpo_config import materialize_hpo_run
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_entry import campaign_startup_preview
    from gf_models.hpo_config import model_defaults_v4
    from gf_models.rust_player_device import CudaCheckpointProfile
    from gf_training.hpo_config import HPOTrainingConfig
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    raw['model'], raw['training'] = asdict(model_defaults_v4('mlp')), asdict(HPOTrainingConfig())
    raw['model']['resources']['cuda_context_bytes'] = 1024**3
    raw['space'] = {'parameters': [], 'coupled': []}
    raw['storage'].update(parent_root=str(tmp_path), numeric_root=str(tmp_path), optuna_path=str(tmp_path / 'study.sqlite3'))
    loaded = campaign_config(raw, origin=ROOT)
    requested = canonical_control(loaded.config)
    configured = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    assert requested.identity == configured.identity
    assert requested.model.resources.cuda_context_bytes == 512 * 1024**2
    assert configured.model.resources.cuda_context_bytes == 1024**3
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        manifest = initialize_campaign(store, loaded, {'scope': 'CONFIG-only, no Source/fitness'})
        proposal = CampaignStudy(store, loaded, manifest).propose(0, slot=0, kind='control', requested=requested)
        assert proposal is not None and proposal.settings.identity == requested.identity
        actual = materialize_hpo_run(loaded, proposal.settings, learning_seed=loaded.config.learning_seeds[0],
            run_id='00000000-0000-0000-0000-000000000000').config
        profile = CudaCheckpointProfile(actual.model.architecture(layout=layout), actual.model.inference, layout=layout)
        persisted, _sha = store.get('proposals/000000/proposal.json')
        preview = persisted['resource_admission']
        assert persisted['effective_model']['resources']['cuda_context_bytes'] == 1024**3
        assert preview['startup']['host_fixed_bytes'] == profile.fixed_bytes
        assert preview == campaign_startup_preview(loaded, settings=proposal.settings)
