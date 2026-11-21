"""Shared serving adoption and opt-in metadata preserve historical contracts."""
from dataclasses import asdict

import pytest

from gf_common import card_instance
from gf_common.domain import ContractError, digest


def test_shared_profile_has_its_own_exact_normative_inventory():
    contract = card_instance.shared_card_instance_contract()
    assert contract.simulator_revision == '072dabbca1458c9526897781d2fd64edeeb4f9ca'
    assert contract != card_instance.corrected_card_instance_contract()
    specs = card_instance.reviewed_specs(contract)
    assert digest(specs) == contract.spec_inventory_sha256
    assert specs['docs/spec/07-external-interfaces-adapters.md'] == '8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2'
    assert card_instance.parse_card_instance_contract(asdict(contract)) == contract
    changed = asdict(contract) | {'spec_inventory_sha256': card_instance.CORRECTED_SPEC_INVENTORY_HASH}
    with pytest.raises(ContractError):
        card_instance.parse_card_instance_contract(changed)


def test_shared_geometry_resolves_from_games_and_stays_outside_legacy_runtime():
    from gf_common.hpo_resident import resident_runtime
    from gf_common.shared_inference import shared_runtime
    runtime = shared_runtime({'mode': 'shared-mlp-v1'}, game_concurrency=4)
    assert runtime == {'mode': 'shared-mlp-v1', 'max_batch_size': 4, 'formation_wait_ms': 1.0,
        'cache_entries': 6, 'cache_weight_bytes': 134217728}
    assert shared_runtime(runtime | {'max_batch_size': 2}, game_concurrency=4)['max_batch_size'] == 2
    with pytest.raises(ContractError):
        resident_runtime(runtime)
    for bad in (True, 0, -1):
        with pytest.raises(ContractError):
            shared_runtime(runtime | {'max_batch_size': bad}, game_concurrency=4)


def test_shared_runtime_receipt_pins_source_dirty_status_and_new_inventory():
    from pathlib import Path

    from gf_common.card_instance_runtime import validate_runtime_receipt
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    source = Path('/tmp/gf-issue72-shared-inference')
    binary = Path('/tmp/gfb-shared-simulator-072dabb/target/release/gf-test-server')
    if not source.exists() or not binary.exists():
        pytest.skip('reviewed Simulator source or binary unavailable')
    owner = CardInstanceRuntimeEvidence(binary, source,
        contract=card_instance.shared_card_instance_contract())
    receipt = owner.preflight()
    assert receipt['schema_version'] == 2
    assert receipt['source']['dirty'] is False
    assert receipt['source']['spec_files'] == card_instance.reviewed_specs(card_instance.shared_card_instance_contract())
    assert validate_runtime_receipt(receipt) == receipt
    receipt['source']['dirty'] = True
    with pytest.raises(ContractError):
        validate_runtime_receipt(receipt)


def test_normal_and_hpo_presets_select_explicit_shared_mode_without_changing_legacy():
    from pathlib import Path

    from gf_cli.config import RunConfigV2, load_config
    from gf_cli.hpo_config import load_hpo_config, materialize_hpo_run
    from gf_common.shared_inference import serving_config
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_training.hpo_config import effective_settings
    root = Path(__file__).parents[1]
    old = load_config(root/'configs/experiments/hpo-resident-mlp-base.toml')
    assert type(old.config.run) is RunConfigV2
    assert 'inference_service' not in asdict(old.config.run)
    ordinary = load_config(root/'configs/experiments/mlp-shared-inference-smoke.toml')
    assert serving_config(ordinary.config)['max_batch_size'] == 2
    campaign = load_hpo_config(root/'configs/hpo/mlp-shared-inference.toml')
    settings = effective_settings(campaign.config.model,campaign.config.training,{},layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    run = materialize_hpo_run(campaign,settings,learning_seed=17,run_id='00000000-0000-0000-0000-000000000017')
    assert run.config.hpo_spec['schema_version'] == 5
    assert serving_config(run.config)['max_batch_size'] == 3
    from gf_training.hpo_lineage import hpo_spec
    assert hpo_spec({'effective_config':asdict(run.config),'run_variant':'hpo-ppo-owned-shared-v1',
        'hpo_spec_sha256':digest(run.config.hpo_spec)}) == run.config.hpo_spec
    assert old.sha256 == load_config(root/'configs/experiments/hpo-resident-mlp-base.toml').sha256


def test_ordinary_shared_preflight_refuses_joint_cuda_geometry_before_process_start(monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace

    from gf_cli.config import load_config
    from gf_simulator_client import card_instance_runtime
    from gf_training.card_instance_ppo import _GameOperations
    loaded=load_config(Path(__file__).parents[1]/'configs/experiments/mlp-shared-inference-smoke.toml')
    config=loaded.config
    monkeypatch.setattr(card_instance_runtime,'CardInstanceRuntimeEvidence',lambda *args,**kwargs:SimpleNamespace(preflight=lambda:None))
    config=replace(config,run=replace(config.run,global_cuda_budget_bytes=1))
    with pytest.raises(ContractError,match='shared learner and service'):
        _GameOperations().preflight(config,deadline=100.)


def test_shared_coordinator_imports_and_preset_loading_do_not_import_optional_runtimes():
    import subprocess
    import sys
    from pathlib import Path
    script="""
import sys
import gf_cli
import gf_cli.hpo
import gf_simulator_client.shared_service
from gf_cli.config import load_config
from gf_cli.hpo_config import load_hpo_config
load_config('configs/experiments/mlp-shared-inference-smoke.toml')
load_hpo_config('configs/hpo/mlp-shared-inference.toml')
assert not ({'torch','torchrl','tensordict','optuna','lightgbm','pyarrow'} & sys.modules.keys())
"""
    subprocess.run([sys.executable,'-E','-c',script],cwd=Path(__file__).parents[1],check=True)


@pytest.mark.parametrize('command',['collect','evaluate','smoke','benchmark','arbitrary-real-operation'])
@pytest.mark.parametrize('run_fields',[{}, {'run':None}])
def test_shared_storage_discovery_preserves_partial_config_early_contract_guards(command,run_fields):
    from types import SimpleNamespace

    from gf_cli.config import require_storage_command
    require_storage_command(SimpleNamespace(**run_fields),command)


def test_shared_storage_discovery_still_refuses_standalone_serving_commands():
    from pathlib import Path

    from gf_cli.config import load_config, require_storage_command
    loaded=load_config(Path(__file__).parents[1]/'configs/experiments/mlp-shared-inference-smoke.toml')
    for command in ('collect','evaluate','smoke','benchmark'):
        with pytest.raises(ContractError,match='shared serving'):
            require_storage_command(loaded.config,command)
    for command in ('train','validate'):require_storage_command(loaded.config,command)
    require_storage_command(loaded.config,'collect',standalone=False)
