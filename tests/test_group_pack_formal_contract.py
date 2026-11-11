"""Storage-only formal values, without original-parent or execution authority."""
import copy
import hashlib
from pathlib import Path

import pytest

from gf_common.card_instance_formal import (
    ALGORITHM_SHA256,
    PACKED_DECLARATION_FIELDS,
    PACKED_RECIPE_ID,
    PACKED_RECIPE_SHA256,
    PACKED_RUN_VARIANT,
    RECIPE_SHA256,
    checked_packed_config,
    declaration_reference,
    packed_recipe_bytes,
    stop_receipt,
    validate_declaration,
    validate_formal_state,
)
from gf_common.domain import ContractError, canonical, digest
from gf_common.group_pack_game import storage_binding
from gf_training.card_instance_formal import load_packed_recipe, load_recipe, verify_kernel_ast
from tests.test_formal_ppo_contract import config_for, state_for

ROOT = Path(__file__).resolve().parents[1]


def packed_config(**kwargs):
    value = config_for(**kwargs)
    value['storage'] = {**load_packed_recipe()['storage_defaults'], 'db_root': '/unused/db', 'ram_root': '/unused/ram'}
    return value


def packed_state():
    state = state_for()
    state.update(schema_version=4, run_variant=PACKED_RUN_VARIANT, storage_binding=storage_binding())
    state['effective_config'] = packed_config()
    state['formal_declaration'].update(recipe_id=PACKED_RECIPE_ID, recipe_sha256=PACKED_RECIPE_SHA256)
    return state


def test_exact_asset_pins_base_algorithm_canonical_bytes_and_all_eight_kernels():
    raw = (ROOT / 'src/gf_common/contracts/card-instance-formal-ppo-group-pack-v1.json').read_bytes()
    assert raw == (ROOT / 'docs/contracts/card-instance-formal-ppo-group-pack-v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PACKED_RECIPE_SHA256
    recipe = packed_recipe_bytes(raw)
    assert raw == (canonical(recipe) + '\n').encode()
    assert recipe['base_recipe_sha256'] == RECIPE_SHA256
    assert recipe['algorithm_recipe_sha256'] == ALGORITHM_SHA256
    assert len(load_recipe()['algorithm_recipe']['kernel_sha256']) == 8
    verify_kernel_ast(load_recipe())
    with pytest.raises(ContractError):
        packed_recipe_bytes(raw + b'\n')


@pytest.mark.parametrize('model', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('backend', ['python', 'rust'])
@pytest.mark.parametrize('seed', [17, 29, 43])
def test_storage_projection_preserves_whole_old_literal(model, device, backend, seed):
    config = packed_config(model=model, device=device, backend=backend, seed=seed)
    logical = checked_packed_config(config, load_recipe())
    assert logical == config_for(model, device, backend, seed)
    assert (logical['training']['max_updates'], logical['training']['max_owned_bytes'],
        logical['training']['admission_reservation_bytes']) == (1000, 10 * 1024**3, 1024**3)
    assert tuple(logical['run'][key] for key in ('wall_seconds', 'admission_seconds', 'execution_seconds', 'finalization_seconds')) == (7200, 5400, 7140, 60)
    assert config['storage'] == packed_config()['storage']


@pytest.mark.parametrize('key,value', [('format', 'directory'), ('budget_mib', 2049), ('queue_mib', True),
    ('max_pack_mib', 1), ('db_root', ''), ('ram_root', 'relative'), ('other', 1)])
def test_projection_validates_storage_before_removing_only_exact_table(key, value):
    config = packed_config()
    config['storage'][key] = value
    with pytest.raises(ContractError):
        checked_packed_config(config, load_recipe())


@pytest.mark.parametrize('table,key,value', [('training', 'max_updates', 2), ('run', 'wall_seconds', 100),
    ('run', 'concurrency', 1), ('actions', 'max_candidates', 64)])
def test_projection_cannot_hide_algorithm_or_budget_change(table, key, value):
    config = packed_config()
    config[table][key] = value
    with pytest.raises(ContractError):
        checked_packed_config(config, load_recipe())


def test_packed_reference_and_stop_are_explicit_schema2_with_unchanged_arithmetic():
    state = packed_state()
    reference = declaration_reference(state['formal_declaration'], packed=True)
    assert reference == state['formal_declaration']
    with pytest.raises(ContractError):
        declaration_reference(reference)
    receipt = stop_receipt(state, invocation_unix=1_000_000., invocation_monotonic=1000.,
        observed_monotonic=6400., owned_bytes=1234)
    assert receipt['schema_version'] == 2
    assert receipt['kind'] == 'card-instance-formal-packed-stop'
    assert receipt['recipe_id'] == PACKED_RECIPE_ID
    assert receipt['reason'] == 'admission_deadline'
    state.update(status='complete', formal_stop=receipt)
    validate_formal_state(state)
    for field, value in [('schema_version', 1), ('kind', 'card-instance-formal-stop'), ('recipe_sha256', RECIPE_SHA256)]:
        changed = copy.deepcopy(state)
        changed['formal_stop'][field] = value
        with pytest.raises(ContractError):
            validate_formal_state(changed)


def test_declaration_binds_full_and_logical_config_storage_and_same_base_identity():
    from gf_common.card_instance_formal import planned_execution
    state = packed_state()
    config = state['effective_config']
    config_inputs = []
    base_identity = {'configuration_inputs_sha256': digest(config_inputs),
        'integration': {'server_config_sha256': load_recipe()['server_config_sha256']}}
    state.update(config_inputs=config_inputs, config_sha256=digest(config),
        execution=planned_execution(config), identity={**base_identity, 'formal_parent': state['formal_declaration']})
    declaration = {'schema_version': 2, 'kind': 'card-instance-formal-packed-parent-declaration',
        'recipe_id': PACKED_RECIPE_ID, 'recipe_sha256': PACKED_RECIPE_SHA256,
        'algorithm_recipe_sha256': ALGORITHM_SHA256, 'seed_slot': 17,
        'config_sha256': digest(config), 'logical_config_sha256': digest(checked_packed_config(config, load_recipe())),
        'config_inputs_sha256': digest(config_inputs), 'input_binding': load_recipe()['input_binding'],
        'execution_identity_sha256': digest(state['execution']), 'identity_sha256': digest(base_identity),
        'storage_binding': storage_binding()}
    assert set(declaration) == PACKED_DECLARATION_FIELDS
    validate_declaration(declaration, state, load_recipe())
    for field in ('logical_config_sha256', 'config_sha256'):
        changed = {**declaration, field: '0' * 64}
        with pytest.raises(ContractError):
            validate_declaration(changed, state, load_recipe())
    with pytest.raises(ContractError):
        validate_declaration({**declaration, 'extra': True}, state, load_recipe())


@pytest.mark.parametrize('schema,variant', [(3, PACKED_RUN_VARIANT), (4, 'formal-parent-v1'), (True, PACKED_RUN_VARIANT)])
def test_cross_variant_schema_and_storage_refuse(schema, variant):
    state = packed_state()
    state.update(schema_version=schema, run_variant=variant)
    with pytest.raises(ContractError):
        validate_formal_state(state)


def test_packed_formal_preflight_uses_same_literal_and_rejects_short_budget():
    from gf_training.card_instance_formal import formal_preflight
    from tests.test_formal_ppo_contract import loaded_for
    assert formal_preflight(loaded_for(packed_config())) == load_recipe()
    changed = packed_config()
    changed['training']['max_updates'] = 2
    with pytest.raises(ContractError):
        formal_preflight(loaded_for(changed))


def test_native_packed_formal_requires_both_exact_markers():
    from types import SimpleNamespace

    from gf_training.card_instance_formal import require_native_formal_controls
    info = {'card_instance_formal_controls': 'run3-formal-parent-v1;declaration1;stop1'}
    with pytest.raises(ContractError):
        require_native_formal_controls(SimpleNamespace(build_info=lambda: info), packed=True)
    info.update(card_instance_packed_formal_controls='run4-formal-packed-parent-v1;declaration2;stop2',
        card_instance_group_pack='provider1;proof2;plan4')
    require_native_formal_controls(SimpleNamespace(build_info=lambda: info), packed=True)
    del info['card_instance_group_pack']
    with pytest.raises(ContractError):
        require_native_formal_controls(SimpleNamespace(build_info=lambda: info), packed=True)


def test_real_control_lock_writes_variant2_before_any_generation_but_grants_no_source(tmp_path):
    import json
    import time
    from dataclasses import asdict

    from gf_cli.config import LoadedConfig, load_config
    from gf_common.card_instance_formal import planned_execution
    from gf_models.structured_sealed import SealedScan
    from gf_training.card_instance_formal import (
        FormalControl,
        FormalGameParent,
        read_formal_control,
    )
    from gf_training.lineage import run_lock
    server = tmp_path / 'server.toml'
    server.write_bytes((ROOT / 'configs/test-server.toml').read_bytes())
    config = packed_config()
    config['server']['config'] = str(server)
    source = ROOT / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml'
    text = source.read_text()
    for part in ('actions', 'models', 'preprocessing'):
        text = text.replace('"../' + part + '/', '"' + str(ROOT / 'configs' / part) + '/')
    input_path = tmp_path / 'formal-packed.toml'
    input_path.write_text(text + '\n[storage]\ndb_root="/unused/db"\nram_root="/unused/ram"\n')
    value = load_config(input_path, {key: config[key] for key in ('run', 'server', 'training')})
    loaded = LoadedConfig(value.config, asdict(value.config), digest(asdict(value.config)), value.inputs)
    control = FormalControl(loaded)
    control.preflight()
    directory = tmp_path / 'parent'
    with run_lock(directory, create=True) as lock:
        identity = {'configuration_inputs_sha256': digest(loaded.inputs),
            'integration': {'server_config_sha256': load_recipe()['server_config_sha256']}}
        execution = planned_execution(loaded.effective)
        control.attach_identity(directory, identity, execution, mode='fresh', deadline=time.monotonic() + 30, ownership=lock)
        state = packed_state()
        state.update(effective_config=loaded.effective, config_sha256=loaded.sha256, config_inputs=list(loaded.inputs),
            identity=identity, execution=execution)
        control.initialize_state(state)
        declaration = json.loads((directory / 'formal-declaration.json').read_bytes())
        assert declaration['schema_version'] == 2
        assert declaration['storage_binding'] == storage_binding()
        assert state['run_variant'] == PACKED_RUN_VARIANT
        assert not (directory / 'artifacts').exists()
        read_formal_control(directory, state, SealedScan(max_owned_bytes=1024**3, deadline=time.monotonic() + 30))
    control.finish()
    with pytest.raises((ContractError, FileNotFoundError)), FormalGameParent(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30):
        pytest.fail('controlled declaration cannot qualify a packed original Source')


def test_completion_counts_packed_external_parent_and_catalog_in_final_guard(tmp_path, monkeypatch):
    import time

    import gf_training.ppo_run as coordinator
    from gf_training.card_instance_formal import FormalCompletion
    from gf_training.journal import atomic_json
    state = packed_state()
    state.update(run_id='unqualified-control', identity={'formal_parent': state['formal_declaration']},
        generation_sha256='b' * 64, accepted_decisions=1, config_sha256=digest(state['effective_config']))
    directory = tmp_path / 'control'
    directory.mkdir()
    atomic_json(directory / 'run.json', state)
    owner = FormalCompletion()
    owner.publish(directory, state, deadline=time.monotonic() + 30, max_owned_bytes=1024**3)
    calls = []
    def used(path, deadline, *, external_roots=()):
        calls.append(external_roots)
        return 10 * 1024**3 + 1
    monkeypatch.setattr(coordinator, 'owned_bytes', used)
    try:
        with pytest.raises(ContractError, match='outer completion'):
            owner.finish()
        assert calls == [(Path('/unused/db/unqualified-control'), Path('/unused/db/catalog.sqlite3'),
            Path('/unused/db/catalog.sqlite3-wal'), Path('/unused/db/catalog.sqlite3-shm'),
            Path('/unused/db/catalog.sqlite3-journal'))]
    finally:
        owner.revoke()
