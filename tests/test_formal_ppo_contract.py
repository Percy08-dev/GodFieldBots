"""Pure formal-control predicates; these records are NOT original game Sources."""
import ast
import copy
import hashlib
import importlib.resources
import json
import sys
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.card_instance_formal import (
    ALGORITHM_SHA256,
    RECIPE_ID,
    RECIPE_SHA256,
    RUN_VARIANT,
    checked_config,
    clock_values,
    recipe_bytes,
    stop_receipt,
    validate_formal_state,
    validate_stop,
)
from gf_common.domain import ContractError, digest
from gf_training.card_instance_formal import (
    formal_preflight,
    load_recipe,
    require_native_formal_controls,
    verify_kernel_ast,
)

ROOT = Path(__file__).resolve().parents[1]


def config_for(model='mlp', device='cpu', backend='python', seed=17):
    recipe = load_recipe()
    value = copy.deepcopy(recipe['base_config'])
    value['model'] = copy.deepcopy(recipe['model_profiles'][model])
    value['model']['inference']['device'] = device
    value['preprocessing']['backend'] = backend
    value['training']['seeds'] = [seed]
    value['run'].update(name='component-control-only', output_dir='/component/unused-output')
    value['server'].update(source_dir='/component/unused-source', binary='/component/unused-binary', config='/component/unused-server.toml')
    return value


def loaded_for(value=None):
    value = value or config_for()
    kind = make_dataclass('PureFormalConfig', [(key, object) for key in value])
    config = kind(**copy.deepcopy(value))
    return SimpleNamespace(config=config, effective=copy.deepcopy(value), sha256=digest(value), inputs=())


def state_for(*, updates=1):
    config = config_for()
    return {'effective_config': config, 'created_unix': 1_000_000., 'last_unix': 1_007_100.,
        'wall_deadline_unix': 1_007_200., 'admission_deadline_unix': 1_005_400., 'execution_deadline_unix': 1_007_140.,
        'updates': updates, 'completed_updates': updates, 'optimizer_steps': updates * 4, 'elapsed_seconds': 7100.,
        'formal_declaration': {'relative_path': 'formal-declaration.json', 'sha256': 'a' * 64,
            'recipe_id': RECIPE_ID, 'recipe_sha256': RECIPE_SHA256, 'algorithm_recipe_sha256': ALGORITHM_SHA256, 'seed_slot': 17},
        'run_variant': RUN_VARIANT, 'formal_stop': None, 'status': 'running'}


@pytest.mark.parametrize('model', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('backend', ['python', 'rust'])
@pytest.mark.parametrize('seed', [17, 29, 43])
def test_all36_literal_variants_reject_retired_binding_without_optional_imports(model, device, backend, seed):
    optional_before = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    value = config_for(model, device, backend, seed)
    assert checked_config(value, load_recipe()) == value
    with pytest.raises(ContractError, match='^formal planned input/static semantics differ from literal$'):
        formal_preflight(loaded_for(value))
    optional_after = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    assert optional_after.keys() == optional_before.keys()
    assert all(optional_after[name] is module for name, module in optional_before.items())


def test_real_preflight_rejects_frozen_v1_recipe_outside_unadmitted_control_helpers():
    recipe = load_recipe()
    assert recipe['input_binding']['rule_signature'] == '3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d'
    with pytest.raises(ContractError, match='^formal planned input/static semantics differ from literal$'):
        formal_preflight(loaded_for())


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('training', 'max_updates', 2), ('training', 'device', 'cpu'), ('training', 'max_owned_bytes', 100),
    ('training', 'admission_reservation_bytes', 1), ('training', 'seeds', [17, 29]), ('training', 'seeds', [True]),
    ('training', 'seeds', [11]), ('run', 'wall_seconds', 7199), ('run', 'admission_seconds', 5399),
    ('run', 'execution_seconds', 7139), ('run', 'finalization_seconds', 59), ('run', 'concurrency', 1),
    ('preprocessing', 'workers', 2), ('preprocessing', 'reader', 'structured-v1'),
    ('server', 'external_url', 'http://unowned'), ('actions', 'max_candidates', 64),
])
def test_changed_full_or_inactive_recipe_values_fail(section, key, value):
    config = config_for()
    config[section][key] = value
    with pytest.raises(ContractError):
        formal_preflight(loaded_for(config))


@pytest.mark.parametrize('change', ['hand', 'inactive', 'visibility', 'extra', 'profile'])
def test_no_general_overlay_or_mutable_default_fallback(change):
    value = config_for()
    if change == 'hand':
        value['model']['hand']['kind'] = 'direct'
    elif change == 'inactive':
        value['model']['temporal']['feedforward_size'] = 512
    elif change == 'visibility':
        value['model']['cuda_visible_devices'] = '0'
    elif change == 'profile':
        value['actions']['action_contract']['profile_id'] = 'old-profile'
    else:
        value['formal'] = True
    with pytest.raises(ContractError):
        formal_preflight(loaded_for(value))


def test_packaged_literal_bytes_and_algorithm_content_are_pinned():
    raw = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-v1.json').read_bytes()
    assert raw == (ROOT / 'docs/contracts/card-instance-formal-ppo-v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECIPE_SHA256
    recipe_bytes(raw)
    with pytest.raises(ContractError):
        recipe_bytes(raw + b'\n')
    recipe = load_recipe()
    name = next(iter(recipe['algorithm_recipe']['kernel_sha256']))
    recipe['algorithm_recipe']['kernel_sha256'][name] = 'f' * 64
    with pytest.raises(ContractError, match='kernel'):
        verify_kernel_ast(recipe)


def test_planned_adam_mismatch_fails_before_any_torch(monkeypatch):
    optional_before = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    import gf_models.structured_artifacts as module
    monkeypatch.setattr(module, 'ADAM_RECIPE', {**module.ADAM_RECIPE, 'foreach': False})
    with pytest.raises(ContractError, match='Adam'):
        formal_preflight(loaded_for())
    optional_after = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    assert optional_after.keys() == optional_before.keys()
    assert all(optional_after[name] is module for name, module in optional_before.items())


def test_rust_formal_marker_is_separate_and_current_reader_marker_is_insufficient():
    with pytest.raises(ContractError, match='no fallback'):
        require_native_formal_controls(SimpleNamespace(build_info=lambda: [('card_instance_contract', 'ordinary')]))
    require_native_formal_controls(SimpleNamespace(build_info=lambda: [('card_instance_formal_controls', 'run3-formal-parent-v1;declaration1;stop1')]))


@pytest.mark.parametrize(('updates', 'elapsed', 'usage', 'reason'), [
    (1000, 5401, 10 * 1024**3, 'max_updates'), (1, 5400, 9 * 1024**3 + 1, 'admission_deadline'),
    (1, 100, 9 * 1024**3 + 1, 'storage_admission'), (1, 100, 9 * 1024**3, None),
])
def test_stop_priority_and_strict_storage_boundary(updates, elapsed, usage, reason):
    state = state_for(updates=updates)
    receipt = stop_receipt(state, invocation_unix=1_000_000., invocation_monotonic=1000.,
        observed_monotonic=1000. + elapsed, owned_bytes=usage)
    assert (None if receipt is None else receipt['reason']) == reason
    if receipt:
        state.update(status='complete', formal_stop=receipt)
        validate_formal_state(state)
        validate_stop(state)


@pytest.mark.parametrize(('elapsed', 'usage'), [(7140, 0), (7200, 0), (10, 10 * 1024**3 + 1)])
def test_execution_wall_and_actual_storage_overrun_precede_every_success_reason(elapsed, usage):
    with pytest.raises(ContractError, match='overrun'):
        stop_receipt(state_for(updates=1000), invocation_unix=1_000_000., invocation_monotonic=1000.,
            observed_monotonic=1000. + elapsed, owned_bytes=usage)


def test_zero_updates_and_nonfinite_clocks_refuse_and_pause_has_no_stop():
    with pytest.raises(ContractError, match='zero-update'):
        stop_receipt(state_for(updates=0), invocation_unix=1_000_000., invocation_monotonic=1000.,
            observed_monotonic=6400., owned_bytes=0)
    with pytest.raises(ContractError, match='finite'):
        stop_receipt(state_for(), invocation_unix=1_000_000., invocation_monotonic=1000.,
            observed_monotonic=float('nan'), owned_bytes=0)
    state = state_for()
    state['status'] = 'paused'
    validate_formal_state(state)
    state['formal_stop'] = {}
    with pytest.raises(ContractError):
        validate_formal_state(state)


def test_deadline_operation_order_paused_time_and_receipt_is_not_cleanup_storage():
    state = state_for()
    unix, mono = 1_003_000.1234567, 1000.1234567
    clocks = clock_values(state, unix, mono, mono + 2500.)
    for name, original in (('admission', 'admission_deadline_unix'), ('execution', 'execution_deadline_unix'), ('wall', 'wall_deadline_unix')):
        assert clocks[name + '_deadline_monotonic'] == (mono + state[original]) - unix
    receipt = stop_receipt(state, invocation_unix=unix, invocation_monotonic=mono,
        observed_monotonic=mono + 2500., owned_bytes=1234)
    state.update(status='complete', formal_stop=receipt)
    assert receipt['elapsed_seconds'] > 5500 and receipt['owned_bytes'] == 1234
    validate_stop(state)
    state['formal_stop'] = {**receipt, 'owned_bytes': 1235, 'reason': 'storage_admission'}
    with pytest.raises(ContractError):
        validate_stop(state)


def test_extracted_terminal_helpers_are_byte_exact_and_tree_private_aliases_remain():
    import gf_common.owned_completion as common
    import gf_models.structured_tree_pairing as tree
    assert tree._close_owned_descriptor is common._close_owned_descriptor
    assert tree._invalidate_owned_completion is common._invalidate_owned_completion
    # Original baseline raw-source pins, computed BEFORE the approved move.
    # Expected hashes never come from the moved implementation at runtime.
    original = {'_close_owned_descriptor': 'c2694eb9cb4d6a72dd64761d6654bfe892252201aa5add7ff73e4e6dea2cdcae', '_invalidate_owned_completion': '890708b021128b4b319c1c0554b8aa11b0ff43f35b2ace100f2d3542925dfb1d'}
    moved = Path(common.__file__).read_text()
    for name, expected in original.items():
        node = next(node for node in ast.parse(moved).body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert hashlib.sha256(ast.get_source_segment(moved, node).encode()).hexdigest() == expected


def test_actual_paired_adam_metadata_compared_to_literal_not_mutable_defaults():
    from gf_training.card_instance_formal import verify_actual_optimizer_metadata
    original = load_recipe()['algorithm_recipe']['adam']
    verify_actual_optimizer_metadata({'optimizer_recipe': original})
    for key, value in [('foreach', False), ('capturable', True), ('lr', .001), ('eps', 1e-7), ('weight_decay', .1), ('decoupled_weight_decay', True)]:
        with pytest.raises(ContractError, match='actual named Adam'):
            verify_actual_optimizer_metadata({'optimizer_recipe': {**original, key: value}})


def test_isolated_E_child_imports_all_seven_copy_packages_and_no_learning_runtime():
    import subprocess
    code = '''
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
paths={name:Path(importlib.import_module(name).__file__).resolve() for name in ('godfieldbots','gf_common','gf_models','gf_training','gf_cli','gf_evaluation','gf_simulator_client')}
assert all(path.is_relative_to(root/'src') for path in paths.values())
assert not ({'torch','lightgbm','gf_native'} & sys.modules.keys())
print(json.dumps({'copy_packages':len(paths),'isolated':True,'torch':False,'lightgbm':False}))
'''
    result = subprocess.run([sys.executable, '-E', '-c', code, str(ROOT)], cwd=ROOT / 'src', capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'copy_packages': 7, 'isolated': True, 'torch': False, 'lightgbm': False}


@pytest.mark.parametrize('version,gil', [((3, 15), True), ((3, 14), False)])
def test_formal_python_ast_and_gil_runtime_is_pinned_before_control_allocation(monkeypatch, version, gil):
    import gf_training.card_instance_formal as module
    monkeypatch.setattr(module, 'sys', SimpleNamespace(version_info=version, _is_gil_enabled=lambda: gil))
    with pytest.raises(ContractError, match='Python3.14'):
        formal_preflight(loaded_for())


@pytest.mark.parametrize('skew,accepted', [(-.05, True), (-.1, True), (-.1001, False), (-.11, False)])
def test_stop_final_clock_uses_same_point_one_skew_and_absolute_float_order(skew, accepted):
    state = state_for()
    state['created_unix'] = 1_793_000_000.1234567
    state['admission_deadline_unix'] = state['created_unix'] + 5400
    state['execution_deadline_unix'] = state['created_unix'] + 7140
    state['wall_deadline_unix'] = state['created_unix'] + 7200
    unix, mono = state['created_unix'], 123456.9876543
    observed = mono + 5401
    receipt = stop_receipt(state, invocation_unix=unix, invocation_monotonic=mono,
        observed_monotonic=observed, owned_bytes=1234)
    expected_stop_unix = unix + (observed - mono)
    state.update(status='complete', formal_stop=receipt, last_unix=expected_stop_unix + skew,
                 elapsed_seconds=(expected_stop_unix + skew) - state['created_unix'])
    assert (state['last_unix'] < expected_stop_unix - .1) is (not accepted)
    if accepted:
        validate_stop(state)
    else:
        with pytest.raises(ContractError):
            validate_stop(state)
