"""Bounded checks for native benchmark plans, source proofs and owner adapters."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

TOOL = Path(__file__).resolve().parents[1] / 'tools/benchmark_workload.py'
spec = importlib.util.spec_from_file_location('native_benchmark_workload', TOOL)
assert spec is not None and spec.loader is not None
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


def fake_preflight(source: Path, kind: str, **kwargs: Any) -> dict[str, Any]:
    proof: dict[str, Any] = {'source': str(source), 'actual_imports': {}}
    if kwargs.get('native'):
        proof['native_identity'] = {'extension_sha256': 'binary-sha', 'build': {'abi': 'cp314-gil'}}
    return proof


@pytest.mark.parametrize('workers', [1, 2, 4, 8])
def test_native_plan_is_only_matching_budget_and_alternating_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workers: int,
) -> None:
    monkeypatch.setattr(workload, 'preflight_source', fake_preflight)
    monkeypatch.setenv('VIRTUAL_ENV', '/wrong-venv')
    monkeypatch.setenv('PYTHONPATH', '/explicit-native-path')
    output = tmp_path / 'plan'
    monkeypatch.setattr(sys, 'argv', [str(TOOL), 'paired', '--kind', 'preparation',
        '--comparison', 'native', '--output', str(output), '--workers', str(workers),
        '--max-in-flight', str(workers), '--native-pythonpath', str(tmp_path / 'native')])
    workload.main()
    decoded = json.loads(capsys.readouterr().out)
    plan = decoded['plan']
    assert [(row['trial'], row['variant']) for row in plan] == [
        (-1, 'python'), (-1, 'rust'), (0, 'python'), (0, 'rust'),
        (1, 'rust'), (1, 'python'), (2, 'python'), (2, 'rust')]
    assert {row['cell'] for row in plan} == {'standard'}
    assert decoded['aliases'] == {}
    assert not output.exists()
    for row in plan:
        command = row['command']
        assert command[:2] == [sys.executable, str(workload.ROOT / 'tools/benchmark_performance.py')]
        assert command[command.index('--memory-mib') + 1] == '2048'
        assert command[command.index('--chunk-mib') + 1] == '16'
        assert command[command.index('env') + 1:command.index('env') + 3] == ['-u', 'VIRTUAL_ENV']
        assert command.count(sys.executable) == 2
        search = next(item.removeprefix('PYTHONPATH=') for item in command if item.startswith('PYTHONPATH='))
        if row['variant'] == 'python':
            assert search == '/tmp/gfb-native-reference-42ce5e6/src'
            assert '--workers' not in command and '--max-in-flight' not in command
            assert '--native-sha256' not in command
        else:
            assert search.split(os.pathsep) == [str(workload.ROOT / 'src'),
                str(tmp_path / 'native'), '/explicit-native-path']
            assert command[command.index('--workers') + 1] == str(workers)
            assert command[command.index('--max-in-flight') + 1] == str(workers)
            assert command[command.index('--native-sha256') + 1] == 'binary-sha'


@pytest.mark.parametrize('options, message', [
    (['--kind', 'fixed'], 'requires preparation'),
    (['--kind', 'preparation', '--workers', '2', '--max-in-flight', '1'], 'workers/max-in-flight'),
])
def test_invalid_native_plan_cannot_probe_or_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], message: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('invalid plan reached source probe')
    monkeypatch.setattr(workload, 'preflight_source', forbidden)
    output = tmp_path / 'invalid'
    monkeypatch.setattr(sys, 'argv', [str(TOOL), 'paired', '--comparison', 'native',
                                   '--output', str(output), *options])
    with pytest.raises(ValueError, match=message):
        workload.main()
    assert not output.exists()


def make_source(source: Path, *, native: bool = False) -> None:
    for name in workload.source_modules('preparation', native=native):
        path = source / Path(*name.split('.'))
        path = path / '__init__.py' if '.' not in name else path.with_suffix('.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')


def test_reference_probe_rejects_fallback_to_current_python_source(tmp_path: Path) -> None:
    source = tmp_path / 'reference/src'
    make_source(source)
    current = tmp_path / 'current/gf_training'
    current.mkdir(parents=True)
    for name in ('fixed_data', 'q_data', 'journal', 'lineage'):
        (current / (name + '.py')).write_text('')
    (source / 'gf_training/__init__.py').write_text(f'__path__ = [{str(current)!r}]\n')
    with pytest.raises(ValueError, match='shadowed source import'):
        workload.preflight_source(source, 'preparation')


def test_absent_backend_fails_native_preflight_before_inventory_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / 'reference'
    root = tmp_path / 'current'
    make_source(reference / 'src')
    make_source(root / 'src', native=True)
    (root / 'src/gf_training/native_backend.py').write_text(
        'def binary_identity():\n    raise ImportError("native backend absent")\n')
    monkeypatch.setattr(workload, 'ROOT', root)
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('failed native probe reached inventory')
    monkeypatch.setattr(workload, 'source_inventory', forbidden)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', [str(TOOL), 'paired', '--kind', 'preparation',
        '--comparison', 'native', '--reference', str(reference), '--output', str(output), '--execute'])
    with pytest.raises(ValueError, match='native backend absent'):
        workload.main()
    assert not output.exists()


def test_native_binary_proof_hashes_actual_extension_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gf_training.native_backend import binary_identity

    binary_path = tmp_path / ('_native' + EXTENSION_SUFFIXES[0])
    binary_path.write_bytes(b'actual loaded binary bytes')
    binary = ModuleType('benchmark_fake_native_extension')
    binary.__file__ = str(binary_path)
    monkeypatch.setitem(sys.modules, binary.__name__, binary)
    def build_info() -> dict[str, str]:
        return {'abi': 'cp314-gil', 'recipe_identity': 'native-preprocessing-v1'}
    build_info.__module__ = binary.__name__
    native = ModuleType('benchmark_fake_native')
    native.build_info = build_info  # type: ignore[attr-defined]
    backend = ModuleType('gf_training.native_backend')
    backend.binary_identity = lambda: binary_identity(native)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'gf_training.native_backend', backend)
    identity = workload.native_identity()
    assert identity['extension_path'] == str(binary_path)
    assert identity['extension_sha256'] == hashlib.sha256(binary_path.read_bytes()).hexdigest()
    assert identity['build'] == build_info()
    binary.__file__ = str(tmp_path / 'fake.py')
    Path(binary.__file__).write_text('# never a native binary')
    with pytest.raises(ValueError, match='loaded extension binary'):
        workload.native_identity()


def test_workload_source_proof_refuses_different_binary_before_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workload, 'source_probe', lambda *args: {})
    monkeypatch.setattr(workload, 'native_identity', lambda: {'extension_sha256': 'different'})
    args = workload.argument_parser().parse_args(['preparation', '--variant', 'rust',
                                                '--native-sha256', 'expected'])
    with pytest.raises(ValueError, match='changed after preflight'):
        workload.workload_source_proof(args)


@pytest.mark.parametrize('variant', ['python', 'rust'])
def test_preparation_passes_backend_settings_to_owner_and_preserves_output_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    dataset_path = tmp_path / 'dataset.json'
    dataset_path.write_text('{}')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setenv('GFB_BENCHMARK_SCRATCH', str(scratch))
    calls: dict[str, Any] = {}
    diagnostics = {'sample_identity_sha256': 'samples', 'ffi_data_calls': 7,
                   'payload_copy_stats': {'feature_payload_bytes': 0}, 'read_bytes': 123}
    def finalize():
        calls.update(finalized=True)
        diagnostics['ffi_data_calls'] = 99
    dataset = {'content_sha256': workload.FROZEN, 'train_decisions': 81523, 'validation_decisions': 2260}
    owner = SimpleNamespace(dataset=dataset, diagnostics=diagnostics,
        write_numeric=lambda path: path.write_bytes(b'numeric'),
        finalize=finalize, abort=lambda: calls.update(aborted=True))
    def prepare(path: Path, encoder: Any, **kwargs: Any) -> Any:
        calls.update(kwargs)
        return owner
    config_module = ModuleType('gf_models.config')
    config_module.EncoderConfig = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    preparation_module = ModuleType('gf_common.preprocessing')
    preparation_module.PreparationConfig = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fixed_module = ModuleType('gf_training.fixed_data')
    fixed_module.prepare_fixed_inputs = prepare  # type: ignore[attr-defined]
    for module in (config_module, preparation_module, fixed_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(workload, 'archive_proof', lambda path: {'x': {'sha256': 'raw-array'}})
    args = SimpleNamespace(variant=variant, encoder='history', memory_mib=2048, chunk_mib=16,
        workers=4, max_in_flight=4, dataset=dataset_path, command_started=0., seconds=7200.)
    result = workload.preparation(args)
    settings: dict[str, Any] = {'memory_budget_mib': 2048, 'chunk_target_mib': 16, 'cache_budget_mib': 512}
    if variant == 'rust':
        settings.update(backend='rust', reader='legacy', workers=4, max_in_flight=4)
        assert result['native_transfer_metrics']['ffi_data_calls']['value'] == 99
    assert calls['preprocessing'] == settings
    assert calls['finalized'] is True
    assert result['data_stage_preparation_diagnostics']['ffi_data_calls'] == 7
    assert result['nested_preparation_diagnostics']['ffi_data_calls'] == 99
    assert result['accepted_actions'] == 83783
    assert result['sample_identity_sha256'] == 'samples'
    assert result['numeric_archive_sha256'] == hashlib.sha256(b'numeric').hexdigest()
    assert result['arrays'] == {'x': {'sha256': 'raw-array'}}
    assert not (scratch / 'arrays.npz').exists()
    dataset['train_decisions'] = 1
    with pytest.raises(ValueError, match='full 83,783-row'):
        workload.preparation(args)
    assert calls['aborted'] is True


def test_missing_native_transfer_counters_remain_unavailable() -> None:
    metrics = workload.native_transfer_metrics({})
    assert all(item['value'] is None and item['unavailable_reason'] for item in metrics.values())
    measured = workload.native_transfer_metrics({'ffi_data_calls': 0, 'payload_copy_stats': {'bytes': 0}})
    assert measured['ffi_data_calls'] == {'value': 0, 'unavailable_reason': None}
    assert measured['payload_copy_stats']['value'] == {'bytes': 0}
