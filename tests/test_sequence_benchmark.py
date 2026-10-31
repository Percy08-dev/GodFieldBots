"""Tiny synthetic functional checks, without observer, source campaign or GPU runs."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKLOAD = Path(__file__).resolve().parents[1] / 'benchmarks/sequence_ppo/workload.py'


def helper():
    spec = importlib.util.spec_from_file_location('sequence_benchmark_workload', WORKLOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    from gf_models.config import GRUModelConfig
    from gf_training.sequence_numeric import Learner
    from tests.test_gru_ppo import persisted
    persisted(tmp_path, Learner(GRUModelConfig(), 17, 'cpu'), (2, 0, 3))
    return tmp_path / 'rollout' / 'rollout.json'


def arguments(module, source, *, model='transformer', mode='inference', order='cached-first'):
    return module.parser().parse_args(['--model', model, '--mode', mode, '--source', str(source), '--inference-order', order])


def test_parser_import_and_help_require_only_standard_library():
    program = f'''
import runpy, sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch', 'gf_models', 'gf_training'):
            raise AssertionError('optional/model import during parser setup: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
module = runpy.run_path({str(WORKLOAD)!r})
args = module['parser']().parse_args(['--model','gru','--mode','update','--source','/tmp/rollout.json'])
assert args.sequence_length == 32
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    module = helper()
    for window in ('1', '31', '1024', 'True'):
        with pytest.raises(SystemExit):
            module.parser().parse_args(['--model', 'gru', '--mode', 'update', '--source', '/tmp/rollout.json', '--sequence-length', window])


def test_cpu_rebase_changes_only_scalar_evidence_and_preserves_source_owners_and_bytes(source):
    import numpy as np

    from gf_models.arrays import sha256
    from gf_models.config import TransformerModelConfig
    from gf_training.sequence_numeric import Learner
    from gf_training.sequence_rollout import read_rollout
    module = helper()
    before = module.file_hashes(module.files(source))
    cpu = Learner(TransformerModelConfig(), 17, 'cpu')
    original = read_rollout(source.parent, sha256(source), cpu.model)
    proof = module.retained_proof(original)
    old = [(shard.old_logprob.copy(), shard.old_value.copy()) for shard in original.shards]
    inputs = module.tensor_inputs(original, cpu.torch)
    rebased = module.rebase(original, cpu, inputs)
    assert module.retained_proof(rebased) == proof
    assert cpu.parity(rebased)['decisions'] == 5
    assert rebased.banks is original.banks
    for index, (actual, expected) in enumerate(zip(rebased.shards, original.shards, strict=True)):
        for key in actual.arrays():
            if key not in ('old_logprob', 'old_value'):
                assert getattr(actual, key) is getattr(expected, key)
        assert not np.shares_memory(actual.old_logprob, expected.old_logprob)
        assert not np.shares_memory(actual.old_value, expected.old_value)
        np.testing.assert_array_equal(expected.old_logprob, old[index][0])
        np.testing.assert_array_equal(expected.old_value, old[index][1])
    assert any(not np.array_equal(a.old_value, b.old_value) for a, b in zip(rebased.shards, original.shards, strict=True))
    assert module.file_hashes(module.files(source)) == before
    state = json.dumps(cpu.shuffle.bit_generator.state, sort_keys=True)
    assert module.update_counts(cpu, rebased)['microbatches_per_epoch'] == [3] * 4
    assert json.dumps(cpu.shuffle.bit_generator.state, sort_keys=True) == state


@pytest.mark.parametrize('order', ['cached-first', 'full-first'])
def test_tiny_inference_paths_compare_every_candidate_and_leave_fixture_immutable(source, order):
    module = helper()
    result = module.run(arguments(module, source, order=order))
    assert result['status'] == 'complete' and result['source_unchanged']
    assert result['source_sha256_before'] == result['source_sha256_after']
    assert result['parity']['decisions'] == result['decisions'] == 5
    assert set(result['paths']) == {'cached', 'full'}
    assert all(value['kernel_seconds'] > 0 and value['p95_ms'] >= value['p50_ms'] for value in result['paths'].values())
    assert 'off-policy diagnostic' in result['purpose']
    assert 'final_weights_sha256' not in result


def test_tiny_update_fresh_rng_weights_and_complete_phase_accounting(source):
    module = helper()
    args = arguments(module, source, model='gru', mode='update')
    first, second = module.run(args), module.run(args)
    assert first['status'] == second['status'] == 'complete'
    assert first['initial_weights_sha256'] == second['initial_weights_sha256']
    assert first['final_weights_sha256'] == second['final_weights_sha256'] != first['initial_weights_sha256']
    assert first['array_proof'] == second['array_proof']
    assert first['microbatches_per_epoch'] == [3] * 4
    assert first['metrics']['optimizer_steps'] == 4 and first['finite_metrics']
    assert first['metrics']['parity']['decisions'] == 5
    assert first['update_seconds'] == first['parity_seconds'] + first['remaining_update_seconds']
    assert first['source_unchanged'] and second['source_unchanged']
    assert not list(source.parent.glob('weights*')) and not list(source.parent.glob('training.json'))


def test_corrupt_fixture_and_inference_cuda_fail_without_source_repair_or_fallback(source):
    module = helper()
    args = arguments(module, source)
    args.device = 'cuda:0'
    rejected = module.run(args)
    assert rejected['status'] == 'failed' and rejected['source_unchanged']
    assert 'requires CPU' in rejected['failure']
    bank = source.parent / 'bank-000000.npz'
    bank.write_bytes(bank.read_bytes() + b'corrupt temporary fixture')
    result = module.run(arguments(module, source))
    assert result['status'] == 'failed' and result['source_unchanged']
    assert 'hash mismatch' in result['failure']


def test_unsafe_source_member_and_output_inside_fixture_refuse_before_write(source, monkeypatch):
    module = helper()
    monkeypatch.setenv('GFB_BENCHMARK_RESULT', str(source.parent / 'forbidden.json'))
    monkeypatch.setattr(sys, 'argv', ['workload', '--model', 'gru', '--mode', 'inference', '--source', str(source)])
    with pytest.raises(ValueError, match='outside'):
        module.main()
    assert not (source.parent / 'forbidden.json').exists()
    manifest = json.loads(source.read_text())
    manifest['banks'][0]['file'] = '../escape.npz'
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe'):
        module.files(source)


def test_cli_writes_only_observer_result_and_preserves_existing_result(source, monkeypatch):
    module = helper()
    output = source.parent.parent / 'workload.json'
    monkeypatch.setenv('GFB_BENCHMARK_RESULT', str(output))
    monkeypatch.setattr(sys, 'argv', ['workload', '--model', 'gru', '--mode', 'inference', '--source', str(source)])
    assert module.main() == 0
    result = json.loads(output.read_text())
    assert result['status'] == 'complete' and result['source_unchanged']
    original = output.read_bytes()
    monkeypatch.setattr(module, 'run', lambda *args: pytest.fail('existing output must fail before compute'))
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_bytes() == original


def test_result_partial_serialization_failure_never_publishes_final_json(tmp_path, monkeypatch):
    module = helper()
    output = tmp_path / 'workload.json'
    primary = OSError('injected partial serialization failure')
    def broken_dump(result, stream, **options):
        stream.write('{"status":')
        assert Path(stream.name).parent == tmp_path
        assert not output.exists()
        raise primary
    monkeypatch.setattr(module.json, 'dump', broken_dump)
    with pytest.raises(OSError) as raised:
        module.publish(output, {'status': 'complete'})
    assert raised.value is primary
    assert not output.exists() and not list(tmp_path.iterdir())


def test_result_temporary_cleanup_failure_preserves_primary_exception(tmp_path, monkeypatch):
    module = helper()
    output = tmp_path / 'workload.json'
    primary = OSError('primary serialization error')
    def broken_dump(result, stream, **options):
        stream.write('{')
        raise primary
    with monkeypatch.context() as patch:
        patch.setattr(module.json, 'dump', broken_dump)
        patch.setattr(Path, 'unlink', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('cleanup failure')))
        with pytest.raises(OSError) as raised:
            module.publish(output, {'status': 'complete'})
    assert raised.value is primary
    assert any('cleanup failure' in note for note in primary.__notes__)
    assert not output.exists()
    for path in tmp_path.iterdir():
        path.unlink()


def test_result_publication_is_closed_synced_and_never_clobbers_race_winner(tmp_path, monkeypatch):
    module = helper()
    output = tmp_path / 'workload.json'
    calls = []
    original_sync, original_link = module.os.fsync, module.os.link
    def sync(descriptor):
        calls.append('sync')
        original_sync(descriptor)
    def link(temporary, final):
        assert calls == ['sync']
        assert json.loads(Path(temporary).read_text()) == {'status': 'complete'}
        output.write_text('{"status":"other owner"}\n')
        return original_link(temporary, final)
    monkeypatch.setattr(module.os, 'fsync', sync)
    monkeypatch.setattr(module.os, 'link', link)
    with pytest.raises(FileExistsError):
        module.publish(output, {'status': 'complete'})
    assert json.loads(output.read_text()) == {'status': 'other owner'}
    assert list(tmp_path.iterdir()) == [output]


def test_full_prefix_diagnostic_never_allocates_or_resets_an_unused_kv_cache(source, monkeypatch):
    from gf_models.arrays import sha256
    from gf_models.config import TransformerModelConfig
    from gf_training.sequence_numeric import Learner
    from gf_training.sequence_rollout import read_rollout
    module = helper()
    cpu = Learner(TransformerModelConfig(), 17, 'cpu')
    rollout = read_rollout(source.parent, sha256(source), cpu.model)
    inputs = module.tensor_inputs(rollout, cpu.torch)
    monkeypatch.setattr(cpu.network, 'create_cache', lambda: pytest.fail('full-prefix mode must not allocate KV inference storage'))
    assert len(list(module.forward_rows(cpu.network, rollout, cpu.torch, inputs, 'full', timed=True))) == 5


def test_owned_empty_runtime_cache_cleanup_preserves_unexpected_files(tmp_path):
    module = helper()
    cache = tmp_path / 'sequence-inductor-cache'
    module.cleanup_empty_cache(cache)
    cache.mkdir()
    module.cleanup_empty_cache(cache)
    assert not cache.exists()
    cache.mkdir()
    artifact = cache / 'unexpected'
    artifact.write_bytes(b'retain failure evidence')
    with pytest.raises(OSError):
        module.cleanup_empty_cache(cache)
    assert artifact.read_bytes() == b'retain failure evidence'


def test_cleanup_failure_keeps_the_original_diagnostic_failure(source, tmp_path, monkeypatch):
    module = helper()
    args = arguments(module, source)
    args.device = 'cuda:0'
    monkeypatch.setenv('GFB_BENCHMARK_SCRATCH', str(tmp_path / 'owned-scratch'))
    monkeypatch.setattr(module, 'cleanup_empty_cache', lambda path: (_ for _ in ()).throw(OSError('cache cleanup failed')))
    result = module.run(args)
    assert result['status'] == 'failed' and result['source_unchanged']
    assert 'requires CPU' in result['failure']
    assert 'cache cleanup failed' in result['cleanup_failure']
