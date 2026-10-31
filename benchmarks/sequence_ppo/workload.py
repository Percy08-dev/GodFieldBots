"""Diagnostic fixed-array compute only; invoke through tools/benchmark_performance.py.

No training eligibility, game-quality evidence, checkpoints or weights are published.
The caller runs every warmup/trial in a fresh process with the same immutable source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', required=True, choices=('gru', 'transformer'))
    result.add_argument('--mode', required=True, choices=('inference', 'update'))
    result.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    result.add_argument('--sequence-length', type=int, choices=(32, 64, 128), default=32)
    result.add_argument('--source', required=True, type=Path)
    result.add_argument('--inference-order', choices=('cached-first', 'full-first'), default='cached-first')
    return result


def files(source: Path) -> list[Path]:
    if source.name != 'rollout.json':
        raise ValueError('source must be the immutable schema-2 rollout.json')
    manifest = json.loads(source.read_text())
    output = [source]
    for kind in ('bank', 'shard'):
        for index, entry in enumerate(manifest[kind + 's']):
            if entry['file'] != f'{kind}-{index:06}.npz':
                raise ValueError('unsafe source member')
            path = source.parent / entry['file']
            output.append(path)
            if path.with_suffix('.json').exists():
                output.append(path.with_suffix('.json'))
    return output


def file_hashes(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        with path.open('rb') as stream:
            result[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def array_digest(items: Any) -> str:
    checksum = hashlib.sha256()
    for name, value in items:
        checksum.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        if value.nbytes:
            checksum.update(memoryview(value).cast('B'))
    return checksum.hexdigest()


def retained_proof(rollout: Any) -> str:
    return array_digest([(f'bank.{i}', bank) for i, bank in enumerate(rollout.banks)]
                        + [(f'shard.{i}.{key}', value) for i, shard in enumerate(rollout.shards)
                           for key, value in shard.arrays().items() if key not in ('old_logprob', 'old_value')])


def tensor_inputs(rollout: Any, torch: Any) -> tuple[Any, Any]:
    return (tuple(torch.from_numpy(bank.copy()) for bank in rollout.banks),
            tuple(torch.from_numpy(shard.actor.copy()) for shard in rollout.shards))


def forward_rows(network: Any, rollout: Any, torch: Any, inputs: Any, mode: str, *, timed: bool = False) -> Any:
    banks, actors = inputs
    cache, bank_id = None, -1
    for index in range(len(rollout)):
        shard_id = int(rollout.offsets.searchsorted(index, side='right')) - 1
        shard, row = rollout.shards[shard_id], index - int(rollout.offsets[shard_id])
        current, position = int(shard.bank[row]), int(shard.position[row])
        if current != bank_id:
            if mode == 'cached':
                if cache is None:
                    cache = network.create_cache()
                cache.reset()
            bank_id = current
        token = banks[current][position:position + 1]
        actor = actors[shard_id][shard.offsets[row]:shard.offsets[row + 1]][None]
        mask = torch.ones((1, actor.shape[1]), dtype=torch.bool)
        history = banks[current][:position + 1][None]
        started = time.perf_counter_ns() if timed else 0
        with torch.no_grad():
            if mode == 'cached':
                context = network.step(token, cache)
                logs, value = network.heads(context, token, actor, mask)
            else:
                logs, value = network(history, actor, mask)
        elapsed = time.perf_counter_ns() - started if timed else 0
        yield shard_id, row, logs[0].numpy(), float(value[0]), elapsed


def rebase(rollout: Any, cpu: Any, inputs: Any) -> Any:
    from gf_training.sequence_rollout import SequenceRollout
    shards = tuple(replace(shard, old_logprob=shard.old_logprob.copy(), old_value=shard.old_value.copy()) for shard in rollout.shards)
    for shard_id, row, logs, value, _ in forward_rows(cpu.network, rollout, cpu.torch, inputs, 'cached'):
        shards[shard_id].old_logprob[row] = logs[int(shards[shard_id].selected[row])]
        shards[shard_id].old_value[row] = value
    for shard in shards:
        shard.old_logprob.flags.writeable = shard.old_value.flags.writeable = False
    return SequenceRollout(rollout.banks, shards)


def inference(rollout: Any, cpu: Any, inputs: Any, order: str) -> dict[str, Any]:
    import numpy as np
    logs = np.empty(sum(len(shard.actor) for shard in rollout.shards), dtype=np.float32)
    values = np.empty(len(rollout), dtype=np.float32)
    result = {}
    for pass_id, mode in enumerate(('cached', 'full') if order == 'cached-first' else ('full', 'cached')):
        times = np.empty(len(rollout), dtype=np.int64)
        candidate = 0
        started = time.perf_counter()
        for index, (_, _, row, value, elapsed) in enumerate(forward_rows(cpu.network, rollout, cpu.torch, inputs, mode, timed=True)):
            if pass_id == 0:
                logs[candidate:candidate + len(row)], values[index] = row, value
            elif not np.allclose(logs[candidate:candidate + len(row)], row, atol=1e-5, rtol=1e-5) or not np.isclose(values[index], value, atol=1e-5, rtol=1e-5):
                raise ValueError('all-step cached/full candidate/value parity failed')
            candidate += len(row)
            times[index] = elapsed
        result[mode] = {'kernel_seconds': float(times.sum()) / 1e9, 'instrumented_diagnostic_wall_seconds': time.perf_counter() - started,
                        'p50_ms': float(np.percentile(times, 50)) / 1e6, 'p95_ms': float(np.percentile(times, 95)) / 1e6}
    return {'paths': result, 'parity': {'decisions': len(rollout), 'atol': 1e-5, 'rtol': 1e-5},
            'scope': 'compare kernel_seconds/p50_ms/p95_ms only: CPU step/full-prefix forward and heads excludes loading, packing, checks and sampling; instrumented diagnostic wall includes asymmetric first-reference storage versus second-pass comparisons'}


def update_counts(learner: Any, rollout: Any) -> dict[str, Any]:
    import numpy as np

    from gf_training.sequence_numeric import microbatches
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = learner.shuffle.bit_generator.state
    counts = []
    for _ in range(4):
        order = rng.permutation(len(rollout)).tolist()
        counts.append(sum(len(microbatches(order[start:start + 128], rollout, learner.sequence_length,
                                          attention_multiplier=learner.attention_multiplier)) for start in range(0, len(order), 128)))
    return {'epochs': 4, 'microbatches_per_epoch': counts,
            'logical_minibatches_per_epoch': (len(rollout) + 127) // 128,
            'count_scope': 'independent initial PCG64 copy and production grouper; no learner RNG mutation'}


def run(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    scratch = os.environ.get('GFB_BENCHMARK_SCRATCH')
    if scratch is not None:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(scratch) / 'sequence-inductor-cache')
    for key, value in {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
        os.environ[key] = value
    from gf_models.config import GRUModelConfig, TransformerModelConfig
    from gf_models.torch_runtime import weights
    from gf_training.ppo_run import source_identity
    from gf_training.sequence_numeric import Learner, microbatches
    from gf_training.sequence_rollout import read_rollout
    paths = files(args.source)
    before = file_hashes(paths)
    result: dict[str, Any] = {'schema_version': 1, 'purpose': 'off-policy diagnostic compute fixture with fixed source targets and privately rebased old evidence; never training-eligible, on-policy, or strength evidence',
                              'status': 'failed', 'arguments': {**vars(args), 'source': str(args.source.resolve())},
                              'source_sha256_before': before, 'seed': 17,
                              'memory_scope': 'observer whole-command RSS includes runtime/model setup, full fixture loading and tensor copies, CPU rebasing and code/source hashing; not production participant RSS',
                              'campaign_scope': 'per-command source immutability only; campaign owner must freeze these same source digests across all trials'}
    try:
        if args.mode == 'inference' and args.device != 'cpu':
            raise ValueError('participant inference benchmark requires CPU')
        model = GRUModelConfig() if args.model == 'gru' else TransformerModelConfig()
        rollout = read_rollout(args.source.parent, before['rollout.json'], model)
        cpu = Learner(model, 17, 'cpu', args.sequence_length)
        model_bytes = sum(t.numel() * t.element_size() for t in (*cpu.network.parameters(), *cpu.network.buffers()))
        if model_bytes + cpu.network.cache_storage_bytes > model.inference.cache_budget_bytes:
            raise ValueError('participant model/cache resource limit exceeded')
        for start in range(0, len(rollout), 128):
            microbatches(list(range(start, min(start + 128, len(rollout)))), rollout, args.sequence_length,
                         attention_multiplier=cpu.attention_multiplier, retained_cache_bytes=cpu.network.cache_storage_bytes)
        inputs = tensor_inputs(rollout, cpu.torch)
        rebased = rebase(rollout, cpu, inputs)
        proof = retained_proof(rollout)
        if retained_proof(rebased) != proof:
            raise ValueError('rebase changed frozen arrays')
        result.update(model=asdict(model), code=source_identity(), helper_sha256=file_hashes([Path(__file__)])[Path(__file__).name],
                      initial_weights_sha256=array_digest(sorted(weights(cpu.network).items())), array_proof=proof,
                      rebased_old_evidence_sha256=array_digest([(f'{i}.{key}', getattr(shard, key)) for i, shard in enumerate(rebased.shards) for key in ('old_logprob', 'old_value')]),
                      decisions=len(rollout), model_bytes=model_bytes, cache_bytes=cpu.network.cache_storage_bytes,
                      rebased_scalar_bytes=sum(shard.old_logprob.nbytes + shard.old_value.nbytes for shard in rebased.shards),
                      input_bytes=sum(bank.nbytes for bank in rollout.banks) + sum(value.nbytes for shard in rollout.shards for value in shard.arrays().values()))
        if args.mode == 'inference':
            result.update(inference(rebased, cpu, inputs, args.inference_order))
        else:
            del inputs
            learner = cpu if args.device == 'cpu' else Learner(model, 17, args.device, args.sequence_length)
            if args.device != 'cpu':
                del cpu
            actual_initial = array_digest(sorted(weights(learner.network).items()))
            if actual_initial != result['initial_weights_sha256']:
                raise ValueError('CPU rebasing and learner initial weights differ')
            result.update(hardware=learner.hardware(), initial_shuffle_rng=learner.shuffle.bit_generator.state,
                          **update_counts(learner, rebased))
            original, phase = learner.parity, {}
            def parity(batch: Any) -> Any:
                learner.synchronize()
                tick = time.perf_counter()
                output = original(batch)
                learner.synchronize()
                phase['parity_seconds'] = time.perf_counter() - tick
                return output
            setattr(learner, 'parity', parity)  # noqa: B010 -- timing hook on this fresh learner
            learner.synchronize()
            tick = time.perf_counter()
            metrics = learner.update(rebased)
            learner.synchronize()
            elapsed = time.perf_counter() - tick
            if metrics['optimizer_steps'] != 4 * result['logical_minibatches_per_epoch']:
                raise ValueError('unexpected optimizer step count')
            result['finite_metrics'] = all(math.isfinite(metrics[key]) for key in ('policy_loss', 'value_mse', 'entropy', 'approx_kl', 'total_loss'))
            result.update(metrics=metrics, update_seconds=elapsed, **phase,
                          remaining_update_seconds=elapsed - phase['parity_seconds'],
                          final_weights_sha256=array_digest(sorted(weights(learner.network).items())))
        if retained_proof(rollout) != proof or retained_proof(rebased) != proof:
            raise ValueError('compute changed frozen input arrays')
        result['status'] = 'complete'
    except Exception as error:  # noqa: BLE001 -- emit a failed diagnostic after immutable-source checks
        result['failure'] = f'{type(error).__name__}: {error}'
    finally:
        result['source_sha256_after'] = file_hashes(paths)
        result['source_unchanged'] = before == result['source_sha256_after']
        if not result['source_unchanged']:
            result.update(status='failed', failure='immutable source changed')
        result['whole_workload_seconds'] = time.perf_counter() - started
        result['whole_command_scope'] = 'observer measurement.json wall_seconds includes interpreter startup, imports, this workload and result serialization'
    if scratch is not None:
        try:
            cleanup_empty_cache(Path(scratch) / 'sequence-inductor-cache')
        except OSError as error:
            result['cleanup_failure'] = f'{type(error).__name__}: {error}'
            if result['status'] == 'complete':
                result['failure'] = result['cleanup_failure']
            result['status'] = 'failed'
    return result


def cleanup_empty_cache(path: Path) -> None:
    # This workload never compiles kernels. Remove only its own empty runtime
    # cache directory; unexpected files stay in place and fail the command.
    if path.exists():
        path.rmdir()


def publish(output: Path, result: dict[str, Any]) -> None:
    """Publish a fully closed/fsynced JSON without replacing another result."""
    temporary: Path | None = None
    primary: BaseException | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix=f'.{output.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, allow_nan=False, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    except BaseException as error:  # Preserve primary through fallible temporary cleanup.
        primary = error
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup:
                if primary is None:
                    raise
                primary.add_note(f'benchmark result temporary cleanup failed: {cleanup}')


def main() -> int:
    args = parser().parse_args()
    output = Path(os.environ['GFB_BENCHMARK_RESULT']).resolve()
    if output.is_relative_to(args.source.resolve().parent):
        raise ValueError('benchmark result must be outside the immutable source directory')
    if output.exists(follow_symlinks=False):
        raise FileExistsError('benchmark result already exists')
    result = run(args)
    publish(output, result)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
