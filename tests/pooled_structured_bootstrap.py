"""Test-only epoch transport driving the original protocol2 main per episode."""
from __future__ import annotations

import importlib
import io
import random
import sys
from pathlib import Path
from types import SimpleNamespace

from gf_common.domain import ContractError, closed, digest, integer
from gf_simulator_client.framing import MAX_FRAME_BYTES, read_frame, write_frame

POOL_VERSION = 'pytest-cpu-participant-v1'


def envelope(operation, viewer, epoch, **fields):
    return {'pytest_pool': POOL_VERSION, 'operation': operation, 'viewer': viewer, 'epoch': epoch, **fields}


def check_envelope(value, operation, viewer, epoch, fields=()):
    closed(value, {'pytest_pool', 'operation', 'viewer', 'epoch', *fields})
    integer(value['viewer'], 1, 2)
    integer(value['epoch'], 1, 2**31 - 1)
    if (value['pytest_pool'] != POOL_VERSION or value['operation'] != operation
            or value['viewer'] != viewer or value['epoch'] != epoch):
        raise ContractError('pytest pool viewer/epoch/operation mismatch')
    return value


class PolicyCache:
    """One exact identity; ordinary artifact metadata and hashes checked each loan."""

    def __init__(self):
        self.identity = None
        self.policy = None
        self.loads = self.reuses = 0

    def factory(self, initial, original):
        identity = digest({key: value for key, value in initial.items() if key != 'sampling_seed'})
        def create(model_id, options, action_config_hash, **classification):
            if identity != self.identity:
                self.policy = self.identity = None
            if self.policy is not None:
                if options is not None:
                    from gf_models.structured_artifacts import inspect_inference
                    from gf_models.structured_config import InferenceBindingV2, model_config_v2
                    inspect_inference(Path(options['checkpoint']), expected_digest=options['expected_manifest_digest'],
                        model=model_config_v2(options['model']), binding=InferenceBindingV2(**options['inference']),
                        action_config_hash=action_config_hash, **classification)
                self.reuses += 1
            else:
                self.policy = original(model_id, options, action_config_hash, **classification)
                self.identity = identity
                self.loads += 1
            # Reset both at the handoff and in original main with the episode seed.
            self.reset(0)
            return self.policy
        return create

    def verify(self, seed):
        from gf_models.structured_baselines import StructuredBaseline
        policy = self.policy
        if isinstance(policy, StructuredBaseline):
            if policy.rng.getstate() != random.Random(seed).getstate():
                raise ContractError('pytest baseline RNG reset failed')
            return
        from gf_models.structured_policy import StructuredPolicy
        if not isinstance(policy, StructuredPolicy) or policy.binding.device != 'cpu':
            raise ContractError('pytest pool admits only CPU structured policies')
        cache = policy.cache
        if ((cache.position, cache.owner, cache.revision, cache.event_end) != (0, None, -1, 0)
                or policy.last_timing or not policy.torch.equal(policy.rng.get_state(),
                    policy.torch.Generator(device='cpu').manual_seed(seed).get_state())):
            raise ContractError('pytest neural policy RNG/temporal reset failed')
        for buffer in (cache.keys, cache.values, cache.past):
            if buffer is not None and bool(buffer.count_nonzero()):
                raise ContractError('pytest temporal storage reset failed')

    def reset(self, seed):
        if self.policy is None:
            raise ContractError('pytest reset requires initialized policy')
        self.policy.reset(seed)
        self.verify(seed)


class EpisodeInput:
    """Present unchanged framed packets; only a validated end yields clean EOF."""

    def __init__(self, source, initial, viewer, epoch):
        self.source, self.viewer, self.epoch = source, viewer, epoch
        self.ended = False
        self.pending = self._packet(initial)

    @staticmethod
    def _packet(payload):
        framed = io.BytesIO()
        write_frame(framed, payload)
        framed.seek(0)
        return framed

    def read(self, size):
        if self.pending.tell() == len(self.pending.getbuffer()):
            if self.ended:
                return b''
            value = read_frame(self.source)  # Unexpected/truncated EOF never sets ended.
            if value.get('operation') == 'end':
                check_envelope(value, 'end', self.viewer, self.epoch)
                self.ended = True
                return b''
            check_envelope(value, 'select', self.viewer, self.epoch, ('payload',))
            payload = value['payload']
            if (type(payload) is not dict or payload.get('frame', {}).get('actor') != self.viewer
                    or payload.get('delta', {}).get('actor') != self.viewer):
                raise ContractError('pytest pool cannot consume opposite viewer frame/delta')
            self.pending = self._packet(payload)
        return self.pending.read(size)

    def drained(self):
        return self.ended and self.pending.tell() == len(self.pending.getbuffer())


class EpisodeOutput:
    def __init__(self, destination, viewer, epoch, cache, seed):
        self.destination, self.viewer, self.epoch = destination, viewer, epoch
        self.cache, self.seed = cache, seed
        self.pending = bytearray()
        self.failed = self.ready = False

    def write(self, data):
        self.pending.extend(data)
        if len(self.pending) >= 8:
            size = int.from_bytes(self.pending[:8], 'big')
            if not 0 < size <= MAX_FRAME_BYTES or len(self.pending) > size + 8:
                raise ContractError('pytest output framing violation')
            if len(self.pending) == size + 8:
                value = read_frame(io.BytesIO(self.pending))
                self.pending.clear()
                self.failed |= 'worker_error' in value
                if value.get('kind') == 'ready':
                    self.cache.verify(self.seed)
                    self.ready = True
                write_frame(self.destination, envelope('reply', self.viewer, self.epoch, payload=value))
        return len(data)

    def flush(self):
        self.destination.flush()


def main():
    # Capture the fixed viewer before the original bootstrap clears argv/env.
    if len(sys.argv) != 2 or sys.argv[1] not in ('1', '2'):
        raise ContractError('fixed pytest pool viewer required')
    viewer = int(sys.argv[1])
    from gf_simulator_client import structured_bootstrap as original_bootstrap
    source, destination = sys.stdin.buffer, sys.stdout.buffer
    cache, epoch = PolicyCache(), 0
    while True:
        try:
            begin = read_frame(source)
        except EOFError:
            return
        epoch += 1
        check_envelope(begin, 'begin', viewer, epoch, ('payload',))
        initial = begin['payload']
        if initial.get('model_id') not in ('random', 'heuristic', 'mlp', 'gru', 'transformer'):
            raise ContractError('pytest CPU pool model is unsupported')
        options = initial.get('policy_options')
        if options is not None and (options.get('inference', {}).get('device') != 'cpu'
                                    or options.get('cuda_visible_devices') != ''):
            raise ContractError('pytest pool requires CPU and empty CUDA visibility')
        descriptor = closed(initial['factory'], {'module', 'name'})
        module = importlib.import_module(descriptor['module'])
        original_factory = getattr(module, descriptor['name'])
        episode_input = EpisodeInput(source, initial, viewer, epoch)
        episode_output = EpisodeOutput(destination, viewer, epoch, cache, initial['sampling_seed'])
        old_stdin, old_stdout = sys.stdin, sys.stdout
        try:
            setattr(module, descriptor['name'], cache.factory(initial, original_factory))
            sys.stdin = SimpleNamespace(buffer=episode_input)
            sys.stdout = SimpleNamespace(buffer=episode_output)
            original_bootstrap.main()
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
            setattr(module, descriptor['name'], original_factory)
        if not episode_input.drained() or episode_output.failed or episode_output.pending or not episode_output.ready:
            raise ContractError('pytest episode ended without healthy explicit reset boundary')
        cache.reset(0)
        write_frame(destination, envelope('reset', viewer, epoch, loads=cache.loads, reuses=cache.reuses))


if __name__ == '__main__':
    main()
