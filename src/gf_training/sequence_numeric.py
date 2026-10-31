"""Sequence endpoint PPO with current-weight detached prefixes and all-batch parity."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, integer
from gf_models.arrays import MAX_ARRAY_BYTES, load_arrays
from gf_models.config import SequenceModelConfig, TransformerModelConfig
from gf_models.sequence import (
    SEQUENCE_RECIPE,
    inspect_artifact,
    make_network,
    model_backend,
    save_artifact,
)
from gf_models.torch_runtime import hardware, restore_weights, torch_runtime
from gf_training.ppo_math import losses
from gf_training.ppo_state import restore_resume, save_resume
from gf_training.sequence_rollout import SequenceRollout


def _admitted(batch: int, length: int, candidates: int, state_width: int, actor_width: int,
              retained_cache_bytes: int, *, slots: int, tokens: int, input_bytes: int,
              attention_multiplier: int, attention_slots: int) -> bool:
    # Reserve CPU packing, learner copy and parity reference input simultaneously,
    # plus retained sequential contexts, output/masks, hidden and prefix state.
    # Attention score capacity is separate from numeric input/retained KV.
    score_slots = attention_multiplier * batch * length * length
    numeric = (3 * batch * (length * state_width + candidates * actor_width) * 4
               + batch * candidates * 32 + batch * length * 128 * 4
               + batch * 128 * 4 * 3 + retained_cache_bytes)
    if attention_multiplier:
        # Prefix KV and concatenated prefix/suffix KV can coexist for all
        # layers. Reserve both, both causal-mask temporaries and position axes.
        numeric += 8 * batch * length * 128 * 4 + 2 * batch * length * length + 16 * batch * length
    return (batch * candidates <= slots and batch * length <= tokens and numeric <= input_bytes
            and score_slots <= attention_slots)


def microbatches(indices: list[int], rollout: SequenceRollout, sequence_length: int, *,
                 slots: int = 4096, tokens: int = 32768, input_bytes: int = MAX_ARRAY_BYTES,
                 retained_cache_bytes: int = 0, attention_multiplier: int = 0,
                 attention_slots: int = 8388608) -> list[list[int]]:
    integer(sequence_length, 1, 1024)
    # The caller selects <=128 logical loss endpoints first. Stable grouping is
    # strictly local; no epoch-wide reorder or change to endpoint loss weights.
    if len(indices) > 128:
        raise ContractError('sequence logical minibatch exceeds 128 endpoints')
    grouped: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        history, _ = rollout.endpoint(index)
        suffix = min(len(history), sequence_length)
        grouped.setdefault((len(history) - suffix, suffix), []).append(index)
    output = []
    for (prefix, suffix), group in grouped.items():
        current: list[int] = []
        maximum = 0
        for index in group:
            history, actor = rollout.endpoint(index)
            count = len(actor)
            fits = _admitted(len(current) + 1, prefix + suffix, max(maximum, count), history.shape[1], actor.shape[1],
                             retained_cache_bytes, slots=slots, tokens=tokens, input_bytes=input_bytes,
                             attention_multiplier=attention_multiplier, attention_slots=attention_slots)
            if current and not fits:
                output.append(current)
                current, maximum = [], 0
            if not _admitted(1, prefix + suffix, count, history.shape[1], actor.shape[1], retained_cache_bytes,
                             slots=slots, tokens=tokens, input_bytes=input_bytes,
                             attention_multiplier=attention_multiplier, attention_slots=attention_slots):
                raise ContractError('single sequence endpoint exceeds microbatch resource limits')
            current.append(index)
            maximum = max(maximum, count)
        if current:
            output.append(current)
    return output


def packed(rollout: SequenceRollout, indices: list[int], device: str) -> tuple[Any, ...]:
    import torch
    first_tokens, first_actor = rollout.endpoint(indices[0])
    count = max(len(rollout.endpoint(index)[1]) for index in indices)
    tokens = np.empty((len(indices), len(first_tokens), first_tokens.shape[1]), dtype=np.float32)
    actor = np.zeros((len(indices), count, first_actor.shape[1]), dtype=np.float32)
    mask = np.zeros((len(indices), count), dtype=np.bool_)
    for row, index in enumerate(indices):
        history, candidates = rollout.endpoint(index)
        if len(history) != len(first_tokens):
            raise ContractError('sequence packing requires equal endpoint boundaries')
        tokens[row] = history
        actor[row, :len(candidates)] = candidates
        mask[row, :len(candidates)] = True
    return tuple(torch.from_numpy(value).to(device) for value in (tokens, actor, mask))


def _cached_contexts(cpu: Any, rollout: SequenceRollout) -> Any:
    """Frozen CPU incremental reference, linear storage and no history copies."""
    import torch
    contexts = np.empty((len(rollout), 128), dtype=np.float32)
    row = 0
    with torch.no_grad():
        cache = cpu.create_cache()
        for bank in rollout.banks:
            cache.reset()
            # One bounded bank copy because source banks are readonly NumPy.
            tokens = torch.from_numpy(bank.copy())
            for position in range(len(bank)):
                output = cpu.step(tokens[position:position + 1], cache)
                contexts[row] = output[0].numpy()
                row += 1
    return contexts


class Learner:
    def __init__(self, model: SequenceModelConfig, seed: int, device: str, sequence_length: int = 32) -> None:
        integer(sequence_length, 1, 1024)
        self.torch = torch_runtime(device)
        self.torch.random.default_generator.manual_seed(seed)
        if device != 'cpu':
            self.torch.cuda.manual_seed_all(seed)
        self.model, self.device, self.sequence_length = model, device, sequence_length
        self.network = make_network(model, seed).to(device)
        self.backend = model_backend(model)
        self.attention_multiplier = 8 if isinstance(model, TransformerModelConfig) else 0
        self.optimizer = self.torch.optim.Adam(self.network.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8)
        self.shuffle = np.random.Generator(np.random.PCG64(seed))
        self.steps = 0

    def hardware(self) -> Json:
        return {**hardware(self.device), 'sequence_backend': self.backend}

    def synchronize(self) -> None:
        if self.device != 'cpu':
            self.torch.cuda.synchronize(self.device)

    def parity(self, rollout: SequenceRollout) -> Json:
        torch = self.torch
        cpu = make_network(self.model, 0)
        cpu.load_state_dict({key: value.detach().cpu() for key, value in self.network.state_dict().items()})
        cache_bytes = len(rollout) * 128 * 4 + cpu.cache_storage_bytes
        if cache_bytes + max(bank.nbytes for bank in rollout.banks) > MAX_ARRAY_BYTES:
            raise ContractError('sequence parity cache/input resource limit exceeded')
        # Admission happens before allocating sequential-reference storage.
        for start in range(0, len(rollout), 128):
            microbatches(list(range(start, min(start + 128, len(rollout)))), rollout,
                         self.sequence_length, retained_cache_bytes=cache_bytes, attention_multiplier=self.attention_multiplier)
        contexts = _cached_contexts(cpu, rollout)
        maximum = 0.
        for indices in (indices for start in range(0, len(rollout), 128)
                        for indices in microbatches(list(range(start, min(start + 128, len(rollout)))), rollout,
                                                    self.sequence_length, retained_cache_bytes=cache_bytes, attention_multiplier=self.attention_multiplier)):
            tokens, actor, mask = packed(rollout, indices, 'cpu')
            with torch.no_grad():
                reference_logs, reference_values = cpu(tokens, actor, mask)
                cached_logs, cached_values = cpu.heads(torch.from_numpy(contexts[indices]), tokens[:, -1], actor, mask)
                device_tokens, device_actor, device_mask = tokens.to(self.device), actor.to(self.device), mask.to(self.device)
                device_logs, device_values = self.network(device_tokens, device_actor, device_mask)
            # Mandatory actual learner-device split, including on CPU. Keep grad
            # enabled through the suffix and drop this graph before proceeding.
            prefix = max(0, tokens.shape[1] - self.sequence_length)
            split_logs, split_values = self.network.split(device_tokens, device_actor, device_mask, prefix)
            selected = torch.from_numpy(rollout.vector(indices, 'selected', np.int64))
            old = torch.from_numpy(rollout.vector(indices, 'old_logprob'))
            old_value = torch.from_numpy(rollout.vector(indices, 'old_value'))
            checks = ((reference_logs.gather(1, selected[:, None]).squeeze(1), old), (reference_values, old_value),
                      (reference_logs[mask], cached_logs[mask]), (reference_values, cached_values),
                      (reference_logs[mask], device_logs.detach().cpu()[mask]), (reference_values, device_values.detach().cpu()),
                      (reference_logs[mask], split_logs.detach().cpu()[mask]), (reference_values, split_values.detach().cpu()))
            for actual, expected in checks:
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
                    raise ContractError('frozen saved/cache/full/device/split parity failed before update')
                maximum = max(maximum, float((actual - expected).abs().max()))
            del split_logs, split_values, checks
        return {'atol': 1e-5, 'rtol': 1e-5, 'max_abs_error': maximum, 'decisions': len(rollout),
                'checks': ['saved', 'cpu-cache', 'cpu-full', 'learner-full', 'learner-split'],
                'sequence_length': self.sequence_length}

    def update(self, rollout: SequenceRollout) -> Json:
        torch = self.torch
        if self.device != 'cpu':
            torch.cuda.reset_peak_memory_stats(self.device)
        parity = self.parity(rollout)
        self.synchronize()
        parity_allocated = torch.cuda.max_memory_allocated(self.device) if self.device != 'cpu' else 0
        parity_reserved = torch.cuda.max_memory_reserved(self.device) if self.device != 'cpu' else 0
        if self.device != 'cpu':
            torch.cuda.reset_peak_memory_stats(self.device)
        totals: dict[str, float] = {}
        denominator = 0
        self.network.train()
        for _ in range(4):
            order = self.shuffle.permutation(len(rollout)).tolist()
            for start in range(0, len(order), 128):
                logical = order[start:start + 128]
                self.optimizer.zero_grad(set_to_none=True)
                for indices in microbatches(logical, rollout, self.sequence_length, attention_multiplier=self.attention_multiplier):
                    tokens, actor, mask = packed(rollout, indices, self.device)
                    logs, values = self.network.split(tokens, actor, mask, max(0, tokens.shape[1] - self.sequence_length))
                    selected = torch.from_numpy(rollout.vector(indices, 'selected', np.int64)).to(self.device)
                    vectors = [torch.from_numpy(rollout.vector(indices, key)).to(self.device)
                               for key in ('old_logprob', 'advantage', 'target')]
                    loss, measured = losses(logs, values, mask, selected, *vectors)
                    (loss * (len(indices) / len(logical))).backward()
                    for key, value in measured.items():
                        totals[key] = totals.get(key, 0.) + value * len(indices)
                    denominator += len(indices)
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), .5, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ContractError('nonfinite sequence PPO gradients')
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        result = {key: value / denominator for key, value in totals.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise ContractError('nonfinite sequence PPO metrics')
        return {**result, 'parity': parity, 'optimizer_steps': self.steps, 'sequence_length': self.sequence_length,
                'cuda_memory_scope': 'phase high-water marks include resident model/optimizer and allocator caches; CPU values are zero',
                'cuda_parity_peak_allocated_bytes': parity_allocated, 'cuda_parity_peak_reserved_bytes': parity_reserved,
                'cuda_peak_scope': 'optimizer update after parity',
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device != 'cpu' else 0,
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device != 'cpu' else 0}

    def _resume_identity(self) -> Json:
        return {'model_id': self.model.model_id, 'sequence_recipe': SEQUENCE_RECIPE,
                'sequence_length': self.sequence_length, 'backend': self.backend}

    def save(self, directory: Path, action_hash: str) -> Json:
        self.synchronize()
        inference = save_artifact(self.network, directory, self.model, action_hash)
        save_resume(directory, self.network, self.optimizer, self.shuffle, self.steps, self.device,
                    schema_version=2, identity=self._resume_identity())
        restored = load_arrays(directory / 'weights.npz', inference['weights'])
        for key, value in self.network.state_dict().items():
            if not np.array_equal(value.detach().cpu().numpy(), restored[key]):
                raise ContractError('sequence CPU checkpoint roundtrip failed')
        return {'weights': inference['weights']['sha256'], 'optimizer_steps': self.steps}

    def restore(self, directory: Path) -> None:
        inference = inspect_artifact(directory, model=self.model)
        self.steps = restore_resume(directory, self.network, self.optimizer, self.shuffle, self.device,
                                    schema_version=2, identity=self._resume_identity())
        restore_weights(self.network, load_arrays(directory / 'weights.npz', inference['weights']))
