"""Version-two token banks and bounded candidate shards, without window copies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, closed, integer
from gf_models.arrays import MAX_ARRAY_BYTES, load_arrays, save_arrays, sha256
from gf_models.config import SequenceModelConfig
from gf_models.encoder import feature_names, signature
from gf_models.features import critic_indices
from gf_models.sequence import MAX_HISTORY, SEQUENCE_RECIPE
from gf_training.journal import atomic_json
from gf_training.native_learning import BulkRollout
from gf_training.q_data import check_deadline


@dataclass(frozen=True)
class SequenceShard:
    actor: Any
    offsets: Any
    bank: Any
    position: Any
    selected: Any
    old_logprob: Any
    old_value: Any
    advantage: Any
    target: Any

    def arrays(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


class SequenceRollout:
    def __init__(self, banks: tuple[Any, ...], shards: tuple[SequenceShard, ...]) -> None:
        self.banks, self.shards = banks, shards
        self.offsets = np.cumsum([0] + [len(shard.selected) for shard in shards], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, index: int) -> tuple[SequenceShard, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self.offsets, index, side='right')) - 1
        return self.shards[shard], index - int(self.offsets[shard])

    def endpoint(self, index: int) -> tuple[Any, Any]:
        shard, row = self.locate(index)
        bank, position = int(shard.bank[row]), int(shard.position[row])
        return self.banks[bank][:position + 1], shard.actor[shard.offsets[row]:shard.offsets[row + 1]]

    def vector(self, indices: list[int], key: str, dtype: Any = np.float32) -> Any:
        # At most one logical minibatch; never expand the stored history windows.
        return np.asarray([getattr(shard, key)[row] for shard, row in (self.locate(i) for i in indices)], dtype=dtype)


def _semantics(model: SequenceModelConfig) -> Json:
    return {'schema_version': 2, 'sequence_recipe': SEQUENCE_RECIPE, 'max_history': MAX_HISTORY,
            'encoder_signature': signature(model.encoder), 'actor_width': len(feature_names(model.encoder)),
            'state_width': len(critic_indices(model.encoder))}


def write_rollout(directory: Path, rollout: BulkRollout, model: SequenceModelConfig, *, deadline: float = float('inf')) -> Json:
    directory.mkdir(parents=True, exist_ok=False)
    banks: list[Json] = []
    shards: list[Json] = []
    count = 0
    for chunk in rollout.chunks:
        for episode in range(len(chunk.episode_offsets) - 1):
            check_deadline(deadline)
            start, stop = map(int, chunk.episode_offsets[episode:episode + 2])
            length = stop - start
            integer(length, 0, MAX_HISTORY)
            bank_index = len(banks)
            name = f'bank-{bank_index:06}.npz'
            metadata = save_arrays(directory / name, {'tokens': np.ascontiguousarray(chunk.state[start:stop], dtype=np.float32)})
            banks.append({'file': name, 'metadata': metadata, 'length': length})
            first = start
            while first < stop:
                last = first
                candidate_start = int(chunk.offsets[first])
                # 8*(N+1) offsets + 2 int64 and 5 scalar arrays per endpoint.
                while last < stop:
                    candidate_stop = int(chunk.offsets[last + 1])
                    size = (candidate_stop - candidate_start) * chunk.actor.shape[1] * 4 + (last + 1 - first) * 48 + 8
                    if size > MAX_ARRAY_BYTES:
                        break
                    last += 1
                if first == last:
                    raise ContractError('single sequence decision exceeds numeric shard bound')
                arrays = {'actor': np.ascontiguousarray(chunk.actor[candidate_start:int(chunk.offsets[last])], dtype=np.float32),
                          'offsets': chunk.offsets[first:last + 1] - candidate_start,
                          'bank': np.full(last - first, bank_index, dtype=np.int64),
                          'position': np.arange(first - start, last - start, dtype=np.int64),
                          'selected': chunk.selected[first:last].astype(np.int64),
                          **{key: getattr(chunk, key)[first:last].astype(np.float32)
                             for key in ('old_logprob', 'old_value', 'advantage', 'target')}}
                name = f'shard-{len(shards):06}.npz'
                check_deadline(deadline)
                shards.append({'file': name, 'metadata': save_arrays(directory / name, arrays),
                               'bank_sha256': metadata['sha256']})
                count += last - first
                first = last
    if not count:
        raise ContractError('sequence rollout requires accepted decisions')
    manifest = {**_semantics(model), 'decision_count': count, 'banks': banks, 'shards': shards}
    check_deadline(deadline)
    atomic_json(directory / 'rollout.json', manifest)
    check_deadline(deadline)
    return manifest


def read_rollout(directory: Path, expected_hash: str, model: SequenceModelConfig) -> SequenceRollout:
    if sha256(directory / 'rollout.json') != expected_hash:
        raise ContractError('sequence rollout digest mismatch')
    manifest = json.loads((directory / 'rollout.json').read_text())
    closed(manifest, set(_semantics(model)) | {'decision_count', 'banks', 'shards'})
    if any(manifest[key] != value for key, value in _semantics(model).items()):
        raise ContractError('sequence rollout recipe/signature mismatch')
    integer(manifest['decision_count'], 1)
    banks = []
    for index, bank in enumerate(manifest['banks']):
        closed(bank, {'file', 'metadata', 'length'})
        if bank['file'] != f'bank-{index:06}.npz':
            raise ContractError('unsafe or reordered sequence bank path')
        integer(bank['length'], 0, MAX_HISTORY)
        arrays = load_arrays(directory / bank['file'], bank['metadata'])
        if set(arrays) != {'tokens'}:
            raise ContractError('sequence bank keys mismatch')
        tokens = arrays['tokens']
        if tokens.dtype != np.float32 or tokens.shape != (bank['length'], manifest['state_width']):
            raise ContractError('sequence bank dimensions mismatch')
        tokens.flags.writeable = False
        banks.append(tokens)
    shards = []
    seen = np.zeros(len(banks), dtype=np.int64)
    previous_bank = 0
    for index, shard in enumerate(manifest['shards']):
        closed(shard, {'file', 'metadata', 'bank_sha256'})
        if shard['file'] != f'shard-{index:06}.npz':
            raise ContractError('unsafe or reordered sequence shard path')
        arrays = load_arrays(directory / shard['file'], shard['metadata'])
        if set(arrays) != set(SequenceShard.__dataclass_fields__):
            raise ContractError('sequence shard keys mismatch')
        n = len(arrays['selected'])
        for key in ('bank', 'position', 'selected'):
            if arrays[key].dtype != np.int64 or arrays[key].shape != (n,):
                raise ContractError('sequence endpoint reference mismatch')
        for key in ('old_logprob', 'old_value', 'advantage', 'target'):
            if arrays[key].dtype != np.float32 or arrays[key].shape != (n,):
                raise ContractError('sequence scalar array mismatch')
        actor, offsets = arrays['actor'], arrays['offsets']
        if (not n or actor.dtype != np.float32 or actor.ndim != 2 or actor.shape[1] != manifest['actor_width']
                or offsets.dtype != np.int64 or offsets.shape != (n + 1,) or offsets[0] != 0
                or offsets[-1] != len(actor) or np.any(np.diff(offsets) <= 0)):
            raise ContractError('sequence candidate dimensions/offsets mismatch')
        if np.any(arrays['selected'] < 0) or np.any(arrays['selected'] >= np.diff(offsets)):
            raise ContractError('invalid sequence selected index')
        bank = int(arrays['bank'][0])
        if not previous_bank <= bank < len(banks) or np.any(arrays['bank'] != bank):
            raise ContractError('sequence bank order/reference mismatch')
        if (shard['bank_sha256'] != manifest['banks'][bank]['metadata']['sha256']
                or not np.array_equal(arrays['position'], np.arange(seen[bank], seen[bank] + n))
                or seen[bank] + n > len(banks[bank])):
            raise ContractError('sequence endpoint ordering/bank hash mismatch')
        seen[bank] += n
        previous_bank = bank
        for value in arrays.values():
            value.flags.writeable = False
        shards.append(SequenceShard(**arrays))
    if any(seen[i] != len(bank) for i, bank in enumerate(banks)) or int(seen.sum()) != manifest['decision_count']:
        raise ContractError('sequence rollout complete endpoint count mismatch')
    return SequenceRollout(tuple(banks), tuple(shards))
