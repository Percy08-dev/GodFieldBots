"""Private numeric persistence and Torch handoff from one sealed Rust Source."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from gf_common.artifacts import json_bound
from gf_common.domain import ContractError, Json, digest, integer
from gf_common.hpo import optimizer_delta, selected_locations
from gf_common.rust_ppo_source import source_mode
from gf_models.arrays import MAX_ARRAY_BYTES, sha256
from gf_models.sharded_arrays import (
    _write_producer_bundle,
    create_output_directory,
    write_owned_json,
)
from gf_models.structured_artifacts import semantics_v3
from gf_models.structured_config import recorded_architecture_identity, recorded_token_signature
from gf_models.structured_learning import FrameStagingPlan, PackedLearningWindow, slab_plan
from gf_training.structured_rollout import _hpo_semantics, _RolloutContext


class _RustTrainingRollout:
    """Own one use of the same actual captured arrays by the writer and learner."""

    def __init__(self, source: Any) -> None:
        import gf_native

        from gf_training.rust_player_source import RustPlayerSource

        if type(source) is not RustPlayerSource or source.state != 'SEALED':
            raise ContractError('actual sealed native training Source required')
        source.require()
        registry = source.registry
        learner = registry.local.learner
        collection = source.collection
        plan = source.plan
        self._state = 'BUILDING'
        self._budget = registry.budget
        self._snapshot_owner = source.parent
        self._check = source._packing_deadline
        self._deadline = plan.execution_deadline
        self.model = learner.model
        self.training = learner.training if registry.hpo else None
        self.generation = source.parent.generation
        self._context = _RolloutContext(registry.config.actions, learner._context.rules)
        self.input_binding = self._context.classification()
        self.action_config_hash = digest(asdict(self._context.actions))
        self.collected_decisions = sum(len(episode.rows) for episode in collection.training)
        selected = selected_locations(source.selection) if registry.hpo else None
        self.locations = selected if selected is not None else tuple(
            (index, pos) for index, episode in enumerate(collection.training) for pos in range(len(episode.rows)))
        self.used_decisions = len(self.locations)
        self._episodes: tuple[Any, ...] = ()
        self._scalars: tuple[Any, ...] = ()
        self._arrays: dict[str, Any] = {}
        self._metadata: Json = {}
        self._native = None
        self._charge = None
        self._numeric_limit = (registry.config.hpo_spec['operating']['numeric_budget_bytes']
            if registry.hpo else self.model.resources.max_owned_bytes)
        mode = source_mode(registry.state['effective_config'])
        if mode is None or collection.pack_verification is None or not self.locations:
            raise ContractError('native numerics require nonempty verified training collection')
        # Existing grouping plan budget covers Python per-decision views, names,
        # and output descriptors. Rust fields already carry their own lease.
        planning = 65536 * (1 + self.collected_decisions)
        self._charge = self._budget.reserve(planning)
        try:
            self._native = gf_native._build_player_ppo_batch(
                [episode.native for episode in collection.training],
                self.training.gae_lambda if self.training is not None else .95,
                list(selected) if selected is not None else None, self._deadline)
            self.resident_bytes = self._native.resident_bytes
            self._prepare(collection.training, mode, source.selection if registry.hpo else None)
            self._state = 'SEALED'
        except BaseException:
            self.close()
            raise

    def _prepare(self, episodes: tuple[Any, ...], mode: Json, selection: Json | None) -> None:
        assert self._native is not None and self._charge is not None
        owned = self._charge.size
        layout = self._context.layout
        classification = self.input_binding
        signature = recorded_token_signature(self.model, layout=layout)
        metadata = []
        token_episodes = []
        scalars = []
        labels = set()
        for index, episode in enumerate(episodes):
            self._check()
            prefix = f'e.{index}'
            raw_blocks = []
            for start in range(0, len(episode.events), 1024):
                end = min(start + 1024, len(episode.events))
                key = f'{prefix}.raw-events.{start}'
                size = 2 + end - start - 1 + sum(len(raw) for raw in episode.events[start:end])
                # Joining existing canonical event bytes is lossless. Charge the
                # assembly copy and the immutable final backing while both live.
                self._charge.resize(owned + 2 * size)
                raw = b'[' + b','.join(episode.events[start:end]) + b']'
                owned += len(raw)
                self._charge.resize(owned)
                self._arrays[key] = np.frombuffer(raw, dtype=np.uint8)
                raw_blocks.append({'start': start, 'end': end, 'key': key})
            frames, candidates, entries, selections = [], [], [], []
            for position, row in enumerate(episode.rows):
                self._check()
                frame = row.frame
                decision = asdict(row.selection)
                labels.add(row.selection.policy_version)
                frame_prefix = f'{prefix}.f.{position}'
                event_prefix = f'{prefix}.event-tokens.{position}'
                payload_key = frame_prefix + '.payload'
                self._arrays[payload_key] = np.frombuffer(frame.payload, dtype=np.uint8)
                fields, table = self._native.row_arrays(index, position)
                for name, values in fields.items():
                    token_prefix = event_prefix if name.startswith('interval_') else frame_prefix
                    self._arrays[f'{token_prefix}.{name}'] = values
                self._arrays.update({f'{frame_prefix}.candidate.{name}': values for name, values in table.items()})
                frames.append(SimpleNamespace(**fields, frame_hash=frame.frame_hash, recipe_signature=signature,
                    aggregate_window=self.model.event.aggregate_window, actor=frame.actor, episode=frame.episode,
                    revision=frame.revision, event_start=frame.interval_start, event_end=frame.boundary.end))
                candidates.append(SimpleNamespace(**table, table_hash=row.selection.table_hash,
                    frame_hash=frame.frame_hash, recipe_signature=signature))
                entries.append({**classification, 'actor': frame.actor, 'episode': frame.episode,
                    'revision': frame.revision, 'interval_start': frame.interval_start,
                    'boundary': asdict(frame.boundary), 'interval_start_hash': frame.interval_start_hash,
                    'frame_hash': frame.frame_hash, 'table_hash': row.selection.table_hash,
                    'payload_key': payload_key, 'token_prefix': frame_prefix, 'event_token_prefix': event_prefix})
                selections.append(decision)
            episode_scalars = self._native.episode_arrays(index)
            has_targets = bool(episode.rows)
            names = ('selected', 'old_logprob', 'old_value') + (('advantage', 'target') if has_targets else ())
            self._arrays.update({f'{prefix}.{name}': episode_scalars[name] for name in names})
            scalars.append(episode_scalars)
            token_episodes.append((tuple(frames), tuple(candidates)))
            metadata.append({**classification, 'actor': episode.facts['viewer'],
                'episode': episode.facts['episode_id'], 'boundary': asdict(episode.boundary),
                'end_kind': 'natural', 'outcome': episode.outcome, 'reason': None, 'raw_blocks': raw_blocks,
                'frames': entries, 'selections': selections, 'has_targets': has_targets})
        if len(labels) != 1:
            raise ContractError('one frozen learner policy required across native trajectories')
        self._episodes, self._scalars = tuple(token_episodes), tuple(scalars)
        self._metadata = {'schema_version': 7 if selection is None else 8, 'kind': 'structured-rollout',
            **classification, 'architecture_identity': recorded_architecture_identity(self.model, layout=layout),
            'action_config_hash': self.action_config_hash, 'policy_version': labels.pop(),
            'decision_count': self.used_decisions, 'episodes': metadata, 'ppo_source': mode}
        if selection is None:
            self._metadata.update(semantics_v3(self.model, layout=layout,
                action_contract=self._context.actions.action_contract, rules=self._context.rules))
        else:
            assert self.training is not None
            self._metadata.update(_hpo_semantics(self.model, self._context),
                origin='producer-owned-v1', encoder='canonical-rust', snapshot_identity=self._snapshot_owner.identity,
                training_recipe=asdict(self.training), selection=selection,
                expected_optimizer_step_delta=optimizer_delta(self.used_decisions,
                    epochs=self.training.epochs, minibatch_size=self.training.minibatch_size))

    @property
    def metadata(self) -> Json:
        return self._metadata

    def __len__(self) -> int:
        return self.used_decisions

    def require_learning(self) -> None:
        if self._state not in ('PERSISTED', 'DISPATCHED') or self._native is None:
            raise ContractError('native rollout is absent or has already been consumed')

    def endpoint(self, index: int) -> tuple[tuple[Any, ...], Any]:
        episode, pos = self.locations[integer(index, 0, len(self) - 1)]
        frames, candidates = self._episodes[episode]
        return frames[:pos + 1], candidates[pos]

    def vector(self, index: int) -> tuple[int, float, float, float, float]:
        episode, pos = self.locations[integer(index, 0, len(self) - 1)]
        data = self._scalars[episode]
        return (int(data['selected'][pos]), float(data['old_logprob'][pos]), float(data['old_value'][pos]),
            float(data['advantage'][pos]), float(data['target'][pos]))

    def pack_fits(self, indices: tuple[int, ...], staging: FrameStagingPlan | None = None) -> bool:
        self.require_learning()
        assert self._native is not None
        locations = [self.locations[index] for index in indices]
        if staging is None:
            parts = [(locations, 'all')]
        else:
            parts = [(locations, 'candidates')]
            parts.extend(([locations[index] for index in chunk], 'frames') for chunk in staging.suffix_chunks)
        return self._native._pack_fits(parts, self._deadline)

    def require_singleton_packing(self, index: int, staging: FrameStagingPlan | None = None) -> None:
        if not self.pack_fits((index,), staging):
            raise ContractError('native PPO singleton padding exceeds storage/numeric reservation')

    def pack(self, network: Any, indices: tuple[int, ...], part: str = 'all') -> PackedLearningWindow:
        self.require_learning()
        assert self._native is not None
        slabs, layouts, lengths = self._native.pack([self.locations[index] for index in indices], self._deadline, part)
        import torch
        shapes = {name: (tuple(shape), slabs[index].itemsize) for name, (index, _, shape) in
            sorted(layouts.items(), key=lambda pair: pair[1][:2])}
        segments = slab_plan(shapes, network.config.resources.max_tensor_bytes, split=part != 'all')
        cpu, device, operands = [], [], {}
        retain_host = part != 'frames' or torch.device(network.device).type == 'cpu'
        for segment in segments:
            index, start, _ = layouts[segment[0]]
            size = sum(math.prod(shapes[name][0]) for name in segment)
            host = torch.from_numpy(slabs[index][start:start + size])
            # A staged CUDA frame has no remaining host consumer after this
            # blocking transfer. The candidate host slab stays through backward.
            target = host.to(network.device, non_blocking=False)
            if retain_host:
                cpu.append(host)
            device.append(target)
            for name in segment:
                _, offset, shape = layouts[name]
                at = offset - start
                operands[name] = target[at:at + math.prod(shape)].view(shape)
        return PackedLearningWindow(operands, tuple(device), tuple(cpu), tuple(lengths), len(lengths), len(device))

    def persist(self, directory: Path) -> tuple[Json, Json]:
        if self._state != 'SEALED':
            raise ContractError('native rollout persistence is single use')
        self._state = 'FAILED'
        # Shared array codec admission mirrors the established HPO producer.
        metadata_bytes = 32 * json_bound(self._metadata) + 65536
        total = sum(array.nbytes for array in self._arrays.values())
        max_array = max(array.nbytes for array in self._arrays.values())
        remaining = min(self._budget.limit - self._budget.used,
            self._numeric_limit - self.resident_bytes)
        shard_bytes = min(MAX_ARRAY_BYTES, (remaining - metadata_bytes) // 35)
        if shard_bytes < max(1, max_array):
            raise ContractError('native numeric persistence cannot fit one array within its available budget')
        with self._budget.reserve(35 * min(shard_bytes, total) + metadata_bytes):
            self._check()
            create_output_directory(directory, 'rollout.json')
            self._metadata['data'] = _write_producer_bundle(directory, 'rollout', self._arrays,
                shard_bytes=shard_bytes, _check=self._check)
            write_owned_json(directory / 'rollout.json', self._metadata)
            files = {**self._metadata['data']['files'], 'rollout.json': sha256(directory / 'rollout.json')}
            self._check()
        self._arrays.clear()
        self._state = 'PERSISTED'
        return self.metadata, files

    def _begin_update(self) -> None:
        if self._state != 'PERSISTED':
            raise ContractError('native numeric update is single use')
        self._state = 'DISPATCHED'

    def close(self) -> None:
        if self._state == 'DISPATCHED':
            raise ContractError('native numeric update must release its actual consumer')
        self._metadata = {}
        self._episodes = self._scalars = ()
        self._arrays.clear()
        self._native = None
        if self._charge is not None:
            self._charge.close()
            self._charge = None
        self._state = 'RETIRED'

    def require_retired(self) -> None:
        if self._state != 'RETIRED':
            raise ContractError('native numeric batch is still held')
