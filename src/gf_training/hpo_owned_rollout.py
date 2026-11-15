"""Capture-origin canonical Python encoding and direct sealed numeric6 inputs."""
from __future__ import annotations

import json
import time
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from gf_common.artifacts import ArtifactAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, digest, integer
from gf_common.hpo import checked_selection, optimizer_delta
from gf_common.hpo import selected_locations as selection_locations
from gf_common.hpo_owned_tokens import (
    CANDIDATE_ARRAY_FIELDS,
    FRAME_ARRAY_FIELDS,
    _OwnedCandidateTokens,
    _OwnedFrameTokens,
)
from gf_common.ram_bank import MAX_BANK_BYTES, ReadonlyRamBankSet
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.arrays import MAX_ARRAY_BYTES, sha256
from gf_models.hpo_config import validate_model_v4
from gf_models.sharded_arrays import (
    _write_producer_bundle,
    create_output_directory,
    write_owned_json,
)
from gf_models.structured_config import (
    aggregate_recipe,
    recorded_architecture_identity,
    recorded_token_signature,
)
from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config
from gf_training.hpo_owned_capture import OwnedHpoBatch
from gf_training.ppo import gae, normalize_advantages
from gf_training.structured_rollout import (
    _array,
    _blob,
    _hpo_semantics,
    _plan_rollout_arrays,
    _RolloutContext,
)

_TOKEN = object()


class _NumericRetirement:
    def __init__(self) -> None:
        self.retired=False
        self.error: BaseException | None = None


_QUARANTINED_STORAGE: list[tuple[Any, Any, _NumericRetirement]] = []
_DEFERRED_STORAGE: dict[int,tuple[Any,Any,_NumericRetirement]] = {}


def _retire_storage(banks: ReadonlyRamBankSet, reservation: Any, receipt: _NumericRetirement) -> None:
    try:
        banks.close()
        if reservation is not None:
            reservation.close()
        receipt.retired = True
        _DEFERRED_STORAGE.pop(id(receipt),None)
    except BufferError:
        # Known surviving exports/leases have not attempted an uncertain close.
        # Keep the actual owners, then retry only through the retirement receipt.
        _DEFERRED_STORAGE[id(receipt)]=(banks,reservation,receipt)
    except BaseException as error:  # noqa: BLE001 - retain actual storage after any uncertain finalizer failure
        _DEFERRED_STORAGE.pop(id(receipt),None)
        receipt.error = error
        _QUARANTINED_STORAGE.append((banks, reservation, receipt))


class _NumericStorage:
    def __init__(self, banks: ReadonlyRamBankSet, reservation: Any, receipt: _NumericRetirement) -> None:
        self.banks = banks
        weakref.finalize(self, _retire_storage, banks, reservation, receipt)


@dataclass(frozen=True)
class _CapturedRecord:
    bank: Any
    frames: tuple[Any, ...]
    tables: tuple[Any, ...]
    selections: tuple[Any, ...]
    action_contract: Any
    outcome: Any
    purpose: str = 'qualified_game'
    synthetic_capacity: bool = False
    end_kind: str = 'natural'
    reason: None = None


class OwnedNumericBatch:
    """One-shot learner input; only capture encoding or owned FD receive creates it."""

    def __init__(self, token: object, banks: ReadonlyRamBankSet, metadata: Json, *,
                 model: Any, training: HPOTrainingConfig, context: _RolloutContext,
                 budget: ArtifactAdmission, generation: int, metadata_reservation: Any = None) -> None:
        if token is not _TOKEN:
            raise ContractError('private captured numeric construction required')
        self.banks, self.model, self.training, self._context = banks, model, training, context
        self.input_binding = context.classification()
        self.action_config_hash = metadata['action_config_hash']
        self.policy_version = metadata['policy_version']
        self.collected_decisions = sum(len(episode['frames']) for episode in metadata['episodes'])
        self.used_decisions = metadata['decision_count']
        self.generation, self.resident_bytes = generation, banks.admitted_bytes
        self._numeric_limit = MAX_BANK_BYTES
        self.timings: dict[str,float] = {}
        self._metadata_bytes = canonical(metadata).encode()
        self._identity = digest({'generation':generation,'metadata':metadata})
        self._retirement = _NumericRetirement()
        self._storage_owner: Any = _NumericStorage(banks,metadata_reservation,self._retirement)
        self.locations = selection_locations(metadata['selection'])
        self._budget = budget
        self._state = 'SEALED'
        self._snapshot_owner: Any = None
        self._files: Json | None = None
        self._frames: tuple[Any,...] = ()
        self._tables: tuple[Any,...] = ()
        self._selections: tuple[Any,...] = ()
        self._prepare_views(metadata)

    @property
    def metadata(self) -> Json:
        return json.loads(self._metadata_bytes)

    @property
    def identity(self) -> str:
        return self._identity

    def __len__(self) -> int:
        return self.used_decisions

    def _prepare_views(self, metadata: Json) -> None:
        signature = recorded_token_signature(self.model, layout=LAYOUT)
        episode_frames, episode_tables, selections = [], [], []
        for episode in metadata['episodes']:
            frames, tables = [], []
            for item in episode['frames']:
                prefix, event = item['token_prefix'], item['event_token_prefix']
                scalars = {'frame_hash':item['frame_hash'],'recipe_signature':signature,
                    'aggregate_window':self.model.event.aggregate_window,'actor':item['actor'],
                    'episode':item['episode'],'revision':item['revision'],
                    'event_start':item['interval_start'],'event_end':item['boundary']['end']}
                keys={field:(event if field.startswith('interval_') else prefix)+'.'+field
                    for field in FRAME_ARRAY_FIELDS}
                frames.append(_OwnedFrameTokens(self.banks,scalars,keys,self._storage_owner))
                tables.append(_OwnedCandidateTokens(self.banks,
                    {'table_hash':item['table_hash'],'frame_hash':item['frame_hash'],'recipe_signature':signature},
                    {field:prefix+'.candidate.'+field for field in CANDIDATE_ARRAY_FIELDS},self._storage_owner))
            episode_frames.append(tuple(frames))
            episode_tables.append(tuple(tables))
            selections.append(tuple((value['index'],float(value['logprob']),float(value['value']))
                for value in episode['selections']))
        self._frames,self._tables,self._selections=tuple(episode_frames),tuple(episode_tables),tuple(selections)

    def endpoint(self, index: int) -> tuple[tuple[_OwnedFrameTokens, ...], _OwnedCandidateTokens]:
        if self._state == 'RETIRED':
            raise ContractError('owned numeric container is retired')
        integer(index,0,len(self)-1)
        episode,position=self.locations[index]
        return self._frames[episode][:position+1],self._tables[episode][position]

    def vector(self, index: int) -> tuple[int, float, float, float, float]:
        if self._state == 'RETIRED':
            raise ContractError('owned numeric container is retired')
        integer(index,0,len(self)-1)
        episode,position=self.locations[index]
        selected,old,value=self._selections[episode][position]
        return (selected,old,value,float(self.banks.array(f'e.{episode}.advantage')[position]),
            float(self.banks.array(f'e.{episode}.target')[position]))

    def persist(self, directory: Path, *, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
        if self._state != 'SEALED':
            raise ContractError('numeric persistence is single use')
        self._state = 'FAILED'
        started = time.perf_counter()
        # Codec scratch and metadata coexist with raw capture and sealed banks.
        scratch = 35 * min(shard_bytes, self.banks.size) + 32 * len(self._metadata_bytes) + 65536
        if self.resident_bytes + scratch > self._numeric_limit:
            raise ContractError('owned numeric persistence aggregate envelope exceeded before codec')
        with self._budget.reserve(scratch):
            create_output_directory(directory, 'rollout.json')
            metadata = self.metadata
            arrays = {name: self.banks.array(name) for name in self.banks.names}
            metadata['data'] = _write_producer_bundle(directory, 'rollout', arrays, shard_bytes=shard_bytes)
            write_owned_json(directory/'rollout.json', metadata)
            files = {**metadata['data']['files'], 'rollout.json': sha256(directory/'rollout.json')}
        self._files = files
        self._state = 'PERSISTED'
        self.timings['numeric_write_seconds'] = time.perf_counter()-started
        return {'manifest': metadata, 'files': dict(files)}

    def _begin_update(self) -> None:
        if self._state != 'PERSISTED':
            raise ContractError('owned update requires persisted batch, single use')
        self._state = 'DISPATCHED'

    def close(self) -> None:
        if self._state == 'DISPATCHED':
            raise ContractError('owned batch requires actual completion or child reap before close')
        self._frames=self._tables=self._selections=()
        self._metadata_bytes=b''
        self._storage_owner=None
        self._state='RETIRED'
        pending=_DEFERRED_STORAGE.get(id(self._retirement))
        if pending is not None:
            _retire_storage(*pending)
        if self._retirement.error is not None:
            raise ContractError('owned numeric storage retirement is quarantined') from self._retirement.error

    def require_retired(self) -> None:
        pending=_DEFERRED_STORAGE.get(id(self._retirement))
        if pending is not None:
            _retire_storage(*pending)
        if not self._retirement.retired:
            raise ContractError('owned numeric storage still has retained views/leases or uncertain cleanup')


def encode_owned_batch(batch: OwnedHpoBatch, *, owner: object, plan: Any, snapshot_owner: object,
                       model: Any, training: HPOTrainingConfig, actions: Any, rules: Any,
                       selected_locations: tuple[tuple[int, int], ...], selection: Json,
                       budget: ArtifactAdmission, numeric_budget_bytes: int = MAX_BANK_BYTES) -> OwnedNumericBatch:
    started = time.perf_counter()
    token_seconds = 0.0
    if type(batch) is not OwnedHpoBatch:
        raise ContractError('actual private captured batch required')
    batch.require_owner(owner, plan=plan, snapshot_owner=snapshot_owner)
    if batch._retention is None or budget is not batch._retention.budget:
        raise ContractError('owned encoding requires SAME actual capture Source account')
    from gf_training.hpo_owned_snapshot import OwnedModelSnapshot
    if type(snapshot_owner) is not OwnedModelSnapshot:
        from gf_training.hpo_owned_process import RemoteModelSnapshot
        if type(snapshot_owner) is not RemoteModelSnapshot:
            raise ContractError('actual exported or Wire7-owned model snapshot required')
    snapshot_budget=(snapshot_owner._budget if isinstance(snapshot_owner,OwnedModelSnapshot) else snapshot_owner._process.budget)
    if budget is not snapshot_budget:
        raise ContractError('owned capture and frozen snapshot require SAME invocation Source account')
    if snapshot_owner.generation != batch.generation:
        raise ContractError('capture generation differs from actual frozen snapshot')
    validate_model_v4(model, layout=LAYOUT)
    training = hpo_training_config(asdict(training))
    context = _RolloutContext(actions, rules)
    selection = checked_selection(selection)
    counts = tuple(len(episode.learner.decisions) for episode in batch.episodes)
    if (tuple(selection['episode_counts']) != counts or selected_locations != selection_locations(selection)
            or not selected_locations):
        raise ContractError('owned endpoint selection differs from actual complete capture')
    integer(numeric_budget_bytes, 1, MAX_BANK_BYTES)
    # Conservative token/Python planning and array temporaries before tokenization.
    temporary = (65536 * (1 + sum(counts)) + sum(
        episode.learner.history.boundary.end * 40 * 64
        + sum(4096 * len(d.table.kinds) + 128 * len(d.table.slots) + 8 * len(d.frame.payload)
            for d in episode.learner.decisions) for episode in batch.episodes))
    if temporary > numeric_budget_bytes:
        raise ContractError('owned token temporary envelope exceeds numeric budget before encoding')
    with budget.reserve(temporary):
        episodes = []
        for episode in batch.episodes:
            own = episode.learner
            integer(len(own.decisions), 0, model.temporal.max_decisions)
            integer(own.history.boundary.end, 0, model.event.max_episode)
            frames, tables = [], []
            for decision in own.decisions:
                integer(decision.frame.boundary.end - decision.frame.interval_start, 0, model.event.max_interval)
                integer(len(decision.table.kinds), 1, actions.max_candidates)
                if any(b-a > model.max_positions for a,b in zip(decision.table.offsets[:-1], decision.table.offsets[1:], strict=True)):
                    raise ContractError('owned candidate position capacity exceeded')
                token_started = time.perf_counter()
                frames.append(tokenize_frame(decision.frame, own.history,
                    aggregate_window=model.event.aggregate_window, aggregate_recipe=aggregate_recipe(model), layout=LAYOUT))
                tables.append(tokenize_table(decision.table, decision.frame, model.max_positions,
                    aggregate_window=model.event.aggregate_window, aggregate_recipe=aggregate_recipe(model), layout=LAYOUT))
                token_seconds += time.perf_counter()-token_started
            record = _CapturedRecord(own.history, tuple(d.frame for d in own.decisions),
                tuple(d.table for d in own.decisions), tuple(d.selection for d in own.decisions),
                actions.action_contract, episode.outcome)
            reward = 0 if episode.outcome == 'draw' else 1 if episode.outcome['win']['winner'] == own.viewer else -1
            advantage, target = gae([d.selection.value for d in own.decisions], reward, lam=training.gae_lambda) if frames else (np.empty(0), np.empty(0))
            episodes.append(SimpleNamespace(records=record, frames=tuple(frames), tables=tuple(tables),
                advantage=advantage, target=target.astype(np.float32)))
        normalized = normalize_advantages([episodes[e].advantage[p] for e,p in selected_locations])
        for encoded_episode in episodes:
            encoded_episode.advantage = np.zeros(len(encoded_episode.frames), np.float32)
        for (episode_index, position), value in zip(selected_locations, normalized, strict=True):
            episodes[episode_index].advantage[position] = value
        component = SimpleNamespace(_context=context, episodes=episodes, resident_bytes=0)
        def admit(size: int) -> None:
            if temporary + size > numeric_budget_bytes:
                raise ContractError('owned packing/output aggregate envelope exceeded')
        episode_metadata, planned, _, planned_bytes = _plan_rollout_arrays(component,
            shard_bytes=MAX_ARRAY_BYTES, admit=admit, poll=lambda: None, hpo=True)
        admit(2 * planned_bytes)
        metadata = {'schema_version': 6, 'kind': 'structured-rollout', **context.classification(),
            **_hpo_semantics(model, context), 'origin': 'producer-owned-v1', 'encoder': 'canonical-python',
            'snapshot_identity': snapshot_owner.identity,
            'architecture_identity': recorded_architecture_identity(model, layout=LAYOUT),
            'action_config_hash': digest(asdict(actions)),
            'policy_version': next(d.selection.policy_version for e in batch.episodes for d in e.learner.decisions),
            'decision_count': len(selected_locations), 'episodes': episode_metadata,
            'training_recipe': asdict(training), 'selection': selection,
            'expected_optimizer_step_delta': optimizer_delta(len(selected_locations), epochs=training.epochs, minibatch_size=training.minibatch_size)}
        metadata_charge = 16 * json_bound(metadata) + 65536
        backing_count = (len(planned) + 255) // 256
        # Upper bound aligned pages/descriptors before ndarray/memfd allocation.
        overhead = 1088 * len(planned) + (65536 + 4096) * backing_count
        if temporary + 2 * planned_bytes + overhead + metadata_charge > numeric_budget_bytes:
            raise ContractError('owned aggregate metadata/output envelope exceeded before arrays')
        metadata_reservation = budget.reserve(metadata_charge)
        banks = None
        try:
            arrays = {name: _blob(value) if blob else _array(value, dtype)
                for name, (value, dtype, blob) in planned.items()}
            seal_started = time.perf_counter()
            banks = ReadonlyRamBankSet.build(arrays, budget,
                max_bytes=numeric_budget_bytes-temporary-planned_bytes-metadata_charge)
            output = OwnedNumericBatch(_TOKEN, banks, metadata, model=model, training=training,
                context=context, budget=budget, generation=batch.generation, metadata_reservation=metadata_reservation)
            output.resident_bytes += metadata_charge
            output._numeric_limit = numeric_budget_bytes
            output._snapshot_owner = snapshot_owner
            output.timings = {'encode_seconds':time.perf_counter()-started,
                'tokenize_seconds':token_seconds,'seal_seconds':time.perf_counter()-seal_started}
            return output
        except BaseException:
            metadata_reservation.close()
            if banks is not None:
                banks.close()
            raise


def _validate_owned_layout(metadata: Json, banks: ReadonlyRamBankSet, *, model: Any, actions: Any) -> None:
    """External closed layout/count/finite guards; never raw semantic reconciliation."""
    from gf_common.domain import closed
    descriptors = {descriptor.name: descriptor for bank in banks.banks for descriptor in bank.descriptors}
    expected: set[str] = set()
    def take(name: str, dtype: str, shape: tuple[int, ...] | None) -> Any:
        descriptor = descriptors.get(name)
        if descriptor is None or descriptor.dtype != np.dtype(dtype).str or shape is not None and descriptor.shape != shape:
            raise ContractError('owned numeric dtype/shape/layout mismatch')
        if descriptor.nbytes > MAX_ARRAY_BYTES:
            raise ContractError('owned numeric tensor exceeds shard bound')
        expected.add(name)
        return descriptor
    episodes = metadata.get('episodes')
    if type(episodes) is not list or not episodes:
        raise ContractError('owned numeric complete episode metadata required')
    counts = []
    for episode_index, episode in enumerate(episodes):
        count = integer(len(episode['frames']), 0, model.temporal.max_decisions)
        counts.append(count)
        prefix = f'e.{episode_index}'
        if episode['end_kind'] != 'natural' or episode['reason'] is not None:
            raise ContractError('owned numeric natural episode required')
        for block in episode['raw_blocks']:
            descriptor = take(block['key'], 'uint8', None)
            if len(descriptor.shape) != 1:
                raise ContractError('owned raw event block shape mismatch')
        if len(episode['selections']) != count:
            raise ContractError('owned numeric selection metadata count mismatch')
        for position, item in enumerate(episode['frames']):
            frame, event = prefix+f'.f.{position}', prefix+f'.event-tokens.{position}'
            if item['token_prefix'] != frame or item['event_token_prefix'] != event or item['payload_key'] != frame+'.payload':
                raise ContractError('owned numeric frame key/order mismatch')
            descriptor = take(item['payload_key'], 'uint8', None)
            if len(descriptor.shape) != 1:
                raise ContractError('owned raw payload shape mismatch')
            shapes = {'state_numerics': (len(LAYOUT.state_numeric_fields),),
                'state_present': (len(LAYOUT.state_numeric_fields),),
                'state_categories': (len(LAYOUT.state_categories),),
                'state_category_present': (len(LAYOUT.state_categories),),
                'card_numerics': (18,len(LAYOUT.card_numeric_fields)),
                'card_present': (18,len(LAYOUT.card_numeric_fields)),
                'card_categories': (18,len(LAYOUT.card_categories)),
                'card_mask': (18,), 'opponent_learned': (18,), 'history': (LAYOUT.history_width,)}
            interval = integer(item['boundary']['end']-item['interval_start'],0,model.event.max_interval)
            shape: tuple[int,...]
            for field in FRAME_ARRAY_FIELDS:
                if field.startswith('interval_'):
                    width = len(LAYOUT.event_categories) if field=='interval_categories' else len(LAYOUT.event_numeric_fields)
                    shape = (interval,width) if interval else (0,)
                    key = event+'.'+field
                else:
                    shape, key = shapes[field], frame+'.'+field
                take(key, 'uint8' if 'present' in field or 'mask' in field else 'int64', shape)
            kinds = take(frame+'.candidate.kinds','int64',None)
            if len(kinds.shape) != 1:
                raise ContractError('owned candidate kinds shape mismatch')
            candidates = integer(kinds.shape[0],1,actions.max_candidates)
            for field,shape in {'targets':(candidates,), 'numerics':(candidates,4), 'present':(candidates,4),
                               'offsets':(candidates+1,)}.items():
                take(frame+'.candidate.'+field,'uint8' if field=='present' else 'int64',shape)
            slots = take(frame+'.candidate.slots','int64',None)
            if len(slots.shape) != 1 or slots.shape[0] > candidates*model.max_positions:
                raise ContractError('owned candidate slots shape/cap mismatch')
            selection = closed(episode['selections'][position], {'table_hash','index','policy_version','logprob','value'})
            integer(selection['index'],0,candidates-1)
            if selection['policy_version'] != metadata['policy_version']:
                raise ContractError('owned numeric frozen policy metadata mismatch')
        for name in ('selected','old_logprob','old_value'):
            take(prefix+'.'+name,'int64' if name=='selected' else 'float32',(count,))
        if episode['has_targets'] is not bool(count):
            raise ContractError('owned numeric target presence mismatch')
        if count:
            for name in ('advantage','target'):
                take(prefix+'.'+name,'float32',(count,))
    selection = checked_selection(metadata['selection'])
    if tuple(selection['episode_counts']) != tuple(counts) or metadata['decision_count'] != selection['used_decisions']:
        raise ContractError('owned numeric selected endpoint count mismatch')
    if set(descriptors) != expected:
        raise ContractError('owned numeric missing/extra array layout')
    for name in expected:
        array = banks.array(name)
        if array.dtype.kind == 'f' and not np.isfinite(array).all():
            raise ContractError('nonfinite owned numeric input')
        del array
