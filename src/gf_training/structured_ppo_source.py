"""Frozen-plan original Source/numeric correspondence for engineering components.

No live parent/lock authorization, trainer loop or qualified Source authority.
The composing coordinator supplies its separately validated original plan and
generation inference declaration. Public real-I/O gates remain independent.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    closed,
    digest,
    integer,
)
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.schedule import serialize_schedule
from gf_common.structured import ActionConfigV2
from gf_models.structured_binding import (
    check_input_binding,
    recorded_options_v2,
    recorded_policy_label,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    RecordedModelConfigV2,
    recorded_token_signature,
)
from gf_models.structured_sealed import SEMANTIC_KEYS, checksum, validate_recorded_semantics
from gf_training.collection import CollectionPlan
from gf_training.structured_rollout import (
    EpisodeRecords,
    StructuredRollout,
    _owned_bytes,
    _predicted_token_bytes,
    _read_recorded_rollout,
    _write_rollout,
)
from gf_training.structured_source import (
    _component_source_snapshot,
    _SourceSnapshot,
    component_episode_records,
)


def _frozen_batch(snapshot: _SourceSnapshot, plan: CollectionPlan, actions: ActionConfigV2,
                  model: RecordedModelConfigV2, learner_label: str, inference_digest: str) -> tuple[EpisodeRecords, ...]:
    checksum(inference_digest)
    if snapshot.collection.directory != plan.output.absolute():
        raise ContractError('original component collection differs from frozen output')
    manifest = snapshot.collection.manifest
    link = closed(plan.linkage, {'parent_run_id', 'parent_run_path', 'batch_id', 'collection_generation', 'learner_seats'})
    integer(link['collection_generation'], 0, 999)
    if any(type(link[key]) is not str or not link[key] for key in ('parent_run_id', 'parent_run_path', 'batch_id')):
        raise ContractError('frozen parent/batch identities required')
    expected_schedule = serialize_schedule(plan.groups, 4)
    specs = [spec for group in plan.groups for spec in group.episodes]
    seats = {spec.episode_id: integer(spec.learner_viewer, 1, 2) for spec in specs}
    if len(specs) != 16 or len(seats) != 16 or seats != link['learner_seats']:
        raise ContractError('frozen component batch requires all16 original roles')
    for spec in specs:
        if spec.policies[seats[spec.episode_id] - 1] != learner_label:
            raise ContractError('frozen learner label/explicit role mismatch')
    if (manifest['schema_version'] != 4 or manifest['collection_kind'] != 'ppo_batch'
            or canonical(manifest['schedule']) != canonical(expected_schedule)
            or canonical(manifest['policy_bindings']) != canonical({key: asdict(value) for key, value in plan.bindings.items()})
            or canonical(manifest['ppo']) != canonical(link)):
        raise ContractError('original collection differs from frozen plan/parent/bindings')
    if learner_label not in plan.bindings or plan.bindings[learner_label].model_id != model.model_id:
        raise ContractError('original learner model identity mismatch')
    options = recorded_options_v2(plan.bindings[learner_label].policy_options,
        action_contract=actions.action_contract, purpose='component', synthetic_capacity=False,
        token_signature=recorded_token_signature(model))
    binding = InferenceBindingV2(**options['inference'])
    if (canonical(options['model']) != canonical(asdict(model))
            or options['expected_manifest_digest'] != inference_digest
            or learner_label != recorded_policy_label(model, binding, inference_digest,
                                              cuda_visible_devices=options['cuda_visible_devices'])):
        raise ContractError('original learner differs from frozen generation inference')
    records = component_episode_records(snapshot.collection, learner_seats=seats)
    if any(item.model_id != model.model_id for item in snapshot.collection.episodes
           if seats[item.episode_id] == item.viewer):
        raise ContractError('original learner participant model mismatch')
    return records


@dataclass(frozen=True)
class _ComponentBatchSource:
    _snapshot: _SourceSnapshot
    records: tuple[EpisodeRecords, ...]
    model: RecordedModelConfigV2
    actions: ActionConfigV2
    declaration: Json
    learner_label: str

    @property
    def proof(self) -> Json:
        # Structural evidence stays explicitly false/not_performed.
        return self._snapshot.collection.proof

    @property
    def _limit(self) -> int:
        return min(self.model.resources.max_owned_bytes, self._snapshot.max_owned_bytes)

    def read_numeric(self, directory: Path, expected_digest: str) -> StructuredRollout:
        checksum(expected_digest)
        self._snapshot.check()
        rollout = _read_recorded_rollout(directory, expected_digest, self.model,
            action_contract=self.actions.action_contract, declaration=self.declaration, expected_records=self.records,
            max_owned_bytes=self._limit, extra_owned_bytes=self._snapshot.resident_bytes,
            check=self._snapshot.check, reserve=self._snapshot.reserve)
        if rollout.action_config_hash != digest(asdict(self.actions)) or rollout.policy_version not in (self.learner_label, None):
            raise ContractError('numeric original actions/frozen policy identity mismatch')
        self._snapshot.recheck()
        return rollout

    def write_numeric(self, directory: Path) -> Json:
        self._snapshot.check()
        # Only exact shared object identities may reduce the joint reservation.
        extra = self._snapshot.extra_bytes_for(self.records)
        self._snapshot.reserve(self._snapshot.additional_bytes((self.records, self.model)) + _predicted_token_bytes(self.records))
        rollout = StructuredRollout(self.records, self.model, digest(asdict(self.actions)),
            action_contract=self.actions.action_contract, purpose='component', synthetic_capacity=False,
            _reserved_bytes=extra)
        owned = (rollout.model, rollout.episodes, rollout.locations, rollout.action_config_hash,
                 rollout.policy_version, rollout.input_binding)
        additional = self._snapshot.additional_bytes(owned)
        self._snapshot.reserve(additional)
        shared = _owned_bytes(owned) - additional
        def reserve(peak: int) -> None:
            self._snapshot.reserve(max(0, peak - shared))
        extra = self._snapshot.extra_bytes_for(owned)
        manifest = _write_rollout(directory, rollout, max_owned_bytes=self._limit, extra_owned_bytes=extra,
            check=self._snapshot.check, reserve=reserve)
        self._snapshot.recheck()
        return manifest


@contextmanager
def component_batch_source(plan: CollectionPlan, *, actions: ActionConfigV2, model: RecordedModelConfigV2,
                           learner_label: str, inference_digest: str, declaration: Json,
                           settings: PreparationConfig, deadline: float, extra_owned_bytes: int = 0,
                           max_owned_bytes: int | None = None) -> Iterator[_ComponentBatchSource]:
    """Retain ORIGINAL Source snapshots through projection/publication/inspection.

    Declaration must come from the caller's S2-validated generation inference;
    this component cannot authorize that parent or qualify engineering data.
    All closures remain present. The learner separately requires a natural cohort.
    """
    closed(declaration, SEMANTIC_KEYS)
    validate_recorded_semantics(declaration, model)
    check_input_binding({'action_contract': asdict(actions.action_contract), 'purpose': 'component', 'synthetic_capacity': False},
        action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
    with _component_source_snapshot(plan.output, actions=actions, settings=settings, expected_split='train',
                                    deadline=min(deadline, plan.execution_deadline), extra_owned_bytes=extra_owned_bytes,
                                    max_owned_bytes=max_owned_bytes) as snapshot:
        records = _frozen_batch(snapshot, plan, actions, model, learner_label, inference_digest)
        snapshot.check()
        yield _ComponentBatchSource(snapshot, records, model, actions, declaration, learner_label)


def _recorded_plan(directory: Path, state: Json, records: list[Json], generation: int,
                   deadline: float, *, descriptor_model: RecordedModelConfigV2 | None = None) -> tuple[CollectionPlan, str, Json]:
    """Recompute original16roles/streams/pool from immutable previous controls."""
    import numpy as np

    from gf_training.collection import training_groups
    from gf_training.lineage import checkpoint_path
    from gf_training.structured_lineage import _state
    actions, model = _state(state, recorded_signature=recorded_token_signature(descriptor_model) if descriptor_model is not None else None)
    inference = InferenceBindingV2(**state['execution']['inference']['binding'])
    visibility = state['execution']['inference']['cuda_visible_devices']
    def binding(number: int) -> tuple[str, RuntimePolicyBinding]:
        record = records[number]
        label = recorded_policy_label(model, inference, record['inference_sha256'], cuda_visible_devices=visibility)
        return label, RuntimePolicyBinding(model.model_id, {'version': '2',
            'checkpoint': str(checkpoint_path(directory, number)), 'expected_manifest_digest': record['inference_sha256'],
            'model': asdict(model), 'inference': asdict(inference), 'cuda_visible_devices': visibility,
            'action_contract': asdict(actions.action_contract), 'purpose': 'component', 'synthetic_capacity': False})
    learner_label, learner = binding(generation)
    registry = {'random-structured-v1': RuntimePolicyBinding('random'),
                'heuristic-structured-v1': RuntimePolicyBinding('heuristic'), learner_label: learner}
    controls = records[generation]['parent_controls']
    snapshots = []
    for number in controls['snapshot_pool']:
        label, value = binding(number)
        registry[label] = value
        snapshots.append(label)
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = controls['schedule_opponent_rng']
    groups = training_groups(state['seed'], generation, learner_label, tuple(snapshots), rng, actions_version=2)
    batch = f'batch-{generation:06}'
    linkage = {'parent_run_id': state['run_id'], 'parent_run_path': str(directory), 'batch_id': batch,
               'collection_generation': generation,
               'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes}}
    return CollectionPlan(groups, registry, directory / 'collections' / batch, linkage, deadline, deadline), learner_label, dict(rng.bit_generator.state)


def check_original_collection_parent(manifest: Json, state: Json) -> None:
    """Bind original collection control provenance to the recorded parent only.

    CollectionPlan supplies output/schedule/bindings without changing Config;
    there are no implicit run/path overrides in this recipe. No current checkout,
    installed package or hardware query can substitute for the recorded parent.
    """
    if (manifest.get('config_sha256') != state['config_sha256']
            or canonical(manifest.get('effective_config')) != canonical(state['effective_config'])
            or canonical(manifest.get('config_inputs')) != canonical(state['config_inputs'])):
        raise ContractError('original collection full Config differs from recorded parent')
    integration = state['identity']['integration']
    for key in ('binary_sha256', 'server_config_sha256'):
        if manifest.get(key) != integration[key]:
            raise ContractError('original collection binary/server-config provenance differs from recorded parent')
    if (manifest.get('simulator_contract_revision') != integration['declared_simulator_revision']
            or canonical(manifest.get('code')) != canonical(state['identity']['code'])
            or manifest.get('simulator', {}).get('revision') != integration['source_revision']
            or canonical(manifest.get('simulator', {}).get('spec_sha256')) != canonical(integration['source_spec_sha256'])):
        raise ContractError('original collection recorded code/Simulator provenance differs from parent')


def validate_component_parent(directory: Path, *, require_complete: bool = True, max_owned_bytes: int,
                               deadline: float, extra_owned_bytes: int = 0) -> tuple[Json, list[Json]]:
    """Whole engineering-component evidence, explicitly no formal eligibility.

    Sealed metadata is inspected without current Torch/runtime/hardware queries.
    Every original cohort/schedule/role/raw numeric link is then reconstructed;
    the S1 proof itself remains false/not_performed rather than being upgraded.
    """
    from gf_models.structured_runtime import PROCESS_ENVELOPE_RECIPE
    from gf_training.lineage import checkpoint_path, generation_path
    from gf_training.structured_lineage import (
        _ChainScan,
        _stat_identity,
        _state,
        _validate_component_chain,
    )
    from gf_training.structured_source import _SourceOwner
    directory = directory.resolve()
    integer(extra_owned_bytes, 0, max_owned_bytes - 1)
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes - extra_owned_bytes, deadline=deadline)
    state, records = _validate_component_chain(directory, require_complete=require_complete, require_retained=False, scan=scan)
    if state['identity'].get('process_envelope_recipe') != PROCESS_ENVELOPE_RECIPE:
        raise ContractError('component process-envelope recipe missing/mismatched; no default upgrade')
    bootstrap = scan.object(checkpoint_path(directory, 0) / 'inference.json', records[0]['inference_sha256'])
    actions, model = _state(state, recorded_signature=bootstrap['token_signature'])
    import numpy as np
    initial_rng = np.random.Generator(np.random.PCG64(state['seed']))
    if canonical(records[0]['parent_controls']) != canonical({'schedule_opponent_rng': dict(initial_rng.bit_generator.state),
            'snapshot_pool': [0], 'collection_position': 0}):
        raise ContractError('original generation0 PCG64 state differs from frozen learning seed')
    settings = preparation_config(state['effective_config']['preprocessing'])
    source_inventories = []
    for number in range(1, len(records)):
        scan.check()
        plan, label, after_rng = _recorded_plan(directory, state, records, number - 1, deadline, descriptor_model=model)
        if canonical(after_rng) != canonical(records[number]['parent_controls']['schedule_opponent_rng']):
            raise ContractError('original component schedule/opponent RNG advance mismatch')
        update = scan.object(directory / 'updates' / f'{number:06}.json', records[number]['update_sha256'])
        if (update['batch_id'] != plan.linkage['batch_id']
                or update['collection'] != str(plan.output.relative_to(directory))
                or canonical(update['policy_bindings']) != canonical({key: asdict(value) for key, value in plan.bindings.items()})
                or canonical(update['learner_seats']) != canonical(plan.linkage['learner_seats'])):
            raise ContractError('original component update differs from immutable frozen schedule/bindings')
        inference = scan.object(checkpoint_path(directory, number - 1) / 'inference.json', records[number - 1]['inference_sha256'])
        with component_batch_source(plan, actions=actions, model=model, learner_label=label,
                inference_digest=records[number - 1]['inference_sha256'], declaration={key: inference[key] for key in SEMANTIC_KEYS},
                settings=settings, deadline=deadline, extra_owned_bytes=scan.retained + extra_owned_bytes,
                max_owned_bytes=max_owned_bytes) as source:
            check_original_collection_parent(source._snapshot.collection.manifest, state)
            if canonical(source.proof) != canonical(update['source_proof']):
                raise ContractError('original component Source proof differs from committed evidence')
            numeric = directory / update['numeric']
            rollout = source.read_numeric(numeric, update['numeric_sha256'])
            rollout.require_learning()
            if len(rollout) != update['accepted_decisions']:
                raise ContractError('original learner endpoints differ from committed numeric count')
            del rollout
            owner = source._snapshot._owner
            # Carry only original file/inventory METADATA, never the Source
            # collection/records/rollout whose raw storage retires this batch.
            for name, entry in owner.files.items():
                scan.check()
                path = owner.root / name
                snapshot = (tuple(entry['identity']), entry['sha256'])
                if path in scan.files:
                    if scan.files[path] != snapshot:
                        raise ContractError('original Source changed between parent and batch snapshots')
                else:
                    scan.charge(1024 + 4 * len(str(path)))
                    scan.files[path] = snapshot
            scan.charge(_owned_bytes(owner.inventories) + 512 + 4 * len(str(owner.root)))
            source_inventories.append((owner.root, owner.deadline, tuple(owner.inventories)))
            del owner
        del source
        # Keep the original chain owner through Source work; never adopt a
        # fresh scan as the baseline for parent control/generation records.
        if scan.files[generation_path(directory, number)][1] != state['generation_sha256'] and number == len(records) - 1:
            raise ContractError('original final generation identity mismatch')
    def original_inventories() -> None:
        for root, original_deadline, inventories in source_inventories:
            scan.check()
            comparison = _SourceOwner(root, settings, min(deadline, original_deadline),
                extra_owned_bytes=scan.retained + extra_owned_bytes, max_owned_bytes=max_owned_bytes)
            for pattern, names in inventories:
                if comparison._inventory_names(pattern, retain=False) != names:
                    raise ContractError('original Source inventory changed across component batches')
    original_inventories()
    scan.finalize()  # includes ALL original journals/reports/sidecars/closures.
    original_inventories()
    for path, (identity, _) in scan.files.items():
        scan.check()
        if _stat_identity(path.lstat()) != identity:
            raise ContractError('original component Source changed after final inventory sweep')
    return state, records


def validate_owned_collection(plan: CollectionPlan, ledger: Any, *, max_owned_bytes: int, deadline: float) -> None:
    """Exact live parent + concrete learner owner, before collection/Simulator I/O."""
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.lineage import _OWNED_LOCKS
    from gf_training.structured_learning_process import StructuredLearningProcess
    from gf_training.structured_lineage import _state
    if not isinstance(ledger, ReservationLedger) or ledger.learner is None or ledger.updating or ledger.workers:
        raise ContractError('owned collection requires the SAME live learner ledger before participant startup')
    lease = ledger.learner
    owner = lease._cleanup_owner
    if (not isinstance(owner, StructuredLearningProcess) or owner.ledger is not ledger or owner.lease is not lease
            or lease.released or lease.creation_pending or owner.steps is None or owner.process is None
            or owner.process.returncode is not None or lease.pid != owner.process.pid):
        raise ContractError('owned collection concrete learner owner is missing/mismatched')
    directory = Path(plan.linkage['parent_run_path']).resolve()
    if directory not in _OWNED_LOCKS or owner.directory.resolve() != directory:
        raise ContractError('owned collection requires its original live parent lock/learner')
    maximum = min(max_owned_bytes, owner.model.resources.max_owned_bytes)
    state, records = validate_component_parent(directory, require_complete=False, max_owned_bytes=maximum, deadline=deadline)
    actions, model = _state(state)
    if type(model) is not ModelConfigV2:
        raise ContractError('historical descriptor cannot authorize an active participant')
    run = state['effective_config']['run']
    if (ledger.global_cuda_bytes != run['global_cuda_budget_bytes'] or ledger.max_workers != run['max_participant_workers']
            or ledger.learner_bytes != 0):
        raise ContractError('owned collection ledger limits differ from frozen parent')
    from gf_training.structured_numeric_worker import wire_identity
    expected_identity = wire_identity(model, InferenceBindingV2(**state['execution']['learner']['binding']),
        digest(asdict(actions)), state['resume_identity'], action_contract=actions.action_contract,
        purpose='component', synthetic_capacity=False,
        cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'])
    expected, _, _ = _recorded_plan(directory, state, records, state['updates'], deadline)
    if (state['status'] != 'running' or canonical(plan.linkage) != canonical(expected.linkage)
            or canonical(serialize_schedule(plan.groups, 4)) != canonical(serialize_schedule(expected.groups, 4))
            or canonical({key: asdict(value) for key, value in plan.bindings.items()}) != canonical({key: asdict(value) for key, value in expected.bindings.items()})
            or plan.output.resolve() != expected.output or plan.kind != 'ppo_batch'
            or canonical(asdict(owner.model)) != canonical(asdict(model))
            or canonical(asdict(owner.binding)) != canonical(state['execution']['learner']['binding'])
            or canonical(owner.identity) != canonical(expected_identity)
            or owner.visibility != state['execution']['learner']['cuda_visible_devices']
            or owner.steps != records[-1]['optimizer_steps']
            or canonical(owner.hardware) != canonical(state['hardware'])):
        raise ContractError('owned collection differs from original parent/frozen plan/learner identity')
