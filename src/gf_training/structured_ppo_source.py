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

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.preprocessing import PreparationConfig
from gf_common.schedule import serialize_schedule
from gf_common.structured import ActionConfigV2
from gf_models.structured_binding import check_input_binding, policy_label, safe_options_v2
from gf_models.structured_config import InferenceBindingV2, ModelConfigV2
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
                  model: ModelConfigV2, learner_label: str, inference_digest: str) -> tuple[EpisodeRecords, ...]:
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
    options = safe_options_v2(plan.bindings[learner_label].policy_options,
        action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
    binding = InferenceBindingV2(**options['inference'])
    if (canonical(options['model']) != canonical(asdict(model))
            or options['expected_manifest_digest'] != inference_digest
            or learner_label != policy_label(model, binding, inference_digest,
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
    model: ModelConfigV2
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
def component_batch_source(plan: CollectionPlan, *, actions: ActionConfigV2, model: ModelConfigV2,
                           learner_label: str, inference_digest: str, declaration: Json,
                           settings: PreparationConfig, deadline: float) -> Iterator[_ComponentBatchSource]:
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
                                    deadline=min(deadline, plan.execution_deadline)) as snapshot:
        records = _frozen_batch(snapshot, plan, actions, model, learner_label, inference_digest)
        snapshot.check()
        yield _ComponentBatchSource(snapshot, records, model, actions, declaration, learner_label)
