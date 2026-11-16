"""One actual learner-state export, distinct inference/resume memfd owners."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.artifacts import ArtifactAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, digest, integer
from gf_common.ram_bank import MAX_BANK_BYTES, ReadonlyRamBankSet
from gf_models.arrays import MAX_ARRAY_BYTES, sha256
from gf_models.sharded_arrays import (
    _write_producer_bundle,
    create_output_directory,
    write_owned_json,
)
from gf_models.structured_artifacts import (
    _cpu_array,
    _validate_shuffle_owner,
    producer_identity,
    validate_save_context,
)
from gf_models.structured_binding import inference_model
from gf_models.structured_config import architecture_identity
from gf_training.hpo_artifacts import _recipe_options, checked_phase_limits, save_estimate

_TOKEN = object()


def _backings(banks: ReadonlyRamBankSet) -> tuple[tuple[int, int, int], ...]:
    return tuple((stat.st_dev, stat.st_ino, stat.st_size)
        for stat in (os.fstat(bank.fileno()) for bank in banks.banks))


class OwnedSaveAck:
    """Issued only after this snapshot's actual writer completes all files."""
    def __init__(self, token: object, snapshot: OwnedModelSnapshot, writer: object, files: Json, inference_manifest: Json) -> None:
        if token is not _TOKEN or snapshot._writer is not writer:
            raise ContractError('actual snapshot writer completion required')
        self._snapshot, self._writer = snapshot, writer
        self._files = canonical(files).encode()
        self._inference_manifest=(canonical(inference_manifest)+'\n').encode()
        if hashlib.sha256(self._inference_manifest).hexdigest()!=files['inference.json']:
            raise ContractError('actual inference writer manifest/hash differs')
        self.generation, self.optimizer_steps = snapshot.generation, snapshot.optimizer_steps
        self._inference, self._resume = _backings(snapshot.inference), _backings(snapshot.resume)

    @property
    def files(self) -> Json:
        return json.loads(self._files)

    @property
    def inference_manifest(self) -> Json:
        return json.loads(self._inference_manifest)

    def require(self, snapshot: OwnedModelSnapshot) -> None:
        if (snapshot is not self._snapshot or snapshot.save_ack is not self or snapshot._writer is not self._writer
                or self.generation != snapshot.generation or self.optimizer_steps != snapshot.optimizer_steps
                or self._inference != _backings(snapshot.inference) or self._resume != _backings(snapshot.resume)):
            raise ContractError('save ACK differs from actual writer/snapshot/generation/backings')


class OwnedModelSnapshot:
    def __init__(self, token: object, learner: Any, inference: ReadonlyRamBankSet,
                 resume: ReadonlyRamBankSet, inference_metadata: Json, resume_metadata: Json,
                 *, generation: int, budget: ArtifactAdmission, reservation: Any) -> None:
        if token is not _TOKEN:
            raise ContractError('snapshot must originate at actual learner export')
        self.inference, self.resume = inference, resume
        self.generation, self.optimizer_steps = generation, learner.steps
        self._inference_metadata = canonical(inference_metadata).encode()
        self._resume_metadata = canonical(resume_metadata).encode()
        self._learner, self._budget, self._reservation = learner, budget, reservation
        self.timings: dict[str,float] = {}
        self._save_budget_limit = 512*1024**2
        self.resident_bytes = inference.admitted_bytes+resume.admitted_bytes+16*json_bound((inference_metadata,resume_metadata))+65536
        self._manifest_reservation: Any = None
        self._writer: object | None = None
        self.save_ack: OwnedSaveAck | None = None
        self._actor_capability: Any = None
        self._state = 'SEALED'
        self.identity = digest({'generation': generation, 'steps': self.optimizer_steps,
            'inference': _backings(inference), 'resume': _backings(resume),
            'inference_metadata': inference_metadata, 'resume_metadata': resume_metadata})

    @property
    def inference_manifest(self) -> Json:
        if self.save_ack is None: raise ContractError('actual inference writer completion required')
        self.save_ack.require(self)
        return self.save_ack.inference_manifest

    @property
    def actor_metadata(self) -> Json:
        if self.save_ack is None:
            raise ContractError('actor metadata requires actual successful save')
        self.save_ack.require(self)
        return {'protocol_version': 9, 'command': 'SYNC_OWNED', 'generation': self.generation,
            'snapshot_identity': self.identity, 'inference_digest': self.save_ack.files['inference.json'],
            'inference': json.loads(self._inference_metadata),
            'arrays': [asdict(descriptor) for bank in self.inference.banks for descriptor in bank.descriptors]}

    @property
    def actor_capability(self) -> Any:
        from gf_common.hpo_actor_snapshot import _issue_actor_snapshot
        _check_actor_origin(self)
        if self._actor_capability is None:
            self._actor_capability=_issue_actor_snapshot(self,inference=self.inference,
                metadata=self.actor_metadata,check_origin=_check_actor_origin,budget=self._budget)
        self._actor_capability.require()
        return self._actor_capability

    def save(self, directory: Path, *, shard_bytes: int = MAX_ARRAY_BYTES) -> OwnedSaveAck:
        if self._state != 'SEALED':
            raise ContractError('snapshot save is single use')
        started = time.perf_counter()
        self._state, self._writer = 'FAILED', object()
        integer(shard_bytes, 1, MAX_ARRAY_BYTES)
        scratch = (35 * min(shard_bytes, max(self.inference.size, self.resume.size))
            + 32 * (len(self._inference_metadata) + len(self._resume_metadata)) + 65536)
        try:
            if self.resident_bytes+scratch>self._save_budget_limit:
                raise ContractError('owned snapshot/save aggregate envelope exceeded before codec')
            with self._budget.reserve(scratch):
                create_output_directory(directory, 'inference.json')
                inference = json.loads(self._inference_metadata)
                inference['weights'] = _write_producer_bundle(directory, 'weights',
                    {name: self.inference.array(name) for name in self.inference.names}, shard_bytes=shard_bytes)
                write_owned_json(directory/'inference.json', inference)
                inference_digest = sha256(directory/'inference.json')
                resume = json.loads(self._resume_metadata)
                resume['inference_digest'] = inference_digest
                resume['state'] = _write_producer_bundle(directory, 'resume',
                    {name: self.resume.array(name) for name in self.resume.names}, shard_bytes=shard_bytes)
                write_owned_json(directory/'resume.json', resume)
                files = {**inference['weights']['files'], **resume['state']['files'],
                    'inference.json': inference_digest, 'resume.json': sha256(directory/'resume.json')}
                manifest_charge=32*json_bound(inference)+65536
                if self.resident_bytes+scratch+manifest_charge>self._save_budget_limit:
                    raise ContractError('owned saved inference metadata exceeds aggregate save envelope')
                self._manifest_reservation=self._budget.reserve(manifest_charge)
                self.resident_bytes+=manifest_charge
                ack = OwnedSaveAck(_TOKEN, self, self._writer, files, inference)
            self.save_ack, self._state = ack, 'SAVED'
            self.timings['save_seconds'] = time.perf_counter()-started
            return ack
        except BaseException:
            self._learner._owned_failed = True
            raise

    def close(self) -> None:
        self.inference.close()
        self.resume.close()
        self._reservation.close()
        if self._manifest_reservation is not None: self._manifest_reservation.close()
        if self._actor_capability is not None:
            self._actor_capability._retire_origin(self)
        self._state = 'RETIRED'


def _check_actor_origin(owner: OwnedModelSnapshot) -> None:
    if type(owner) is not OwnedModelSnapshot or owner._state!='SAVED' or owner.save_ack is None:
        raise ContractError('actor capability requires actual complete local snapshot save')
    owner.save_ack.require(owner)


def export_learner_snapshot(learner: Any, *, budget: ArtifactAdmission,
                            save_budget_bytes: int = 512*1024**2,
                            code_revision: str | None = None) -> OwnedModelSnapshot:
    from gf_training.hpo_numeric import HpoLearner
    started = time.perf_counter()
    if type(learner) is not HpoLearner or getattr(learner, '_owned_failed', False):
        raise ContractError('actual healthy HPO learner required')
    if learner._owned_legacy:
        raise ContractError('legacy update/restore cannot issue producer-owned snapshot generation')
    if learner._owned_source_budget is not None and learner._owned_source_budget is not budget:
        raise ContractError('owned snapshot export requires SAME invocation Source account')
    existing = getattr(learner, '_owned_snapshot', None)
    if existing is not None:
        return existing
    checked_phase_limits({'numeric_budget_bytes':MAX_BANK_BYTES,'save_budget_bytes':save_budget_bytes})
    devices = 0 if learner.binding.device == 'cpu' else learner.torch.cuda.device_count()
    estimate = save_estimate(learner.model, cuda_devices=devices)
    if estimate['additional_bytes'] > save_budget_bytes:
        raise ContractError('owned snapshot/save envelope exceeded before export')
    options = _recipe_options(learner.training)
    learner._context.bind(learner.network)
    learner._context.check(learner.input_binding)
    recipe = validate_save_context(learner.network, learner.model, learner.binding, learner.optimizer,
        _optimizer_recipe=options['_optimizer_recipe'], **learner._options)
    _validate_shuffle_owner(learner.shuffle)
    learner.synchronize()
    learner._owned_source_budget=budget
    # Reserve the complete CPU export envelope before ANY model/Adam/RNG copy.
    with budget.reserve(estimate['additional_bytes']):
        weights = {name: _cpu_array(tensor) for name, tensor in learner.network.state_dict().items()}
        state = {'rng.torch_cpu': _cpu_array(learner.torch.get_rng_state())}
        cuda = learner.torch.cuda.get_rng_state_all() if devices else []
        state.update({f'rng.torch_cuda.{index}': _cpu_array(value) for index, value in enumerate(cuda)})
        presence = {}
        for name, parameter in learner.network.named_parameters():
            values = learner.optimizer.state.get(parameter)
            presence[name] = {'present': bool(values)}
            if values:
                step = integer(int(values['step'].item()), 1, learner.steps)
                state[f'adam.{name}.step'] = np.asarray(step, np.int64)
                for key in ('exp_avg', 'exp_avg_sq'):
                    state[f'adam.{name}.{key}'] = _cpu_array(values[key])
        context = learner._context
        common = {'schema_version': context.schema, **context.classification(), **context.semantics(learner.model),
            'action_config_hash': learner.action_config_hash}
        inference_metadata = {**common, 'kind': 'structured-inference', 'model': inference_model(learner.model),
            'architecture_identity': architecture_identity(learner.model, **learner._options),
            'producer': producer_identity(code_revision)}
        resume_metadata = {**common, 'kind': 'structured-resume', **options['_resume_metadata'],
            'model': asdict(learner.model), 'binding': asdict(learner.binding), 'resume_identity': learner.resume_identity,
            'producer': producer_identity(), 'optimizer_steps': learner.steps, 'optimizer_recipe': recipe,
            'parameter_states': presence, 'cuda_rng_count': len(cuda), 'shuffle_rng': learner.shuffle.bit_generator.state}
        metadata_charge=16*json_bound((inference_metadata,resume_metadata))+65536
        names=len(weights)+len(state)
        backing_count=(len(weights)+255)//256+(len(state)+255)//256
        bank_charge=sum(array.nbytes for array in (*weights.values(),*state.values()))+1088*names+(65536+4096)*backing_count
        if estimate['additional_bytes']+metadata_charge+bank_charge>save_budget_bytes:
            raise ContractError('owned export/metadata/bank aggregate envelope exceeded before memfd')
        reservation = budget.reserve(metadata_charge)
        inference = resume = None
        try:
            inference = ReadonlyRamBankSet.build(weights, budget, max_bytes=min(MAX_BANK_BYTES,save_budget_bytes))
            resume = ReadonlyRamBankSet.build(state, budget, max_bytes=min(MAX_BANK_BYTES,save_budget_bytes-inference.admitted_bytes))
            output = OwnedModelSnapshot(_TOKEN, learner, inference, resume, inference_metadata, resume_metadata,
                generation=getattr(learner, '_owned_generation', 0), budget=budget, reservation=reservation)
            output._save_budget_limit = save_budget_bytes
            output.timings['snapshot_seconds'] = time.perf_counter()-started
            learner._owned_snapshot = output
            return output
        except BaseException:
            if inference is not None:
                inference.close()
            if resume is not None:
                resume.close()
            reservation.close()
            raise
