"""Concrete model2 helpers for the existing single PPO coordinator loop.

The private engineering entry accepts an injected component collector. It grants
no real-I/O or qualified Source permission; the normal environment guards remain.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, RuntimePolicyBinding, digest
from gf_common.structured import INPUT_RECIPE, ActionConfigV2
from gf_models.config import PPOTrainingConfig, SequencePPOTrainingConfig
from gf_models.sharded_arrays import read_json
from gf_models.structured_binding import (
    ConfiguredNeuralModelV2,
    child_cuda_visibility,
    input_binding,
)
from gf_models.structured_config import InferenceBindingV2, ModelConfigV2
from gf_models.structured_runtime import PROCESS_ENVELOPE_RECIPE
from gf_models.structured_sealed import SEMANTIC_KEYS
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import CollectionPlan
from gf_training.lineage import checkpoint_path
from gf_training.structured_learning_process import StructuredLearningProcess
from gf_training.structured_lineage import (
    _ChainScan,
    _write,
    commit_component_generation,
    owned_component_binding,
    resume_identity,
    retain_component_generations,
)
from gf_training.structured_ppo_source import component_batch_source


def decorate_identity(identity: Json) -> None:
    identity['process_envelope_recipe'] = PROCESS_ENVELOPE_RECIPE


def validate_owned_collection(plan: CollectionPlan, ledger: ReservationLedger, *, config: Any, deadline: float) -> None:
    from gf_training.structured_ppo_source import validate_owned_collection as validate
    validate(plan, ledger, max_owned_bytes=cap(config), deadline=deadline)


def validate_config(config: Any) -> ModelConfigV2:
    if (not isinstance(config.actions, ActionConfigV2) or not isinstance(config.model, ConfiguredNeuralModelV2)
            or config.model.checkpoint is not None or not isinstance(config.training, PPOTrainingConfig)
            or isinstance(config.training, SequencePPOTrainingConfig) or config.run.split != 'train'):
        raise ContractError('component PPO2 requires model2/actions2 and one model temporal window')
    input_binding(config.actions.action_contract, 'component', False)
    return config.model.architecture()


def cap(config: Any) -> int:
    return min(config.training.max_owned_bytes, config.preprocessing.memory_budget_mib * 1024**2)


def read_control(path: Path, config: Any, *, deadline: float) -> tuple[Json, _ChainScan]:
    """Caller-bounded metadata read; retain its original identity until use."""
    root = path.parent.parent.parent if path.name == 'manifest.json' else path.parent
    scan = _ChainScan(directory=root, max_owned_bytes=cap(config), deadline=deadline)
    value = scan.object(path)
    scan.finalize()
    return value, scan


def saved_deadline(value: Json) -> float:
    number = value.get('execution_deadline_unix')
    if number is None:
        raise ContractError('recorded execution deadline must be finite')
    try:
        valid = type(number) in (int, float) and math.isfinite(number) and number > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ContractError('recorded execution deadline must be finite')
    return float(number)


def persist_control(path: Path, value: Json, config: Any, *, deadline: float,
                    original: _ChainScan | None = None) -> None:
    scan = original or _ChainScan(directory=path.parent, max_owned_bytes=cap(config), deadline=deadline)
    scan.deadline = min(scan.deadline, deadline)
    scan.check()
    if path in scan.files:
        scan.finalize()
    elif path.exists() or path.is_symlink():
        scan.object(path)
    _write(path, value, scan=scan)
    scan.finalize()


def preprocessing_identity(config: Any, *, deadline: float) -> Json:
    if time.monotonic() >= deadline:
        raise TimeoutError('structured PPO preprocessing deadline')
    if config.preprocessing.reader != 'structured-v1':
        raise ContractError('PPO2 requires explicit structured-v1 reader')
    if config.preprocessing.backend == 'python':
        return {'backend': 'python', 'reader_contract': 'structured-v1', 'implementation': 'python-reference-structured-v1'}
    from gf_training.native_backend import load_native, reader_settings
    reader_settings(config.preprocessing, load_native())
    raise ContractError('structured native PPO integration unavailable; no Python fallback')


def execution(config: Any) -> Json:
    inference = config.model.inference
    learner = replace(inference, device=config.training.device)
    return {role: {'binding': asdict(binding), 'cuda_visible_devices': child_cuda_visibility(binding.device, config.model.cuda_visible_devices)}
            for role, binding in (('learner', learner), ('inference', inference))}


def initialize_state(state: Json, config: Any) -> None:
    model = config.model.architecture()
    classification = input_binding(config.actions.action_contract, 'component', False)
    state.update(schema_version=2, model_id=model.model_id, model_version='2', input_recipe=INPUT_RECIPE,
                 execution=execution(config), **classification)
    state['resume_identity'] = resume_identity(state['config_sha256'], state['identity'], state['execution'], classification)


def make_child(directory: Path, config: Any, state: Json, *, deadline: float) -> tuple[StructuredLearningProcess, ReservationLedger]:
    ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
    binding = InferenceBindingV2(**state['execution']['learner']['binding'])
    child = StructuredLearningProcess(directory, config.training.max_log_bytes, config.model.architecture(), binding,
        digest(asdict(config.actions)), state['resume_identity'], action_contract=config.actions.action_contract,
        ledger=ledger, cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'], cleanup_deadline=deadline)
    return child, ledger


def commit(directory: Path, state: Json, config: Any, *, generation: int, update: Json | None,
           controls: Json, acknowledgement: Json, deadline: float) -> Json:
    return commit_component_generation(directory, state, generation=generation, update=update, controls=controls,
        artifact_files=acknowledgement['artifact']['files'], max_owned_bytes=cap(config), deadline=deadline)


def binding(directory: Path, generation: int, config: Any, deadline: float) -> tuple[str, RuntimePolicyBinding]:
    return owned_component_binding(directory, generation, max_owned_bytes=cap(config), deadline=deadline)


def retain(directory: Path, generation: int, config: Any, deadline: float) -> None:
    retain_component_generations(directory, generation, max_owned_bytes=cap(config), deadline=deadline)


def validate_parent(directory: Path, config: Any, deadline: float, *, extra_owned_bytes: int = 0) -> tuple[Json, list[Json]]:
    from gf_training.structured_ppo_source import validate_component_parent
    return validate_component_parent(directory, require_complete=False, max_owned_bytes=cap(config),
                                     deadline=deadline, extra_owned_bytes=extra_owned_bytes)


@contextmanager
def project(plan: CollectionPlan, numeric: Path, config: Any, learner_label: str, generation: int, deadline: float,
            *, expected_manifest_sha256: str, extra_owned_bytes: int = 0) -> Iterator[tuple[int, Json, Json]]:
    options = plan.bindings[learner_label].policy_options
    if options is None:
        raise ContractError('structured learner options required')
    previous = read_json(checkpoint_path(Path(plan.linkage['parent_run_path']), generation) / 'inference.json',
                         expected_digest=options['expected_manifest_digest'], max_owned_bytes=cap(config))
    state = read_json(Path(plan.linkage['parent_run_path']) / 'run.json', max_owned_bytes=cap(config))
    from gf_training.structured_ppo_source import check_original_collection_parent
    from gf_training.structured_rollout import _owned_bytes
    with component_batch_source(plan, actions=config.actions, model=config.model.architecture(), learner_label=learner_label,
            inference_digest=options['expected_manifest_digest'], declaration={key: previous[key] for key in SEMANTIC_KEYS},
            settings=config.preprocessing, deadline=deadline, extra_owned_bytes=_owned_bytes((previous, state)) + extra_owned_bytes,
            max_owned_bytes=cap(config)) as source:
        if source.proof['manifest_sha256'] != expected_manifest_sha256:
            raise ContractError('original collection changed after frozen callback guard')
        check_original_collection_parent(source._snapshot.collection.manifest, state)
        manifest = source.write_numeric(numeric)
        from gf_models.arrays import sha256
        numeric_hash = sha256(numeric / 'rollout.json', deadline)
        rollout = source.read_numeric(numeric, numeric_hash)
        rollout.require_learning()  # reject the WHOLE interrupted/empty cohort.
        count = len(rollout)
        del rollout
        source._snapshot.recheck()  # original evidence before the first Adam request.
        yield count, source.proof, {'rollout.json': numeric_hash, **manifest['data']['files']}


async def _train_component_ppo(loaded: Any, collector: Any, directory: Path, *, mode: str = 'fresh',
                              pause_after_update: int | None = None) -> Json:
    """Private synthetic/FakeHTTP composition; no public real-I/O bypass."""
    validate_config(loaded.config)
    from gf_training.ppo_run import _run_ppo
    return await _run_ppo(loaded, collector, directory, mode=mode, pause_after_update=pause_after_update,
                          _structured_component=True)
