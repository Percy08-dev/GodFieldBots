"""Closed corrected Python operations for the existing single PPO coordinator."""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.config import PPOTrainingConfig, SequencePPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3, input_binding_game_v3
from gf_models.structured_config import InferenceBindingV2, ModelConfigV3
from gf_models.structured_runtime import admit_planned_game_training_v3
from gf_models.structured_sealed import inspect_retained_game_artifact_v3
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training import structured_ppo
from gf_training.card_instance_source import (
    _binding,
    _OnlineGameOwner,
    frozen_linkage,
    game_batch_source,
    validate_game_parent,
    validate_owned_collection,
)
from gf_training.lineage import checkpoint_path
from gf_training.structured_learning_process import GameLearningProcess
from gf_training.structured_lineage import (
    _ChainScan,
    _validate_component_chain,
    commit_game_generation,
    resume_identity,
    retain_game_generations,
)


def _validate_packed_config(config: Any, *, formal: bool = False) -> None:
    if not hasattr(config, 'storage'):
        return
    from gf_models.structured_tree_config import ConfiguredTreeModelV3
    if (type(config.actions) is not ActionConfigV3 or type(config.model) is not ConfiguredNeuralModelV3
            or type(config.training) is not PPOTrainingConfig or config.run.split != 'train'
            or formal or config.preprocessing.backend not in ('python', 'rust')
            or config.preprocessing.reader != 'participant-structured-v2'
            or type(config.opponent) is ConfiguredTreeModelV3):
        raise ContractError('group Pack storage PPO requires ordinary corrected PPO')
    if config.preprocessing.backend == 'rust':
        from gf_training.native_backend import group_pack_reader_settings
        group_pack_reader_settings(config.preprocessing)


class _GameOperations:
    """One known composition; no configurable registry, callbacks or trainer loop."""
    actions_version = 3
    registry_schema = 3

    def __init__(self, *, formal: Any = None) -> None:
        if formal is not None:
            from gf_training.card_instance_formal import FormalControl
            if type(formal) is not FormalControl:
                raise ContractError('one concrete formal control context required')
        self.formal = formal
        self.online: _OnlineGameOwner | None = None
        self.ownership: Any = None
        self.state: Json | None = None
        self.ledger: ReservationLedger | None = None
        self.projection: Any = None
        self.runtime: Any = None
        self.native: Any = None
        self.finalized_collection: Any = None

    def __getattr__(self, name: str) -> Any:
        if name in {'cap', 'saved_deadline', 'execution', 'decorate_identity'}:
            return getattr(structured_ppo, name)
        raise AttributeError(name)

    def validate_config(self, config: Any) -> ModelConfigV3:
        _validate_packed_config(config, formal=self.formal is not None)
        if (type(config.actions) is not ActionConfigV3 or type(config.model) is not ConfiguredNeuralModelV3
                or config.model.checkpoint is not None or type(config.training) is not PPOTrainingConfig
                or isinstance(config.training, SequencePPOTrainingConfig) or config.run.split != 'train'
                or config.preprocessing.backend not in ('python', 'rust') or config.preprocessing.reader != 'participant-structured-v2'):
            raise ContractError('corrected PPO requires fresh game Model3/actions3/python/participant-structured-v2')
        input_binding_game_v3(config.actions.action_contract, card_instance_rules())
        if config.preprocessing.backend == 'rust':
            from gf_training.native_backend import card_instance_reader_settings, load_native
            self.native = self.native or load_native()
            card_instance_reader_settings(config.preprocessing, self.native)
        return config.model.architecture(layout=LAYOUT)

    def preprocessing_identity(self, config: Any, *, deadline: float) -> Json:
        self.validate_config(config)
        if time.monotonic() >= deadline:
            raise TimeoutError('corrected PPO preprocessing deadline')
        binding = input_binding_game_v3(config.actions.action_contract, card_instance_rules())
        if config.preprocessing.backend == 'rust':
            from gf_training.native_backend import binary_identity
            return {'backend': 'rust', 'reader_contract': 'participant-structured-v2',
                'implementation': 'rust-player-card-refs-v2', 'binary': binary_identity(self.native), **binding}
        return {'backend': 'python', 'reader_contract': 'participant-structured-v2',
            'implementation': 'python-reference-player-card-refs-v2', **binding}

    def preflight(self, config: Any, *, deadline: float) -> None:
        model = self.validate_config(config)
        if config.server.external_url is not None or config.server.source_dir is None:
            raise ContractError('corrected PPO requires locally owned approved runtime source')
        from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
        self.runtime = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir), deadline=deadline)
        self.runtime.preflight()
        admit_planned_game_training_v3(config.actions, model, config.model.inference,
            replace(config.model.inference, device=config.training.device), rules=card_instance_rules(),
            game_concurrency=config.run.concurrency, global_cuda_bytes=config.run.global_cuda_budget_bytes,
            max_workers=config.run.max_participant_workers)

    def integration_identity(self, config: Any, deadline: float) -> Json:
        from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
        from gf_models.arrays import sha256
        from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
        runtime = self.runtime or CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir), deadline=deadline)
        receipt = runtime.preflight()
        return {'binary_sha256': receipt['binary']['sha256'], 'server_config_sha256': sha256(Path(config.server.config), deadline),
            'declared_simulator_revision': config.server.simulator_revision, 'binary_build_revision': None,
            'source_revision': receipt['source']['revision'], 'source_spec_sha256': dict(CORRECTED_REVIEWED_SPEC_HASHES)}

    def initialize_state(self, state: Json, config: Any) -> None:
        state.update(schema_version=3, model_id=config.model.model_id, model_version='3', input_recipe=LAYOUT.input_recipe,
            execution=structured_ppo.execution(config), **input_binding_game_v3(config.actions.action_contract, card_instance_rules()))
        classification = input_binding_game_v3(config.actions.action_contract, card_instance_rules())
        if self.formal is not None:
            self.formal.initialize_state(state)
        if hasattr(config, 'storage'):
            from gf_common.group_pack_game import storage_binding
            state.update(schema_version=4, storage_binding=storage_binding())
        state['resume_identity'] = resume_identity(state['config_sha256'], state['identity'], state['execution'], classification,
            storage_binding=state.get('storage_binding'))

    def external_roots(self, state: Json) -> tuple[Path, ...]:
        if state['schema_version'] != 4:
            return ()
        root = Path(state['effective_config']['storage']['db_root'])
        return (root / state['run_id'], root / 'catalog.sqlite3',
            root / 'catalog.sqlite3-wal', root / 'catalog.sqlite3-shm', root / 'catalog.sqlite3-journal')

    def accept_collection(self, collection: Any, config: Any) -> None:
        from gf_training.collection import FinalizedCollection, PackedFinalizedCollection
        expected = PackedFinalizedCollection if hasattr(config, 'storage') else FinalizedCollection
        if type(collection) is not expected:
            raise ContractError('callback requires its exact configured finalized collection variant')
        if expected is PackedFinalizedCollection and (cast(PackedFinalizedCollection, collection).storage_metrics.get('publication_scope') != 'published'
                or collection.report.get('healthy') is not True):
            raise ContractError('healthy original packed publication required')
        self.finalized_collection = collection

    def read_control(self, path: Path, config: Any, *, deadline: float) -> Any:
        if not hasattr(config, 'storage') or path.name != 'manifest.json':
            return structured_ppo.read_control(path, config, deadline=deadline)
        from gf_common.preprocessing import preparation_config
        from gf_training.group_pack_game_source import PackedBatchCustody, PackedSourceOwner
        online = self.online
        if online is None:
            raise ContractError('packed callback controls require SAME live parent')
        online.check()
        if online.packed_batch is None:
            settings = preparation_config(online.state['effective_config']['preprocessing'])
            expected = Path(config.storage.db_root) / online.state['run_id'] / 'collections' / path.parent.name
            original = PackedSourceOwner(path.parent, settings, deadline,
                storage_binding=online.state['storage_binding'], max_owned_bytes=structured_ppo.cap(config),
                storage_limit=config.storage.budget_mib * 1024**2, expected_descriptor_root=expected)
            online.packed_batch = PackedBatchCustody(original)
        elif online.packed_batch.owner.root != path.parent:
            raise ContractError('previous original packed batch has not retired')
        original = online.packed_batch.owner
        result = self.finalized_collection
        if (result is None or result.directory.absolute() != original.root
                or result.envelope_sha256 != original.locator.sha256
                or result.locator.path != original.artifacts._originals.descriptor_file.path):
            raise ContractError('callback locator/hash differs from actual original batch custody')
        scan = _ChainScan(directory=online.ownership.directory, max_owned_bytes=structured_ppo.cap(config), deadline=deadline)
        scan.configure_packed(online.state)
        online.packed_batch.borrow(scan)
        return scan.object(path), scan

    def close_batch(self) -> None:
        if self.online is not None and self.online.packed_batch is not None:
            self.online.packed_batch.close()
            self.online.packed_batch = None
            self.finalized_collection = None

    def persist_control(self, path: Path, value: Json, config: Any, *, deadline: float, original: Any = None) -> None:
        if (self.formal is not None and path.name == 'run.json'
                and self.formal.persist(path, value, deadline=deadline, max_owned_bytes=structured_ppo.cap(config))):
            return
        structured_ppo.persist_control(path, value, config, deadline=deadline, original=original)

    def bind_run(self, ownership: Any, state: Json) -> None:
        self.ownership, self.state = ownership, state

    def make_child(self, directory: Path, config: Any, state: Json, *, deadline: float) -> Any:
        ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
        binding = InferenceBindingV2(**state['execution']['learner']['binding'])
        child = GameLearningProcess(directory, config.training.max_log_bytes, config.model.architecture(layout=LAYOUT), binding,
            digest(asdict(config.actions)), state['resume_identity'], actions=config.actions, ledger=ledger,
            cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'], cleanup_deadline=deadline,
            storage_binding=state.get('storage_binding'))
        self.online = _OnlineGameOwner(self.ownership, state, child, ledger)
        self.online.formal_control = self.formal
        child.online_owner = self.online
        self.ledger = ledger
        return child, ledger

    def linkage(self, state: Json, directory: Path, generation: int, groups: Any, registry: Any) -> Json:
        return frozen_linkage(state, directory, generation, groups, registry)

    def validate_parent(self, directory: Path, config: Any, deadline: float, *, extra_owned_bytes: int = 0) -> Any:
        return validate_game_parent(directory, require_complete=False, max_owned_bytes=structured_ppo.cap(config),
            deadline=deadline, extra_owned_bytes=extra_owned_bytes)

    def validate_owned_collection(self, plan: Any, ledger: Any, *, config: Any, deadline: float) -> None:
        validate_owned_collection(plan, ledger, max_owned_bytes=structured_ppo.cap(config), deadline=deadline)

    def binding(self, directory: Path, generation: int, config: Any, deadline: float) -> Any:
        scan = _ChainScan(directory=directory, max_owned_bytes=structured_ppo.cap(config), deadline=deadline)
        with scan:
            state, records = _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan, _game=True)
            registry = scan.object(directory / 'artifact-registry.json')
            if registry['generations'][str(generation)]['status'] != 'retained':
                raise ContractError('game policy cannot load a pruned generation')
            inspect_retained_game_artifact_v3(checkpoint_path(directory, generation), expected_inference_digest=records[generation]['inference_sha256'],
                actions=config.actions, rules=card_instance_rules(), scan=scan)
            result = _binding(directory, state, records[generation], generation)
            scan.finalize()
            return result

    @contextmanager
    def project(self, plan: Any, numeric: Path, config: Any, learner_label: str, generation: int, deadline: float,
                *, expected_manifest_sha256: str, extra_owned_bytes: int = 0) -> Iterator[Any]:
        from gf_models.arrays import sha256
        with game_batch_source(plan, self.ledger, max_owned_bytes=structured_ppo.cap(config), deadline=deadline,
                               extra_owned_bytes=extra_owned_bytes) as source:
            if source.proof['manifest_sha256'] != expected_manifest_sha256 or source.learner_label != learner_label:
                raise ContractError('original game Source differs from frozen callback result')
            manifest = source.write_numeric(numeric)
            files = {'rollout.json': sha256(numeric / 'rollout.json', deadline), **manifest['data']['files']}
            rollout = source.read_numeric(numeric, files['rollout.json'])
            rollout.require_learning()
            count = len(rollout)
            del rollout
            for name, expected in files.items():
                with source.controls.file(numeric / name, expected):
                    pass
            handle = source.issue(numeric, files)
            self.projection = handle
            primary: BaseException | None = None
            try:
                yield count, source.proof, files, handle
            except BaseException as error:
                primary = error
                raise
            finally:
                self.projection = None
                try:
                    self.close_batch()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.__dict__['cleanup_owner'] = self.online.packed_batch if self.online is not None else self
                    primary.add_note('packed batch cleanup: ' + str(cleanup))

    def update_record(self, update: Json, state: Json, numeric_files: Json) -> None:
        assert self.online is not None
        classification = input_binding_game_v3(self.online.process.actions.action_contract, card_instance_rules())
        update.update(schema_version=4 if state['schema_version'] == 4 else 3, model_id=state['model_id'], model_version='3', input_recipe=state['input_recipe'],
            **classification, numeric_files=numeric_files)
        if state['schema_version'] == 4:
            update['storage_binding'] = state['storage_binding']

    def commit(self, directory: Path, state: Json, config: Any, *, generation: int, update: Json | None,
               controls: Json, acknowledgement: Json, deadline: float) -> Json:
        if self.formal is not None:
            from gf_training.card_instance_formal import read_formal_control
            scan = _ChainScan(directory=directory, max_owned_bytes=structured_ppo.cap(config), deadline=deadline)
            read_formal_control(directory, state, scan, self.formal.recipe)
            scan.finalize()
            self.formal.originals.finalize()
        return commit_game_generation(directory, state, generation=generation, update=update, controls=controls,
            artifact_files=acknowledgement['artifact']['files'], max_owned_bytes=structured_ppo.cap(config),
            deadline=deadline, projection=self.projection if generation else None)

    def retain(self, directory: Path, generation: int, config: Any, deadline: float) -> None:
        retain_game_generations(directory, generation, max_owned_bytes=structured_ppo.cap(config), deadline=deadline)


async def train_card_instance_ppo(loaded: Any, collector: Any, directory: Path, *, mode: str = 'fresh',
                                  pause_after_update: int | None = None) -> Json:
    _validate_packed_config(loaded.config)
    operations = _GameOperations()
    operations.validate_config(loaded.config)
    from gf_training.ppo_run import _run_ppo
    return await _run_ppo(loaded, collector, directory, mode=mode, pause_after_update=pause_after_update,
                          _game_ops=operations)


async def train_formal_card_instance_ppo(loaded: Any, collector: Any, directory: Path, *, mode: str = 'fresh',
                                         pause_after_update: int | None = None) -> Json:
    """Explicit full-recipe entry, keeping completion revocable through lock exit."""
    if hasattr(loaded.config, 'storage'):
        raise ContractError('group Pack storage formal PPO awaits its accepted Source integration')
    from gf_training.card_instance_formal import FormalControl
    from gf_training.ppo_run import _run_ppo
    control = FormalControl(loaded)
    operations = _GameOperations(formal=control)
    try:
        result = await _run_ppo(loaded, collector, directory, mode=mode, pause_after_update=pause_after_update,
                              _game_ops=operations)
        control.finish()
        return result
    except BaseException as error:
        control.fail(error)  # FIRST: invalidate an exposed owned completion.
        raise
