"""Same-process numerical owners for the fixed supervised training composition.

Numerical kernels and checkpoint producers remain the existing Model3/4 code.
Source/collection authorization belongs to the concrete coordinator, not these
component facades. CPU components do not fabricate GPU or process authority.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json
from gf_models.arrays import sha256
from gf_models.rust_player import serialized_torch
from gf_training.rust_player_supervision import HostAdmission

_SAVED = object()


def _admit_device(binding: Any, admission: HostAdmission | None) -> None:
    if admission is not None:
        if type(admission) is not HostAdmission:
            raise ContractError("actual same-host admission required")
        admission.require()
    if binding.device != "cpu" and admission is None:
        raise ContractError("CUDA allocation requires actual HostAdmission")


class LocalSavedPair:
    """Actual completed local writer and unchanged inference/resume files."""

    def __init__(self, token: object, owner: Any, directory: Path, result: Json,
                 *, snapshot: Any = None) -> None:
        if token is not _SAVED:
            raise ContractError("actual local completed save required")
        self._owner, self.directory, self.snapshot = owner, directory, snapshot
        self._files = dict(result["files"])
        self.inference_digest = self._files["inference.json"]
        self.resume_digest = self._files["resume.json"]

    @property
    def files(self) -> Json:
        return dict(self._files)

    def require(self, owner: Any) -> None:
        owner.require()
        if owner is not self._owner or owner._saved.get(self.directory) is not self:
            raise ContractError("saved pair differs from original local writer")
        if self.snapshot is not None:
            self.snapshot.save_ack.require(self.snapshot)
        for name, expected in self._files.items():
            if sha256(self.directory / name) != expected:
                raise ContractError("actual saved pair original changed")

    def __reduce__(self) -> Any:
        raise TypeError("local saved writer authority cannot be serialized")


class _LocalLearner:
    def _own(self, admission: HostAdmission | None) -> None:
        self.admission = admission
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._failed = False
        self._saved: dict[Path, LocalSavedPair] = {}

    def require(self) -> None:
        if self._failed or (self._pid, self._thread) != (os.getpid(), threading.get_ident()):
            raise ContractError("original healthy local learner owner required")
        if self.admission is not None:
            self.admission.require()

    def _record_save(self, directory: Path, result: Json, *, snapshot: Any = None) -> LocalSavedPair:
        pair = LocalSavedPair(_SAVED, self, directory, result, snapshot=snapshot)
        self._saved[directory] = pair
        pair.require(self)
        return pair

    def __reduce__(self) -> Any:
        raise TypeError("local numerical owner cannot be serialized")


class LocalGameLearner(_LocalLearner):
    def __init__(self, model: Any, seed: int, binding: Any, resume_identity: str,
                 *, actions: Any, rules: Any, admission: HostAdmission | None = None) -> None:
        _admit_device(binding, admission)
        from gf_training.structured_numeric import GameLearner

        self._own(admission)
        with serialized_torch():
            self.learner = GameLearner(model, seed, binding, resume_identity,
                actions=actions, rules=rules)

    def update(self, rollout: Any) -> Json:
        self.require()
        with serialized_torch():
            try:
                return self.learner.update(rollout)
            except BaseException:
                self._failed = True
                raise

    def save(self, directory: Path, **options: Any) -> LocalSavedPair:
        self.require()
        directory = directory.absolute()
        with serialized_torch():
            try:
                result = self.learner.save(directory, **options)
                return self._record_save(directory, result)
            except BaseException:
                self._failed = True
                raise

    def restore(self, directory: Path) -> None:
        self.require()
        with serialized_torch():
            self.learner.restore(directory)

    def restore_policy(self, policy: Any) -> None:
        from gf_training.rust_player_restore import RustRestoredPolicy
        if type(policy) is not RustRestoredPolicy:
            raise ContractError('actual fully audited restored policy required')
        self.require()
        if policy.resume_manifest is None:
            raise ContractError('actual full restored resume metadata required')
        policy.require_restore(self, policy.resume_manifest)
        with serialized_torch():
            self.learner.restore(policy.directory,
                expected_resume_sha256=policy._held['resume.json'].sha256)
        policy.require()


class LocalHpoLearner(_LocalLearner):
    def __init__(self, model: Any, training: Any, seed: int, binding: Any,
                 resume_identity: str, *, actions: Any, rules: Any, budget: Any,
                 admission: HostAdmission | None = None,
                 device_admission: Json | None = None) -> None:
        _admit_device(binding, admission)
        from gf_training.hpo_numeric import HpoLearner

        self._own(admission)
        self.budget, self.actions = budget, actions
        with serialized_torch():
            self.learner = HpoLearner(model, training, seed, binding, resume_identity,
                actions=actions, rules=rules, _owned_device_admission=device_admission)
            self.learner._owned_source_budget = budget

    def update_owned(self, rollout: Any) -> Json:
        self.require()
        with serialized_torch():
            try:
                return self.learner.update_owned(rollout)
            except BaseException:
                self._failed = True
                raise

    def snapshot_owned(self, **options: Any) -> Any:
        self.require()
        with serialized_torch():
            return self.learner.snapshot_owned(budget=self.budget, **options)

    def restore_snapshot(self, snapshot: Any, *, resume_manifest: Json) -> None:
        self.require()
        from gf_training.hpo_artifacts import _restore_owned_learner

        with serialized_torch():
            _restore_owned_learner(self.learner, inference=snapshot.inference,
                resume=snapshot.resume, actor_metadata=None, resume_manifest=resume_manifest,
                generation=snapshot.generation, budget=self.budget, actions=self.actions,
                local_snapshot=snapshot)

    def restore_policy(self, policy: Any) -> None:
        from gf_training.hpo_artifacts import _restore_owned_learner
        from gf_training.rust_player_restore import RustRestoredPolicy
        if type(policy) is not RustRestoredPolicy:
            raise ContractError('actual fully audited restored policy required')
        self.require()
        if policy.resume_manifest is None:
            raise ContractError('actual full restored resume metadata required')
        policy.require_restore(self, policy.resume_manifest)
        with serialized_torch():
            _restore_owned_learner(self.learner, inference=policy.inference, resume=policy.resume,
                actor_metadata=None, resume_manifest=policy.resume_manifest, generation=policy.generation,
                budget=self.budget, actions=self.actions, restored_policy=policy)
        policy.require()

    def save(self, directory: Path, *, save_budget_bytes: int = 512 * 1024**2, **options: Any) -> LocalSavedPair:
        self.require()
        directory = directory.absolute()
        with serialized_torch():
            try:
                snapshot = self.snapshot_owned(save_budget_bytes=save_budget_bytes)
                ack = snapshot.save(directory, **options)
                ack.require(snapshot)
                return self._record_save(directory, {"files": ack.files}, snapshot=snapshot)
            except BaseException:
                self._failed = True
                raise


class RustGameOperations:
    """Closed Model3 operations for the existing common PPO coordinator.

    Local numerical owners and native collection remain distinct actual objects;
    no child-process facade, legacy ledger or configurable training loop exists.
    """

    formal = None
    registry_schema = 5
    _is_hpo = False

    def __init__(self, control: Any, loaded: Any, admission: HostAdmission) -> None:
        from dataclasses import asdict

        from gf_common.domain import canonical, digest
        from gf_training.rust_player_supervision import HostControl
        if type(control) is not HostControl or type(admission) is not HostAdmission:
            raise ContractError('actual same-host control/admission required')
        control._require()
        admission.require()
        if (admission._control is not control or control.operation['schema_version'] != 2
                or control.operation['operation_type'] != ('hpo-train' if self._is_hpo else 'ordinary-train')
                or control.operation['config_sha256'] != loaded.sha256
                or canonical(loaded.effective) != canonical(asdict(loaded.config))
                or digest(loaded.effective) != loaded.sha256):
            raise ContractError('actual operation2/LoadedConfig/admission binding required')
        self.control, self.loaded, self.admission = control, loaded, admission
        self.config = loaded.config
        self.state: Any = None
        self.pool: Any = None
        self.writer: Any = None
        self.registry: Any = None
        self.collector: Any = None
        self._ownership: Any = None
        self.host_health: Json
        self.resume_audit: Any = None
        self._failure_traceback: Any = None
        self._outgoing_owner: Any = None
        self.validate_config(self.config)

    def validate_config(self, config: Any) -> Any:
        from gf_common.card_instance import ActionConfigV3
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.config import PPOTrainingConfig, SequencePPOTrainingConfig
        from gf_models.structured_binding import ConfiguredNeuralModelV3
        if (config is not self.config or type(config.actions) is not ActionConfigV3
                or type(config.model) is not ConfiguredNeuralModelV3 or config.model.model_id != 'mlp'
                or config.model.checkpoint is not None or type(config.training) is not PPOTrainingConfig
                or isinstance(config.training, SequencePPOTrainingConfig)
                or config.run.split != 'train' or config.preprocessing.backend != 'python'
                or config.preprocessing.reader != 'participant-structured-v2'
                or not hasattr(config.run, 'player_runtime')):
            raise ContractError('actual Rust Model3 MLP/Python training profile required')
        return config.model.architecture(layout=layout)

    def deadlines(self) -> dict[str, float]:
        from gf_common.rust_player import deadline_seconds
        self.admission.require()
        return {key: deadline_seconds(value) for key, value in self.control.operation['deadlines'].items()}

    def run_scope(self, directory: Path, *, mode: str) -> Any:
        from contextlib import contextmanager, nullcontext

        from gf_training.group_pack_pool import PackedArtifactPool
        from gf_training.hpo_owned_lineage import OwnedControlWriter
        from gf_training.lineage import _HELD_RUN_OWNERS, borrow_run_lock, require_hpo_run_owner

        @contextmanager
        def held() -> Any:
            if directory != Path(self.control.operation['parent_directory']) or mode != self.control.operation['execution_request']['mode']:
                raise ContractError('actual parent/request ownership differs')
            if self._outgoing_owner is not None:
                actual = require_hpo_run_owner(directory, self._outgoing_owner)
                if _HELD_RUN_OWNERS[directory].control is not self.control:
                    raise ContractError('actual transferred owner belongs to another control')
                scope: Any = nullcontext(actual)
            else:
                scope = borrow_run_lock(directory, self.control)
            with scope as ownership:
                self._ownership = ownership
                # Cleanup may fund final parent writes under the original term
                # cutoff; every Source/numeric operation separately uses original
                # execution bits. No later call creates a fresh grace period.
                self.pool = PackedArtifactPool(max_owned_bytes=self.admission.observation['budget']['source_workspace_bytes'],
                    deadline=self.deadlines()['term_bits'])
                self.writer = OwnedControlWriter(ownership, budget=self.pool.budget,
                    limit=min(self.pool.budget.limit, self.config.hpo_spec['operating']['parent_budget_bytes']
                        if self._is_hpo else self.config.training.max_owned_bytes))
                # The fixed handler retains this SAME pool for the final
                # protocol serialization after the borrower FD closes. Data
                # loans retire earlier; the control allowance closes after ACK.
                yield ownership
                if self.resume_audit is not None:
                    self.resume_audit.close()
        return held()

    def preprocessing_identity(self, config: Any, *, deadline: float) -> Json:
        from gf_common.card_instance_rules import card_instance_rules
        from gf_models.structured_binding import input_binding_game_v3
        from gf_training.group_pack_resources import remaining
        self.validate_config(config)
        remaining(deadline)
        return {'backend': 'python', 'reader_contract': 'participant-structured-v2',
            'implementation': 'python-reference-player-card-refs-v2',
            **input_binding_game_v3(config.actions.action_contract, card_instance_rules())}

    def preflight(self, config: Any, *, deadline: float) -> None:
        from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
        self.validate_config(config)
        self.runtime = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir),
            contract=config.actions.action_contract, deadline=deadline)
        self.runtime.preflight()

    def integration_identity(self, config: Any, deadline: float) -> Json:
        from gf_common.card_instance import reviewed_specs
        from gf_models.arrays import sha256
        receipt = self.runtime.preflight()
        return {'binary_sha256': receipt['binary']['sha256'],
            'server_config_sha256': sha256(Path(config.server.config), deadline),
            'declared_simulator_revision': config.server.simulator_revision, 'binary_build_revision': None,
            'source_revision': receipt['source']['revision'], 'source_dirty': receipt['source']['dirty'],
            'source_spec_sha256': reviewed_specs(config.actions.action_contract)}

    def decorate_identity(self, identity: Json) -> None:
        from gf_common.domain import digest
        from gf_common.rust_player import runtime_binding
        actual = self.admission.observation
        bound = runtime_binding(self.config.hpo_spec['runtime'] if self._is_hpo else self.config.run.player_runtime, actual['producer'])
        if digest(bound) != actual['runtime_binding_sha256']:
            raise ContractError('actual admitted runtime binding differs')
        identity['player_runtime'] = bound

    def execution(self, config: Any) -> Json:
        from gf_training.structured_ppo import execution
        return execution(config)

    def initialize_state(self, state: Json, config: Any) -> None:
        from gf_common.card_instance_rules import card_instance_rules
        from gf_common.group_pack_game import storage_binding
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.structured_binding import input_binding_game_v3
        from gf_training.rust_player_lineage import resume_identity
        actual = self.admission.observation
        classification = input_binding_game_v3(config.actions.action_contract, card_instance_rules())
        state.update(schema_version=5, run_variant='ppo-rust-player-v1', model_id='mlp', model_version='3',
            input_recipe=layout.input_recipe, execution=self.execution(config), **classification,
            storage_binding=storage_binding(), runtime_binding_sha256=actual['runtime_binding_sha256'],
            monotonic_authority={'boot_id': self.control.launch['clock_binding']['boot_id'],
                'deadlines': self.control.operation['deadlines']}, invocations={}, host_result=None)
        state['resume_identity'] = resume_identity(effective_config=state['effective_config'],
            identity=state['identity'], execution=state['execution'], classification=classification,
            storage_binding=state['storage_binding'])

    def persist_control(self, path: Path, value: Json, config: Any, *, deadline: float, original: Any = None) -> str:
        if self.writer is None:
            raise ContractError('actual current parent writer required')
        if path.name == 'run.json' and value['status'] in ('complete', 'paused', 'incomplete'):
            if (not hasattr(self, 'host_health') or self.host_health['errors']
                    or not all(self.host_health[name] for name in ('actors_joined', 'service_joined',
                        'writers_joined', 'servers_joined', 'buffers_released', 'gpu_fenced'))):
                raise ContractError('actual complete host retirement required before awaiting finalization')
            value['host_result'] = 'awaiting-supervised-finalization'
        return self.writer.write(path, value, deadline=deadline)

    def bind_run(self, ownership: Any, state: Json) -> None:
        from gf_training.lineage import require_hpo_run_owner
        if require_hpo_run_owner(ownership.directory, ownership) is not self._ownership:
            raise ContractError('actual same parent numerical mutation owner required')
        self.state = state

    def read_cold(self, ownership: Any, config: Any, *, deadline: float) -> Any:
        from gf_training.rust_player_restore import RustResumeAudit
        self.preflight(config, deadline=deadline)
        self.resume_audit = RustResumeAudit(self.control, self.admission, ownership,
            loaded=self.loaded, pool=self.pool, writer=self.writer)
        return self.resume_audit.state, [self.resume_audit.records[generation]
            for generation in range(self.resume_audit.state['updates'] + 1)]

    def restore_initial(self, local: Any, state: Json, *, deadline: float) -> Json:
        from gf_models.torch_runtime import hardware
        from gf_training.group_pack_resources import remaining
        remaining(deadline)
        audit = self.resume_audit
        policies = audit.load(local)
        current = policies[state['updates']]
        local.restore_policy(current)
        self.verify_restored_state(local, current)
        audit.scan.finalize()
        self.registry.adopt_restored(audit, local)
        for generation in sorted(policies):
            self.registry.publish(generation, deadline=deadline)
        current.retire_resume()
        self.last_source_manifest_sha256 = audit.last_source_manifest_sha256
        if self._is_hpo:
            assert type(self) is RustHpoOperations
            self.sample_cuda('actual-restored-banks-published')
        return {'hardware': hardware(local.learner.binding.device)}

    def verify_restored_state(self, local: Any, policy: Any) -> None:
        import numpy as np

        from gf_common.domain import canonical
        if policy.resume is None or policy.resume_manifest is None:
            raise ContractError('actual full restored state comparison requires resume banks')
        learner = local.learner
        # Reuse the SAME saved-inspection scratch high-water, not a second
        # concurrent phase allowance. One FP32 D2H tensor plus bool comparison.
        largest = max(value.numel() for value in learner.network.state_dict().values())
        policy.audit.scan.charge(5 * largest + 65536, temporary=True)
        for name, value in learner.network.state_dict().items():
            if not np.array_equal(value.detach().cpu().numpy(), policy.inference.array(name)):
                raise ContractError('actual restored named weights differ from held banks')
        for name, parameter in learner.network.named_parameters():
            saved = policy.resume_manifest['parameter_states'][name]['present']
            actual = learner.optimizer.state.get(parameter)
            if saved != bool(actual):
                raise ContractError('actual restored named Adam presence differs')
            if saved:
                for key in ('exp_avg', 'exp_avg_sq'):
                    if not np.array_equal(actual[key].detach().cpu().numpy(), policy.resume.array(f'adam.{name}.{key}')):
                        raise ContractError('actual restored named Adam tensor differs')
                if int(actual['step'].item()) != int(policy.resume.array(f'adam.{name}.step')):
                    raise ContractError('actual restored named Adam step differs')
        if not np.array_equal(learner.torch.get_rng_state().numpy(), policy.resume.array('rng.torch_cpu')):
            raise ContractError('actual restored CPU RNG differs')
        if learner.binding.device != 'cpu':
            for index, value in enumerate(learner.torch.cuda.get_rng_state_all()):
                if not np.array_equal(value.numpy(), policy.resume.array(f'rng.torch_cuda.{index}')):
                    raise ContractError('actual restored CUDA RNG differs')
        if canonical(learner.shuffle.bit_generator.state) != canonical(policy.resume_manifest['shuffle_rng']):
            raise ContractError('actual restored PCG64 differs')
        self.writer.write(self._ownership.directory / 'retirement' / f'restore-{self.admission.observation["invocation_id"]}.json',
            {'schema_version': 1, 'kind': 'rust-player-restored-state-comparison-v1',
                'named_weights_equal': True, 'named_adam_equal': True, 'cpu_rng_equal': True,
                'cuda_rng_equal': True, 'pcg64_equal': True, 'optimizer_steps': learner.steps,
                'original_lineage_identity': policy.identity, 'current_bank_identity': policy.current_bank_identity,
                'current_inference_backings': policy._inference_backings}, deadline=self.deadlines()['execution_bits'], immutable=True)

    def initial_registry(self, state: Json) -> Json:
        result = {'schema_version': self.registry_schema, 'run_id': state['run_id'],
            'runtime_binding_sha256': state['runtime_binding_sha256'], 'generations': {},
            'evaluation_references': {}, 'service_references': {}}
        if self._is_hpo:
            result.update(origin='producer-owned-v1', encoder='canonical-python')
        return result

    def external_roots(self, state: Json) -> tuple[Path, ...]:
        root = Path(self.config.storage.db_root)
        return (root / state['run_id'], *(root / name for name in (
            'catalog.sqlite3', 'catalog.sqlite3-wal', 'catalog.sqlite3-shm', 'catalog.sqlite3-journal')))

    def make_child(self, directory: Path, config: Any, state: Json, *, deadline: float) -> tuple[Any, Any]:
        from dataclasses import replace

        from gf_common.card_instance_rules import card_instance_rules
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.rust_player import RustPlayerInference
        from gf_training.rust_player_collection import RustPlayerCollector
        from gf_training.rust_player_registry import RustRunRegistry
        if directory != self._ownership.directory or state is not self.state:
            raise ContractError('actual same parent/local run required')
        binding = replace(config.model.inference, device=config.training.device)
        if self._is_hpo:
            from gf_training.hpo_config import hpo_training_config
            assert type(self) is RustHpoOperations
            self._checked_device_admission = self.component_device_admission()
            local: Any = LocalHpoLearner(config.model.architecture(layout=layout),
                hpo_training_config(config.hpo_spec['training_recipe']), config.training.seeds[0],
                binding, state['resume_identity'], actions=config.actions, rules=card_instance_rules(),
                admission=self.admission, budget=self.pool.budget,
                device_admission=getattr(self, '_checked_device_admission', None))
        else:
            local = LocalGameLearner(config.model.architecture(layout=layout), config.training.seeds[0],
                binding, state['resume_identity'], actions=config.actions, rules=card_instance_rules(), admission=self.admission)
        # Five fixed normal pool policies + current + the newly prepared next
        # generation. This is cache occupancy, not per-thread/process memory.
        service = RustPlayerInference(device=config.model.inference.device,
            cache_entries=config.hpo_spec['operating']['opponent_pool_size'] + 2 if self._is_hpo else 7,
            cache_weight_bytes=self.admission.observation['budget']['managed_ram_bytes']
                if config.model.inference.device == 'cpu' else self.admission.observation['budget']['managed_cuda_bytes'],
            visibility=config.model.cuda_visible_devices)
        self.registry = RustRunRegistry(self._ownership, state, config=config, admission=self.admission,
            pool=self.pool, writer=self.writer, local=local, service=service)
        self.collector = RustPlayerCollector(self.registry)
        if self._is_hpo:
            assert type(self) is RustHpoOperations
            self.sample_cuda('local-service-initialized')
        return local, self.registry

    def initialize_local(self, local: LocalGameLearner, output: Path, *, deadline: float) -> Json:
        from gf_models.torch_runtime import hardware
        from gf_training.group_pack_resources import remaining
        remaining(deadline)
        if self.registry.local is not local:
            raise ContractError('actual current local numerical owner required')
        pair = local.save(output, **({'save_budget_bytes': self.config.hpo_spec['operating']['save_budget_bytes']} if self._is_hpo else {}))
        remaining(deadline)
        return {'pair': pair, 'hardware': hardware(local.learner.binding.device)}

    def binding(self, directory: Path, generation: int, config: Any, deadline: float) -> Any:
        from gf_training.group_pack_resources import remaining
        remaining(deadline)
        return self.registry.binding(generation)

    def linkage(self, state: Json, directory: Path, generation: int, groups: Any, bindings: Any) -> Json:
        from gf_training.card_instance_source import frozen_linkage
        return frozen_linkage(state, directory, generation, groups, bindings)

    def validate_owned_collection(self, plan: Any, registry: Any, *, config: Any, deadline: float) -> None:
        from gf_training.group_pack_resources import remaining
        remaining(deadline)
        self.registry.require()
        if registry is not self.registry or registry.pending is not None or self.state['updates'] != plan.linkage['collection_generation']:
            raise ContractError('actual unused same-generation native collection required')

    def accept_collection(self, collection: Any, config: Any) -> None:
        from gf_training.rust_player_collection import RustCollectedBatch
        if type(collection) is not RustCollectedBatch:
            raise ContractError('actual native/writer collection required')
        collection.require(self.registry, collection.plan)
        self.finalized_collection = collection

    async def collect(self, plan: Any, registry: Any) -> Any:
        if registry is not self.registry:
            raise ContractError('actual current native collection registry required')
        collection = await self.collector.collect(plan, registry)
        directory = self._ownership.directory / 'retirement'
        directory.mkdir(exist_ok=True)
        accounting = {'schema_version': 1, 'kind': 'rust-player-source-accounting-v1',
            'phase': 'sealed-before-source', 'source_limit': self.pool.budget.limit,
            'source_used': self.pool.budget.used, 'source_peak': self.pool.budget.peak,
            'writer_holder_absent': collection.writer._workspace_lease is None,
            'inventory_holder_absent': self.collector._inventory is None,
            'static_context_absent': self.collector._source_control_charge is None,
            'pool_base': self.pool._base_charge,
            'inference_bank_data_bytes': sum(policy.inference.size for policy in self.registry.policies.values()),
            'saved_metadata_bytes': sum(charge.size for policy in self.registry.policies.values() for charge in policy._charges)}
        self.writer.write(directory / f'accounting-{plan.linkage["batch_id"]}.json', accounting,
            deadline=plan.execution_deadline, immutable=True)
        if self._is_hpo:
            assert type(self) is RustHpoOperations
            self.sample_cuda('collection-settled')
        return collection

    def project(self, collection: Any, plan: Any, numeric: Path, config: Any, deadline: float) -> Any:
        from contextlib import contextmanager

        from gf_training.rust_player_source import RustPlayerSource

        @contextmanager
        def projection() -> Any:
            import traceback
            collection.require(self.registry, plan)
            source = RustPlayerSource.project(collection)
            try:
                source.persist(numeric)
                self.last_consumer_settlement_sha256 = source.manifest['logical_retirement']['service_settlement_sha256']
                yield source.decision_count, source.proof, dict(source.numeric_files), source
            except BaseException as error:
                # Stopped producer/kernel frames can still own arrays and
                # metadata. Retire those aliases before their concrete Source
                # loans; executing common-loop aliases retire via its scope.
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                source.close()
        return projection()

    def update_local(self, source: Any, output: Path, *, deadline: float) -> Json:
        from gf_training.group_pack_resources import remaining
        from gf_training.rust_player_source import RustPlayerSource
        remaining(deadline)
        if type(source) is not RustPlayerSource or source.registry is not self.registry:
            raise ContractError('actual same-host persisted Source required')
        metrics, pair = source.consume(output)
        if self._is_hpo:
            assert type(self) is RustHpoOperations
            self.sample_cuda('update-saved')
        return {'metrics': metrics, 'timings': {}, 'pair': pair}

    def update_record(self, update: Json, state: Json, numeric_files: Json) -> None:
        from gf_models.structured_binding import input_binding_game_v3
        source = self.registry.pending
        update.update(schema_version=6, model_id='mlp', model_version='3', input_recipe=state['input_recipe'],
            **input_binding_game_v3(self.config.actions.action_contract, self.registry.local.learner._context.rules),
            numeric_files=numeric_files, runtime_binding_sha256=state['runtime_binding_sha256'],
            collection_retirement_sha256=source.collection.retirement_sha256,
            service_usage=source.manifest['service_usage'], parent_snapshot_identity=source.parent.identity)
        # Result identity is inserted by registry.commit only after its genuine
        # provisional saved-policy owner exists, before the complete Source guard.

    def commit(self, directory: Path, state: Json, config: Any, *, generation: int, update: Json | None,
               controls: Json, acknowledgement: Json, deadline: float) -> Json:
        from gf_common.domain import canonical
        from gf_training.group_pack_resources import remaining
        remaining(deadline)
        if self.registry.state is not state or directory != self.registry.directory or generation != state['updates'] + int(update is not None):
            raise ContractError('actual same-parent saved commit required')
        row = self.registry.commit(acknowledgement['pair'], controls=controls, update=update, deadline=deadline)
        if canonical(self.registry.records[generation]) != canonical(row):
            raise ContractError('actual completed generation differs from current registry')
        self.registry.publish(generation, deadline=deadline)
        if self._is_hpo:
            assert type(self) is RustHpoOperations
            self.sample_cuda(f'generation-{generation}-published')
        if update is not None:
            self.last_source_manifest_sha256 = update['collection_manifest_sha256']
        return row

    def close_batch(self) -> None:
        if self.registry is not None and self.registry.pending is not None:
            self.registry.pending.close()

    def retain(self, directory: Path, generation: int, config: Any, deadline: float) -> None:
        if directory != self.registry.directory or generation != self.registry.state['updates']:
            raise ContractError('actual current normal generation retention required')
        self.registry.retain_ordinary(self.collector.runtime,
            consumer_settlement_sha256=self.last_consumer_settlement_sha256, deadline=deadline)

    def enter_invocation(self, state: Json, mode: str, started: float) -> None:
        from gf_common.domain import clone, digest
        actual = self.admission.observation
        invocation = actual['invocation_id']
        if invocation in state['invocations']:
            raise ContractError('actual invocation is single use')
        state['invocations'][invocation] = {
            'operation_path': str(self.control.operation_path), 'operation_sha256': self.control.operation_sha256,
            'execution_request': clone(self.control.operation['execution_request']),
            'admission_path': str(self.admission.path), 'admission_sha256': digest(actual),
            'mode': mode, 'elapsed_seconds': 0., 'host_health': None}

    def finish_invocation(self, state: Json, facts: Json) -> None:
        actual = self.admission.observation
        state['invocations'][actual['invocation_id']].update(facts)

    async def close_local(self, local: LocalGameLearner, *, deadline: float,
                          error: BaseException | None = None) -> None:
        import asyncio

        from gf_common.domain import clone, digest
        actual = self.admission.observation
        invocation, run_id = actual['invocation_id'], actual['run_id']
        directory = self._ownership.directory / 'retirement'
        directory.mkdir(exist_ok=True)
        self.host_health = dict.fromkeys(('actors_joined', 'service_joined', 'writers_joined',
            'servers_joined', 'buffers_released', 'gpu_fenced'), False)
        self.host_health.update(dict.fromkeys(('native_retirement_sha256', 'writer_retirement_sha256',
            'service_retirement_sha256', 'cache_retirement_sha256')))
        self.host_health['errors'] = []
        failures: list[BaseException] = []
        def failed(stage: str, failure: BaseException) -> None:
            failures.append(failure)
            self.host_health['errors'].append(stage + ':' + type(failure).__name__)
            vars(failure)['cleanup_owner'] = self
        try:
            self.close_batch()
        except BaseException as failure:  # noqa: BLE001 -- independently stop original producers too.
            failed('source', failure)
        runtime = self.collector.runtime
        if error is not None or failures:
            try:
                runtime.cancel('failed parent retirement')
            except BaseException as failure:  # noqa: BLE001 -- retain actual runtime, continue other owners.
                failed('cancel', failure)
        servers = []
        for item in self.collector.server_history:
            server, receipt = item['server'], item['runtime_evidence']
            try:
                health = await server.stop()
                if (not server.stopped or server.process is None or server.process.returncode is None
                        or health.get('log_error') is not None or health.get('reap_failure') is not None
                        or any(not task.done() or task.cancelled() or task.exception() is not None for task in server.drainers)):
                    raise ContractError('actual original server/log retirement remains unconfirmed')
                if receipt is None:
                    raise ContractError('actual server identity was not captured before failure')
                code = server.process.returncode
                servers.append({'pid': receipt['executed']['pid'], 'start_ticks': receipt['executed']['start_time'],
                    'exit_code': code if code >= 0 else None, 'signal': -code if code < 0 else None,
                    'logs_closed': True})
            except BaseException as failure:  # noqa: BLE001 -- one failed server cannot skip another original.
                failed('server', failure)
        self.host_health['servers_joined'] = len(servers) == len(self.collector.server_history)
        writers = []
        for item in self.collector.writer_history:
            writer = item['writer']
            try:
                if not writer._sealed:
                    writer.cancel('failed parent retirement')
                report = await asyncio.to_thread(writer.join)
                writers.append({'directory': str(writer.directory),
                    'descriptor_sha256': item['descriptor_sha256'], 'report': report})
            except BaseException as failure:  # noqa: BLE001 -- keep actual writer and attempt native retirement.
                failed('writer', failure)
        self.host_health['writers_joined'] = len(writers) == len(self.collector.writer_history)
        try:
            runtime.quiesce()
            report = await asyncio.to_thread(runtime.join)
            if runtime._holds or runtime._active_routes or runtime.dispatcher.checkpoint_references() or any(self.registry.service.references.values()):
                raise ContractError('actual service/actor references remain after original joins')
            with serialized_torch():
                local.learner.synchronize()
                if self.registry.service.device != 'cpu':
                    import torch
                    torch.cuda.synchronize(self.registry.service.device)
            self.host_health.update(actors_joined=True, service_joined=True, gpu_fenced=True)
        except BaseException as failure:  # noqa: BLE001 -- no retirement proof or refunds for uncertain consumers.
            failed('native-service', failure)
        if failures:
            primary = failures[0]
            for later_failure in failures[1:]:
                primary.add_note('independent original cleanup: ' + str(later_failure))
            raise primary
        native = {'schema_version': 1, 'kind': 'rust-player-native-retirement-v1',
            'run_id': run_id, 'invocation_id': invocation, 'report': report, 'report_sha256': digest(report)}
        native_sha = self.writer.write(directory / f'native-{invocation}.json', native, deadline=deadline, immutable=True)
        settlement = {'schema_version': 1, 'kind': 'rust-player-host-consumer-settlement-v1',
            'run_id': run_id, 'invocation_id': invocation, 'service_id': self.registry.service_id,
            'native_retirement_sha256': native_sha, 'active_policy_references': 0,
            'native_pending': 0, 'started_producers': 0, 'gpu_fenced': True}
        settlement_sha = self.writer.write(directory / f'consumers-{invocation}.json', settlement, deadline=deadline, immutable=True)
        service = {'schema_version': 1, 'kind': 'rust-player-service-retirement-v1',
            'run_id': run_id, 'invocation_id': invocation, 'service_id': self.registry.service_id,
            'native_retirement_sha256': native_sha, 'consumer_settlement_sha256': settlement_sha,
            'service_joined': True, 'gpu_fenced': True}
        service_sha = self.writer.write(directory / f'service-{invocation}.json', service, deadline=deadline, immutable=True)
        cache = self.registry.retire_cache(runtime, service_retirement_sha256=service_sha,
            consumer_settlement_sha256=settlement_sha, deadline=deadline)
        cache_sha = self.writer.write(directory / f'cache-{invocation}.json', cache, deadline=deadline, immutable=True)
        writer_record = {'schema_version': 1, 'kind': 'rust-player-writers-retirement-v1',
            'run_id': run_id, 'invocation_id': invocation, 'writers': writers}
        writer_sha = self.writer.write(directory / f'writers-{invocation}.json', writer_record, deadline=deadline, immutable=True)
        account = runtime.recording_account()
        if account.native_bytes or account.external_bytes:
            raise ContractError('actual recording buffers remain live after all last consumers')
        self.host_health.update(buffers_released=True, native_retirement_sha256=native_sha,
            writer_retirement_sha256=writer_sha, service_retirement_sha256=service_sha,
            cache_retirement_sha256=cache_sha)
        if error is not None:
            self.host_health['errors'].append(type(error).__name__)
        self.servers_retired = clone(servers)
        if self._is_hpo:
            assert type(self) is RustHpoOperations
            self.sample_cuda('consumer-cache-retired')

    def freeze_state(self, state: Json, *, deadline: float) -> None:
        from gf_common.domain import digest
        from gf_training.rust_player_lineage import validate_run
        validate_run(state, hpo=self._is_hpo)
        if state['host_result'] != 'awaiting-supervised-finalization' or state['status'] not in ('complete', 'paused', 'incomplete'):
            raise ContractError('actual healthy computational boundary required')
        actual = self.admission.observation
        invocation = actual['invocation_id']
        current = state['invocations'][invocation]
        if (current['operation_sha256'] != self.control.operation_sha256
                or current['operation_path'] != str(self.control.operation_path)
                or current['execution_request'] != self.control.operation['execution_request']
                or current['admission_sha256'] != digest(actual)):
            raise ContractError('actual frozen invocation operation/request/admission differs')
        path = self._ownership.directory / f'state-{invocation}.json'
        checksum = self.writer.write(path, state, deadline=deadline, immutable=True)
        run_receipt = self.writer._completed[self._ownership.directory / 'run.json']
        run_receipt.require(self.writer, self._ownership.directory / 'run.json', checksum)
        self.frozen_state_path, self.frozen_state_sha256 = path, checksum

    def _write_protocol(self, path: Path, body: Json, *, deadline: float) -> str:
        from gf_common.artifacts import json_bound
        from gf_training.group_pack_resources import remaining
        from gf_training.rust_player_supervision import write_control_record
        with self.writer.parent_budget.reserve(32 * json_bound(body) + 65536):
            remaining(deadline)
            checksum = write_control_record(path, body)
            remaining(deadline)
            return checksum

    def prepare(self) -> Any:
        from gf_common.artifacts import json_bound
        from gf_common.domain import digest
        from gf_training.group_pack_resources import remaining
        from gf_training.rust_player_supervision import write_control_record
        # run_scope has closed its original borrower FD. The supervisor still
        # owns the actual OFD; this fixed control phase owns no parent mutations.
        if self.control._parent_borrowers:
            raise ContractError('actual parent borrowers must close before Prepared')
        actual = self.admission.observation
        invocation = actual['invocation_id']
        deadline = self.deadlines()['term_bits']
        remaining(deadline)
        if self.pool.sources or self.pool._close_holds or self.host_health['errors']:
            raise ContractError('actual Source/control retirement remains incomplete')
        common = {key: actual[key] for key in ('supervision_id', 'invocation_id', 'operation_id',
            'run_id', 'runtime_binding_sha256')}
        common['admission_sha256'] = digest(actual)
        servers = {'schema_version': 1, 'kind': 'rust-player-server-retirement-v1',
            **{key: common[key] for key in ('supervision_id', 'invocation_id', 'operation_id')},
            'servers': self.servers_retired}
        servers_path = self.control.root / f'servers-{invocation}.json'
        servers_sha = self._write_protocol(servers_path, servers, deadline=deadline)
        close = {'schema_version': 1, 'kind': 'rust-player-host-close-ack-v1', **common,
            'actors_joined': self.host_health['actors_joined'], 'service_joined': self.host_health['service_joined'],
            'writer_joined': self.host_health['writers_joined'], 'buffers_released': self.host_health['buffers_released'],
            'gpu_fenced': self.host_health['gpu_fenced'],
            **{key: self.host_health[key] for key in ('native_retirement_sha256', 'writer_retirement_sha256',
                'service_retirement_sha256', 'cache_retirement_sha256')},
            'servers_path': str(servers_path), 'servers_sha256': servers_sha, 'server_count': len(self.servers_retired)}
        close_path = self.control.root / f'close-{invocation}.json'
        close_sha = self._write_protocol(close_path, close, deadline=deadline)
        row = self.registry.records[self.state['updates']]
        prepared = {'schema_version': 1, 'kind': 'rust-player-prepared-v1', **common,
            'config_sha256': self.loaded.sha256, 'state': self.state['status'],
            'committed_update': self.state['updates'], 'generation': self.state['updates'],
            'state_sha256': self.frozen_state_sha256, 'generation_sha256': self.state['generation_sha256'],
            'artifacts_sha256': digest(row['artifact_files']),
            'source_manifest_sha256': self.last_source_manifest_sha256,
            'close_ack_path': str(close_path), 'close_ack_sha256': close_sha}
        path = self.control.root / f'prepared-{invocation}.json'
        with self.writer.parent_budget.reserve(32 * json_bound(prepared) + 65536):
            remaining(deadline)
            write_control_record(path, prepared)
            remaining(deadline)
            return self.control.prepare(self.admission, path)

    def close_control_resources(self) -> None:
        if getattr(self, '_failure_traceback', None) is not None:
            import traceback
            traceback.clear_frames(self._failure_traceback)
        state = getattr(self, 'state', None)
        if state is not None:
            state.clear()
        if self.registry is not None:
            self.registry._drop_control_aliases()
        if self.resume_audit is not None:
            self.resume_audit.close()
        if self.registry is not None:
            self.registry._retire_restored_metadata()
        if self.pool is not None:
            self.pool.close()


def run_ordinary(control: Any, operation: Json, *, loaded: Any) -> Json:
    """Fixed internal ordinary entry; public command activation is separate."""
    return _run_local(control, operation, loaded=loaded, hpo=False)


def run_hpo(control: Any, operation: Json, *, loaded: Any) -> Json:
    """Fixed internal Model4 entry under actual global HPO admission."""
    return _run_local(control, operation, loaded=loaded, hpo=True)


def _configure_device_environment(config: Any) -> bool:
    gpu = config.training.device != 'cpu' or config.model.inference.device != 'cpu'
    if not gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif config.model.cuda_visible_devices is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.model.cuda_visible_devices
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return gpu


def _run_local(control: Any, operation: Json, *, loaded: Any, hpo: bool) -> Json:
    import asyncio
    import hashlib
    from contextlib import nullcontext
    from dataclasses import asdict
    from types import SimpleNamespace

    from gf_common.domain import canonical, digest
    from gf_common.rust_player import runtime_binding
    from gf_models.rust_player import sampling_runtime_identity
    from gf_training.lineage import run_lock, transfer_run_lock
    from gf_training.ppo_run import _run_ppo
    from gf_training.rust_player_supervision import HostControl, runtime_producer
    if (type(control) is not HostControl or operation is not control.operation
            or operation['schema_version'] != 2 or operation['operation_type'] != ('hpo-train' if hpo else 'ordinary-train')
            or canonical(loaded.effective) != canonical(asdict(loaded.config))):
        raise ContractError('actual known operation2/LoadedConfig required')
    config = loaded.config
    declaration = config.hpo_spec['runtime'] if hpo else config.run.player_runtime
    gpu = _configure_device_environment(config)
    # This is actual kernel-observed original supervisor data, not a substitute
    # control capability. Native admission independently checks this exact SHA.
    companion = SimpleNamespace(manifest={'executable_sha256': hashlib.sha256(
        Path(f'/proc/{control._parent_pid}/exe').read_bytes()).hexdigest()})
    producer = runtime_producer(companion, sampling_runtime_sha256=sampling_runtime_identity())
    bound = runtime_binding(declaration, producer)
    if digest(bound) != operation['runtime_binding_sha256'] or loaded.sha256 != operation['config_sha256']:
        raise ContractError('actual input/producer/runtime differs from operation2')
    budget = {'managed_ram_bytes': config.hpo_spec['operating']['managed_ram_bytes'] if hpo else 24 * 1024**3,
        'managed_cuda_bytes': config.hpo_spec['operating']['managed_cuda_bytes'] if hpo else config.run.global_cuda_budget_bytes if gpu else 0,
        'game_concurrency': declaration['game_concurrency'], 'player_worker_threads': declaration['player_worker_threads'],
        'max_batch_size': declaration['max_batch_size'], 'storage_bytes': config.storage.settings().budget_bytes,
        'source_workspace_bytes': config.preprocessing.memory_budget_mib * 1024**2}
    request = operation['execution_request']
    parent = Path(operation['parent_directory'])
    inherited = 'parent' in control._borrowed
    if not inherited and (control._phase != 'training' or control._cold is None or control._cold._finalized_sha256 is None):
        raise ContractError('actual finalized bind is required for outgoing parent acquisition')
    scope = nullcontext(None) if inherited else run_lock(parent)
    ops: Any = None
    try:
        with scope as outgoing:
            if outgoing is not None:
                transfer_run_lock(outgoing, control)
            admission = control.admit(budget, producer)
            ops = RustHpoOperations(control, loaded, admission) if hpo else RustGameOperations(control, loaded, admission)
            ops._outgoing_owner = outgoing
            report = asyncio.run(_run_ppo(loaded, ops.collect, parent,
                mode=request['mode'], pause_after_update=request['pause_after_update'], _game_ops=ops))
        # The outgoing ACK-owned context has now positively closed WITHOUT
        # LOCK_UN; native custody survives through Prepared/physical exit.
        prepared = ops.prepare()
        return {**report, 'computational_status': report['status'],
            'status': 'awaiting-supervised-finalization', 'prepared_sha256': prepared.sha256}
    except BaseException as error:
        if ops is not None:
            ops._failure_traceback = error.__traceback__
            vars(error).setdefault('cleanup_owner', ops)
        raise
    finally:
        if ops is not None:
            ops.close_control_resources()


def finalize_cold(control: Any, cold: Any, *, loaded: Any) -> Any:
    from gf_training.rust_player_completion import finalize_cold as actual_finalize
    return actual_finalize(control, cold, loaded=loaded)


class RustHpoOperations(RustGameOperations):
    """Exact Model4/spec6 operations; same common loop, real global admission."""

    registry_schema = 8
    _is_hpo = True
    resume_audit: Any

    def __init__(self, control: Any, loaded: Any, admission: HostAdmission) -> None:
        super().__init__(control, loaded, admission)
        actual = admission.observation
        if actual['global_ticket'] is None or actual['context'] != control.operation['expected_context']:
            raise ContractError('actual global HPO admission/context required')
        self.stop_reason: str | None = None
        self.coordinator_phase = 'startup'
        self._gpu_samples: list[Json] = []

    def sample_cuda(self, phase: str) -> None:
        import torch
        self.admission.require()
        with serialized_torch():
            self._gpu_samples.append({'phase': phase,
                'allocated_bytes': torch.cuda.memory_allocated(0),
                'reserved_bytes': torch.cuda.memory_reserved(0),
                'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                'process_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                'captured_policy_backings': {str(generation): policy._inference_backings
                    for generation, policy in self.registry.policies.items()}})
        directory = self._ownership.directory / 'retirement'
        directory.mkdir(exist_ok=True)
        self.writer.write(directory / f'cuda-{self.admission.observation["invocation_id"]}.json',
            {'schema_version': 1, 'kind': 'rust-player-process-cuda-metrics-v1',
                'scope': 'current process Torch allocator samples/cumulative peaks; not whole-device usage or hard quota',
                'samples': self._gpu_samples}, deadline=self.deadlines()['term_bits'])

    def validate_config(self, config: Any) -> Any:
        from dataclasses import asdict

        from gf_common.domain import canonical
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.structured_binding import ConfiguredNeuralModelV4
        from gf_training.hpo_config import hpo_training_config, normalize_training_for_model
        if (config is not self.config or type(config.model) is not ConfiguredNeuralModelV4
                or config.model.model_id != 'mlp' or config.model.checkpoint is not None
                or config.hpo_spec['schema_version'] != 6
                or config.training.device != 'cuda:0' or config.model.inference.device != 'cuda:0'
                or config.preprocessing.backend != 'python' or config.run.split != 'train'):
            raise ContractError('actual supported Model4/spec6 CUDA/Python HPO profile required')
        recipe = hpo_training_config(config.hpo_spec['training_recipe'])
        if canonical(asdict(normalize_training_for_model('mlp', recipe))) != canonical(config.hpo_spec['training_recipe']):
            raise ContractError('actual normalized Model4 recipe required')
        return config.model.architecture(layout=layout)

    def component_device_admission(self) -> Json:
        """Approved bounded component ONLY; no generic collection fit claim."""
        from math import prod

        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.learner_cuda import owned_learner_cuda_admission
        from gf_models.structured_config import parameter_shapes, resource_estimate
        config, actual = self.config, self.admission.observation
        model = config.model.architecture(layout=layout)
        recipe = config.hpo_spec['training_recipe']
        runtime = config.hpo_spec['runtime']
        request = self.control.operation['execution_request']
        logical = resource_estimate(model, config.model.inference,
            suffix_decisions=recipe['tbptt_length'], layout=layout)['total_bytes']
        if (config.hpo_spec['decision_budget'] != 64 or config.training.max_updates != 2
                or recipe['collection_games'] != 16 or runtime['game_concurrency'] != 1
                or runtime['player_worker_threads'] != 1 or runtime['max_batch_size'] != 1
                or logical != 2_966_109_712 or actual['budget']['managed_cuda_bytes'] != 12 * 1024**3
                or request['mode'] == 'fresh' and request['pause_after_update'] != 1
                or request['mode'] == 'resume' and self.resume_audit.state['used_decisions'] != 64):
            raise ContractError('general Rust CUDA phase planning is not approved; bounded component differs')
        weights = 4 * sum(prod(shape) for shape in parameter_shapes(model, layout=layout).values())
        positions = 4 * model.temporal.max_decisions * model.temporal.hidden_size
        # This component explicitly loads original pool0/current1 on resume;
        # there is no future update, but those two actual cache networks count.
        generations = {0, 1} if request['mode'] == 'fresh' else set(
            self.resume_audit.current['parent_controls']['snapshot_pool']) | {self.resume_audit.state['updates']}
        pool_bytes = len(generations) * (weights + positions)
        if pool_bytes != 15_492_112:
            raise ContractError('actual reachable/imported component cache differs from approved bound')
        return owned_learner_cuda_admission(model, config.model.inference,
            logical_bytes=logical, global_cuda_bytes=actual['budget']['managed_cuda_bytes'],
            pool_cuda_bytes=pool_bytes, overlap_scratch_bytes=0, layout=layout)

    def initialize_state(self, state: Json, config: Any) -> None:
        from gf_common.domain import digest
        from gf_training.rust_player_lineage import resume_identity
        super().initialize_state(state, config)
        state.update(run_variant='hpo-ppo-owned-rust-player-v1', model_version='4',
            hpo_spec_sha256=digest(config.hpo_spec), used_decisions=0, collected_decisions=0, stop_reason=None)
        classification = {name: state[name] for name in ('action_contract', 'purpose',
            'synthetic_capacity', 'rule_signature', 'layout_signature')}
        state['resume_identity'] = resume_identity(effective_config=state['effective_config'],
            identity=state['identity'], execution=state['execution'], classification=classification,
            storage_binding=state['storage_binding'], hpo=True)

    def controls(self, rng: Any, pool: list[int], position: int) -> Json:
        return {'schedule_opponent_rng': rng.bit_generator.state,
            'snapshot_pool': pool, 'collection_position': position}

    def phase(self, phase: str, **facts: Any) -> None:
        self.coordinator_phase = phase

    def groups(self, config: Any, generation: int, label: str, snapshots: tuple[str, ...], rng: Any) -> Any:
        from dataclasses import asdict

        from gf_common.training_opponents import training_opponents
        from gf_training.hpo_collection import training_groups_hpo, training_groups_hpo_refresh
        mixed = training_opponents(asdict(config))
        if mixed is not None:
            return training_groups_hpo_refresh(config.training.seeds[0], generation, label, snapshots, rng,
                collection_games=config.hpo_spec['training_recipe']['collection_games'],
                actions=config.actions, training_opponents=mixed)
        return training_groups_hpo(config.training.seeds[0], generation, label, snapshots, rng,
            collection_games=config.hpo_spec['training_recipe']['collection_games'])

    def linkage(self, state: Json, directory: Path, generation: int, groups: Any, bindings: Any) -> Json:
        from gf_common.hpo import selection_context
        link = super().linkage(state, directory, generation, groups, bindings)
        link['selection_context'] = selection_context(decision_budget=self.config.hpo_spec['decision_budget'],
            used_before=state['used_decisions'], learning_seed=state['seed'], batch=generation)
        return link

    def stop(self, state: Json, config: Any, usage: int, now: float, execution: float, admission: float) -> bool:
        if state['used_decisions'] == config.hpo_spec['decision_budget']:
            return True
        if usage > config.training.max_owned_bytes or now >= execution:
            raise RuntimeError('HPO actual execution/storage limit exceeded')
        if state['updates'] >= config.training.max_updates:
            self.stop_reason = 'update_cap'
        elif now >= admission:
            self.stop_reason = 'time_cap'
        elif usage + config.training.admission_reservation_bytes > config.training.max_owned_bytes:
            self.stop_reason = 'storage_cap'
        return self.stop_reason is not None

    def update_record(self, update: Json, state: Json, numeric_files: Json) -> None:
        from gf_common.domain import digest
        super().update_record(update, state, numeric_files)
        source = self.registry.pending
        proof = source.proof
        update.update(schema_version=8, model_version='4', origin='producer-owned-v1', encoder='canonical-python',
            hpo_spec_sha256=digest(self.config.hpo_spec), collected_decisions=proof['selection']['collected_decisions'],
            used_decisions=proof['selection']['used_decisions'],
            overshoot_decisions=proof['selection']['collected_decisions'] - proof['selection']['used_decisions'],
            expected_optimizer_step_delta=proof['expected_optimizer_step_delta'])

    def retain(self, directory: Path, generation: int, config: Any, deadline: float) -> None:
        self.registry.retain_hpo(self.collector.runtime,
            consumer_settlement_sha256=self.last_consumer_settlement_sha256, deadline=deadline)
