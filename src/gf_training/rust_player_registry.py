"""Actual saved inference ownership and new-mode generation/cache transactions."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, clone, digest
from gf_common.ram_bank import MAX_BANK_BYTES, ReadonlyRamBankSet
from gf_common.rust_ppo_source import source_encoder
from gf_models.rust_player import serialized_torch
from gf_training.group_pack_source import OriginalFile
from gf_training.hpo_owned_lineage import OwnedControlWriter
from gf_training.hpo_update_storage import compressed_updates, update_name
from gf_training.lineage import require_hpo_run_owner
from gf_training.rust_player_lineage import validate_record_shape
from gf_training.rust_player_supervision import HostAdmission

_ISSUER = object()


def _backings(banks: ReadonlyRamBankSet) -> tuple[tuple[int, int, int], ...]:
    return tuple((value.st_dev, value.st_ino, value.st_size)
        for value in (os.fstat(bank.fileno()) for bank in banks.banks))


def policy_binding_data(*, model: Any, binding: Any, directory: Path, manifest_digest: str,
                        actions: Any, rules: Any) -> tuple[str, Any]:
    """Pure same-policy descriptor; neither saved nor restored authority."""
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.structured_binding import (
        child_cuda_visibility,
        input_binding_game_v3,
        policy_label_game_v3,
        policy_label_game_v4,
        safe_options_game_v3,
        safe_options_game_v4,
    )
    parser = safe_options_game_v4 if model.version == '4' else safe_options_game_v3
    labeler = policy_label_game_v4 if model.version == '4' else policy_label_game_v3
    visibility = child_cuda_visibility(binding.device, os.environ.get('CUDA_VISIBLE_DEVICES'))
    options = parser({'version': model.version, 'checkpoint': str(directory),
        'expected_manifest_digest': manifest_digest, 'model': asdict(model),
        'inference': asdict(binding), 'cuda_visible_devices': visibility,
        'action_config_hash': digest(asdict(actions)),
        **input_binding_game_v3(actions.action_contract, rules)}, actions=actions, rules=rules)
    label = labeler(model, binding, manifest_digest, action_contract=actions.action_contract,
        rules=rules, action_config_hash=digest(asdict(actions)), cuda_visible_devices=visibility)
    return label, RuntimePolicyBinding(model.model_id, options)


class _PolicyLease:
    def __init__(self, policy: Any, owner: object) -> None:
        self.policy, self.owner, self.closed = policy, owner, False

    def close(self, owner: object) -> None:
        if owner is not self.owner:
            raise ContractError("SAME actual inference consumer required")
        if not self.closed:
            self.policy._consumers.remove(self)
            self.closed = True


class RustSavedPolicy:
    """Genuine save -> actual generation write -> inference-only role.

    This data owner is not admission/capture/Source/final-parent authority.
    The original complete save ACK is used only while both banks are valid.
    After resume-bank retirement, actual inference backing/file ownership takes
    over; no claim is made that the old full-snapshot ACK remains applicable.
    """

    _pair: Any
    _budget: Any
    _learner: Any
    generation: int
    run_id: str
    runtime_binding_sha256: str
    actions: Any
    rules: Any
    binding: Any
    model: Any
    optimizer_steps: int
    files: Json
    identity: str
    _inference_backings: Any

    def __init__(self, issuer: object) -> None:
        if issuer is not _ISSUER:
            raise ContractError("actual saved-policy preparation required")
        self._state = "PREPARING"
        self._held: dict[str, OriginalFile] = {}
        self._cleanup_children: list[Any] = []
        self._charges: list[Any] = []
        self._consumers: set[_PolicyLease] = set()
        self._snapshot: Any = None
        self.inference: ReadonlyRamBankSet | None = None
        self._resume_retired = False

    @classmethod
    def prepare(cls, pair: Any, *, budget: Any, generation: int, run_id: str,
                runtime_binding_sha256: str, actions: Any, rules: Any,
                inference: Any) -> RustSavedPolicy:
        from gf_training.rust_player_learning import LocalSavedPair

        if type(pair) is not LocalSavedPair:
            raise ContractError("actual local completed saved pair required")
        pair.require(pair._owner)
        output = cls(_ISSUER)
        output._pair, output._budget = pair, budget
        output.generation, output.run_id = generation, run_id
        output.runtime_binding_sha256 = runtime_binding_sha256
        output.actions, output.rules, output.binding = actions, rules, inference
        output._learner = pair._owner.learner
        output.model, output.optimizer_steps = output._learner.model, output._learner.steps
        output.files = pair.files
        try:
            with serialized_torch():
                resume = json.loads((pair.directory / "resume.json").read_bytes())
                if resume["optimizer_steps"] != output.optimizer_steps:
                    raise ContractError("saved pair differs from current actual numerical state")
                if pair.snapshot is not None:
                    snapshot = pair.snapshot
                    if snapshot._budget is not budget or snapshot.generation != generation:
                        raise ContractError("saved snapshot requires SAME Source account/generation")
                    snapshot.save_ack.require(snapshot)
                    output._snapshot = snapshot
                    output.inference = snapshot.inference
                    output._manifest_bytes = snapshot.save_ack._inference_manifest
                    output.identity = snapshot.identity
                else:
                    path = pair.directory / "inference.json"
                    # The existing JSON envelope owns this newly decoded metadata;
                    # HPO above aliases its already-charged actual saved manifest.
                    output._charges.append(budget.reserve(32 * path.stat().st_size + 65536))
                    output._manifest_bytes = path.read_bytes()
                    tensors = output._learner.network.state_dict()
                    sizes = [value.numel() * value.element_size() for value in tensors.values()]
                    # Exact simultaneous CPU array staging + one current D2H
                    # parameter, rather than a second full snapshot/IPC allowance.
                    with budget.reserve(sum(sizes) + max(sizes, default=0)):
                        arrays = {name: value.detach().cpu().numpy().copy()
                            for name, value in tensors.items()}
                        output.inference = ReadonlyRamBankSet.build(arrays, budget,
                            max_bytes=min(MAX_BANK_BYTES, budget.limit - budget.used))
                        del arrays
                    output.identity = digest({"kind": "rust-player-saved-policy-v1",
                        "run_id": run_id, "generation": generation, "files": output.files,
                        "inference_backings": _backings(output.inference)})
                output._inference_backings = _backings(output.inference)
                manifest = output.manifest
                for name, expected in {"inference.json": pair.inference_digest,
                        **manifest["weights"]["files"]}.items():
                    original = OriginalFile(pair.directory / name, budget.deadline)
                    output._held[name] = original
                    if original.sha256 != expected:
                        raise ContractError("saved inference original differs from actual writer")
                shallow = (sys.getsizeof(output) + sys.getsizeof(vars(output))
                    + sys.getsizeof(output._held)
                    + sum(sys.getsizeof(value) + sys.getsizeof(vars(value)) for value in output._held.values()))
                output._charges.append(budget.reserve(shallow))
                output._state = "EXPORTED"
                return output
        except BaseException as error:
            from gf_training.group_pack_resources import retain_cleanup
            retain_cleanup(output, error)
            raise

    @property
    def manifest(self) -> Json:
        return json.loads(self._manifest_bytes)

    def commit(self, writer: OwnedControlWriter, path: Path, checksum: str) -> None:
        if self._state != "EXPORTED" or writer.budget is not self._budget:
            raise ContractError("actual exported policy and SAME generation writer required")
        self._pair.require(self._pair._owner)
        receipt = writer._completed.get(path)
        if receipt is None:
            raise ContractError("actual completed generation write required")
        receipt.require(writer, path, checksum)
        row = json.loads(path.read_bytes())
        validate_record_shape(row, "generation", hpo=self._snapshot is not None)
        if (row["run_id"] != self.run_id or row["generation"] != self.generation
                or row["snapshot_identity"] != self.identity or row["artifact_files"] != self.files
                or row["inference_sha256"] != self.files["inference.json"]
                or row["resume_sha256"] != self.files["resume.json"]
                or row["optimizer_steps"] != self.optimizer_steps
                or row["runtime_binding_sha256"] != self.runtime_binding_sha256):
            raise ContractError("committed generation differs from SAME saved policy")
        self.generation_sha256 = checksum
        self._commit_receipt, self._commit_writer, self._commit_path = receipt, writer, path
        self._state = "COMMITTED"

    def require(self) -> None:
        if self._state != "COMMITTED":
            raise ContractError("original committed inference policy required")
        self._commit_receipt.require(self._commit_writer, self._commit_path, self.generation_sha256)
        assert self.inference is not None
        if _backings(self.inference) != self._inference_backings:
            raise ContractError("actual inference backing differs from original saved policy")
        for original in self._held.values():
            original.recheck()

    def retire_resume(self) -> None:
        self.require()
        if not self._resume_retired:
            if self._snapshot is not None:
                self._snapshot.resume.close()  # positive close only; aliases keep the SAME charge
            self._resume_retired = True

    def retain(self, owner: object) -> _PolicyLease:
        self.require()
        lease = _PolicyLease(self, owner)
        self._consumers.add(lease)
        return lease

    def policy_binding(self) -> tuple[str, Any]:
        self.require()
        return policy_binding_data(model=self.model, binding=self.binding, directory=self._pair.directory,
            manifest_digest=self.files['inference.json'], actions=self.actions, rules=self.rules)

    def journal_metadata(self) -> Json:
        from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT
        _, binding = self.policy_binding()
        return {"input_contract": EXTERNAL_INPUT_CONTRACT, "policy_options": binding.policy_options,
            "inference_manifest": self.manifest, "files": [
                {"path": name, "sha256": original.sha256, "identity": [getattr(original.identity, key)
                    for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')]}
                for name, original in sorted(self._held.items())]}

    def close(self) -> None:
        if self._state == "CLOSED":
            return
        if self._consumers:
            raise ContractError("actual inference consumers remain live")
        if self._snapshot is not None:
            self._snapshot.close()
        elif self.inference is not None:
            self.inference.close()
        from gf_training.group_pack_resources import close_resources
        close_resources(self, list(self._held.values()))
        self._held.clear()
        self._manifest_bytes = b""
        for charge in self._charges:
            charge.close()
        self._charges.clear()
        self._state = "CLOSED"


class RustRunRegistry:
    """Actual supervised local run/save/cache owners, without legacy process attachment."""

    def __init__(self, ownership: Any, state: Json, *, config: Any, admission: HostAdmission,
                 pool: Any, writer: OwnedControlWriter, local: Any, service: Any) -> None:
        from gf_common.rust_player import validate_runtime_binding
        from gf_models.rust_player import RustPlayerInference

        self.ownership = require_hpo_run_owner(ownership.directory, ownership)
        if type(admission) is not HostAdmission or type(service) is not RustPlayerInference:
            raise ContractError("actual same-host admission and inference service required")
        admission.require()
        actual = admission.observation
        bound = validate_runtime_binding(state["identity"]["player_runtime"])
        if (local.admission is not admission or writer.owner is not ownership or writer.budget is not pool.budget
                or actual["run_id"] != state["run_id"] or actual["config_sha256"] != state["config_sha256"]
                or actual["runtime_binding_sha256"] != digest(bound) or actual["producer"] != bound["producer"]
                or service.runtime != bound["sampling_runtime_sha256"]
                or Path(admission._control.operation["parent_directory"]) != ownership.directory
                or admission._control.operation["schema_version"] != 2):
            raise ContractError("registry differs from actual parent/config/runtime/owner/admission")
        self.directory, self.state, self.config = ownership.directory, state, config
        self.admission, self.pool, self.budget = admission, pool, pool.budget
        self.writer, self.local, self.service = writer, local, service
        self.hpo = getattr(config, "hpo_spec", {}).get("schema_version") == 6
        self.service_id = os.urandom(16).hex()
        self.policies: dict[int, Any] = {}
        self.records: dict[int, Json] = {}
        self.cache: dict[int, Any] = {}
        self.released_references: Json = {}
        self.pending: Any = None
        self._retirement_owners: list[Any] = []
        self._cleanup_children: list[Any] = []
        self._restored_metadata_loans: list[Any] = []
        self.failed, self.closed = False, False
        self.current: Any = None
        self.registry = {"schema_version": 8 if self.hpo else 5, "run_id": state["run_id"],
            "runtime_binding_sha256": state["runtime_binding_sha256"], "generations": {},
            "evaluation_references": {}, "service_references": {}}
        if self.hpo:
            self.registry.update(origin="producer-owned-v1", encoder=source_encoder(state['effective_config']))

    def require(self) -> None:
        require_hpo_run_owner(self.directory, self.ownership)
        self.admission.require()
        if self.failed or self.closed:
            raise ContractError("original healthy Rust registry required")

    def adopt_restored(self, audit: Any, local: Any) -> None:
        from gf_training.rust_player_restore import RustResumeAudit
        self.require()
        if (type(audit) is not RustResumeAudit or audit.phase != 'LOADED'
                or audit.pool is not self.pool or audit.writer is not self.writer
                or audit.ownership is not self.ownership or local is not self.local
                or audit.state is not self.state or self.current is not None or self.policies):
            raise ContractError('actual SAME audited restored owner/parent/Source required')
        audit.require()
        self.policies = dict(audit.policies)
        self._adopt_restored_metadata(audit)
        self.current = self.policies[self.state['updates']]

    def _adopt_restored_metadata(self, audit: Any) -> None:
        """Transfer existing DATA loans after adopt_restored's live-owner checks."""
        self.records = dict(audit.records)
        self.registry = audit.registry
        self._restored_metadata_loans, audit._loans = audit._loans, []

    def _retire_restored_metadata(self) -> None:
        """After prepare's final read, drop shared graphs before the SAME refunds."""
        self._drop_control_aliases()
        for loan in self._restored_metadata_loans:
            loan.close()
        self._restored_metadata_loans.clear()

    def _drop_control_aliases(self) -> None:
        """Terminal last-use cleanup, including aliases created before adoption."""
        self.state.clear()
        self.records.clear()
        self.registry.clear()

    def commit(self, pair: Any, *, controls: Json, update: Json | None, deadline: float) -> Json:
        self.require()
        generation = self.state["updates"] + int(update is not None)
        if pair._owner is not self.local:
            raise ContractError("save differs from actual local numerical owner")
        if update is not None:
            from gf_training.rust_player_source import RustPlayerSource
            if type(self.pending) is not RustPlayerSource:
                raise ContractError("actual single-use persisted Source transaction required")
        elif self.current is not None or generation:
            raise ContractError("initial generation is single use")
        try:
            policy = RustSavedPolicy.prepare(pair, budget=self.budget, generation=generation,
                run_id=self.state["run_id"], runtime_binding_sha256=self.state["runtime_binding_sha256"],
                actions=self.config.actions, rules=self.local.learner._context.rules, inference=self.config.model.inference)
        except BaseException as error:
            self.failed = True
            self._provisional_policy = vars(error).get('cleanup_owner')
            raise
        # EXPORTED is not committed/publishable. Retain this SAME provisional
        # owner before validating the result binding or performing any write.
        self.policies[generation] = policy
        row = {"schema_version": 6 if self.hpo else 4, "run_id": self.state["run_id"],
            "generation": generation, "parent_sha256": self.state["generation_sha256"],
            "snapshot_identity": policy.identity, "inference_sha256": pair.inference_digest,
            "resume_sha256": pair.resume_digest, "artifact_files": pair.files, "update_sha256": None,
            "parent_controls": clone(controls), "optimizer_steps": self.state["optimizer_steps"],
            "accepted_decisions": self.state["accepted_decisions"],
            "runtime_binding_sha256": self.state["runtime_binding_sha256"]}
        if self.hpo:
            row.update(origin="producer-owned-v1", encoder=source_encoder(self.state['effective_config']),
                used_decisions=self.state["used_decisions"], collected_decisions=self.state["collected_decisions"])
        else:
            row.update({key: self.state[key] for key in ("model_id", "model_version", "input_recipe",
                "action_contract", "purpose", "synthetic_capacity", "rule_signature", "layout_signature")})
        try:
            if update is not None:
                update['result_snapshot_identity'] = policy.identity
                self.pending.require_saved_result(self, pair, update, policy)
                from gf_models.rust_player_device import bind_update_device_plan
                actual_plan = bind_update_device_plan(update,
                    operation_id=self.admission.observation['operation_id'],
                    learner_device=self.config.training.device, serving_device=self.config.model.inference.device,
                    global_bytes=self.admission.observation['budget']['managed_cuda_bytes'])
                if actual_plan is not None and (self.local.last_device_plan is None
                        or actual_plan != self.local.last_device_plan.record()):
                    raise ContractError('update plan differs from SAME local immutable operation')
                validate_record_shape(update, "update", hpo=self.hpo)
                compressed = self.hpo and compressed_updates(self.state)
                row["update_sha256"] = self.writer.write(
                    self.directory / "updates" / update_name(generation, compressed=compressed),
                    update, deadline=deadline, immutable=True,
                    update_compression=compressed if self.hpo else None)
            validate_record_shape(row, "generation", hpo=self.hpo)
            path = self.directory / "generations" / f"{generation:06}.json"
            checksum = self.writer.write(path, row, deadline=deadline, immutable=True)
            self.registry["generations"][str(generation)] = {"inference_sha256": pair.inference_digest,
                "snapshot_identity": policy.identity, "files": {name: {"sha256": value,
                    "status": "retained", "deletion_intent": None, "deletion_receipt": None}
                    for name, value in pair.files.items()}}
            self.writer.write(self.directory / "artifact-registry.json", self.registry, deadline=deadline)
            self.writer.write(self.directory / "latest.json", {"generation": generation, "sha256": checksum}, deadline=deadline)
            self.state.update(updates=generation, generation_sha256=checksum)
            self.writer.write(self.directory / "run.json", self.state, deadline=deadline)
            policy.commit(self.writer, path, checksum)
            policy.retire_resume()
            self.current, self.records[generation] = policy, clone(row)
            if self.pending is not None:
                self.pending.commit_completed(policy)
                self.pending = None
            return clone(row)
        except BaseException:
            self.failed = True
            raise

    def binding(self, generation: int) -> tuple[str, Any]:
        self.require()
        return self.policies[generation].policy_binding()

    def journal_inputs(self, bindings: Any) -> Json:
        from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT
        self.require()
        policies = {}
        for label, binding in bindings.items():
            if binding.model_id != "mlp":
                continue
            generation = int(Path(binding.policy_options["checkpoint"]).name)
            if (label, binding) != self.binding(generation):
                raise ContractError("inference differs from actual committed policy")
            policies[label] = self.policies[generation].journal_metadata()
        return {"input_contract": EXTERNAL_INPUT_CONTRACT, "policies": policies}

    def publish(self, generation: int, *, deadline: float) -> Json:
        self.require()
        if generation in self.cache:
            return clone(self.cache[generation]["publication"])
        policy = self.policies[generation]
        policy.require()
        label, binding = policy.policy_binding()
        actual, token = self.admission.observation, os.urandom(16).hex()
        data = {"run_id": self.state["run_id"], "token": token, "generation": generation,
            "inference_sha256": policy.files["inference.json"], "generation_sha256": policy.generation_sha256,
            "policy_sha256": digest(binding.policy_options), "weights_sha256": digest(policy.manifest["weights"]),
            "snapshot_identity": policy.identity, "runtime_binding_sha256": self.state["runtime_binding_sha256"],
            "invocation_id": actual["invocation_id"], "admission_sha256": digest(actual), "service_id": self.service_id,
            "cache_identity": digest({"invocation_id": actual["invocation_id"], "token": token,
                "actual_inference_backings": policy._inference_backings})}
        owner = {"binding": data, "policy": policy, "lease": None, "publication": None}
        self.cache[generation] = owner
        owner["lease"] = policy.retain(owner)
        directory = self.directory / "rust-cache-references" / token
        directory.mkdir(parents=True, exist_ok=False)
        acquired = self.writer.write(directory / "acquisition.json",
            {"schema_version": 1, "kind": "rust-player-cache-acquisition-v1", "binding": data},
            deadline=deadline, immutable=True)
        self.registry["service_references"][token] = {"binding": clone(data), "acquisition_sha256": acquired,
            "publication_sha256": None, "release_sha256": None, "status": "acquired"}
        self.writer.write(self.directory / "artifact-registry.json", self.registry, deadline=deadline)
        try:
            self.service.publish_banks(policy.model, policy.binding, policy.inference, policy.files["inference.json"])
            assert policy.inference is not None
            ack = {"schema_version": 1, "kind": "rust-player-cache-publication-ack-v1",
                "invocation_id": actual["invocation_id"], "service_id": self.service_id,
                "checkpoint": policy.files["inference.json"], "generation": generation,
                "ordinal": self.service.registry.publications, "cache_identity": data["cache_identity"],
                "source_binding_sha256": digest(data),
                "weights_bytes": sum(policy.inference.array(name).nbytes for name in policy.inference.names)}
            publication = {"schema_version": 1, "kind": "rust-player-cache-publication-v1",
                "acquisition_sha256": acquired, "ordinal": ack["ordinal"], "checkpoint": ack["checkpoint"],
                "generation": generation, "source_binding_sha256": digest(data), "ack": ack, "ack_sha256": digest(ack)}
            published = self.writer.write(directory / "publication.json", publication, deadline=deadline, immutable=True)
            self.registry["service_references"][token].update(publication_sha256=published, status="published")
            self.writer.write(self.directory / "artifact-registry.json", self.registry, deadline=deadline)
            owner.update(publication=publication, acquisition_sha256=acquired,
                publication_sha256=published, directory=directory, label=label)
            return clone(publication)
        except BaseException:
            self.failed = True
            raise

    def retire_cache(self, runtime: Any, *, service_retirement_sha256: str,
                     consumer_settlement_sha256: str, deadline: float) -> Json:
        """Actual cache removal/releases, after original independent joins."""
        require_hpo_run_owner(self.directory, self.ownership)
        self.admission.require()
        if (not runtime._joined or runtime._service_thread is not None and runtime._service_thread.is_alive()
                or runtime._active_routes or runtime._holds or runtime.dispatcher.checkpoint_references()
                or any(self.service.references.values())):
            raise ContractError('actual independent consumer retirement required before cache release')
        actual = self.admission.observation
        releases = dict(self.released_references)
        for owner in self.cache.values():
            data = owner['binding']
            checkpoint = data['inference_sha256']
            if checkpoint in self.service.registry.networks:
                self.service.evict(checkpoint, dispatcher=runtime.dispatcher)
            with serialized_torch():
                if self.service.device != 'cpu':
                    import torch
                    torch.cuda.synchronize(self.service.device)
            if checkpoint in self.service.registry.networks or checkpoint in self.service.references:
                raise ContractError('actual immutable cache entry remains present')
            evidence = {'schema_version': 1, 'kind': 'rust-player-cache-retirement-ack-v1',
                'invocation_id': actual['invocation_id'], 'service_id': self.service_id,
                'checkpoint': checkpoint, 'cache_identity': data['cache_identity'],
                'service_retirement_sha256': service_retirement_sha256,
                'consumer_settlement_sha256': consumer_settlement_sha256,
                'entry_absent': True, 'gpu_fenced': True}
            release = {'schema_version': 1, 'kind': 'rust-player-cache-release-v1',
                'binding': data, 'acquisition_sha256': owner['acquisition_sha256'],
                'publication_sha256': owner['publication_sha256'],
                'evidence': evidence, 'evidence_sha256': digest(evidence)}
            checksum = self.writer.write(owner['directory'] / 'release.json', release,
                deadline=deadline, immutable=True)
            self.registry['service_references'][data['token']].update(
                release_sha256=checksum, status='released')
            self.writer.write(self.directory / 'artifact-registry.json', self.registry, deadline=deadline)
            owner['lease'].close(owner)
            releases[data['token']] = checksum
        if any(owner.phase != 'complete' for owner in self._retirement_owners):
            raise ContractError('artifact retirement remains quarantined')
        for policy in self.policies.values():
            policy.close()
        aggregate = {'schema_version': 1, 'kind': 'rust-player-cache-retirement-v1',
            'run_id': self.state['run_id'], 'invocation_id': actual['invocation_id'],
            'service_id': self.service_id, 'service_retirement_sha256': service_retirement_sha256,
            'consumer_settlement_sha256': consumer_settlement_sha256, 'releases': releases}
        self.closed = True
        return aggregate

    def retain_ordinary(self, runtime: Any, *, consumer_settlement_sha256: str, deadline: float) -> None:
        """Retire only actual inference caches outside the fixed normal pool."""
        self.require()
        if self.hpo:
            raise ContractError('ordinary retention requires its exact normal recipe')
        self._retain_inference_cache(runtime, consumer_settlement_sha256=consumer_settlement_sha256, deadline=deadline)
        self._retain_artifact_files(deadline=deadline)

    def retain_hpo(self, runtime: Any, *, consumer_settlement_sha256: str, deadline: float) -> None:
        self.require()
        if not self.hpo:
            raise ContractError('HPO retention requires its exact spec6 recipe')
        self._retain_inference_cache(runtime, consumer_settlement_sha256=consumer_settlement_sha256, deadline=deadline)
        self._retain_artifact_files(deadline=deadline)

    def _retain_inference_cache(self, runtime: Any, *, consumer_settlement_sha256: str, deadline: float) -> None:
        generation = self.state['updates']
        keep = set(self.records[generation]['parent_controls']['snapshot_pool']) | {generation}
        for number in tuple(self.cache):
            if number in keep:
                continue
            owner = self.cache[number]
            policy, data = owner['policy'], owner['binding']
            if policy._consumers != {owner['lease']}:
                raise ContractError('actual Source/snapshot consumers remain before cache eviction')
            checkpoint = data['inference_sha256']
            self.service.evict(checkpoint, dispatcher=runtime.dispatcher)
            with serialized_torch():
                if self.service.device != 'cpu':
                    import torch
                    torch.cuda.synchronize(self.service.device)
            if checkpoint in self.service.registry.networks:
                raise ContractError('actual evicted inference cache entry remains present')
            evidence = {'schema_version': 1, 'kind': 'rust-player-cache-eviction-ack-v1',
                'invocation_id': self.admission.observation['invocation_id'], 'service_id': self.service_id,
                'checkpoint': checkpoint, 'cache_identity': data['cache_identity'],
                'active_policy_references': 0, 'native_pending': 0, 'started_producers': 0,
                'consumer_settlement_sha256': consumer_settlement_sha256, 'entry_absent': True}
            release = {'schema_version': 1, 'kind': 'rust-player-cache-release-v1', 'binding': data,
                'acquisition_sha256': owner['acquisition_sha256'], 'publication_sha256': owner['publication_sha256'],
                'evidence': evidence, 'evidence_sha256': digest(evidence)}
            checksum = self.writer.write(owner['directory'] / 'release.json', release,
                deadline=deadline, immutable=True)
            self.registry['service_references'][data['token']].update(release_sha256=checksum, status='released')
            self.writer.write(self.directory / 'artifact-registry.json', self.registry, deadline=deadline)
            self.released_references[data['token']] = checksum
            owner['lease'].close(owner)
            policy.close()
            del self.cache[number], self.policies[number]


    def _retain_artifact_files(self, *, deadline: float) -> None:
        from gf_training.rust_player_retention import RustArtifactRetirement, protected_files
        self.require()
        pending = [self.pending.parent.generation] if self.pending is not None else []
        protected = protected_files(self.records, self.registry, generation=self.state['updates'],
            operating=self.config.hpo_spec['operating'] if self.hpo else None, pending=pending)
        for generation, policy in tuple(self.policies.items()):
            policy.retire_resume()
            if not (set(policy.manifest['weights']['files']) | {'inference.json'}) <= protected[generation]:
                if policy._consumers:
                    protected[generation].update(self.records[generation]['artifact_files'])
                else:
                    policy.close()
                    del self.policies[generation]
        for generation, record in self.records.items():
            for name in record['artifact_files']:
                row = self.registry['generations'][str(generation)]['files'][name]
                if name in protected[generation] or row['status'] == 'deleted':
                    continue
                retirement = RustArtifactRetirement(self, generation, name, row, deadline)
                retirement.run()
