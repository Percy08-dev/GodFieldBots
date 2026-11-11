"""Collection5 metadata and final owned-inode publication; no Source admission."""

from __future__ import annotations

import math
import os
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gf_training.collection import PackedRunResult
    from gf_training.group_pack_publication import GroupPackCollectionPublication

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.card_instance_runtime import game_binding, validate_runtime_receipt
from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    clone,
    closed,
    digest,
    integer,
)
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_tokens import signature
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT, validate_neural_metadata
from gf_models.card_instance_game_tree_runtime import TREE_INPUT_CONTRACT, validate_tree_metadata


def _publication_schema(manifest: Json) -> bool:
    if type(manifest.get('schema_version')) is not int:
        return False
    if manifest['schema_version'] == 5:
        return True
    if manifest['schema_version'] != 6:
        return False
    if manifest.get('collection_role') == 'hpo-evaluation':
        from gf_common.card_instance_collection import checked_hpo_evaluation_manifest
        checked_hpo_evaluation_manifest(manifest)
        return True
    from gf_training.hpo_collection import check_hpo_collection
    check_hpo_collection(manifest)
    return True


def collection_metadata(
    *,
    actions: ActionConfigV3,
    config: Json,
    runtime_evidence: list[Json],
    schedule: list[Json],
    inventory: list[Json],
    policies: list[str],
    bots_revision: str,
    policy_bindings: dict[str, RuntimePolicyBinding] | None = None,
    neural_inputs: Json | None = None,
    tree_inputs: Json | None = None,
    _hpo: bool = False,
) -> Json:
    binding = game_binding(
        actions.action_contract, rule_signature(actions.action_contract, card_instance_rules())
    )
    if not runtime_evidence or not schedule or not inventory:
        raise ContractError("collection5 requires original complete runtime/schedule/file evidence")
    receipts = [
        validate_runtime_receipt(value, require_executed=True) for value in runtime_evidence
    ]
    neural_fields: Json = {}
    inputs = {"neural_inputs": neural_inputs, "tree_inputs": tree_inputs}
    supplied = {key: value for key, value in inputs.items() if value is not None}
    if (policy_bindings is None) != (not supplied):
        raise ContractError("learned collection requires both policy bindings and retained inputs")
    if supplied:
        assert policy_bindings is not None
        for key, raw_inputs in supplied.items():
            closed(raw_inputs, {"input_contract", "policies"})
            expected_contract = TREE_INPUT_CONTRACT if key == "tree_inputs" else EXTERNAL_INPUT_CONTRACT
            if (raw_inputs["input_contract"] != expected_contract or type(raw_inputs["policies"]) is not dict
                    or not raw_inputs["policies"]):
                raise ContractError("collection requires closed retained own-policy inputs")
        if set(policy_bindings) != set(policies):
            raise ContractError("collection policy registry differs from declared policy labels")
        expected_by_type: dict[str, set[str]] = {"neural_inputs": set(), "tree_inputs": set()}
        for label, execution in policy_bindings.items():
            if type(execution) is not RuntimePolicyBinding:
                raise ContractError("closed runtime policy binding required")
            if execution.policy_options is None:
                if execution.model_id not in ("random", "heuristic") or label != execution.model_id + "-card-instance-v2":
                    raise ContractError("collection baseline binding mismatch")
                continue
            tree = execution.model_id == "lightgbm"
            key = "tree_inputs" if tree else "neural_inputs"
            expected_by_type[key].add(label)
            raw = supplied.get(key, {}).get("policies", {}).get(label)
            if type(raw) is not dict:
                raise ContractError("missing retained own-policy metadata")
            validator = validate_tree_metadata if tree else validate_neural_metadata
            safe, expected = validator(raw, actions=actions, rules=card_instance_rules())
            if _hpo and (tree or safe['version'] != '4'):
                raise ContractError('Collection6 requires explicit neural4 registry')
            if expected != label or canonical(safe) != canonical(execution.policy_options) or safe["model"]["model_id"] != execution.model_id:
                raise ContractError("collection retained policy binding mismatch")
        for key, labels in expected_by_type.items():
            if set(supplied.get(key, {}).get("policies", {})) != labels:
                raise ContractError("collection retained input inventory differs from typed registry")
        neural_fields = {"policy_bindings": {label: asdict(value) for label, value in policy_bindings.items()},
            **{key: clone(value) for key, value in supplied.items()}}
    elif not policies or any(
        value not in ("random-card-instance-v2", "heuristic-card-instance-v2") for value in policies
    ):
        raise ContractError("collection5 supports corrected baselines only")
    return {
        "schema_version": 6 if _hpo else 5,
        **binding,
        "config": config,
        "config_hash": digest(config),
        "action_config_hash": digest(asdict(actions)),
        "bots_revision": bots_revision,
        "simulator_revision": actions.action_contract.simulator_revision,
        "runtime_evidence": receipts,
        "schedule": schedule,
        "files": inventory,
        "policies": policies,
        "journal_schema_version": 6,
        "derived_schema_version": 6,
        "participant_protocol_version": 3,
        "candidate_codec": "candidate-columns-v2",
        "model_version": "4" if _hpo else "3" if neural_fields else "1",
        "input_recipe": CORRECTED_CARD_INSTANCE_LAYOUT.input_recipe,
        "token_signature": signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        "source_eligibility": "not-authorized-by-collection-metadata",
        **neural_fields,
    }


def hpo_evaluation_collection_metadata(*, actions: ActionConfigV3, config: Json,
                            runtime_evidence: list[Json], schedule: list[Json],
                            inventory: list[Json], policies: list[str], bots_revision: str,
                            policy_bindings: dict[str, RuntimePolicyBinding],
                            neural_inputs: Json, linkage: Json) -> Json:
    """Closed Eval6 metadata; neither final64 nor a training Source capability."""
    from gf_common.card_instance_collection import checked_hpo_evaluation_metadata
    value = collection_metadata(actions=actions, config=config, runtime_evidence=runtime_evidence,
        schedule=schedule, inventory=inventory, policies=policies, bots_revision=bots_revision,
        policy_bindings=policy_bindings, neural_inputs=neural_inputs, _hpo=True)
    value.update(collection_role='hpo-evaluation', linkage=clone(linkage))
    return checked_hpo_evaluation_metadata(value)


def ppo_collection_metadata(*, actions: ActionConfigV3, config: Json,
                            runtime_evidence: list[Json], schedule: list[Json],
                            inventory: list[Json], policies: list[str], bots_revision: str,
                            policy_bindings: dict[str, RuntimePolicyBinding],
                            neural_inputs: Json, linkage: Json) -> Json:
    """Closed original PPO linkage; only its live owner can admit the Source."""
    raw = closed(linkage, {"parent_run_id", "parent_run_path", "batch_id", "collection_generation",
                           "learner_seats", "schedule_digest", "policy_registry_digest"})
    for key in ("parent_run_id", "parent_run_path", "batch_id"):
        if type(raw[key]) is not str or not raw[key]:
            raise ContractError("PPO collection requires explicit parent/batch linkage")
    integer(raw["collection_generation"], 0)
    if (not Path(raw["parent_run_path"]).is_absolute()
            or raw["batch_id"] != f"batch-{raw['collection_generation']:06}"):
        raise ContractError("PPO collection parent path/batch generation mismatch")
    value = collection_metadata(actions=actions, config=config, runtime_evidence=runtime_evidence,
        schedule=schedule, inventory=inventory, policies=policies, bots_revision=bots_revision,
        policy_bindings=policy_bindings, neural_inputs=neural_inputs)
    if "policy_bindings" not in value:
        raise ContractError("PPO collection requires the exact neural policy registry")
    if (raw["schedule_digest"] != digest(value["schedule"])
            or raw["policy_registry_digest"] != digest(value["policy_bindings"])):
        raise ContractError("PPO collection frozen schedule/registry digest mismatch")
    seats = raw["learner_seats"]
    if type(seats) is not dict:
        raise ContractError("PPO collection requires explicit learner seats")
    expected = {}
    for group in value["schedule"]:
        for episode in group["episodes"]:
            viewer = integer(episode.get("learner_viewer"), 1, 2)
            if episode["episode_id"] in expected:
                raise ContractError("duplicate PPO episode identity")
            expected[episode["episode_id"]] = viewer
    if not expected or canonical(seats) != canonical(expected):
        raise ContractError("PPO collection learner seats differ from frozen schedule")
    return {**value, "collection_role": "ppo-batch", "linkage": clone(raw)}


def final_validation_collection_metadata(*, actions: ActionConfigV3, config: Json,
                                        runtime_evidence: list[Json], schedule: list[Json],
                                        inventory: list[Json], policies: list[str], bots_revision: str,
                                        policy_bindings: dict[str, RuntimePolicyBinding], neural_inputs: Json,
                                        linkage: Json, config_projection: Json,
                                        validation_invocation: Json, elapsed_seconds: float) -> Json:
    """Closed final64 metadata; neither linkage nor receipt supplies authority."""
    from gf_common.card_instance_collection import (
        FINAL_VALIDATION,
        PROJECTION_KEYS,
        PROJECTION_RECIPE,
        checked_invocation,
        checked_linkage,
    )
    value = collection_metadata(actions=actions, config=config, runtime_evidence=runtime_evidence,
        schedule=schedule, inventory=inventory, policies=policies, bots_revision=bots_revision,
        policy_bindings=policy_bindings, neural_inputs=neural_inputs)
    if config.get('run', {}).get('split') != 'validation':
        raise ContractError('final validation requires validation split')
    link = checked_linkage(linkage, role=FINAL_VALIDATION, schedule=schedule, registry=value['policy_bindings'])
    projection = closed(config_projection, PROJECTION_KEYS)
    if (projection['recipe'] != PROJECTION_RECIPE or projection['projected_config_sha256'] != value['config_hash']
            or any(type(projection[key]) is not str or len(projection[key]) != 64
                or any(char not in '0123456789abcdef' for char in projection[key])
                for key in PROJECTION_KEYS - {'recipe'})):
        raise ContractError('closed final validation effective Config projection required')
    invocation = checked_invocation(validation_invocation, run=config['run'])
    if (type(elapsed_seconds) not in (float, int) or not math.isfinite(elapsed_seconds)
            or not 0 <= elapsed_seconds <= invocation['wall_deadline_monotonic'] - invocation['started_monotonic']):
        raise ContractError('final validation cost exceeds original capped invocation')
    label = link['policy_label']
    if (set(value['policy_bindings']) != {label, 'random-card-instance-v2', 'heuristic-card-instance-v2'}
            or set(value['neural_inputs']['policies']) != {label}
            or value['policy_bindings'][label]['policy_options']['expected_manifest_digest'] != link['final_inference_sha256']):
        raise ContractError('final validation exact final neural binding required')
    return {**value, 'collection_role': FINAL_VALIDATION.role, 'linkage': clone(link),
        'config_projection': clone(projection), 'validation_invocation': clone(invocation), 'elapsed_seconds': elapsed_seconds}


@dataclass
class _OwnedHandle:
    descriptor: int | None
    state: str = "live"
    uncertain_number: int | None = None

    def release(self) -> None:
        if self.state != "live" or self.descriptor is None:
            raise ContractError("only a live owned collection descriptor may be closed")
        descriptor = self.descriptor
        # close may release the number and THEN report an error. Never inspect,
        # write, or retry that number after a failed attempt, even on EBADF.
        self.state, self.descriptor, self.uncertain_number = "uncertain", None, descriptor
        os.close(descriptor)
        self.state, self.uncertain_number = "released", None


@dataclass
class _OwnedOutput:
    name: str
    handle: _OwnedHandle
    raw: bytes
    inode: tuple[int, int] | None = None
    published: bool = False
    invalidated: bool = False

    @property
    def recorded_name(self) -> str:
        return self.name if self.published else self.name + ".tmp"


def _inode(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


class CollectionPublication:
    """One closed two-file commit with exact-inode recovery after close errors."""

    def __init__(self, directory: Path, *, work_deadline: float, cleanup_deadline: float,
                 group_pack: GroupPackCollectionPublication | None = None) -> None:
        self.group_pack = group_pack
        self.directory = directory
        self.work_deadline, self.cleanup_deadline = work_deadline, cleanup_deadline
        self.directory_identity: tuple[int, int] | None = None
        self._directory_handle: _OwnedHandle | None = None
        self._handles: list[_OwnedHandle] = []
        # Ordinary held-FD owners and retained inode records have different
        # lifetimes. Released/uncertain handles never remain in `files`.
        self.files: list[_OwnedOutput] = []
        self.outputs: list[_OwnedOutput] = []
        self.committed = False
        self.quarantined = False
        self._final_staged = False
        self._final_memory_retired = False

    @property
    def directory_fd(self) -> int | None:
        handle = self._directory_handle
        return handle.descriptor if handle is not None and handle.state == "live" else None

    def _check(self) -> None:
        if time.monotonic() >= self.work_deadline:
            raise TimeoutError("collection final publication deadline")
        assert self.directory_fd is not None
        current, owned = self.directory.lstat(), os.fstat(self.directory_fd)
        if (
            not stat.S_ISDIR(current.st_mode)
            or not stat.S_ISDIR(owned.st_mode)
            or _inode(current) != self.directory_identity
            or _inode(owned) != self.directory_identity
        ):
            raise ContractError("collection owned publication namespace changed")

    def _release_output(self, output: _OwnedOutput) -> None:
        self.files.remove(output)
        output.handle.release()

    def _release(self) -> None:
        for output in tuple(self.files):
            self._release_output(output)
        if self._directory_handle is not None:
            self._directory_handle.release()

    def _cleanup_check(self) -> None:
        if time.monotonic() >= self.cleanup_deadline:
            raise ContractError("collection invalidation unconfirmed: cleanup deadline exhausted")

    def _own_handle(self, descriptor: int) -> _OwnedHandle:
        handle = _OwnedHandle(descriptor)
        self._handles.append(handle)
        return handle

    def _recovery_directory(self) -> int:
        self._cleanup_check()
        if self.directory_fd is not None:
            descriptor = self.directory_fd
        else:
            # Recovery uses only the recorded pathname and saved directory inode.
            # It does not search renamed directories or retry an uncertain number.
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._own_handle(descriptor)
        current = os.fstat(descriptor)
        if not stat.S_ISDIR(current.st_mode) or _inode(current) != self.directory_identity:
            raise ContractError("collection recovery directory differs from owned inode")
        return descriptor

    def _invalidate_output(self, output: _OwnedOutput) -> None:
        self._cleanup_check()
        if output.invalidated:
            return
        if output.inode is None:
            raise ContractError("collection output inode identity was not captured")
        if output.handle.state == "live":
            assert output.handle.descriptor is not None
            descriptor = output.handle.descriptor
        else:
            directory_fd = self._recovery_directory()
            descriptor = os.open(
                output.recorded_name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
            self._own_handle(descriptor)  # own fresh resources before validation can fail.
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or _inode(current) != output.inode:
            raise ContractError("collection recovery output differs from owned inode")
        self._cleanup_check()
        if os.pwrite(descriptor, b"!", 0) != 1 or os.pread(descriptor, 1, 0) != b"!":
            raise ContractError("collection invalidation unconfirmed: owned inode write failed")
        self._cleanup_check()
        os.fsync(descriptor)
        output.invalidated = True

    def revoke(self) -> None:
        if self.group_pack is not None:
            self.committed = False
            self.group_pack.revoke()
            return
        errors: list[BaseException] = []
        # Failure for one record never prevents an independent attempt for the
        # other record, within the same original cleanup allowance.
        invalidations = (sorted(self.outputs, key=lambda output: output.name != 'manifest.json')
                         if self._final_staged else self.outputs)
        for output in invalidations:
            try:
                self._invalidate_output(output)
            except BaseException as error:  # noqa: BLE001 - retain all independent cleanup failures.
                errors.append(error)
        for output in tuple(self.files):
            try:
                self._cleanup_check()
                self._release_output(output)
            except BaseException as error:  # noqa: BLE001 - attempt other unquestionably owned handles.
                errors.append(error)
        for handle in self._handles:
            if handle.state != "live":
                continue
            try:
                self._cleanup_check()
                handle.release()
            except BaseException as error:  # noqa: BLE001 - uncertain handles are never retried.
                errors.append(error)
        if any(handle.state == "uncertain" for handle in self._handles):
            errors.append(
                ContractError(
                    "collection descriptor release remains uncertain; stale numbers quarantined"
                )
            )
        if errors:
            self.quarantined = True
            unconfirmed = ContractError("collection invalidation/resource release unconfirmed")
            for failure in errors:
                unconfirmed.add_note(type(failure).__name__ + ": " + str(failure))
            raise unconfirmed

    def _write(self, name: str, value: Json, *, _stage_manifest: bool = False,
               _raw: bytes | None = None) -> None:
        self._check()
        assert self.directory_fd is not None
        raw = (canonical(value) + "\n").encode() if _raw is None else _raw
        descriptor = os.open(
            name + ".tmp",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.directory_fd,
        )
        output = _OwnedOutput(name, self._own_handle(descriptor), raw)
        self.files.append(output)
        self.outputs.append(output)
        output.inode = _inode(os.fstat(descriptor))
        for start in range(0, len(raw), 1024**2):
            self._check()
            chunk = raw[start : start + 1024**2]
            if os.pwrite(descriptor, chunk, start) != len(chunk):
                raise OSError("collection publication incomplete write")
        os.fsync(descriptor)
        self._verify(name + ".tmp", descriptor, raw)
        if _stage_manifest:
            return
        os.link(
            name + ".tmp",
            name,
            src_dir_fd=self.directory_fd,
            dst_dir_fd=self.directory_fd,
            follow_symlinks=False,
        )
        output.published = True
        self._verify(name, descriptor, raw)
        self._verify(name + ".tmp", descriptor, raw)
        os.unlink(name + ".tmp", dir_fd=self.directory_fd)

    def _verify(self, name: str, descriptor: int, raw: bytes) -> None:
        self._check()
        assert self.directory_fd is not None
        before = os.fstat(descriptor)
        path = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
        actual = bytearray()
        for offset in range(0, len(raw) + 1, 1024**2):
            self._check()
            actual.extend(os.pread(descriptor, min(1024**2, len(raw) + 1 - offset), offset))
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

        if (
            not stat.S_ISREG(path.st_mode)
            or _inode(path) != _inode(before)
            or identity(before) != identity(after)
            or identity(current) != identity(after)
            or bytes(actual) != raw
        ):
            raise ContractError("collection held output/path/content changed")

    def _publish(self, report: Json, manifest: Json, *, final: bool) -> None:
        if self.committed or self.outputs or self._directory_handle is not None:
            raise ContractError("collection publication owner already consumed")
        if report.get("healthy") is not True or not _publication_schema(manifest):
            raise ContractError("collection healthy report and manifest5 required")
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._directory_handle = self._own_handle(descriptor)
        self.directory_identity = _inode(os.fstat(descriptor))
        self._write("report.json", report)
        if final:
            from gf_common.card_instance_collection import INVOCATION_KEYS
            clock = closed(manifest['validation_invocation'], INVOCATION_KEYS)
            elapsed = time.monotonic() - clock['started_monotonic']
            if (not math.isfinite(elapsed) or not report['wall_seconds'] <= elapsed
                    <= clock['wall_deadline_monotonic'] - clock['started_monotonic']):
                raise ContractError('final validation post-report cost exceeds original capped invocation')
            manifest = {**manifest, 'elapsed_seconds': elapsed}
        if final:
            self._write("manifest.json", manifest, _stage_manifest=True)
        else:
            self._write("manifest.json", manifest)
        for output in self.outputs:
            assert output.handle.descriptor is not None
            self._verify(output.recorded_name, output.handle.descriptor, output.raw)
        assert self.directory_fd is not None
        os.fsync(self.directory_fd)
        self._check()  # final check while both original output descriptors are live.

    def commit(self, report: Json, manifest: Json, *, retained_input_guard: Callable[[], None] | None = None) -> PackedRunResult | None:
        if self.group_pack is not None:
            from gf_training.group_pack_publication import GroupPackCollectionPublication
            if type(self.group_pack) is not GroupPackCollectionPublication:
                raise ContractError("packed publication requires its concrete owned composite")
            if self.committed:
                raise ContractError("collection publication owner already consumed")
            try:
                result = self.group_pack.commit(report, manifest, retained_input_guard=retained_input_guard)
            except BaseException as error:
                vars(error)["_card_instance_collection_cleanup_owner"] = self
                raise
            self.committed = True
            return result
        if manifest.get('collection_role') == 'ppo-final-validation' or self._final_staged:
            raise ContractError('final validation requires its concrete retained publication owner')
        if self.committed or self.outputs or self._directory_handle is not None:
            raise ContractError('collection publication owner already consumed')
        if report.get('healthy') is not True or not _publication_schema(manifest):
            raise ContractError('collection healthy report and manifest5 required')
        try:
            if retained_input_guard is not None:
                retained_input_guard()
            self._publish(report, manifest, final=False)
            if retained_input_guard is not None:
                retained_input_guard()
            self._release()  # close errors are publication failures, even after actual release.
            if retained_input_guard is not None:
                retained_input_guard()
        except BaseException as error:
            try:
                self.revoke()
            except BaseException as cleanup:  # noqa: BLE001 - retain reachable invalidation owner.
                self.quarantined = True
                vars(error)["_card_instance_collection_cleanup_owner"] = self
                error.add_note("collection publication quarantine: " + str(cleanup))
                for note in getattr(cleanup, "__notes__", ()):
                    error.add_note(note)
            raise
        # No fallible operation follows successful release.
        self.committed = True
        return None

    @property
    def cleanup_complete(self) -> bool:
        if self.group_pack is not None:
            return self.group_pack.cleanup_complete
        released = all(handle.state == 'released' for handle in self._handles)
        return released and (self.committed or all(output.invalidated for output in self.outputs))

    @property
    def final_pending(self) -> bool:
        if self.group_pack is not None:
            return self.group_pack.final_pending
        return self._final_staged and not self.committed and not self.quarantined and not any(
            output.invalidated for output in self.outputs)

    @property
    def resident_bytes(self) -> int:
        if self.group_pack is not None:
            return self.group_pack.resident_bytes
        return sum(32 * len(output.raw) + 65536 for output in self.outputs) + 65536

    def fail_final_validation(self, initiating: BaseException) -> None:
        if self.cleanup_complete and not self.committed:
            return
        try:
            self.revoke()  # Owned pending/published manifest is invalidated FIRST.
        except BaseException as cleanup:  # noqa: BLE001 - no stale descriptor retries or primary replacement
            self.quarantined = True
            vars(initiating)['_card_instance_collection_cleanup_owner'] = self
            try:
                initiating.add_note('final validation publication retirement: ' + str(cleanup))
            except BaseException:  # noqa: BLE001, S110 - diagnostic failure cannot replace primary
                pass

    def begin_final_validation(self, report: Json, manifest: Json) -> None:
        if manifest.get('collection_role') != 'ppo-final-validation' or self._final_staged:
            raise ContractError('one closed final validation role/lifetime required for retained publication')
        if self.committed or self.outputs or self._directory_handle is not None:
            raise ContractError('collection publication owner already consumed')
        self._final_staged = True
        try:
            if self.group_pack is not None:
                self.group_pack.begin_final_validation(report, manifest)
            else:
                self._publish(report, manifest, final=True)
        except BaseException as error:
            self.fail_final_validation(error)
            raise

    def pending_manifest(self) -> str:
        if self.group_pack is not None:
            return self.group_pack.pending_manifest()
        if not self.final_pending:
            raise ContractError('SAME pending final validation publication required')
        self._check()
        try:
            (self.directory / 'manifest.json').lstat()
        except FileNotFoundError:
            pass
        else:
            raise ContractError('canonical final validation manifest must remain absent before closure')
        for output in self.outputs:
            if output.handle.state != 'live' or output.handle.descriptor is None:
                raise ContractError('pending control descriptor no longer owned')
            self._verify(output.recorded_name, output.handle.descriptor, output.raw)
        return 'manifest.json.tmp'

    def finish_final_validation(self, *, context: object = None) -> Json | None:
        if self.group_pack is not None:
            result = self.group_pack.finish_final_validation(self, context)
            self.committed = True
            return result
        if not self.final_pending:
            raise ContractError('one active pending final publication required for finish')
        try:
            self.pending_manifest()
            output = next(value for value in self.outputs if value.name == 'manifest.json')
            assert self.directory_fd is not None and output.handle.descriptor is not None
            os.link('manifest.json.tmp', 'manifest.json', src_dir_fd=self.directory_fd,
                dst_dir_fd=self.directory_fd, follow_symlinks=False)
            output.published = True
            self._verify('manifest.json', output.handle.descriptor, output.raw)
            self._verify('manifest.json.tmp', output.handle.descriptor, output.raw)
            os.unlink('manifest.json.tmp', dir_fd=self.directory_fd)
            self._verify('manifest.json', output.handle.descriptor, output.raw)
            os.fsync(self.directory_fd)
            self._check()
            self._release()
            if time.monotonic() >= self.work_deadline:
                raise TimeoutError('final validation terminal release exceeded original deadline')
        except BaseException as error:
            self.fail_final_validation(error)
            raise
        self.committed = True  # All reportable terminal work has finished.
        return None

    def retire_final_memory(self) -> None:
        if not self.cleanup_complete:
            raise ContractError('confirmed final publication cleanup required before memory retirement')
        for output in self.outputs:
            output.raw = b''
        self._final_memory_retired = True
