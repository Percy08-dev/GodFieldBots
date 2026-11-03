"""Collection5 metadata and final owned-inode publication; no Source admission."""

from __future__ import annotations

import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
)
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_tokens import signature
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT, validate_neural_metadata


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
    if (policy_bindings is None) != (neural_inputs is None):
        raise ContractError("neural collection requires both policy bindings and retained inputs")
    if neural_inputs is not None:
        assert policy_bindings is not None
        closed(neural_inputs, {"input_contract", "policies"})
        if neural_inputs["input_contract"] != EXTERNAL_INPUT_CONTRACT or not isinstance(neural_inputs["policies"], dict) or not neural_inputs["policies"]:
            raise ContractError("collection requires sealed retained external neural inputs")
        if set(policy_bindings) != set(policies):
            raise ContractError("collection policy registry differs from declared policy labels")
        expected_neural = set()
        for label, execution in policy_bindings.items():
            if type(execution) is not RuntimePolicyBinding:
                raise ContractError("closed runtime policy binding required")
            if execution.policy_options is None:
                if execution.model_id not in ("random", "heuristic") or label != execution.model_id + "-card-instance-v2":
                    raise ContractError("collection baseline binding mismatch")
            else:
                expected_neural.add(label)
                raw = neural_inputs["policies"].get(label)
                if type(raw) is not dict:
                    raise ContractError("missing sealed neural policy metadata")
                safe, expected = validate_neural_metadata(raw, actions=actions, rules=card_instance_rules())
                if expected != label or canonical(safe) != canonical(execution.policy_options) or safe["model"]["model_id"] != execution.model_id:
                    raise ContractError("collection sealed neural policy mismatch")
        if set(neural_inputs["policies"]) != expected_neural:
            raise ContractError("collection external input inventory differs from neural registry")
        neural_fields = {"policy_bindings": {label: asdict(value) for label, value in policy_bindings.items()},
            "neural_inputs": clone(neural_inputs)}
    elif not policies or any(
        value not in ("random-card-instance-v2", "heuristic-card-instance-v2") for value in policies
    ):
        raise ContractError("collection5 supports corrected baselines only")
    return {
        "schema_version": 5,
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
        "model_version": "3" if neural_fields else "1",
        "input_recipe": CORRECTED_CARD_INSTANCE_LAYOUT.input_recipe,
        "token_signature": signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        "source_eligibility": "not-authorized-by-collection-metadata",
        **neural_fields,
    }


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

    def __init__(self, directory: Path, *, work_deadline: float, cleanup_deadline: float) -> None:
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
        errors: list[BaseException] = []
        # Failure for one record never prevents an independent attempt for the
        # other record, within the same original cleanup allowance.
        for output in self.outputs:
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

    def _write(self, name: str, value: Json) -> None:
        self._check()
        assert self.directory_fd is not None
        raw = (canonical(value) + "\n").encode()
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

    def commit(self, report: Json, manifest: Json) -> None:
        if self.committed or self.outputs or self._directory_handle is not None:
            raise ContractError("collection publication owner already consumed")
        if report.get("healthy") is not True or manifest.get("schema_version") != 5:
            raise ContractError("collection healthy report and manifest5 required")
        try:
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._directory_handle = self._own_handle(descriptor)
            self.directory_identity = _inode(os.fstat(descriptor))
            self._write("report.json", report)
            self._write("manifest.json", manifest)
            for output in self.outputs:
                assert output.handle.descriptor is not None
                self._verify(output.name, output.handle.descriptor, output.raw)
            assert self.directory_fd is not None
            os.fsync(self.directory_fd)
            self._check()  # final check while both original output descriptors are live.
            self._release()  # close errors are publication failures, even after actual release.
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
