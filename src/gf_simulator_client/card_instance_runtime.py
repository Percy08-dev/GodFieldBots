"""Filesystem and live-process owner for the closed corrected release approval."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.card_instance import (
    CardInstanceContract,
    corrected_card_instance_contract,
    reviewed_specs,
    shared_card_instance_contract,
)
from gf_common.card_instance_runtime import (
    APPROVED_BINARY_SHA256,
    APPROVED_CHANGE_FILES,
    APPROVED_CHANGE_SHA256,
    SHARED_APPROVED_BINARY_SHA256,
    validate_runtime_receipt,
)
from gf_common.domain import ContractError, Json, clone


def _identity(value: os.stat_result) -> list[int]:
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _read_hash(path: Path, deadline: float) -> tuple[str, list[int]]:
    if time.monotonic() >= deadline:
        raise TimeoutError("runtime evidence read deadline")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK
    )  # /proc/PID/exe is intentionally followed.
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("runtime evidence must be a regular file")
        value = hashlib.sha256()
        offset = 0
        while chunk := os.pread(descriptor, 1024**2, offset):
            if time.monotonic() >= deadline:
                raise TimeoutError("runtime evidence read deadline")
            value.update(chunk)
            offset += len(chunk)
        if _identity(before) != _identity(os.fstat(descriptor)) or offset != before.st_size:
            raise ContractError("runtime evidence changed during read")
        if time.monotonic() >= deadline:
            raise TimeoutError("runtime evidence read deadline")
        return value.hexdigest(), _identity(before)
    finally:
        os.close(descriptor)


class CardInstanceRuntimeEvidence:
    """No caller-supplied approval hashes; retain original source/binary evidence."""

    def __init__(self, binary: Path, source_root: Path, *, deadline: float = float("inf"), contract: CardInstanceContract | None = None) -> None:
        self.contract = corrected_card_instance_contract() if contract is None else contract
        self.shared = self.contract == shared_card_instance_contract()
        self.binary, self.source_root = binary.absolute(), source_root.absolute()
        self.deadline = deadline
        self._receipt: Json | None = None
        self._closed = False

    def _source(self) -> Json:
        try:
            revision = subprocess.run(
                ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=max(0.001, min(10.0, self.deadline - time.monotonic())),
            ).stdout.strip()
        except subprocess.SubprocessError as error:
            raise ContractError("runtime source revision read failed") from error
        maps = []
        for expected in (reviewed_specs(self.contract), APPROVED_CHANGE_FILES):
            actual = {
                name: _read_hash(self.source_root / name, self.deadline)[0] for name in expected
            }
            if actual != dict(expected):
                raise ContractError("runtime source content differs from approved file inventory")
            maps.append(actual)
        result: Json = {
            "path": str(self.source_root),
            "revision": revision,
            "spec_files": maps[0],
            "change_files": maps[1],
        }
        if self.shared:
            status = subprocess.run(["git","-C",str(self.source_root),"status","--porcelain","--untracked-files=normal"],
                check=True,capture_output=True,text=True,timeout=max(.001,min(10.,self.deadline-time.monotonic())))
            result["dirty"] = bool(status.stdout.strip())
        return result

    def preflight(self) -> Json:
        if self._closed:
            raise ContractError("runtime owner already finalized")
        contract = self.contract
        sha, identity = _read_hash(self.binary, self.deadline)
        actual = {
            "schema_version": 2 if self.shared else 1,
            "action_contract": asdict(contract),
            "build": {
                "binary_sha256": SHARED_APPROVED_BINARY_SHA256 if self.shared else APPROVED_BINARY_SHA256,
                "spec_inventory_sha256": contract.spec_inventory_sha256,
                "change_inventory_sha256": APPROVED_CHANGE_SHA256,
                "spec_files": reviewed_specs(contract),
                "change_files": dict(APPROVED_CHANGE_FILES),
            },
            "binary": {"path": str(self.binary), "sha256": sha, "identity": identity},
            "source": self._source(),
            "executed": None,
        }
        validate_runtime_receipt(actual)
        if self._receipt is not None:
            original = {**self._receipt, "executed": None}
            if actual != original:
                raise ContractError("runtime path/source evidence changed after preflight")
        else:
            self._receipt = actual
        return self.receipt()

    def receipt(self, *, require_executed: bool = False) -> Json:
        if self._receipt is None:
            raise ContractError("runtime evidence preflight required before allocation")
        return validate_runtime_receipt(self._receipt, require_executed=require_executed)

    def _start_time(self, process: Any) -> int:
        if process.returncode is not None:
            raise ContractError("owned runtime process is not live")
        # comm may contain spaces/parentheses; fields after final ')' start at state(3).
        status = Path(f"/proc/{process.pid}/stat").read_text()
        fields = status.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            raise ContractError("owned runtime process is a zombie")
        return int(fields[19])  # Linux starttime is field22.

    def verify_executed(self, process: Any) -> Json:
        if self._closed:
            raise ContractError("runtime owner already finalized")
        self.receipt()
        before = self._start_time(process)
        sha, identity = _read_hash(Path(f"/proc/{process.pid}/exe"), self.deadline)
        if self._start_time(process) != before:
            raise ContractError("runtime process identity changed during executable read")
        assert self._receipt is not None
        executed = {"pid": process.pid, "start_time": before, "sha256": sha, "identity": identity}
        validate_runtime_receipt({**self._receipt, "executed": executed}, require_executed=True)
        self._receipt["executed"] = executed
        return self.receipt(require_executed=True)

    def recheck(self) -> Json:
        self.preflight()
        return self.receipt(require_executed=True)

    def finalize(self) -> Json:
        value = self.recheck()
        self._closed = True
        return clone(value)
