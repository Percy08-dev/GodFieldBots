"""Installed runtime-wheel companion discovery. Never searches PATH or builds."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from gf_common.domain import ContractError, Json, canonical, closed
from gf_common.rust_player import LINUX_F_ADD_SEALS, LINUX_REQUIRED_LAUNCH_SEALS, validate_launch

BUILD_KEYS = frozenset(
    {
        "version",
        "rustc",
        "target",
        "profile",
        "rustflags",
        "cargo_lock_sha256",
        "native_source_sha256",
        "abi",
        "allocator",
    }
)


_MISSING_RUNTIME = (
    "rust-player requires the installed native runtime wheel companion. "
    "From the GodFieldBots checkout, build and install it with:\n"
    "  uv run --no-sync python tools/build_rust_player_wheel.py "
    "--python .venv/bin/python --out dist/rust-player-runtime\n"
    "  uv pip install --python .venv/bin/python --reinstall-package godfieldbots-native "
    "dist/rust-player-runtime/godfieldbots_native-*.whl\n"
    "Launch .venv/bin/gf-bots or uv run --no-sync gf-bots. "
    "uv run without --no-sync may replace the runtime wheel with the "
    "companion-free editable native workspace package. No runtime build or fallback occurs."
)


@dataclass(frozen=True)
class RuntimeCompanion:
    executable: Path
    manifest: Json


def _regular(path: Path, *, executable: bool = False) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or executable and not os.access(path, os.X_OK):
        raise ContractError("runtime companion must be a regular executable")


def discover_companion(native_identity: Json) -> RuntimeCompanion:
    """Bind exact installed inventory, binary hash and handshake to loaded code."""
    extension_hash = native_identity.get("extension_sha256")
    if (
        type(extension_hash) is not str
        or len(extension_hash) != 64
        or any(character not in "0123456789abcdef" for character in extension_hash)
    ):
        raise ContractError("invalid loaded extension identity")
    try:
        distribution = metadata.distribution("godfieldbots-native")
    except metadata.PackageNotFoundError as error:
        raise ContractError(_MISSING_RUNTIME) from error
    inventory = {str(item): item for item in distribution.files or ()}
    names = ("gf_native_runtime/gf-player-supervisor", "gf_native_runtime/manifest.json")
    if any(name not in inventory for name in names):
        raise ContractError(_MISSING_RUNTIME)
    executable, manifest_path = (
        Path(str(distribution.locate_file(inventory[name]))) for name in names
    )
    _regular(executable, executable=True)
    _regular(manifest_path)
    try:
        manifest = closed(
            json.loads(manifest_path.read_text()),
            {
                "schema_version",
                "kind",
                "executable_sha256",
                "build",
                "features",
                "extension_features",
            },
        )
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["kind"] != "rust-player-supervisor-manifest-v1"
            or manifest["extension_features"] != ["pyo3/extension-module"]
            or type(manifest["features"]) is not list
            or manifest["features"] not in ([], ["allocator-mimalloc"])
        ):
            raise ContractError("unsupported runtime companion manifest")
        build = closed(manifest["build"], set(BUILD_KEYS))
        loaded = native_identity["build"]
        if (
            build["profile"] != "release"
            or build["abi"] != "cp314-gil"
            or any(loaded.get(key) != value for key, value in build.items())
            or not native_identity.get("extension_sha256")
        ):
            raise ContractError("runtime companion and loaded extension build mismatch")
        if hashlib.sha256(executable.read_bytes()).hexdigest() != manifest["executable_sha256"]:
            raise ContractError("runtime companion executable hash mismatch")
        completed = subprocess.run(
            [str(executable), "--build-info"],
            check=True,
            capture_output=True,
            timeout=10,
            text=True,
        )
        handshake = json.loads(completed.stdout)
        if handshake != {
            "schema_version": 1,
            "kind": "rust-player-supervisor-build-v1",
            "build": build,
        }:
            raise ContractError("runtime companion build handshake mismatch")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError("invalid runtime companion or build handshake") from error
    return RuntimeCompanion(executable, manifest)


def sealed_launch_fd(launch: Json) -> int:
    validate_launch(launch)
    data = canonical(launch).encode()
    if len(data) > 65536:
        raise ContractError("compact launch envelope exceeds control bound")
    fd = os.memfd_create("gf-rust-player-launch", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(data)
            output.flush()
        fcntl.fcntl(
            fd,
            LINUX_F_ADD_SEALS,
            LINUX_REQUIRED_LAUNCH_SEALS,
        )
        return fd
    except BaseException:
        os.close(fd)
        raise


def exec_supervisor(companion: RuntimeCompanion, launch: Json) -> None:
    """Replace the CLI; retain one Python host only after the Rust exec boundary."""
    validate_launch(launch)
    descriptor = os.open(companion.executable, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    launch_fd = None
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != companion.manifest["executable_sha256"]:
            raise ContractError("verified runtime companion changed before exec")
        launch_fd = sealed_launch_fd(launch)
        os.set_inheritable(launch_fd, True)
        for lock in launch["locks"]:
            os.set_inheritable(lock["fd"], True)
        # Linux execve accepts an executable FD; no pathname replacement race.
        os.execve(
            descriptor, [str(companion.executable), "--launch-fd", str(launch_fd)], dict(os.environ)
        )
    finally:
        os.close(descriptor)
        if launch_fd is not None:
            os.close(launch_fd)
