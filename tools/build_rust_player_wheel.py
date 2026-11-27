#!/usr/bin/env python3
"""Explicit build-time runtime wheel. Runtime discovery never invokes this tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ASSETS = (
    "structured-token-recipe-v1.json",
    "structured-token-recipe-public-exact-v2.json",
    "card-instance-token-recipe-corrected-v2.json",
    "card-instance-validation-pins-v2.json",
    "card-instance-defense-v3-release.json",
    "card-instance-formal-ppo-v1.json",
    "card-instance-formal-ppo-group-pack-v1.json",
    "card-instance-tree3-layout-recipe-v1.json",
)


def _inventory(native: Path) -> tuple[str, str]:
    checksum = hashlib.sha256()
    paths = [
        *native.joinpath("src").rglob("*.rs"),
        native / "build.rs",
        *(native / "assets" / name for name in ASSETS),
    ]
    for path in sorted(paths):
        checksum.update(str(path.relative_to(native)).encode() + b"\0" + path.read_bytes() + b"\0")
    return checksum.hexdigest(), hashlib.sha256((native / "Cargo.lock").read_bytes()).hexdigest()


def _build(native: Path, output: Path, python: str, features: tuple[str, ...]) -> Path:
    staging = native / "gf_native_runtime"
    # Never adopt, replace or remove another invocation's staging.
    staging.mkdir(mode=0o700)
    owned = staging.stat()
    try:
        original = _inventory(native)
        command = [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            str(native / "Cargo.toml"),
            "--bin",
            "gf-player-supervisor",
        ]
        if features:
            command += ["--features", ",".join(features)]
        subprocess.run(command, check=True)
        cargo = json.loads(
            subprocess.run(
                [
                    "cargo",
                    "metadata",
                    "--locked",
                    "--no-deps",
                    "--format-version",
                    "1",
                    "--manifest-path",
                    str(native / "Cargo.toml"),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        binary = Path(cargo["target_directory"]) / "release/gf-player-supervisor"
        handshake = json.loads(
            subprocess.run(
                [str(binary), "--build-info"], check=True, capture_output=True, text=True
            ).stdout
        )
        build = handshake["build"]
        if (
            handshake["kind"] != "rust-player-supervisor-build-v1"
            or build["native_source_sha256"] != original[0]
            or build["cargo_lock_sha256"] != original[1]
            or build["profile"] != "release"
            or _inventory(native) != original
        ):
            raise RuntimeError("supervisor source/lock/build changed during compilation")
        staged_binary = staging / "gf-player-supervisor"
        shutil.copyfile(binary, staged_binary)
        staged_binary.chmod(0o755)
        manifest = {
            "schema_version": 1,
            "kind": "rust-player-supervisor-manifest-v1",
            "executable_sha256": hashlib.sha256(staged_binary.read_bytes()).hexdigest(),
            "build": build,
            "features": list(features),
            "extension_features": ["pyo3/extension-module"],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        )
        output.mkdir(parents=True, exist_ok=True)
        # Publish a wheel only after source/lock consistency is confirmed again.
        with tempfile.TemporaryDirectory(prefix=".rust-player-build-", dir=output) as scratch:
            command = [
                "uv",
                "tool",
                "run",
                "--offline",
                "maturin",
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                str(native / "Cargo.toml"),
                "--interpreter",
                python,
                "--out",
                scratch,
            ]
            if features:
                command += ["--features", ",".join(features)]
            subprocess.run(command, check=True)
            if _inventory(native) != original:
                raise RuntimeError("native source/lock changed during wheel build")
            wheels = list(Path(scratch).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("expected exactly one native wheel")
            destination = output / wheels[0].name
            os.link(wheels[0], destination)  # do not overwrite an existing artifact
            return destination
    finally:
        current = staging.lstat()
        if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise RuntimeError("runtime wheel staging ownership changed; cleanup refused")
        allowed = {"gf-player-supervisor", "manifest.json"}
        if any(
            path.name not in allowed or path.is_symlink() or not path.is_file()
            for path in staging.iterdir()
        ):
            raise RuntimeError("runtime wheel staging contains foreign entries; cleanup refused")
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--allocator-mimalloc", action="store_true")
    args = parser.parse_args()
    native = Path(__file__).resolve().parents[1] / "native"
    features = ("allocator-mimalloc",) if args.allocator_mimalloc else ()
    print(_build(native, args.out.resolve(), args.python, features))


if __name__ == "__main__":
    main()
