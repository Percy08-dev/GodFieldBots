"""Original supervision and scheduling authority; no GPU/data qualification."""

from importlib.util import find_spec

import pytest


def test_supervised_protocol_available_without_optional_ml_imports():
    # Missing/misrouted closed control definitions must fail before game/GPU I/O.
    assert find_spec("gf_common.rust_player") is not None


def test_missing_runtime_companion_refuses_without_path_or_runtime_build(monkeypatch):
    import importlib

    import pytest

    from gf_common.domain import ContractError

    module = importlib.import_module("gf_simulator_client.rust_player_companion")
    monkeypatch.setattr(
        module.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(module.metadata.PackageNotFoundError(name)),
    )
    with pytest.raises(ContractError, match="runtime wheel"):
        module.discover_companion({"extension_sha256": "a" * 64, "build": {}})


def test_wheel_builder_never_removes_preexisting_staging(tmp_path):
    import importlib.util
    from pathlib import Path

    import pytest

    script = Path(__file__).parents[1] / "tools/build_rust_player_wheel.py"
    assert script.is_file(), "runtime-wheel build entry is missing"
    spec = importlib.util.spec_from_file_location("runtime_wheel_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = tmp_path / "native"
    staging = native / "gf_native_runtime"
    staging.mkdir(parents=True)
    sentinel = staging / "someone-elses-file"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        module._build(native, tmp_path / "wheels", "/unused/python", ())
    assert sentinel.read_text() == "preserve"


def _companion_fixture(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from gf_simulator_client import rust_player_companion as companion

    build = dict.fromkeys(companion.BUILD_KEYS, "identity")
    build.update(
        profile="release",
        abi="cp314-gil",
        rustflags="",
        allocator="system",
        native_source_sha256="b" * 64,
        cargo_lock_sha256="c" * 64,
    )
    root = tmp_path / "gf_native_runtime"
    root.mkdir()
    binary = root / "gf-player-supervisor"
    handshake = json.dumps(
        {"schema_version": 1, "kind": "rust-player-supervisor-build-v1", "build": build}
    )
    binary.write_text("#!/bin/sh\nprintf '%s\\n' '" + handshake + "'\n")
    binary.chmod(0o755)
    manifest = {
        "schema_version": 1,
        "kind": "rust-player-supervisor-manifest-v1",
        "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "build": build,
        "features": [],
        "extension_features": ["pyo3/extension-module"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    inventory = ["gf_native_runtime/gf-player-supervisor", "gf_native_runtime/manifest.json"]
    distribution = SimpleNamespace(files=inventory, locate_file=lambda name: tmp_path / name)
    monkeypatch.setattr(companion.metadata, "distribution", lambda name: distribution)
    return companion, binary, manifest, {"extension_sha256": "a" * 64, "build": dict(build)}


def test_installed_inventory_hash_and_real_build_handshake_are_all_required(tmp_path, monkeypatch):
    import pytest

    from gf_common.domain import ContractError

    module, binary, manifest, loaded = _companion_fixture(tmp_path, monkeypatch)
    found = module.discover_companion(loaded)
    assert found.executable == binary
    assert found.manifest == manifest
    loaded["build"]["native_source_sha256"] = "wrong-source"
    with pytest.raises(ContractError, match="build mismatch"):
        module.discover_companion(loaded)
    loaded["build"] = manifest["build"]
    binary.write_text("tampered")
    with pytest.raises(ContractError, match="hash mismatch"):
        module.discover_companion(loaded)


def test_companion_symlink_cannot_satisfy_regular_installed_executable(tmp_path, monkeypatch):
    import pytest

    from gf_common.domain import ContractError

    module, binary, _, loaded = _companion_fixture(tmp_path, monkeypatch)
    elsewhere = binary.with_name("elsewhere")
    binary.rename(elsewhere)
    binary.symlink_to(elsewhere)
    with pytest.raises(ContractError, match="regular executable"):
        module.discover_companion(loaded)


def test_failed_wheel_compilation_removes_only_its_own_empty_staging(tmp_path, monkeypatch):
    import importlib.util
    import subprocess
    from pathlib import Path

    import pytest

    script = Path(__file__).parents[1] / "tools/build_rust_player_wheel.py"
    spec = importlib.util.spec_from_file_location("runtime_wheel_builder_failure", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = tmp_path / "native"
    (native / "src").mkdir(parents=True)
    (native / "assets").mkdir()
    (native / "build.rs").write_text("fn main(){}")
    (native / "Cargo.lock").write_text("version = 4\n")
    for name in module.ASSETS:
        (native / "assets" / name).write_text("{}")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cargo")),
    )
    with pytest.raises(subprocess.CalledProcessError):
        module._build(native, tmp_path / "wheels", "/unused/python", ())
    assert not (native / "gf_native_runtime").exists()
    assert (native / "build.rs").read_text() == "fn main(){}"


def test_loaded_extension_hash_must_be_actual_sha256(tmp_path, monkeypatch):
    import pytest

    from gf_common.domain import ContractError

    module, _, _, loaded = _companion_fixture(tmp_path, monkeypatch)
    loaded["extension_sha256"] = "invented"
    with pytest.raises(ContractError, match="identity"):
        module.discover_companion(loaded)


def test_original_deadline_bits_round_trip_and_invalid_values_reject():
    import math

    import pytest

    from gf_common.domain import ContractError
    from gf_common.rust_player import deadline_bits, deadline_seconds

    original = 0.1 + 123456.0
    assert deadline_seconds(deadline_bits(original)) == original
    for value in (True, math.inf, math.nan, -1, 0):
        with pytest.raises(ContractError):
            deadline_bits(value)
    for bits in ("3FF0000000000000", "7ff0000000000000", "bff0000000000000", "0" * 16):
        with pytest.raises(ContractError):
            deadline_seconds(bits)


def _new_namespace_records():
    import time

    from gf_common.domain import digest
    from gf_common.rust_player import deadline_bits

    now = time.monotonic()
    build = {
        key: "actual-build"
        for key in (
            "version",
            "rustc",
            "target",
            "profile",
            "rustflags",
            "cargo_lock_sha256",
            "native_source_sha256",
            "abi",
            "allocator",
        )
    }
    build.update(
        profile="release",
        rustflags="",
        abi="cp314-gil",
        allocator="system",
        cargo_lock_sha256="a" * 64,
        native_source_sha256="b" * 64,
    )
    producer = {
        "schema_version": 1,
        "kind": "rust-player-producer-v1",
        "build": build,
        **dict.fromkeys(
            (
                "python_source_digest",
                "python_runtime_digest",
                "extension_sha256",
                "supervisor_sha256",
                "sampling_runtime_sha256",
            ),
            "c" * 64,
        ),
    }
    context = {
        "campaign_id": "campaign",
        "operation_id": "operation",
        "kind": "training",
        "run_id": "run",
        "evaluation_id": None,
    }
    local = {
        "schema_version": 1,
        "kind": "rust-player-admission-v1",
        "supervision_id": "1" * 32,
        "invocation_id": "2" * 32,
        "nonce": "3" * 32,
        "supervisor": {"pid": 41001, "start_ticks": 1},
        "host": {"pid": 41002, "pgid": 41002, "start_ticks": 2},
        "context": context,
        "operation_id": "operation",
        "run_id": "run",
        "config_sha256": "d" * 64,
        "runtime_binding_sha256": "e" * 64,
        "global_ticket": "4" * 32,
        "producer": producer,
        "deadlines": {
            key: deadline_bits(now + index + 1)
            for index, key in enumerate(
                ("admission_bits", "execution_bits", "term_bits", "kill_bits", "finish_bits")
            )
        },
        "budget": dict.fromkeys(
            (
                "managed_ram_bytes",
                "managed_cuda_bytes",
                "game_concurrency",
                "player_worker_threads",
                "max_batch_size",
                "storage_bytes",
                "source_workspace_bytes",
            ),
            1,
        ),
    }
    admission = {
        "schema_version": 2,
        "kind": "rust-player-gpu-admission-v1",
        "ticket": "4" * 32,
        "context": context,
        "boot_id": "boot",
        "owner_pid": 41001,
        "admitted_unix": time.time(),
        "host_admission": local,
    }
    exit_record = {
        "schema_version": 1,
        "kind": "rust-player-exit-v1",
        "supervision_id": local["supervision_id"],
        "invocation_id": local["invocation_id"],
        "nonce": local["nonce"],
        "admission_sha256": digest(local),
        "prepared_sha256": None,
        "host_original": local["host"],
        "wait": {"exit_code": None, "signal": 9, "core_dumped": False},
        "descendants": [],
        "signals": [{"pgid": 41002, "signal": 9, "delivered": True}],
        "descendants_reaped": True,
        "cooperative": False,
        "finished_unix_ns": time.time_ns(),
    }
    receipt = {
        "schema_version": 3,
        "kind": "rust-player-gpu-termination-v1",
        "ticket": admission["ticket"],
        "admission_sha256": digest(admission),
        "context_sha256": digest(context),
        "boot_id": "boot",
        "confirmed_unix": time.time(),
        "exit_sha256": digest(exit_record),
        "exit_record": exit_record,
    }
    return admission, receipt


def test_legacy_scheduling_reader_dispatches_new_physical_records_without_old_job_reconstruction(
    tmp_path, monkeypatch
):
    from dataclasses import asdict

    import pytest

    from gf_common.domain import ContractError, digest
    from gf_training import hpo_gpu_lease as gpu

    root = tmp_path / "namespace"
    monkeypatch.setattr(gpu, "_GLOBAL_ROOT", root)
    admission, receipt = _new_namespace_records()
    gpu._admission(admission, admission["ticket"])
    gpu._termination(receipt, admission)
    gpu._ensure_root(root)
    (root / "jobs").mkdir(mode=0o700)
    directory = root / "jobs" / admission["ticket"]
    directory.mkdir(mode=0o700)
    gpu._write(directory / "admission.json", admission, immutable=True)
    gpu._write(
        root / "current.json",
        {"schema_version": 1, "ticket": admission["ticket"], "admission_sha256": digest(admission)},
        immutable=False,
    )
    with pytest.raises(gpu.RecoveryRequired):
        gpu.admit_gpu_job(
            gpu.JobContext("next", "operation", "training", "run", None),
            global_cuda_bytes=1,
            max_workers=1,
        )
    gpu._write(directory / "termination.json", receipt, immutable=True)
    with gpu.admit_gpu_job(
        gpu.JobContext("next", "operation", "training", "run", None),
        global_cuda_bytes=1,
        max_workers=1,
    ) as actual:
        assert actual._admission["schema_version"] == 1
        original = actual.confirm_termination()
        assert asdict(original)["kind"] == "hpo-gpu-termination"
    receipt["exit_record"]["descendants_reaped"] = False
    receipt["exit_sha256"] = digest(receipt["exit_record"])
    with pytest.raises(ContractError):
        gpu._termination(receipt, admission)


def test_new_control_files_are_exclusive_canonical_without_newline(tmp_path):
    import hashlib

    import pytest

    from gf_common.domain import ContractError, canonical
    from gf_training.rust_player_supervision import read_control_record, write_control_record

    value = {"integer": 2**64 - 1, "null": None, "bits": "3ff0000000000000", "unicode": "資料😀"}
    path = tmp_path / "control.json"
    sha = write_control_record(path, value)
    assert path.read_bytes() == canonical(value).encode()
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_control_record(path, tmp_path) == (value, sha)
    assert write_control_record(path, value) == sha
    with pytest.raises(ContractError):
        write_control_record(path, {"changed": True})


def _host_fixture(mode, control_fd, launch_fd, phase):
    """Private actual-protocol fixture. Emits incomplete component data, never Source."""
    import copy
    import fcntl
    import hashlib
    import os
    import signal
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from gf_common.domain import digest
    from gf_training.rust_player_supervision import (
        HostControl,
        runtime_producer,
        write_control_record,
    )

    if mode == "handshake-exit":
        os._exit(17)
    control = HostControl(control_fd, launch_fd, phase=phase)
    root = control.root
    (root / (phase + "-started")).write_text(str(os.getpid()))
    (root / (phase + "-started-" + control.operation["operation_id"])).write_text(str(os.getpid()))
    second = control.operation["operation_id"] == "operation-second"
    prefix = "second-" if second else ""
    parent = Path(control.operation["parent_directory"])

    def competitor_blocked():
        fd = os.open(parent / ".lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            raise AssertionError("actual parent custody lost")
        finally:
            os.close(fd)

    if phase == "cold":
        competitor_blocked()
        if mode == "cold-fail":
            os._exit(11)
        if mode in ("cold-delay", "cold-expire"):
            from gf_common.rust_player import deadline_seconds

            cutoff = deadline_seconds(
                control.operation["deadlines"][
                    "finish_bits" if mode == "cold-expire" else "term_bits"
                ]
            )
            while time.monotonic() < cutoff + 0.12:
                time.sleep(0.005)
        if mode == "session-expire":
            from gf_common.rust_player import deadline_seconds

            session_cutoff = deadline_seconds(control.launch["session_deadlines"]["finish_bits"])
        if mode == "unfinalized-bind":
            control.bind_operation(control.operation_path)
            raise AssertionError("unfinalized bind was accepted")
        _, prepared, exit_record = control.cold.require_pair()
        assert control.cold.original_exit.require() == exit_record
        assert control.cold.physical_termination.require() == exit_record
        for capability in (control.cold.original_exit, control.cold.physical_termination):
            try:
                copy.copy(capability)
            except TypeError:
                pass
            else:
                raise AssertionError("actual physical capability was copyable")
        assert prepared["state"] == "incomplete"  # no final-parent/data qualification
        completion = {
            "schema_version": 1,
            "kind": "rust-player-training-completion-v1",
            "prepared_sha256": digest(prepared),
            "exit_sha256": digest(exit_record),
            "component_only": True,
            "state": "incomplete",
        }
        path = root / (prefix + "component-finalizer-output.json")
        write_control_record(path, completion)
        control.cold.acknowledge_finalizer_output(path)
        control.cold.acknowledge_finalizer_output(path)  # exact-pair retry is idempotent
        if mode == "session-expire":
            control.release_parent()
            while time.monotonic() < session_cutoff + 0.12:
                time.sleep(0.005)
        if mode == "same-operation":
            control.release_parent()
            changed = copy.deepcopy(control.operation)
            from gf_common.rust_player import deadline_bits

            now = time.monotonic()
            changed["deadlines"] = {
                key: deadline_bits(now + offset)
                for key, offset in zip(
                    ("admission_bits", "execution_bits", "term_bits", "kill_bits", "finish_bits"),
                    (1, 1.5, 2, 2.5, 3.5),
                )
            }
            path = root / "same-operation-changed.json"
            write_control_record(path, changed)
            control.bind_operation(path)
            raise AssertionError("same operation renewed original deadlines")
        if mode == "two-operation" and not second:
            from gf_common.rust_player import deadline_bits, deadline_seconds

            control.release_parent()
            previous_finish = deadline_seconds(control.operation["deadlines"]["finish_bits"])
            while time.monotonic() < previous_finish + 0.12:
                time.sleep(0.005)
            changed = copy.deepcopy(control.operation)
            now = time.monotonic()
            changed.update(
                operation_id="operation-second",
                run_id="run-second",
                parent_directory=str(root / "parent-second"),
            )
            changed["expected_context"].update(operation_id="operation-second", run_id="run-second")
            changed["deadlines"] = {
                key: deadline_bits(now + offset)
                for key, offset in zip(
                    ("admission_bits", "execution_bits", "term_bits", "kill_bits", "finish_bits"),
                    (1, 1.5, 2, 2.5, 3.5),
                )
            }
            parent = Path(changed["parent_directory"])
            parent.mkdir()
            path = root / "operation-second.json"
            write_control_record(path, changed)
            control.bind_operation(path)
            prefix = "second-"
        else:
            control.close()
            return
    fd = os.open(parent / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control.transfer_lock("parent", fd)
    os.close(fd)  # authenticated close-only detach, not LOCK_UN
    competitor_blocked()
    companion = SimpleNamespace(
        manifest={
            "executable_sha256": hashlib.sha256(
                Path(f"/proc/{os.getppid()}/exe").read_bytes()
            ).hexdigest()
        }
    )
    producer = runtime_producer(companion, sampling_runtime_sha256="a" * 64)
    budget = {
        key: 1
        for key in (
            "managed_ram_bytes",
            "game_concurrency",
            "player_worker_threads",
            "max_batch_size",
            "storage_bytes",
            "source_workspace_bytes",
        )
    }
    budget["managed_cuda_bytes"] = 0
    admission = control.admit(budget, producer)
    try:
        copy.copy(admission)
    except TypeError:
        pass
    else:
        raise AssertionError("live admission was copyable")
    if mode in ("duplicate-stop-ack", "legitimate-stop"):
        while not control.poll_stop():
            time.sleep(0.005)
        if mode == "duplicate-stop-ack":
            control._send("stop_ack", control._server_sequence - 1, {})
    if mode == "unsolicited-stop-ack":
        control._request("stop", {"reason": "component-stop"}, "stop_ack", set())
        control._send("stop_ack", 0, {})
    if mode == "exit-before-prepare":
        os._exit(7)
    if mode == "wrong-nonce":
        control._envelope["nonce"] = "0" * 32
        control._send("stop", control._next_sequence, {"reason": "component-fault"})
        os._exit(0)
    if mode == "control-eof-kill":
        control._socket.close()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    local = admission.observation
    common = {
        key: local[key]
        for key in (
            "supervision_id",
            "invocation_id",
            "operation_id",
            "run_id",
            "runtime_binding_sha256",
        )
    }
    common["admission_sha256"] = digest(local)
    servers = {
        "schema_version": 1,
        "kind": "rust-player-server-retirement-v1",
        **{key: common[key] for key in ("supervision_id", "invocation_id", "operation_id")},
        "servers": [],
    }
    servers_path = root / (prefix + "component-servers.json")
    servers_sha = write_control_record(servers_path, servers)
    close = {
        "schema_version": 1,
        "kind": "rust-player-host-close-ack-v1",
        **common,
        **dict.fromkeys(
            ("actors_joined", "service_joined", "writer_joined", "buffers_released", "gpu_fenced"),
            True,
        ),
        **dict.fromkeys(
            (
                "native_retirement_sha256",
                "writer_retirement_sha256",
                "service_retirement_sha256",
                "cache_retirement_sha256",
            ),
            "b" * 64,
        ),
        "servers_path": str(servers_path),
        "servers_sha256": servers_sha,
        "server_count": 0,
    }
    close_path = root / (prefix + "component-close.json")
    close_sha = write_control_record(close_path, close)
    prepared = {
        "schema_version": 1,
        "kind": "rust-player-prepared-v1",
        **common,
        "config_sha256": local["config_sha256"],
        "state": "incomplete",
        "committed_update": 0,
        "generation": 0,
        **dict.fromkeys(
            ("state_sha256", "generation_sha256", "artifacts_sha256", "source_manifest_sha256"),
            "c" * 64,
        ),
        "close_ack_path": str(close_path),
        "close_ack_sha256": close_sha,
    }
    prepared_path = root / (prefix + "component-prepared.json")
    write_control_record(prepared_path, prepared)
    control.prepare(admission, prepared_path)
    try:
        admission.require()
    except ValueError:
        pass
    else:
        raise AssertionError("prepared admission remained open")
    competitor_blocked()
    if mode == "prepared-nonzero":
        os._exit(13)
    if mode == "exit-write-fail":
        root.chmod(0o500)
    if mode == "prepared-tamper":
        prepared_path.write_text("{}")
    control.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 6 and sys.argv[1] == "_host":
        _host_fixture(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        raise SystemExit(2)


def test_actual_linux_memfd_launch_is_sealed_even_when_python_omits_seal_constants(tmp_path):
    import os
    import time
    from pathlib import Path

    import pytest

    from gf_common.rust_player import deadline_bits
    from gf_simulator_client.rust_player_companion import sealed_launch_fd
    from gf_training.rust_player_supervision import _launch_original

    admission, _ = _new_namespace_records()
    now = time.monotonic()
    launch = {
        "schema_version": 1,
        "kind": "rust-player-launch-v1",
        "supervision_id": "1" * 32,
        "python_executable": "/unused/python",
        "operation_path": str(tmp_path / "op.json"),
        "operation_sha256": "a" * 64,
        "record_directory": str(tmp_path),
        "mode": "ordinary",
        "session_deadlines": {"finish_bits": deadline_bits(now + 30)},
        "clock_binding": {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "monotonic_before_bits": deadline_bits(now),
            "monotonic_after_bits": deadline_bits(now),
            "unix_anchor_bits": deadline_bits(time.time()),
        },
        "binding": {
            "python_abi": "cpython-314",
            "extension_sha256": "b" * 64,
            "build": admission["host_admission"]["producer"]["build"],
        },
        "locks": [],
    }
    fd = sealed_launch_fd(launch)
    try:
        assert _launch_original(fd) == launch
        with pytest.raises(PermissionError):
            os.pwrite(fd, b"x", 0)
        with pytest.raises(PermissionError):
            os.ftruncate(fd, 0)
    finally:
        os.close(fd)


@pytest.mark.parametrize("missing", ["distribution", "inventory"])
def test_fix1_missing_companion_gives_exact_build_install_and_no_sync_launch(monkeypatch, missing):
    from types import SimpleNamespace

    from gf_common.domain import ContractError
    from gf_simulator_client import rust_player_companion as companion

    def distribution(name):
        if missing == "distribution":
            raise companion.metadata.PackageNotFoundError(name)
        return SimpleNamespace(files=[])

    monkeypatch.setattr(companion.metadata, "distribution", distribution)
    with pytest.raises(ContractError) as error:
        companion.discover_companion({"extension_sha256": "a" * 64, "build": {}})
    message = str(error.value)
    for required in (
        "tools/build_rust_player_wheel.py",
        "--python",
        "--out",
        "uv pip install",
        "godfieldbots_native-*.whl",
        ".venv/bin/gf-bots",
        "uv run --no-sync",
        "editable",
    ):
        assert required in message
