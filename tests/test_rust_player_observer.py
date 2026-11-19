"""Standalone observer smoke tests; never a training/benchmark controller."""

from __future__ import annotations

import ctypes
import json
import os
import select
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/rust_player_observer.rs"


@pytest.fixture(scope="module")
def observer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert SOURCE.is_file(), "standalone observer implementation is missing"
    output = tmp_path_factory.mktemp("rust-observer-bin") / "observer"
    subprocess.run(
        ["rustc", "--edition=2024", "-D", "warnings", str(SOURCE), "-o", str(output)], check=True
    )
    return output


def invoke(observer: Path, output: Path, *command: str, sampling: bool = False, preexec_fn=None):
    args = [str(observer), "--output", str(output)]
    if not sampling:
        args.append("--no-sampling")
    args.extend(["--", *command])
    completed = subprocess.run(
        args, check=False, preexec_fn=preexec_fn, capture_output=True, timeout=15
    )
    return completed, json.loads((output / "summary.json").read_text())


def test_closed_json_logs_and_no_sampling(observer: Path, tmp_path: Path) -> None:
    result, summary = invoke(
        observer,
        tmp_path / "out",
        "/bin/sh",
        "-c",
        "printf 'out'; printf 'err' >&2",
        'quote"\n\\\t',
    )
    assert result.returncode == 0, result.stderr
    assert set(summary) == {
        "schema_version",
        "kind",
        "command",
        "clock",
        "observer",
        "command_wait",
        "sampling",
        "peak_complete_pss_bytes",
        "coverage",
        "errors",
        "limitations",
    }
    assert summary["schema_version"] == 1
    assert summary["kind"] == "rust-player-process-observation-v1"
    assert summary["command"]["argv"][-1] == 'quote"\n\\\t'
    assert summary["command_wait"] == {"exit_code": 0, "signal": None, "observed": True}
    assert summary["peak_complete_pss_bytes"] is None
    assert summary["coverage"]["pss_sweeps"] == 0
    assert (
        summary["command"]["wall_ns"]
        == summary["command"]["finish_monotonic_ns"] - summary["command"]["start_monotonic_ns"]
    )
    assert (tmp_path / "out" / "command.stdout.log").read_text() == "out"
    assert (tmp_path / "out" / "command.stderr.log").read_text() == "err"
    assert (tmp_path / "out" / "process.jsonl").read_text() == ""
    assert (tmp_path / "out" / "pss.jsonl").read_text() == ""


@pytest.mark.parametrize(
    ("script", "wrapper", "code", "sig"),
    [("exit 23", 23, 23, None), ("exit 143", 143, 143, None), ("kill -TERM $$", 143, None, 15)],
)
def test_actual_status_distinguishes_code_and_signal(
    observer: Path, tmp_path: Path, script: str, wrapper: int, code: int | None, sig: int | None
) -> None:
    result, summary = invoke(observer, tmp_path / "out", "/bin/sh", "-c", script)
    assert result.returncode == wrapper
    assert summary["command_wait"] == {"exit_code": code, "signal": sig, "observed": True}


def ignore_sigchld() -> None:
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


def auto_reap_sigchld() -> None:
    class Action(ctypes.Structure):
        _fields_ = [
            ("handler", ctypes.c_void_p),
            ("mask", ctypes.c_ulong * 16),
            ("flags", ctypes.c_int),
            ("restorer", ctypes.c_void_p),
        ]

    action = Action()
    action.flags = 2
    if ctypes.CDLL(None).sigaction(signal.SIGCHLD, ctypes.byref(action), None) != 0:
        os._exit(99)


@pytest.mark.parametrize("disposition", [ignore_sigchld, auto_reap_sigchld])
def test_inherited_autoreap_is_cleared(observer: Path, tmp_path: Path, disposition) -> None:
    result, summary = invoke(
        observer, tmp_path / "out", "/bin/sh", "-c", "exit 19", preexec_fn=disposition
    )
    assert result.returncode == 19, (result.stderr, summary)
    assert summary["command_wait"] == {"exit_code": 19, "signal": None, "observed": True}


def test_quick_sampled_command_never_invents_zero_peak(observer: Path, tmp_path: Path) -> None:
    result, summary = invoke(observer, tmp_path / "out", "/bin/true", sampling=True)
    assert result.returncode == 0, (result.stderr, summary)
    assert summary["peak_complete_pss_bytes"] is None
    assert summary["coverage"]["peak_eligible_pss_sweeps"] == 0
    for name in ("process.jsonl", "pss.jsonl"):
        for line in (tmp_path / "out" / name).read_text().splitlines():
            json.loads(line)


def test_setup_failure_does_not_spawn(observer: Path, tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    marker = tmp_path / "spawned"
    result = subprocess.run(
        [str(observer), "--output", str(output), "--", "/usr/bin/touch", str(marker)],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert not marker.exists()


def wait_for(predicate, process: subprocess.Popen, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        assert process.poll() is None, "observer exited before handshake"
        select.select([], [], [], 0.01)
    raise AssertionError("handshake timed out")


def test_descendant_sampling_with_handshake(observer: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    gate = tmp_path / "gate"
    os.mkfifo(gate)
    ready = tmp_path / "ready"
    script = (
        """/bin/sh -c 'printf ready > "$2"; read line < "$1"' child "$1" "$2" & wait; exit 27"""
    )
    process = subprocess.Popen(
        [
            str(observer),
            "--output",
            str(output),
            "--",
            "/bin/sh",
            "-c",
            script,
            "parent",
            str(gate),
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(ready.exists, process)

        def descendants():
            path = output / "process.jsonl"
            if not path.exists():
                return False
            return any(
                len(json.loads(line)["processes"]) >= 2
                for line in path.read_text().splitlines()
                if line.endswith("}")
            )

        wait_for(descendants, process)
        assert process.poll() is None
        with gate.open("w") as stream:
            stream.write("done\n")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 27, (stdout, stderr)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["coverage"]["observed_processes"] >= 2
        assert summary["command_wait"]["exit_code"] == 27
    finally:
        if process.poll() is None:
            fd = os.open(gate, os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"done\n")
            os.close(fd)
            process.communicate(timeout=10)


def test_failed_spawn_is_unknown_never_success(observer: Path, tmp_path: Path) -> None:
    result, summary = invoke(observer, tmp_path / "out", "/does/not/exist")
    assert result.returncode == 2
    assert summary["command_wait"] == {"exit_code": None, "signal": None, "observed": False}
    assert summary["command"]["pid"] is None
    assert summary["errors"]


def test_complete_pss_and_closed_raw_records(observer: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    gate = tmp_path / "gate"
    ready = tmp_path / "ready"
    os.mkfifo(gate)
    process = subprocess.Popen(
        [
            str(observer),
            "--output",
            str(output),
            "--",
            "/bin/sh",
            "-c",
            'printf ready > "$2"; read line < "$1"',
            "held",
            str(gate),
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(ready.exists, process)

        def complete_sweep():
            path = output / "pss.jsonl"
            if not path.exists():
                return False
            return any(
                json.loads(line)["complete"] and json.loads(line)["processes"]
                for line in path.read_text().splitlines()
                if line.endswith("}")
            )

        wait_for(complete_sweep, process)
        with gate.open("w") as stream:
            stream.write("done\n")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["peak_complete_pss_bytes"] > 0
        assert summary["coverage"]["peak_eligible_pss_sweeps"] >= 1
        assert (
            summary["observer"]["finish_monotonic_ns"] >= summary["command"]["finish_monotonic_ns"]
        )
        for line in (output / "process.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert set(row) == {"monotonic_ns", "processes", "errors"}
            for entry in row["processes"]:
                assert set(entry) == {
                    "pid",
                    "start_ticks",
                    "ppid",
                    "state",
                    "user_ticks",
                    "system_ticks",
                    "rss_bytes",
                    "threads",
                    "executable",
                    "argv",
                    "tasks",
                }
                for task in entry["tasks"] or []:
                    assert set(task) == {"tid", "name"}
        for line in (output / "pss.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert set(row) == {
                "start_monotonic_ns",
                "finish_monotonic_ns",
                "complete",
                "total_pss_bytes",
                "processes",
                "errors",
            }
            for entry in row["processes"]:
                assert set(entry) == {
                    "pid",
                    "start_ticks",
                    "start_monotonic_ns",
                    "finish_monotonic_ns",
                    "pss_bytes",
                    "error",
                }
    finally:
        if process.poll() is None:
            fd = os.open(gate, os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"done\n")
            os.close(fd)
            process.communicate(timeout=10)


def limited_diagnostic_files() -> None:
    import resource

    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (8192, 8192))


@pytest.mark.parametrize(("child_code", "expected_wrapper"), [(0, 2), (27, 27)])
def test_sampler_write_error_keeps_actual_child_until_wait(
    observer: Path, tmp_path: Path, child_code: int, expected_wrapper: int
) -> None:
    output = tmp_path / "out"
    gate = tmp_path / "gate"
    ready = tmp_path / "ready"
    os.mkfifo(gate)
    script = f'printf ready > "$2"; read line < "$1"; exit {child_code}'
    process = subprocess.Popen(
        [
            str(observer),
            "--output",
            str(output),
            "--",
            "/bin/sh",
            "-c",
            script,
            "held",
            str(gate),
            str(ready),
            "x" * 3500,
        ],
        preexec_fn=limited_diagnostic_files,  # noqa: PLW1509 -- Single-threaded fixture sets inherited Linux limits before exec.
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(ready.exists, process)

        def raw_write_failed():
            raw = output / "process.jsonl"
            return raw.exists() and raw.stat().st_size == 8192

        wait_for(raw_write_failed, process)
        assert process.poll() is None, "diagnostic failure dropped the live original child"
        with gate.open("w") as stream:
            stream.write("done\n")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == expected_wrapper, (stdout, stderr)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["command_wait"] == {
            "exit_code": child_code,
            "signal": None,
            "observed": True,
        }
        assert any("sampler failed" in error for error in summary["errors"])
        assert summary["peak_complete_pss_bytes"] is None
    finally:
        if process.poll() is None:
            fd = os.open(gate, os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"done\n")
            os.close(fd)
            process.communicate(timeout=10)


def limited_summary_file() -> None:
    import resource

    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))


@pytest.mark.parametrize(("child_code", "expected_wrapper"), [(0, 2), (27, 27)])
def test_summary_and_stderr_failure_preserve_actual_wait(
    observer: Path, tmp_path: Path, child_code: int, expected_wrapper: int
) -> None:
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # Real broken diagnostic stderr; command has its own log.
    try:
        result = subprocess.run(
            [
                str(observer),
                "--output",
                str(tmp_path / "out"),
                "--no-sampling",
                "--",
                "/bin/sh",
                "-c",
                f"exit {child_code}",
            ],
            preexec_fn=limited_summary_file,  # Set inherited Linux limits before exec.
            stdout=subprocess.DEVNULL,
            stderr=write_fd,
            check=False,
            timeout=10,
        )
        assert result.returncode == expected_wrapper
    finally:
        os.close(write_fd)


@pytest.mark.parametrize("replacement", ["symlink", "regular"])
@pytest.mark.parametrize("child_code", [0, 27])
def test_replaced_pss_path_never_certifies_foreign_peak(
    observer: Path, tmp_path: Path, replacement: str, child_code: int
) -> None:
    output = tmp_path / "out"
    gate = tmp_path / "gate"
    ready = tmp_path / "ready"
    os.mkfifo(gate)
    process = subprocess.Popen(
        [
            str(observer),
            "--output",
            str(output),
            "--",
            "/bin/sh",
            "-c",
            f'printf ready > "$2"; read line < "$1"; exit {child_code}',
            "held",
            str(gate),
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(ready.exists, process)
        stamp = time.monotonic_ns()
        forged = tmp_path / "foreign.jsonl"
        forged.write_text(
            json.dumps(
                {
                    "start_monotonic_ns": stamp,
                    "finish_monotonic_ns": stamp,
                    "complete": True,
                    "total_pss_bytes": 1 << 50,
                    "processes": [{}],
                    "errors": [],
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        published = output / "pss.jsonl"
        if replacement == "symlink":
            published.unlink()
            published.symlink_to(forged)
        else:
            os.replace(forged, published)
        with gate.open("w") as stream:
            stream.write("done\n")
        stdout, stderr = process.communicate(timeout=10)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["peak_complete_pss_bytes"] is None, summary
        assert summary["coverage"]["peak_eligible_pss_sweeps"] == 0
        assert summary["errors"], "replaced published raw file must mark incomplete"
        assert summary["command_wait"] == {
            "exit_code": child_code,
            "signal": None,
            "observed": True,
        }
        assert process.returncode == (2 if child_code == 0 else child_code), (stdout, stderr)
    finally:
        if process.poll() is None:
            fd = os.open(gate, os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"done\n")
            os.close(fd)
            process.communicate(timeout=10)
