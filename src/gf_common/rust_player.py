"""Closed supervised-runtime observations. No I/O, native or learning imports.

Validators grant no live admission, termination, Source or final-parent authority.
"""

from __future__ import annotations

import math
import struct
from typing import Any, cast

from gf_common.domain import ContractError, Json, clone, closed, digest, integer

# Linux UAPI fcntl.h values. Some verified CPython builds omit these constants.
LINUX_F_ADD_SEALS = 1033
LINUX_F_GET_SEALS = 1034
LINUX_REQUIRED_LAUNCH_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008

BUILD_KEYS = {
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
BUDGET_KEYS = {
    "managed_ram_bytes",
    "managed_cuda_bytes",
    "game_concurrency",
    "player_worker_threads",
    "max_batch_size",
    "storage_bytes",
    "source_workspace_bytes",
}
DEADLINE_KEYS = ("admission_bits", "execution_bits", "term_bits", "kill_bits", "finish_bits")
PLAYER_RUNTIME_MODE = "rust-player-runtime-v1"


def player_runtime(value: Any, *, game_concurrency: int) -> Json:
    """Normalize the explicit declaration; never grants runtime admission."""
    raw = closed(
        value,
        {
            "mode",
            "scheduler",
            "game_concurrency",
            "player_worker_threads",
            "max_batch_size",
            "formation_wait_ms",
        },
    )
    if raw["mode"] != PLAYER_RUNTIME_MODE or raw["scheduler"] not in ("tokio", "fixed"):
        raise ContractError("explicit Rust player mode and tokio|fixed scheduler required")
    integer(game_concurrency, 1)
    for key in ("game_concurrency", "player_worker_threads", "max_batch_size"):
        integer(raw[key], 1)
    if raw["game_concurrency"] != game_concurrency:
        raise ContractError("Rust player game_concurrency differs from declared concurrency")
    wait = raw["formation_wait_ms"]
    try:
        if type(wait) not in (int, float) or not math.isfinite(wait) or wait < 0:
            raise ContractError("finite nonnegative Rust player formation wait required")
        wait = float(wait)
    except OverflowError as error:
        raise ContractError("finite nonnegative Rust player formation wait required") from error
    return dict(raw, formation_wait_ms=wait)


def require_legacy_player_entry(config: Any) -> None:
    """Keep new selection outside every historical numerical execution path."""
    spec = getattr(config, "hpo_spec", {})
    if (
        hasattr(getattr(config, "run", None), "player_runtime")
        or spec.get("schema_version") == 6
        or spec.get("runtime", {}).get("mode") == PLAYER_RUNTIME_MODE
    ):
        raise ContractError("Rust player selection requires an authenticated Rust-supervised entry")


def runtime_binding(runtime: Json, producer: Json) -> Json:
    """Stable metadata only; actual host/control owners grant live authority."""
    return clone(validate_runtime_binding({"schema_version": 1,
        "kind": "rust-player-runtime-binding-v1", "runtime": runtime,
        "producer": producer, "sampling_runtime_sha256": producer.get("sampling_runtime_sha256")}))


def validate_runtime_binding(value: Any) -> Json:
    raw = record(value, "rust-player-runtime-binding-v1",
        {"runtime", "producer", "sampling_runtime_sha256"})
    if type(raw["runtime"]) is not dict:
        raise ContractError("complete materialized runtime declaration required")
    normalized = player_runtime(raw["runtime"],
        game_concurrency=cast(int, raw["runtime"].get("game_concurrency")))
    if normalized != raw["runtime"]:
        raise ContractError("normalized runtime declaration required")
    producer = validate_producer(raw["producer"])
    hex_identity(raw["sampling_runtime_sha256"])
    if raw["sampling_runtime_sha256"] != producer["sampling_runtime_sha256"]:
        raise ContractError("runtime binding sampling identity differs from its producer")
    return raw


def text(value: Any) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise ContractError("nonempty runtime identity required")
    return value


def hex_identity(value: Any, length: int = 64) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ContractError("invalid runtime digest/identity")
    return value


def deadline_bits(value: Any) -> str:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ContractError("finite positive original deadline required")
    return struct.pack(">d", value).hex()


def deadline_seconds(value: Any) -> float:
    hex_identity(value, 16)
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    deadline_bits(result)
    return result


def validate_deadlines(value: Any, *, session_finish: str | None = None) -> Json:
    value = closed(value, set(DEADLINE_KEYS))
    times = [deadline_seconds(value[key]) for key in DEADLINE_KEYS]
    if times != sorted(times) or times[-2] >= times[-1]:
        raise ContractError("unordered original operation milestones")
    if session_finish is not None and times[-1] > deadline_seconds(session_finish):
        raise ContractError("operation extends original session")
    return value


def validate_build(value: Any) -> Json:
    value = closed(value, BUILD_KEYS)
    for key in BUILD_KEYS - {"rustflags"}:
        text(value[key])
    if type(value["rustflags"]) is not str:
        raise ContractError("actual runtime flags required")
    for key in ("cargo_lock_sha256", "native_source_sha256"):
        hex_identity(value[key])
    if value["abi"] != "cp314-gil" or value["allocator"] not in ("system", "mimalloc"):
        raise ContractError("unsupported native runtime ABI/allocator")
    return value


def validate_context(value: Any) -> Json:
    value = closed(value, {"campaign_id", "operation_id", "kind", "run_id", "evaluation_id"})
    for key in ("campaign_id", "operation_id", "run_id"):
        text(value[key])
    if value["kind"] != "training" or value["evaluation_id"] is not None:
        raise ContractError("supervised training context required")
    return value


def validate_producer(value: Any) -> Json:
    value = record(
        value,
        "rust-player-producer-v1",
        {
            "python_source_digest",
            "python_runtime_digest",
            "extension_sha256",
            "build",
            "supervisor_sha256",
            "sampling_runtime_sha256",
        },
    )
    validate_build(value["build"])
    for key in set(value) - {"schema_version", "kind", "build"}:
        hex_identity(value[key])
    return value


def record(value: Any, kind: str, keys: set[str], version: int = 1) -> Json:
    value = closed(value, keys | {"schema_version", "kind"})
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != version
        or value["kind"] != kind
    ):
        raise ContractError("unsupported rust-player record kind/version")
    return value


def validate_lock(value: Any) -> Json:
    value = closed(value, {"lock_kind", "fd", "device", "inode"})
    if value["lock_kind"] not in ("parent", "campaign"):
        raise ContractError("unknown supervisor lock kind")
    for key in ("fd", "device", "inode"):
        integer(value[key], 0 if key == "device" else 1)
    return value


def validate_operation(value: Any, *, session_finish: str | None = None) -> Json:
    version2 = type(value) is dict and type(value.get("schema_version")) is int and value["schema_version"] == 2
    value = record(
        value,
        "rust-player-operation-v2" if version2 else "rust-player-operation-v1",
        {
            "operation_id",
            "run_id",
            "mode",
            "operation_type",
            "config_path",
            "config_sha256",
            "runtime_binding_sha256",
            "deadlines",
            "parent_directory",
            "expected_context",
        } | ({"execution_request"} if version2 else set()),
        2 if version2 else 1,
    )
    for key in ("operation_id", "config_path"):
        text(value[key])
    for key in ("config_sha256", "runtime_binding_sha256"):
        hex_identity(value[key])
    if value["mode"] not in ("ordinary", "hpo") or value["operation_type"] not in (
        "ordinary-train",
        "hpo-train",
        "cold-finalize",
        "campaign-continue",
    ):
        raise ContractError("unknown fixed host operation")
    if (
        value["operation_type"] == "ordinary-train"
        and value["mode"] != "ordinary"
        or (value["operation_type"] == "hpo-train" and value["mode"] != "hpo")
    ):
        raise ContractError("operation/mode mismatch")
    if version2:
        execution = value["execution_request"]
        if value["operation_type"] in ("ordinary-train", "hpo-train"):
            closed(execution, {"mode", "pause_after_update"})
            if execution["mode"] not in ("fresh", "resume"):
                raise ContractError("explicit fresh/resume execution request required")
            if execution["pause_after_update"] is not None:
                integer(execution["pause_after_update"], 1)
        elif execution is not None:
            raise ContractError("non-training operation has no training execution request")
    if value["run_id"] is None or value["parent_directory"] is None:
        if value["operation_type"] != "campaign-continue" or any(
            value[key] is not None
            for key in (
                "run_id",
                "parent_directory",
                "expected_context",
            )
        ):
            raise ContractError("only pre-parent campaign continuation may omit parent")
    else:
        text(value["run_id"])
        text(value["parent_directory"])
        if value["mode"] == "hpo":
            context = validate_context(value["expected_context"])
            if (
                context["run_id"] != value["run_id"]
                or context["operation_id"] != value["operation_id"]
            ):
                raise ContractError("operation context mismatch")
        elif value["expected_context"] is not None:
            raise ContractError("ordinary operation has no HPO context")
    validate_deadlines(value["deadlines"], session_finish=session_finish)
    return value


def validate_launch(value: Any) -> Json:
    value = record(
        value,
        "rust-player-launch-v1",
        {
            "supervision_id",
            "python_executable",
            "operation_path",
            "operation_sha256",
            "record_directory",
            "mode",
            "session_deadlines",
            "clock_binding",
            "binding",
            "locks",
        },
    )
    hex_identity(value["supervision_id"], 32)
    for key in ("python_executable", "operation_path", "record_directory"):
        text(value[key])
        if not value[key].startswith("/"):
            raise ContractError("absolute fixed host path required")
    hex_identity(value["operation_sha256"])
    if value["mode"] not in ("ordinary", "hpo"):
        raise ContractError("unknown supervised mode")
    deadline_seconds(closed(value["session_deadlines"], {"finish_bits"})["finish_bits"])
    clock = closed(
        value["clock_binding"],
        {
            "boot_id",
            "monotonic_before_bits",
            "monotonic_after_bits",
            "unix_anchor_bits",
        },
    )
    text(clock["boot_id"])
    if deadline_seconds(clock["monotonic_before_bits"]) > deadline_seconds(
        clock["monotonic_after_bits"]
    ):
        raise ContractError("invalid original checked clock anchor")
    deadline_seconds(clock["unix_anchor_bits"])
    binding = closed(value["binding"], {"python_abi", "extension_sha256", "build"})
    if binding["python_abi"] != "cpython-314":
        raise ContractError("unsupported host Python ABI")
    hex_identity(binding["extension_sha256"])
    validate_build(binding["build"])
    if type(value["locks"]) is not list:
        raise ContractError("original lock mapping required")
    kinds = [validate_lock(lock)["lock_kind"] for lock in value["locks"]]
    if len(kinds) != len(set(kinds)):
        raise ContractError("duplicate original custody")
    return value


LOCAL_KEYS = {
    "supervision_id",
    "invocation_id",
    "nonce",
    "supervisor",
    "host",
    "context",
    "budget",
    "operation_id",
    "run_id",
    "config_sha256",
    "runtime_binding_sha256",
    "deadlines",
    "producer",
    "global_ticket",
}


def validate_local_admission(value: Any) -> Json:
    value = record(value, "rust-player-admission-v1", LOCAL_KEYS)
    for key in ("supervision_id", "invocation_id", "nonce"):
        hex_identity(value[key], 32)
    for key in ("operation_id", "run_id"):
        text(value[key])
    for key in ("config_sha256", "runtime_binding_sha256"):
        hex_identity(value[key])
    for name in ("supervisor", "host"):
        identity = closed(
            value[name], {"pid", "start_ticks"} | ({"pgid"} if name == "host" else set())
        )
        for entry in identity.values():
            integer(entry, 1)
        if name == "host" and identity["pid"] != identity["pgid"]:
            raise ContractError("host must own its isolated original group")
    budget = closed(value["budget"], BUDGET_KEYS)
    for key in BUDGET_KEYS:
        integer(budget[key], 0 if key in ("managed_cuda_bytes", "source_workspace_bytes") else 1)
    validate_deadlines(value["deadlines"])
    validate_producer(value["producer"])
    if value["global_ticket"] is None:
        if value["context"] is not None:
            raise ContractError("ordinary admission has no global context")
    else:
        hex_identity(value["global_ticket"], 32)
        context = validate_context(value["context"])
        if context["run_id"] != value["run_id"] or context["operation_id"] != value["operation_id"]:
            raise ContractError("global admission context mismatch")
    return value


def validate_prepared(value: Any, admission: Json) -> Json:
    value = record(
        value,
        "rust-player-prepared-v1",
        {
            "supervision_id",
            "invocation_id",
            "admission_sha256",
            "operation_id",
            "run_id",
            "config_sha256",
            "runtime_binding_sha256",
            "state",
            "committed_update",
            "generation",
            "state_sha256",
            "generation_sha256",
            "artifacts_sha256",
            "source_manifest_sha256",
            "close_ack_path",
            "close_ack_sha256",
        },
    )
    validate_local_admission(admission)
    for key in (
        "supervision_id",
        "invocation_id",
        "operation_id",
        "run_id",
        "config_sha256",
        "runtime_binding_sha256",
    ):
        if value[key] != admission[key]:
            raise ContractError("prepared admission identity mismatch")
    if value["admission_sha256"] != digest(admission) or value["state"] not in (
        "complete",
        "paused",
        "incomplete",
    ):
        raise ContractError("unbound prepared result")
    for key in ("committed_update", "generation"):
        integer(value[key])
    for key in (
        "state_sha256",
        "generation_sha256",
        "artifacts_sha256",
        "source_manifest_sha256",
        "close_ack_sha256",
    ):
        hex_identity(value[key])
    text(value["close_ack_path"])
    return value


def validate_wait(value: Any) -> Json:
    value = closed(value, {"exit_code", "signal", "core_dumped"})
    if (value["exit_code"] is None) == (value["signal"] is None) or type(
        value["core_dumped"]
    ) is not bool:
        raise ContractError("actual original wait status required")
    if value["exit_code"] is not None:
        integer(value["exit_code"], 0, 255)
        if value["core_dumped"]:
            raise ContractError("exit status cannot be a core dump")
    else:
        integer(value["signal"], 1, 64)
    return value


def validate_exit(value: Any, admission: Json) -> Json:
    value = record(
        value,
        "rust-player-exit-v1",
        {
            "supervision_id",
            "invocation_id",
            "nonce",
            "admission_sha256",
            "prepared_sha256",
            "host_original",
            "wait",
            "descendants",
            "signals",
            "descendants_reaped",
            "cooperative",
            "finished_unix_ns",
        },
    )
    validate_local_admission(admission)
    if any(
        value[key] != admission[key] for key in ("supervision_id", "invocation_id", "nonce")
    ) or (
        value["admission_sha256"] != digest(admission)
        or value["host_original"] != admission["host"]
    ):
        raise ContractError("original host exit identity mismatch")
    validate_wait(value["wait"])
    if value["prepared_sha256"] is not None:
        hex_identity(value["prepared_sha256"])
    if type(value["descendants"]) is not list or type(value["signals"]) is not list:
        raise ContractError("actual descendant/signal observations required")
    seen = {admission["host"]["pid"]}
    for child in value["descendants"]:
        child = closed(child, {"pid", "start_ticks", "pgid", "wait"})
        for key in ("pid", "start_ticks", "pgid"):
            integer(child[key], 1)
        if child["pid"] in seen or child["pgid"] != admission["host"]["pgid"]:
            raise ContractError("duplicate/outgroup descendant cannot confirm physical closure")
        seen.add(child["pid"])
        validate_wait(child["wait"])
    for signal in value["signals"]:
        signal = closed(signal, {"pgid", "signal", "delivered"})
        if (
            signal["pgid"] != admission["host"]["pgid"]
            or signal["signal"] not in (15, 9)
            or type(signal["delivered"]) is not bool
        ):
            raise ContractError("invalid owned group signal observation")
    if value["descendants_reaped"] is not True or type(value["cooperative"]) is not bool:
        raise ContractError("unconfirmed descendant retirement")
    integer(value["finished_unix_ns"], 1)
    if value["cooperative"] and (
        value["wait"]["exit_code"] != 0 or value["signals"] or value["prepared_sha256"] is None
    ):
        raise ContractError("forced/unprepared exit cannot be cooperative")
    return value


def validate_global_admission(value: Any, ticket: str) -> Json:
    value = record(
        value,
        "rust-player-gpu-admission-v1",
        {
            "ticket",
            "context",
            "boot_id",
            "owner_pid",
            "admitted_unix",
            "host_admission",
        },
        2,
    )
    hex_identity(ticket, 32)
    if value["ticket"] != ticket:
        raise ContractError("global ticket mismatch")
    validate_context(value["context"])
    text(value["boot_id"])
    integer(value["owner_pid"], 1)
    if (
        type(value["admitted_unix"]) not in (int, float)
        or not math.isfinite(value["admitted_unix"])
        or value["admitted_unix"] < 0
    ):
        raise ContractError("invalid global observation timestamp")
    host = validate_local_admission(value["host_admission"])
    if (
        host["global_ticket"] != ticket
        or host["context"] != value["context"]
        or host["supervisor"]["pid"] != value["owner_pid"]
    ):
        raise ContractError("global/local original capability mismatch")
    return value


def validate_global_termination(value: Any, admission: Json) -> Json:
    value = record(
        value,
        "rust-player-gpu-termination-v1",
        {
            "ticket",
            "admission_sha256",
            "context_sha256",
            "boot_id",
            "confirmed_unix",
            "exit_sha256",
            "exit_record",
        },
        3,
    )
    validate_global_admission(admission, admission["ticket"])
    if (
        value["ticket"] != admission["ticket"]
        or value["admission_sha256"] != digest(admission)
        or value["context_sha256"] != digest(admission["context"])
        or value["boot_id"] != admission["boot_id"]
    ):
        raise ContractError("global physical receipt binding mismatch")
    if (
        type(value["confirmed_unix"]) not in (int, float)
        or not math.isfinite(value["confirmed_unix"])
        or value["confirmed_unix"] < 0
    ):
        raise ContractError("invalid physical receipt observation timestamp")
    exit_record = validate_exit(value["exit_record"], admission["host_admission"])
    if value["exit_sha256"] != digest(exit_record):
        raise ContractError("physical receipt original exit hash mismatch")
    return value
