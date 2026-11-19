//! Closed control observations and owned immutable publication; no data eligibility.
use crate::{Result, build_info, invalid};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Component, Path},
    time::{SystemTime, UNIX_EPOCH},
};

pub(crate) fn object<'a>(v: &'a Value, keys: &[&str]) -> Result<&'a Value> {
    let map = v
        .as_object()
        .ok_or_else(|| invalid("control object required"))?;
    if map.len() != keys.len() || keys.iter().any(|k| !map.contains_key(*k)) {
        return Err(invalid("unknown/missing control fields"));
    }
    Ok(v)
}
pub(crate) fn text(v: &Value) -> Result<&str> {
    v.as_str()
        .filter(|s| !s.is_empty() && !s.contains('\0'))
        .ok_or_else(|| invalid("control text required"))
}
pub(crate) fn uint(v: &Value) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| invalid("control unsigned integer required"))
}
pub(crate) fn positive(v: &Value) -> Result<u64> {
    let n = uint(v)?;
    if n == 0 {
        return Err(invalid("positive control integer required"));
    }
    Ok(n)
}
pub(crate) fn hex(v: &Value, len: usize) -> Result<&str> {
    let s = text(v)?;
    if s.len() != len
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("invalid control digest"));
    }
    Ok(s)
}
pub(crate) fn record<'a>(
    v: &'a Value,
    kind: &str,
    version: u64,
    keys: &[&str],
) -> Result<&'a Value> {
    let mut all = keys.to_vec();
    all.extend(["schema_version", "kind"]);
    object(v, &all)?;
    if v["schema_version"].as_u64() != Some(version) || v["kind"] != kind {
        return Err(invalid("unsupported control record"));
    }
    Ok(v)
}
pub(crate) fn digest(v: &Value) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(v).expect("JSON value"))
    )
}
pub(crate) fn bytes_sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn random_id() -> Result<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|n| format!("{n:02x}")).collect())
}
pub(crate) fn boot_id() -> Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_owned())
}
pub(crate) fn unix_ns() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("invalid observed wall clock"))?
        .as_nanos()
        .try_into()
        .map_err(|_| invalid("unsupported observed wall time"))
}
pub(crate) fn observation_time(v: &Value) -> Result<()> {
    if !v.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0) {
        return Err(invalid("invalid observed timestamp"));
    }
    Ok(())
}
pub(crate) fn bits(v: &Value) -> Result<f64> {
    let n = f64::from_bits(
        u64::from_str_radix(hex(v, 16)?, 16).map_err(|_| invalid("bad deadline bits"))?,
    );
    if !n.is_finite() || n <= 0.0 {
        return Err(invalid("invalid original deadline"));
    }
    Ok(n)
}
pub(crate) const DEADLINES: [&str; 5] = [
    "admission_bits",
    "execution_bits",
    "term_bits",
    "kill_bits",
    "finish_bits",
];
pub(crate) fn deadlines(v: &Value, session: f64) -> Result<[f64; 5]> {
    object(v, &DEADLINES)?;
    let mut times = [0.0; 5];
    for (i, k) in DEADLINES.iter().enumerate() {
        times[i] = bits(&v[k])?;
    }
    if times.windows(2).any(|w| w[0] > w[1]) || times[3] >= times[4] || times[4] > session {
        return Err(invalid("original operation/session milestones mismatch"));
    }
    Ok(times)
}
pub(crate) const BUILD: [&str; 9] = [
    "version",
    "rustc",
    "target",
    "profile",
    "rustflags",
    "cargo_lock_sha256",
    "native_source_sha256",
    "abi",
    "allocator",
];
pub(crate) fn build(v: &Value) -> Result<()> {
    object(v, &BUILD)?;
    for k in BUILD {
        if k == "rustflags" {
            v[k].as_str()
                .ok_or_else(|| invalid("actual flags required"))?;
        } else {
            text(&v[k])?;
        }
    }
    for k in ["cargo_lock_sha256", "native_source_sha256"] {
        hex(&v[k], 64)?;
    }
    if v["abi"] != "cp314-gil" || !["system", "mimalloc"].iter().any(|s| v["allocator"] == *s) {
        return Err(invalid("unsupported build ABI/allocator"));
    }
    Ok(())
}
pub(crate) fn context(v: &Value, training_only: bool) -> Result<()> {
    object(
        v,
        &[
            "campaign_id",
            "operation_id",
            "kind",
            "run_id",
            "evaluation_id",
        ],
    )?;
    for k in ["campaign_id", "operation_id", "run_id"] {
        text(&v[k])?;
    }
    match text(&v["kind"])? {
        "training" => {
            if !v["evaluation_id"].is_null() {
                return Err(invalid("training evaluation context"));
            }
        }
        "evaluation" if !training_only => {
            text(&v["evaluation_id"])?;
        }
        _ => return Err(invalid("unknown GPU job kind")),
    }
    Ok(())
}
pub(crate) fn producer(v: &Value) -> Result<()> {
    record(
        v,
        "rust-player-producer-v1",
        1,
        &[
            "python_source_digest",
            "python_runtime_digest",
            "extension_sha256",
            "build",
            "supervisor_sha256",
            "sampling_runtime_sha256",
        ],
    )?;
    build(&v["build"])?;
    for k in [
        "python_source_digest",
        "python_runtime_digest",
        "extension_sha256",
        "supervisor_sha256",
        "sampling_runtime_sha256",
    ] {
        hex(&v[k], 64)?;
    }
    Ok(())
}
pub(crate) fn host_identity(v: &Value) -> Result<()> {
    object(v, &["pid", "start_ticks", "pgid"])?;
    for k in ["pid", "start_ticks", "pgid"] {
        positive(&v[k])?;
    }
    if v["pid"] != v["pgid"] {
        return Err(invalid("host does not own group"));
    }
    Ok(())
}
pub(crate) const BUDGET: [&str; 7] = [
    "managed_ram_bytes",
    "managed_cuda_bytes",
    "game_concurrency",
    "player_worker_threads",
    "max_batch_size",
    "storage_bytes",
    "source_workspace_bytes",
];
pub(crate) fn local(v: &Value) -> Result<()> {
    record(
        v,
        "rust-player-admission-v1",
        1,
        &[
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
        ],
    )?;
    for k in ["supervision_id", "invocation_id", "nonce"] {
        hex(&v[k], 32)?;
    }
    object(&v["supervisor"], &["pid", "start_ticks"])?;
    positive(&v["supervisor"]["pid"])?;
    positive(&v["supervisor"]["start_ticks"])?;
    host_identity(&v["host"])?;
    for k in ["operation_id", "run_id"] {
        text(&v[k])?;
    }
    for k in ["config_sha256", "runtime_binding_sha256"] {
        hex(&v[k], 64)?;
    }
    object(&v["budget"], &BUDGET)?;
    for k in BUDGET {
        if ["managed_cuda_bytes", "source_workspace_bytes"].contains(&k) {
            uint(&v["budget"][k])?;
        } else {
            positive(&v["budget"][k])?;
        }
    }
    deadlines(&v["deadlines"], f64::MAX)?;
    producer(&v["producer"])?;
    if v["global_ticket"].is_null() {
        if !v["context"].is_null() {
            return Err(invalid("ordinary HPO context forbidden"));
        }
    } else {
        hex(&v["global_ticket"], 32)?;
        context(&v["context"], true)?;
        if v["context"]["run_id"] != v["run_id"]
            || v["context"]["operation_id"] != v["operation_id"]
        {
            return Err(invalid("local context identity mismatch"));
        }
    }
    Ok(())
}
pub(crate) fn operation(v: &Value, mode: &str, session: f64) -> Result<[f64; 5]> {
    let version2 = v["schema_version"].as_u64() == Some(2);
    let mut keys = vec![
        "operation_id", "run_id", "mode", "operation_type", "config_path",
        "config_sha256", "runtime_binding_sha256", "deadlines",
        "parent_directory", "expected_context",
    ];
    if version2 { keys.push("execution_request"); }
    record(
        v,
        if version2 { "rust-player-operation-v2" } else { "rust-player-operation-v1" },
        if version2 { 2 } else { 1 },
        &keys,
    )?;
    if v["mode"] != mode {
        return Err(invalid("fixed operation mode mismatch"));
    }
    for k in ["operation_id", "config_path"] {
        text(&v[k])?;
    }
    for k in ["config_sha256", "runtime_binding_sha256"] {
        hex(&v[k], 64)?;
    }
    let kind = text(&v["operation_type"])?;
    if ![
        "ordinary-train",
        "hpo-train",
        "cold-finalize",
        "campaign-continue",
    ]
    .contains(&kind)
        || kind == "ordinary-train" && mode != "ordinary"
        || kind == "hpo-train" && mode != "hpo"
    {
        return Err(invalid("unknown fixed host operation"));
    }
    if version2 {
        let execution = &v["execution_request"];
        if ["ordinary-train", "hpo-train"].contains(&kind) {
            object(execution, &["mode", "pause_after_update"])?;
            if !["fresh", "resume"].contains(&text(&execution["mode"])?) {
                return Err(invalid("explicit fresh/resume execution request required"));
            }
            if !execution["pause_after_update"].is_null() {
                positive(&execution["pause_after_update"])?;
            }
        } else if !execution.is_null() {
            return Err(invalid("non-training execution request must be null"));
        }
    }
    if v["run_id"].is_null() || v["parent_directory"].is_null() {
        if kind != "campaign-continue"
            || !["run_id", "parent_directory", "expected_context"]
                .iter()
                .all(|k| v[k].is_null())
        {
            return Err(invalid("missing materialized parent"));
        }
    } else {
        text(&v["run_id"])?;
        text(&v["parent_directory"])?;
        if mode == "hpo" {
            context(&v["expected_context"], true)?;
            if v["expected_context"]["run_id"] != v["run_id"]
                || v["expected_context"]["operation_id"] != v["operation_id"]
            {
                return Err(invalid("operation context mismatch"));
            }
        } else if !v["expected_context"].is_null() {
            return Err(invalid("ordinary context must be null"));
        }
    }
    deadlines(&v["deadlines"], session)
}

#[cfg(test)]
mod operation_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn operation2_request_is_explicit_and_operation1_remains_closed() {
        let bits = |value: f64| format!("{:016x}", value.to_bits());
        let mut value = json!({"schema_version":2,"kind":"rust-player-operation-v2",
            "operation_id":"op","run_id":"run","mode":"ordinary","operation_type":"ordinary-train",
            "config_path":"/tmp/config","config_sha256":"a".repeat(64),
            "runtime_binding_sha256":"b".repeat(64),"parent_directory":"/tmp/parent",
            "expected_context":null,"deadlines":{"admission_bits":bits(100.),
            "execution_bits":bits(110.),"term_bits":bits(120.),"kill_bits":bits(130.),
            "finish_bits":bits(140.)},"execution_request":{"mode":"fresh","pause_after_update":1}});
        assert!(operation(&value, "ordinary", 200.).is_ok());
        for execution in [json!(null), json!({}), json!({"mode":"fresh","pause_after_update":true}),
            json!({"mode":"resume","pause_after_update":0}),
            json!({"mode":"infer","pause_after_update":null})] {
            let mut bad = value.clone();
            bad["execution_request"] = execution;
            assert!(operation(&bad, "ordinary", 200.).is_err());
        }
        value["schema_version"] = json!(1);
        value["kind"] = json!("rust-player-operation-v1");
        assert!(operation(&value, "ordinary", 200.).is_err());
        value.as_object_mut().unwrap().remove("execution_request");
        assert!(operation(&value, "ordinary", 200.).is_ok());
    }
}
pub(crate) fn launch(v: &Value) -> Result<f64> {
    record(
        v,
        "rust-player-launch-v1",
        1,
        &[
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
        ],
    )?;
    hex(&v["supervision_id"], 32)?;
    for k in ["python_executable", "operation_path", "record_directory"] {
        let path = Path::new(text(&v[k])?);
        if !path.is_absolute() {
            return Err(invalid("absolute bootstrap path required"));
        }
    }
    hex(&v["operation_sha256"], 64)?;
    if !["ordinary", "hpo"].iter().any(|m| v["mode"] == *m) {
        return Err(invalid("unknown host mode"));
    }
    object(&v["session_deadlines"], &["finish_bits"])?;
    let finish = bits(&v["session_deadlines"]["finish_bits"])?;
    crate::deadline_from_bits(text(&v["session_deadlines"]["finish_bits"])?)?;
    object(
        &v["clock_binding"],
        &[
            "boot_id",
            "monotonic_before_bits",
            "monotonic_after_bits",
            "unix_anchor_bits",
        ],
    )?;
    if v["clock_binding"]["boot_id"] != boot_id()?
        || bits(&v["clock_binding"]["monotonic_before_bits"])?
            > bits(&v["clock_binding"]["monotonic_after_bits"])?
        || bits(&v["clock_binding"]["monotonic_after_bits"])? > crate::monotonic()?
    {
        return Err(invalid("original boot/clock binding mismatch"));
    }
    bits(&v["clock_binding"]["unix_anchor_bits"])?;
    object(&v["binding"], &["python_abi", "extension_sha256", "build"])?;
    hex(&v["binding"]["extension_sha256"], 64)?;
    build(&v["binding"]["build"])?;
    if v["binding"]["python_abi"] != "cpython-314" || v["binding"]["build"] != build_info() {
        return Err(invalid("host/supervisor build mismatch"));
    }
    let mut kinds = std::collections::BTreeSet::new();
    for lock in v["locks"]
        .as_array()
        .ok_or_else(|| invalid("lock mapping array required"))?
    {
        object(lock, &["lock_kind", "fd", "device", "inode"])?;
        let kind = text(&lock["lock_kind"])?;
        if !["parent", "campaign"].contains(&kind) || !kinds.insert(kind) {
            return Err(invalid("duplicate/unknown inherited lock"));
        }
        positive(&lock["fd"])?;
        uint(&lock["device"])?;
        positive(&lock["inode"])?;
    }
    Ok(finish)
}
pub(crate) fn prepared(v: &Value, a: &Value) -> Result<()> {
    record(
        v,
        "rust-player-prepared-v1",
        1,
        &[
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
        ],
    )?;
    local(a)?;
    for k in [
        "supervision_id",
        "invocation_id",
        "operation_id",
        "run_id",
        "config_sha256",
        "runtime_binding_sha256",
    ] {
        if v[k] != a[k] {
            return Err(invalid("prepared original admission mismatch"));
        }
    }
    if v["admission_sha256"] != digest(a)
        || !["complete", "paused", "incomplete"]
            .iter()
            .any(|s| v["state"] == *s)
    {
        return Err(invalid("prepared admission/state mismatch"));
    }
    for k in ["committed_update", "generation"] {
        uint(&v[k])?;
    }
    for k in [
        "state_sha256",
        "generation_sha256",
        "artifacts_sha256",
        "source_manifest_sha256",
        "close_ack_sha256",
    ] {
        hex(&v[k], 64)?;
    }
    text(&v["close_ack_path"])?;
    Ok(())
}
pub(crate) fn close_ack(v: &Value, p: &Value) -> Result<()> {
    record(
        v,
        "rust-player-host-close-ack-v1",
        1,
        &[
            "supervision_id",
            "invocation_id",
            "operation_id",
            "run_id",
            "admission_sha256",
            "runtime_binding_sha256",
            "actors_joined",
            "service_joined",
            "writer_joined",
            "buffers_released",
            "gpu_fenced",
            "native_retirement_sha256",
            "writer_retirement_sha256",
            "service_retirement_sha256",
            "cache_retirement_sha256",
            "servers_path",
            "servers_sha256",
            "server_count",
        ],
    )?;
    for k in [
        "supervision_id",
        "invocation_id",
        "operation_id",
        "run_id",
        "admission_sha256",
        "runtime_binding_sha256",
    ] {
        if v[k] != p[k] {
            return Err(invalid("cooperative close identity mismatch"));
        }
    }
    for k in [
        "actors_joined",
        "service_joined",
        "writer_joined",
        "buffers_released",
        "gpu_fenced",
    ] {
        if v[k] != true {
            return Err(invalid("cooperative close remains unconfirmed"));
        }
    }
    for k in [
        "native_retirement_sha256",
        "writer_retirement_sha256",
        "service_retirement_sha256",
        "cache_retirement_sha256",
        "servers_sha256",
    ] {
        hex(&v[k], 64)?;
    }
    text(&v["servers_path"])?;
    uint(&v["server_count"])?;
    Ok(())
}
pub(crate) fn wait(v: &Value) -> Result<()> {
    object(v, &["exit_code", "signal", "core_dumped"])?;
    let core = v["core_dumped"]
        .as_bool()
        .ok_or_else(|| invalid("actual core status required"))?;
    if v["exit_code"].is_null() == v["signal"].is_null() {
        return Err(invalid("actual wait status required"));
    }
    if !v["exit_code"].is_null() {
        if uint(&v["exit_code"])? > 255 || core {
            return Err(invalid("invalid exit status"));
        }
    } else if positive(&v["signal"])? > 64 {
        return Err(invalid("invalid signal status"));
    }
    Ok(())
}
pub(crate) fn exit(v: &Value, a: &Value) -> Result<()> {
    record(
        v,
        "rust-player-exit-v1",
        1,
        &[
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
        ],
    )?;
    local(a)?;
    for k in ["supervision_id", "invocation_id", "nonce"] {
        if v[k] != a[k] {
            return Err(invalid("original exit host identity mismatch"));
        }
    }
    if v["admission_sha256"] != digest(a)
        || v["host_original"] != a["host"]
        || v["descendants_reaped"] != true
    {
        return Err(invalid("physical closure unconfirmed"));
    }
    wait(&v["wait"])?;
    if !v["prepared_sha256"].is_null() {
        hex(&v["prepared_sha256"], 64)?;
    }
    let mut seen = std::collections::BTreeSet::from([positive(&a["host"]["pid"])?]);
    for d in v["descendants"]
        .as_array()
        .ok_or_else(|| invalid("descendants array required"))?
    {
        object(d, &["pid", "start_ticks", "pgid", "wait"])?;
        let pid = positive(&d["pid"])?;
        positive(&d["start_ticks"])?;
        if !seen.insert(pid) || d["pgid"] != a["host"]["pgid"] {
            return Err(invalid("duplicate/outgroup descendants"));
        }
        wait(&d["wait"])?;
    }
    let signals = v["signals"]
        .as_array()
        .ok_or_else(|| invalid("signals array required"))?;
    for s in signals {
        object(s, &["pgid", "signal", "delivered"])?;
        if s["pgid"] != a["host"]["pgid"]
            || ![9, 15].iter().any(|n| s["signal"] == *n)
            || !s["delivered"].is_boolean()
        {
            return Err(invalid("invalid group signal evidence"));
        }
    }
    let cooperative = v["cooperative"]
        .as_bool()
        .ok_or_else(|| invalid("cooperative exit boolean required"))?;
    if cooperative
        && (v["wait"]["exit_code"] != 0 || !signals.is_empty() || v["prepared_sha256"].is_null())
    {
        return Err(invalid("forced exit cannot qualify cooperative completion"));
    }
    positive(&v["finished_unix_ns"])?;
    Ok(())
}
pub(crate) fn global_admission(v: &Value, ticket: &str) -> Result<()> {
    record(
        v,
        "rust-player-gpu-admission-v1",
        2,
        &[
            "ticket",
            "context",
            "boot_id",
            "owner_pid",
            "admitted_unix",
            "host_admission",
        ],
    )?;
    hex(&v["ticket"], 32)?;
    if v["ticket"] != ticket {
        return Err(invalid("GPU ticket mismatch"));
    }
    context(&v["context"], true)?;
    text(&v["boot_id"])?;
    positive(&v["owner_pid"])?;
    observation_time(&v["admitted_unix"])?;
    local(&v["host_admission"])?;
    let a = &v["host_admission"];
    if a["global_ticket"] != ticket
        || a["context"] != v["context"]
        || a["supervisor"]["pid"] != v["owner_pid"]
    {
        return Err(invalid("global/local admission mismatch"));
    }
    Ok(())
}
pub(crate) fn global_termination(v: &Value, a: &Value) -> Result<()> {
    record(
        v,
        "rust-player-gpu-termination-v1",
        3,
        &[
            "ticket",
            "admission_sha256",
            "context_sha256",
            "boot_id",
            "confirmed_unix",
            "exit_sha256",
            "exit_record",
        ],
    )?;
    global_admission(a, text(&a["ticket"])?)?;
    if v["ticket"] != a["ticket"]
        || v["admission_sha256"] != digest(a)
        || v["context_sha256"] != digest(&a["context"])
        || v["boot_id"] != a["boot_id"]
        || v["exit_sha256"] != digest(&v["exit_record"])
    {
        return Err(invalid("GPU physical receipt mismatch"));
    }
    observation_time(&v["confirmed_unix"])?;
    exit(&v["exit_record"], &a["host_admission"])
}

pub(crate) fn private_directory(path: &Path) -> Result<()> {
    let info = fs::symlink_metadata(path)?;
    if !info.is_dir() || info.uid() != unsafe { libc::getuid() } || info.mode() & 0o077 != 0 {
        return Err(invalid("owned private nonsymlink directory required"));
    }
    Ok(())
}
pub(crate) fn sync_dir(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)?
        .sync_all()
}
pub(crate) fn ensure_private(path: &Path) -> Result<()> {
    let mut missing = Vec::new();
    let mut at = path;
    while !at.exists() {
        missing.push(at.to_path_buf());
        at = at
            .parent()
            .ok_or_else(|| invalid("invalid namespace root"))?;
    }
    for directory in missing.iter().rev() {
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        match builder.create(directory) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
        sync_dir(directory.parent().unwrap())?;
    }
    private_directory(path)?;
    for ancestor in path.ancestors() {
        let before = fs::symlink_metadata(ancestor)?;
        if !before.is_dir() {
            return Err(invalid("symlink in namespace ancestry"));
        }
        sync_dir(ancestor)?;
        let after = fs::symlink_metadata(ancestor)?;
        if (before.dev(), before.ino()) != (after.dev(), after.ino()) {
            return Err(invalid("namespace ancestry changed"));
        }
    }
    Ok(())
}
pub(crate) fn read_owned(path: &Path, root: Option<&Path>) -> Result<(Value, String)> {
    if let Some(root) = root {
        if !path.is_absolute()
            || !path.starts_with(root)
            || path.components().any(|c| matches!(c, Component::ParentDir))
        {
            return Err(invalid("control record outside owned namespace"));
        }
        let mut parent = path.parent();
        while let Some(p) = parent {
            if !fs::symlink_metadata(p)?.is_dir() {
                return Err(invalid("symlink control ancestry"));
            }
            if p == root {
                break;
            }
            parent = p.parent();
        }
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let before = file.metadata()?;
    if !before.is_file() || before.uid() != unsafe { libc::getuid() } || before.nlink() != 1 {
        return Err(invalid("unowned/nonregular control record"));
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    let current = fs::symlink_metadata(path)?;
    if (
        before.dev(),
        before.ino(),
        before.len(),
        before.mtime(),
        before.mtime_nsec(),
        before.ctime(),
        before.ctime_nsec(),
    ) != (
        after.dev(),
        after.ino(),
        after.len(),
        after.mtime(),
        after.mtime_nsec(),
        after.ctime(),
        after.ctime_nsec(),
    ) || (after.dev(), after.ino()) != (current.dev(), current.ino())
    {
        return Err(invalid("control original changed"));
    }
    let value = serde_json::from_slice(&bytes).map_err(|_| invalid("corrupt control record"))?;
    Ok((value, bytes_sha(&bytes)))
}
pub(crate) fn resync(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?
        .sync_all()?;
    sync_dir(path.parent().unwrap())
}
pub(crate) fn write(path: &Path, value: &Value, immutable: bool) -> Result<String> {
    let data = serde_json::to_vec(value).map_err(|_| invalid("invalid control JSON"))?;
    if immutable && path.exists() {
        let (old, _) = read_owned(path, None)?;
        if old != *value {
            return Err(invalid("immutable control record differs"));
        }
        resync(path)?;
        return Ok(digest(value));
    }
    let temporary = path.with_file_name(format!(
        "{}.tmp-{}",
        path.file_name().unwrap().to_string_lossy(),
        random_id()?
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary)?;
        file.write_all(&data)?;
        file.sync_all()?;
        if immutable {
            fs::hard_link(&temporary, path)?;
            fs::remove_file(&temporary)?;
        } else {
            fs::rename(&temporary, path)?;
        }
        sync_dir(path.parent().unwrap())?;
        Ok(bytes_sha(&data))
    })();
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    result
}
