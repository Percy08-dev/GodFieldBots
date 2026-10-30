//! Standalone ADR0016 Journal experiment. No production or learning output.
#![allow(dead_code)] // Generated DTO leaf helpers also serve contract fixtures.
mod common;
#[path = "../../../../native/src/typed_journal/dto/mod.rs"]
mod shared_dto;
use shared_dto::{borrowed, contract};
#[cfg(test)]
mod contract_tests;
#[cfg(feature = "allocation_counts")]
mod counter;
mod feasibility;
mod memory;
mod owned;

#[cfg(all(feature = "allocation_counts", feature = "allocator-mimalloc"))]
compile_error!("System allocation counting and mimalloc must use separate builds");
#[cfg(feature = "allocator-mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn allocator_name() -> &'static str {
    if cfg!(feature = "allocator-mimalloc") {
        "mimalloc"
    } else {
        "System"
    }
}

use contract::Validate;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, hint::black_box, path::PathBuf, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
    #[serde(alias = "bytes")]
    logical_bytes: usize,
}
#[derive(Default, Serialize)]
struct Phase {
    wall_seconds: f64,
    calls: u64,
    #[cfg(feature = "allocation_counts")]
    allocation_requests: u64,
    #[cfg(feature = "allocation_counts")]
    requested_bytes: u64,
    #[cfg(feature = "allocation_counts")]
    deallocation_requests: u64,
    #[cfg(feature = "allocation_counts")]
    reallocation_requests: u64,
}
struct Clock {
    start: Instant,
    #[cfg(feature = "allocation_counts")]
    counts: [u64; 4],
}
fn tick() -> Clock {
    #[cfg(feature = "allocation_counts")]
    let counts = counter::snapshot();
    Clock {
        start: Instant::now(),
        #[cfg(feature = "allocation_counts")]
        counts,
    }
}
fn add(phases: &mut BTreeMap<&'static str, Phase>, name: &'static str, clock: Clock) {
    let wall = clock.start.elapsed().as_secs_f64();
    #[cfg(feature = "allocation_counts")]
    let counts = counter::snapshot();
    let value = phases.entry(name).or_default();
    value.wall_seconds += wall;
    value.calls += 1;
    #[cfg(feature = "allocation_counts")]
    {
        value.allocation_requests += counts[0] - clock.counts[0];
        value.requested_bytes += counts[1] - clock.counts[1];
        value.deallocation_requests += counts[2] - clock.counts[2];
        value.reallocation_requests += counts[3] - clock.counts[3];
    }
}
fn parse<'de, T: Deserialize<'de>>(lines: &[&'de [u8]]) -> Result<Vec<T>, String> {
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| serde_json::from_slice(line).map_err(|e| format!("record {i}: {e}")))
        .collect()
}
fn validate<T: Validate>(records: &[T]) -> Result<(), String> {
    for (i, record) in records.iter().enumerate() {
        record.validate().map_err(|e| format!("record {i}: {e}"))?;
    }
    Ok(())
}
fn export<T: Serialize>(path: &PathBuf, records: &[T]) -> Result<(), String> {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &(path, records)).map_err(|e| e.to_string())?;
    output.write_all(b"\n").map_err(|e| e.to_string())
}
fn main_result() -> Result<(), String> {
    if std::env::var_os("LD_PRELOAD").is_some() {
        return Err("LD_PRELOAD prohibited".into());
    }
    if std::env::args().nth(1).as_deref() == Some("--identity") {
        println!(
            "{}",
            serde_json::json!({"allocator":allocator_name(),
            "arbitrary_precision":cfg!(feature="arbitrary_precision"),
            "allocation_counts":cfg!(feature="allocation_counts"),
            "source_sha256":option_env!("GFB_EXPERIMENT_SOURCE_SHA256").unwrap_or("unrecorded"),
            "contract":"typed-journal-experiment-v1"})
        );
        return Ok(());
    }
    let mut args = std::env::args().skip(1);
    let mut inventory = None;
    let mut variant = None;
    let mut mode = None;
    let mut seconds = 1800f64;
    let mut memory_mib = None;
    while let Some(key) = args.next() {
        let value = args.next().ok_or("flag requires value")?;
        match key.as_str() {
            "--inventory" => inventory = Some(value),
            "--variant" => variant = Some(value),
            "--mode" => mode = Some(value),
            "--seconds" => seconds = value.parse().map_err(|_| "seconds")?,
            "--memory-budget-mib" => {
                memory_mib = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| "memory budget integer")?,
                )
            }
            _ => return Err(format!("unknown flag {key}")),
        }
    }
    let variant = variant.ok_or("--variant A|B|C|D required")?;
    let mode = mode.ok_or("--mode measure|export|counter required")?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("positive finite seconds required".into());
    }
    if !matches!(mode.as_str(), "measure" | "export" | "counter") {
        return Err("unknown mode".into());
    }
    if cfg!(feature = "allocation_counts") != (mode == "counter") {
        return Err("counter and primary binaries/modes must be separate".into());
    }
    if (variant == "D") == cfg!(feature = "arbitrary_precision") {
        return Err("variant/AP build mismatch".into());
    }
    if !matches!(variant.as_str(), "A" | "B" | "C" | "D") {
        return Err("unknown representation".into());
    }
    if memory_mib != Some(2048) {
        return Err("fixed --memory-budget-mib 2048 required".into());
    }
    let mut managed = memory::Memory::new(2048 * memory::MIB);
    let inventory = PathBuf::from(inventory.ok_or("--inventory required")?);
    let request_len = usize::try_from(fs::metadata(&inventory).map_err(|e| e.to_string())?.len())
        .map_err(|_| "request size overflow")?;
    if request_len > common::MAX_FILE {
        return Err("request file limit".into());
    }
    let controls = memory::control_bytes(request_len)?;
    let request_capacity = memory::raw_bytes(request_len)?;
    managed.reserve(
        controls
            .checked_add(request_capacity)
            .ok_or("request admission overflow")?,
    )?;
    let request = memory::read_admitted(&inventory, request_len, request_capacity, &managed)?;
    let inputs: Vec<Input> = serde_json::from_slice(&request).map_err(|e| e.to_string())?;
    drop(request);
    managed.release(request_capacity)?;
    if inputs.is_empty() || inputs.len() > 13568 {
        return Err("inventory count outside frozen corpus bound".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for input in &inputs {
        if input.logical_bytes == 0
            || input.logical_bytes > common::MAX_FILE
            || input.sha256.len() != 64
            || !input.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            || input.path.as_os_str().is_empty()
            || !seen.insert(&input.path)
        {
            return Err("invalid/duplicate frozen inventory entry".into());
        }
    }
    drop(seen);
    let mut phases = BTreeMap::new();
    let mut files = 0;
    let mut records = 0;
    let mut bytes = 0;
    let whole = tick();
    for input in &inputs {
        if whole.start.elapsed().as_secs_f64() >= seconds {
            return Err("original harness deadline".into());
        }
        let clock = tick();
        let metadata = fs::metadata(&input.path).map_err(|e| e.to_string())?;
        if metadata.len() != input.logical_bytes as u64 {
            return Err("frozen source size changed before admission".into());
        }
        let (raw_charge, json_charge) = managed.admit_file(input.logical_bytes)?;
        let raw = memory::read_admitted(&input.path, input.logical_bytes, raw_charge, &managed)?;
        add(&mut phases, "read", clock);
        let clock = tick();
        let lines = common::records(&raw, &input.sha256)
            .map_err(|e| format!("{}: {e}", input.path.display()))?;
        records += lines.len();
        bytes += raw.len();
        files += 1;
        add(&mut phases, "common_checks", clock);
        match variant.as_str() {
            "A" | "D" => {
                let clock = tick();
                let values: Vec<serde_json::Value> = parse(&lines)?;
                black_box(&values);
                add(&mut phases, "deserialize", clock);
                if mode == "export" {
                    export(&input.path, &values)?;
                }
                let clock = tick();
                drop(lines);
                drop(raw);
                add(&mut phases, "input_release", clock);
                managed.release(raw_charge)?;
                let clock = tick();
                drop(values);
                add(&mut phases, "dto_drop", clock);
                managed.release(json_charge)?;
            }
            "B" => {
                let clock = tick();
                let values: Vec<owned::Record> = parse(&lines)?;
                black_box(&values);
                add(&mut phases, "deserialize", clock);
                let clock = tick();
                validate(&values)?;
                add(&mut phases, "typed_shape_validation", clock);
                if mode == "export" {
                    export(&input.path, &values)?;
                }
                let clock = tick();
                drop(lines);
                drop(raw);
                add(&mut phases, "input_release", clock);
                managed.release(raw_charge)?;
                let clock = tick();
                drop(values);
                add(&mut phases, "dto_drop", clock);
                managed.release(json_charge)?;
            }
            "C" => {
                let clock = tick();
                let values: Vec<borrowed::Record> = parse(&lines)?;
                black_box(&values);
                add(&mut phases, "deserialize", clock);
                let clock = tick();
                validate(&values)?;
                add(&mut phases, "typed_shape_validation", clock);
                if mode == "export" {
                    export(&input.path, &values)?;
                }
                let clock = tick();
                drop(values);
                add(&mut phases, "dto_drop", clock);
                let clock = tick();
                drop(lines);
                drop(raw);
                add(&mut phases, "input_release", clock);
                managed.release(raw_charge)?;
                managed.release(json_charge)?;
            }
            _ => unreachable!(),
        }
    }
    if whole.start.elapsed().as_secs_f64() >= seconds {
        return Err("original harness deadline after cleanup".into());
    }
    add(&mut phases, "whole_natural_cycle", whole);
    if mode != "export" {
        println!(
            "{}",
            serde_json::json!({"schema_version":1,"contract":"typed-journal-experiment-v1",
            "variant":variant,"mode":mode,"allocator":allocator_name(),"arbitrary_precision":cfg!(feature="arbitrary_precision"),
            "allocation_counts":cfg!(feature="allocation_counts"),"files":files,"records":records,"bytes":bytes,
            "phases":phases,"phase_cpu_seconds":null,
            "managed_memory_budget_bytes":managed.limit,"coarse_managed_peak_bytes":managed.peak,
            "post_release_managed_bytes":managed.used,"control_metadata_bytes":controls,
            "memory_scope":"coarse32x+64KiB JSON, separate admitted rawcapacity, persistent64x inventory+32MiB controls; estimates, not OS/RSS quota",
            "allocation_scope":"counter builds only; deserialize/drop snapshots exclude rawinput/read/guard/bookkeeping; read/common and whole counters are separate scopes, never additive",
            "common_guard_provenance":"native/src/json.rs reserved member and native/src/source.rs UTF8/CR/CRLF record framing at baseline e441053; unchanged semantics, no parser replacement",
            "phase_cpu_limitation":"unavailable; observer wait4 owns precise whole-process user/system CPU",
            "phase_scope":"read and common checks separate; typed_shape_validation B/C only; whole includes clocks/loop overhead; phases not a reconciled wall partition",
            "arbitrary_json_fields":[],"post_release_status":fs::read_to_string("/proc/self/status").ok(),
            "post_release_smaps_rollup":fs::read_to_string("/proc/self/smaps_rollup").ok(),
            "not_proven":"no typed feature/array equivalence or learning eligibility"})
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = main_result() {
        eprintln!("{error}");
        std::process::exit(2)
    }
}
