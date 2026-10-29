//! Counters for the actual NativeSession FFI boundary, not process-wide claims.
use crate::error::{Error, Result};
use std::time::Instant;
use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering},
};

/// Elapsed phases may nest or overlap across workers; they are never CPU time.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    SourceReadHash,
    SourceDecode,
    JournalInputValidation,
    ReplayValidation,
    FeatureEncoding,
    ProofStaging,
}
impl Phase {
    const ALL: [Self; 6] = [
        Self::SourceReadHash,
        Self::SourceDecode,
        Self::JournalInputValidation,
        Self::ReplayValidation,
        Self::FeatureEncoding,
        Self::ProofStaging,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::SourceReadHash => "source_read_hash",
            Self::SourceDecode => "source_decode",
            Self::JournalInputValidation => "journal_input_validation",
            Self::ReplayValidation => "replay_validation",
            Self::FeatureEncoding => "feature_encoding",
            Self::ProofStaging => "proof_staging",
        }
    }
}
#[derive(Default)]
struct PhaseCounters {
    ns: AtomicU64,
    calls: AtomicU64,
}
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum DecodeKind {
    Journal,
    Replay,
    Json,
}
impl DecodeKind {
    fn name(self) -> &'static str {
        match self {
            Self::Journal => "journal",
            Self::Replay => "replay",
            Self::Json => "json",
        }
    }
}
#[derive(Default)]
struct DecodeCounters {
    files: AtomicU64,
    bytes: AtomicU64,
    ns: AtomicU64,
    failed: AtomicU64,
}
pub struct DecodeTimer<'a> {
    counters: &'a DecodeCounters,
    started: Instant,
    bytes: usize,
    success: bool,
}
impl DecodeTimer<'_> {
    pub fn success(&mut self) {
        self.success = true;
    }
}
impl Drop for DecodeTimer<'_> {
    fn drop(&mut self) {
        self.counters.files.fetch_add(1, Ordering::Relaxed);
        self.counters
            .bytes
            .fetch_add(self.bytes as u64, Ordering::Relaxed);
        self.counters
            .ns
            .fetch_add(self.started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !self.success {
            self.counters.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Exclusive sequential coordinator phases. Decoder worker timers are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum CoordinatorPhase {
    CollectionSetup,
    DecodeWaitMerge,
    JournalValidation,
    SourceMetadata,
    ProofLedger,
    EpisodeFactsBuffers,
    FeaturesSpooling,
    TemporaryRelease,
    CollectionProof,
    GlobalChecks,
    Other,
}
impl CoordinatorPhase {
    const ALL: [Self; 11] = [
        Self::CollectionSetup,
        Self::DecodeWaitMerge,
        Self::JournalValidation,
        Self::SourceMetadata,
        Self::ProofLedger,
        Self::EpisodeFactsBuffers,
        Self::FeaturesSpooling,
        Self::TemporaryRelease,
        Self::CollectionProof,
        Self::GlobalChecks,
        Self::Other,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::CollectionSetup => "collection_setup_index_inventory",
            Self::DecodeWaitMerge => "decode_window_wait_and_source_merge",
            Self::JournalValidation => "journal_validation",
            Self::SourceMetadata => "source_metadata_linkage_canonical_checks",
            Self::ProofLedger => "proof_ledger_serialization",
            Self::EpisodeFactsBuffers => "episode_facts_layout_buffers",
            Self::FeaturesSpooling => "features_and_numeric_spooling",
            Self::TemporaryRelease => "episode_and_collection_temporary_release",
            Self::CollectionProof => "collection_source_proof_check",
            Self::GlobalChecks => "global_identity_split_source_manifest_checks",
            Self::Other => "other",
        }
    }
}
#[derive(Clone, Copy)]
struct WallState {
    current: CoordinatorPhase,
    since: Instant,
    ns: [u64; 11],
}
/// Single-coordinator stopwatch; Cell borrows never span any application hook.
/// The final Drop charges all remaining elapsed time, including error unwinding.
pub struct ScanWall<'a> {
    telemetry: &'a Telemetry,
    started: Instant,
    state: Cell<WallState>,
    success: Cell<bool>,
}
pub struct WallScope<'a, 'b> {
    wall: &'a ScanWall<'b>,
    previous: CoordinatorPhase,
}
impl ScanWall<'_> {
    pub fn set(&self, phase: CoordinatorPhase) {
        let now = Instant::now();
        let mut state = self.state.get();
        state.ns[state.current as usize] += now.duration_since(state.since).as_nanos() as u64;
        state.since = now;
        state.current = phase;
        self.state.set(state);
    }
    pub fn scope(&self, phase: CoordinatorPhase) -> WallScope<'_, '_> {
        let previous = self.state.get().current;
        self.set(phase);
        WallScope {
            wall: self,
            previous,
        }
    }
    pub fn success(&self) {
        self.success.set(true);
    }
}
impl Drop for WallScope<'_, '_> {
    fn drop(&mut self) {
        self.wall.set(self.previous);
    }
}
impl Drop for ScanWall<'_> {
    fn drop(&mut self) {
        let now = Instant::now();
        let mut state = self.state.get();
        state.ns[state.current as usize] += now.duration_since(state.since).as_nanos() as u64;
        for (counter, ns) in self.telemetry.coordinator_ns.iter().zip(state.ns) {
            counter.fetch_add(ns, Ordering::Relaxed);
        }
        self.telemetry.coordinator_parent_ns.fetch_add(
            now.duration_since(self.started).as_nanos() as u64,
            Ordering::Relaxed,
        );
        self.telemetry
            .coordinator_calls
            .fetch_add(1, Ordering::Relaxed);
        if !self.success.get() {
            self.telemetry
                .coordinator_failed
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}
#[derive(Default)]
struct WorkerSlot {
    jobs: AtomicU64,
    failed: AtomicU64,
    ns: AtomicU64,
}
pub struct DecodeJob<'a> {
    telemetry: &'a Telemetry,
    slot: usize,
    started: Instant,
    success: bool,
}
impl DecodeJob<'_> {
    pub fn success(&mut self) {
        self.success = true;
    }
}
impl Drop for DecodeJob<'_> {
    fn drop(&mut self) {
        let slot = &self.telemetry.slots[self.slot];
        slot.jobs.fetch_add(1, Ordering::Relaxed);
        slot.ns
            .fetch_add(self.started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !self.success {
            slot.failed.fetch_add(1, Ordering::Relaxed);
        }
        self.telemetry.active_decode.fetch_sub(1, Ordering::Relaxed);
    }
}
#[derive(Default)]
pub struct Telemetry {
    coordinator_ns: [AtomicU64; 11],
    coordinator_parent_ns: AtomicU64,
    coordinator_calls: AtomicU64,
    coordinator_failed: AtomicU64,
    configured_workers: AtomicU64,
    configured_in_flight: AtomicU64,
    windows: AtomicU64,
    window_episodes: AtomicU64,
    threads_started: AtomicU64,
    effective_slots_peak: AtomicU64,
    active_decode: AtomicU64,
    active_decode_peak: AtomicU64,
    slots: [WorkerSlot; 8],
    #[cfg(test)]
    decode_start_barrier: std::sync::Mutex<
        Option<(
            std::sync::Arc<std::sync::Barrier>,
            std::sync::Arc<std::sync::Barrier>,
        )>,
    >,
    decode_kinds: [DecodeCounters; 3],
    data_calls: AtomicU64,
    batches: AtomicU64,
    eof: AtomicU64,
    control_calls: AtomicU64,
    input_borrowed: AtomicU64,
    input_copied: AtomicU64,
    control_output_copied: AtomicU64,
    shared_exports: AtomicU64,
    shared_bytes: AtomicU64,
    numeric_copy_events: AtomicU64,
    numeric_copy_bytes: AtomicU64,
    preflight_ns: AtomicU64,
    next_ns: AtomicU64,
    scan_ns: AtomicU64,
    cleanup_ns: AtomicU64,
    stage_read: AtomicU64,
    stage_write: AtomicU64,
    scratch_peak: AtomicU64,
    phases: [PhaseCounters; 6],
}
pub struct Timer<'a> {
    counter: &'a AtomicU64,
    calls: Option<&'a AtomicU64>,
    started: Instant,
}
impl Drop for Timer<'_> {
    fn drop(&mut self) {
        self.counter
            .fetch_add(self.started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(calls) = self.calls {
            calls.fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl Telemetry {
    pub fn scan_wall(&self) -> ScanWall<'_> {
        let started = Instant::now();
        ScanWall {
            telemetry: self,
            started,
            state: Cell::new(WallState {
                current: CoordinatorPhase::Other,
                since: started,
                ns: [0; 11],
            }),
            success: Cell::new(false),
        }
    }
    pub fn configure_workers(&self, workers: usize, in_flight: usize) {
        self.configured_workers
            .store(workers as u64, Ordering::Relaxed);
        self.configured_in_flight
            .store(in_flight as u64, Ordering::Relaxed);
    }
    pub fn decode_window(&self, workers: usize, episodes: usize) {
        self.windows.fetch_add(1, Ordering::Relaxed);
        self.window_episodes
            .fetch_add(episodes as u64, Ordering::Relaxed);
        self.effective_slots_peak
            .fetch_max(workers.min(episodes) as u64, Ordering::Relaxed);
    }
    pub fn thread_started(&self) {
        self.threads_started.fetch_add(1, Ordering::Relaxed);
    }
    pub fn decode_job(&self, slot: usize) -> DecodeJob<'_> {
        let job = DecodeJob {
            telemetry: self,
            slot,
            started: Instant::now(),
            success: false,
        };
        let active = self.active_decode.fetch_add(1, Ordering::Relaxed) + 1;
        self.active_decode_peak.fetch_max(active, Ordering::Relaxed);
        #[cfg(test)]
        {
            let barrier = self.decode_start_barrier.lock().unwrap().clone();
            if let Some((entered, release)) = barrier {
                entered.wait();
                release.wait();
            }
        }
        job
    }
    #[cfg(test)]
    pub(crate) fn test_decode_barriers(
        &self,
        count: usize,
    ) -> (
        std::sync::Arc<std::sync::Barrier>,
        std::sync::Arc<std::sync::Barrier>,
    ) {
        let barriers = (
            std::sync::Arc::new(std::sync::Barrier::new(count)),
            std::sync::Arc::new(std::sync::Barrier::new(count)),
        );
        *self.decode_start_barrier.lock().unwrap() = Some(barriers.clone());
        barriers
    }
    fn coordinator_json(&self) -> String {
        let parent = self.coordinator_parent_ns.load(Ordering::Relaxed);
        let partitions: serde_json::Map<String, serde_json::Value> = CoordinatorPhase::ALL
            .iter()
            .map(|phase| {
                (
                    phase.name().to_owned(),
                    serde_json::json!(self.coordinator_ns[*phase as usize].load(Ordering::Relaxed)),
                )
            })
            .collect();
        let sum: u64 = self
            .coordinator_ns
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        serde_json::json!({"scope": "exclusive sequential scan coordinator wall; worker diagnostics are nested and excluded from this sum", "parent_nanoseconds": parent,
            "partition_nanoseconds": partitions, "partition_sum_nanoseconds": sum,
            "residual_nanoseconds": i128::from(parent) - i128::from(sum),
            "parent_seconds": parent as f64 / 1e9,
            "calls": self.coordinator_calls.load(Ordering::Relaxed),
            "failed_calls": self.coordinator_failed.load(Ordering::Relaxed)}).to_string()
    }
    fn parallel_json(&self) -> String {
        let value = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let slots: Vec<_> = self.slots.iter().enumerate().map(|(index, slot)| serde_json::json!({
            "logical_slot": index, "jobs": value(&slot.jobs), "failed_jobs": value(&slot.failed),
            "elapsed_nanoseconds": value(&slot.ns), "elapsed_seconds": value(&slot.ns) as f64 / 1e9,
        })).collect();
        serde_json::json!({"scope": "decoder jobs only; logical slots reused across recreated window threads; elapsed is not CPU time; active peak is overlap, not CPU utilization",
            "configured_workers": value(&self.configured_workers), "configured_max_in_flight": value(&self.configured_in_flight),
            "windows": value(&self.windows), "scheduled_episodes": value(&self.window_episodes),
            "spawned_threads": value(&self.threads_started), "effective_slots_peak": value(&self.effective_slots_peak),
            "active_jobs": value(&self.active_decode), "active_decode_peak": value(&self.active_decode_peak),
            "logical_slots": slots}).to_string()
    }

    pub fn decode_timer(&self, kind: DecodeKind, bytes: usize) -> DecodeTimer<'_> {
        DecodeTimer {
            counters: &self.decode_kinds[kind as usize],
            started: Instant::now(),
            bytes,
            success: false,
        }
    }
    /// One start/stop pair per file or coarse operation, including failed attempts.
    /// A live guard is not included in snapshots until it is dropped.
    pub fn phase_timer(&self, phase: Phase) -> Timer<'_> {
        let counters = &self.phases[phase as usize];
        Timer {
            counter: &counters.ns,
            calls: Some(&counters.calls),
            started: Instant::now(),
        }
    }
    pub fn preflight(&self, started: Instant) {
        self.preflight_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn scan(&self, started: Instant) {
        self.scan_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn data_timer(&self) -> Timer<'_> {
        Timer {
            counter: &self.next_ns,
            calls: None,
            started: Instant::now(),
        }
    }
    pub fn cleanup_timer(&self) -> Timer<'_> {
        Timer {
            counter: &self.cleanup_ns,
            calls: None,
            started: Instant::now(),
        }
    }
    pub fn stage_read(&self, bytes: usize) {
        self.stage_read.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub fn stage_write(&self, bytes: usize) {
        self.stage_write.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub fn scratch_peak(&self, bytes: usize) {
        self.scratch_peak.fetch_max(bytes as u64, Ordering::Relaxed);
    }
    pub fn control(&self, borrowed: usize, copied: usize) {
        self.control_calls.fetch_add(1, Ordering::Relaxed);
        self.input_borrowed
            .fetch_add(borrowed as u64, Ordering::Relaxed);
        self.input_copied
            .fetch_add(copied as u64, Ordering::Relaxed);
    }
    pub fn output_copy(&self, bytes: usize) {
        self.control_output_copied
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub fn data_call(&self) {
        self.data_calls.fetch_add(1, Ordering::Relaxed);
    }
    pub fn delivered(&self, batch: bool) {
        if batch {
            self.batches.fetch_add(1, Ordering::Relaxed);
        } else {
            self.eof.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn export(&self, native: usize, numpy: usize, bytes: usize) -> Result<()> {
        if bytes > 0 && native != numpy {
            self.numeric_copy_events.fetch_add(1, Ordering::Relaxed);
            self.numeric_copy_bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
            return Err(Error::contract(
                "NumPy export unexpectedly copied native data",
            ));
        }
        self.shared_exports.fetch_add(1, Ordering::Relaxed);
        self.shared_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }
    pub fn json(
        &self,
        reallocations: usize,
        reallocated_bytes: usize,
        extend_copy_bytes: usize,
    ) -> String {
        let value = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let worker_seconds = Phase::ALL
            .iter()
            .map(|phase| {
                format!(
                    "\"{}\":{}",
                    phase.name(),
                    value(&self.phases[*phase as usize].ns) as f64 / 1e9
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let worker_calls = Phase::ALL
            .iter()
            .map(|phase| {
                format!(
                    "\"{}\":{}",
                    phase.name(),
                    value(&self.phases[*phase as usize].calls)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let output = format!(
            concat!(
                "{{\"scope\":\"NativeSession boundary; numeric pointer probes and explicit control-byte copies\",",
                "\"accounting_policy\":\"numeric-capacity-and-coarse-json-document-admission-v2\",\"json_heap_scope\":\"32*raw_input_bytes+64KiB document/file estimates; not exact serde heap or RSS\",\"copy_counter_scope\":\"explicit BudgetVec growth/slice copies only; serde heap, clone, String/Vec/Map allocations and copies are untracked\",",
                "\"snapshot\":\"before this telemetry response copy\",\"ffi_data_calls\":{},\"ffi_batches_returned\":{},\"ffi_eof_returns\":{},\"ffi_control_calls\":{},",
                "\"stage_io_scope\":\"owned numeric/proof/identity logical staging; excludes Python serialization and OS/cache accounting\",\"stage_read_bytes\":{},\"stage_write_bytes\":{},\"scratch_peak_bytes\":{},",
                "\"phase_seconds\":{{\"constructor_native\":{},\"next_batch_inclusive\":{},\"global_scan_and_encode\":{},\"finalize_or_abort\":{}}},",
                "\"aggregated_worker_wall_scope\":\"summed elapsed wall time across owned native executor/workers; phases may nest and workers overlap; not additive, critical-path wall time or CPU time; calls include failed attempts and exclude active guards\",",
                "\"aggregated_worker_wall_seconds\":{{{}}},\"aggregated_worker_phase_calls\":{{{}}},",
                "\"payload_copy_stats\":{{\"control_input_borrowed_bytes\":{},\"control_input_copied_bytes\":{},\"control_output_copied_bytes\":{},",
                "\"verified_shared_numeric_exports\":{},\"verified_shared_numeric_bytes\":{},\"numeric_copy_events\":{},\"numeric_copy_bytes\":{},",
                "\"managed_buffer_reallocation_events\":{},\"managed_buffer_reallocation_copy_bytes\":{},\"managed_extend_from_slice_copy_bytes\":{}}}}}"
            ),
            value(&self.data_calls),
            value(&self.batches),
            value(&self.eof),
            value(&self.control_calls),
            value(&self.stage_read),
            value(&self.stage_write),
            value(&self.scratch_peak),
            value(&self.preflight_ns) as f64 / 1e9,
            value(&self.next_ns) as f64 / 1e9,
            value(&self.scan_ns) as f64 / 1e9,
            value(&self.cleanup_ns) as f64 / 1e9,
            worker_seconds,
            worker_calls,
            value(&self.input_borrowed),
            value(&self.input_copied),
            value(&self.control_output_copied),
            value(&self.shared_exports),
            value(&self.shared_bytes),
            value(&self.numeric_copy_events),
            value(&self.numeric_copy_bytes),
            reallocations,
            reallocated_bytes,
            extend_copy_bytes
        );
        let kinds = [DecodeKind::Journal, DecodeKind::Replay, DecodeKind::Json]
            .iter()
            .map(|kind| {
                let c = &self.decode_kinds[*kind as usize];
                format!(
                    "\"{}\":{{\"files\":{},\"input_bytes\":{},\"wall_seconds\":{},\"failed\":{}}}",
                    kind.name(),
                    value(&c.files),
                    value(&c.bytes),
                    value(&c.ns) as f64 / 1e9,
                    value(&c.failed)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{},\"source_kind_decode\":{{{}}},\"json_memory_policy\":\"{}\",\"coordinator_scan_wall\":{},\"decode_parallelism\":{}}}",
            &output[..output.len() - 1],
            kinds,
            crate::json::MEMORY_POLICY,
            self.coordinator_json(),
            self.parallel_json()
        )
    }
}

#[cfg(test)]
mod wall_tests {
    use super::*;

    #[test]
    fn nested_wall_scopes_restore_after_errors_and_reconcile_exactly() {
        let telemetry = Telemetry::default();
        {
            let wall = telemetry.scan_wall();
            wall.set(CoordinatorPhase::CollectionSetup);
            let result: Result<()> = (|| {
                let _outer = wall.scope(CoordinatorPhase::SourceMetadata);
                {
                    let _inner = wall.scope(CoordinatorPhase::ProofLedger);
                    assert_eq!(wall.state.get().current, CoordinatorPhase::ProofLedger);
                }
                assert_eq!(wall.state.get().current, CoordinatorPhase::SourceMetadata);
                Err(Error::contract("injected nested failure"))
            })();
            assert!(result.is_err());
            assert_eq!(wall.state.get().current, CoordinatorPhase::CollectionSetup);
        }
        let value: serde_json::Value = serde_json::from_str(&telemetry.coordinator_json()).unwrap();
        let total = value["parent_nanoseconds"].as_u64().unwrap();
        let sum: u64 = value["partition_nanoseconds"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_u64().unwrap())
            .sum();
        assert_eq!(total, sum);
        assert_eq!(value["partition_sum_nanoseconds"], sum);
        assert_eq!(value["residual_nanoseconds"], 0);
        assert_eq!(value["failed_calls"], 1);
        {
            let wall = telemetry.scan_wall();
            wall.success();
        }
        let value: serde_json::Value = serde_json::from_str(&telemetry.coordinator_json()).unwrap();
        assert_eq!(value["calls"], 2);
        assert_eq!(value["failed_calls"], 1);
        assert_eq!(value["residual_nanoseconds"], 0);
    }
}
