//! Counters for the actual NativeSession FFI boundary, not process-wide claims.
use crate::error::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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

#[derive(Default)]
pub struct Telemetry {
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
            "{},\"source_kind_decode\":{{{}}},\"json_memory_policy\":\"{}\"}}",
            &output[..output.len() - 1],
            kinds,
            crate::json::MEMORY_POLICY
        )
    }
}
