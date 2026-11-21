//! Bounded current-collection diagnostics, never simulator/policy authority.
use super::{ensure, error};
use crate::error::Result;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
const COUNTERS: [&str; 14] = [
    "event_page_successes",
    "event_page_failures",
    "event_transport_attempts",
    "inference_issued",
    "inference_successes",
    "inference_cancellations",
    "inference_expiries",
    "inference_rejections",
    "inference_failures",
    "pyo3_next_batch",
    "pyo3_submit_results",
    "pyo3_other",
    "writer_admissions",
    "writer_admission_failures",
];
const HISTOGRAMS: [&str; 5] = [
    "event_page",
    "inference_roundtrip",
    "storage_admission_wait",
    "storage_queue_age",
    "collection_storage_serialization_wait",
];
pub fn monotonic_ns() -> u64 {
    #[cfg(test)]
    if let Some(ns) = TEST_CLOCK.with(|clock| clock.get()) {
        return ns;
    }
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // CLOCK_MONOTONIC is supported by the existing Linux runtime contract.
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) },
        0
    );
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
pub struct Histogram {
    counts: Box<[u64; 2049]>,
    count: u64,
    overflow: u64,
    min: Option<u64>,
    max: Option<u64>,
    sum: f64,
}
impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: Box::new([0; 2049]),
            count: 0,
            overflow: 0,
            min: None,
            max: None,
            sum: 0.,
        }
    }
}
impl Histogram {
    pub fn add(&mut self, x: u128) {
        self.count += 1;
        self.sum += x as f64 / 1e9;
        if x > u64::MAX as u128 {
            self.overflow += 1;
            return;
        }
        let n = x as u64;
        self.min = Some(self.min.map_or(n, |v| v.min(n)));
        self.max = Some(self.max.map_or(n, |v| v.max(n)));
        let index = if n == 0 {
            0
        } else {
            let e = 63 - n.leading_zeros() as usize;
            1 + 32 * e + (((x - (1u128 << e)) * 32) / (1u128 << e)) as usize
        };
        self.counts[index] += 1;
    }
    pub fn value(&self) -> Value {
        json!({"recipe":"ns-exp2-sub32-v1","count":self.count,"counts":self.counts.as_slice(),"overflow_count":self.overflow,"min_ns":self.min,"max_ns":self.max,"sum_seconds":self.sum})
    }
}
struct Window {
    scope: Value,
    start: u64,
    counters: BTreeMap<String, u64>,
    histograms: BTreeMap<String, Histogram>,
    batches: BTreeMap<usize, u64>,
    errors: Vec<String>,
    queue: [usize; 4],
    queue_failures: u64,
}
#[derive(Default)]
pub struct Metrics {
    active: AtomicBool,
    window: Mutex<Option<Window>>,
}
impl Metrics {
    pub fn now(&self) -> Option<u64> {
        self.active.load(Ordering::Relaxed).then(monotonic_ns)
    }
    pub fn begin(&self, scope: Value, start: u64) -> Result<()> {
        let mut state = self.window.lock().unwrap();
        ensure(state.is_none(), "measurement scope already active")?;
        *state = Some(Window {
            scope,
            start,
            counters: COUNTERS.iter().map(|k| (k.to_string(), 0)).collect(),
            histograms: HISTOGRAMS
                .iter()
                .map(|k| (k.to_string(), Histogram::default()))
                .collect(),
            batches: BTreeMap::new(),
            errors: vec![],
            queue: [0; 4],
            queue_failures: 0,
        });
        self.active.store(true, Ordering::Release);
        Ok(())
    }
    pub fn increment(&self, name: &str, n: u64) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        if let Some(w) = self.window.lock().unwrap().as_mut() {
            *w.counters.get_mut(name).expect("known metric counter") += n;
        }
    }
    pub fn interval(&self, name: &str, start: Option<u64>, end: Option<u64>) {
        let (Some(start), Some(end)) = (start, end) else {
            return;
        };
        if let Some(w) = self.window.lock().unwrap().as_mut() {
            if end < start {
                let e = format!("invalid_clock_interval:{name}");
                if !w.errors.contains(&e) {
                    w.errors.push(e);
                }
            } else {
                w.histograms
                    .get_mut(name)
                    .expect("known histogram")
                    .add((end - start) as u128);
            }
        }
    }
    pub fn batch(&self, rows: usize) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        if let Some(w) = self.window.lock().unwrap().as_mut() {
            *w.batches.entry(rows).or_default() += 1;
        }
    }
    pub fn queue(&self, queued: usize, in_service: usize, outstanding: usize, depth: usize) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        if let Some(w) = self.window.lock().unwrap().as_mut() {
            for (peak, value) in w
                .queue
                .iter_mut()
                .zip([queued, in_service, outstanding, depth])
            {
                *peak = (*peak).max(value);
            }
        }
    }
    pub fn end(&self, end: u64, complete: bool) -> Result<Value> {
        let mut state = self.window.lock().unwrap();
        let mut w = state
            .take()
            .ok_or_else(|| error("no active measurement scope"))?;
        self.active.store(false, Ordering::Release);
        if end < w.start {
            w.errors.push("invalid_scope_clock_interval".into());
        }
        let issued = w.counters["inference_issued"];
        let completed = w.counters["inference_successes"];
        let failures = [
            "inference_cancellations",
            "inference_expiries",
            "inference_rejections",
            "inference_failures",
        ]
        .iter()
        .map(|k| w.counters[*k])
        .sum::<u64>();
        let histograms: serde_json::Map<String, Value> = w
            .histograms
            .iter()
            .map(|(k, h)| {
                (
                    k.clone(),
                    if k == "collection_storage_serialization_wait" {
                        Value::Null
                    } else {
                        h.value()
                    },
                )
            })
            .collect();
        Ok(
            json!({"scope":w.scope,"start_monotonic_ns":w.start,"end_monotonic_ns":end,"complete":complete&&w.errors.is_empty(),"errors":w.errors,"counters":w.counters,"histograms":histograms,"batch_rows":{"method":"native-issued-batches-v1","request_size_counts":null,"batch_size_counts":w.batches,"issued_requests":issued,"completed_responses":completed,"quiescent_start":true,"quiescent_end":complete,"failures":failures},"stage_seconds":{"assembly":null,"cpu_share":null,"h2d":null,"forward":null,"sampling":null,"return":null,"native_wait":null,"native_submit":null},"queue":{"model":"native-ingress","queued_bytes_peak":w.queue[0],"in_service_bytes_peak":w.queue[1],"outstanding_bytes_peak":w.queue[2],"depth_peak":w.queue[3],"failed_attempts":w.queue_failures}}),
        )
    }
}
/// A logical validated page, including all internal retries. Failures never sample success latency.
pub struct PageSpan<'a> {
    metrics: &'a Metrics,
    start: Option<u64>,
    success: bool,
}
impl<'a> PageSpan<'a> {
    pub fn new(metrics: &'a Metrics) -> Self {
        Self {
            metrics,
            start: metrics.now(),
            success: false,
        }
    }
    pub fn success(&mut self) {
        self.success = true;
    }
}
impl Drop for PageSpan<'_> {
    fn drop(&mut self) {
        if self.start.is_none() {
            return;
        }
        self.metrics.increment(
            if self.success {
                "event_page_successes"
            } else {
                "event_page_failures"
            },
            1,
        );
        if self.success {
            self.metrics
                .interval("event_page", self.start, self.metrics.now());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_bins_rank_and_overflow_match_python_recipe() {
        let mut h = Histogram::default();
        for x in [0, 1, 2, 31, 32, 33, 63, 64, u64::MAX as u128, 1u128 << 64] {
            h.add(x);
        }
        let v = h.value();
        assert_eq!(v["counts"].as_array().unwrap().len(), 2049);
        assert_eq!(v["count"], 10);
        assert_eq!(v["overflow_count"], 1);
        assert_eq!(v["max_ns"], u64::MAX);
    }
    #[test]
    fn request_failures_and_cancelled_futures_never_sample_success() {
        let m = Metrics::default();
        m.begin(json!({}), 0).unwrap();
        {
            let _cancelled = InferenceSpan::new(&m);
        }
        {
            let mut span = InferenceSpan::new(&m);
            span.finish::<()>(&Err(crate::error::Error::Deadline));
        }
        {
            let mut span = InferenceSpan::new(&m);
            span.finish::<()>(&Err(error("rejected reply")));
        }
        {
            let _failed = StorageAdmission::new(&m);
        }
        let data = m.end(monotonic_ns(), true).unwrap();
        assert_eq!(data["counters"]["inference_cancellations"], 1);
        assert_eq!(data["counters"]["inference_expiries"], 1);
        assert_eq!(data["counters"]["inference_rejections"], 1);
        assert_eq!(data["histograms"]["inference_roundtrip"]["count"], 0);
        assert_eq!(data["counters"]["writer_admission_failures"], 1);
    }
    #[test]
    fn disabled_scope_partition_and_drain_are_bounded() {
        let m = Metrics::default();
        assert!(m.now().is_none());
        m.increment("event_page_successes", 1);
        m.begin(serde_json::json!({"batch_id":"first"}), 10)
            .unwrap();
        m.increment("event_page_successes", 1);
        m.interval("event_page", Some(11), Some(17));
        m.interval("inference_roundtrip", Some(20), Some(19));
        let first = m.end(30, true).unwrap();
        assert_eq!(first["counters"]["event_page_successes"], 1);
        assert_eq!(first["histograms"]["event_page"]["count"], 1);
        assert_eq!(first["complete"], false);
        assert!(m.now().is_none());
        m.begin(serde_json::json!({"batch_id":"second"}), 40)
            .unwrap();
        let second = m.end(50, true).unwrap();
        assert_eq!(second["counters"]["event_page_successes"], 0);
        assert_eq!(second["histograms"]["event_page"]["count"], 0);
        assert!(m.end(60, true).is_err());
    }
}

impl Metrics {
    pub fn end_scope(&self, scope: &Value, end: u64, complete: bool) -> Result<Value> {
        {
            let w = self.window.lock().unwrap();
            ensure(
                w.as_ref().is_some_and(|w| &w.scope == scope),
                "measurement scope differs",
            )?;
        }
        self.end(end, complete)
    }
}
pub struct InferenceSpan<'a> {
    metrics: &'a Metrics,
    start: Option<u64>,
    finished: bool,
}
impl<'a> InferenceSpan<'a> {
    pub fn new(metrics: &'a Metrics) -> Self {
        let start = metrics.now();
        if start.is_some() {
            metrics.increment("inference_issued", 1);
        }
        Self {
            metrics,
            start,
            finished: false,
        }
    }
    pub fn finish<T>(&mut self, result: &Result<T>) {
        if self.start.is_none() {
            self.finished = true;
            return;
        }
        let counter = match result {
            Ok(_) => "inference_successes",
            Err(crate::error::Error::Deadline) => "inference_expiries",
            Err(crate::error::Error::Cancelled) => "inference_cancellations",
            Err(crate::error::Error::Contract(_)) => "inference_rejections",
            Err(_) => "inference_failures",
        };
        self.metrics.increment(counter, 1);
        if result.is_ok() {
            self.metrics
                .interval("inference_roundtrip", self.start, self.metrics.now());
        }
        self.finished = true;
    }
}
impl Drop for InferenceSpan<'_> {
    fn drop(&mut self) {
        if self.start.is_some() && !self.finished {
            self.metrics.increment("inference_cancellations", 1);
        }
    }
}
pub struct StorageAdmission<'a> {
    metrics: &'a Metrics,
    start: Option<u64>,
    finished: bool,
}
impl<'a> StorageAdmission<'a> {
    pub fn new(metrics: &'a Metrics) -> Self {
        Self {
            metrics,
            start: metrics.now(),
            finished: false,
        }
    }
    pub fn admitted(&mut self) {
        if self.start.is_some() {
            self.metrics.increment("writer_admissions", 1);
            self.metrics
                .interval("storage_admission_wait", self.start, self.metrics.now());
        }
        self.finished = true;
    }
}
impl Drop for StorageAdmission<'_> {
    fn drop(&mut self) {
        if self.start.is_some() && !self.finished {
            self.metrics.increment("writer_admission_failures", 1);
            self.metrics
                .interval("storage_admission_wait", self.start, self.metrics.now());
            if let Some(w) = self.metrics.window.lock().unwrap().as_mut() {
                w.queue_failures += 1;
            }
        }
    }
}

#[cfg(test)]
std::thread_local! { static TEST_CLOCK: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub struct TestClock;
#[cfg(test)]
impl Drop for TestClock {
    fn drop(&mut self) {
        TEST_CLOCK.with(|clock| clock.set(None));
    }
}
#[cfg(test)]
pub fn test_clock(ns: u64) -> TestClock {
    set_test_clock(ns);
    TestClock
}
#[cfg(test)]
pub fn set_test_clock(ns: u64) {
    TEST_CLOCK.with(|clock| clock.set(Some(ns)));
}
