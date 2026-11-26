//! Immutable capture transport. Acceptance transfers custody, never proves persistence.
use super::capture::OwnedAction;
use super::super::{
    ensure, error,
    http::HttpFacts,
    training::{Episode, PlayerPpoEpisode},
};
use crate::{
    budget::{Budget, BudgetVec, Control, monotonic},
    error::{Error, Result},
    json::{self, ValueExt},
    structured_journal::card_instance,
};
use ndarray::Array1;
use numpy::PyArray1;
use pyo3::{prelude::*, types::PyBytes};
use serde::{Deserialize, Serialize, de::IgnoredAny};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    io::{self, Write},
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

pub(crate) mod rollout;

struct Count(usize);
impl Write for Count {
    fn write(&mut self, v: &[u8]) -> std::io::Result<usize> {
        self.0 += v.len();
        Ok(v.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[derive(Default)]
struct Ledger {
    native: usize,
    external: usize,
    training: usize,
    provisional: usize,
    waiters: usize,
    peak: usize,
}
pub struct Account {
    metrics: Arc<super::super::metrics::Metrics>,
    limit: usize,
    training_limit: usize,
    ledger: Mutex<Ledger>,
    available: Condvar,
}
impl Account {
    fn change(
        &self,
        native: bool,
        training: bool,
        old: usize,
        new: usize,
        deadline: Option<f64>,
    ) -> Result<()> {
        let mut s = self.ledger.lock().unwrap();
        let mut waiting = false;
        let result = loop {
            if deadline.is_some_and(|expires| monotonic() >= expires) {
                break Err(Error::Deadline);
            }
            let Some(total) = (s.native + s.external - old)
                .checked_add(new)
                .filter(|n| *n <= self.limit)
            else {
                break Err(Error::resource("recording byte budget exhausted"));
            };
            let next_training = if training {
                let Some(bytes) = (s.training - old)
                    .checked_add(new)
                    .filter(|n| *n <= self.training_limit)
                else {
                    break Err(Error::resource("PPO numeric byte budget exhausted"));
                };
                bytes
            } else {
                s.training
            };
            if total <= self.limit - s.provisional {
                if native {
                    s.native = s.native - old + new;
                } else {
                    s.external = s.external - old + new;
                }
                s.training = next_training;
                s.peak = s.peak.max(total + s.provisional);
                if new < old && s.waiters != 0 {
                    self.available.notify_all();
                }
                break Ok(());
            }

            // Only transient encoders block this otherwise admissible request.
            // Their synchronous leases never wait to grow or to become retained.
            let timeout = match deadline
                .map(super::super::requests::wait_duration)
                .transpose()
            {
                Ok(timeout) => timeout,
                Err(failure) => break Err(failure),
            };
            if !waiting {
                s.waiters += 1;
                waiting = true;
            }
            s = match timeout {
                Some(remaining) => self.available.wait_timeout(s, remaining).unwrap().0,
                None => self.available.wait(s).unwrap(),
            };
        };
        if waiting {
            s.waiters -= 1;
        }
        result
    }
    fn reserve(self: &Arc<Self>, bytes: usize, native: bool) -> Result<Reservation> {
        self.reserve_until(bytes, native, None)
    }
    fn reserve_until(
        self: &Arc<Self>,
        bytes: usize,
        native: bool,
        deadline: Option<f64>,
    ) -> Result<Reservation> {
        self.change(native, false, 0, bytes, deadline)?;
        Ok(Reservation {
            bytes,
            native,
            training: false,
            account: self.clone(),
        })
    }
    pub(crate) fn reserve_training(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        self.change(true, true, 0, bytes, None)?;
        Ok(Reservation { bytes, native: true, training: true, account: self.clone() })
    }
    pub(crate) fn training_fits(&self, bytes: usize) -> bool {
        let ledger = self.ledger.lock().unwrap();
        bytes <= self.limit - ledger.native - ledger.external
            && bytes <= self.training_limit - ledger.training
    }
    fn try_provision(self: &Arc<Self>, bytes: usize) -> Option<Provisional<'_>> {
        let mut ledger = self.ledger.lock().unwrap();
        let used = ledger.native + ledger.external + ledger.provisional;
        if ledger.waiters != 0 || bytes > self.limit - used {
            return None;
        }
        ledger.provisional += bytes;
        ledger.peak = ledger.peak.max(used + bytes);
        Some(Provisional {
            account: self,
            bytes,
        })
    }
}
pub struct Reservation {
    bytes: usize,
    native: bool,
    training: bool,
    account: Arc<Account>,
}
impl Reservation {
    pub(crate) fn bytes(&self) -> usize { self.bytes }
    fn resize(&mut self, size: usize) -> Result<()> {
        self.account
            .change(self.native, self.training, self.bytes, size, None)?;
        self.bytes = size;
        Ok(())
    }

    fn split(&mut self, bytes: usize) -> Result<Self> {
        self.bytes = self.bytes.checked_sub(bytes)
            .ok_or_else(|| error("split exceeds the admitted recording reservation"))?;
        Ok(Self {
            bytes,
            native: self.native,
            training: self.training,
            account: self.account.clone(),
        })
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        // A refund cannot exceed the admitted total; no allocation or callback.
        self.account
            .change(self.native, self.training, self.bytes, 0, None)
            .unwrap();
    }
}

struct Provisional<'a> {
    account: &'a Arc<Account>,
    bytes: usize,
}
impl Provisional<'_> {
    fn promote(&mut self) -> Option<Reservation> {
        let mut ledger = self.account.ledger.lock().unwrap();
        if ledger.waiters != 0 {
            return None;
        }
        let bytes = std::mem::take(&mut self.bytes);
        ledger.provisional -= bytes;
        ledger.native += bytes;
        Some(Reservation {
            bytes,
            native: true,
            training: false,
            account: self.account.clone(),
        })
    }
}
impl Drop for Provisional<'_> {
    fn drop(&mut self) {
        if self.bytes != 0 {
            let mut ledger = self.account.ledger.lock().unwrap();
            ledger.provisional -= self.bytes;
            if ledger.waiters != 0 {
                self.account.available.notify_all();
            }
        }
    }
}

struct CaptureEncoder<'a> {
    // Keep the allocation charged until after the buffer has been freed.
    bytes: Vec<u8>,
    reservation: Provisional<'a>,
    retry_exact: bool,
}
impl CaptureEncoder<'_> {
    fn replace_buffer(&mut self, capacity: usize) -> bool {
        let Some(reservation) = self.reservation.account.try_provision(capacity) else {
            return false;
        };
        let mut replacement = Vec::new();
        if replacement.try_reserve_exact(capacity).is_err() || replacement.capacity() != capacity {
            return false;
        }
        replacement.extend_from_slice(&self.bytes);
        self.bytes = replacement;
        self.reservation = reservation;
        true
    }
}
impl Write for CaptureEncoder<'_> {
    fn write(&mut self, chunk: &[u8]) -> io::Result<usize> {
        let needed = self.bytes.len().saturating_add(chunk.len());
        let capacity = self.bytes.capacity();
        if needed > capacity && !self.replace_buffer(needed.max(capacity.saturating_mul(2)).max(8))
        {
            self.retry_exact = true;
            return Err(io::Error::other("provisional capture buffer unavailable"));
        }
        self.bytes.extend_from_slice(chunk);
        Ok(chunk.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct RecordingAccount {
    pub(crate) account: Arc<Account>,
}
#[pymethods]
impl RecordingAccount {
    #[getter]
    fn limit(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        self.account.limit
    }
    #[getter]
    fn used(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        let s = self.account.ledger.lock().unwrap();
        s.native + s.external + s.provisional
    }
    #[getter]
    fn peak(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        self.account.ledger.lock().unwrap().peak
    }
    #[getter]
    fn native_bytes(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        let s = self.account.ledger.lock().unwrap();
        s.native + s.provisional
    }
    #[getter]
    fn external_bytes(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        self.account.ledger.lock().unwrap().external
    }
    fn same_account(&self, other: &Self) -> bool {
        self.account.metrics.increment("pyo3_other", 1);
        Arc::ptr_eq(&self.account, &other.account)
    }
    fn owns(&self, bundle: &StorageBundle) -> bool {
        self.account.metrics.increment("pyo3_other", 1);
        Arc::ptr_eq(&self.account, &bundle.record._reservation.account)
    }
    fn reserve(&self, bytes: usize) -> PyResult<RecordingLease> {
        self.account.metrics.increment("pyo3_other", 1);
        Ok(RecordingLease {
            metrics: self.account.metrics.clone(),
            lease: Mutex::new(Some(
                self.account
                    .reserve(bytes, false)
                    .map_err(crate::python_error)?,
            )),
        })
    }
}
#[pyclass(frozen)]
pub struct RecordingLease {
    metrics: Arc<super::super::metrics::Metrics>,
    lease: Mutex<Option<Reservation>>,
}
#[pymethods]
impl RecordingLease {
    #[getter]
    fn size(&self) -> usize {
        self.metrics.increment("pyo3_other", 1);
        self.lease.lock().unwrap().as_ref().map_or(0, |r| r.bytes)
    }
    fn resize(&self, bytes: usize) -> PyResult<()> {
        self.metrics.increment("pyo3_other", 1);
        let mut s = self.lease.lock().unwrap();
        let r = s
            .as_mut()
            .ok_or_else(|| crate::python_error(error("recording lease closed")))?;
        r.resize(bytes).map_err(crate::python_error)
    }
    fn close(&self) {
        self.metrics.increment("pyo3_other", 1);
        self.lease.lock().unwrap().take();
    }
}
#[derive(Clone, Copy)]
pub(crate) struct TerminalHttpDiagnostics {
    pub facts: HttpFacts,
    pub abnormal: bool,
}

pub struct Record {
    pub id: u64,
    pub kind: String,
    pub episode: String,
    pub game: String,
    array: Array1<u8>,
    transport_sha256: String,
    release_key: Option<(String, String, String)>,
    training: Option<Arc<Episode>>,
    http_diagnostics: Option<TerminalHttpDiagnostics>,
    action_capture: Mutex<Option<Weak<OwnedAction>>>,
    _reservation: Reservation,
}

impl Record {
    fn take_action_capture(&self) -> Option<Weak<OwnedAction>> {
        self.action_capture.lock().unwrap().take()
    }

    pub(crate) fn discard_action_capture(&self) {
        drop(self.take_action_capture());
    }
}
struct Entry {
    record: Arc<Record>,
    answer: Arc<StorageAnswer>,
    exported: bool,
    deadline: f64,
    enqueued_ns: Option<u64>,
}
#[derive(Default)]
struct StorageAnswer {
    result: Mutex<Option<Result<()>>>,
    wake: tokio::sync::Notify,
}
impl StorageAnswer {
    fn finish(&self, result: Result<()>) {
        let mut s = self.result.lock().unwrap();
        if s.is_none() {
            *s = Some(result);
            self.wake.notify_waiters();
        }
    }
    async fn wait(&self, deadline: f64) -> Result<()> {
        loop {
            let wake = self.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if let Some(r) = self.result.lock().unwrap().as_ref() {
                if r.is_ok() && monotonic() >= deadline {
                    return Err(Error::Deadline);
                }
                return r.clone();
            }
            if monotonic() >= deadline {
                return Err(Error::Deadline);
            }
            let _ =
                tokio::time::timeout(super::super::requests::wait_duration(deadline)?, wake).await;
        }
    }
}
struct State {
    next: u64,
    queue: VecDeque<u64>,
    entries: HashMap<u64, Entry>,
    closed: Option<Error>,
}
pub struct Mailbox {
    state: Mutex<State>,
    wake: Condvar,
    async_wake: tokio::sync::Notify,
    account: Arc<Account>,
    capacity: usize,
    native_rollout: AtomicBool,
    pub metrics: Arc<super::super::metrics::Metrics>,
}

struct NativeRolloutLease {
    mailbox: Arc<Mailbox>,
}

impl Drop for NativeRolloutLease {
    fn drop(&mut self) {
        // Coordinate with publishers inserting a record and a successor claiming
        // the same mailbox. The writer retires its workers before dropping this.
        let _state = self.mailbox.state.lock().unwrap();
        self.mailbox.native_rollout.store(false, Ordering::Release);
    }
}

impl Mailbox {
    pub fn new(limit: usize, capacity: usize) -> Result<Arc<Self>> {
        Self::new_monitored(limit, capacity, Arc::new(super::super::metrics::Metrics::default()))
    }
    pub fn new_monitored(limit: usize, capacity: usize, metrics: Arc<super::super::metrics::Metrics>) -> Result<Arc<Self>> {
        Self::new_with_training_limit(limit, capacity, limit, metrics)
    }
    pub(crate) fn new_with_training_limit(limit: usize, capacity: usize, training_limit: usize,
        metrics: Arc<super::super::metrics::Metrics>) -> Result<Arc<Self>> {
        ensure(
            limit > 0 && capacity > 0,
            "positive storage bounds required",
        )?;
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                next: 0,
                queue: VecDeque::new(),
                entries: HashMap::new(),
                closed: None,
            }),
            wake: Condvar::new(),
            async_wake: tokio::sync::Notify::new(),
            account: Arc::new(Account {
                metrics: metrics.clone(),
                limit,
                training_limit,
                ledger: Mutex::new(Ledger::default()),
                available: Condvar::new(),
            }),
            capacity,
            native_rollout: AtomicBool::new(false),
            metrics,
        }))
    }
    pub(in crate::player_runtime) fn native_rollout_active(&self) -> bool {
        self.native_rollout.load(Ordering::Acquire)
    }
    fn claim_native_rollout(self: &Arc<Self>) -> Result<NativeRolloutLease> {
        let _state = self.state.lock().unwrap();
        self.native_rollout
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| error("native Journal reader already owns this mailbox"))?;
        Ok(NativeRolloutLease { mailbox: self.clone() })
    }
    pub fn measurement_quiet(&self) -> bool {
        let s = self.state.lock().unwrap();
        s.queue.is_empty() && s.entries.is_empty()
    }
    fn observe_queue(&self, s: &State) {
        if self.metrics.now().is_none() { return; }
        let mut queued = 0; let mut in_service = 0;
        for entry in s.entries.values() {
            if entry.exported { in_service += entry.record.array.len(); }
            else { queued += entry.record.array.len(); }
        }
        self.metrics.queue(queued, in_service, queued + in_service, s.queue.len());
    }
    pub fn used(&self) -> usize {
        let s = self.account.ledger.lock().unwrap();
        s.native + s.provisional
    }
    pub fn account(&self) -> RecordingAccount {
        RecordingAccount {
            account: self.account.clone(),
        }
    }
    fn notify(&self) {
        self.wake.notify_all();
        self.async_wake.notify_waiters();
    }
    fn encode_capture<T: Serialize>(
        &self,
        value: &T,
        deadline: f64,
    ) -> Result<(Vec<u8>, Reservation)> {
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        if let Some(reservation) = self.account.try_provision(0) {
            let mut encoder = CaptureEncoder {
                bytes: Vec::new(),
                reservation,
                retry_exact: false,
            };
            match serde_json::to_writer(&mut encoder, value) {
                Ok(()) => {
                    if monotonic() >= deadline {
                        return Err(Error::Deadline);
                    }
                    let len = encoder.bytes.len();
                    // Exported records retain their allocation through every custody alias.
                    if (encoder.bytes.capacity() == len || encoder.replace_buffer(len))
                        && let Some(reservation) = encoder.reservation.promote()
                    {
                        return Ok((encoder.bytes, reservation));
                    }
                }
                Err(failure) if failure.is_io() && encoder.retry_exact => {}
                Err(failure) => return Err(error(failure.to_string())),
            }
        }

        // Retry exact only after all provisional bytes of this capture have retired.
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        let mut count = Count(0);
        serde_json::to_writer(&mut count, value).map_err(|e| error(e.to_string()))?;
        let admitted = self.account.reserve_until(count.0, true, Some(deadline));
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        let reservation = match admitted {
            Ok(reservation) => reservation,
            Err(Error::ResourceLimit(_)) => {
                let failure = Error::resource("native capture storage byte budget exhausted");
                self.close(failure.clone());
                return Err(failure);
            }
            Err(failure) => return Err(failure),
        };
        let mut encoded = Vec::with_capacity(count.0);
        serde_json::to_writer(&mut encoded, value).map_err(|e| error(e.to_string()))?;
        Ok((encoded, reservation))
    }
    pub async fn publish<T: Serialize + Sync>(
        &self,
        kind: &str,
        episode: &str,
        game: &str,
        value: &T,
        release_key: Option<(String, String, String)>,
        deadline: f64,
    ) -> Result<()> {
        self.publish_with_training(kind, episode, game, value, release_key, deadline, None, None).await
    }
    pub(super) async fn publish_action(&self, action: &Arc<OwnedAction>, deadline: f64) -> Result<()> {
        let (kind, episode, game) = action.routing();
        self.publish_original(kind, episode, game, action.as_ref(), None, deadline, None, None,
            Some(Arc::downgrade(action))).await
    }
    pub(crate) async fn publish_with_training<T: Serialize + Sync>(
        &self,
        kind: &str,
        episode: &str,
        game: &str,
        value: &T,
        release_key: Option<(String, String, String)>,
        deadline: f64,
        training: Option<Arc<Episode>>,
        http_diagnostics: Option<TerminalHttpDiagnostics>,
    ) -> Result<()> {
        self.publish_original(kind, episode, game, value, release_key, deadline, training,
            http_diagnostics, None).await
    }
    async fn publish_original<T: Serialize + Sync>(
        &self,
        kind: &str,
        episode: &str,
        game: &str,
        value: &T,
        release_key: Option<(String, String, String)>,
        deadline: f64,
        mut training: Option<Arc<Episode>>,
        http_diagnostics: Option<TerminalHttpDiagnostics>,
        mut action_capture: Option<Weak<OwnedAction>>,
    ) -> Result<()> {
        super::super::requests::wait_duration(deadline)?;
        let prepared = self.encode_capture(value, deadline)?;
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        let transport_sha256 = format!("{:x}", Sha256::digest(&prepared.0));
        let answer = Arc::new(StorageAnswer::default());
        let mut owned = Some((Array1::from_vec(prepared.0), prepared.1));
        let mut admission = super::super::metrics::StorageAdmission::new(&self.metrics);
        loop {
            let wake = self.async_wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            {
                let mut s = self.state.lock().unwrap();
                if let Some(e) = &s.closed {
                    return Err(e.clone());
                }
                if monotonic() >= deadline {
                    return Err(Error::Deadline);
                }
                if s.entries.len() < self.capacity {
                    let id = s.next;
                    s.next += 1;
                    let (array, _reservation) = owned.take().unwrap();
                    let record = Arc::new(Record {
                        id,
                        kind: kind.into(),
                        episode: episode.into(),
                        game: game.into(),
                        array,
                        transport_sha256: transport_sha256.clone(),
                        release_key: release_key.clone(),
                        training: training.take(),
                        http_diagnostics,
                        action_capture: Mutex::new(
                            action_capture.take().filter(|_| self.native_rollout_active()),
                        ),
                        _reservation,
                    });
                    s.entries.insert(
                        id,
                        Entry {
                            record,
                            answer: answer.clone(),
                            exported: false,
                            deadline,
                            enqueued_ns: self.metrics.now(),
                        },
                    );
                    s.queue.push_back(id);
                    admission.admitted();
                    self.observe_queue(&s);
                    self.notify();
                    break;
                }
            }
            if monotonic() >= deadline {
                return Err(Error::Deadline);
            }
            let _ =
                tokio::time::timeout(super::super::requests::wait_duration(deadline)?, wake).await;
        }
        answer.wait(deadline).await
    }
    fn pop_ready(&self, s: &mut State, deadline: f64, before_terminal: bool) -> Result<Option<Arc<Record>>> {
        if let Some(e) = &s.closed {
            return Err(e.clone());
        }
        let now = monotonic();
        if now >= deadline {
            return Err(Error::Deadline);
        }
        let Some(id) = s.queue.front() else { return Ok(None); };
        let entry = s.entries.get_mut(id).expect("queued original");
        if now >= entry.deadline {
            return Err(Error::Deadline);
        }
        if before_terminal && entry.record.kind == "terminal" {
            return Ok(None);
        }
        s.queue.pop_front();
        entry.exported = true;
        self.metrics.interval("storage_queue_age", entry.enqueued_ns, self.metrics.now());
        let record = entry.record.clone();
        self.observe_queue(s);
        Ok(Some(record))
    }
    pub fn next(&self, deadline: f64) -> Result<Arc<Record>> {
        super::super::requests::wait_duration(deadline)?;
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(record) = self.pop_ready(&mut s, deadline, false)? {
                return Ok(record);
            }
            s = self.wake.wait_timeout(s, super::super::requests::wait_duration(deadline)?).unwrap().0;
        }
    }
    fn next_before_terminal(&self, deadline: f64) -> Result<Option<Arc<Record>>> {
        self.pop_ready(&mut self.state.lock().unwrap(), deadline, true)
    }
    fn ack_prepared(&self, records: &[Arc<Record>]) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        if let Some(e) = &s.closed {
            return Err(e.clone());
        }
        ensure(records.windows(2).all(|p| p[0].id < p[1].id), "prepared receipt order differs")?;
        let now = monotonic();
        for record in records {
            let entry = s.entries.get(&record.id).ok_or_else(|| error("missing prepared receipt"))?;
            ensure(entry.exported && Arc::ptr_eq(&entry.record, record), "prepared receipt original differs")?;
            if now >= entry.deadline {
                return Err(Error::Deadline);
            }
        }
        for record in records {
            s.entries.remove(&record.id).expect("prepared original").answer.finish(Ok(()));
        }
        self.observe_queue(&s);
        self.notify();
        Ok(())
    }
    pub fn ack(&self, record: &Arc<Record>, failure: Option<String>) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        let entry = s
            .entries
            .get(&record.id)
            .ok_or_else(|| error("duplicate or foreign storage acknowledgement"))?;
        ensure(
            entry.exported && Arc::ptr_eq(&entry.record, record),
            "storage acknowledgement requires same immutable reservation",
        )?;
        let expired = failure.is_none() && monotonic() >= entry.deadline;
        let entry = s.entries.remove(&record.id).unwrap();
        self.observe_queue(&s);
        if expired {
            entry.answer.finish(Err(Error::Deadline));
            self.notify();
            return Err(Error::Deadline);
        }
        if let Some(failure) = failure {
            let e = error(format!("writer failed: {failure}"));
            entry.answer.finish(Err(e.clone()));
            s.closed = Some(e.clone());
            for (_, entry) in s.entries.drain() {
                entry.answer.finish(Err(e.clone()));
            }
            s.queue.clear();
        } else {
            entry.answer.finish(Ok(()));
        }
        self.notify();
        Ok(())
    }
    pub fn close(&self, reason: Error) {
        let mut s = self.state.lock().unwrap();
        if s.closed.is_none() {
            s.closed = Some(reason.clone());
        }
        for (_, entry) in s.entries.drain() {
            entry.answer.finish(Err(reason.clone()));
        }
        s.queue.clear();
        self.notify();
    }
}
// Only the dedicated writer uses this scratch budget, within its existing
// workspace reservation. It issues no recording reservation of its own.
#[pyclass(frozen, name = "_JournalFrameScratch", module = "gf_native")]
pub struct JournalFrameScratch {
    budget: Arc<Budget>,
}

pub(super) type FrameMetadata = (u8, String, u64, usize, (usize, String), String, String);

#[derive(Deserialize)]
#[serde(transparent)]
struct CaptureKey<'a>(#[serde(borrow)] Cow<'a, str>);

fn expect_delimiter(raw: &[u8], delimiter: u8) -> Result<&[u8]> {
    let mut rest = raw;
    while matches!(rest.first(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        rest = &rest[1..];
    }
    rest.strip_prefix(&[delimiter])
        .ok_or_else(|| Error::contract("invalid pending frame member delimiter"))
}

fn capture_frame<'a>(raw: &'a [u8], control: &Control) -> Result<&'a [u8]> {
    let mut rest = expect_delimiter(raw, b'{')?;
    loop {
        control.check()?;
        let mut keys = serde_json::Deserializer::from_slice(rest).into_iter::<CaptureKey>();
        let key = keys
            .next()
            .ok_or_else(|| Error::contract("missing pending frame"))?
            .map_err(|error| Error::contract(error.to_string()))?;
        let value = expect_delimiter(&rest[keys.byte_offset()..], b':')?;
        let mut items = serde_json::Deserializer::from_slice(value).into_iter::<IgnoredAny>();
        items
            .next()
            .ok_or_else(|| Error::contract("missing pending member value"))?
            .map_err(|error| Error::contract(error.to_string()))?;
        let end = items.byte_offset();
        if key.0 == "frame" {
            // The typed candidate parser still owns the whole record,
            // including duplicate frame fields and malformed suffixes.
            return Ok(&value[..end]);
        }
        rest = expect_delimiter(&value[end..], b',')?;
    }
}

pub(super) fn checked_journal_frame(
    raw: &[u8],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<(FrameMetadata, BudgetVec<u8>)> {
    let document = json::parse(capture_frame(raw, control)?, budget, control)?;
    let frame = &document.root;
    // This verifier uses the historical corrected frame layout pin. The
    // operational Sacrifice (or earlier shared) action contract belongs to
    // the candidate table, not the unchanged frame hash binding.
    card_instance::frame(frame, budget, control)?;
    let boundary = frame.require("boundary")?;
    let meta = (
        frame.require("actor")?.integer(1, 2)? as u8,
        frame.require("episode")?.string()?.to_owned(),
        frame.require("revision")?.integer(0, u64::MAX as i128)? as u64,
        frame.require("interval_start")?.integer(0, i32::MAX as i128)? as usize,
        (
            boundary.require("end")?.integer(0, i32::MAX as i128)? as usize,
            boundary.require("prefix_hash")?.string()?.to_owned(),
        ),
        frame.require("interval_start_hash")?.string()?.to_owned(),
        frame.require("frame_hash")?.string()?.to_owned(),
    );
    let payload = json::canonical(frame.require("payload")?, budget, control)?;
    Ok((meta, payload))
}

#[pyclass(frozen)]
pub struct StorageBundle {
    pub(crate) record: Arc<Record>,
}
#[pymethods]
impl StorageBundle {
    /// The trusted Python writer calls this on an original terminal bundle to
    /// compare its accepted Journal and Replay facts before the collection seals.
    #[pyo3(name = "_training_episode")]
    fn training_episode(&self) -> Option<PlayerPpoEpisode> {
        self.record.training.as_ref().map(|episode| PlayerPpoEpisode::new(episode.clone()))
    }
    /// Opaque HTTP diagnostic observations; never encoded in the capture.
    fn _http_diagnostics(&self) -> Option<(u64, u64, u64, bool)> {
        self.record.http_diagnostics.map(|info| (
            info.facts.responses, info.facts.error_responses,
            info.facts.transport_errors, info.abnormal,
        ))
    }
    #[getter]
    fn id(&self) -> u64 {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        self.record.id
    }
    #[getter]
    fn kind(&self) -> &str {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        &self.record.kind
    }
    #[getter]
    fn episode(&self) -> &str {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        &self.record.episode
    }
    #[getter]
    fn game(&self) -> &str {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        &self.record.game
    }
    #[getter]
    fn transport_sha256(&self) -> &str {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        &self.record.transport_sha256
    }
    #[getter]
    fn release_key(&self) -> Option<(String, String, String)> {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        self.record.release_key.clone()
    }
    #[getter]
    fn reserved_bytes(&self) -> usize {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        self.record.array.len()
    }
    #[staticmethod]
    fn _journal_frame_scratch(workspace_bytes: usize) -> JournalFrameScratch {
        JournalFrameScratch {
            budget: Budget::new(workspace_bytes),
        }
    }
    fn _journal_frame<'py>(
        &self,
        py: Python<'py>,
        scratch: &JournalFrameScratch,
        deadline: f64,
    ) -> PyResult<(FrameMetadata, Bound<'py, PyBytes>)> {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        ensure(self.record.kind == "pending", "writer frame requires pending")
            .map_err(crate::python_error)?;
        let original = self.record.array.as_slice().expect("own contiguous capture");
        let control = Control::new(deadline).map_err(crate::python_error)?;
        let budget = &scratch.budget;
        let (meta, payload, copy) = py
            .detach(|| {
                let (meta, payload) = checked_journal_frame(original, budget, &control)?;
                let copy = budget.reserve(payload.len())?;
                control.check()?;
                Ok((meta, payload, copy))
            })
            .map_err(crate::python_error)?;
        let owned = PyBytes::new(py, &payload);
        drop(payload);
        drop(copy);
        control.check().map_err(crate::python_error)?;
        Ok((meta, owned))
    }
    /// Independently validate and encode the writer's original pending candidate table.
    /// The caller reserves the codec Vec and Python bytes using the same recording account.
    fn _journal_candidates<'py>(
        &self,
        py: Python<'py>,
        max_positions: usize,
        output_bytes: usize,
    ) -> PyResult<(
        &str, &str, usize, usize, usize, Bound<'py, PyBytes>, Bound<'py, PyBytes>, String,
    )> {
        self.record._reservation.account.metrics.increment("pyo3_other", 1);
        ensure(self.record.kind == "pending", "writer candidates require pending")
            .map_err(crate::python_error)?;
        let whole = self.record.array.as_slice().expect("own contiguous capture");
        let checked = py.detach(|| super::super::journal_candidates::check(whole, max_positions, output_bytes))
            .map_err(crate::python_error)?;
        Ok((
            checked.frame_hash, checked.table_hash, checked.selected, checked.count, checked.elements,
            PyBytes::new(py, &checked.action), PyBytes::new(py, &checked.payload), checked.sha256,
        ))
    }
    fn view<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        slf.borrow().record._reservation.account.metrics.increment("pyo3_other", 1);
        let record = slf.borrow().record.clone();
        // Base owns StorageBundle -> Arc<Record> -> Array + SAME reservation.
        let array = unsafe { PyArray1::borrow_from_array(&record.array, slf.into_any()) };
        array.call_method1("setflags", (false,))?;
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn journal_frame_reads_the_first_decoded_key_at_the_top_level() {
        let control = Control::new(monotonic() + 5.).unwrap();
        let raw = br#" { "other": {"frame": "inside", "quote": "x\"y"}, "fra\u006de" : {"expected":true}, invalid"#;
        let frame = capture_frame(raw, &control).unwrap();
        assert_eq!(frame, br#" {"expected":true}"#);
        for raw in [
            br#"{}"#.as_slice(),
            br#"{"frame" {}}"#,
            br#"{"other":{}, "frame"}"#,
            br#"{"other": [1 2], "frame": {}}"#,
            br#"{"frame": {"bad": }}"#,
        ] {
            assert!(capture_frame(raw, &control).is_err(), "{raw:?}");
        }
    }

    #[test]
    fn journal_frame_and_whole_candidate_checks_jointly_bind_the_original() {
        let doc: json::Value = serde_json::from_slice(include_bytes!(
            "../../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let decision = &doc["episodes"][0]["decisions"][0];
        let table = &decision["generated"]["table"];
        let count = table["kinds"].as_array().unwrap().len();
        let elements = table["slots"].as_array().unwrap().len();
        let table_bytes = ((2 * count).next_multiple_of(8) + 36 * count).next_multiple_of(8)
            + 8 * (count + 1) + elements;
        let frame = &decision["frame"];
        let record = format!(
            "{{\"frame\":{frame},\"generated\":{{\"table\":{table}}},\"selected\":0}}"
        );
        let budget = Budget::new(32 * 1024 * 1024);
        let control = Control::new(monotonic() + 5.).unwrap();
        let (meta, payload) = checked_journal_frame(record.as_bytes(), &budget, &control).unwrap();
        let candidates = super::super::super::journal_candidates::check(
            record.as_bytes(), 18, table_bytes,
        )
        .unwrap();
        assert_eq!(meta.6, candidates.frame_hash);
        assert_eq!(&*payload, serde_json::to_vec(&frame["payload"]).unwrap());
        drop(payload);
        assert_eq!(budget.used(), 0);

        for (suffix, expected) in [
            (format!(r#", "fra\u006de": {frame}}}"#), "duplicate field"),
            (", broken}".into(), "key must be a string"),
        ] {
            let invalid = format!("{}{suffix}", record.strip_suffix('}').unwrap());
            assert!(checked_journal_frame(invalid.as_bytes(), &budget, &control).is_ok());
            let error = super::super::super::journal_candidates::check(
                invalid.as_bytes(), 18, table_bytes,
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(budget.used(), 0);
        }
        let expired = Control::new(monotonic() - 1.).unwrap();
        assert!(matches!(checked_journal_frame(record.as_bytes(), &budget, &expired), Err(Error::Deadline)));
        let tiny = Budget::new(64 * 1024);
        assert!(matches!(checked_journal_frame(record.as_bytes(), &tiny, &control), Err(Error::ResourceLimit(_))));
        assert_eq!(tiny.used(), 0);
    }

    #[test]
    #[ignore = "requires pairs of frame captures and independent Python payloads from a sealed sample"]
    fn journal_frame_sealed_sample_payload_and_fault_parity() {
        let base = std::path::PathBuf::from(std::env::var_os("GFB_FRAME_PARITY_DIR").unwrap());
        let budget = Budget::new(32 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        for name in ["early", "late"] {
            let captures = std::fs::read_to_string(base.join(format!("{name}.captures"))).unwrap();
            let expected = std::fs::read_to_string(base.join(format!("{name}.payloads"))).unwrap();
            assert_eq!(captures.lines().count(), expected.lines().count());
            for (index, (raw, expected)) in captures.lines().zip(expected.lines()).enumerate() {
                let result = checked_journal_frame(raw.as_bytes(), &budget, &control);
                if expected == "null" {
                    assert!(matches!(result, Err(Error::Contract(_))), "{name} {index}");
                } else {
                    let (_, bytes) = result.unwrap_or_else(|e| panic!("{name} {index}: {e}"));
                    assert!(&*bytes == expected.as_bytes(), "{name} {index}");
                }
                assert_eq!(budget.used(), 0);
            }
        }
    }

    struct Probe {
        calls: AtomicUsize,
        delayed_call: usize,
    }
    impl Serialize for Probe {
        fn serialize<S: serde::Serializer>(
            &self,
            serializer: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == self.delayed_call {
                std::thread::sleep(Duration::from_millis(30));
            }
            serializer.serialize_u8(1)
        }
    }
    fn run(future: impl std::future::Future<Output = ()>) {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future);
    }
    #[test]
    fn training_limit_is_subordinate_to_the_same_recording_allocation() {
        let mailbox = Mailbox::new_with_training_limit(100, 1, 60, Arc::new(Default::default())).unwrap();
        let account = mailbox.account().account;
        let native_capture = account.reserve(30, true).unwrap();
        let native_training = account.reserve_training(60).unwrap();
        assert_eq!(mailbox.used(), 90);
        assert!(!account.training_fits(1));
        assert!(account.reserve_training(1).is_err());
        assert!(account.reserve(11, false).is_err());
        drop(native_capture);
        assert_eq!(mailbox.used(), 60);
        drop(native_training);
        assert_eq!(mailbox.used(), 0);
        let other = account.reserve(70, false).unwrap();
        assert!(account.training_fits(30));
        assert!(!account.training_fits(31));
        drop(other);
    }
    #[test]
    fn retained_recording_admission_has_priority_over_provisional_encoding() {
        let mailbox =
            Mailbox::new_with_training_limit(100, 1, 35, Arc::new(Default::default())).unwrap();
        let account = mailbox.account().account;
        let mut retained = account.reserve(35, false).unwrap();
        let mut provisional = account.try_provision(50).unwrap();
        assert_eq!(mailbox.used(), 50);
        assert!(account.training_fits(30));
        let deadline = monotonic() + 2.;
        assert!(matches!(
            account.reserve_until(66, true, Some(deadline)),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            account.reserve_training(36),
            Err(Error::ResourceLimit(_))
        ));

        let (ready, admitted) = std::sync::mpsc::channel();
        let worker_account = account.clone();
        let waiter = std::thread::spawn(move || {
            ready.send(worker_account.reserve_training(30)).unwrap();
        });
        while account.ledger.lock().unwrap().waiters == 0 {
            assert!(monotonic() < deadline);
            std::thread::yield_now();
        }
        assert!(admitted.try_recv().is_err());
        assert!(account.try_provision(1).is_none());
        assert!(provisional.promote().is_none());

        // Ordinary refunds must wake the waiter even while the encoder is live.
        retained.resize(20).unwrap();
        let training = admitted
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(account.ledger.lock().unwrap().peak, 100);
        assert_eq!(mailbox.used(), 80);
        waiter.join().unwrap();
        drop((training, retained, provisional));
        assert_eq!(mailbox.used(), 0);
    }

    #[test]
    fn temporary_capacity_wait_obeys_the_original_deadline_without_closing_the_mailbox() {
        let mailbox = Mailbox::new(20, 1).unwrap();
        let account = mailbox.account().account;
        let temporary = account.try_provision(19).unwrap();
        run(async {
            assert!(matches!(
                mailbox
                    .publish("pending", "e", "g", &1234, None, monotonic() + 0.02)
                    .await,
                Err(Error::Deadline)
            ));
        });
        assert!(mailbox.state.lock().unwrap().closed.is_none());
        assert_eq!(account.ledger.lock().unwrap().waiters, 0);
        assert_eq!(mailbox.used(), 19);
        drop(temporary);
        assert_eq!(mailbox.used(), 0);
    }
    #[test]
    fn exact_fallback_reuses_budget_after_growth_or_compaction_is_refused() {
        let value = serde_json::json!({"x": 1});
        let expected = serde_json::to_vec(&value).unwrap();
        assert_eq!(expected.len(), 7);
        // At seven bytes the first growth is refused. At fourteen bytes the
        // eight-byte buffer fits and the seven-byte replacement does not.
        for (limit, peak) in [(7, 7), (14, 8)] {
            let mailbox = Mailbox::new(limit, 1).unwrap();
            let (encoded, reservation) = mailbox.encode_capture(&value, monotonic() + 2.).unwrap();
            assert_eq!(encoded, expected);
            assert_eq!(encoded.capacity(), expected.len());
            assert_eq!(mailbox.used(), expected.len());
            assert_eq!(mailbox.account.ledger.lock().unwrap().peak, peak);
            assert!(mailbox.state.lock().unwrap().closed.is_none());
            drop((encoded, reservation));
            assert_eq!(mailbox.used(), 0);
        }
    }
    #[test]
    fn ingress_admission_wait_excludes_original_capture_preparation() {
        use std::sync::atomic::{AtomicUsize,Ordering};
        use super::super::super::metrics::{Metrics,test_clock,set_test_clock};
        struct TimedEncoding(AtomicUsize);
        impl Serialize for TimedEncoding {
            fn serialize<S:serde::Serializer>(&self,serializer:S)->std::result::Result<S::Ok,S::Error>{
                let pass=self.0.fetch_add(1,Ordering::Relaxed)+1;
                set_test_clock(pass as u64 * 1_000);
                serializer.serialize_u64(1)
            }
        }
        let _clock=test_clock(0);
        let metrics=Arc::new(Metrics::default());
        metrics.begin(serde_json::json!({"batch_id":"queue"}),0).unwrap();
        let mailbox=Mailbox::new_monitored(4096,1,metrics.clone()).unwrap();
        run(async {
            let producer=mailbox.clone();
            let task=tokio::spawn(async move {
                producer.publish("pending","episode","game",&TimedEncoding(AtomicUsize::new(0)),None,monotonic()+5.).await
            });
            tokio::task::yield_now().await;
            let record=mailbox.next(monotonic()+5.).unwrap();
            mailbox.ack(&record,None).unwrap();
            task.await.unwrap().unwrap();
        });
        let data=metrics.end(3_000,true).unwrap();
        assert_eq!(data["histograms"]["storage_admission_wait"]["count"],1);
        assert_eq!(data["histograms"]["storage_admission_wait"]["min_ns"],0);
    }
    #[test]
    fn monitored_ingress_uses_real_enqueue_export_ack_boundaries() {
        let metrics=Arc::new(super::super::super::metrics::Metrics::default());
        metrics.begin(serde_json::json!({"batch_id":"ingress"}),0).unwrap();
        let mailbox=Mailbox::new_monitored(4096,1,metrics.clone()).unwrap();
        run(async {
            let producer=mailbox.clone();
            let task=tokio::spawn(async move {producer.publish("pending","episode","game",&serde_json::json!({"x":1}),None,monotonic()+5.).await});
            tokio::task::yield_now().await;
            assert!(!mailbox.measurement_quiet());
            let record=mailbox.next(monotonic()+5.).unwrap();
            mailbox.ack(&record,None).unwrap();
            task.await.unwrap().unwrap();
            assert!(mailbox.measurement_quiet());
        });
        let data=metrics.end(super::super::super::metrics::monotonic_ns(),true).unwrap();
        assert_eq!(data["counters"]["writer_admissions"],1);
        assert_eq!(data["histograms"]["storage_queue_age"]["count"],1);
        assert!(data["queue"]["queued_bytes_peak"].as_u64().unwrap()>0);
        assert_eq!(data["queue"]["queued_bytes_peak"],data["queue"]["in_service_bytes_peak"]);
    }
    #[test]
    fn expired_publication_never_counts_encodes_or_enqueues() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let probe = Probe {
            calls: AtomicUsize::new(0),
            delayed_call: 0,
        };
        run(async {
            assert!(matches!(
                mailbox
                    .publish("terminal", "e", "g", &probe, None, monotonic() - 1.)
                    .await,
                Err(Error::Deadline)
            ));
        });
        assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        assert!(mailbox.state.lock().unwrap().entries.is_empty());
        assert_eq!(mailbox.used(), 0);
    }
    #[test]
    fn a_timely_publication_serializes_once_and_holds_exact_original_bytes() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let probe = Probe {
            calls: AtomicUsize::new(0),
            delayed_call: 0,
        };
        let consumer = mailbox.clone();
        let thread = std::thread::spawn(move || {
            let record = consumer.next(monotonic() + 2.).unwrap();
            assert_eq!(record.array.as_slice().unwrap(), b"1");
            assert_eq!(record.transport_sha256, format!("{:x}", Sha256::digest(b"1")));
            assert_eq!(consumer.used(), 1);
            consumer.ack(&record, None).unwrap();
            record
        });
        run(async {
            mailbox
                .publish("pending", "e", "g", &probe, None, monotonic() + 2.)
                .await
                .unwrap();
        });
        assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
        let held = thread.join().unwrap();
        assert_eq!(mailbox.used(), 1);
        drop(held);
        assert_eq!(mailbox.used(), 0);
    }
    #[test]
    fn fallback_count_and_encode_cannot_spend_then_renew_publication_cutoff() {
        for delayed_call in [1, 2, 3] {
            let mailbox = Mailbox::new(1, 1).unwrap();
            let probe = Probe {
                calls: AtomicUsize::new(0),
                delayed_call,
            };
            run(async {
                assert!(matches!(
                    mailbox
                        .publish("terminal", "e", "g", &probe, None, monotonic() + 0.01)
                        .await,
                    Err(Error::Deadline)
                ));
            });
            assert_eq!(probe.calls.load(Ordering::SeqCst), delayed_call);
            assert!(mailbox.state.lock().unwrap().entries.is_empty());
            assert_eq!(mailbox.used(), 0);
        }
    }
    #[test]
    fn ready_success_after_original_cutoff_is_not_successful_settlement() {
        let answer = StorageAnswer::default();
        answer.finish(Ok(()));
        run(async {
            assert!(matches!(
                answer.wait(monotonic() - 1.).await,
                Err(Error::Deadline)
            ));
        });
    }
    #[test]
    fn late_ack_refuses_success_but_actual_consumer_retains_original_reservation() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let consumer = mailbox.clone();
        let thread = std::thread::spawn(move || {
            let record = consumer.next(monotonic() + 1.).unwrap();
            std::thread::sleep(Duration::from_millis(30));
            assert!(matches!(consumer.ack(&record, None), Err(Error::Deadline)));
            record
        });
        run(async {
            assert!(matches!(
                mailbox
                    .publish("terminal", "e", "g", &1, None, monotonic() + 0.01)
                    .await,
                Err(Error::Deadline)
            ));
        });
        let held = thread.join().unwrap();
        mailbox.close(Error::Deadline);
        assert!(mailbox.used() > 0);
        drop(held);
        assert_eq!(mailbox.used(), 0);
    }
    #[test]
    fn prepared_receipts_fail_atomically_and_ready_drain_stops_at_terminal() {
        let mailbox = Mailbox::new(1024, 4).unwrap();
        let deadline = monotonic() + 5.;
        let mut threads = vec![];
        for (episode, kind) in [("e1", "pending"), ("e2", "pending"), ("e3", "terminal")] {
            let shared = mailbox.clone();
            threads.push(std::thread::spawn(move || run(async move {
                assert!(shared.publish(kind, episode, "g", &1, None, deadline).await.is_err());
            })));
            while mailbox.state.lock().unwrap().queue.len() < threads.len() {
                assert!(monotonic() < deadline);
                std::thread::yield_now();
            }
        }
        let first = mailbox.next(deadline).unwrap();
        let second = mailbox.next_before_terminal(deadline).unwrap().unwrap();
        assert!(mailbox.next_before_terminal(deadline).unwrap().is_none());
        assert_eq!(mailbox.next(deadline).unwrap().kind, "terminal");
        assert!(mailbox.ack_prepared(&[first.clone(), first.clone()]).is_err());
        mailbox.state.lock().unwrap().entries.get_mut(&second.id).unwrap().deadline = monotonic() - 1.;
        assert!(matches!(mailbox.ack_prepared(&[first.clone(), second.clone()]), Err(Error::Deadline)));
        assert!(mailbox.state.lock().unwrap().entries[&first.id].answer.result.lock().unwrap().is_none());
        mailbox.close(Error::Deadline);
        for worker in threads {
            worker.join().unwrap();
        }
        drop((first, second));
        assert_eq!(mailbox.used(), 0);
    }

    #[test]
    fn split_recording_reservation_preserves_the_total_until_each_alias_retires() {
        let mailbox = Mailbox::new(8, 1).unwrap();
        let account = mailbox.account().account;
        let mut members = account.reserve(8, true).unwrap();
        let evidence = members.split(3).unwrap();
        assert_eq!(members.bytes(), 5);
        assert_eq!(evidence.bytes(), 3);
        assert_eq!(mailbox.used(), 8);
        assert!(account.reserve(1, false).is_err());
        drop(evidence);
        assert_eq!(mailbox.used(), 5);
        drop(members);
        assert_eq!(mailbox.used(), 0);
    }

    #[test]
    fn timely_original_ack_still_succeeds() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let consumer = mailbox.clone();
        let thread = std::thread::spawn(move || {
            let record = consumer.next(monotonic() + 1.).unwrap();
            consumer.ack(&record, None).unwrap();
        });
        run(async {
            mailbox
                .publish("pending", "e", "g", &1, None, monotonic() + 1.)
                .await
                .unwrap();
        });
        thread.join().unwrap();
        assert_eq!(mailbox.used(), 0);
    }
    #[test]
    fn queued_original_cannot_export_under_expired_consumer_or_original_cutoff() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let producer = mailbox.clone();
        let thread = std::thread::spawn(move || {
            run(async {
                assert!(matches!(
                    producer
                        .publish("terminal", "e", "g", &1, None, monotonic() + 0.05)
                        .await,
                    Err(Error::Deadline)
                ));
            });
        });
        let observation_deadline = monotonic() + 1.;
        while mailbox.state.lock().unwrap().entries.is_empty() {
            assert!(monotonic() < observation_deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(matches!(
            mailbox.next(monotonic() - 1.),
            Err(Error::Deadline)
        ));
        assert!(
            !mailbox
                .state
                .lock()
                .unwrap()
                .entries
                .values()
                .next()
                .unwrap()
                .exported
        );
        thread.join().unwrap();
        assert!(matches!(
            mailbox.next(monotonic() + 1.),
            Err(Error::Deadline)
        ));
        assert!(
            !mailbox
                .state
                .lock()
                .unwrap()
                .entries
                .values()
                .next()
                .unwrap()
                .exported
        );
        mailbox.close(Error::Deadline);
        assert_eq!(mailbox.used(), 0);
    }
    #[test]
    fn ready_failure_keeps_its_failure_identity_after_cutoff() {
        let answer = StorageAnswer::default();
        answer.finish(Err(Error::resource("actual recording failure")));
        run(async {
            let failure = answer.wait(monotonic() - 1.).await.unwrap_err();
            assert!(failure.to_string().contains("actual recording failure"));
        });
    }

    fn owned_ack() -> Arc<OwnedAction> {
        Arc::new(OwnedAction::Ack(super::super::capture::OwnedAck {
            schema: "rust-player-ack-v1",
            game: "g".into(),
            episode: "e".into(),
            actor: 1,
            command: serde_json::json!({"action": {"type": "pray"}}),
            acknowledgement: serde_json::json!({"status":"active"}),
        }))
    }

    #[test]
    fn captured_action_bytes_and_weak_alias_follow_native_and_legacy_custody() {
        let mailbox = Mailbox::new(4096, 1).unwrap();
        let _lease = mailbox.claim_native_rollout().unwrap();
        let action = owned_ack();
        let original = serde_json::to_vec(action.as_ref()).unwrap();
        for legacy in [false, true] {
            run(async {
                let deadline = monotonic() + 2.;
                let (result, ()) = tokio::join!(mailbox.publish_action(&action, deadline), async {
                    let record = mailbox.next(deadline).unwrap();
                    assert_eq!(record.array.as_slice().unwrap(), original.as_slice());
                    assert_eq!(
                        record.transport_sha256,
                        format!("{:x}", Sha256::digest(&original))
                    );
                    if legacy {
                        record.discard_action_capture();
                    } else {
                        let typed = record.take_action_capture().unwrap().upgrade().unwrap();
                        assert!(Arc::ptr_eq(&typed, &action));
                    }
                    assert!(record.take_action_capture().is_none());
                    mailbox.ack(&record, None).unwrap();
                });
                result.unwrap();
            });
        }
        assert_eq!(mailbox.used(), 0);
    }

    #[test]
    fn after_native_retirement_a_blocked_action_is_only_published_as_raw_bytes() {
        use std::future::{Future, poll_fn};
        use std::task::Poll;

        let mailbox = Mailbox::new(4096, 1).unwrap();
        let lease = mailbox.claim_native_rollout().unwrap();
        let action = owned_ack();
        run(async {
            let deadline = monotonic() + 2.;
            let mut occupied = Box::pin(mailbox.publish("pending", "e", "g", &0, None, deadline));
            poll_fn(|cx| {
                assert!(occupied.as_mut().poll(cx).is_pending());
                Poll::Ready(())
            })
            .await;
            let mut waiting = Box::pin(mailbox.publish_action(&action, deadline));
            poll_fn(|cx| {
                assert!(waiting.as_mut().poll(cx).is_pending());
                Poll::Ready(())
            })
            .await;

            drop(lease);
            let original = mailbox.next(deadline).unwrap();
            mailbox.ack(&original, None).unwrap();
            occupied.await.unwrap();
            let (result, ()) = tokio::join!(waiting, async {
                let record = mailbox.next(deadline).unwrap();
                assert!(record.take_action_capture().is_none());
                assert_eq!(
                    record.array.as_slice().unwrap(),
                    serde_json::to_vec(action.as_ref()).unwrap().as_slice()
                );
                mailbox.ack(&record, None).unwrap();
            });
            result.unwrap();
        });
        assert_eq!(mailbox.used(), 0);
    }
}
