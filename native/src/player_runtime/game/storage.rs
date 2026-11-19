//! Immutable capture transport. Acceptance transfers custody, never proves persistence.
use super::super::{ensure, error};
use crate::{
    budget::monotonic,
    error::{Error, Result},
};
use ndarray::Array1;
use numpy::PyArray1;
use pyo3::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    sync::{Arc, Condvar, Mutex},
};

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
    peak: usize,
}
pub struct Account {
    limit: usize,
    ledger: Mutex<Ledger>,
}
impl Account {
    fn change(&self, native: bool, old: usize, new: usize) -> Result<()> {
        let mut s = self.ledger.lock().unwrap();
        let total = s
            .native
            .checked_add(s.external)
            .and_then(|n| n.checked_sub(old))
            .and_then(|n| n.checked_add(new))
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| Error::resource("recording byte budget exhausted"))?;
        if native {
            s.native = s.native - old + new;
        } else {
            s.external = s.external - old + new;
        }
        s.peak = s.peak.max(total);
        Ok(())
    }
    fn reserve(self: &Arc<Self>, bytes: usize, native: bool) -> Result<Reservation> {
        self.change(native, 0, bytes)?;
        Ok(Reservation {
            bytes,
            native,
            account: self.clone(),
        })
    }
}
pub struct Reservation {
    bytes: usize,
    native: bool,
    account: Arc<Account>,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        // A refund cannot exceed the admitted total; no allocation or callback.
        self.account.change(self.native, self.bytes, 0).unwrap();
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
        self.account.limit
    }
    #[getter]
    fn used(&self) -> usize {
        let s = self.account.ledger.lock().unwrap();
        s.native + s.external
    }
    #[getter]
    fn peak(&self) -> usize {
        self.account.ledger.lock().unwrap().peak
    }
    #[getter]
    fn native_bytes(&self) -> usize {
        self.account.ledger.lock().unwrap().native
    }
    #[getter]
    fn external_bytes(&self) -> usize {
        self.account.ledger.lock().unwrap().external
    }
    fn same_account(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.account, &other.account)
    }
    fn owns(&self, bundle: &StorageBundle) -> bool {
        Arc::ptr_eq(&self.account, &bundle.record._reservation.account)
    }
    fn reserve(&self, bytes: usize) -> PyResult<RecordingLease> {
        Ok(RecordingLease {
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
    lease: Mutex<Option<Reservation>>,
}
#[pymethods]
impl RecordingLease {
    #[getter]
    fn size(&self) -> usize {
        self.lease.lock().unwrap().as_ref().map_or(0, |r| r.bytes)
    }
    fn resize(&self, bytes: usize) -> PyResult<()> {
        let mut s = self.lease.lock().unwrap();
        let r = s
            .as_mut()
            .ok_or_else(|| crate::python_error(error("recording lease closed")))?;
        r.account
            .change(false, r.bytes, bytes)
            .map_err(crate::python_error)?;
        r.bytes = bytes;
        Ok(())
    }
    fn close(&self) {
        self.lease.lock().unwrap().take();
    }
}
pub struct Record {
    pub id: u64,
    pub kind: String,
    pub episode: String,
    pub game: String,
    array: Array1<u8>,
    transport_sha256: String,
    release_key: Option<(String, String, String)>,
    _reservation: Reservation,
}
struct Entry {
    record: Arc<Record>,
    answer: Arc<StorageAnswer>,
    exported: bool,
    deadline: f64,
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
}
impl Mailbox {
    pub fn new(limit: usize, capacity: usize) -> Result<Arc<Self>> {
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
                limit,
                ledger: Mutex::new(Ledger::default()),
            }),
            capacity,
        }))
    }
    pub fn used(&self) -> usize {
        self.account.ledger.lock().unwrap().native
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
    pub async fn publish<T: Serialize + Sync>(
        &self,
        kind: &str,
        episode: &str,
        game: &str,
        value: &T,
        release_key: Option<(String, String, String)>,
        deadline: f64,
    ) -> Result<()> {
        super::super::requests::wait_duration(deadline)?;
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        // Encoding workspace is temporary actor state. Count first, reserve exact
        // retained bytes before allocation; one lease follows every exported view.
        let mut count = Count(0);
        serde_json::to_writer(&mut count, value).map_err(|e| error(e.to_string()))?;
        let bytes = count.0;
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        let reservation = match self.account.reserve(bytes, true) {
            Ok(reservation) => reservation,
            Err(_) => {
                let failure = Error::resource("native capture storage byte budget exhausted");
                self.close(failure.clone());
                return Err(failure);
            }
        };
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        let mut encoded = Vec::with_capacity(bytes);
        serde_json::to_writer(&mut encoded, value).map_err(|e| error(e.to_string()))?;
        if monotonic() >= deadline {
            return Err(Error::Deadline);
        }
        let transport_sha256 = format!("{:x}", Sha256::digest(&encoded));
        let answer = Arc::new(StorageAnswer::default());
        let mut owned = Some((Array1::from_vec(encoded), reservation));
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
                        _reservation,
                    });
                    s.entries.insert(
                        id,
                        Entry {
                            record,
                            answer: answer.clone(),
                            exported: false,
                            deadline,
                        },
                    );
                    s.queue.push_back(id);
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
    pub fn next(&self, deadline: f64) -> Result<Arc<Record>> {
        super::super::requests::wait_duration(deadline)?;
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(e) = &s.closed {
                return Err(e.clone());
            }
            if monotonic() >= deadline {
                return Err(Error::Deadline);
            }
            if let Some(id) = s.queue.front() {
                if monotonic() >= s.entries[id].deadline {
                    return Err(Error::Deadline);
                }
            }
            if let Some(id) = s.queue.pop_front() {
                let entry = s.entries.get_mut(&id).unwrap();
                entry.exported = true;
                return Ok(entry.record.clone());
            }
            if monotonic() >= deadline {
                return Err(Error::Deadline);
            }
            s = self
                .wake
                .wait_timeout(s, super::super::requests::wait_duration(deadline)?)
                .unwrap()
                .0;
        }
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
#[pyclass(frozen)]
pub struct StorageBundle {
    pub(crate) record: Arc<Record>,
}
#[pymethods]
impl StorageBundle {
    #[getter]
    fn id(&self) -> u64 {
        self.record.id
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.record.kind
    }
    #[getter]
    fn episode(&self) -> &str {
        &self.record.episode
    }
    #[getter]
    fn game(&self) -> &str {
        &self.record.game
    }
    #[getter]
    fn transport_sha256(&self) -> &str {
        &self.record.transport_sha256
    }
    #[getter]
    fn release_key(&self) -> Option<(String, String, String)> {
        self.record.release_key.clone()
    }
    #[getter]
    fn reserved_bytes(&self) -> usize {
        self.record.array.len()
    }
    fn view<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<u8>>> {
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
    fn count_and_encode_cannot_spend_then_renew_publication_cutoff() {
        for delayed_call in [1, 2] {
            let mailbox = Mailbox::new(1024, 1).unwrap();
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
}
