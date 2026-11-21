//! Immutable capture transport. Acceptance transfers custody, never proves persistence.
use super::super::{ensure, error, training::{Episode, PlayerPpoEpisode}};
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
    training: usize,
    peak: usize,
}
pub struct Account {
    metrics: Arc<super::super::metrics::Metrics>,
    limit: usize,
    training_limit: usize,
    ledger: Mutex<Ledger>,
}
impl Account {
    fn change(&self, native: bool, training: bool, old: usize, new: usize) -> Result<()> {
        let mut s = self.ledger.lock().unwrap();
        let total = s
            .native
            .checked_add(s.external)
            .and_then(|n| n.checked_sub(old))
            .and_then(|n| n.checked_add(new))
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| Error::resource("recording byte budget exhausted"))?;
        if training {
            s.training = s.training.checked_sub(old)
                .and_then(|n| n.checked_add(new))
                .filter(|n| *n <= self.training_limit)
                .ok_or_else(|| Error::resource("PPO numeric byte budget exhausted"))?;
        }
        if native {
            s.native = s.native - old + new;
        } else {
            s.external = s.external - old + new;
        }
        s.peak = s.peak.max(total);
        Ok(())
    }
    fn reserve(self: &Arc<Self>, bytes: usize, native: bool) -> Result<Reservation> {
        self.change(native, false, 0, bytes)?;
        Ok(Reservation {
            bytes,
            native,
            training: false,
            account: self.clone(),
        })
    }
    pub(crate) fn reserve_training(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        self.change(true, true, 0, bytes)?;
        Ok(Reservation { bytes, native: true, training: true, account: self.clone() })
    }
    pub(crate) fn training_fits(&self, bytes: usize) -> bool {
        let ledger = self.ledger.lock().unwrap();
        bytes <= self.limit - ledger.native - ledger.external
            && bytes <= self.training_limit - ledger.training
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
}
impl Drop for Reservation {
    fn drop(&mut self) {
        // A refund cannot exceed the admitted total; no allocation or callback.
        self.account.change(self.native, self.training, self.bytes, 0).unwrap();
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
        s.native + s.external
    }
    #[getter]
    fn peak(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        self.account.ledger.lock().unwrap().peak
    }
    #[getter]
    fn native_bytes(&self) -> usize {
        self.account.metrics.increment("pyo3_other", 1);
        self.account.ledger.lock().unwrap().native
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
        r.account
            .change(false, false, r.bytes, bytes)
            .map_err(crate::python_error)?;
        r.bytes = bytes;
        Ok(())
    }
    fn close(&self) {
        self.metrics.increment("pyo3_other", 1);
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
    training: Option<Arc<Episode>>,
    _reservation: Reservation,
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
    pub metrics: Arc<super::super::metrics::Metrics>,
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
            }),
            capacity,
            metrics,
        }))
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
        self.publish_with_training(kind, episode, game, value, release_key, deadline, None).await
    }
    pub(crate) async fn publish_with_training<T: Serialize + Sync>(
        &self,
        kind: &str,
        episode: &str,
        game: &str,
        value: &T,
        release_key: Option<(String, String, String)>,
        deadline: f64,
        mut training: Option<Arc<Episode>>,
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
                self.metrics.interval("storage_queue_age", entry.enqueued_ns, self.metrics.now());
                let record = entry.record.clone();
                self.observe_queue(&s);
                return Ok(record);
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
    fn ingress_admission_wait_excludes_original_count_encode_and_hash_preparation() {
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
