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
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
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
pub struct Reservation {
    bytes: usize,
    used: Arc<AtomicUsize>,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::AcqRel);
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
    used: Arc<AtomicUsize>,
    limit: usize,
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
            used: Arc::new(AtomicUsize::new(0)),
            limit,
            capacity,
        }))
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
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
        // Encoding workspace is temporary actor state. Count first, reserve exact
        // retained bytes before allocation; one lease follows every exported view.
        let mut count = Count(0);
        serde_json::to_writer(&mut count, value).map_err(|e| error(e.to_string()))?;
        let bytes = count.0;
        if self
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|n| *n <= self.limit)
            })
            .is_err()
        {
            let failure = Error::resource("native capture storage byte budget exhausted");
            self.close(failure.clone());
            return Err(failure);
        }
        let reservation = Reservation {
            bytes,
            used: self.used.clone(),
        };
        let mut encoded = Vec::with_capacity(bytes);
        serde_json::to_writer(&mut encoded, value).map_err(|e| error(e.to_string()))?;
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
        let entry = s.entries.remove(&record.id).unwrap();
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
