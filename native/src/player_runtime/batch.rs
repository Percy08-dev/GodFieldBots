//! One bounded formation owner shared by the later game actors and Python service.
use super::{
    ensure, error,
    numeric::ServingRow,
    requests::{
        Answer, Identity, Participant, Promise, RequestHandle, Response, checked_duration,
        wait_duration,
    },
};
use crate::{
    budget::monotonic,
    error::{Error, Result},
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Condvar, Mutex},
};

type ParticipantKey = (String, String, String, u8);
pub struct DispatchRow {
    pub identity: Identity,
    pub inputs: ServingRow,
    pub sampling_seed: u64,
    pub rng_state: Option<Vec<u8>>,
}
pub struct Batch {
    pub id: u64,
    pub rows: Vec<DispatchRow>,
    pub runtime: String,
}
struct Pending {
    identity: Identity,
    inputs: Option<ServingRow>,
    candidates: usize,
    received: f64,
    retired: Option<Error>,
    batch: Option<u64>,
    handle: RequestHandle,
}
struct State {
    closed: Option<Error>,
    queue: VecDeque<u64>,
    pending: HashMap<u64, Pending>,
    participants: HashMap<ParticipantKey, Participant>,
    producers: HashMap<ParticipantKey, usize>,
    next: u64,
    next_batch: u64,
    forming: bool,
    formation_deadline: Option<f64>,
}
pub struct Dispatcher {
    capacity: usize,
    max_batch: usize,
    wait: f64,
    pub runtime: String,
    state: Mutex<State>,
    wake: Condvar,
    async_wake: tokio::sync::Notify,
}
// Counts real started admission calls, including capacity waits. Dropping an
// async admission future releases its registration without a cleanup callback.
struct Admission<'a> {
    dispatcher: &'a Dispatcher,
    key: ParticipantKey,
}
impl Drop for Admission<'_> {
    fn drop(&mut self) {
        let mut s = self.dispatcher.state.lock().unwrap();
        let count = s
            .producers
            .get_mut(&self.key)
            .expect("registered admission");
        *count -= 1;
        if *count == 0 {
            s.producers.remove(&self.key);
        }
        self.dispatcher.notify();
    }
}
impl Dispatcher {
    pub fn new(
        capacity: usize,
        max_batch: usize,
        formation_wait_ms: f64,
        runtime: String,
    ) -> Result<Arc<Self>> {
        ensure(
            capacity > 0
                && max_batch > 0
                && formation_wait_ms.is_finite()
                && formation_wait_ms >= 0.
                && !runtime.is_empty(),
            "invalid inference dispatcher configuration",
        )?;
        checked_duration(formation_wait_ms / 1000.)?;
        Ok(Arc::new(Self {
            capacity,
            max_batch,
            wait: formation_wait_ms / 1000.,
            runtime,
            state: Mutex::new(State {
                closed: None,
                queue: VecDeque::new(),
                pending: HashMap::new(),
                participants: HashMap::new(),
                producers: HashMap::new(),
                next: 0,
                next_batch: 0,
                forming: false,
                formation_deadline: None,
            }),
            wake: Condvar::new(),
            async_wake: tokio::sync::Notify::new(),
        }))
    }
    fn notify(&self) {
        self.wake.notify_all();
        self.async_wake.notify_waiters();
    }
    fn register(&self, identity: &Identity) -> Result<Admission<'_>> {
        let mut s = self.state.lock().unwrap();
        if let Some(e) = &s.closed {
            return Err(e.clone());
        }
        let key = identity.participant();
        *s.producers.entry(key.clone()).or_insert(0) += 1;
        Ok(Admission {
            dispatcher: self,
            key,
        })
    }
    fn admit(
        &self,
        s: &mut State,
        identity: &Identity,
        inputs: &mut Option<ServingRow>,
        seed: u64,
    ) -> Result<Option<(u64, RequestHandle)>> {
        if let Some(e) = &s.closed {
            return Err(e.clone());
        }
        let now = monotonic();
        if now >= identity.deadline {
            return Err(Error::Deadline);
        }
        let row = inputs.as_ref().unwrap();
        ensure(
            row.actor == identity.player && row.revision == identity.revision,
            "request own frame identity mismatch",
        )?;
        Self::expire(s, now);
        ensure(
            !s.pending
                .values()
                .any(|p| p.identity.game_key() == identity.game_key()),
            "game already has outstanding inference",
        )?;
        let p = s
            .participants
            .entry(identity.participant())
            .or_insert(Participant {
                epoch: identity.epoch,
                last_request: None,
                seed,
                state: None,
                retired: false,
            });
        if identity.epoch > p.epoch {
            p.epoch = identity.epoch;
            p.seed = seed;
            p.last_request = None;
            p.state = None;
            p.retired = false;
        }
        ensure(
            p.epoch == identity.epoch && !p.retired && p.seed == seed,
            "participant epoch or sampling seed mismatch",
        )?;
        ensure(
            p.last_request.is_none_or(|n| identity.request > n),
            "reused or stale inference request",
        )?;
        if s.pending.len() >= self.capacity {
            return Ok(None);
        }
        let formation = if s.queue.is_empty() {
            checked_duration(self.wait)?;
            Some(now + self.wait)
        } else {
            s.formation_deadline
        };
        s.participants
            .get_mut(&identity.participant())
            .unwrap()
            .last_request = Some(identity.request);
        let token = s.next;
        s.next += 1;
        let handle = Arc::new(Promise::default());
        let row = inputs.take().unwrap();
        s.pending.insert(
            token,
            Pending {
                identity: identity.clone(),
                candidates: row.candidates,
                inputs: Some(row),
                received: now,
                retired: None,
                batch: None,
                handle: handle.clone(),
            },
        );
        s.formation_deadline = formation;
        s.queue.push_back(token);
        self.notify();
        Ok(Some((token, handle)))
    }
    /// Fixed workers/component callers use the same bounded admission as async actors.
    pub fn enqueue(
        &self,
        identity: Identity,
        inputs: ServingRow,
        seed: u64,
    ) -> Result<(u64, RequestHandle)> {
        identity.validate()?;
        let mut inputs = Some(inputs);
        let _admission = self.register(&identity)?;
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(answer) = self.admit(&mut s, &identity, &mut inputs, seed)? {
                return Ok(answer);
            }
            let expiry = s
                .pending
                .values()
                .map(|p| p.identity.deadline)
                .filter(|d| *d > monotonic())
                .fold(identity.deadline, f64::min);
            s = self.wake.wait_timeout(s, wait_duration(expiry)?).unwrap().0;
        }
    }
    /// No thread pool, callbacks, Python event loop or second formation window.
    pub async fn enqueue_async(
        &self,
        identity: Identity,
        inputs: ServingRow,
        seed: u64,
    ) -> Result<(u64, RequestHandle)> {
        identity.validate()?;
        let mut inputs = Some(inputs);
        let _admission = self.register(&identity)?;
        loop {
            let notified = self.async_wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let expiry = {
                let mut s = self.state.lock().unwrap();
                if let Some(answer) = self.admit(&mut s, &identity, &mut inputs, seed)? {
                    return Ok(answer);
                }
                s.pending
                    .values()
                    .map(|p| p.identity.deadline)
                    .filter(|d| *d > monotonic())
                    .fold(identity.deadline, f64::min)
            };
            let _ = tokio::time::timeout(wait_duration(expiry)?, notified).await;
        }
    }
    fn expire(s: &mut State, now: f64) {
        // Exported work retains its original lease slot/identity until the service
        // settles it. Only queued (unexported) allocation custody can end here.
        s.pending.retain(|_, p| {
            if p.identity.deadline > now {
                return true;
            }
            p.handle.complete(Err(Error::Deadline));
            p.retired.get_or_insert(Error::Deadline);
            p.batch.is_some()
        });
        let State { pending, queue, .. } = &mut *s;
        queue.retain(|k| pending.contains_key(k));
        if s.queue.is_empty() {
            s.formation_deadline = None;
        }
    }
    pub fn next_batch(&self, deadline: f64) -> Result<Batch> {
        wait_duration(deadline)?;
        let mut s = self.state.lock().unwrap();
        ensure(!s.forming, "one batch formation owner required")?;
        s.forming = true;
        let result = loop {
            if let Some(e) = &s.closed {
                break Err(e.clone());
            }
            let now = monotonic();
            Self::expire(&mut s, now);
            self.notify();
            if now >= deadline {
                break Err(Error::Deadline);
            }
            let formation = s.formation_deadline;
            if !s.queue.is_empty() && (s.queue.len() >= self.max_batch || now >= formation.unwrap())
            {
                let id = s.next_batch;
                s.next_batch += 1;
                let mut rows = vec![];
                for _ in 0..self.max_batch {
                    let Some(k) = s.queue.pop_front() else { break };
                    let p = s.pending.get_mut(&k).unwrap();
                    p.batch = Some(id);
                    let identity = p.identity.clone();
                    let inputs = p.inputs.take().unwrap();
                    let participant = &s.participants[&identity.participant()];
                    rows.push(DispatchRow {
                        identity,
                        inputs,
                        sampling_seed: participant.seed,
                        rng_state: participant.state.clone(),
                    });
                }
                s.formation_deadline = s.queue.front().map(|k| s.pending[k].received + self.wait);
                break Ok(Batch {
                    id,
                    rows,
                    runtime: self.runtime.clone(),
                });
            }
            let expiry = s
                .pending
                .values()
                .map(|p| p.identity.deadline)
                .filter(|d| *d > now)
                .fold(deadline, f64::min);
            let until = formation.unwrap_or(deadline).min(deadline).min(expiry);
            let duration = match wait_duration(until) {
                Ok(d) => d,
                Err(e) => break Err(e),
            };
            s = self.wake.wait_timeout(s, duration).unwrap().0;
        };
        s.forming = false;
        self.notify();
        result
    }
    /// A whole batch is identity/shape validated before any participant commit.
    pub fn submit_results(&self, batch: u64, responses: Vec<Response>) -> Result<usize> {
        let mut s = self.state.lock().unwrap();
        if let Some(e) = &s.closed {
            return Err(e.clone());
        }
        let mut originals: Vec<_> = s
            .pending
            .iter()
            .filter(|(_, p)| p.batch == Some(batch))
            .map(|(k, _)| *k)
            .collect();
        originals.sort();
        ensure(!originals.is_empty(), "unknown or retired inference batch")?;
        let valid = originals.len() == responses.len()
            && originals.iter().zip(&responses).all(|(k, r)| {
                let p = &s.pending[k];
                p.identity == r.identity
                    && r.runtime == self.runtime
                    && r.logs.len() == p.candidates
                    && r.sample < p.candidates
                    && !r.next_state.is_empty()
                    && r.logs.iter().all(|n| n.is_finite())
                    && r.value.is_finite()
            });
        if !valid {
            for k in originals {
                s.pending
                    .remove(&k)
                    .unwrap()
                    .handle
                    .complete(Err(error("inference reply binding or shape mismatch")));
            }
            self.notify();
            return Err(error("inference reply binding or shape mismatch"));
        }
        let mut accepted = 0;
        for (k, r) in originals.into_iter().zip(responses) {
            let p = s.pending.remove(&k).unwrap();
            let participant = s.participants.get_mut(&r.identity.participant()).unwrap();
            if let Some(e) = p.retired {
                p.handle.complete(Err(e));
            } else if monotonic() >= r.identity.deadline {
                p.handle.complete(Err(Error::Deadline));
            } else if participant.epoch != r.identity.epoch {
                p.handle.complete(Err(error("stale inference epoch")));
            } else {
                participant.state = Some(r.next_state);
                p.handle.complete(Ok(Answer {
                    identity: r.identity,
                    logs: r.logs,
                    value: r.value,
                    sample: r.sample,
                }));
                accepted += 1;
            }
        }
        self.notify();
        Ok(accepted)
    }
    /// Epoch retirement is actor-owned; stale result never restores the prior stream.
    pub fn retire(&self, identity: &Identity) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        let keys: Vec<_> = s
            .pending
            .iter()
            .filter(|(_, p)| {
                p.identity.participant() == identity.participant()
                    && p.identity.epoch == identity.epoch
            })
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            let p = s.pending.get_mut(&k).unwrap();
            p.handle.complete(Err(Error::Cancelled));
            p.retired = Some(Error::Cancelled);
            if p.batch.is_none() {
                s.pending.remove(&k);
            }
        }
        let State { pending, queue, .. } = &mut *s;
        queue.retain(|k| pending.contains_key(k));
        if s.queue.is_empty() {
            s.formation_deadline = None;
        }
        let p = s
            .participants
            .entry(identity.participant())
            .or_insert(Participant {
                epoch: identity.epoch,
                last_request: None,
                seed: 0,
                state: None,
                retired: true,
            });
        if identity.epoch >= p.epoch {
            p.epoch = identity.epoch;
            p.last_request = None;
            p.state = None;
            p.retired = true;
        }
        self.notify();
        Ok(())
    }
    pub fn release_game(&self, run: &str, trial: &str, game: &str) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        ensure(
            !s.pending.values().any(|p| {
                p.identity.run == run && p.identity.trial == trial && p.identity.game == game
            }),
            "cannot release game with outstanding inference",
        )?;
        ensure(
            !s.producers
                .keys()
                .any(|(r, t, g, _)| r == run && t == trial && g == game),
            "cannot release game with active inference admission",
        )?;
        s.participants
            .retain(|(r, t, g, _), _| r != run || t != trial || g != game);
        Ok(())
    }
    pub fn checkpoint_references(&self) -> HashMap<String, usize> {
        let s = self.state.lock().unwrap();
        let mut out = HashMap::new();
        for p in s.pending.values() {
            *out.entry(p.identity.checkpoint.clone()).or_insert(0) += 1;
        }
        out
    }
    pub fn close(&self, reason: Error) {
        let mut s = self.state.lock().unwrap();
        if s.closed.is_none() {
            s.closed = Some(reason.clone());
        }
        for (_, p) in s.pending.drain() {
            p.handle.complete(Err(reason.clone()));
        }
        s.queue.clear();
        s.participants.clear();
        self.notify();
    }
}
impl Drop for Dispatcher {
    fn drop(&mut self) {
        self.close(Error::Closed);
    }
}

// Compact control/owned-array Python boundary. The later actors call Dispatcher directly.
use numpy::PyReadonlyArray1;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
#[pyclass]
pub struct PlayerInferenceDispatcher {
    core: Arc<Dispatcher>,
    handles: Mutex<HashMap<u64, RequestHandle>>,
}
impl PlayerInferenceDispatcher {
    /// Host runtime shares the original actor dispatcher, never constructs a second queue.
    pub fn from_core(core: Arc<Dispatcher>) -> Self {
        Self {
            core,
            handles: Mutex::new(HashMap::new()),
        }
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlInput {
    identity: Identity,
    frame: crate::json::Value,
    events: Vec<crate::json::Value>,
    actions: crate::json::Value,
    rules: crate::json::Value,
    table: super::search::Table,
    window: usize,
    max_positions: usize,
    sampling_seed: u64,
}
#[pymethods]
impl PlayerInferenceDispatcher {
    #[new]
    fn new(
        capacity: usize,
        max_batch_size: usize,
        formation_wait_ms: f64,
        runtime_identity: String,
    ) -> PyResult<Self> {
        Ok(Self {
            core: Dispatcher::new(
                capacity,
                max_batch_size,
                formation_wait_ms,
                runtime_identity,
            )
            .map_err(crate::python_error)?,
            handles: Mutex::new(HashMap::new()),
        })
    }
    fn enqueue(&self, py: Python<'_>, encoded: &[u8]) -> PyResult<u64> {
        let (token, handle) = py
            .detach(|| -> Result<_> {
                let c: ControlInput =
                    serde_json::from_slice(encoded).map_err(|e| error(e.to_string()))?;
                let config = super::Config::parse(&c.actions, &c.rules)?;
                let budget = crate::budget::Budget::new(12usize << 30);
                let ctl = crate::budget::Control::new(c.identity.deadline)?;
                let input = super::Input::parse(
                    &c.frame,
                    &serde_json::json!(c.events),
                    &config,
                    &budget,
                    &ctl,
                )?;
                c.table.validate(&input, &config)?;
                let row = ServingRow::build(
                    &c.frame,
                    &c.events,
                    &input,
                    &config,
                    &c.table,
                    c.window,
                    c.max_positions,
                    &budget,
                    &ctl,
                )?;
                self.core.enqueue(c.identity, row, c.sampling_seed)
            })
            .map_err(crate::python_error)?;
        self.handles.lock().unwrap().insert(token, handle);
        Ok(token)
    }
    fn next_batch(&self, py: Python<'_>, deadline: f64) -> PyResult<Py<PyDict>> {
        let batch = py
            .detach(|| self.core.next_batch(deadline))
            .map_err(crate::python_error)?;
        // Move each allocation exactly once into NumPy's owning capsule. No Rust reuse.
        let result = (|| {
            let out = PyDict::new(py);
            out.set_item("batch_id", batch.id)?;
            out.set_item("runtime", &batch.runtime)?;
            let rows = PyList::empty(py);
            for r in batch.rows {
                let row = PyDict::new(py);
                row.set_item(
                    "identity",
                    PyBytes::new(py, &serde_json::to_vec(&r.identity).unwrap()),
                )?;
                row.set_item("sampling_seed", r.sampling_seed)?;
                if let Some(state) = r.rng_state {
                    row.set_item("rng_state", PyBytes::new(py, &state))?;
                } else {
                    row.set_item("rng_state", py.None())?;
                }
                row.set_item("candidate_count", r.inputs.candidates)?;
                row.set_item("event_length", r.inputs.events)?;
                row.set_item("assembly_seconds", r.inputs.assembly_seconds)?;
                row.set_item("buffer_bytes", r.inputs.bytes())?;
                let (inputs, offsets) = r.inputs.export(py)?;
                row.set_item("inputs", inputs)?;
                row.set_item("candidate_offsets", offsets)?;
                rows.append(row)?;
            }
            out.set_item("rows", rows)?;
            Ok(out.unbind())
        })();
        if result.is_err() {
            self.core
                .close(error("inference buffer publication failed"));
        }
        result
    }
    fn submit_results(
        &self,
        py: Python<'_>,
        batch_id: u64,
        rows: &Bound<'_, PyList>,
    ) -> PyResult<usize> {
        let parsed = (|| -> PyResult<Vec<Response>> {
            let mut responses = vec![];
            for item in rows.iter() {
                let row = item.cast::<PyDict>()?;
                let get = |name: &str| -> PyResult<Bound<'_, PyAny>> {
                    row.get_item(name)?.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!("missing {name}"))
                    })
                };
                let identity: Vec<u8> = get("identity")?.extract()?;
                let logs = get("logs")?;
                let logs = logs
                    .extract::<PyReadonlyArray1<'_, f32>>()?
                    .as_slice()?
                    .to_vec();
                responses.push(Response {
                    identity: serde_json::from_slice(&identity)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
                    logs,
                    value: get("value")?.extract()?,
                    sample: get("sample")?.extract()?,
                    next_state: get("next_state")?.extract()?,
                    runtime: get("runtime")?.extract()?,
                });
            }
            Ok(responses)
        })();
        match parsed {
            Ok(responses) => py
                .detach(|| self.core.submit_results(batch_id, responses))
                .map_err(crate::python_error),
            Err(e) => {
                self.core.close(error("inference reply decode failed"));
                Err(e)
            }
        }
    }
    fn wait_result(&self, py: Python<'_>, token: u64, deadline: f64) -> PyResult<Py<PyDict>> {
        wait_duration(deadline).map_err(crate::python_error)?;
        let handle = self
            .handles
            .lock()
            .unwrap()
            .get(&token)
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown inference handle"))?;
        let result = py.detach(|| handle.wait(deadline));
        self.handles.lock().unwrap().remove(&token);
        let answer = result.map_err(crate::python_error)?;
        let out = PyDict::new(py);
        out.set_item(
            "identity",
            PyBytes::new(py, &serde_json::to_vec(&answer.identity).unwrap()),
        )?;
        use numpy::IntoPyArray;
        out.set_item("logs", answer.logs.into_pyarray(py))?;
        out.set_item("value", answer.value)?;
        out.set_item("sample", answer.sample)?;
        Ok(out.unbind())
    }
    fn retire(&self, encoded: &[u8]) -> PyResult<()> {
        let identity: Identity = serde_json::from_slice(encoded)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.core.retire(&identity).map_err(crate::python_error)
    }
    fn release_game(&self, run: &str, trial: &str, game: &str) -> PyResult<()> {
        self.core
            .release_game(run, trial, game)
            .map_err(crate::python_error)
    }
    fn checkpoint_references(&self) -> HashMap<String, usize> {
        self.core.checkpoint_references()
    }
    fn cancel(&self, reason: String) {
        self.core.close(error(reason));
    }
    fn close(&self) {
        self.core.close(Error::Closed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        budget::{Budget, Control},
        json::Value,
    };
    use std::time::Duration;
    fn row(game: &str, deadline: f64) -> (Identity, ServingRow) {
        let saved: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let e = &saved["episodes"][3];
        let d = &e["decisions"][0];
        let config = super::super::Config::parse(&e["actions"], &e["rules"]).unwrap();
        let budget = Budget::new(256 << 20);
        let control = Control::new(f64::MAX).unwrap();
        let input = super::super::Input::parse(
            &d["frame"],
            &serde_json::json!([]),
            &config,
            &budget,
            &control,
        )
        .unwrap();
        let table = serde_json::from_value(d["generated"]["table"].clone()).unwrap();
        let inputs = ServingRow::build(
            &d["frame"],
            &[],
            &input,
            &config,
            &table,
            128,
            18,
            &budget,
            &control,
        )
        .unwrap();
        (
            Identity {
                run: "r".into(),
                trial: "t".into(),
                game: game.into(),
                player: 1,
                epoch: 0,
                request: 1,
                revision: 0,
                checkpoint: "a".repeat(64),
                deadline,
            },
            inputs,
        )
    }
    fn response(identity: Identity) -> Response {
        Response {
            identity,
            logs: vec![0.],
            value: 0.,
            sample: 0,
            next_state: vec![1, 2, 3],
            runtime: "test".into(),
        }
    }
    #[tokio::test(flavor = "current_thread")]
    async fn async_full_admission_and_result_wait_leave_single_worker_responsive() {
        let d = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
        let (first, inputs) = row("g1", monotonic() + 5.);
        let (_, first_handle) = d.enqueue_async(first.clone(), inputs, 17).await.unwrap();
        let (second, inputs) = row("g2", monotonic() + 5.);
        let other = d.clone();
        let waiting = tokio::spawn(async move { other.enqueue_async(second, inputs, 17).await });
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(!waiting.is_finished()); // Timer executed on the same one-worker runtime.
        d.retire(&first).unwrap();
        let (_, handle) = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            first_handle.wait_async(monotonic() + 1.).await.unwrap_err(),
            Error::Cancelled
        );
        let waiter = tokio::spawn(async move { handle.wait_async(monotonic() + 2.).await });
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(!waiter.is_finished());
        d.close(Error::Closed);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .unwrap()
                .unwrap()
                .unwrap_err(),
            Error::Closed
        );
    }
    #[tokio::test(flavor = "current_thread")]
    async fn async_result_wake_before_or_after_subscription_never_lost() {
        for index in 0..64 {
            let promise = Arc::new(Promise::default());
            let target = promise.clone();
            let task = tokio::spawn(async move { target.wait_async(monotonic() + 1.).await });
            if index % 2 == 0 {
                tokio::task::yield_now().await;
            }
            promise.complete(Err(Error::Cancelled));
            assert_eq!(task.await.unwrap().unwrap_err(), Error::Cancelled);
        }
    }
    #[test]
    fn expired_exported_row_keeps_original_binding_while_other_row_commits() {
        let d = Dispatcher::new(2, 2, 0., "test".into()).unwrap();
        let (mut first, first_inputs) = row("g1", monotonic() + 5.);
        let (second, inputs) = row("g2", monotonic() + 5.);
        first.deadline = monotonic() + 0.04;
        let (_, a) = d.enqueue(first.clone(), first_inputs, 17).unwrap();
        let (_, b) = d.enqueue(second.clone(), inputs, 18).unwrap();
        let batch = d.next_batch(monotonic() + 1.).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(matches!(
            d.next_batch(monotonic() + 0.002),
            Err(Error::Deadline)
        ));
        assert_eq!(
            d.submit_results(
                batch.id,
                vec![response(first.clone()), response(second.clone())]
            )
            .unwrap(),
            1
        );
        assert_eq!(a.wait(monotonic() + 1.).unwrap_err(), Error::Deadline);
        assert_eq!(b.wait(monotonic() + 1.).unwrap().sample, 0);
        let s = d.state.lock().unwrap();
        assert!(s.participants[&first.participant()].state.is_none());
        assert_eq!(
            s.participants[&second.participant()].state,
            Some(vec![1, 2, 3])
        );
    }
    #[test]
    fn direct_actor_admission_binds_row_player_and_revision() {
        for field in ["player", "revision"] {
            let d = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
            let (mut identity, inputs) = row("g1", monotonic() + 5.);
            if field == "player" {
                identity.player = 2;
            } else {
                identity.revision += 1;
            }
            assert!(d.enqueue(identity, inputs, 17).is_err());
        }
    }
    #[test]
    fn retire_clears_private_state_and_release_game_clears_routing_tombstone() {
        let d = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
        let (identity, inputs) = row("g1", monotonic() + 5.);
        d.enqueue(identity.clone(), inputs, 17).unwrap();
        let batch = d.next_batch(monotonic() + 1.).unwrap();
        assert!(d.release_game("r", "t", "g1").is_err());
        d.retire(&identity).unwrap();
        assert!(d.release_game("r", "t", "g1").is_err()); // exported lease still physically active
        assert_eq!(
            d.submit_results(batch.id, vec![response(identity.clone())])
                .unwrap(),
            0
        );
        {
            let s = d.state.lock().unwrap();
            let p = &s.participants[&identity.participant()];
            assert!(p.state.is_none() && p.retired);
        }
        d.release_game("r", "t", "g1").unwrap();
        assert!(d.state.lock().unwrap().participants.is_empty());
    }
    #[tokio::test(flavor = "current_thread")]
    async fn fix1_retire_blocked_async_epoch_and_release_waiting_producer() {
        let d = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
        let (first, inputs) = row("g1", monotonic() + 5.);
        d.enqueue(first.clone(), inputs, 17).unwrap();
        let (second, inputs) = row("g2", monotonic() + 5.);
        let other = d.clone();
        let id = second.clone();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            started.send(()).unwrap();
            other.enqueue_async(id, inputs, 17).await
        });
        ready.await.unwrap();
        assert!(d.release_game("r", "t", "g2").is_err());
        d.retire(&second).unwrap();
        assert!(d.release_game("r", "t", "g2").is_err());
        d.retire(&first).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        {
            let s = d.state.lock().unwrap();
            let p = &s.participants[&second.participant()];
            assert!(p.retired && p.state.is_none());
            assert!(!s.producers.contains_key(&second.participant()));
        }
        let (mut later, inputs) = row("g2", monotonic() + 5.);
        later.epoch = 1;
        let (_, handle) = d.enqueue_async(later.clone(), inputs, 99).await.unwrap();
        let batch = d.next_batch(monotonic() + 1.).unwrap();
        assert!(batch.rows[0].rng_state.is_none());
        assert_eq!(
            d.submit_results(batch.id, vec![response(later)]).unwrap(),
            1
        );
        handle.wait_async(monotonic() + 1.).await.unwrap();
        d.release_game("r", "t", "g2").unwrap();
    }
    #[tokio::test(flavor = "current_thread")]
    async fn fix1_dropped_async_admission_releases_its_producer_registration() {
        let d = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
        let (first, inputs) = row("g1", monotonic() + 5.);
        d.enqueue(first, inputs, 17).unwrap();
        let (second, inputs) = row("g2", monotonic() + 5.);
        let other = d.clone();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            started.send(()).unwrap();
            other.enqueue_async(second, inputs, 17).await
        });
        ready.await.unwrap();
        assert!(d.release_game("r", "t", "g2").is_err());
        task.abort();
        assert!(matches!(task.await,Err(e) if e.is_cancelled()));
        d.release_game("r", "t", "g2").unwrap();
    }
    #[tokio::test(flavor = "current_thread")]
    async fn fix1_unsupported_timers_leave_sync_and_async_admission_usable() {
        for deadline in [f64::MAX, 1e19] {
            let d = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
            let (mut identity, inputs) = row("g1", monotonic() + 5.);
            identity.deadline = deadline;
            assert!(d.enqueue(identity, inputs, 17).is_err());
            let (mut identity, inputs) = row("g2", monotonic() + 5.);
            identity.deadline = deadline;
            assert!(d.enqueue_async(identity, inputs, 17).await.is_err());
            assert!(d.state.lock().unwrap().producers.is_empty());
            assert!(d.next_batch(deadline).is_err());
            assert!(!d.state.lock().unwrap().forming);
            let (identity, inputs) = row("valid", monotonic() + 5.);
            let (_, handle) = d.enqueue_async(identity.clone(), inputs, 17).await.unwrap();
            let batch = d.next_batch(monotonic() + 1.).unwrap();
            assert_eq!(
                d.submit_results(batch.id, vec![response(identity)])
                    .unwrap(),
                1
            );
            handle.wait_async(monotonic() + 1.).await.unwrap();
            d.close(Error::Closed);
        }
        assert!(Dispatcher::new(1, 1, f64::MAX, "test".into()).is_err());
    }
}
