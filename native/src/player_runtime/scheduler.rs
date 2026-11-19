//! A common actor on either Tokio W threads or deterministic fixed W runtimes.
use super::{
    batch::{Dispatcher, PlayerInferenceDispatcher},
    ensure, error,
    game::{
        self, GameReport, GameSpec,
        storage::{Mailbox, RecordingAccount, StorageBundle},
    },
    http,
    requests::wait_duration,
};
use crate::{
    budget::monotonic,
    error::{Error, Result},
};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
};
use tokio::sync::{Semaphore, mpsc};

#[derive(Default)]
pub struct Cancellation {
    flag: AtomicBool,
    wake: tokio::sync::Notify,
}
impl Cancellation {
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.wake.notify_waiters();
    }
    pub async fn wait(&self) {
        loop {
            let wake = self.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if self.flag.load(Ordering::Acquire) {
                return;
            }
            wake.await;
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    pub scheduler: String,
    pub player_worker_threads: usize,
    pub game_concurrency: usize,
    pub max_batch_size: usize,
    pub formation_wait_ms: f64,
    pub storage_bytes: usize,
    pub storage_queue: usize,
    pub runtime_identity: String,
}
struct Completion {
    reports: Vec<GameReport>,
    report_keys: HashMap<(String, String, String), usize>,
    completed: usize,
    errors: Vec<String>,
}
struct Shared {
    done: Mutex<Completion>,
    wake: Condvar,
    worker_done: AtomicUsize,
    worker_target: AtomicUsize,
    worker_wake: tokio::sync::Notify,
    submitted: AtomicUsize,
}
impl Shared {
    fn finish(&self, key: (String, String, String), report: GameReport) {
        let mut s = self.done.lock().unwrap();
        let ordinal = s.reports.len();
        assert!(s.report_keys.insert(key, ordinal).is_none());
        s.reports.push(report);
        s.completed += 1;
        self.wake.notify_all();
    }
    fn failed(&self, message: String) {
        let mut s = self.done.lock().unwrap();
        s.errors.push(message);
        self.wake.notify_all();
    }
    async fn global_worker_end(&self) {
        loop {
            let wake = self.worker_wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if self.worker_done.load(Ordering::Acquire)
                == self.worker_target.load(Ordering::Acquire)
            {
                return;
            }
            wake.await;
        }
    }
}
// Count the original workers that actually started, including failed workers.
// A startup error/panic must wake surviving runtimes instead of stranding their
// global connection-driver lifetime gate at the requested worker count.
struct WorkerRetirement {
    shared: Arc<Shared>,
    cancel: Arc<Cancellation>,
    storage: Arc<Mailbox>,
    dispatcher: Arc<Dispatcher>,
    arrived: bool,
}
impl WorkerRetirement {
    fn arrive(&mut self) {
        self.arrived = true;
        self.shared.worker_done.fetch_add(1, Ordering::AcqRel);
        self.shared.worker_wake.notify_waiters();
    }
}
impl Drop for WorkerRetirement {
    fn drop(&mut self) {
        if !self.arrived {
            self.shared
                .failed("original player worker stopped before owned shutdown".into());
            self.cancel.cancel();
            self.storage.close(error("original worker failed"));
            self.dispatcher.close(error("original worker failed"));
            self.arrive();
        }
    }
}
struct Lifecycle {
    senders: Vec<mpsc::UnboundedSender<GameSpec>>,
    threads: Vec<JoinHandle<()>>,
    started: bool,
    accepting: bool,
    joined: bool,
    next: usize,
    episodes: HashSet<(String, String, String)>,
}
#[pyclass]
pub struct PlayerRuntime {
    config: RuntimeConfig,
    dispatcher: Arc<Dispatcher>,
    storage: Arc<Mailbox>,
    cancel: Arc<Cancellation>,
    shared: Arc<Shared>,
    life: Mutex<Lifecycle>,
}
impl PlayerRuntime {
    fn submit_inner(&self, games: Vec<GameSpec>) -> Result<()> {
        // Validate the entire accepted wave before any native game admission.
        let mut incoming = HashSet::new();
        for game in &games {
            game.validate()?;
            ensure(
                incoming.insert((game.run.clone(), game.trial.clone(), game.episode.clone())),
                "duplicate planned episode",
            )?;
        }
        let mut life = self.life.lock().unwrap();
        ensure(
            life.accepting && !life.joined && !self.cancel.flag.load(Ordering::Acquire),
            "runtime admission closed",
        )?;
        ensure(
            incoming.iter().all(|k| !life.episodes.contains(k)),
            "planned episode already submitted",
        )?;
        if !life.started {
            if let Err(e) = self.launch(&mut life) {
                self.cancel.cancel();
                self.storage.close(e.clone());
                self.dispatcher.close(e.clone());
                return Err(e);
            }
            life.started = true;
        }
        for game in games {
            let worker = if self.config.scheduler == "fixed" {
                life.next % self.config.player_worker_threads
            } else {
                0
            };
            life.next += 1;
            life.episodes
                .insert((game.run.clone(), game.trial.clone(), game.episode.clone()));
            self.shared.submitted.fetch_add(1, Ordering::AcqRel);
            life.senders[worker]
                .send(game)
                .map_err(|_| error("native scheduler receiver closed"))?;
        }
        Ok(())
    }
    fn launch(&self, life: &mut Lifecycle) -> Result<()> {
        let client = http::client()?;
        let slots = Arc::new(Semaphore::new(self.config.game_concurrency));
        let count = if self.config.scheduler == "fixed" {
            self.config.player_worker_threads
        } else {
            1
        };
        for worker in 0..count {
            let (tx, mut rx) = mpsc::unbounded_channel();
            life.senders.push(tx);
            let client = client.clone();
            let dispatcher = self.dispatcher.clone();
            let storage = self.storage.clone();
            let cancel = self.cancel.clone();
            let shared = self.shared.clone();
            let slots = slots.clone();
            let config = self.config.clone();
            let thread=std::thread::Builder::new().name(format!("gf-player-{}-{worker}",config.scheduler)).spawn(move||{
                let mut retirement=WorkerRetirement{shared:shared.clone(),cancel:cancel.clone(),storage:storage.clone(),dispatcher:dispatcher.clone(),arrived:false};
                let mut builder=if config.scheduler=="fixed"{tokio::runtime::Builder::new_current_thread()}else{let mut b=tokio::runtime::Builder::new_multi_thread();b.worker_threads(config.player_worker_threads);b.thread_name("gf-player-tokio");b};
                let runtime=match builder.enable_all().build(){Ok(v)=>v,Err(e)=>{shared.failed(format!("Tokio startup: {e}"));cancel.cancel();storage.close(Error::Io(e.to_string()));dispatcher.close(Error::Io(e.to_string()));return;}};
                let local=tokio::task::LocalSet::new();
                let runtime_handle=runtime.handle().clone();
                // Fixed tasks are never Send-spawned; one worker owns each game
                // forever. Both paths call exactly game::run, including all waits.
                local.block_on(&runtime,async move {
                    let mut tasks=tokio::task::JoinSet::new();
                    loop {
                        tokio::select! {
                            item=rx.recv()=>match item {
                                Some(spec)=>{
                                    let client=client.clone();let dispatcher=dispatcher.clone();let storage=storage.clone();let cancel=cancel.clone();let shared=shared.clone();let slots=slots.clone();
                                    let job=async move {
                                        let permit=tokio::select!{biased;_=cancel.wait()=>None,permit=slots.acquire_owned()=>permit.ok()};
                                        // C includes the terminal record until same-reservation
                                        // writer acceptance. Queued cancelled games still emit
                                        // explicit outcomes, without acquiring/reusing a live slot.
                                        let key=(spec.run.clone(),spec.trial.clone(),spec.episode.clone());
                                        let report=game::run(spec,client,dispatcher,storage,cancel).await;
                                        shared.finish(key,report);drop(permit);
                                    };
                                    if config.scheduler=="fixed"{tasks.spawn_local(job);}else{tasks.spawn_on(job,&runtime_handle);}
                                },None=>break,
                            },
                            done=tasks.join_next(),if !tasks.is_empty()=>{if let Some(Err(e))=done{shared.failed(format!("game task failed: {e}"));cancel.cancel();storage.close(error("game task failed"));dispatcher.close(error("game task failed"));}},
                        }
                    }
                    while let Some(result)=tasks.join_next().await{if let Err(e)=result{shared.failed(format!("game task failed: {e}"));cancel.cancel();storage.close(error("game task failed"));dispatcher.close(error("game task failed"));}}
                    // An idle B worker keeps driving shared reqwest connections and
                    // timers until all original worker receivers/tasks have ended.
                    retirement.arrive();
                    shared.global_worker_end().await;
                });
            }).map_err(Error::from)?;
            life.threads.push(thread);
            self.shared.worker_target.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }
    fn join_inner(&self) -> Result<Vec<u8>> {
        let threads = {
            let mut life = self.life.lock().unwrap();
            life.accepting = false;
            life.senders.clear();
            std::mem::take(&mut life.threads)
        };
        for thread in threads {
            if thread.join().is_err() {
                self.shared.failed("original player thread panicked".into());
            }
        }
        let mut life = self.life.lock().unwrap();
        life.joined = true;
        self.storage.close(Error::Closed);
        self.dispatcher.close(Error::Closed);
        let done = self.shared.done.lock().unwrap();
        ensure(
            done.errors.is_empty(),
            &format!("native runtime failure: {:?}", done.errors),
        )?;
        ensure(
            done.completed == self.shared.submitted.load(Ordering::Acquire),
            "native runtime missing original game completion",
        )?;
        let result=serde_json::to_vec(&serde_json::json!({"joined":true,"games":done.reports,"player_threads":if life.started{self.config.player_worker_threads}else{0},"control_threads":if life.started&&self.config.scheduler=="tokio"{1}else{0},"io_threads":0,"native_batcher_threads":0,"python_service_threads":"host-reported","torch_threads":"host-reported","retained_capture_bytes":self.storage.used()})).map_err(|e|error(e.to_string()))?;
        self.storage.close(Error::Closed);
        self.dispatcher.close(Error::Closed);
        Ok(result)
    }
}
#[pymethods]
impl PlayerRuntime {
    #[new]
    fn new(encoded: &[u8]) -> PyResult<Self> {
        let config: RuntimeConfig = serde_json::from_slice(encoded)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        ensure(
            ["tokio", "fixed"].contains(&config.scheduler.as_str())
                && config.player_worker_threads > 0
                && config.game_concurrency > 0,
            "invalid native scheduler geometry",
        )
        .map_err(crate::python_error)?;
        let dispatcher = Dispatcher::new(
            config.game_concurrency,
            config.max_batch_size,
            config.formation_wait_ms,
            config.runtime_identity.clone(),
        )
        .map_err(crate::python_error)?;
        let storage = Mailbox::new(config.storage_bytes, config.storage_queue)
            .map_err(crate::python_error)?;
        Ok(Self {
            config,
            dispatcher,
            storage,
            cancel: Arc::new(Cancellation::default()),
            shared: Arc::new(Shared {
                done: Mutex::new(Completion {
                    reports: vec![],
                    report_keys: HashMap::new(),
                    completed: 0,
                    errors: vec![],
                }),
                wake: Condvar::new(),
                worker_done: AtomicUsize::new(0),
                worker_target: AtomicUsize::new(0),
                worker_wake: tokio::sync::Notify::new(),
                submitted: AtomicUsize::new(0),
            }),
            life: Mutex::new(Lifecycle {
                senders: vec![],
                threads: vec![],
                started: false,
                accepting: true,
                joined: false,
                next: 0,
                episodes: HashSet::new(),
            }),
        })
    }
    #[getter]
    fn dispatcher(&self) -> PlayerInferenceDispatcher {
        PlayerInferenceDispatcher::from_core(self.dispatcher.clone())
    }
    fn start(&self, py: Python<'_>, encoded: &[u8]) -> PyResult<()> {
        let games: Vec<GameSpec> = serde_json::from_slice(encoded)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        py.detach(|| self.submit_inner(games))
            .map_err(crate::python_error)
    }
    fn next_storage(&self, py: Python<'_>, deadline: f64) -> PyResult<StorageBundle> {
        let record = py
            .detach(|| self.storage.next(deadline))
            .map_err(crate::python_error)?;
        Ok(StorageBundle { record })
    }
    #[pyo3(signature=(bundle, failure=None))]
    fn storage_ack(&self, bundle: &StorageBundle, failure: Option<String>) -> PyResult<()> {
        self.storage
            .ack(&bundle.record, failure)
            .map_err(crate::python_error)
    }
    fn recording_account(&self) -> RecordingAccount {
        self.storage.account()
    }
    fn storage_bytes(&self) -> usize {
        self.storage.used()
    }
    fn wait(&self, py: Python<'_>, deadline: f64) -> PyResult<()> {
        py.detach(|| -> Result<()> {
            wait_duration(deadline)?;
            let mut done = self.shared.done.lock().unwrap();
            while done.completed < self.shared.submitted.load(Ordering::Acquire) {
                if !done.errors.is_empty() {
                    return Err(error(format!("native game task failed: {:?}", done.errors)));
                }
                if monotonic() >= deadline {
                    return Err(Error::Deadline);
                }
                done = self
                    .shared
                    .wake
                    .wait_timeout(done, wait_duration(deadline)?)
                    .unwrap()
                    .0;
            }
            Ok(())
        })
        .map_err(crate::python_error)
    }
    fn completed_reports<'py>(&self, py: Python<'py>, keys: Vec<(String, String, String)>)
        -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let encoded = py.detach(|| -> Result<Vec<u8>> {
            let unique: HashSet<_> = keys.iter().collect();
            ensure(unique.len() == keys.len(), "duplicate completed-report key")?;
            let life = self.life.lock().unwrap();
            ensure(keys.iter().all(|key| life.episodes.contains(key)),
                "unissued completed-report key")?;
            let done = self.shared.done.lock().unwrap();
            let mut rows = Vec::with_capacity(keys.len());
            for key in &keys {
                let ordinal = done.report_keys.get(key)
                    .ok_or_else(|| error("incomplete completed-report key"))?;
                rows.push(serde_json::json!({"run":key.0,"trial":key.1,
                    "episode":key.2,"report":done.reports[*ordinal]}));
            }
            serde_json::to_vec(&rows).map_err(|e| error(e.to_string()))
        }).map_err(crate::python_error)?;
        Ok(pyo3::types::PyBytes::new(py, &encoded))
    }
    fn quiesce(&self) {
        let mut life = self.life.lock().unwrap();
        life.accepting = false;
        life.senders.clear();
    }
    fn cancel(&self, reason: String) {
        self.cancel.cancel();
        self.storage.close(error(reason)); /* Retire actual games; keep exported service leases until original submit/close. */
    }
    fn join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let report = py
            .detach(|| self.join_inner())
            .map_err(crate::python_error)?;
        Ok(pyo3::types::PyBytes::new(py, &report))
    }
}
impl Drop for PlayerRuntime {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.storage.close(Error::Closed);
        self.dispatcher.close(Error::Closed);
        self.life.lock().unwrap().senders.clear(); /* Explicit join alone proves original thread completion. */
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test(flavor = "current_thread")]
    async fn failed_original_worker_wakes_surviving_shutdown_gate() {
        let shared = Arc::new(Shared {
            done: Mutex::new(Completion {
                reports: vec![],
                report_keys: HashMap::new(),
                completed: 0,
                errors: vec![],
            }),
            wake: Condvar::new(),
            worker_done: AtomicUsize::new(0),
            worker_target: AtomicUsize::new(2),
            worker_wake: tokio::sync::Notify::new(),
            submitted: AtomicUsize::new(0),
        });
        let cancel = Arc::new(Cancellation::default());
        let storage = Mailbox::new(1024, 1).unwrap();
        let dispatcher = Dispatcher::new(1, 1, 0., "test".into()).unwrap();
        let mut normal = WorkerRetirement {
            shared: shared.clone(),
            cancel: cancel.clone(),
            storage: storage.clone(),
            dispatcher: dispatcher.clone(),
            arrived: false,
        };
        let failed = WorkerRetirement {
            shared: shared.clone(),
            cancel: cancel.clone(),
            storage: storage.clone(),
            dispatcher: dispatcher.clone(),
            arrived: false,
        };
        normal.arrive();
        let own = shared.clone();
        let waiter = tokio::spawn(async move {
            own.global_worker_end().await;
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(failed);
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        assert!(cancel.flag.load(Ordering::Acquire));
        assert_eq!(shared.done.lock().unwrap().errors.len(), 1);
        assert!(dispatcher.next_batch(monotonic() + 1.).is_err());
        assert!(storage.next(monotonic() + 1.).is_err());
    }
}
