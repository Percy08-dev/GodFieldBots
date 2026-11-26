//! Detached Journal generation over the original mailbox lease.

use super::{Mailbox, NativeRolloutLease, Record, Reservation, StorageBundle};
use crate::{
    budget::{Budget, BudgetVec, Control, Lease, monotonic},
    episode_pool::STACK_BYTES,
    error::{Error, Result},
    player_runtime::{
        ensure, error,
        game::rollout_journal::{EpisodeWriter, JournalValidation, LearnerEvidence},
    },
    python_error,
};
use pyo3::{prelude::*, types::PyBytes};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Context {
    episode: String,
    headers: Option<[Value; 2]>,
    template_ids: Option<[usize; 2]>,
    run: Option<String>,
    runtime_evidence: Option<Value>,
    trial: String,
    policies: [Value; 2],
    numeric_costs: Value,
    clock: Value,
    learner_viewer: u8,
    deadline: f64,
    rules: Value,
    classification: Value,
}

struct Active {
    writer: EpisodeWriter,
    originals: Vec<Arc<Record>>,
    retained: Reservation,
    control: Control,
}

#[derive(Serialize)]
struct Receipt {
    id: u64,
    kind: String,
    episode: String,
    game: String,
    transport_sha256: String,
    bytes: usize,
}

#[pyclass(frozen)]
pub(crate) struct RolloutTerminal {
    record: Arc<Record>,
    receipts: Vec<Receipt>,
    // The old writer held these to pair commit. This capability has the same lifetime.
    _originals: Vec<Arc<Record>>,
    output: Mutex<Option<Output>>,
    report: Value,
    proof: Mutex<Option<EvidenceOutput>>,
}

struct Output {
    members: Vec<(String, BudgetVec<u8>)>,
    _retained: Reservation,
}

struct EvidenceOutput {
    evidence: LearnerEvidence,
    _retained: Reservation,
}

type EvidencePayload<'py> = (
    Bound<'py, PyBytes>,
    Vec<Bound<'py, PyBytes>>,
    Vec<Bound<'py, PyBytes>>,
);

#[pymethods]
impl RolloutTerminal {
    fn bundle(&self) -> StorageBundle {
        StorageBundle {
            record: self.record.clone(),
        }
    }

    fn same_original(&self, bundle: &StorageBundle) -> bool {
        Arc::ptr_eq(&self.record, &bundle.record)
    }

    fn prior_receipts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| serde_json::to_vec(&self.receipts))
            .map_err(|e| python_error(error(e.to_string())))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn members(&self) -> PyResult<Vec<(String, usize)>> {
        let output = self.output.lock().unwrap();
        let output = output.as_ref().ok_or_else(|| python_error(Error::Closed))?;
        Ok(output
            .members
            .iter()
            .map(|(name, data)| (name.clone(), data.len()))
            .collect())
    }

    fn copy_member<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyBytes>> {
        let output = self.output.lock().unwrap();
        let (_, bytes) = output
            .as_ref()
            .and_then(|o| o.members.get(index))
            .ok_or_else(|| python_error(error("unknown completed member")))?;
        Ok(PyBytes::new(py, bytes))
    }

    fn has_evidence(&self) -> bool {
        self.proof.lock().unwrap().is_some()
    }

    fn report_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| serde_json::to_vec(&self.report))
            .map_err(|e| python_error(error(e.to_string())))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn evidence_bytes(&self) -> PyResult<usize> {
        let owner = self.proof.lock().unwrap();
        let proof = &owner
            .as_ref()
            .ok_or_else(|| python_error(Error::Closed))?
            .evidence;
        Ok(proof.events.iter().map(|v| v.len()).sum::<usize>()
            + proof.rows.iter().map(|v| v.payload.len()).sum::<usize>())
    }

    fn copy_evidence<'py>(&self, py: Python<'py>) -> PyResult<EvidencePayload<'py>> {
        let meta = py
            .detach(|| -> Result<Vec<u8>> {
                let owner = self.proof.lock().unwrap();
                let proof = &owner.as_ref().ok_or(Error::Closed)?.evidence;
                let rows = proof
                    .rows
                    .iter()
                    .map(|row| (&row.correspondence, row.candidates))
                    .collect::<Vec<_>>();
                serde_json::to_vec(
                    &serde_json::json!({"facts":proof.facts,"boundary":proof.boundary,
                "outcome":proof.outcome,"rows":rows,"report":self.report}),
                )
                .map_err(|e| error(e.to_string()))
            })
            .map_err(python_error)?;
        let owner = self.proof.lock().unwrap();
        let proof = &owner
            .as_ref()
            .ok_or_else(|| python_error(Error::Closed))?
            .evidence;
        Ok((
            PyBytes::new(py, &meta),
            proof
                .events
                .iter()
                .map(|bytes| PyBytes::new(py, bytes))
                .collect(),
            proof
                .rows
                .iter()
                .map(|row| PyBytes::new(py, &row.payload))
                .collect(),
        ))
    }

    fn release_evidence(&self) {
        self.proof.lock().unwrap().take();
    }

    // Bytes are copied only after the existing sink has admitted the new allocation.
    fn release_members(&self) {
        self.output.lock().unwrap().take();
    }
}

#[derive(Default, Serialize)]
struct Stage {
    count: u64,
    wall_seconds: f64,
}
impl Stage {
    fn observe(&mut self, start: Instant) {
        self.count += 1;
        self.wall_seconds += start.elapsed().as_secs_f64();
    }
}

#[derive(Default, Serialize)]
struct Stages {
    journal_workers: usize,
    mailbox: Stage,
    pending: Stage,
    ack: Stage,
    terminal: Stage,
    parallel_batches: Stage,
    parallel_items: usize,
}

struct WriterThreads {
    pool: Option<rayon::ThreadPool>,
    handles: Vec<JoinHandle<()>>,
    _stacks: Lease,
}

impl WriterThreads {
    fn start(workers: usize, budget: &Arc<Budget>) -> Result<Self> {
        let stacks = workers
            .checked_mul(STACK_BYTES)
            .ok_or_else(|| Error::resource("rollout writer stack size overflow"))?;
        let mut owner = Self {
            pool: None,
            handles: Vec::new(),
            _stacks: budget.reserve(stacks)?,
        };
        owner.pool = Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|i| format!("gf-rollout-writer-{i}"))
                .spawn_handler(|thread| {
                    let mut builder = std::thread::Builder::new().stack_size(STACK_BYTES);
                    if let Some(name) = thread.name() {
                        builder = builder.name(name.to_owned());
                    }
                    owner.handles.push(builder.spawn(move || thread.run())?);
                    Ok(())
                })
                .build()
                .map_err(|e| {
                    Error::resource(format!("native Journal pool construction failed: {e}"))
                })?,
        );
        Ok(owner)
    }

    fn join(&mut self) -> Result<()> {
        // Drop the last owning Rayon pool first, which signals idle workers.
        self.pool.take();
        let mut failed = false;
        for handle in self.handles.drain(..) {
            failed |= handle.join().is_err();
        }
        ensure(!failed, "native Journal worker panicked")
    }
}

impl Drop for WriterThreads {
    fn drop(&mut self) {
        let _ = self.join();
    }
}

#[pyclass(frozen)]
pub(crate) struct RolloutSubmitWorkspace {
    lease: Mutex<Option<Lease>>,
}

#[pymethods]
impl RolloutSubmitWorkspace {
    fn close(&self) {
        self.lease.lock().unwrap().take();
    }
}

#[derive(Default)]
struct Contexts {
    active: HashMap<String, Active>,
    validation: Option<Arc<JournalValidation>>,
    lease: Option<NativeRolloutLease>,
}

impl Contexts {
    fn require_lease(&self) -> Result<()> {
        self.lease.as_ref().map(|_| ()).ok_or(Error::Closed)
    }
}

#[pyclass(frozen)]
pub(crate) struct RolloutWriter {
    mailbox: Arc<Mailbox>,
    templates: Mutex<Vec<Value>>,
    budget: Arc<Budget>,
    stages: Mutex<Stages>,
    workers: Mutex<Option<WriterThreads>>,
    // In the implicit drop path, the workers join before the mailbox lease drops.
    contexts: Mutex<Contexts>,
}

impl RolloutWriter {
    pub(crate) fn new(mailbox: Arc<Mailbox>, bytes: usize, workers: usize) -> Result<Self> {
        ensure(
            bytes != 0 && workers != 0,
            "actual rollout workspace and workers required",
        )?;
        let budget = Budget::new(bytes);
        let threads = if workers == 1 {
            None
        } else {
            Some(WriterThreads::start(workers, &budget)?)
        };
        let lease = mailbox.claim_native_rollout()?;
        Ok(Self {
            mailbox,
            templates: Mutex::new(Vec::new()),
            budget,
            stages: Mutex::new(Stages {
                journal_workers: workers,
                ..Stages::default()
            }),
            workers: Mutex::new(threads),
            contexts: Mutex::new(Contexts {
                lease: Some(lease),
                ..Contexts::default()
            }),
        })
    }

    fn join_workers(&self) -> Result<()> {
        if let Some(mut owner) = self.workers.lock().unwrap().take() {
            owner.join()?;
        }
        Ok(())
    }

    fn retire(&self, failed: bool) -> Result<()> {
        let mut contexts = self.contexts.lock().unwrap();
        if failed {
            contexts.active.clear();
        } else {
            ensure(
                contexts.active.is_empty(),
                "unfinished original Journal contexts",
            )?;
        }
        contexts.validation.take();
        self.templates.lock().unwrap().clear();
        self.join_workers()?;
        drop(contexts.lease.take());
        Ok(())
    }

    fn prepare(&self, active: &mut Active, record: &Arc<Record>) -> Result<()> {
        let raw = record.array.as_slice().expect("own contiguous original");
        ensure(
            format!("{:x}", Sha256::digest(raw)) == record.transport_sha256,
            "rollout original transport SHA differs",
        )?;
        if let Some(owned) = record.take_action_capture().and_then(|weak| weak.upgrade()) {
            active.writer.consume_owned(
                &record.kind,
                raw,
                &owned,
                &self.budget,
                &active.control,
            )?;
        } else {
            active
                .writer
                .consume(&record.kind, raw, &self.budget, &active.control)?;
        }
        ensure(
            active.writer.game_id() == (!record.game.is_empty()).then_some(record.game.as_str()),
            "rollout original record game differs from body",
        )?;
        active.retained.resize(active.writer.retained_bytes())
    }

    fn receipt(record: &Record) -> Receipt {
        Receipt {
            id: record.id,
            kind: record.kind.clone(),
            episode: record.episode.clone(),
            game: record.game.clone(),
            transport_sha256: record.transport_sha256.clone(),
            bytes: record.array.len(),
        }
    }

    fn prepare_parallel(
        &self,
        pool: &rayon::ThreadPool,
        records: Vec<Arc<Record>>,
        contexts: &mut HashMap<String, Active>,
        stages: &mut Stages,
        receipts: &mut Vec<Receipt>,
    ) -> Result<()> {
        for (i, record) in records.iter().enumerate() {
            ensure(
                contexts.contains_key(&record.episode)
                    && records[..i].iter().all(|r| r.episode != record.episode),
                "parallel rollout requires distinct original bound episodes",
            )?;
        }
        let began = Instant::now();
        let jobs: Vec<_> = records
            .iter()
            .map(|r| (r.clone(), contexts.remove(&r.episode).unwrap()))
            .collect();
        let prepared: Vec<_> = pool.install(|| {
            jobs.into_par_iter()
                .map(|(record, mut active)| {
                    let start = Instant::now();
                    let result = self.prepare(&mut active, &record);
                    (record, active, result, start.elapsed().as_secs_f64())
                })
                .collect()
        });
        let mut failure = None;
        for (record, mut active, result, seconds) in prepared {
            let stage = if record.kind == "pending" {
                &mut stages.pending
            } else {
                &mut stages.ack
            };
            stage.count += 1;
            stage.wall_seconds += seconds;
            if let Err(e) = result {
                failure.get_or_insert(e);
            }
            // Every result has joined and restored its original owner before cancellation.
            active.originals.push(record.clone());
            contexts.insert(record.episode.clone(), active);
        }
        stages.parallel_batches.observe(began);
        stages.parallel_items += records.len();
        if let Some(error) = failure {
            return Err(error);
        }
        self.mailbox.ack_prepared(&records)?;
        receipts.extend(records.iter().map(|r| Self::receipt(r)));
        Ok(())
    }

    fn receive(&self, deadline: f64) -> Result<RolloutTerminal> {
        // This private consumer starts only after the entire cohort has bound
        // its context. A retired handle cannot close a successor's mailbox.
        let mut contexts = self.contexts.lock().unwrap();
        contexts.require_lease()?;
        self.receive_active(deadline, &mut contexts.active)
            .inspect_err(|failure| self.mailbox.close(failure.clone()))
    }

    fn receive_active(
        &self,
        deadline: f64,
        contexts: &mut HashMap<String, Active>,
    ) -> Result<RolloutTerminal> {
        let mut stages = self.stages.lock().unwrap();
        let workers = self.workers.lock().unwrap();
        let pool = workers.as_ref().and_then(|owner| owner.pool.as_ref());
        let mut receipts = vec![];
        loop {
            let wait = Instant::now();
            let mut record = self.mailbox.next(deadline)?;
            stages.mailbox.observe(wait);
            if record.kind != "terminal" {
                ensure(
                    matches!(record.kind.as_str(), "pending" | "ack"),
                    "rollout record kind",
                )?;
                if let Some(pool) = pool {
                    let mut ready = Vec::with_capacity(pool.current_num_threads());
                    ready.push(record);
                    for _ in 1..pool.current_num_threads() {
                        let Some(next) = self.mailbox.next_before_terminal(deadline)? else {
                            break;
                        };
                        stages.mailbox.count += 1;
                        ready.push(next);
                    }
                    if ready.len() > 1 {
                        self.prepare_parallel(pool, ready, contexts, &mut stages, &mut receipts)?;
                        continue;
                    }
                    record = ready.pop().expect("one original");
                }
                let active = contexts
                    .get_mut(&record.episode)
                    .ok_or_else(|| error("rollout original lacks bound episode"))?;
                let began = Instant::now();
                self.prepare(active, &record)?;
                let stage = if record.kind == "pending" {
                    &mut stages.pending
                } else {
                    &mut stages.ack
                };
                stage.observe(began);
                self.mailbox.ack(&record, None)?;
                receipts.push(Self::receipt(&record));
                active.originals.push(record);
                continue;
            }

            let began = Instant::now();
            let mut active = contexts
                .remove(&record.episode)
                .ok_or_else(|| error("rollout terminal lacks bound episode"))?;
            self.prepare(&mut active, &record)?;
            let report = active
                .writer
                .terminal()
                .ok_or_else(|| error("verified original report required"))?
                .clone();
            let (members, evidence) = if active.writer.requires_training() {
                let numeric = record
                    .training
                    .as_ref()
                    .ok_or_else(|| error("actual terminal numeric Episode required"))?;
                let (members, evidence) =
                    active
                        .writer
                        .into_completed(numeric, &self.budget, &active.control)?;
                (members, Some(evidence))
            } else {
                (active.writer.into_members()?, None)
            };
            // Both backings are live; resize their aggregate before splitting
            // the same reservation into their independently retired owners.
            let member_bytes: usize = members
                .iter()
                .map(|(_, bytes)| bytes.capacity_bytes())
                .sum();
            let evidence_bytes = evidence.as_ref().map_or(0, LearnerEvidence::retained_bytes);
            let total = member_bytes
                .checked_add(evidence_bytes)
                .ok_or_else(|| Error::resource("rollout terminal recording bytes overflow"))?;
            active.retained.resize(total)?;
            let proof = evidence
                .map(|evidence| -> Result<_> {
                    let lease = active.retained.split(evidence_bytes)?;
                    Ok(EvidenceOutput {
                        evidence,
                        _retained: lease,
                    })
                })
                .transpose()?;
            stages.terminal.observe(began);
            return Ok(RolloutTerminal {
                record,
                receipts,
                _originals: active.originals,
                report,
                output: Mutex::new(Some(Output {
                    members,
                    _retained: active.retained,
                })),
                proof: Mutex::new(proof),
            });
        }
    }
}

#[pyclass(frozen)]
pub(crate) struct RolloutPack {
    bytes: BudgetVec<u8>,
    #[pyo3(get)]
    raw_bytes: usize,
}

#[pymethods]
impl RolloutPack {
    #[getter]
    fn size(&self) -> usize {
        self.bytes.len()
    }
    fn copy<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }
}

#[pymethods]
impl RolloutWriter {
    fn add_header_template(&self, py: Python<'_>, encoded: &[u8]) -> PyResult<usize> {
        py.detach(|| -> Result<usize> {
            let contexts = self.contexts.lock().unwrap();
            contexts.require_lease()?;
            let header: Value =
                serde_json::from_slice(encoded).map_err(|e| error(e.to_string()))?;
            ensure(
                header.is_object() && header["kind"] == "header" && header["schema_version"] == 6,
                "validated Journal6 template required",
            )?;
            let mut templates = self.templates.lock().unwrap();
            let id = templates.len();
            templates.push(header);
            Ok(id)
        })
        .map_err(python_error)
    }

    fn finish_header_templates(&self, py: Python<'_>) {
        py.detach(|| self.templates.lock().unwrap().clear());
    }

    fn bind(&self, py: Python<'_>, encoded: &[u8]) -> PyResult<()> {
        py.detach(|| -> Result<()> {
            let mut contexts = self.contexts.lock().unwrap();
            contexts.require_lease()?;
            let ctx: Context = serde_json::from_slice(encoded).map_err(|e| error(e.to_string()))?;
            ensure(monotonic() < ctx.deadline, "rollout context expired")?;
            let headers = match (ctx.headers, ctx.template_ids, ctx.run, ctx.runtime_evidence) {
                (Some(headers), None, None, None) => headers,
                (None, Some(ids), Some(run), Some(receipt)) => {
                    ensure(
                        !run.is_empty() && !ctx.episode.is_empty(),
                        "rollout header identity required",
                    )?;
                    let templates = self.templates.lock().unwrap();
                    let instantiate = |i: usize| -> Result<Value> {
                        let mut h = templates
                            .get(ids[i])
                            .ok_or_else(|| error("unknown rollout header template"))?
                            .clone();
                        ensure(
                            receipt["action_contract"] == h["action_contract"],
                            "rollout header receipt differs",
                        )?;
                        let seed = ctx.policies[i]["seed"]
                            .as_u64()
                            .ok_or_else(|| error("planned rollout seed required"))?;
                        h["run_id"] = serde_json::json!(run);
                        h["episode_id"] = serde_json::json!(ctx.episode);
                        h["bank_episode"] = serde_json::json!(ctx.episode);
                        h["viewer"] = serde_json::json!(i + 1);
                        h["sampling_seed"] = serde_json::json!(seed);
                        h["runtime_evidence"] = receipt.clone();
                        Ok(h)
                    };
                    [instantiate(0)?, instantiate(1)?]
                }
                _ => return Err(error("ambiguous rollout header registration")),
            };
            ensure(
                headers.iter().all(|h| h["episode_id"] == ctx.episode),
                "rollout header episode differs",
            )?;
            ensure(
                !contexts.active.contains_key(&ctx.episode),
                "rollout context already bound",
            )?;
            let control = Control::new(ctx.deadline)?;
            control.check()?;
            let validation = match &contexts.validation {
                Some(validation) => validation.clone(),
                None => Arc::new(JournalValidation::new(&self.budget, &control)?),
            };
            let writer = EpisodeWriter::new(
                validation.clone(),
                headers,
                ctx.trial,
                ctx.policies,
                ctx.numeric_costs,
                ctx.clock,
                ctx.learner_viewer,
                ctx.deadline,
                ctx.rules,
                ctx.classification,
            )?;
            let retained = self.mailbox.account.reserve(0, true)?;
            control.check()?;
            contexts.validation = Some(validation);
            contexts.active.insert(
                ctx.episode,
                Active {
                    writer,
                    originals: vec![],
                    retained,
                    control,
                },
            );
            Ok(())
        })
        .map_err(python_error)
    }

    fn next_terminal(&self, py: Python<'_>, deadline: f64) -> PyResult<RolloutTerminal> {
        py.detach(|| self.receive(deadline)).map_err(python_error)
    }

    fn stage_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| serde_json::to_vec(&*self.stages.lock().unwrap()))
            .map_err(|e| python_error(error(e.to_string())))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.retire(false)).map_err(python_error)
    }

    fn retire_failed_drain(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.retire(true)).map_err(python_error)
    }

    fn reserve_submit_workspace(&self, raw_size: usize) -> PyResult<RolloutSubmitWorkspace> {
        // Pack decompression and its decoded index can overlap native work.
        // Keep the existing shared read-workspace admission for this path.
        let bytes = raw_size
            .checked_mul(2)
            .ok_or_else(|| python_error(Error::resource("Pack read workspace overflow")))?;
        let lease = self.budget.reserve(bytes).map_err(python_error)?;
        Ok(RolloutSubmitWorkspace {
            lease: Mutex::new(Some(lease)),
        })
    }

    fn pack<'py>(
        &self,
        py: Python<'py>,
        members: Vec<(String, String, Bound<'py, PyBytes>)>,
        raw_limit: usize,
        deadline: f64,
    ) -> PyResult<RolloutPack> {
        let bytes = members
            .iter()
            .map(|(kind, name, data)| (kind.as_str(), name.as_str(), data.as_bytes()))
            .collect::<Vec<_>>();
        let control = Control::new(deadline).map_err(python_error)?;
        let (pack, uncompressed) = py
            .detach(|| crate::rollout_group_pack::encode(&bytes, raw_limit, &self.budget, &control))
            .map_err(python_error)?;
        Ok(RolloutPack {
            bytes: pack,
            raw_bytes: uncompressed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_rollout_lease_is_exclusive_and_stale_reader_cannot_close_successor() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let writer = RolloutWriter::new(mailbox.clone(), 64 << 20, 1).unwrap();
        assert!(mailbox.native_rollout_active());
        assert!(RolloutWriter::new(mailbox.clone(), 64 << 20, 1).is_err());
        writer.retire(false).unwrap();
        assert!(!mailbox.native_rollout_active());

        let successor = RolloutWriter::new(mailbox.clone(), 64 << 20, 1).unwrap();
        assert!(matches!(writer.receive(f64::NAN), Err(Error::Closed)));
        assert!(mailbox.state.lock().unwrap().closed.is_none());
        assert!(matches!(
            writer.contexts.lock().unwrap().require_lease(),
            Err(Error::Closed)
        ));
        writer.retire(true).unwrap();
        drop(writer);
        assert!(mailbox.native_rollout_active());
        assert!(successor.receive(f64::NAN).is_err());
        assert!(mailbox.state.lock().unwrap().closed.is_some());
        drop(successor);
        assert!(!mailbox.native_rollout_active());
    }

    #[test]
    fn failed_worker_join_keeps_lease_until_writer_drops() {
        let mailbox = Mailbox::new(1024, 1).unwrap();
        let writer = RolloutWriter::new(mailbox.clone(), 64 << 20, 1).unwrap();
        *writer.workers.lock().unwrap() = Some(WriterThreads {
            pool: None,
            handles: vec![std::thread::spawn(|| panic!("test writer panic"))],
            _stacks: writer.budget.reserve(0).unwrap(),
        });
        assert!(writer.retire(false).is_err());
        assert!(mailbox.native_rollout_active());
        assert!(RolloutWriter::new(mailbox.clone(), 64 << 20, 1).is_err());
        drop(writer);
        assert!(!mailbox.native_rollout_active());
        assert!(RolloutWriter::new(mailbox.clone(), 64 << 20, 1).is_ok());
    }

    #[test]
    fn retirement_refunds_shared_validation_even_without_a_reader() {
        for failed in [false, true] {
            let writer = RolloutWriter::new(Mailbox::new(1024, 1).unwrap(), 64 << 20, 1).unwrap();
            let control = Control::new(f64::MAX).unwrap();
            let validation = Arc::new(JournalValidation::new(&writer.budget, &control).unwrap());
            let borrowed = Arc::downgrade(&validation);
            writer.contexts.lock().unwrap().validation = Some(validation);
            assert!(writer.budget.used() > 0);
            writer.retire(failed).unwrap();
            assert!(borrowed.upgrade().is_none());
            assert_eq!(writer.budget.used(), 0);
        }
    }
}
