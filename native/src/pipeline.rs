//! Complete-episode numeric spooling and all-or-nothing structural validation.
use crate::json::ValueExt;
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    error::{Error, Result},
    feature::PreparedDecision,
    journal::{self, string},
    json::{self, Value},
    numeric_io::{read_f64, read_i64, write_f64, write_i64},
    projection,
    session::Session,
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);
pub struct Episode {
    pub path: PathBuf,
    pub run: String,
    pub game: String,
    pub viewer: i64,
    pub split: i64,
    pub winner: i64,
    pub rewards: [f64; 2],
    pub decisions: usize,
    pub candidates: usize,
    pub bytes: usize,
    pub export_bytes: usize,
    pub order: usize,
    pub index: Option<crate::index::IndexInfo>,
    _scratch: Lease,
    _metadata: Lease,
}
pub struct Stage {
    pub directory: PathBuf,
    pub episodes: BudgetVec<Episode>,
    pub proof: Value,
    pub cursor: usize,
    pub scratch: Arc<Budget>,
    closed: bool,
    _index_output: Option<Lease>,
}
impl Stage {
    #[cfg(test)]
    pub fn test_failed_close(budget: &Arc<Budget>) -> Self {
        Self {
            directory: std::env::temp_dir().join(format!(
                "gf-native-absent-{}-{}",
                std::process::id(),
                crate::budget::monotonic().to_bits()
            )),
            episodes: BudgetVec::new(budget),
            proof: Value::Null,
            cursor: 0,
            scratch: Budget::new(1024 * 1024),
            closed: false,
            _index_output: None,
        }
    }
    pub fn close(&mut self) -> Result<()> {
        if !self.closed {
            fs::remove_dir_all(&self.directory)?;
            self.closed = true;
            self.episodes.clear();
        }
        Ok(())
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        if !self.closed {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}
fn temporary() -> Result<PathBuf> {
    for _ in 0..100 {
        let name = format!(
            "gf-native-preparation-{}-{}-{}",
            std::process::id(),
            crate::budget::monotonic().to_bits(),
            NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(Error::resource(
        "could not create fresh native scratch directory",
    ))
}

struct Identity {
    run: String,
    game: String,
    viewer: i128,
    _metadata: Lease,
}
struct Lineage {
    seed: String,
    ancestry: Option<String>,
    split: i64,
    _metadata: Lease,
}

struct ProofRecord {
    path: String,
    offset: u64,
    length: usize,
    _scratch: Lease,
    _metadata: Lease,
}
struct ProofLedger {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    records: BudgetVec<ProofRecord>,
    offset: u64,
    _buffer: Lease,
    telemetry: Arc<crate::telemetry::Telemetry>,
}
impl ProofLedger {
    fn new(
        directory: &Path,
        budget: &Arc<Budget>,
        telemetry: &Arc<crate::telemetry::Telemetry>,
    ) -> Result<Self> {
        let path = directory.join("source-proofs.bin");
        let buffer = budget.reserve(32768)?;
        let writer = BufWriter::with_capacity(
            32768,
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?,
        );
        Ok(Self {
            path,
            writer: Some(writer),
            records: BudgetVec::new(budget),
            offset: 0,
            _buffer: buffer,
            telemetry: telemetry.clone(),
        })
    }
    fn append(&mut self, path: &Path, encoded: &[u8], scratch: &Arc<Budget>) -> Result<()> {
        self.records.reserve(1)?;
        let record = ProofRecord {
            _metadata: self
                .records
                .budget()
                .reserve(crate::session::path_text(path)?.len() + size_of::<String>())?,
            path: (crate::session::path_text(path)?).to_owned(),
            offset: self.offset,
            length: encoded.len(),
            _scratch: scratch.reserve(encoded.len())?,
        };
        self.writer
            .as_mut()
            .ok_or(Error::Closed)?
            .write_all(encoded)?;
        self.offset += encoded.len() as u64;
        self.telemetry.stage_write(encoded.len());
        self.records.push(record)
    }
    fn hash_range(&mut self, start: usize, end: usize, control: &Control) -> Result<String> {
        self.writer.as_mut().ok_or(Error::Closed)?.flush()?;
        let mut ordered = BudgetVec::with_capacity(self.records.budget(), end - start)?;
        for record in &self.records[start..end] {
            ordered.push(record)?;
        }
        // Participant source paths are unique in an eligible collection;
        // duplicate source identities are rejected before proof publication.
        crate::budget::sort_unstable_by(
            &mut ordered,
            |a, b| a.path.as_bytes().cmp(b.path.as_bytes()),
            control,
        )?;
        let mut file = File::open(&self.path)?;
        let mut hasher = Sha256::new();
        hasher.update(b"[");
        let mut block = [0u8; 32768];
        for (index, record) in ordered.iter().enumerate() {
            control.check()?;
            if index > 0 {
                hasher.update(b",");
            }
            file.seek(SeekFrom::Start(record.offset))?;
            let mut remaining = record.length;
            while remaining > 0 {
                control.check()?;
                let count = remaining.min(block.len());
                file.read_exact(&mut block[..count])?;
                self.telemetry.stage_read(count);
                hasher.update(&block[..count]);
                remaining -= count;
            }
        }
        hasher.update(b"]");
        Ok(format!("{:x}", hasher.finalize()))
    }
    fn close(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
            drop(writer);
            fs::remove_file(&self.path)?;
            self.records.clear();
        }
        Ok(())
    }
}
struct IdentityLedger {
    paths: [PathBuf; 2],
    writers: Option<[BufWriter<File>; 2]>,
    leases: BudgetVec<Lease>,
    _buffers: Lease,
    telemetry: Arc<crate::telemetry::Telemetry>,
}
impl IdentityLedger {
    fn new(
        directory: &Path,
        budget: &Arc<Budget>,
        telemetry: &Arc<crate::telemetry::Telemetry>,
    ) -> Result<Self> {
        let paths = [
            directory.join("train-identities.jsonl"),
            directory.join("validation-identities.jsonl"),
        ];
        let buffers = budget.reserve(65536)?;
        let writers = [
            BufWriter::with_capacity(
                32768,
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&paths[0])?,
            ),
            BufWriter::with_capacity(
                32768,
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&paths[1])?,
            ),
        ];
        Ok(Self {
            paths,
            writers: Some(writers),
            leases: BudgetVec::new(budget),
            _buffers: buffers,
            telemetry: telemetry.clone(),
        })
    }
    fn append(
        &mut self,
        split: usize,
        header: &Value,
        decision: &Value,
        budget: &Arc<Budget>,
        scratch: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        let values = [
            json::text(if split == 0 { "train" } else { "validation" }),
            json::copy(header.require("run_id")?, control)?,
            json::copy(header.require("game_id")?, control)?,
            json::copy(header.require("viewer")?, control)?,
            json::copy(decision.require("decision")?, control)?,
        ];
        let encoded = json::canonical(&json::array(values), budget, control)?;
        self.leases.reserve(1)?;
        let lease = scratch.reserve(encoded.len() + 1)?;
        let writer = &mut self.writers.as_mut().ok_or(Error::Closed)?[split];
        writer.write_all(&encoded)?;
        writer.write_all(b"\n")?;
        self.telemetry.stage_write(encoded.len() + 1);
        self.leases.push(lease)
    }
    fn finish(&mut self, control: &Control) -> Result<String> {
        if let Some(mut writers) = self.writers.take() {
            for writer in &mut writers {
                writer.flush()?;
            }
            drop(writers);
        }
        let mut hasher = Sha256::new();
        let mut bytes = [0u8; 32768];
        for path in &self.paths {
            let mut file = File::open(path)?;
            loop {
                control.check()?;
                let count = file.read(&mut bytes)?;
                if count == 0 {
                    break;
                }
                hasher.update(&bytes[..count]);
                self.telemetry.stage_read(count);
            }
            drop(file);
            fs::remove_file(path)?;
        }
        self.leases.clear();
        Ok(format!("{:x}", hasher.finalize()))
    }
}

pub fn scan(session: &mut Session) -> Result<Stage> {
    session.check()?;
    if !session.authorized {
        return Err(Error::contract(
            "native session lacks schedule authorization",
        ));
    }
    let budget = &session.budget;
    let control = &session.control;
    let scratch = Budget::new(session.settings.scratch);
    let mut stage = Stage {
        directory: temporary()?,
        episodes: BudgetVec::new(budget),
        proof: Value::Null,
        cursor: 0,
        scratch,
        closed: false,
        _index_output: None,
    };
    let mut proof_ledger = ProofLedger::new(&stage.directory, budget, &session.telemetry)?;
    let mut source_count = 0usize;
    let mut identity_ledger = IdentityLedger::new(&stage.directory, budget, &session.telemetry)?;
    let mut identities = BudgetVec::new(budget);
    let mut lineages = BudgetVec::new(budget);
    let mut counts = [0usize; 2];
    let mut action_hash = None::<String>;
    for collection in session.collections.iter() {
        control.check()?;
        projection::validate_manifest(collection, budget, control, &session.sources)?;
        let m = &collection.manifest.root;
        let episode_index = projection::EpisodeIndex::new(m, budget, control)?;
        let mut episodes = BudgetVec::new(budget);
        for episode in m.require("episodes")?.array()? {
            let directory = collection.directory.join(string(episode, "episode_id")?);
            episodes.push(directory)?;
        }
        crate::budget::sort_unstable_by(
            &mut episodes,
            |a, b| {
                a.as_os_str()
                    .as_encoded_bytes()
                    .iter()
                    .chain(b"/")
                    .cmp(b.as_os_str().as_encoded_bytes().iter().chain(b"/"))
            },
            control,
        )?;
        verify_journal_inventory(&collection.directory, &episodes, budget, control)?;
        let collection_start = proof_ledger.records.len();
        let mut reader = crate::parallel::EpisodeReader::new(
            &episodes,
            session.settings.workers,
            session.settings.in_flight,
            budget,
            control,
            session.settings.file,
            session.settings.record,
        );
        while let Some((directory, data)) = reader.next(&mut session.sources)? {
            control.check()?;
            let id = directory
                .file_name()
                .and_then(|v| v.to_str())
                .ok_or_else(|| Error::contract("invalid episode path"))?;
            let links = episode_index.links(id, control)?;
            let crate::parallel::EpisodeData {
                records1,
                records2,
                replay1,
                replay2,
            } = data;
            let journal1 = {
                let _timer = session
                    .telemetry
                    .phase_timer(crate::telemetry::Phase::JournalInputValidation);
                journal::validate(&records1, budget, control)?
            };
            let journal2 = {
                let _timer = session
                    .telemetry
                    .phase_timer(crate::telemetry::Phase::JournalInputValidation);
                journal::validate(&records2, budget, control)?
            };
            let first_viewer = journal1.header.require("viewer")?.integer(1, 2)? as usize;
            let second_viewer = journal2.header.require("viewer")?.integer(1, 2)? as usize;
            if first_viewer == second_viewer {
                return Err(Error::contract("duplicate source participant identity"));
            }
            let by_viewer = if first_viewer == 1 {
                [&journal1, &journal2]
            } else {
                [&journal2, &journal1]
            };
            let journals = [&journal1, &journal2];
            let replays = [&replay1.root, &replay2.root];
            for (slot, journal) in journals.iter().enumerate() {
                control.check()?;
                let viewer = journal.header.require("viewer")?.integer(1, 2)? as usize;
                let replay = replays[viewer - 1];
                let source = directory.join(format!("player-{}.jsonl", slot + 1));
                let metadata = projection::source_metadata(
                    collection,
                    &source,
                    replay,
                    journal,
                    &links,
                    &session.sources,
                    budget,
                    control,
                )?;
                let _proof_timer = session
                    .telemetry
                    .phase_timer(crate::telemetry::Phase::ProofStaging);
                let encoded = json::canonical(&metadata, budget, control)?;
                proof_ledger.append(&std::fs::canonicalize(&source)?, &encoded, &stage.scratch)?;
                source_count += 1;
                identities.push(Identity {
                    _metadata: budget.reserve(
                        string(journal.header, "run_id")?.len()
                            + string(journal.header, "game_id")?.len()
                            + 2 * size_of::<String>(),
                    )?,
                    run: (string(journal.header, "run_id")?).to_owned(),
                    game: (string(journal.header, "game_id")?).to_owned(),
                    viewer: viewer as i128,
                })?;
                let lineage = metadata.require("lineage")?;
                lineages.push(Lineage {
                    _metadata: budget.reserve(
                        string(lineage, "seed")?.len()
                            + if lineage.require("ancestry")?.is_null() {
                                0
                            } else {
                                string(lineage, "ancestry")?.len()
                            }
                            + 2 * size_of::<String>(),
                    )?,
                    seed: (string(lineage, "seed")?).to_owned(),
                    ancestry: if lineage.require("ancestry")?.is_null() {
                        None
                    } else {
                        Some((string(lineage, "ancestry")?).to_owned())
                    },
                    split: if string(lineage, "split")? == "train" {
                        0
                    } else {
                        1
                    },
                })?;
                let current = string(journal.header, "action_config_hash")?;
                if action_hash
                    .as_ref()
                    .is_some_and(|prior| prior.as_bytes() != current.as_bytes())
                {
                    return Err(Error::contract(
                        "frozen sources must share identical action conditions",
                    ));
                }
                if action_hash.is_none() {
                    action_hash = Some((current).to_owned());
                }
            }
            let learner = links.spec.require("learner_viewer")?.integer(1, 2)? as usize;
            let journal = by_viewer[learner - 1];
            let outcomes = [
                by_viewer[0]
                    .end
                    .ok_or_else(|| Error::contract("missing journal end"))?,
                by_viewer[1]
                    .end
                    .ok_or_else(|| Error::contract("missing journal end"))?,
            ];
            if !outcomes[0]
                .require("outcome")?
                .eq_python(outcomes[1].require("outcome")?)
            {
                return Err(Error::contract("participant terminal facts disagree"));
            }
            let outcome = outcomes[0].require("outcome")?;
            let winner = if outcome.string().is_ok_and(|s| s == "draw") {
                0
            } else {
                outcome.require("win")?.require("winner")?.integer(1, 2)? as i64
            };
            let rewards = [
                outcomes[0].require("terminal_reward")?.integer(-1, 1)? as f64,
                outcomes[1].require("terminal_reward")?.integer(-1, 1)? as f64,
            ];
            let split = if string(m, "split")? == "train" { 0 } else { 1 };
            let decisions = journal.decisions.len();
            let mut candidates = 0usize;
            for decision in journal.decisions.iter() {
                control.check()?;
                identity_ledger.append(
                    split as usize,
                    journal.header,
                    decision,
                    budget,
                    &stage.scratch,
                    control,
                )?;
                let length = if session.online {
                    decision
                        .require("candidates")?
                        .require("actions")?
                        .array()?
                        .len()
                } else {
                    1
                };
                candidates = candidates
                    .checked_add(length)
                    .ok_or_else(|| Error::resource("candidate count overflow"))?;
            }
            let width = if ["prepare", "ppo"].contains(&session.operation.as_str()) {
                session.config.width()
            } else {
                0
            };
            let bytes = candidates
                .checked_add(decisions)
                .and_then(|n| n.checked_mul(width))
                .and_then(|n| n.checked_add(7 * decisions + 1))
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| Error::resource("episode numeric size overflow"))?;
            let scratch_lease = stage.scratch.reserve(bytes)?;
            let path = stage
                .directory
                .join(format!("episode-{:08}.bin", stage.episodes.len()));
            let _write_buffer = budget.reserve(32768)?;
            let mut file = BufWriter::with_capacity(
                32768,
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)?,
            );
            let mut offsets = BudgetVec::with_capacity(budget, decisions + 1)?;
            offsets.push(0i64)?;
            let mut selected = BudgetVec::with_capacity(budget, decisions)?;
            let mut logprob = BudgetVec::with_capacity(budget, decisions)?;
            let mut values = BudgetVec::with_capacity(budget, decisions)?;
            let mut logprob_present = BudgetVec::with_capacity(budget, decisions)?;
            let mut value_present = BudgetVec::with_capacity(budget, decisions)?;
            let mut state_features = BudgetVec::with_capacity(budget, decisions * width)?;
            let mut candidate_counts = BudgetVec::with_capacity(budget, decisions)?;
            let mut row = BudgetVec::with_capacity(budget, width)?;
            for _ in 0..width {
                row.push(0.)?;
            }
            let mut candidate_count = 0usize;
            for decision in journal.decisions.iter() {
                control.check()?;
                let input = decision.require("input")?;
                let selection = decision.require("selection")?;
                let saved_index =
                    selection.require("index")?.integer(0, u64::MAX as i128)? as usize;
                let old_logprob = optional_number(selection.require("logprob")?)?;
                let old_value = optional_number(selection.require("value")?)?;
                if session.online
                    && (selection.require("logprob")?.is_null()
                        || selection.require("value")?.is_null()
                        || !old_logprob.is_finite()
                        || !old_value.is_finite())
                {
                    return Err(Error::contract(
                        "saved PPO probability/value evidence required",
                    ));
                }
                let prepared = if width > 0 {
                    let _timer = session
                        .telemetry
                        .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                    Some(PreparedDecision::new(
                        input,
                        &session.config,
                        budget,
                        control,
                    )?)
                } else {
                    None
                };
                if let Some(prepared) = &prepared {
                    let _timer = session
                        .telemetry
                        .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                    prepared.state(&mut row, control)?;
                    state_features.extend_from_slice(&row)?;
                }
                if session.online {
                    for action in decision
                        .require("candidates")?
                        .require("actions")?
                        .array()?
                    {
                        {
                            let _timer = session
                                .telemetry
                                .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                            prepared
                                .as_ref()
                                .ok_or_else(|| Error::contract("PPO feature context missing"))?
                                .encode(action, &mut row, control)?;
                        }
                        let _timer = session
                            .telemetry
                            .phase_timer(crate::telemetry::Phase::ProofStaging);
                        write_f64(&mut file, &row, control)?;
                        candidate_count += 1;
                    }
                    selected.push(saved_index as i64)?;
                } else {
                    if let Some(prepared) = &prepared {
                        let _timer = session
                            .telemetry
                            .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                        prepared.encode(
                            decision.require("command")?.require("action")?,
                            &mut row,
                            control,
                        )?;
                    }
                    let _timer = session
                        .telemetry
                        .phase_timer(crate::telemetry::Phase::ProofStaging);
                    write_f64(&mut file, &row, control)?;
                    candidate_count += 1;
                    selected.push(saved_index as i64)?;
                }
                offsets.push(candidate_count as i64)?;
                logprob.push(old_logprob)?;
                values.push(old_value)?;
                logprob_present.push((!selection.require("logprob")?.is_null()) as i64)?;
                value_present.push((!selection.require("value")?.is_null()) as i64)?;
                candidate_counts.push(
                    decision
                        .require("candidates")?
                        .require("actions")?
                        .array()?
                        .len() as i64,
                )?;
            }
            let _staging_timer = session
                .telemetry
                .phase_timer(crate::telemetry::Phase::ProofStaging);
            write_f64(&mut file, &state_features, control)?;
            write_i64(&mut file, &offsets, control)?;
            write_i64(&mut file, &selected, control)?;
            write_f64(&mut file, &logprob, control)?;
            write_f64(&mut file, &values, control)?;
            write_i64(&mut file, &logprob_present, control)?;
            write_i64(&mut file, &value_present, control)?;
            write_i64(&mut file, &candidate_counts, control)?;
            file.flush()?;
            if file.get_ref().metadata()?.len() != bytes as u64 {
                return Err(Error::contract("numeric staging length mismatch"));
            }
            session.telemetry.stage_write(bytes);
            drop(file);
            let metadata_lease = budget.reserve(
                path.as_os_str().len()
                    + string(journal.header, "run_id")?.len()
                    + string(journal.header, "game_id")?.len()
                    + 256,
            )?;
            let order = links.ordinal;
            let index = if session.operation == "index" {
                Some(crate::index::IndexInfo::new(
                    collection,
                    &links,
                    journal.header,
                    rewards[learner - 1],
                    &session
                        .dataset
                        .as_ref()
                        .ok_or_else(|| Error::contract("index requires a frozen dataset"))?
                        .root,
                    &session.sources,
                    budget,
                    control,
                )?)
            } else {
                None
            };
            let export_bytes = if let Some(info) = &index {
                bytes
                    .checked_add(info.producer_bytes(decisions)?)
                    .ok_or_else(|| Error::resource("index export size overflow"))?
            } else {
                bytes
            };
            stage.episodes.push(Episode {
                path,
                run: (string(journal.header, "run_id")?).to_owned(),
                game: (string(journal.header, "game_id")?).to_owned(),
                viewer: learner as i64,
                split,
                winner,
                rewards,
                decisions,
                candidates,
                bytes,
                export_bytes,
                order,
                index,
                _scratch: scratch_lease,
                _metadata: metadata_lease,
            })?;
            counts[split as usize] += decisions;
        }
        if let Some(expected) = &collection.expected_source {
            if proof_ledger.hash_range(collection_start, proof_ledger.records.len(), control)?
                != *expected
            {
                return Err(Error::contract(
                    "offline source journal/Replay proof changed since PPO update",
                ));
            }
        }
    }
    if session.online {
        // One staged learner episode per unique manifest position.
        crate::budget::sort_unstable_by(
            &mut stage.episodes,
            |a, b| a.order.cmp(&b.order),
            control,
        )?;
    }
    if session.operation == "index" {
        crate::budget::sort_unstable_by(
            &mut stage.episodes,
            |a, b| {
                a.run
                    .as_bytes()
                    .cmp(b.run.as_bytes())
                    .then(a.game.as_bytes().cmp(b.game.as_bytes()))
                    .then(a.viewer.cmp(&b.viewer))
            },
            control,
        )?;
    }
    check_global(&mut identities, &lineages, budget, control)?;
    let _proof_timer = session
        .telemetry
        .phase_timer(crate::telemetry::Phase::ProofStaging);
    let source_hash = proof_ledger.hash_range(0, proof_ledger.records.len(), control)?;
    proof_ledger.close()?;
    let identity_hash = identity_ledger.finish(control)?;
    if !session.online {
        let mut expected = 0usize;
        for parent in session.parents.iter() {
            expected += parent
                .state
                .root
                .require("accepted_decisions")?
                .integer(0, u64::MAX as i128)? as usize;
        }
        if counts[0] == 0 || counts[1] == 0 || counts[0] != expected {
            return Err(Error::contract(
                "fixed dataset complete learner decision count mismatch",
            ));
        }
    } else if counts[0] == 0 {
        return Err(Error::contract(
            "PPO batch contains no accepted learner decisions",
        ));
    }
    session.sources.verify(control)?;
    let frozen = if session.online {
        Value::Null
    } else {
        crate::session::dataset_manifest(
            session,
            &source_hash,
            counts,
            action_hash
                .as_ref()
                .ok_or_else(|| Error::contract("no action conditions"))?
                .as_str(),
        )?
    };
    if let Some(dataset) = &session.dataset {
        if !frozen.eq_python(&dataset.root) {
            return Err(Error::contract(
                "frozen dataset original evidence/content changed",
            ));
        }
    }
    let index_digest = if session.operation == "index" {
        let (digest, scratch) = crate::index::content_digest(&stage.episodes, budget, control)?;
        stage._index_output = Some(stage.scratch.reserve(scratch)?);
        json::text(&digest)
    } else {
        Value::Null
    };
    stage.proof = json::object([
        ("structural_valid", Value::Bool(true)),
        ("learning_valid", Value::Bool(false)),
        ("source_set_sha256", json::text(&source_hash)),
        ("train_decisions", json::integer(counts[0] as i128)),
        ("validation_decisions", json::integer(counts[1] as i128)),
        ("games", json::integer(stage.episodes.len() as i128)),
        ("sources", json::integer(source_count as i128)),
        (
            "action_config_hash",
            json::text(
                action_hash
                    .as_ref()
                    .ok_or_else(|| Error::contract("no source action conditions"))?
                    .as_str(),
            ),
        ),
        ("dataset_manifest", frozen),
        (
            "manifest_sha256",
            if session.online {
                json::text(session.sources.digest(
                    &session.collections[0].directory.join("manifest.json"),
                    control,
                )?)
            } else {
                Value::Null
            },
        ),
        ("sample_identity_sha256", json::text(&identity_hash)),
        ("index_row_content_sha256", index_digest),
        (
            "index_io_reservation",
            json::text("arrow-index-io-reservation-v1"),
        ),
    ]);
    Ok(stage)
}
fn optional_number(value: &Value) -> Result<f64> {
    if value.is_null() {
        Ok(0.)
    } else {
        value.number()
    }
}
fn verify_journal_inventory(
    directory: &Path,
    episodes: &[PathBuf],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    let estimate = episodes
        .len()
        .checked_mul(128)
        .and_then(|n| n.checked_add(1024))
        .ok_or_else(|| Error::resource("episode inventory index capacity overflow"))?;
    let _index_lease = budget.reserve(estimate)?;
    let mut expected = HashSet::new();
    expected
        .try_reserve(episodes.len())
        .map_err(|_| Error::resource("episode inventory index allocation failed"))?;
    for path in episodes {
        control.check()?;
        if !expected.insert(path.as_path()) {
            return Err(Error::contract("duplicate episode source directory"));
        }
    }
    let mut count = 0usize;
    for entry in fs::read_dir(directory)? {
        control.check()?;
        let entry = entry?;
        if !entry.path().is_dir() {
            continue;
        }
        for child in fs::read_dir(entry.path())? {
            control.check()?;
            let child = child?;
            let name = child.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("player-") && name.ends_with(".jsonl") {
                if !["player-1.jsonl", "player-2.jsonl"].contains(&name.as_ref())
                    || !expected.contains(entry.path().as_path())
                {
                    return Err(Error::contract(
                        "missing/extra original participant journals",
                    ));
                }
                count += 1;
            }
        }
    }
    if count != episodes.len() * 2 {
        return Err(Error::contract(
            "all original participant journals required",
        ));
    }
    Ok(())
}
fn check_global(
    identities: &mut [Identity],
    lineages: &[Lineage],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    crate::budget::sort_unstable_by(
        identities,
        |a, b| {
            a.run
                .as_bytes()
                .cmp(b.run.as_bytes())
                .then(a.game.as_bytes().cmp(b.game.as_bytes()))
                .then(a.viewer.cmp(&b.viewer))
        },
        control,
    )?;
    for pair in identities.windows(2) {
        control.check()?;
        if pair[0].run == pair[1].run
            && pair[0].game == pair[1].game
            && pair[0].viewer == pair[1].viewer
        {
            return Err(Error::contract("duplicate source participant identity"));
        }
    }
    let mut parents = BudgetVec::with_capacity(budget, lineages.len())?;
    for i in 0..lineages.len() {
        parents.push(i)?;
    }
    let mut keys = BudgetVec::with_capacity(budget, lineages.len() * 2)?;
    for (index, lineage) in lineages.iter().enumerate() {
        control.check()?;
        keys.push((0u8, lineage.seed.as_bytes(), index))?;
        if let Some(ancestry) = &lineage.ancestry {
            keys.push((1, ancestry.as_bytes(), index))?;
        }
    }
    // Equal lineage keys join the same DSU component; their order is immaterial.
    crate::budget::sort_unstable_by(&mut keys, |a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)), control)?;
    fn root(parents: &mut [usize], mut i: usize) -> usize {
        while parents[i] != i {
            parents[i] = parents[parents[i]];
            i = parents[i];
        }
        i
    }
    for pair in keys.windows(2) {
        control.check()?;
        if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
            let a = root(&mut parents, pair[0].2);
            let b = root(&mut parents, pair[1].2);
            parents[b] = a;
        }
    }
    let mut splits = BudgetVec::with_capacity(budget, lineages.len())?;
    for _ in lineages {
        splits.push(-1i64)?;
    }
    for (index, lineage) in lineages.iter().enumerate() {
        control.check()?;
        let r = root(&mut parents, index);
        if splits[r] != -1 && splits[r] != lineage.split {
            return Err(Error::contract(
                "equal-seed/common-ancestry sources have conflicting declared splits",
            ));
        }
        splits[r] = lineage.split;
    }
    Ok(())
}

pub struct Batch {
    pub features: BudgetVec<f64>,
    pub state_features: BudgetVec<f64>,
    pub episode_offsets: BudgetVec<i64>,
    pub candidate_offsets: BudgetVec<i64>,
    pub selected: BudgetVec<i64>,
    pub old_logprob: BudgetVec<f64>,
    pub old_value: BudgetVec<f64>,
    pub old_logprob_present: BudgetVec<i64>,
    pub old_value_present: BudgetVec<i64>,
    pub candidate_counts: BudgetVec<i64>,
    pub terminal_rewards: BudgetVec<f64>,
    pub winner: BudgetVec<i64>,
    pub learner_viewer: BudgetVec<i64>,
    pub split: BudgetVec<i64>,
    pub width: usize,
    pub episodes: usize,
    pub candidates: usize,
    pub metadata: BudgetVec<u8>,
    pub consumer: Lease,
    pub index_columns: Option<crate::index::Columns>,
    pub index_consumer: Option<Lease>,
}
#[derive(Clone, Copy)]
pub struct ConsumerLayout {
    pub base: usize,
    pub decision: usize,
    pub candidate: usize,
    pub episode: usize,
}
impl ConsumerLayout {
    pub fn bytes(&self, decisions: usize, candidates: usize, episodes: usize) -> Result<usize> {
        self.decision
            .checked_mul(decisions)
            .and_then(|n| {
                self.candidate
                    .checked_mul(candidates)
                    .and_then(|c| n.checked_add(c))
            })
            .and_then(|n| {
                self.episode
                    .checked_mul(episodes)
                    .and_then(|e| n.checked_add(e))
            })
            .and_then(|n| n.checked_add(self.base))
            .ok_or_else(|| Error::resource("consumer layout capacity overflow"))
    }
}
impl Stage {
    pub fn next(
        &mut self,
        budget: &Arc<Budget>,
        control: &Control,
        width: usize,
        target: usize,
        layout: ConsumerLayout,
        index_mode: bool,
        telemetry: &crate::telemetry::Telemetry,
    ) -> Result<Option<Batch>> {
        control.check()?;
        if self.cursor == self.episodes.len() {
            return Ok(None);
        }
        let start = self.cursor;
        let mut end = start;
        let mut total = 0usize;
        let mut decisions = 0usize;
        let mut candidates = 0usize;
        while end < self.episodes.len() {
            let episode = &self.episodes[end];
            if end > start && total.saturating_add(episode.export_bytes) > target {
                break;
            }
            total = total
                .checked_add(episode.export_bytes)
                .ok_or_else(|| Error::resource("batch size overflow"))?;
            decisions = decisions
                .checked_add(episode.decisions)
                .ok_or_else(|| Error::resource("batch decision count overflow"))?;
            candidates = candidates
                .checked_add(episode.candidates)
                .ok_or_else(|| Error::resource("batch candidate count overflow"))?;
            end += 1;
        }
        let count = end - start;
        let consumer = budget.reserve(layout.bytes(decisions, candidates, count)?)?;
        let index_consumer = if index_mode {
            let mut utf8 = 0usize;
            for episode in &self.episodes[start..end] {
                utf8 = utf8
                    .checked_add(
                        episode
                            .index
                            .as_ref()
                            .ok_or_else(|| Error::contract("missing index metadata"))?
                            .utf8_bytes(episode.decisions)?,
                    )
                    .ok_or_else(|| Error::resource("index consumer capacity overflow"))?;
            }
            let bytes = utf8
                .checked_mul(8)
                .and_then(|n| decisions.checked_mul(128).and_then(|d| n.checked_add(d)))
                .and_then(|n| n.checked_add(16 * 1024 * 1024))
                .ok_or_else(|| Error::resource("index consumer capacity overflow"))?;
            Some(budget.reserve(bytes)?)
        } else {
            None
        };
        let index_columns = if index_mode {
            Some(crate::index::Columns::new(
                &self.episodes[start..end],
                budget,
                control,
            )?)
        } else {
            None
        };
        let mut batch = Batch {
            features: BudgetVec::with_capacity(
                budget,
                candidates
                    .checked_mul(width)
                    .ok_or_else(|| Error::resource("candidate array size overflow"))?,
            )?,
            state_features: BudgetVec::with_capacity(
                budget,
                decisions
                    .checked_mul(width)
                    .ok_or_else(|| Error::resource("state array size overflow"))?,
            )?,
            episode_offsets: BudgetVec::with_capacity(budget, count + 1)?,
            candidate_offsets: BudgetVec::with_capacity(budget, decisions + 1)?,
            selected: BudgetVec::with_capacity(budget, decisions)?,
            old_logprob: BudgetVec::with_capacity(budget, decisions)?,
            old_value: BudgetVec::with_capacity(budget, decisions)?,
            old_logprob_present: BudgetVec::with_capacity(budget, decisions)?,
            old_value_present: BudgetVec::with_capacity(budget, decisions)?,
            candidate_counts: BudgetVec::with_capacity(budget, decisions)?,
            terminal_rewards: BudgetVec::with_capacity(budget, count * 2)?,
            winner: BudgetVec::with_capacity(budget, count)?,
            learner_viewer: BudgetVec::with_capacity(budget, count)?,
            split: BudgetVec::with_capacity(budget, count)?,
            width,
            episodes: count,
            candidates,
            metadata: BudgetVec::new(budget),
            consumer,
            index_columns,
            index_consumer,
        };
        batch.episode_offsets.push(0)?;
        batch.candidate_offsets.push(0)?;
        let mut decision_offset = 0i64;
        let mut candidate_offset = 0i64;
        let mut metadata = Vec::new();
        for episode in &self.episodes[start..end] {
            control.check()?;
            let _read_buffer = budget.reserve(32768)?;
            let mut file = BufReader::with_capacity(32768, File::open(&episode.path)?);
            let candidate_values = episode
                .candidates
                .checked_mul(width)
                .ok_or_else(|| Error::resource("candidate array size overflow"))?;
            let state_values = episode
                .decisions
                .checked_mul(width)
                .ok_or_else(|| Error::resource("state array size overflow"))?;
            read_f64(&mut file, &mut batch.features, candidate_values, control)?;
            read_f64(&mut file, &mut batch.state_features, state_values, control)?;
            // The first offset is a per-episode header; it is not exported.
            let mut first_offset = [0u8; 8];
            file.read_exact(&mut first_offset)?;
            let offset_start = batch.candidate_offsets.len();
            read_i64(
                &mut file,
                &mut batch.candidate_offsets,
                episode.decisions,
                control,
            )?;
            shift_offsets(
                &mut batch.candidate_offsets[offset_start..],
                candidate_offset,
                control,
            )?;
            read_i64(&mut file, &mut batch.selected, episode.decisions, control)?;
            read_f64(
                &mut file,
                &mut batch.old_logprob,
                episode.decisions,
                control,
            )?;
            read_f64(&mut file, &mut batch.old_value, episode.decisions, control)?;
            read_i64(
                &mut file,
                &mut batch.old_logprob_present,
                episode.decisions,
                control,
            )?;
            read_i64(
                &mut file,
                &mut batch.old_value_present,
                episode.decisions,
                control,
            )?;
            read_i64(
                &mut file,
                &mut batch.candidate_counts,
                episode.decisions,
                control,
            )?;
            telemetry.stage_read(episode.bytes);
            decision_offset = decision_offset
                .checked_add(
                    i64::try_from(episode.decisions)
                        .map_err(|_| Error::resource("decision offset overflow"))?,
                )
                .ok_or_else(|| Error::resource("decision offset overflow"))?;
            candidate_offset = candidate_offset
                .checked_add(
                    i64::try_from(episode.candidates)
                        .map_err(|_| Error::resource("candidate offset overflow"))?,
                )
                .ok_or_else(|| Error::resource("candidate offset overflow"))?;
            batch.episode_offsets.push(decision_offset)?;
            batch.terminal_rewards.extend_from_slice(&episode.rewards)?;
            batch.winner.push(episode.winner)?;
            batch.learner_viewer.push(episode.viewer)?;
            batch.split.push(episode.split)?;
            metadata.push(json::object([
                ("run_id", json::text(episode.run.as_str())),
                ("game_id", json::text(episode.game.as_str())),
            ]));
        }
        let metadata = json::object([
            ("episodes", Value::Array(metadata)),
            ("structural_valid", Value::Bool(true)),
            ("learning_valid", Value::Bool(false)),
        ]);
        batch.metadata = json::canonical(&metadata, budget, control)?;
        control.check()?;
        self.cursor = end;
        Ok(Some(batch))
    }
}
fn shift_offsets(values: &mut [i64], base: i64, control: &Control) -> Result<()> {
    for chunk in values.chunks_mut(4096) {
        control.check()?;
        for value in chunk {
            *value = value
                .checked_add(base)
                .ok_or_else(|| Error::resource("candidate offset overflow"))?;
        }
    }
    control.check()
}

#[cfg(test)]
mod bulk_stage_tests {
    use super::*;
    use crate::budget::monotonic;

    fn numeric_bytes(control: &Control) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_f64(&mut bytes, &[1., -0., 3., 4., 5., 6.], control).unwrap();
        write_f64(&mut bytes, &[7., 8., 9., 10.], control).unwrap();
        write_i64(&mut bytes, &[0, 1, 3], control).unwrap();
        write_i64(&mut bytes, &[0, 1], control).unwrap();
        write_f64(&mut bytes, &[-0.1, -0.2], control).unwrap();
        write_f64(&mut bytes, &[0.5, 0.6], control).unwrap();
        for _ in 0..2 {
            write_i64(&mut bytes, &[1, 1], control).unwrap();
        }
        write_i64(&mut bytes, &[1, 2], control).unwrap();
        assert_eq!(bytes.len(), 8 * ((3 + 2) * 2 + 7 * 2 + 1));
        bytes
    }
    fn stage(budget: &Arc<Budget>, bytes: &[u8], copies: usize) -> Stage {
        let directory = temporary().unwrap();
        let scratch = Budget::new(1024 * 1024);
        let mut episodes = BudgetVec::new(budget);
        for order in 0..copies {
            let path = directory.join(format!("episode-{order}.bin"));
            fs::write(&path, bytes).unwrap();
            episodes
                .push(Episode {
                    path,
                    run: "r".to_owned(),
                    game: format!("g{order}"),
                    viewer: 1,
                    split: 0,
                    winner: 1,
                    rewards: [1., -1.],
                    decisions: 2,
                    candidates: 3,
                    bytes: 200,
                    export_bytes: 200,
                    order,
                    index: None,
                    _scratch: scratch.reserve(bytes.len()).unwrap(),
                    _metadata: budget.reserve(256).unwrap(),
                })
                .unwrap();
        }
        Stage {
            directory,
            episodes,
            proof: Value::Null,
            cursor: 0,
            scratch,
            closed: false,
            _index_output: None,
        }
    }
    fn layout() -> ConsumerLayout {
        ConsumerLayout {
            decision: 0,
            candidate: 0,
            episode: 0,
            base: 0,
        }
    }

    #[test]
    fn stage_bulk_roundtrip_and_partial_read_keep_cursor_atomic() {
        let budget = Budget::new(2 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        let bytes = numeric_bytes(&control);
        let mut stage = stage(&budget, &bytes[..bytes.len() - 1], 1);
        let before = budget.used();
        let stats = crate::telemetry::Telemetry::default();
        assert!(matches!(
            stage.next(&budget, &control, 2, 4096, layout(), false, &stats),
            Err(Error::Io(_))
        ));
        assert_eq!(stage.cursor, 0);
        assert_eq!(budget.used(), before);
        fs::write(&stage.episodes[0].path, &bytes).unwrap();
        let batch = stage
            .next(&budget, &control, 2, 4096, layout(), false, &stats)
            .unwrap()
            .unwrap();
        assert_eq!(stage.cursor, 1);
        assert_eq!(&*batch.candidate_offsets, &[0, 1, 3]);
        assert_eq!(&*batch.episode_offsets, &[0, 2]);
        assert_eq!(&*batch.selected, &[0, 1]);
        assert_eq!(&*batch.candidate_counts, &[1, 2]);
        assert_eq!(batch.features[1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(&*batch.state_features, &[7., 8., 9., 10.]);
        drop(batch);
        stage.close().unwrap();
    }

    #[test]
    fn multiple_episode_offsets_and_overflow_preserve_cursor() {
        let budget = Budget::new(2 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        let bytes = numeric_bytes(&control);
        let stats = crate::telemetry::Telemetry::default();
        let mut stage = stage(&budget, &bytes, 2);
        let before = budget.used();
        let mut invalid = bytes.clone();
        // Six candidate and four state values precede the three offset values.
        invalid[12 * 8..13 * 8].copy_from_slice(&i64::MAX.to_le_bytes());
        fs::write(&stage.episodes[1].path, &invalid).unwrap();
        assert!(matches!(
            stage.next(&budget, &control, 2, 4096, layout(), false, &stats),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(stage.cursor, 0);
        assert_eq!(budget.used(), before);
        fs::write(&stage.episodes[1].path, &bytes).unwrap();
        let batch = stage
            .next(&budget, &control, 2, 4096, layout(), false, &stats)
            .unwrap()
            .unwrap();
        assert_eq!(&*batch.candidate_offsets, &[0, 1, 3, 4, 6]);
        assert_eq!(&*batch.episode_offsets, &[0, 2, 4]);
        assert_eq!(stage.cursor, 2);
    }

    #[test]
    fn empty_episode_keeps_only_the_original_offset_header() {
        let budget = Budget::new(2 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        let mut stage = stage(&budget, &0i64.to_le_bytes(), 1);
        stage.episodes[0].decisions = 0;
        stage.episodes[0].candidates = 0;
        stage.episodes[0].bytes = 8;
        let stats = crate::telemetry::Telemetry::default();
        let batch = stage
            .next(&budget, &control, 2, 4096, layout(), false, &stats)
            .unwrap()
            .unwrap();
        assert!(batch.features.is_empty() && batch.selected.is_empty());
        assert_eq!(&*batch.candidate_offsets, &[0]);
        assert_eq!(&*batch.episode_offsets, &[0, 0]);
    }

    #[test]
    fn offset_adjustment_checks_overflow_and_cancellation() {
        let control = Control::new(monotonic() + 30.).unwrap();
        let mut values = [0, 2, 5];
        shift_offsets(&mut values, 7, &control).unwrap();
        assert_eq!(values, [7, 9, 12]);
        assert!(matches!(
            shift_offsets(&mut [i64::MAX], 1, &control),
            Err(Error::ResourceLimit(_))
        ));
        control.cancel();
        assert_eq!(
            shift_offsets(&mut values, 1, &control),
            Err(Error::Cancelled)
        );
    }

    #[test]
    fn inventory_hash_membership_rejects_extra_missing_and_duplicate_paths() {
        let budget = Budget::new(2 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        let root = temporary().unwrap();
        let expected = root.join("e");
        fs::create_dir(&expected).unwrap();
        fs::write(expected.join("player-1.jsonl"), b"{}").unwrap();
        assert!(verify_journal_inventory(&root, &[expected.clone()], &budget, &control).is_err());
        fs::write(expected.join("player-2.jsonl"), b"{}").unwrap();
        verify_journal_inventory(&root, &[expected.clone()], &budget, &control).unwrap();
        assert!(
            verify_journal_inventory(
                &root,
                &[expected.clone(), expected.clone()],
                &budget,
                &control
            )
            .is_err()
        );
        let extra = root.join("extra");
        fs::create_dir(&extra).unwrap();
        fs::write(extra.join("player-1.jsonl"), b"{}").unwrap();
        assert!(verify_journal_inventory(&root, &[expected], &budget, &control).is_err());
        fs::remove_dir_all(root).unwrap();
        assert_eq!(budget.used(), 0);
    }
}
