//! Complete-episode numeric spooling and all-or-nothing structural validation.
use crate::json::ValueExt;
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    error::{Error, Result},
    journal::string,
    json::{self, Value},
    numeric_io::{read_f64, read_i64, write_f64, write_i64},
    projection,
    session::Session,
    telemetry::CoordinatorPhase as WallPhase,
    typed_journal::view::{DecisionView, JournalView},
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
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
    _metadata: Lease,
}
/// Numeric files outlive failed jobs and discarded compact results. Only the
/// stage cleanup owner releases their reservations, after physical deletion.
struct NumericSpools(Mutex<BudgetVec<Lease>>);
impl NumericSpools {
    fn new(budget: &Arc<Budget>) -> Self {
        Self(Mutex::new(BudgetVec::new(budget)))
    }
    fn reserve(&self, scratch: &Arc<Budget>, bytes: usize) -> Result<()> {
        let mut leases = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        leases.reserve(1)?;
        let lease = scratch.reserve(bytes)?;
        leases.push(lease)
    }
    fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }
}
pub struct Stage {
    pub directory: PathBuf,
    pub episodes: BudgetVec<Episode>,
    pub proof: Value,
    pub cursor: usize,
    pub scratch: Arc<Budget>,
    numeric_spools: NumericSpools,
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
            numeric_spools: NumericSpools::new(budget),
            closed: false,
            _index_output: None,
        }
    }
    pub fn close(&mut self) -> Result<()> {
        if !self.closed {
            fs::remove_dir_all(&self.directory)?;
            self.closed = true;
            self.numeric_spools.clear();
            self.episodes.clear();
        }
        Ok(())
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
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
#[cfg(not(test))]
type LedgerFile = File;
#[cfg(test)]
struct LedgerFile {
    file: File,
    fail_after: Option<usize>,
}
#[cfg(test)]
impl From<File> for LedgerFile {
    fn from(file: File) -> Self {
        Self {
            file,
            fail_after: None,
        }
    }
}
#[cfg(test)]
impl Write for LedgerFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.fail_after == Some(0) {
            return Err(std::io::Error::other(
                "injected ledger partial write refusal",
            ));
        }
        let length = self.fail_after.unwrap_or(bytes.len()).min(bytes.len());
        let written = self.file.write(&bytes[..length])?;
        if let Some(remaining) = &mut self.fail_after {
            *remaining -= written;
        }
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

struct ProofLedger {
    path: PathBuf,
    writer: Option<BufWriter<LedgerFile>>,
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
                .open(&path)?
                .into(),
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
        // Keep the full reservation owned by the ledger before write_all can
        // bypass BufWriter, partially write, and fail while siblings still run.
        self.records.push(record)?;
        self.writer
            .as_mut()
            .ok_or(Error::Closed)?
            .write_all(encoded)?;
        self.offset += encoded.len() as u64;
        self.telemetry.stage_write(encoded.len());
        Ok(())
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
    writers: Option<[BufWriter<LedgerFile>; 2]>,
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
                    .open(&paths[0])?
                    .into(),
            ),
            BufWriter::with_capacity(
                32768,
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&paths[1])?
                    .into(),
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
    fn append_bytes(&mut self, split: usize, encoded: &[u8], scratch: &Arc<Budget>) -> Result<()> {
        self.leases.reserve(1)?;
        let lease = scratch.reserve(encoded.len())?;
        self.leases.push(lease)?;
        let writer = &mut self.writers.as_mut().ok_or(Error::Closed)?[split];
        writer.write_all(encoded)?;
        self.telemetry.stage_write(encoded.len());
        Ok(())
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

fn identity_bytes(
    split: usize,
    header: &Value,
    decision: DecisionView<'_, '_>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<BudgetVec<u8>> {
    let values = [
        json::text(if split == 0 { "train" } else { "validation" }),
        json::copy(header.require("run_id")?, control)?,
        json::copy(header.require("game_id")?, control)?,
        json::copy(header.require("viewer")?, control)?,
        decision.identity_number(control)?,
    ];
    let mut encoded = json::canonical(&json::array(values), budget, control)?;
    encoded.push(b'\n')?;
    Ok(encoded)
}

struct CompletedSource {
    path: PathBuf,
    encoded: BudgetVec<u8>,
    identity: Identity,
    lineage: Lineage,
    action_hash: String,
    _metadata: Lease,
}
struct CompletedEpisode {
    episode: Episode,
    sources: crate::source::Sources,
    proofs: BudgetVec<CompletedSource>,
    identities: BudgetVec<u8>,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobMoment {
    Start,
    Read,
    Parsed,
    BeforeSpool,
    SpoolOpened,
    BeforeFlush,
    Released,
    Handoff,
    Discarded,
}
#[cfg(test)]
type EpisodeHook<'a> = dyn Fn(usize, JobMoment) -> Result<()> + Sync + 'a;
struct EpisodeContext<'a> {
    #[cfg(test)]
    hook: Option<&'a EpisodeHook<'a>>,
    collection: &'a crate::session::Collection,
    episode_index: &'a projection::EpisodeIndex<'a>,
    digests: projection::CollectionDigests,
    settings: &'a crate::session::Settings,
    online: bool,
    operation: &'a str,
    config: &'a crate::feature::FeatureConfig,
    dataset: Option<&'a Value>,
    budget: &'a Arc<Budget>,
    scratch: &'a Arc<Budget>,
    numeric_spools: &'a NumericSpools,
    control: &'a Control,
    telemetry: &'a Arc<crate::telemetry::Telemetry>,
    directory: &'a Path,
    spool_start: usize,
}

/// Every large source owner and borrowed view lives in this call; only compact
/// facts/canonical bytes and the flushed spool descriptor leave its thread.
fn process_episode(
    context: &EpisodeContext<'_>,
    directory: &Path,
    ordinal: usize,
) -> Result<CompletedEpisode> {
    let budget = context.budget;
    let control = context.control;
    let collection = context.collection;
    let m = &collection.manifest.root;
    let job_wall = context.telemetry.episode_wall();
    job_wall.set(WallPhase::DecodeWaitMerge);
    let (data, sources) = crate::parallel::read_episode(
        directory,
        budget,
        control,
        context.settings.file,
        context.settings.record,
        context.telemetry,
        context.settings.reader,
    )?;
    #[cfg(test)]
    if let Some(hook) = context.hook {
        hook(ordinal, JobMoment::Read)?;
    }
    job_wall.set(WallPhase::TypedParseWait);
    let parsed1 = data.records1.parse_typed(control, context.telemetry)?;
    let parsed2 = data.records2.parse_typed(control, context.telemetry)?;
    #[cfg(test)]
    if let Some(hook) = context.hook {
        hook(ordinal, JobMoment::Parsed)?;
    }
    let mut proofs = BudgetVec::with_capacity(budget, 2)?;
    let mut identities = BudgetVec::new(budget);
    let completed = (|| {
        control.check()?;
        let id = directory
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or_else(|| Error::contract("invalid episode path"))?;
        let links = context.episode_index.links(id, control)?;
        let crate::parallel::EpisodeData {
            records1,
            records2,
            replay1,
            replay2,
        } = &data;
        job_wall.set(WallPhase::JournalValidation);
        let journal1 = {
            let _timer = context
                .telemetry
                .phase_timer(crate::telemetry::Phase::JournalInputValidation);
            JournalView::validate(
                records1,
                parsed1.as_ref(),
                context.settings.reader,
                budget,
                control,
            )?
        };
        let journal2 = {
            let _timer = context
                .telemetry
                .phase_timer(crate::telemetry::Phase::JournalInputValidation);
            JournalView::validate(
                records2,
                parsed2.as_ref(),
                context.settings.reader,
                budget,
                control,
            )?
        };
        let first_viewer = journal1.header().require("viewer")?.integer(1, 2)? as usize;
        let second_viewer = journal2.header().require("viewer")?.integer(1, 2)? as usize;
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
            job_wall.set(WallPhase::SourceMetadata);
            control.check()?;
            let viewer = journal.header().require("viewer")?.integer(1, 2)? as usize;
            let replay = replays[viewer - 1];
            let source = directory.join(format!("player-{}.jsonl", slot + 1));
            let metadata = projection::source_metadata(
                collection,
                &source,
                replay,
                journal,
                &links,
                &sources,
                &context.digests,
                budget,
                control,
            )?;
            job_wall.set(WallPhase::ProofLedger);
            let _proof_timer = context
                .telemetry
                .phase_timer(crate::telemetry::Phase::ProofStaging);
            let encoded = json::canonical(&metadata, budget, control)?;
            let canonical_path = string(&metadata, "path")?;
            let fragment_metadata = budget.reserve(
                canonical_path.len() + string(journal.header(), "action_config_hash")?.len(),
            )?;
            let path = PathBuf::from(canonical_path);
            job_wall.set(WallPhase::EpisodeFactsBuffers);
            let identity = Identity {
                _metadata: budget.reserve(
                    string(journal.header(), "run_id")?.len()
                        + string(journal.header(), "game_id")?.len()
                        + 2 * size_of::<String>(),
                )?,
                run: (string(journal.header(), "run_id")?).to_owned(),
                game: (string(journal.header(), "game_id")?).to_owned(),
                viewer: viewer as i128,
            };
            let lineage = metadata.require("lineage")?;
            let lineage = Lineage {
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
            };
            let current = string(journal.header(), "action_config_hash")?;
            proofs.push(CompletedSource {
                path,
                encoded,
                identity,
                lineage,
                action_hash: current.to_owned(),
                _metadata: fragment_metadata,
            })?;
        }
        job_wall.set(WallPhase::EpisodeFactsBuffers);
        let learner = links.spec.require("learner_viewer")?.integer(1, 2)? as usize;
        let journal = by_viewer[learner - 1];
        let outcomes = [
            by_viewer[0]
                .end()
                .ok_or_else(|| Error::contract("missing journal end"))?,
            by_viewer[1]
                .end()
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
        let decisions = journal.decision_count();
        let mut candidates = 0usize;
        for decision_index in 0..journal.decision_count() {
            let decision = journal.decision(decision_index)?;
            control.check()?;
            let encoded =
                identity_bytes(split as usize, journal.header(), decision, budget, control)?;
            identities.extend_from_slice(&encoded)?;
            let length = if context.online {
                decision.candidate_count()?
            } else {
                1
            };
            candidates = candidates
                .checked_add(length)
                .ok_or_else(|| Error::resource("candidate count overflow"))?;
        }
        let width = if ["prepare", "ppo"].contains(&context.operation) {
            context.config.width()
        } else {
            0
        };
        let bytes = candidates
            .checked_add(decisions)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_add(7 * decisions + 1))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| Error::resource("episode numeric size overflow"))?;
        // Transfer the reservation to Stage before opening the file. Neither
        // job failure nor result disposal can uncharge a still-live spool.
        context.numeric_spools.reserve(context.scratch, bytes)?;
        let path = context
            .directory
            .join(format!("episode-{:08}.bin", context.spool_start + ordinal));
        #[cfg(test)]
        if let Some(hook) = context.hook {
            hook(ordinal, JobMoment::BeforeSpool)?;
        }
        let _write_buffer = budget.reserve(32768)?;
        let mut file = BufWriter::with_capacity(
            32768,
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?,
        );
        #[cfg(test)]
        if let Some(hook) = context.hook {
            hook(ordinal, JobMoment::SpoolOpened)?;
        }
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
        job_wall.set(WallPhase::FeaturesSpooling);
        let mut candidate_count = 0usize;
        for decision_index in 0..journal.decision_count() {
            let decision = journal.decision(decision_index)?;
            control.check()?;
            let saved_index = decision.selected()?;
            let (saved_logprob, value) = decision.old_statistics()?;
            let old_logprob = saved_logprob.unwrap_or(0.);
            let old_value = value.unwrap_or(0.);
            if context.online
                && (saved_logprob.is_none()
                    || value.is_none()
                    || !old_logprob.is_finite()
                    || !old_value.is_finite())
            {
                return Err(Error::contract(
                    "saved PPO probability/value evidence required",
                ));
            }
            let prepared = if width > 0 {
                let _timer = context
                    .telemetry
                    .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                Some(decision.prepare(&context.config, budget, control)?)
            } else {
                None
            };
            if let Some(prepared) = &prepared {
                let _timer = context
                    .telemetry
                    .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                prepared.state(&mut row, control)?;
                state_features.extend_from_slice(&row)?;
            }
            if context.online {
                for action_index in 0..decision.candidate_count()? {
                    let action = decision.action(action_index)?;
                    {
                        let _timer = context
                            .telemetry
                            .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                        prepared
                            .as_ref()
                            .ok_or_else(|| Error::contract("PPO feature context missing"))?
                            .encode(action, &mut row, control)?;
                    }
                    let _timer = context
                        .telemetry
                        .phase_timer(crate::telemetry::Phase::ProofStaging);
                    write_f64(&mut file, &row, control)?;
                    candidate_count += 1;
                }
                selected.push(saved_index as i64)?;
            } else {
                if let Some(prepared) = &prepared {
                    let _timer = context
                        .telemetry
                        .phase_timer(crate::telemetry::Phase::FeatureEncoding);
                    prepared.encode(decision.command_action()?, &mut row, control)?;
                }
                let _timer = context
                    .telemetry
                    .phase_timer(crate::telemetry::Phase::ProofStaging);
                write_f64(&mut file, &row, control)?;
                candidate_count += 1;
                selected.push(saved_index as i64)?;
            }
            offsets.push(candidate_count as i64)?;
            logprob.push(old_logprob)?;
            values.push(old_value)?;
            logprob_present.push(saved_logprob.is_some() as i64)?;
            value_present.push(value.is_some() as i64)?;
            candidate_counts.push(decision.candidate_count()? as i64)?;
        }
        let _staging_timer = context
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
        #[cfg(test)]
        if let Some(hook) = context.hook {
            hook(ordinal, JobMoment::BeforeFlush)?;
        }
        file.flush()?;
        if file.get_ref().metadata()?.len() != bytes as u64 {
            return Err(Error::contract("numeric staging length mismatch"));
        }
        context.telemetry.stage_write(bytes);
        drop(file);
        job_wall.set(WallPhase::EpisodeFactsBuffers);
        let metadata_lease = budget.reserve(
            path.as_os_str().len()
                + string(journal.header(), "run_id")?.len()
                + string(journal.header(), "game_id")?.len()
                + 256,
        )?;
        let order = links.ordinal;
        let index = if context.operation == "index" {
            Some(crate::index::IndexInfo::new(
                collection,
                &links,
                journal.header(),
                rewards[learner - 1],
                context
                    .dataset
                    .ok_or_else(|| Error::contract("index requires a frozen dataset"))?,
                &context.digests.manifest,
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
        let episode = Episode {
            path,
            run: (string(journal.header(), "run_id")?).to_owned(),
            game: (string(journal.header(), "game_id")?).to_owned(),
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
            _metadata: metadata_lease,
        };
        Ok((episode, proofs, identities))
    })();
    job_wall.set(WallPhase::TemporaryRelease);
    drop(parsed1);
    drop(parsed2);
    drop(data);
    #[cfg(test)]
    if let Some(hook) = context.hook {
        hook(ordinal, JobMoment::Released)?;
    }
    let (episode, proofs, identities) = completed?;
    control.check()?;
    #[cfg(test)]
    if let Some(hook) = context.hook {
        hook(ordinal, JobMoment::Handoff)?;
    }
    Ok(CompletedEpisode {
        episode,
        sources,
        proofs,
        identities,
    })
}

fn record_episode_failure(
    failure: &std::sync::Mutex<Option<Error>>,
    control: &Control,
    error: Error,
) {
    let mut first = failure.lock().expect("episode failure lock");
    if first.is_none() {
        *first = Some(error);
    }
    // Save the primary cause before sibling Control checks can observe cancel.
    control.cancel();
}
fn episode_result(
    context: &EpisodeContext<'_>,
    directory: &Path,
    ordinal: usize,
    failure: &std::sync::Mutex<Option<Error>>,
) -> Result<CompletedEpisode> {
    let mut job = context.telemetry.episode_job();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(test)]
        if let Some(hook) = context.hook {
            hook(ordinal, JobMoment::Start)?;
        }
        context.control.check()?;
        process_episode(context, directory, ordinal)
    }))
    .unwrap_or_else(|_| Err(Error::contract("native complete episode worker panicked")));
    if let Err(error) = &result {
        record_episode_failure(failure, context.control, error.clone());
    } else {
        job.success();
    }
    result
}

fn discard_episode_result(
    context: &EpisodeContext<'_>,
    ordinal: usize,
    result: Result<CompletedEpisode>,
) {
    #[cfg(test)]
    let succeeded = result.is_ok();
    drop(result);
    #[cfg(test)]
    if succeeded && let Some(hook) = context.hook {
        let _ = hook(ordinal, JobMoment::Discarded);
    }
    #[cfg(not(test))]
    let _ = (context, ordinal);
}

/// Rolling stable-ordinal admission: queued, active and completed-but-uncommitted
/// jobs all occupy the same bounded ring. The coordinator never enters the pool.
fn run_collection(
    pool: Option<&rayon::ThreadPool>,
    context: &EpisodeContext<'_>,
    paths: &[PathBuf],
    wall: &crate::telemetry::ScanWall<'_>,
    mut commit: impl FnMut(CompletedEpisode) -> Result<()>,
) -> Result<()> {
    context.telemetry.episode_collection();
    let _failure_capacity = context.budget.reserve(1024)?;
    let failure = std::sync::Mutex::new(None::<Error>);
    let Some(pool) = pool else {
        for (ordinal, path) in paths.iter().enumerate() {
            context.control.check()?;
            context.telemetry.episode_admitted(1);
            wall.set(WallPhase::PoolWait);
            let completed = episode_result(context, path, ordinal, &failure)?;
            wall.set(WallPhase::OrderedCommit);
            if let Err(error) = commit(completed) {
                record_episode_failure(&failure, context.control, error.clone());
                return Err(error);
            }
            context.telemetry.episode_committed();
        }
        return Ok(());
    };
    let capacity = paths.len().min(context.settings.in_flight);
    if capacity == 0 {
        return Ok(());
    }
    let mut ready = BudgetVec::with_capacity(context.budget, capacity)?;
    for _ in 0..capacity {
        ready.push(None::<CompletedEpisode>)?;
    }
    // Transport capacity covers every admitted result, so worker sends cannot
    // block shutdown even if the coordinator fails before its next receive.
    let _transport = context.budget.reserve(
        capacity
            .checked_mul(size_of::<(usize, Result<CompletedEpisode>)>() + 128)
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| Error::resource("episode result transport capacity overflow"))?,
    )?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
    pool.in_place_scope(|scope| {
        let mut admitted = 0usize;
        let mut received = 0usize;
        let mut committed = 0usize;
        let result = (|| -> Result<()> {
            while committed < paths.len() {
                while admitted < paths.len() && admitted - committed < capacity {
                    context.control.check()?;
                    if let Some(error) = failure.lock().expect("episode failure lock").as_ref() {
                        return Err(error.clone());
                    }
                    let ordinal = admitted;
                    let sender = sender.clone();
                    let failure = &failure;
                    scope.spawn(move |_| {
                        let result = episode_result(context, &paths[ordinal], ordinal, failure);
                        // Disconnection means coordinator unwinding; no result or
                        // source owner is retained and the scope still joins us.
                        let _ = sender.send((ordinal, result));
                    });
                    admitted += 1;
                    context.telemetry.episode_admitted(admitted - committed);
                }
                wall.set(WallPhase::PoolWait);
                let (ordinal, result) = receiver
                    .recv()
                    .map_err(|_| Error::contract("episode result transport disconnected"))?;
                received += 1;
                if let Some(error) = failure.lock().expect("episode failure lock").as_ref() {
                    discard_episode_result(context, ordinal, result);
                    return Err(error.clone());
                }
                context.control.check()?;
                ready[ordinal % capacity] = Some(result?);
                context
                    .telemetry
                    .episode_reorder(ready.iter().filter(|slot| slot.is_some()).count());
                wall.set(WallPhase::OrderedCommit);
                while let Some(completed) = ready[committed % capacity].take() {
                    commit(completed)?;
                    committed += 1;
                    context.telemetry.episode_committed();
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            record_episode_failure(&failure, context.control, error);
            for ordinal in committed..admitted {
                if let Some(completed) = ready[ordinal % capacity].take() {
                    discard_episode_result(context, ordinal, Ok(completed));
                }
            }
        }
        wall.set(WallPhase::PoolWait);
        // This drain never runs Control checks: cancellation must not prevent
        // consuming all admitted senders before leaving the joining scope.
        while received < admitted {
            match receiver.recv() {
                Ok((ordinal, result)) => discard_episode_result(context, ordinal, result),
                Err(_) => break,
            }
            received += 1;
        }
        if let Some(error) = failure.lock().expect("episode failure lock").take() {
            Err(error)
        } else {
            Ok(())
        }
    })
}

pub fn scan(session: &mut Session) -> Result<Stage> {
    let wall_telemetry = session.telemetry.clone();
    let wall = wall_telemetry.scan_wall();
    session.check()?;
    if !session.authorized {
        return Err(Error::contract(
            "native session lacks schedule authorization",
        ));
    }
    wall.set(WallPhase::CollectionSetup);
    let budget = &session.budget;
    let scratch = Budget::new(session.settings.scratch);
    let mut stage = Stage {
        directory: temporary()?,
        episodes: BudgetVec::new(budget),
        proof: Value::Null,
        cursor: 0,
        scratch,
        numeric_spools: NumericSpools::new(budget),
        closed: false,
        _index_output: None,
    };
    let pool_budget = budget.clone();
    let pool_telemetry = session.telemetry.clone();
    pool_telemetry.configure_workers(session.settings.workers, session.settings.in_flight);
    pool_telemetry
        .configure_reader(session.settings.reader == crate::session::Reader::TypedBorrowedV1);
    let result = crate::episode_pool::with_pool(
        session.settings.workers,
        &pool_budget,
        &pool_telemetry,
        |pool| {
            let result = scan_into(session, &mut stage, pool, &wall);
            // build_scoped still has to terminate and join the owned threads.
            wall.set(WallPhase::PoolWait);
            result
        },
    );
    wall.set(WallPhase::Other);
    if result.is_ok() {
        wall.success();
    }
    finish_scan(stage, result, &session.control)
}
fn finish_scan(mut stage: Stage, result: Result<()>, control: &Control) -> Result<Stage> {
    match result {
        Ok(()) => Ok(stage),
        Err(primary) => {
            control.cancel();
            match stage.close() {
                Ok(()) => Err(primary),
                Err(cleanup) => Err(Error::Cleanup {
                    primary: Box::new(primary),
                    cleanup: cleanup.to_string(),
                }),
            }
        }
    }
}

fn scan_into(
    session: &mut Session,
    stage: &mut Stage,
    pool: Option<&rayon::ThreadPool>,
    wall: &crate::telemetry::ScanWall<'_>,
) -> Result<()> {
    let budget = &session.budget;
    let control = &session.control;
    let mut proof_ledger = ProofLedger::new(&stage.directory, budget, &session.telemetry)?;
    let mut source_count = 0usize;
    let mut identity_ledger = IdentityLedger::new(&stage.directory, budget, &session.telemetry)?;
    let mut identities = BudgetVec::new(budget);
    let mut lineages = BudgetVec::new(budget);
    let mut counts = [0usize; 2];
    let mut action_hash = None::<String>;
    for collection in session.collections.iter() {
        wall.set(WallPhase::CollectionSetup);
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
        let digests =
            projection::CollectionDigests::new(collection, &session.sources, budget, control)?;
        let context = EpisodeContext {
            #[cfg(test)]
            hook: None,
            collection,
            episode_index: &episode_index,
            digests,
            settings: &session.settings,
            online: session.online,
            operation: &session.operation,
            config: &session.config,
            dataset: session.dataset.as_ref().map(|d| &d.root),
            budget,
            scratch: &stage.scratch,
            numeric_spools: &stage.numeric_spools,
            control,
            telemetry: &session.telemetry,
            directory: &stage.directory,
            spool_start: stage.episodes.len(),
        };
        run_collection(pool, &context, &episodes, wall, |completed| {
            session.sources.merge(completed.sources, control)?;
            let (proofs, _capacity) = completed.proofs.into_parts();
            for proof in proofs {
                proof_ledger.append(&proof.path, &proof.encoded, &stage.scratch)?;
                source_count += 1;
                identities.push(proof.identity)?;
                lineages.push(proof.lineage)?;
                if action_hash
                    .as_ref()
                    .is_some_and(|prior| prior != &proof.action_hash)
                {
                    return Err(Error::contract(
                        "frozen sources must share identical action conditions",
                    ));
                }
                if action_hash.is_none() {
                    action_hash = Some(proof.action_hash);
                }
            }
            identity_ledger.append_bytes(
                completed.episode.split as usize,
                &completed.identities,
                &stage.scratch,
            )?;
            counts[completed.episode.split as usize] += completed.episode.decisions;
            stage.episodes.push(completed.episode)?;
            Ok(())
        })?;
        wall.set(WallPhase::CollectionProof);
        if let Some(expected) = &collection.expected_source {
            if proof_ledger.hash_range(collection_start, proof_ledger.records.len(), control)?
                != *expected
            {
                return Err(Error::contract(
                    "offline source journal/Replay proof changed since PPO update",
                ));
            }
        }
        wall.set(WallPhase::TemporaryRelease);
    }
    wall.set(WallPhase::GlobalChecks);
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
        (
            "reader_contract",
            json::text(session.settings.reader.contract()),
        ),
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
    Ok(())
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
        let numeric_spools = NumericSpools::new(budget);
        for order in 0..copies {
            let path = directory.join(format!("episode-{order}.bin"));
            numeric_spools.reserve(&scratch, bytes.len()).unwrap();
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
            numeric_spools,
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

#[cfg(test)]
mod episode_pipeline_tests {
    use super::*;
    use crate::{
        budget::monotonic,
        session::{Collection, Reader, Settings},
        source::Sources,
        telemetry::Telemetry,
    };
    use std::{sync::Mutex, thread, time::Duration};

    struct Fixture {
        directory: PathBuf,
        paths: Vec<PathBuf>,
        scratch: Arc<Budget>,
        spool_directory: Mutex<Option<PathBuf>>,
        ledger_failure: Option<&'static str>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
    impl Fixture {
        fn new(count: usize) -> Self {
            let directory = temporary().unwrap();
            let config =
                serde_json::json!({"use_action_options":false,"scorer":"public_heuristic_v1"});
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 30.).unwrap();
            let action_hash = crate::controls::digest(&config, &budget, &control).unwrap();
            let mut episodes = Vec::new();
            let mut schedule = Vec::new();
            let mut groups = Vec::new();
            let mut paths = Vec::new();
            for ordinal in 0..count {
                let id = format!("e{ordinal:03}");
                let game = format!("g{ordinal:03}");
                let path = directory.join(&id);
                fs::create_dir(&path).unwrap();
                let empty = ordinal == 0;
                for slot in 1..=2 {
                    // Deliberately swap file slots independently of viewer and learner.
                    let viewer = if ordinal % 2 == 0 { slot } else { 3 - slot };
                    let mut records = crate::typed_journal::journal::tests::fixture();
                    records[0]["episode_id"] = json::text(&id);
                    records[0]["game_id"] = json::text(&game);
                    records[0]["viewer"] = serde_json::json!(viewer);
                    records[0]["action_config_hash"] = json::text(&action_hash);
                    records[1]["input"]["observation"]["viewer_player_id"] =
                        serde_json::json!(viewer);
                    records[1]["input"]["observation"]["current_player"] =
                        serde_json::json!(viewer);
                    records[1]["input"]["observation"]["own"]["player_id"] =
                        serde_json::json!(viewer);
                    records[1]["input"]["observation"]["opponent"]["player_id"] =
                        serde_json::json!(3 - viewer);
                    records[1]["selection"]["policy_version"] = json::text("mlp-ppo-v1-p");
                    records[1]["candidates"]["config_hash"] = json::text(&action_hash);
                    records[4]["final_observation"]["viewer_player_id"] = serde_json::json!(viewer);
                    records[4]["final_observation"]["own"]["player_id"] = serde_json::json!(viewer);
                    records[4]["final_observation"]["opponent"]["player_id"] =
                        serde_json::json!(3 - viewer);
                    if empty {
                        records = vec![records[0].clone(), records[4].clone()];
                    }
                    fs::write(
                        path.join(format!("player-{slot}.jsonl")),
                        crate::typed_journal::journal::tests::wire(&records),
                    )
                    .unwrap();
                }
                for viewer in 1..=2 {
                    let decisions = if empty {
                        vec![]
                    } else {
                        vec![
                            serde_json::json!({"actor":1,"visible_action":{"type":"action","action":{"type":"pray"}}}),
                            serde_json::json!({"actor":2,"visible_action":{"type":"action","action":{"type":"pray"}}}),
                        ]
                    };
                    let replay = serde_json::json!({"game_id":game,"viewer_player_id":viewer,"observation_schema_version":"1",
                        "turns":[{"decisions":decisions}],"episode_end":{"type":"terminal","result":"draw"},
                        "diagnostic_padding":"x".repeat(64 * 1024)});
                    fs::write(
                        path.join(format!("replay-player-{viewer}.json")),
                        serde_json::to_vec(&replay).unwrap(),
                    )
                    .unwrap();
                }
                episodes.push(serde_json::json!({"episode_id":id,"game_id":game,"policies":["p","p"],"pair_id":id,
                    "seed":ordinal.to_string(),"end_kind":"natural","simulator_abort_reason":null,"outcome":"draw"}));
                schedule.push(serde_json::json!({"seed":ordinal.to_string(),"episodes":[{"episode_id":id,"pair_id":id,"policies":["p","p"],"learner_viewer":ordinal % 2 + 1}]}));
                groups.push(serde_json::json!({"episode_ids":[id]}));
                paths.push(path);
            }
            episodes.reverse(); // Spool order and online manifest ordinals are distinct.
            let report = serde_json::json!({"schema_version":1,"comparison_valid":true,"failures":{},"health_failures":{},"group_health_failures":{}});
            let report_wire = serde_json::to_vec(&report).unwrap();
            fs::write(directory.join("report.json"), &report_wire).unwrap();
            let manifest = serde_json::json!({"run_id":"r","split":"train","episodes":episodes,"schedule":schedule,"groups":groups,
                "effective_config":{"actions":config},"report_sha256":format!("{:x}",Sha256::digest(&report_wire)),
                "ppo":{"parent_run_id":"parent","batch_id":"batch","collection_generation":0}});
            fs::write(
                directory.join("manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
            Self {
                directory,
                paths,
                scratch: Budget::new(128 * 1024 * 1024),
                spool_directory: Mutex::new(None),
                ledger_failure: None,
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn run(
            &self,
            budget: &Arc<Budget>,
            control: &Control,
            telemetry: &Arc<Telemetry>,
            reader: Reader,
            workers: usize,
            in_flight: usize,
            operation: &str,
            hook: Option<&EpisodeHook<'_>>,
            fail_commit: bool,
        ) -> Result<Capture> {
            let mut sources =
                Sources::with_telemetry(budget.clone(), 4 * 1024 * 1024, 1024 * 1024, telemetry);
            let manifest = sources.json(&self.directory.join("manifest.json"), false, control)?;
            let report = sources.json(&self.directory.join("report.json"), false, control)?;
            let collection = Collection::test_new(self.directory.clone(), manifest, report, budget);
            let index = projection::EpisodeIndex::new(&collection.manifest.root, budget, control)?;
            let digests =
                projection::CollectionDigests::new(&collection, &sources, budget, control)?;
            let settings = Settings {
                memory: budget.limit(),
                file: 4 * 1024 * 1024,
                record: 1024 * 1024,
                chunk: 1024 * 1024,
                scratch: 128 * 1024 * 1024,
                workers,
                in_flight,
                reader,
            };
            let config = crate::feature::FeatureConfig {
                history: "none".to_owned(),
                max_elements: 3,
                hash_buckets: 2,
                history_window: 128,
            };
            let dataset = serde_json::json!({"content_sha256":"fixed-data-content"});
            let mut stage = Stage {
                directory: temporary()?,
                episodes: BudgetVec::new(budget),
                proof: Value::Null,
                cursor: 0,
                scratch: self.scratch.clone(),
                numeric_spools: NumericSpools::new(budget),
                closed: false,
                _index_output: None,
            };
            let stage_path = stage.directory.clone();
            *self.spool_directory.lock().unwrap() = Some(stage_path.clone());
            let wall = telemetry.scan_wall();
            telemetry.configure_workers(workers, in_flight);
            let mut capture = Capture::default();
            let result = crate::episode_pool::with_pool(workers, budget, telemetry, |pool| {
                let mut proof_ledger = ProofLedger::new(&stage.directory, budget, telemetry)?;
                let mut identity_ledger = IdentityLedger::new(&stage.directory, budget, telemetry)?;
                match self.ledger_failure {
                    Some("proof") => {
                        proof_ledger.writer.as_mut().unwrap().get_mut().fail_after = Some(4096)
                    }
                    Some("identity") => {
                        identity_ledger.writers.as_mut().unwrap()[0]
                            .get_mut()
                            .fail_after = Some(4096)
                    }
                    _ => {}
                }
                // Two collection scopes demonstrate actual owned pool reuse.
                for paths in self.paths.chunks(self.paths.len().div_ceil(2).max(1)) {
                    let context = EpisodeContext {
                        hook,
                        collection: &collection,
                        episode_index: &index,
                        digests: projection::CollectionDigests::new(
                            &collection,
                            &sources,
                            budget,
                            control,
                        )?,
                        settings: &settings,
                        online: operation == "ppo",
                        operation,
                        config: &config,
                        dataset: Some(&dataset),
                        budget,
                        scratch: &stage.scratch,
                        numeric_spools: &stage.numeric_spools,
                        control,
                        telemetry,
                        directory: &stage.directory,
                        spool_start: stage.episodes.len(),
                    };
                    run_collection(pool, &context, paths, &wall, |completed| {
                        if let Some(failure) = self.ledger_failure {
                            let mut encoded = BudgetVec::with_capacity(budget, 65536)?;
                            encoded.resize(65536, b'x')?;
                            match failure {
                                "proof" => proof_ledger.append(
                                    &stage.directory.join("source-fixture"),
                                    &encoded,
                                    &stage.scratch,
                                )?,
                                "identity" => {
                                    identity_ledger.append_bytes(0, &encoded, &stage.scratch)?
                                }
                                _ => unreachable!(),
                            }
                        }
                        if fail_commit {
                            return Err(Error::Io("injected coordinator write refusal".into()));
                        }
                        sources.merge(completed.sources, control)?;
                        for proof in completed.proofs.iter() {
                            proof_ledger.append(&proof.path, &proof.encoded, &stage.scratch)?;
                            capture.proofs.push(proof.encoded.to_vec());
                        }
                        identity_ledger.append_bytes(
                            completed.episode.split as usize,
                            &completed.identities,
                            &stage.scratch,
                        )?;
                        capture.identities.extend_from_slice(&completed.identities);
                        capture.numeric.push(fs::read(&completed.episode.path)?);
                        capture.facts.push((
                            completed.episode.order,
                            completed.episode.viewer,
                            completed.episode.decisions,
                            completed.episode.candidates,
                        ));
                        if let Some(info) = &completed.episode.index {
                            capture
                                .index_rows
                                .push(info.producer_bytes(completed.episode.decisions)?);
                        }
                        stage.episodes.push(completed.episode)?;
                        Ok(())
                    })?;
                }
                sources.verify(control)?;
                capture.source_hash =
                    proof_ledger.hash_range(0, proof_ledger.records.len(), control)?;
                proof_ledger.close()?;
                capture.identity_hash = identity_ledger.finish(control)?;
                Ok(())
            });
            drop(digests);
            let result = finish_scan(stage, result, control);
            if result.is_err() {
                assert!(
                    !stage_path.exists(),
                    "failed scans must reclaim owned spool files"
                );
            }
            let mut stage = result?;
            assert_eq!(
                capture.identity_hash,
                format!("{:x}", Sha256::digest(&capture.identities))
            );
            stage.close()?;
            assert!(!stage_path.exists());
            Ok(capture)
        }
    }
    #[derive(Default, Debug, PartialEq)]
    struct Capture {
        numeric: Vec<Vec<u8>>,
        proofs: Vec<Vec<u8>>,
        identities: Vec<u8>,
        facts: Vec<(usize, i64, usize, usize)>,
        index_rows: Vec<usize>,
        source_hash: String,
        identity_hash: String,
    }
    fn metrics(t: &Telemetry) -> Value {
        serde_json::from_str(&t.json(0, 0, 0)).unwrap()
    }

    #[test]
    fn complete_jobs_preserve_bits_proofs_order_and_pool_reuse_for_both_readers_and_all_operations()
    {
        let fixture = Fixture::new(7);
        for operation in ["prepare", "build", "index", "ppo"] {
            let mut expected = None;
            for reader in [Reader::Legacy, Reader::TypedBorrowedV1] {
                for workers in [1, 2, 4, 8] {
                    for in_flight in [1, workers, workers + 2] {
                        let budget = Budget::new(256 * 1024 * 1024);
                        let control = Control::new(monotonic() + 60.).unwrap();
                        let telemetry = Arc::new(Telemetry::default());
                        let result = fixture
                            .run(
                                &budget, &control, &telemetry, reader, workers, in_flight,
                                operation, None, false,
                            )
                            .unwrap();
                        for (ordinal, facts) in result.facts.iter().enumerate() {
                            assert_eq!(
                                *facts,
                                (
                                    6 - ordinal,
                                    (ordinal % 2 + 1) as i64,
                                    usize::from(ordinal != 0),
                                    usize::from(ordinal != 0)
                                )
                            );
                        }
                        let actual_manifest = format!(
                            "{:x}",
                            Sha256::digest(
                                fs::read(fixture.directory.join("manifest.json")).unwrap()
                            )
                        );
                        for proof in &result.proofs {
                            let proof: Value = serde_json::from_slice(proof).unwrap();
                            assert_eq!(proof["manifest_sha256"], actual_manifest);
                            assert_eq!(
                                proof["report"]["sha256"],
                                proof["report"]["expected_sha256"]
                            );
                        }
                        if let Some(expected) = &expected {
                            assert_eq!(
                                &result, expected,
                                "{reader:?}/{workers}/{in_flight}/{operation}"
                            );
                        } else {
                            expected = Some(result);
                        }
                        let m = metrics(&telemetry);
                        let p = &m["complete_episode_pipeline"];
                        assert_eq!(p["admitted"], 7);
                        assert_eq!(p["completed"], 7);
                        assert_eq!(p["committed"], 7);
                        assert_eq!(p["active_jobs"], 0);
                        assert_eq!(p["collections"], 2);
                        assert_eq!(
                            p["pool_threads_started"],
                            if workers == 1 { 0 } else { workers }
                        );
                        assert_eq!(p["pool_threads_exited"], p["pool_threads_started"]);
                        assert!(
                            p["admitted_uncommitted_peak"].as_u64().unwrap() <= in_flight as u64
                        );
                        assert_eq!(m["decode_parallelism"]["spawned_threads"], 0);
                        assert_eq!(m["typed_parse_parallelism"]["scheduled_journal_tasks"], 0);
                        assert_eq!(budget.used(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn shuffled_completions_release_large_sources_on_the_allocating_thread_before_handoff() {
        let fixture = Fixture::new(6);
        let budget = Budget::new(256 * 1024 * 1024);
        let telemetry = Arc::new(Telemetry::default());
        let control = Control::new(monotonic() + 30.).unwrap();
        let completions = Mutex::new(Vec::new());
        let hook = |ordinal, moment| {
            if moment == JobMoment::Start && ordinal == 0 {
                let started = std::time::Instant::now();
                while !completions.lock().unwrap().contains(&1) {
                    assert!(
                        started.elapsed() < Duration::from_secs(3),
                        "later job did not finish"
                    );
                    thread::yield_now();
                }
            }
            if moment == JobMoment::Handoff {
                completions.lock().unwrap().push(ordinal);
            }
            Ok(())
        };
        fixture
            .run(
                &budget,
                &control,
                &telemetry,
                Reader::TypedBorrowedV1,
                4,
                4,
                "prepare",
                Some(&hook),
                false,
            )
            .unwrap();
        assert_ne!(completions.into_inner().unwrap()[0], 0);
        assert!(
            metrics(&telemetry)["complete_episode_pipeline"]["reorder_peak"]
                .as_u64()
                .unwrap()
                > 1
        );
        assert_eq!(budget.used(), 0);
        for reader in [Reader::Legacy, Reader::TypedBorrowedV1] {
            let control = Control::new(monotonic() + 30.).unwrap();
            let telemetry = Arc::new(Telemetry::default());
            let events = Mutex::new(Vec::new());
            let coordinator = thread::current().id();
            let hook = |ordinal, moment| {
                events.lock().unwrap().push((
                    ordinal,
                    moment,
                    thread::current().id(),
                    budget.used(),
                ));
                Ok(())
            };
            fixture
                .run(
                    &budget,
                    &control,
                    &telemetry,
                    reader,
                    2,
                    1,
                    "prepare",
                    Some(&hook),
                    false,
                )
                .unwrap();
            let events = events.into_inner().unwrap();
            for ordinal in 0..3 {
                // Ordinals are collection-local and repeat in scope two.
                let group: Vec<_> = events.iter().filter(|e| e.0 == ordinal).take(8).collect();
                assert_eq!(group.len(), 8);
                let id = group[0].2;
                assert_ne!(id, coordinator);
                assert!(group.iter().all(|e| e.2 == id));
                let parsed = group.iter().find(|e| e.1 == JobMoment::Parsed).unwrap();
                let released = group.iter().find(|e| e.1 == JobMoment::Released).unwrap();
                assert!(
                    parsed.3 > released.3 + 2 * 1024 * 1024,
                    "actual source admissions must be released before handoff"
                );
                assert_eq!(group.last().unwrap().1, JobMoment::Handoff);
            }
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn worker_panic_later_contract_error_cancel_deadline_and_coordinator_failure_join_and_release()
    {
        for (reader, workers) in [Reader::Legacy, Reader::TypedBorrowedV1]
            .into_iter()
            .flat_map(|reader| [1, 4].map(|workers| (reader, workers)))
        {
            for scenario in [
                "panic",
                "invalid",
                "cancel",
                "deadline",
                "coordinator",
                "write",
                "flush",
                "quota",
            ] {
                let fixture = Fixture::new(6);
                let budget = Budget::new(256 * 1024 * 1024);
                let telemetry = Arc::new(Telemetry::default());
                let control = Control::new(monotonic() + 30.).unwrap();
                let finished = Mutex::new(Vec::new());
                let hook = |ordinal, moment| {
                    if ordinal == 1 && moment == JobMoment::Start {
                        // Let an earlier successful result exist before the later failure.
                        let started = std::time::Instant::now();
                        while !finished.lock().unwrap().contains(&0) {
                            assert!(
                                started.elapsed() < Duration::from_secs(3),
                                "earlier job did not succeed"
                            );
                            thread::yield_now();
                        }
                        match scenario {
                            "panic" => panic!("injected complete episode panic"),
                            "invalid" => {
                                fs::write(fixture.paths[1].join("player-1.jsonl"), b"[\n")?;
                            }
                            "cancel" => control.cancel(),
                            "deadline" => return control.at_deadline(monotonic() - 1.)?.check(),
                            _ => {}
                        }
                    }
                    if ordinal == 1 && moment == JobMoment::BeforeSpool && scenario == "quota" {
                        budget.reserve(budget.limit())?;
                    }
                    if ordinal == 1 && moment == JobMoment::BeforeSpool && scenario == "write" {
                        return Err(Error::Io("injected spool write refusal".into()));
                    }
                    if ordinal == 1 && moment == JobMoment::BeforeFlush && scenario == "flush" {
                        return Err(Error::Io("injected spool flush refusal".into()));
                    }
                    if moment == JobMoment::Handoff {
                        finished.lock().unwrap().push(ordinal);
                    }
                    Ok(())
                };
                let result = fixture.run(
                    &budget,
                    &control,
                    &telemetry,
                    reader,
                    workers,
                    workers,
                    "prepare",
                    Some(&hook),
                    scenario == "coordinator",
                );
                let error = result.unwrap_err();
                match scenario {
                    "panic" => assert!(
                        matches!(error,Error::Contract(message) if message.contains("worker panicked"))
                    ),
                    "invalid" => assert!(matches!(error, Error::Contract(_))),
                    "cancel" => assert_eq!(error, Error::Cancelled),
                    "deadline" => assert_eq!(error, Error::Deadline),
                    "quota" => assert!(matches!(error, Error::ResourceLimit(_))),
                    _ => assert!(matches!(error,Error::Io(message) if message.contains("refusal"))),
                }
                let m = metrics(&telemetry);
                let p = &m["complete_episode_pipeline"];
                assert_eq!(p["active_jobs"], 0);
                assert_eq!(p["admitted"], p["completed"]);
                assert_eq!(
                    p["pool_threads_started"],
                    if workers == 1 { 0 } else { workers }
                );
                assert_eq!(p["pool_threads_exited"], p["pool_threads_started"]);
                assert_eq!(budget.used(), 0, "{scenario}");
            }
        }
    }

    #[test]
    fn failed_partial_and_discarded_success_spools_stay_charged_while_sibling_is_active() {
        use std::sync::{
            Condvar,
            atomic::{AtomicBool, Ordering},
        };
        for reader in [Reader::Legacy, Reader::TypedBorrowedV1] {
            let fixture = Fixture::new(6);
            let budget = Budget::new(256 * 1024 * 1024);
            let telemetry = Arc::new(Telemetry::default());
            let control = Control::new(monotonic() + 30.).unwrap();
            let released = (Mutex::new(false), Condvar::new());
            let success = AtomicBool::new(false);
            let discarded = AtomicBool::new(false);
            let held_started = AtomicBool::new(false);
            let hook = |ordinal, moment| {
                if ordinal == 0 && moment == JobMoment::Start {
                    held_started.store(true, Ordering::Release);
                    let release = released.0.lock().unwrap();
                    let (release, timeout) = released
                        .1
                        .wait_timeout_while(release, Duration::from_secs(5), |value| !*value)
                        .unwrap();
                    assert!(
                        *release && !timeout.timed_out(),
                        "held sibling was not released"
                    );
                }
                if ordinal == 1 && moment == JobMoment::Handoff {
                    success.store(true, Ordering::Release);
                }
                if ordinal == 1 && moment == JobMoment::Discarded {
                    discarded.store(true, Ordering::Release);
                }
                if ordinal == 2 && moment == JobMoment::SpoolOpened {
                    let started = std::time::Instant::now();
                    while !success.load(Ordering::Acquire) {
                        assert!(started.elapsed() < Duration::from_secs(3));
                        thread::yield_now();
                    }
                    return Err(Error::Io("held-sibling partial spool flush refusal".into()));
                }
                Ok(())
            };
            thread::scope(|scope| {
                let handle = scope.spawn(|| {
                    fixture.run(
                        &budget,
                        &control,
                        &telemetry,
                        reader,
                        4,
                        4,
                        "prepare",
                        Some(&hook),
                        false,
                    )
                });
                let started = std::time::Instant::now();
                while !discarded.load(Ordering::Acquire) || !held_started.load(Ordering::Acquire) {
                    assert!(
                        started.elapsed() < Duration::from_secs(3),
                        "successful uncommitted result was not discarded"
                    );
                    thread::yield_now();
                }
                let state = metrics(&telemetry);
                assert_eq!(state["complete_episode_pipeline"]["completed"], 2);
                assert_eq!(state["complete_episode_pipeline"]["active_jobs"], 1);
                assert_eq!(state["complete_episode_pipeline"]["pool_threads_exited"], 0);
                let config = crate::feature::FeatureConfig {
                    history: "none".into(),
                    max_elements: 3,
                    hash_buckets: 2,
                    history_window: 128,
                };
                let spool_bytes = 8 * (2 * config.width() + 8);
                // Ordinals one and two each created one spool. Their job/result
                // owners are gone, but another worker is still held active.
                assert_eq!(fixture.scratch.used(), 2 * spool_bytes);
                let directory = fixture.spool_directory.lock().unwrap().clone().unwrap();
                assert_eq!(
                    fs::metadata(directory.join("episode-00000001.bin"))
                        .unwrap()
                        .len(),
                    spool_bytes as u64
                );
                assert_eq!(
                    fs::metadata(directory.join("episode-00000002.bin"))
                        .unwrap()
                        .len(),
                    0
                );
                assert!(
                    fixture
                        .scratch
                        .reserve(fixture.scratch.limit() - 2 * spool_bytes + 1)
                        .is_err()
                );
                *released.0.lock().unwrap() = true;
                released.1.notify_all();
                assert!(
                    matches!(handle.join().unwrap(),Err(Error::Io(message)) if message.contains("partial spool flush refusal"))
                );
            });
            assert_eq!(fixture.scratch.used(), 0);
            assert_eq!(budget.used(), 0);
            let state = metrics(&telemetry);
            assert_eq!(state["complete_episode_pipeline"]["pool_threads_exited"], 4);
            assert_eq!(state["complete_episode_pipeline"]["active_jobs"], 0);
        }
    }

    #[test]
    fn partial_proof_and_identity_writes_remain_charged_through_held_sibling_drain() {
        use std::sync::{
            Condvar,
            atomic::{AtomicBool, Ordering},
        };
        for reader in [Reader::Legacy, Reader::TypedBorrowedV1] {
            for kind in ["proof", "identity"] {
                let mut fixture = Fixture::new(6);
                fixture.ledger_failure = Some(kind);
                let budget = Budget::new(256 * 1024 * 1024);
                let telemetry = Arc::new(Telemetry::default());
                let control = Control::new(monotonic() + 30.).unwrap();
                let released = (Mutex::new(false), Condvar::new());
                let held_started = AtomicBool::new(false);
                let later_finished = AtomicBool::new(false);
                let hook = |ordinal, moment| {
                    if ordinal == 1 && moment == JobMoment::Start {
                        held_started.store(true, Ordering::Release);
                        let release = released.0.lock().unwrap();
                        let (release, timeout) = released
                            .1
                            .wait_timeout_while(release, Duration::from_secs(5), |value| !*value)
                            .unwrap();
                        assert!(*release && !timeout.timed_out());
                    }
                    if ordinal == 2 && moment == JobMoment::Handoff {
                        later_finished.store(true, Ordering::Release);
                    }
                    if ordinal == 0 && moment == JobMoment::Handoff {
                        let started = std::time::Instant::now();
                        while !later_finished.load(Ordering::Acquire) {
                            assert!(started.elapsed() < Duration::from_secs(3));
                            thread::yield_now();
                        }
                    }
                    Ok(())
                };
                thread::scope(|scope| {
                    let handle = scope.spawn(|| {
                        fixture.run(
                            &budget,
                            &control,
                            &telemetry,
                            reader,
                            4,
                            4,
                            "prepare",
                            Some(&hook),
                            false,
                        )
                    });
                    let started = std::time::Instant::now();
                    loop {
                        let state = metrics(&telemetry);
                        if held_started.load(Ordering::Acquire)
                            && control.check() == Err(Error::Cancelled)
                            && state["complete_episode_pipeline"]["completed"] == 2
                            && state["complete_episode_pipeline"]["active_jobs"] == 1
                        {
                            break;
                        }
                        assert!(
                            started.elapsed() < Duration::from_secs(3),
                            "coordinator did not enter failed-ledger drain"
                        );
                        thread::yield_now();
                    }
                    let directory = fixture.spool_directory.lock().unwrap().clone().unwrap();
                    let numeric_bytes: usize = fs::read_dir(&directory)
                        .unwrap()
                        .map(|entry| entry.unwrap().path())
                        .filter(|path| {
                            path.file_name()
                                .unwrap()
                                .to_string_lossy()
                                .starts_with("episode-")
                        })
                        .map(|path| fs::metadata(path).unwrap().len() as usize)
                        .sum();
                    let ledger = directory.join(if kind == "proof" {
                        "source-proofs.bin"
                    } else {
                        "train-identities.jsonl"
                    });
                    // 64 KiB bypasses the 32 KiB BufWriter; the real File wrote
                    // 4 KiB before failing. Its full lease must stay ledger-owned.
                    assert_eq!(fs::metadata(&ledger).unwrap().len(), 4096);
                    assert_eq!(fixture.scratch.used(), numeric_bytes + 65536);
                    assert!(
                        fixture
                            .scratch
                            .reserve(fixture.scratch.limit() - numeric_bytes - 65536 + 1)
                            .is_err()
                    );
                    assert_eq!(
                        metrics(&telemetry)["complete_episode_pipeline"]["pool_threads_exited"],
                        0
                    );
                    *released.0.lock().unwrap() = true;
                    released.1.notify_all();
                    assert!(
                        matches!(handle.join().unwrap(),Err(Error::Io(message)) if message.contains("ledger partial write refusal"))
                    );
                    assert!(!directory.exists());
                });
                assert_eq!(fixture.scratch.used(), 0);
                assert_eq!(budget.used(), 0);
                let state = metrics(&telemetry);
                assert_eq!(state["complete_episode_pipeline"]["pool_threads_exited"], 4);
                assert_eq!(state["complete_episode_pipeline"]["active_jobs"], 0);
            }
        }
    }

    #[test]
    fn refused_stage_cleanup_keeps_numeric_reservations_until_physical_deletion() {
        let budget = Budget::new(1024 * 1024);
        let mut stage = Stage {
            directory: temporary().unwrap(),
            episodes: BudgetVec::new(&budget),
            proof: Value::Null,
            cursor: 0,
            scratch: Budget::new(1024),
            numeric_spools: NumericSpools::new(&budget),
            closed: false,
            _index_output: None,
        };
        stage.numeric_spools.reserve(&stage.scratch, 64).unwrap();
        fs::write(stage.directory.join("partial.bin"), [0u8; 64]).unwrap();
        let backup = stage.directory.with_extension("saved");
        fs::rename(&stage.directory, &backup).unwrap();
        fs::write(&stage.directory, b"refuse directory cleanup").unwrap();
        assert!(stage.close().is_err());
        assert!(!stage.closed);
        assert_eq!(stage.scratch.used(), 64);
        assert!(backup.join("partial.bin").exists());
        fs::remove_file(&stage.directory).unwrap();
        fs::rename(&backup, &stage.directory).unwrap();
        stage.close().unwrap();
        assert!(!stage.directory.exists());
        assert_eq!(stage.scratch.used(), 0);
        drop(stage);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn pool_stack_quota_and_explicit_cleanup_failure_preserve_primary_and_release() {
        let budget = Budget::new(17 * 1024 * 1024);
        let telemetry = Arc::new(Telemetry::default());
        assert!(crate::episode_pool::with_pool(4, &budget, &telemetry, |_| Ok(())).is_err());
        assert_eq!(budget.used(), 0);
        assert_eq!(
            metrics(&telemetry)["complete_episode_pipeline"]["pool_threads_started"],
            0
        );
        let control = Control::new(monotonic() + 30.).unwrap();
        let stage = Stage::test_failed_close(&budget);
        let primary = Error::contract("original source failure");
        assert!(
            matches!(finish_scan(stage,Err(primary.clone()),&control),Err(Error::Cleanup { primary: error, cleanup }) if *error==primary && !cleanup.is_empty())
        );
        assert_eq!(budget.used(), 0);
    }
}
