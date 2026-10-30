//! A bounded resident episode window; only ordered consumption reaches spooling.
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    json::Document,
    session::Reader,
    source::Sources,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

pub struct EpisodeData {
    pub records1: JournalData,
    pub records2: JournalData,
    pub replay1: Document,
    pub replay2: Document,
}
pub enum JournalData {
    Legacy(Vec<Document>),
    Typed(crate::source::RawJournal),
}
impl JournalData {
    pub fn parse_typed(
        &self,
        control: &Control,
        telemetry: &crate::telemetry::Telemetry,
    ) -> Result<Option<crate::typed_journal::journal::Records<'_>>> {
        match self {
            Self::Legacy(_) => Ok(None),
            Self::Typed(raw) => raw.parse(control, telemetry).map(Some),
        }
    }
}
type Completed = (EpisodeData, Sources);
/// Raw owners and movable source evidence are disjoint: parsing borrows only data.
/// Consumers must drop parsed records before releasing this window.
#[cfg(test)]
pub(crate) struct EpisodeWindow<'a> {
    pub paths: &'a [PathBuf],
    pub data: BudgetVec<EpisodeData>,
    pub evidence: BudgetVec<Option<Sources>>,
}
#[cfg(test)]
pub(crate) struct ParsedEpisode<'raw> {
    pub records1: Option<crate::typed_journal::journal::Records<'raw>>,
    pub records2: Option<crate::typed_journal::journal::Records<'raw>>,
}
#[cfg(test)]
struct ParsedJournal<'raw> {
    episode: usize,
    // Original input file slot, independent of the journal header's viewer.
    file_slot: usize,
    records: Option<crate::typed_journal::journal::Records<'raw>>,
}

pub struct EpisodeReader<'a> {
    paths: &'a [PathBuf],
    position: usize,
    ready: BudgetVec<Option<Completed>>,
    ready_position: usize,
    workers: usize,
    in_flight: usize,
    budget: Arc<Budget>,
    control: Control,
    max_file: usize,
    max_record: usize,
    reader: Reader,
}
impl<'a> EpisodeReader<'a> {
    pub fn new(
        paths: &'a [PathBuf],
        workers: usize,
        in_flight: usize,
        budget: &Arc<Budget>,
        control: &Control,
        max_file: usize,
        max_record: usize,
        reader: Reader,
    ) -> Self {
        Self {
            paths,
            position: 0,
            ready: BudgetVec::new(budget),
            ready_position: 0,
            workers,
            in_flight,
            budget: budget.clone(),
            control: control.clone(),
            max_file,
            max_record,
            reader,
        }
    }
    #[cfg(test)]
    pub(crate) fn next_window(
        &mut self,
        sources: &mut Sources,
    ) -> Result<Option<EpisodeWindow<'a>>> {
        // Preserve legacy decoding, ordered merging and per-episode release.
        if self.reader == Reader::Legacy {
            let start = self.position;
            let Some((_, episode)) = self.next(sources)? else {
                return Ok(None);
            };
            let mut data = BudgetVec::with_capacity(&self.budget, 1)?;
            data.push(episode)?;
            let mut evidence = BudgetVec::with_capacity(&self.budget, 1)?;
            evidence.push(None)?;
            return Ok(Some(EpisodeWindow {
                paths: &self.paths[start..start + 1],
                data,
                evidence,
            }));
        }
        self.control.check()?;
        sources.telemetry().configure_reader(true);
        sources
            .telemetry()
            .configure_workers(self.workers, self.in_flight);
        if self.position == self.paths.len() {
            return Ok(None);
        }
        let end = self
            .position
            .saturating_add(self.in_flight)
            .min(self.paths.len());
        let paths = &self.paths[self.position..end];
        let completed = window(
            paths,
            self.workers,
            &self.budget,
            &self.control,
            self.max_file,
            self.max_record,
            sources.telemetry(),
            self.reader,
        )?;
        let mut data = BudgetVec::with_capacity(&self.budget, paths.len())?;
        let mut evidence = BudgetVec::with_capacity(&self.budget, paths.len())?;
        let (completed, _capacity) = completed.into_parts();
        for result in completed {
            let (episode, source) =
                result.ok_or_else(|| Error::contract("missing ordered native worker result"))?;
            data.push(episode)?;
            evidence.push(Some(source))?;
        }
        self.position = end;
        Ok(Some(EpisodeWindow {
            paths,
            data,
            evidence,
        }))
    }
    pub fn next(&mut self, sources: &mut Sources) -> Result<Option<(&'a Path, EpisodeData)>> {
        self.control.check()?;
        sources
            .telemetry()
            .configure_reader(self.reader == Reader::TypedBorrowedV1);
        sources
            .telemetry()
            .configure_workers(self.workers, self.in_flight);
        if self.ready_position == self.ready.len() {
            if self.position == self.paths.len() {
                return Ok(None);
            }
            let end = self
                .position
                .saturating_add(self.in_flight)
                .min(self.paths.len());
            self.ready = window(
                &self.paths[self.position..end],
                self.workers,
                &self.budget,
                &self.control,
                self.max_file,
                self.max_record,
                sources.telemetry(),
                self.reader,
            )?;
            self.ready_position = 0;
        }
        let directory = &self.paths[self.position];
        let (data, owned_sources) = self.ready[self.ready_position]
            .take()
            .ok_or_else(|| Error::contract("missing ordered native worker result"))?;
        sources.merge(owned_sources, &self.control)?;
        self.position += 1;
        self.ready_position += 1;
        Ok(Some((directory.as_path(), data)))
    }
}
fn decode(
    path: &Path,
    budget: &Arc<Budget>,
    control: &Control,
    max_file: usize,
    max_record: usize,
    telemetry: &Arc<crate::telemetry::Telemetry>,
    slot: usize,
    reader: Reader,
) -> Result<Completed> {
    control.check()?;
    let mut job = telemetry.decode_job(slot);
    let result = read_episode(
        path, budget, control, max_file, max_record, telemetry, reader,
    )?;
    job.success();
    Ok(result)
}
/// Reading is also used by the complete worker pipeline, without recreating a
/// decoder window or incrementing historical decoder-job counters.
pub(crate) fn read_episode(
    path: &Path,
    budget: &Arc<Budget>,
    control: &Control,
    max_file: usize,
    max_record: usize,
    telemetry: &Arc<crate::telemetry::Telemetry>,
    reader: Reader,
) -> Result<Completed> {
    control.check()?;
    let mut sources = Sources::with_telemetry(budget.clone(), max_file, max_record, telemetry);
    let mut journal = |name: &str| match reader {
        Reader::Legacy => sources
            .journal(&path.join(name), control)
            .map(JournalData::Legacy),
        Reader::TypedBorrowedV1 => sources
            .raw_journal(&path.join(name), control)
            .map(JournalData::Typed),
    };
    let records1 = journal("player-1.jsonl")?;
    let records2 = journal("player-2.jsonl")?;
    let replay1 = sources.json(&path.join("replay-player-1.json"), true, control)?;
    let replay2 = sources.json(&path.join("replay-player-2.json"), true, control)?;
    Ok((
        EpisodeData {
            records1,
            records2,
            replay1,
            replay2,
        },
        sources,
    ))
}
fn window(
    paths: &[PathBuf],
    workers: usize,
    budget: &Arc<Budget>,
    control: &Control,
    max_file: usize,
    max_record: usize,
    telemetry: &Arc<crate::telemetry::Telemetry>,
    reader: Reader,
) -> Result<BudgetVec<Option<Completed>>> {
    telemetry.decode_window(workers, paths.len());
    let mut ready = BudgetVec::with_capacity(budget, paths.len())?;
    for _ in paths {
        ready.push(None)?;
    }
    if workers == 1 {
        for (index, path) in paths.iter().enumerate() {
            ready[index] = Some(decode(
                path, budget, control, max_file, max_record, telemetry, 0, reader,
            )?);
        }
        return Ok(ready);
    }
    let _failure_capacity = budget.reserve(1024)?;
    let failure = Arc::new(Mutex::new(None::<Error>));
    std::thread::scope(|scope| -> Result<()> {
        let count = workers.min(paths.len());
        let mut handles = BudgetVec::with_capacity(budget, count)?;
        let mut stacks = BudgetVec::with_capacity(budget, count)?;
        for worker in 0..count {
            // Account reserved native thread stacks along with heap capacities.
            let stack = match budget.reserve(8 * 1024 * 1024) {
                Ok(lease) => lease,
                Err(error) => {
                    let mut first = failure.lock().expect("worker failure lock");
                    if first.is_none() {
                        *first = Some(error);
                    }
                    control.cancel();
                    break;
                }
            };
            let worker_budget = budget.clone();
            let worker_control = control.clone();
            let worker_failure = failure.clone();
            let worker_telemetry = telemetry.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("gf-native-{worker}"))
                .stack_size(8 * 1024 * 1024)
                .spawn_scoped(scope, move || {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> Result<BudgetVec<(usize, Completed)>> {
                            let mut results = BudgetVec::with_capacity(
                                &worker_budget,
                                paths.len().div_ceil(count),
                            )?;
                            for index in (worker..paths.len()).step_by(count) {
                                worker_control.check()?;
                                let result = decode(
                                    &paths[index],
                                    &worker_budget,
                                    &worker_control,
                                    max_file,
                                    max_record,
                                    &worker_telemetry,
                                    worker,
                                    reader,
                                )?;
                                results.push((index, result))?;
                            }
                            Ok(results)
                        },
                    ))
                    .unwrap_or_else(|_| {
                        Err(Error::contract("native preprocessing worker panicked"))
                    });
                    if let Err(error) = &outcome {
                        let mut first = worker_failure.lock().expect("worker failure lock");
                        if first.is_none() {
                            *first = Some(error.clone());
                        }
                        worker_control.cancel();
                    }
                    outcome
                });
            match spawned {
                Ok(handle) => {
                    telemetry.thread_started();
                    stacks.push(stack)?;
                    handles.push(handle)?;
                }
                Err(error) => {
                    let mut first = failure.lock().expect("worker failure lock");
                    if first.is_none() {
                        *first = Some(error.into());
                    }
                    control.cancel();
                    break;
                }
            }
        }
        while let Some(handle) = handles.pop() {
            match handle.join() {
                Ok(Ok(results)) => {
                    let (results, _capacity) = results.into_parts();
                    for (index, result) in results {
                        ready[index] = Some(result);
                    }
                }
                Ok(Err(_)) => {}
                Err(_) => {
                    let mut first = failure.lock().expect("worker failure lock");
                    if first.is_none() {
                        *first = Some(Error::contract("native preprocessing worker panicked"));
                    }
                    control.cancel();
                }
            }
        }
        if let Some(error) = failure.lock().expect("worker failure lock").take() {
            return Err(error);
        }
        Ok(())
    })?;
    Ok(ready)
}

#[cfg(test)]
fn record_parse_failure(failure: &Mutex<Option<Error>>, control: &Control, error: Error) {
    let mut first = failure.lock().expect("parse worker failure lock");
    if first.is_none() {
        *first = Some(error);
    }
    control.cancel();
}
/// Scoped workers borrow stable raw owners. No result can escape their lifetime.
/// All joins drain before any failure is returned or downstream publication starts.
#[cfg(test)]
pub(crate) fn parse_window<'raw>(
    data: &'raw [EpisodeData],
    workers: usize,
    budget: &Arc<Budget>,
    control: &Control,
    telemetry: &Arc<crate::telemetry::Telemetry>,
) -> Result<BudgetVec<Option<ParsedEpisode<'raw>>>> {
    let result = parse_window_inner(data, workers, budget, control, telemetry);
    if result.is_err() {
        control.cancel();
    }
    result
}
#[cfg(test)]
fn parse_window_inner<'raw>(
    data: &'raw [EpisodeData],
    workers: usize,
    budget: &Arc<Budget>,
    control: &Control,
    telemetry: &Arc<crate::telemetry::Telemetry>,
) -> Result<BudgetVec<Option<ParsedEpisode<'raw>>>> {
    control.check()?;
    let mut ready = BudgetVec::with_capacity(budget, data.len())?;
    for _ in data {
        ready.push(None)?;
    }
    if data.iter().all(|episode| {
        matches!(episode.records1, JournalData::Legacy(_))
            && matches!(episode.records2, JournalData::Legacy(_))
    }) {
        return Ok(ready);
    }
    let tasks = data
        .len()
        .checked_mul(2)
        .ok_or_else(|| Error::resource("native Journal parse task count overflow"))?;
    telemetry.parse_window(workers, data.len(), tasks);
    let parse = |ordinal: usize, slot: usize| -> Result<ParsedJournal<'raw>> {
        control.check()?;
        let mut job = telemetry.parse_job(slot);
        #[cfg(test)]
        telemetry.test_parse_hook(ordinal, false);
        control.check()?;
        let episode = ordinal / 2;
        let file_slot = ordinal % 2;
        let journal = if file_slot == 0 {
            &data[episode].records1
        } else {
            &data[episode].records2
        };
        let records = journal.parse_typed(control, telemetry)?;
        control.check()?;
        #[cfg(test)]
        telemetry.test_parse_hook(ordinal, true);
        job.success();
        Ok(ParsedJournal {
            episode,
            file_slot,
            records,
        })
    };
    if workers == 1 {
        for (index, output) in ready.iter_mut().enumerate() {
            let pair = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Ok(ParsedEpisode {
                    records1: parse(2 * index, 0)?.records,
                    records2: parse(2 * index + 1, 0)?.records,
                })
            }))
            .unwrap_or_else(|_| Err(Error::contract("native Journal parse worker panicked")))?;
            *output = Some(pair);
        }
        return Ok(ready);
    }
    let _failure_capacity = budget.reserve(1024)?;
    let failure = Mutex::new(None::<Error>);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| -> Result<()> {
        let count = workers.min(tasks);
        let mut handles = BudgetVec::with_capacity(budget, count)?;
        let mut stacks = BudgetVec::with_capacity(budget, count)?;
        let mut completed = BudgetVec::with_capacity(budget, count)?;
        for worker in 0..count {
            if let Err(error) = control.check() {
                record_parse_failure(&failure, control, error);
                break;
            }
            let stack = match budget.reserve(8 * 1024 * 1024) {
                Ok(stack) => stack,
                Err(error) => {
                    record_parse_failure(&failure, control, error);
                    break;
                }
            };
            let parse = &parse;
            let failure = &failure;
            let next = &next;
            let spawned = std::thread::Builder::new()
                .name(format!("gf-journal-parse-{worker}"))
                .stack_size(8 * 1024 * 1024)
                .spawn_scoped(scope, move || {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> Result<BudgetVec<ParsedJournal<'raw>>> {
                            let mut results =
                                BudgetVec::with_capacity(budget, tasks.div_ceil(count))?;
                            loop {
                                control.check()?;
                                let ordinal = next.fetch_add(1, Ordering::Relaxed);
                                if ordinal >= tasks {
                                    break;
                                }
                                results.push(parse(ordinal, worker)?)?;
                            }
                            Ok(results)
                        },
                    ))
                    .unwrap_or_else(|_| {
                        Err(Error::contract("native Journal parse worker panicked"))
                    });
                    if let Err(error) = &outcome {
                        record_parse_failure(failure, control, error.clone());
                    }
                    outcome
                });
            match spawned {
                Ok(handle) => {
                    telemetry.parse_thread_started();
                    // Capacity was admitted before spawning, so these cannot grow.
                    stacks.push(stack)?;
                    handles.push(handle)?;
                }
                Err(error) => {
                    record_parse_failure(&failure, control, error.into());
                    break;
                }
            }
        }
        while let Some(handle) = handles.pop() {
            match handle.join() {
                Ok(Ok(results)) => {
                    // Keep results worker-local until every started worker has joined.
                    completed.push(results)?;
                }
                Ok(Err(_)) => {}
                Err(_) => record_parse_failure(
                    &failure,
                    control,
                    Error::contract("native Journal parse worker panicked"),
                ),
            }
        }
        if let Some(error) = failure.lock().expect("parse worker failure lock").take() {
            return Err(error);
        }
        control.check()?;
        for output in ready.iter_mut() {
            *output = Some(ParsedEpisode {
                records1: None,
                records2: None,
            });
        }
        for results in completed.iter_mut() {
            while let Some(result) = results.pop() {
                let output = ready[result.episode]
                    .as_mut()
                    .expect("initialized ordered parse result");
                if result.file_slot == 0 {
                    output.records1 = result.records;
                } else {
                    output.records2 = result.records;
                }
            }
        }
        control.check()
    })?;
    Ok(ready)
}

#[cfg(test)]
mod worker_observation_tests {
    use super::*;
    use crate::{budget::monotonic, telemetry::Telemetry};
    use std::fs;

    struct Fixture {
        root: PathBuf,
        paths: Vec<PathBuf>,
    }
    impl Fixture {
        fn new(count: usize) -> Self {
            let root = std::env::temp_dir().join(format!(
                "gf-wall-workers-{}-{}",
                std::process::id(),
                monotonic().to_bits()
            ));
            let mut paths = Vec::new();
            for i in 0..count {
                let path = root.join(format!("e{i}"));
                fs::create_dir_all(&path).unwrap();
                for name in [
                    "player-1.jsonl",
                    "player-2.jsonl",
                    "replay-player-1.json",
                    "replay-player-2.json",
                ] {
                    fs::write(path.join(name), b"{}\n").unwrap();
                }
                paths.push(path);
            }
            Self { root, paths }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }
    fn counters(t: &Telemetry) -> serde_json::Value {
        serde_json::from_str(&t.json(0, 0, 0)).unwrap()
    }

    #[test]
    fn real_decode_overlap_is_observed_and_cancelled_jobs_are_joined() {
        for (reader_kind, workers, cancelled) in [Reader::Legacy, Reader::TypedBorrowedV1]
            .into_iter()
            .flat_map(|reader| {
                [2, 4, 8].into_iter().flat_map(move |workers| {
                    [false, true].map(move |cancelled| (reader, workers, cancelled))
                })
            })
        {
            let fixture = Fixture::new(workers);
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 30.).unwrap();
            let telemetry = Arc::new(Telemetry::default());
            let (entered, release) = telemetry.test_decode_barriers(workers + 1);
            let mut sources =
                Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
            std::thread::scope(|scope| {
                let job = scope.spawn(|| {
                    let mut reader = EpisodeReader::new(
                        &fixture.paths,
                        workers,
                        workers,
                        &budget,
                        &control,
                        1024 * 1024,
                        1024 * 1024,
                        reader_kind,
                    );
                    let mut count = 0;
                    while let Some((path, _data)) = reader.next(&mut sources)? {
                        assert_eq!(path, fixture.paths[count]);
                        count += 1;
                    }
                    Ok::<_, Error>(count)
                });
                entered.wait();
                let observed_active =
                    counters(&telemetry)["decode_parallelism"]["active_jobs"].clone();
                if cancelled {
                    control.cancel();
                }
                release.wait();
                let result = job.join().unwrap();
                assert_eq!(observed_active, workers);
                if cancelled {
                    assert!(matches!(result, Err(Error::Cancelled)));
                } else {
                    assert_eq!(result.unwrap(), workers);
                }
            });
            let metrics = counters(&telemetry);
            let parallel = &metrics["decode_parallelism"];
            assert_eq!(parallel["active_decode_peak"], workers);
            assert_eq!(parallel["active_jobs"], 0);
            assert_eq!(parallel["spawned_threads"], workers);
            for slot in 0..workers {
                assert_eq!(parallel["logical_slots"][slot]["jobs"], 1);
                assert_eq!(
                    parallel["logical_slots"][slot]["failed_jobs"],
                    if cancelled { 1 } else { 0 }
                );
            }
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn configured_eight_with_one_in_flight_observes_only_one_effective_slot() {
        let fixture = Fixture::new(2);
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        let telemetry = Arc::new(Telemetry::default());
        let mut sources =
            Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
        let mut reader = EpisodeReader::new(
            &fixture.paths,
            8,
            1,
            &budget,
            &control,
            1024 * 1024,
            1024 * 1024,
            Reader::Legacy,
        );
        while reader.next(&mut sources).unwrap().is_some() {}
        let metrics = counters(&telemetry);
        let parallel = &metrics["decode_parallelism"];
        assert_eq!(parallel["configured_workers"], 8);
        assert_eq!(parallel["configured_max_in_flight"], 1);
        assert_eq!(parallel["windows"], 2);
        assert_eq!(parallel["scheduled_episodes"], 2);
        assert_eq!(parallel["spawned_threads"], 2);
        assert_eq!(parallel["effective_slots_peak"], 1);
        assert_eq!(parallel["active_decode_peak"], 1);
        assert_eq!(parallel["logical_slots"][0]["jobs"], 2);
        drop(reader);
        drop(sources);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn typed_consumer_failure_drops_queued_raw_results_and_shared_admission() {
        let fixture = Fixture::new(4);
        fs::write(fixture.paths[0].join("player-1.jsonl"), b"[\n").unwrap();
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let telemetry = Arc::new(Telemetry::default());
        let mut sources =
            Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
        let mut reader = EpisodeReader::new(
            &fixture.paths,
            4,
            4,
            &budget,
            &control,
            1024 * 1024,
            1024 * 1024,
            Reader::TypedBorrowedV1,
        );
        let (_, data) = reader.next(&mut sources).unwrap().unwrap();
        assert!(data.records1.parse_typed(&control, &telemetry).is_err());
        assert!(budget.used() > 0); // Current and queued windows still own raw/admission.
        drop(data);
        drop(reader);
        drop(sources);
        assert_eq!(budget.used(), 0);
        let metrics = counters(&telemetry);
        assert_eq!(metrics["decode_parallelism"]["spawned_threads"], 4);
        assert_eq!(metrics["legacy_journal_files"], 0);
        assert_eq!(metrics["typed_journal_files"], 0);
    }
    #[test]
    fn both_reader_windows_refuse_shared_budget_without_retaining_raw_or_docs() {
        for reader_kind in [Reader::Legacy, Reader::TypedBorrowedV1] {
            let fixture = Fixture::new(4);
            let budget = Budget::new(1024 * 1024);
            let control = Control::new(monotonic() + 10.).unwrap();
            let mut sources = Sources::new(budget.clone(), 1024 * 1024, 1024 * 1024);
            let mut reader = EpisodeReader::new(
                &fixture.paths,
                4,
                4,
                &budget,
                &control,
                1024 * 1024,
                1024 * 1024,
                reader_kind,
            );
            assert!(matches!(
                reader.next(&mut sources),
                Err(Error::ResourceLimit(_))
            ));
            drop(reader);
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }
    fn typed_fixture(count: usize) -> Fixture {
        let fixture = Fixture::new(count);
        for (index, path) in fixture.paths.iter().enumerate() {
            for (slot, name) in ["player-1.jsonl", "player-2.jsonl"].into_iter().enumerate() {
                fs::write(
                    path.join(name),
                    // Deliberately reverse the viewers relative to file slots.
                    format!(
                        "{{\"kind\":\"episode{index}-file{slot}\",\"viewer\":{}}}\n",
                        2 - slot
                    ),
                )
                .unwrap();
            }
        }
        fixture
    }

    #[test]
    fn scoped_parse_reversed_finish_order_preserves_raw_and_input_order() {
        use std::sync::Condvar;
        for (workers, episodes) in [(2, 1), (4, 2), (8, 4), (8, 1)] {
            let tasks = 2 * episodes;
            let count = workers.min(tasks);
            let fixture = typed_fixture(episodes);
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 30.).unwrap();
            let telemetry = Arc::new(Telemetry::default());
            let mut sources =
                Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
            let mut reader = EpisodeReader::new(
                &fixture.paths,
                workers,
                workers,
                &budget,
                &control,
                1024 * 1024,
                1024 * 1024,
                Reader::TypedBorrowedV1,
            );
            let mut window = reader.next_window(&mut sources).unwrap().unwrap();
            assert_eq!(sources.entries.len(), 0); // No evidence merged before parsing succeeds.
            let all_started = Arc::new(std::sync::Barrier::new(count));
            let finish_order = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
            let order = finish_order.clone();
            telemetry.test_set_parse_hook(Arc::new(move |index, finished| {
                if !finished {
                    all_started.wait();
                }
                let (lock, wake) = &*order;
                let mut completed = lock.lock().unwrap();
                if finished {
                    completed.push(index);
                    wake.notify_all();
                } else {
                    while completed.len() != tasks - 1 - index {
                        let (state, timeout) = wake
                            .wait_timeout(completed, std::time::Duration::from_secs(5))
                            .unwrap();
                        assert!(!timeout.timed_out(), "parse order rendezvous timed out");
                        completed = state;
                    }
                }
            }));
            let mut parsed =
                parse_window(&window.data, workers, &budget, &control, &telemetry).unwrap();
            assert_eq!(
                *finish_order.0.lock().unwrap(),
                (0..tasks).rev().collect::<Vec<_>>()
            );
            for index in 0..episodes {
                let result = parsed[index].take().unwrap();
                let records1 = result.records1.unwrap();
                let records2 = result.records2.unwrap();
                assert_eq!(
                    records1.records[0].kind.as_ref(),
                    format!("episode{index}-file0")
                );
                assert_eq!(
                    records2.records[0].kind.as_ref(),
                    format!("episode{index}-file1")
                );
                assert_eq!(records1.records[0].viewer.value().unwrap().0, 2);
                assert_eq!(records2.records[0].viewer.value().unwrap().0, 1);
                // Borrowed DTOs stay valid while separately owned evidence moves.
                sources
                    .merge(window.evidence[index].take().unwrap(), &control)
                    .unwrap();
                assert!(matches!(
                    records1.records[0].kind.0,
                    std::borrow::Cow::Borrowed(_)
                ));
                assert!(matches!(
                    records2.records[0].kind.0,
                    std::borrow::Cow::Borrowed(_)
                ));
            }
            let metrics = counters(&telemetry);
            assert_eq!(
                metrics["typed_parse_parallelism"]["active_parse_peak"],
                count
            );
            assert_eq!(metrics["typed_parse_parallelism"]["active_jobs"], 0);
            assert_eq!(metrics["typed_parse_parallelism"]["spawned_threads"], count);
            assert_eq!(
                metrics["typed_parse_parallelism"]["effective_slots_peak"],
                count
            );
            assert_eq!(
                metrics["typed_parse_parallelism"]["scheduled_episodes"],
                episodes
            );
            assert_eq!(
                metrics["typed_parse_parallelism"]["scheduled_journal_tasks"],
                tasks
            );
            assert_eq!(
                metrics["typed_parse_parallelism"]["job_unit"],
                "participant-journal-file"
            );
            assert_eq!(metrics["typed_parse_parallelism"]["job_unit_version"], 2);
            assert_eq!(metrics["typed_journal_files"], tasks);
            assert_eq!(metrics["decode_parallelism"]["spawned_threads"], episodes);
            for slot in 0..count {
                assert_eq!(
                    metrics["typed_parse_parallelism"]["logical_slots"][slot]["jobs"],
                    1
                );
                assert_eq!(
                    metrics["typed_parse_parallelism"]["logical_slots"][slot]["failed_jobs"],
                    0
                );
            }
            drop(parsed);
            drop(window);
            drop(reader);
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn inline_parse_has_no_threads_and_legacy_has_no_parse_jobs() {
        for reader_kind in [Reader::Legacy, Reader::TypedBorrowedV1] {
            let fixture = typed_fixture(4);
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 30.).unwrap();
            let telemetry = Arc::new(Telemetry::default());
            let starts = Arc::new(Mutex::new(Vec::new()));
            let observed_starts = starts.clone();
            telemetry.test_set_parse_hook(Arc::new(move |ordinal, finished| {
                if !finished {
                    observed_starts.lock().unwrap().push(ordinal);
                }
            }));
            let mut sources =
                Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
            let mut reader = EpisodeReader::new(
                &fixture.paths,
                1,
                4,
                &budget,
                &control,
                1024 * 1024,
                1024 * 1024,
                reader_kind,
            );
            let mut seen = 0;
            while let Some(window) = reader.next_window(&mut sources).unwrap() {
                // The legacy release boundary is still a single episode.
                assert_eq!(
                    window.data.len(),
                    if reader_kind == Reader::Legacy { 1 } else { 4 }
                );
                let parsed = parse_window(&window.data, 1, &budget, &control, &telemetry).unwrap();
                for episode in parsed.iter() {
                    if reader_kind == Reader::Legacy {
                        assert!(episode.is_none());
                    } else {
                        assert!(episode.as_ref().unwrap().records1.is_some());
                    }
                }
                seen += window.data.len();
            }
            assert_eq!(seen, 4);
            assert_eq!(
                *starts.lock().unwrap(),
                if reader_kind == Reader::Legacy {
                    Vec::new()
                } else {
                    (0..8).collect::<Vec<_>>()
                }
            );
            let metrics = counters(&telemetry);
            assert_eq!(metrics["typed_parse_parallelism"]["spawned_threads"], 0);
            assert_eq!(metrics["typed_parse_parallelism"]["active_jobs"], 0);
            assert_eq!(
                metrics["typed_parse_parallelism"]["active_parse_peak"],
                if reader_kind == Reader::Legacy { 0 } else { 1 }
            );
            assert_eq!(
                metrics["typed_parse_parallelism"]["logical_slots"][0]["jobs"],
                if reader_kind == Reader::Legacy { 0 } else { 8 }
            );
            assert_eq!(
                metrics["typed_parse_parallelism"]["scheduled_episodes"],
                if reader_kind == Reader::Legacy { 0 } else { 4 }
            );
            assert_eq!(
                metrics["typed_parse_parallelism"]["scheduled_journal_tasks"],
                if reader_kind == Reader::Legacy { 0 } else { 8 }
            );
            drop(reader);
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn participant_dispatch_claims_again_while_a_sibling_file_is_blocked() {
        use std::sync::Condvar;
        let fixture = typed_fixture(3);
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 30.).unwrap();
        let telemetry = Arc::new(Telemetry::default());
        let mut sources =
            Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
        let mut reader = EpisodeReader::new(
            &fixture.paths,
            2,
            3,
            &budget,
            &control,
            1024 * 1024,
            1024 * 1024,
            Reader::TypedBorrowedV1,
        );
        let window = reader.next_window(&mut sources).unwrap().unwrap();
        let finish_order = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let observed = finish_order.clone();
        telemetry.test_set_parse_hook(Arc::new(move |ordinal, finished| {
            let (lock, wake) = &*observed;
            let mut completed = lock.lock().unwrap();
            if finished {
                completed.push(ordinal);
                wake.notify_all();
            } else if ordinal == 0 {
                while completed.len() < 5 {
                    let (state, timeout) = wake
                        .wait_timeout(completed, std::time::Duration::from_secs(5))
                        .unwrap();
                    assert!(!timeout.timed_out(), "participant dispatcher stalled");
                    completed = state;
                }
            }
        }));
        let parsed = parse_window(&window.data, 2, &budget, &control, &telemetry).unwrap();
        assert_eq!(*finish_order.0.lock().unwrap(), vec![1, 2, 3, 4, 5, 0]);
        for (index, episode) in parsed.iter().enumerate() {
            let episode = episode.as_ref().unwrap();
            assert_eq!(
                episode.records1.as_ref().unwrap().records[0].kind.as_ref(),
                format!("episode{index}-file0")
            );
            assert_eq!(
                episode.records2.as_ref().unwrap().records[0].kind.as_ref(),
                format!("episode{index}-file1")
            );
        }
        let metrics = counters(&telemetry);
        let parse = &metrics["typed_parse_parallelism"];
        assert_eq!(parse["scheduled_episodes"], 3);
        assert_eq!(parse["scheduled_journal_tasks"], 6);
        assert_eq!(parse["spawned_threads"], 2);
        assert_eq!(parse["active_parse_peak"], 2);
        let mut jobs: Vec<_> = parse["logical_slots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|slot| slot["jobs"].as_u64().unwrap())
            .filter(|jobs| *jobs > 0)
            .collect();
        jobs.sort_unstable();
        assert_eq!(jobs, vec![1, 5]);
        drop(parsed);
        drop(window);
        drop(reader);
        drop(sources);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn mixed_windows_keep_original_file_slots_and_legacy_no_op_results() {
        for workers in [1, 4] {
            let fixture = typed_fixture(3);
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 30.).unwrap();
            let telemetry = Arc::new(Telemetry::default());
            let mut sources =
                Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
            let mut reader = EpisodeReader::new(
                &fixture.paths,
                workers,
                3,
                &budget,
                &control,
                1024 * 1024,
                1024 * 1024,
                Reader::TypedBorrowedV1,
            );
            let mut window = reader.next_window(&mut sources).unwrap().unwrap();
            window.data[0].records1 = JournalData::Legacy(Vec::new());
            window.data[1].records2 = JournalData::Legacy(Vec::new());
            window.data[2].records1 = JournalData::Legacy(Vec::new());
            window.data[2].records2 = JournalData::Legacy(Vec::new());
            let parsed =
                parse_window(&window.data, workers, &budget, &control, &telemetry).unwrap();
            let first = parsed[0].as_ref().unwrap();
            assert!(first.records1.is_none());
            assert_eq!(
                first.records2.as_ref().unwrap().records[0].kind.as_ref(),
                "episode0-file1"
            );
            let second = parsed[1].as_ref().unwrap();
            assert_eq!(
                second.records1.as_ref().unwrap().records[0].kind.as_ref(),
                "episode1-file0"
            );
            assert!(second.records2.is_none());
            let legacy = parsed[2].as_ref().unwrap();
            assert!(legacy.records1.is_none() && legacy.records2.is_none());
            let metrics = counters(&telemetry);
            let parse = &metrics["typed_parse_parallelism"];
            assert_eq!(parse["scheduled_episodes"], 3);
            assert_eq!(parse["scheduled_journal_tasks"], 6);
            assert_eq!(
                parse["logical_slots"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|slot| slot["jobs"].as_u64().unwrap())
                    .sum::<u64>(),
                6
            );
            assert_eq!(parse["active_jobs"], 0);
            assert_eq!(metrics["typed_journal_files"], 2);
            drop(parsed);
            drop(window);
            drop(reader);
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn parse_syntax_panic_cancel_and_deadline_drain_without_source_publication() {
        for workers in [1, 4] {
            for failure in ["syntax", "panic", "cancel", "deadline", "active-deadline"] {
                let fixture = typed_fixture(4);
                if failure == "syntax" {
                    fs::write(fixture.paths[0].join("player-1.jsonl"), b"[\n").unwrap();
                }
                let budget = Budget::new(128 * 1024 * 1024);
                let control = Control::new(monotonic() + 30.).unwrap();
                let telemetry = Arc::new(Telemetry::default());
                let mut sources =
                    Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
                let mut reader = EpisodeReader::new(
                    &fixture.paths,
                    workers,
                    4,
                    &budget,
                    &control,
                    1024 * 1024,
                    1024 * 1024,
                    Reader::TypedBorrowedV1,
                );
                let window = reader.next_window(&mut sources).unwrap().unwrap();
                let parse_control = if failure == "deadline" || failure == "active-deadline" {
                    Control::new(monotonic() + if failure == "deadline" { 0.001 } else { 1.0 })
                        .unwrap()
                } else {
                    control.clone()
                };
                if failure == "deadline" {
                    std::thread::sleep(std::time::Duration::from_millis(3));
                }
                if failure == "panic" || failure == "cancel" || failure == "active-deadline" {
                    let all_started = Arc::new(std::sync::Barrier::new(workers));
                    let cancel = parse_control.clone();
                    telemetry.test_set_parse_hook(Arc::new(move |index, finished| {
                        if !finished && index < workers {
                            all_started.wait();
                            if failure == "active-deadline" {
                                std::thread::sleep(std::time::Duration::from_millis(1020));
                            }
                            if index == 0 {
                                if failure == "panic" {
                                    panic!("injected parse panic");
                                }
                                if failure == "cancel" {
                                    cancel.cancel();
                                }
                            }
                        }
                    }));
                }
                let result =
                    parse_window(&window.data, workers, &budget, &parse_control, &telemetry);
                match (failure, result) {
                    ("syntax", Err(Error::Contract(message))) => {
                        assert!(message.contains("EOF"), "{message}")
                    }
                    ("panic", Err(Error::Contract(message))) => {
                        assert!(message.contains("parse worker panicked"))
                    }
                    ("cancel", Err(Error::Cancelled)) => {}
                    ("deadline", Err(Error::Deadline)) => {}
                    ("active-deadline", Err(Error::Deadline)) => {}
                    _ => panic!("wrong parse failure for {failure}"),
                }
                assert_eq!(sources.entries.len(), 0);
                assert!(matches!(parse_control.check(), Err(Error::Cancelled)));
                let metrics = counters(&telemetry);
                assert_eq!(metrics["typed_parse_parallelism"]["active_jobs"], 0);
                if failure == "panic" || failure == "cancel" || failure == "active-deadline" {
                    assert_eq!(
                        metrics["typed_parse_parallelism"]["active_parse_peak"],
                        workers
                    );
                }
                drop(window);
                drop(reader);
                drop(sources);
                assert_eq!(budget.used(), 0);
            }
        }
    }

    #[test]
    fn parse_capacity_and_partial_stack_refusal_release_all_leases() {
        for failure in ["capacity", "first-stack", "partial-stacks"] {
            let fixture = typed_fixture(4);
            let budget = Budget::new(if failure == "partial-stacks" {
                20 * 1024 * 1024
            } else {
                4 * 1024 * 1024
            });
            let control = Control::new(monotonic() + 30.).unwrap();
            let telemetry = Arc::new(Telemetry::default());
            let mut sources =
                Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
            // Serial decode admits raw storage without using any worker stacks.
            let mut reader = EpisodeReader::new(
                &fixture.paths,
                1,
                4,
                &budget,
                &control,
                1024 * 1024,
                1024 * 1024,
                Reader::TypedBorrowedV1,
            );
            let window = reader.next_window(&mut sources).unwrap().unwrap();
            let quota = if failure == "capacity" {
                Some(budget.reserve(budget.limit() - budget.used()).unwrap())
            } else {
                None
            };
            assert!(matches!(
                parse_window(&window.data, 4, &budget, &control, &telemetry),
                Err(Error::ResourceLimit(_))
            ));
            let metrics = counters(&telemetry);
            let spawned = metrics["typed_parse_parallelism"]["spawned_threads"]
                .as_u64()
                .unwrap();
            if failure == "partial-stacks" {
                assert!(spawned > 0 && spawned < 4);
            } else {
                assert_eq!(spawned, 0);
            }
            assert_eq!(metrics["typed_parse_parallelism"]["active_jobs"], 0);
            assert_eq!(sources.entries.len(), 0);
            drop(quota);
            drop(window);
            drop(reader);
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }
}
