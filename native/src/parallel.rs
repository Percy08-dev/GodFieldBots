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
    sync::{Arc, Mutex},
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
    job.success();
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
}
