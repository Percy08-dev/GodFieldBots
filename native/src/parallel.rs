//! A bounded resident episode window; only ordered consumption reaches spooling.
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    json::Document,
    source::Sources,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub struct EpisodeData {
    pub records1: Vec<Document>,
    pub records2: Vec<Document>,
    pub replay1: Document,
    pub replay2: Document,
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
        }
    }
    pub fn next(&mut self, sources: &mut Sources) -> Result<Option<(&'a Path, EpisodeData)>> {
        self.control.check()?;
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
) -> Result<Completed> {
    control.check()?;
    let mut sources = Sources::with_telemetry(budget.clone(), max_file, max_record, telemetry);
    let records1 = sources.journal(&path.join("player-1.jsonl"), control)?;
    let records2 = sources.journal(&path.join("player-2.jsonl"), control)?;
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
) -> Result<BudgetVec<Option<Completed>>> {
    let mut ready = BudgetVec::with_capacity(budget, paths.len())?;
    for _ in paths {
        ready.push(None)?;
    }
    if workers == 1 {
        for (index, path) in paths.iter().enumerate() {
            ready[index] = Some(decode(
                path, budget, control, max_file, max_record, telemetry,
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
