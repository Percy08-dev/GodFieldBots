//! One explicitly owned, actually joined pool for a complete scan.
use crate::{
    budget::{Budget, BudgetVec},
    error::{Error, Result},
    telemetry::Telemetry,
};
use std::sync::Arc;

pub(crate) const STACK_BYTES: usize = 8 * 1024 * 1024;

pub(crate) fn with_pool<T>(
    workers: usize,
    budget: &Arc<Budget>,
    telemetry: &Arc<Telemetry>,
    run: impl FnOnce(Option<&rayon::ThreadPool>) -> Result<T>,
) -> Result<T> {
    with_pool_stack(workers, budget, telemetry, STACK_BYTES, run)
}

// The stack-size seam permits deterministic OS spawn refusal in tests. The
// production entry point always supplies the reserved eight MiB per worker.
fn with_pool_stack<T>(
    workers: usize,
    budget: &Arc<Budget>,
    telemetry: &Arc<Telemetry>,
    stack_bytes: usize,
    run: impl FnOnce(Option<&rayon::ThreadPool>) -> Result<T>,
) -> Result<T> {
    if workers == 1 {
        return run(None);
    }
    // Reserve every configured stack before starting any thread. The leases
    // outlive build_scoped, whose std::thread::scope joins even partial builds.
    let mut stacks = BudgetVec::with_capacity(budget, workers)?;
    for _ in 0..workers {
        stacks.push(budget.reserve(STACK_BYTES)?)?;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .stack_size(stack_bytes)
        .thread_name(|index| format!("gf-episode-{index}"))
        .build_scoped(
            |thread| {
                telemetry.episode_thread_started();
                struct Exit<'a>(&'a Telemetry);
                impl Drop for Exit<'_> {
                    fn drop(&mut self) {
                        self.0.episode_thread_exited();
                    }
                }
                let _exit = Exit(telemetry);
                thread.run();
            },
            |pool| run(Some(pool)),
        )
        .map_err(|error| {
            Error::resource(format!("native episode pool construction failed: {error}"))
        })?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_os_spawn_refusal_releases_all_reserved_stacks_without_running_scan() {
        let budget = Budget::new(128 * 1024 * 1024);
        let telemetry = Arc::new(Telemetry::default());
        let result = with_pool_stack(4, &budget, &telemetry, usize::MAX, |_| -> Result<()> {
            panic!("scan must not run after pool spawn refusal")
        });
        assert!(
            matches!(result, Err(Error::ResourceLimit(message)) if message.contains("pool construction failed"))
        );
        assert_eq!(budget.used(), 0);
        let metrics: serde_json::Value = serde_json::from_str(&telemetry.json(0, 0, 0)).unwrap();
        assert_eq!(
            metrics["complete_episode_pipeline"]["pool_threads_started"],
            metrics["complete_episode_pipeline"]["pool_threads_exited"]
        );
    }
}
