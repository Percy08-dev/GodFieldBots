//! Complete request binding and opaque participant sampling custody.
use super::{ensure, error};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Condvar, Mutex};

// Validate the actual Duration and host monotonic-clock representation, not a
// policy time cap. Every wait uses its original absolute deadline.
pub(crate) fn checked_duration(seconds: f64) -> Result<std::time::Duration> {
    let duration = std::time::Duration::try_from_secs_f64(seconds)
        .map_err(|_| error("unsupported inference timer duration"))?;
    ensure(
        std::time::Instant::now().checked_add(duration).is_some(),
        "unsupported inference timer clock deadline",
    )?;
    Ok(duration)
}
pub(crate) fn wait_duration(deadline: f64) -> Result<std::time::Duration> {
    ensure(
        deadline.is_finite(),
        "finite inference timer deadline required",
    )?;
    checked_duration((deadline - crate::budget::monotonic()).max(0.))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub run: String,
    pub trial: String,
    pub game: String,
    pub player: u8,
    pub epoch: u64,
    pub request: u64,
    pub revision: u64,
    pub checkpoint: String,
    pub deadline: f64,
}
impl Identity {
    pub fn validate(&self) -> Result<()> {
        ensure(
            !self.run.is_empty()
                && !self.trial.is_empty()
                && !self.game.is_empty()
                && (1..=2).contains(&self.player)
                && self.deadline.is_finite()
                && self.deadline > 0.
                && self.checkpoint.len() == 64
                && self
                    .checkpoint
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid inference identity",
        )?;
        wait_duration(self.deadline)?;
        Ok(())
    }
    pub fn participant(&self) -> (String, String, String, u8) {
        (
            self.run.clone(),
            self.trial.clone(),
            self.game.clone(),
            self.player,
        )
    }
    pub fn game_key(&self) -> (String, String, String) {
        (self.run.clone(), self.trial.clone(), self.game.clone())
    }
}
#[derive(Clone, Debug)]
pub struct Answer {
    pub identity: Identity,
    pub logs: Vec<f32>,
    pub value: f32,
    pub sample: usize,
}
pub struct Response {
    pub identity: Identity,
    pub logs: Vec<f32>,
    pub value: f32,
    pub sample: usize,
    pub next_state: Vec<u8>,
    pub runtime: String,
}
#[derive(Default)]
pub struct Promise {
    state: Mutex<Option<Result<Answer>>>,
    wake: Condvar,
    async_wake: tokio::sync::Notify,
}
impl Promise {
    pub fn complete(&self, answer: Result<Answer>) {
        let mut state = self.state.lock().unwrap();
        if state.is_none() {
            *state = Some(answer);
            self.wake.notify_all();
            self.async_wake.notify_waiters();
        }
    }
    pub fn wait(&self, deadline: f64) -> Result<Answer> {
        wait_duration(deadline)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| error("inference result lock poisoned"))?;
        loop {
            if let Some(answer) = state.as_ref() {
                return answer.clone();
            }
            let left = deadline - crate::budget::monotonic();
            if left <= 0. {
                return Err(crate::error::Error::Deadline);
            }
            state = self
                .wake
                .wait_timeout(state, wait_duration(deadline)?)
                .unwrap()
                .0;
        }
    }
    /// Tokio game actors await the same promise without blocking an executor worker.
    pub async fn wait_async(&self, deadline: f64) -> Result<Answer> {
        wait_duration(deadline)?;
        loop {
            let notified = self.async_wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(answer) = self.state.lock().unwrap().as_ref() {
                return answer.clone();
            }
            let left = deadline - crate::budget::monotonic();
            if left <= 0. {
                return Err(crate::error::Error::Deadline);
            }
            // Tokio may wake at its clock tick boundary; recheck the original
            // absolute deadline rather than renewing it or accepting early expiry.
            let _ = tokio::time::timeout(wait_duration(deadline)?, notified).await;
        }
    }
}
pub type RequestHandle = Arc<Promise>;
pub struct Participant {
    pub epoch: u64,
    pub last_request: Option<u64>,
    pub seed: u64,
    pub state: Option<Vec<u8>>,
    pub retired: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test(flavor = "current_thread")]
    async fn fix1_unsupported_result_timers_never_poison_own_promise_or_other_waiter() {
        for deadline in [f64::MAX, 1e19] {
            let promise = Arc::new(Promise::default());
            assert!(promise.wait(deadline).is_err());
            assert!(promise.wait_async(deadline).await.is_err());
            let own = promise.clone();
            let waiter =
                tokio::spawn(async move { own.wait_async(crate::budget::monotonic() + 1.).await });
            tokio::task::yield_now().await;
            promise.complete(Err(crate::error::Error::Cancelled));
            assert_eq!(
                waiter.await.unwrap().unwrap_err(),
                crate::error::Error::Cancelled
            );
            assert_eq!(
                promise.wait(crate::budget::monotonic() + 1.).unwrap_err(),
                crate::error::Error::Cancelled
            );
        }
        // This exercises clock overflow separately from float-to-Duration overflow.
        assert!(std::time::Duration::try_from_secs_f64(1e19).is_ok());
        assert!(checked_duration(1e19).is_err());
    }
}
