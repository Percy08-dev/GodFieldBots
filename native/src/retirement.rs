//! Prestarted, budgeted retirement of deeply recursive owned native state.
//!
//! Admission and thread creation happen before the core is constructed. Handoff
//! uses a preallocated single slot, so no failed send can return a core to the
//! caller. Only the retirement thread takes or destroys the transferred value.
use crate::budget::{Budget, Lease};
use crate::error::{Error, Result};
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

pub const STACK_BYTES: usize = 8 * 1024 * 1024;
const METADATA_BYTES: usize = 64 * 1024;

struct Slot<T> {
    ready: bool,
    handed: bool,
    core: Option<T>,
}
struct Shared<T> {
    slot: Mutex<Slot<T>>,
    changed: Condvar,
}
impl<T> Shared<T> {
    fn lock(&self) -> MutexGuard<'_, Slot<T>> {
        self.slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
    fn wait<'a>(&self, guard: MutexGuard<'a, Slot<T>>) -> MutexGuard<'a, Slot<T>> {
        self.changed
            .wait(guard)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A cleanup worker whose lifetime and stack are charged before any core exists.
/// The owner must transfer its core with `retire`; dropping an unused retirement
/// (including partial construction failure) performs an empty handoff and joins.
pub struct Retirement<T: Send + 'static> {
    shared: Arc<Shared<T>>,
    worker: Option<JoinHandle<Result<()>>>,
    _reservation: Lease,
}
impl<T: Send + 'static> Retirement<T> {
    pub fn start(budget: &Arc<Budget>, cleanup: fn(T) -> Result<()>) -> Result<Self> {
        let bytes = STACK_BYTES
            .checked_add(METADATA_BYTES)
            .and_then(|n| n.checked_add(size_of::<Shared<T>>()))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or_else(|| Error::resource("retirement capacity overflow"))?;
        let reservation = budget.reserve(bytes)?;
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                ready: false,
                handed: false,
                core: None,
            }),
            changed: Condvar::new(),
        });
        let work = Arc::clone(&shared);
        // The spawn closure contains no core, including on an OS spawn failure.
        let worker = thread::Builder::new()
            .name("gf-native-retirement".into())
            .stack_size(STACK_BYTES)
            .spawn(move || run(work, cleanup))
            .map_err(|error| {
                Error::resource(format!("retirement thread creation failed: {error}"))
            })?;
        // Install the empty-handoff Drop guard before the readiness wait, so
        // partial start unwinding cannot detach a worker waiting for a value.
        let retirement = Self {
            shared,
            worker: Some(worker),
            _reservation: reservation,
        };
        {
            let mut slot = retirement.shared.lock();
            while !slot.ready {
                slot = retirement.shared.wait(slot);
            }
        }
        Ok(retirement)
    }

    fn handoff(&self, core: Option<T>) {
        let mut slot = self.shared.lock();
        // A consuming public API and private handoff ensure this slot has never
        // held a core. Poisoning is recovered without rejecting the transfer.
        slot.core = core;
        slot.handed = true;
        self.shared.changed.notify_all();
    }

    /// Transfer even when cleanup will fail, then join before returning evidence.
    /// The result contains only the fixed error type, never the retired payload.
    pub fn retire(mut self, core: Option<T>) -> Result<()> {
        self.handoff(core);
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        match worker.join() {
            Ok(result) => result,
            // All code that can own T, including panic-payload destruction, is
            // caught on the worker. This can only describe an internal failure.
            Err(panic) => {
                drop(panic);
                Err(Error::contract("retirement thread failed outside cleanup"))
            }
        }
    }
}
impl<T: Send + 'static> Drop for Retirement<T> {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.handoff(None);
            let _ = self.join();
        }
    }
}

fn discard_panic(mut panic: Box<dyn Any + Send>) {
    // A panic payload can itself own the core (panic_any(core)), or panic while
    // being destroyed. Never return such a payload through JoinHandle::join.
    loop {
        match catch_unwind(AssertUnwindSafe(|| drop(panic))) {
            Ok(()) => return,
            Err(next) => panic = next,
        }
    }
}
fn run<T: Send + 'static>(shared: Arc<Shared<T>>, cleanup: fn(T) -> Result<()>) -> Result<()> {
    // Poison recovery is part of the transfer protocol. Even an unexpected
    // pre-handoff synchronization panic must not let the worker leave while a
    // future core could still be placed in its slot.
    let core = loop {
        match catch_unwind(AssertUnwindSafe(|| {
            let mut slot = shared.lock();
            slot.ready = true;
            shared.changed.notify_all();
            while !slot.handed {
                slot = shared.wait(slot);
            }
            slot.core.take()
        })) {
            Ok(core) => break core,
            Err(panic) => discard_panic(panic),
        }
    };
    match catch_unwind(AssertUnwindSafe(|| match core {
        Some(core) => cleanup(core),
        None => Ok(()),
    })) {
        Ok(result) => result,
        Err(panic) => {
            discard_panic(panic);
            Err(Error::contract("retirement cleanup panicked"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn poisoned_transfer_mutex_cannot_reject_the_core() {
        let budget = Budget::new(STACK_BYTES + 1024 * 1024);
        let retirement = Retirement::start(&budget, |value: Box<u64>| {
            assert_eq!(*value, 17);
            Ok(())
        })
        .unwrap();
        let shared = Arc::clone(&retirement.shared);
        let poisoner = thread::spawn(move || {
            let _guard = shared.slot.lock().unwrap();
            panic!("poison only, before handoff");
        });
        assert!(poisoner.join().is_err());
        retirement.retire(Some(Box::new(17))).unwrap();
        assert_eq!(budget.used(), 0);
    }
}
