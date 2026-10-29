use gf_native::budget::Budget;
use gf_native::error::{Error, Result};
use gf_native::retirement::{Retirement, STACK_BYTES};
use std::hint::black_box;
use std::panic::panic_any;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread::{self, ThreadId};

struct Trace {
    caller: ThreadId,
    drops: AtomicUsize,
    cleanup_calls: AtomicUsize,
    dropped_on_caller: AtomicBool,
    first_drop: Mutex<Option<ThreadId>>,
}
impl Trace {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            caller: thread::current().id(),
            drops: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
            dropped_on_caller: AtomicBool::new(false),
            first_drop: Mutex::new(None),
        })
    }
}
struct Node {
    next: Option<Box<Node>>,
    trace: Arc<Trace>,
}
impl Drop for Node {
    fn drop(&mut self) {
        // Keep a bounded but substantial recursive frame, including in release
        // builds. This makes the 1/2-MiB caller stacks materially different from
        // the admitted 8-MiB retirement stack without constructing source data.
        let frame = [0u8; 256];
        black_box(&frame);
        let here = thread::current().id();
        if self.trace.drops.fetch_add(1, Ordering::SeqCst) == 0 {
            *self.trace.first_drop.lock().unwrap() = Some(here);
        }
        if here == self.trace.caller {
            self.trace.dropped_on_caller.store(true, Ordering::SeqCst);
        }
        drop(self.next.take());
        black_box(frame);
    }
}
struct Deep {
    root: Option<Box<Node>>,
    trace: Arc<Trace>,
}
impl Deep {
    fn new(depth: usize, trace: &Arc<Trace>) -> Self {
        let mut root = None;
        for _ in 0..depth {
            root = Some(Box::new(Node {
                next: root,
                trace: Arc::clone(trace),
            }));
        }
        Self {
            root,
            trace: Arc::clone(trace),
        }
    }
}
fn clean(core: Deep) -> Result<()> {
    core.trace.cleanup_calls.fetch_add(1, Ordering::SeqCst);
    black_box(core.root.as_ref());
    drop(core);
    Ok(())
}
fn failed(core: Deep) -> Result<()> {
    clean(core)?;
    Err(Error::contract("controlled cleanup failure"))
}
fn panicking(core: Deep) -> Result<()> {
    panic_any(core)
}

#[test]
fn recursive_core_drop_uses_retirement_stack_from_small_callers() {
    for stack in [1024 * 1024, 2 * 1024 * 1024] {
        thread::Builder::new()
            .stack_size(stack)
            .spawn(|| {
                let budget = Budget::new(STACK_BYTES + 1024 * 1024);
                let trace = Trace::new();
                let retirement = Retirement::start(&budget, clean).unwrap();
                assert!(budget.used() >= STACK_BYTES);
                retirement.retire(Some(Deep::new(6000, &trace))).unwrap();
                assert_eq!(trace.drops.load(Ordering::SeqCst), 6000);
                assert_eq!(trace.cleanup_calls.load(Ordering::SeqCst), 1);
                assert!(!trace.dropped_on_caller.load(Ordering::SeqCst));
                assert_ne!(*trace.first_drop.lock().unwrap(), Some(trace.caller));
                assert_eq!(budget.used(), 0);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

#[test]
fn cleanup_error_and_core_as_panic_payload_are_retired_on_worker() {
    for cleanup in [failed as fn(Deep) -> Result<()>, panicking] {
        let budget = Budget::new(STACK_BYTES + 1024 * 1024);
        let trace = Trace::new();
        let retirement = Retirement::start(&budget, cleanup).unwrap();
        let error = retirement
            .retire(Some(Deep::new(2000, &trace)))
            .unwrap_err();
        assert!(matches!(error, Error::Contract(_)));
        assert_eq!(trace.drops.load(Ordering::SeqCst), 2000);
        assert!(!trace.dropped_on_caller.load(Ordering::SeqCst));
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn unused_worker_constructor_failure_and_empty_retirement_join_and_release() {
    let budget = Budget::new(STACK_BYTES + 1024 * 1024);
    fn construct(budget: &Arc<Budget>) -> Result<()> {
        let _retirement = Retirement::<Deep>::start(budget, clean)?;
        Err(Error::contract("controlled constructor failure"))
    }
    assert!(construct(&budget).is_err());
    assert_eq!(budget.used(), 0);
    Retirement::<Deep>::start(&budget, clean)
        .unwrap()
        .retire(None)
        .unwrap();
    assert_eq!(budget.used(), 0);
    drop(Retirement::<Deep>::start(&budget, clean).unwrap());
    assert_eq!(budget.used(), 0);
}

#[test]
fn admission_fails_before_thread_or_core_and_outer_repeated_close_is_idempotent() {
    let small = Budget::new(STACK_BYTES - 1);
    assert!(matches!(
        Retirement::<Deep>::start(&small, clean),
        Err(Error::ResourceLimit(_))
    ));
    assert_eq!(small.used(), 0);
    assert_eq!(small.peak(), 0);
    let budget = Budget::new(STACK_BYTES + 1024 * 1024);
    let trace = Trace::new();
    let mut retirement = Some(Retirement::start(&budget, clean).unwrap());
    let mut core = Some(Deep::new(100, &trace));
    for _ in 0..2 {
        if let Some(worker) = retirement.take() {
            worker.retire(core.take()).unwrap();
        }
    }
    assert!(core.is_none());
    assert_eq!(trace.drops.load(Ordering::SeqCst), 100);
    assert_eq!(trace.cleanup_calls.load(Ordering::SeqCst), 1);
    assert_eq!(budget.used(), 0);
}
