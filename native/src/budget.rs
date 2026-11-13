//! Invocation accounting. Reservations precede allocation, including growth peaks.
use crate::error::{Error, Result};
use std::ops::{Deref, DerefMut};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
pub struct Budget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
    reallocations: AtomicUsize,
    reallocated_bytes: AtomicUsize,
    extend_copy_bytes: AtomicUsize,
    json_estimated_used: AtomicUsize,
    json_estimated_peak: AtomicUsize,
    numeric: Mutex<NumericBook>,
}
#[derive(Debug, Default)]
struct NumericBook {
    next: u64,
    active: Option<NumericPhase>,
    failed: bool,
}
#[derive(Debug)]
struct NumericPhase {
    generation: u64,
    limit: usize,
    charged: usize,
}
#[derive(Debug)]
pub struct NumericGuard {
    budget: Arc<Budget>,
    generation: u64,
}
impl Drop for NumericGuard {
    fn drop(&mut self) {
        if let Ok(mut book) = self.budget.numeric.lock() {
            if book
                .active
                .as_ref()
                .is_some_and(|phase| phase.generation == self.generation)
            {
                book.active = None;
            }
        }
    }
}
impl Budget {
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            reallocations: AtomicUsize::new(0),
            reallocated_bytes: AtomicUsize::new(0),
            extend_copy_bytes: AtomicUsize::new(0),
            json_estimated_used: AtomicUsize::new(0),
            json_estimated_peak: AtomicUsize::new(0),
            numeric: Mutex::new(NumericBook::default()),
        })
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
    pub fn limit(&self) -> usize {
        self.limit
    }
    pub fn json_estimated_peak(&self) -> usize {
        self.json_estimated_peak.load(Ordering::Acquire)
    }
    pub fn reserve_json(self: &Arc<Self>, bytes: usize) -> Result<JsonAdmission> {
        let lease = self.reserve(bytes)?;
        let before = self.json_estimated_used.fetch_add(bytes, Ordering::AcqRel);
        self.json_estimated_peak
            .fetch_max(before + bytes, Ordering::AcqRel);
        Ok(JsonAdmission {
            _lease: lease,
            budget: self.clone(),
            bytes,
        })
    }
    pub fn copy_counters(&self) -> (usize, usize, usize) {
        (
            self.reallocations.load(Ordering::Relaxed),
            self.reallocated_bytes.load(Ordering::Relaxed),
            self.extend_copy_bytes.load(Ordering::Relaxed),
        )
    }
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Lease> {
        let mut book = self.numeric.lock().map_err(|_| Error::Closed)?;
        if book.failed {
            return Err(Error::contract(
                "native numeric reservation accounting uncertain",
            ));
        }
        let charge = if let Some(phase) = book.active.as_ref() {
            let next = phase
                .charged
                .checked_add(bytes)
                .filter(|sum| *sum <= phase.limit)
                .ok_or_else(|| {
                    Error::resource("HPO incremental numeric budget exhausted before allocation")
                })?;
            Some((phase.generation, next))
        } else {
            None
        };
        let old = self
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|sum| *sum <= self.limit)
            })
            .map_err(|_| Error::resource("native managed memory budget exhausted"))?;
        self.peak.fetch_max(old + bytes, Ordering::AcqRel);
        if let Some((_, next)) = charge {
            book.active.as_mut().ok_or(Error::Closed)?.charged = next;
        }
        Ok(Lease {
            budget: self.clone(),
            bytes,
            numeric: charge.map(|(generation, _)| (generation, bytes)),
            numeric_keepalive: None,
        })
    }
    pub fn numeric_guard(self: &Arc<Self>, limit: usize) -> Result<Arc<NumericGuard>> {
        if limit == 0 {
            return Err(Error::resource("positive HPO numeric budget required"));
        }
        let mut book = self.numeric.lock().map_err(|_| Error::Closed)?;
        if book.failed {
            return Err(Error::contract(
                "native numeric reservation accounting uncertain",
            ));
        }
        if book.active.is_some() {
            return Err(Error::contract("nested native numeric guard prohibited"));
        }
        let generation = book
            .next
            .checked_add(1)
            .ok_or_else(|| Error::resource("numeric guard generation overflow"))?;
        book.next = generation;
        book.active = Some(NumericPhase {
            generation,
            limit,
            charged: 0,
        });
        Ok(Arc::new(NumericGuard {
            budget: self.clone(),
            generation,
        }))
    }
    pub fn numeric_used(&self) -> Option<usize> {
        self.numeric
            .lock()
            .ok()
            .and_then(|book| book.active.as_ref().map(|phase| phase.charged))
    }
    fn release(&self, bytes: usize, numeric: Option<(u64, usize)>) {
        if let Ok(mut book) = self.numeric.lock() {
            if let (Some(phase), Some((generation, amount))) = (book.active.as_mut(), numeric) {
                if phase.generation == generation {
                    if let Some(next) = phase.charged.checked_sub(amount) {
                        phase.charged = next;
                    } else {
                        book.failed = true;
                    }
                }
            }
            self.used.fetch_sub(bytes, Ordering::AcqRel);
        } else {
            // This lease's reservation effect is known even if the constraint
            // lock is poisoned. Do not invent a numeric refund in that case.
            self.used.fetch_sub(bytes, Ordering::AcqRel);
        }
    }
}
pub struct JsonAdmission {
    _lease: Lease,
    budget: Arc<Budget>,
    bytes: usize,
}
impl Drop for JsonAdmission {
    fn drop(&mut self) {
        self.budget
            .json_estimated_used
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[derive(Debug)]
pub struct Lease {
    budget: Arc<Budget>,
    bytes: usize,
    numeric: Option<(u64, usize)>,
    numeric_keepalive: Option<Arc<NumericGuard>>,
}
impl Lease {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Adjust a logical reservation without briefly dropping existing admission.
    pub fn resize(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            let mut additional = self.budget.reserve(bytes - self.bytes)?;
            let charge = match (self.numeric, additional.numeric) {
                (Some((old, amount)), Some((generation, growth))) if old == generation => Some((
                    generation,
                    amount
                        .checked_add(growth)
                        .ok_or_else(|| Error::resource("numeric lease charge overflow"))?,
                )),
                (_, Some(charge)) => Some(charge),
                (old, None) => old,
            };
            self.bytes = bytes;
            self.numeric = charge;
            additional.bytes = 0; // The existing lease now owns this admitted delta.
            additional.numeric = None;
        } else {
            let numeric = self
                .numeric
                .map(|(generation, amount)| (generation, amount.saturating_sub(bytes)));
            self.budget.release(self.bytes - bytes, numeric);
            self.numeric = self
                .numeric
                .map(|(generation, amount)| (generation, amount.min(bytes)));
            self.bytes = bytes;
        }
        Ok(())
    }
    pub fn retain_numeric_guard(&mut self, guard: Arc<NumericGuard>) {
        self.numeric_keepalive = Some(guard);
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.budget.release(self.bytes, self.numeric);
    }
}

#[derive(Debug)]
pub struct BudgetVec<T> {
    value: Vec<T>,
    lease: Lease,
}
impl<T> BudgetVec<T> {
    pub fn new(budget: &Arc<Budget>) -> Self {
        Self {
            value: Vec::new(),
            lease: budget.reserve(0).expect("zero reservation"),
        }
    }
    pub fn with_capacity(budget: &Arc<Budget>, count: usize) -> Result<Self> {
        let mut result = Self::new(budget);
        result.reserve(count)?;
        Ok(result)
    }
    pub fn budget(&self) -> &Arc<Budget> {
        &self.lease.budget
    }
    pub fn capacity_bytes(&self) -> usize {
        self.value.capacity() * size_of::<T>()
    }
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let needed = self
            .value
            .len()
            .checked_add(additional)
            .ok_or_else(|| Error::resource("capacity overflow"))?;
        if needed <= self.value.capacity() {
            return Ok(());
        }
        let capacity = needed.max(self.value.capacity().saturating_mul(2)).max(4);
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::resource("capacity overflow"))?;
        // Reserve the complete replacement while the old buffer remains charged.
        let lease = self.lease.budget.reserve(bytes)?;
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(capacity)
            .map_err(|_| Error::resource("native allocation failed"))?;
        if replacement.capacity() != capacity {
            return Err(Error::resource("unexpected allocator capacity"));
        }
        if !self.value.is_empty() {
            self.lease
                .budget
                .reallocations
                .fetch_add(1, Ordering::Relaxed);
            self.lease
                .budget
                .reallocated_bytes
                .fetch_add(self.value.len() * size_of::<T>(), Ordering::Relaxed);
        }
        replacement.append(&mut self.value);
        self.value = replacement;
        self.lease = lease;
        Ok(())
    }
    pub fn push(&mut self, value: T) -> Result<()> {
        self.reserve(1)?;
        self.value.push(value);
        Ok(())
    }
    /// Admit growth before safely initializing elements for typed bulk reads.
    pub fn resize(&mut self, length: usize, value: T) -> Result<()>
    where
        T: Clone,
    {
        self.reserve(length.saturating_sub(self.value.len()))?;
        self.value.resize(length, value);
        Ok(())
    }
    pub fn truncate(&mut self, length: usize) {
        self.value.truncate(length);
    }
    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<()>
    where
        T: Clone,
    {
        self.reserve(values.len())?;
        self.value.extend_from_slice(values);
        self.lease
            .budget
            .extend_copy_bytes
            .fetch_add(values.len() * size_of::<T>(), Ordering::Relaxed);
        Ok(())
    }
    pub fn pop(&mut self) -> Option<T> {
        self.value.pop()
    }
    pub fn clear(&mut self) {
        self.value.clear();
    }
    pub fn into_parts(self) -> (Vec<T>, Lease) {
        (self.value, self.lease)
    }
}
impl<T> Deref for BudgetVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.value
    }
}
impl<T> DerefMut for BudgetVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.value
    }
}

#[derive(Debug, Clone)]
pub struct Control {
    cancelled: Arc<AtomicBool>,
    deadline: f64,
}
impl Control {
    pub fn new(deadline: f64) -> Result<Self> {
        if !deadline.is_finite() || deadline <= 0.0 {
            return Err(Error::contract(
                "finite absolute monotonic deadline required",
            ));
        }
        Ok(Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline,
        })
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn at_deadline(&self, deadline: f64) -> Result<Self> {
        if !deadline.is_finite() || deadline <= 0.0 {
            return Err(Error::contract(
                "finite absolute monotonic deadline required",
            ));
        }
        Ok(Self {
            cancelled: self.cancelled.clone(),
            deadline,
        })
    }
    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if monotonic() >= self.deadline {
            return Err(Error::Deadline);
        }
        Ok(())
    }
}

pub fn monotonic() -> f64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: pointer references an initialized timespec for the duration of the call.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
    assert_eq!(status, 0, "CLOCK_MONOTONIC unavailable");
    value.tv_sec as f64 + value.tv_nsec as f64 / 1e9
}

/// Unstable in-place sorting with boundary cancellation. Callers must ensure
/// equal keys are interchangeable, unique, or rejected before publication.
pub fn sort_unstable_by<T>(
    values: &mut [T],
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
    control: &Control,
) -> Result<()> {
    control.check()?;
    values.sort_unstable_by(compare);
    control.check()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_guard_survives_its_retained_payload_lease_and_global_failure_is_atomic() {
        let b = Budget::new(100);
        let source = b.reserve(80).unwrap();
        let guard = b.numeric_guard(50).unwrap();
        assert!(b.reserve(30).is_err());
        assert_eq!(b.numeric_used(), Some(0));
        let mut payload = b.reserve(10).unwrap();
        payload.retain_numeric_guard(guard.clone());
        drop(guard);
        assert!(b.numeric_guard(50).is_err());
        drop(source);
        assert_eq!(b.numeric_used(), Some(10));
        drop(payload);
        assert_eq!(b.numeric_used(), None);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn numeric_guard_installation_and_old_lease_release_races_keep_generations_isolated() {
        let b = Budget::new(1000);
        let running = AtomicBool::new(true);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while running.load(Ordering::Acquire) {
                    if let Ok(lease) = b.reserve(1) {
                        std::thread::yield_now();
                        drop(lease);
                    }
                }
            });
            for _ in 0..100 {
                let guard = b.numeric_guard(50).unwrap();
                assert!(b.numeric_used().unwrap() <= 50);
                drop(guard);
            }
            running.store(false, Ordering::Release);
        });
        assert_eq!(b.used(), 0);
        assert_eq!(b.numeric_used(), None);
    }
    #[test]
    fn numeric_guard_does_not_spend_refunded_original_source_credit() {
        let b = Budget::new(1000);
        let mut original = b.reserve(100).unwrap();
        let guard = b.numeric_guard(50).unwrap();
        original.resize(50).unwrap();
        assert!(b.reserve(100).is_err());
        assert_eq!(b.used(), 50);
        let numeric = b.reserve(50).unwrap();
        assert_eq!(b.numeric_used(), Some(50));
        assert!(b.numeric_guard(50).is_err());
        drop(numeric);
        assert_eq!(b.numeric_used(), Some(0));
        drop(guard);
        drop(original);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn numeric_guard_mixed_resize_and_stale_generation_refunds_are_conservative() {
        let b = Budget::new(1000);
        let mut mixed = b.reserve(100).unwrap();
        let first = b.numeric_guard(50).unwrap();
        mixed.resize(130).unwrap();
        assert_eq!(b.numeric_used(), Some(30));
        mixed.resize(80).unwrap();
        assert_eq!(b.numeric_used(), Some(30));
        mixed.resize(10).unwrap();
        assert_eq!(b.numeric_used(), Some(10));
        let old = b.reserve(20).unwrap();
        drop(first);
        let second = b.numeric_guard(40).unwrap();
        let current = b.reserve(40).unwrap();
        drop(old);
        mixed.resize(0).unwrap();
        assert_eq!(b.numeric_used(), Some(40));
        assert!(b.reserve(1).is_err());
        drop(current);
        drop(second);
        drop(mixed);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn numeric_guard_shared_threads_and_replacement_failure_keep_exact_charges() {
        let b = Budget::new(1000);
        let guard = b.numeric_guard(63).unwrap();
        let values = Arc::new(std::sync::Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let b = b.clone();
                let values = values.clone();
                scope.spawn(move || {
                    if let Ok(lease) = b.reserve(10) {
                        values.lock().unwrap().push(lease);
                    }
                });
            }
        });
        assert_eq!(values.lock().unwrap().len(), 6);
        assert_eq!(b.numeric_used(), Some(60));
        values.lock().unwrap().clear();
        let mut vector = BudgetVec::<u64>::new(&b);
        for n in 0..4 {
            vector.push(n).unwrap();
        }
        assert!(vector.push(4).is_err());
        assert_eq!(b.numeric_used(), Some(32));
        assert_eq!(&*vector, &[0, 1, 2, 3]);
        drop(vector);
        assert_eq!(b.numeric_used(), Some(0));
        drop(guard);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn growth_accounts_overlap_and_releases_failure() {
        let budget = Budget::new(63);
        let mut values = BudgetVec::<u64>::new(&budget);
        for i in 0..4 {
            values.push(i).unwrap();
        }
        assert_eq!(budget.used(), 32);
        assert!(matches!(values.push(4), Err(Error::ResourceLimit(_))));
        assert_eq!(&*values, &[0, 1, 2, 3]);
        drop(values);
        assert_eq!(budget.used(), 0);
    }
}
