//! Invocation accounting. Reservations precede allocation, including growth peaks.
use crate::error::{Error, Result};
use std::ops::{Deref, DerefMut};
use std::sync::{
    Arc,
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
        let old = self
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|sum| *sum <= self.limit)
            })
            .map_err(|_| Error::resource("native managed memory budget exhausted"))?;
        self.peak.fetch_max(old + bytes, Ordering::AcqRel);
        Ok(Lease {
            budget: self.clone(),
            bytes,
        })
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
}
impl Lease {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Adjust a logical reservation without briefly dropping existing admission.
    pub fn resize(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            let mut additional = self.budget.reserve(bytes - self.bytes)?;
            self.bytes = bytes;
            additional.bytes = 0; // The existing lease now owns this admitted delta.
        } else {
            self.budget
                .used
                .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
            self.bytes = bytes;
        }
        Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
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
    json_depth: usize,
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
            json_depth: 990,
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
            json_depth: self.json_depth,
        })
    }
    pub fn with_json_depth(&self, depth: usize) -> Result<Self> {
        if !(1..=990).contains(&depth) {
            return Err(Error::contract(
                "native_json_depth must be an integer in 1..990",
            ));
        }
        Ok(Self {
            cancelled: self.cancelled.clone(),
            deadline: self.deadline,
            json_depth: depth,
        })
    }
    pub fn json_depth(&self) -> usize {
        self.json_depth
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

/// Standard in-place sorting with the declared boundary cancellation policy.
pub fn sort_by<T>(
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
