//! Conservative admission for Python control trees, independent of array exports.
use crate::{
    budget::{Budget, Lease},
    error::{Error, Result},
};
use std::sync::Arc;

pub const HEADROOM: usize = 32 * 1024 * 1024;
pub fn reservation_bytes(summary: usize, other: usize) -> Result<usize> {
    summary
        .checked_add(other)
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| n.checked_add(HEADROOM))
        .ok_or_else(|| Error::resource("Python control reservation overflow"))
}
#[derive(Default)]
pub struct ControlMemory {
    summary: usize,
    authorization: usize,
    proof: usize,
    data_started: bool,
    hpo_finalized: bool,
    released: bool,
    lease: Option<Lease>,
}
impl ControlMemory {
    pub fn summary(
        &mut self,
        budget: &Arc<Budget>,
        summary: usize,
        authorization: usize,
    ) -> Result<()> {
        if self.lease.is_some() || self.released {
            return Err(Error::contract("control summary already delivered"));
        }
        let lease = budget.reserve(reservation_bytes(summary, authorization)?)?;
        self.summary = summary;
        self.authorization = authorization;
        self.lease = Some(lease);
        Ok(())
    }
    pub fn authorization(&self, length: usize) -> Result<()> {
        if self.lease.is_none() || self.released {
            return Err(Error::contract("control summary admission required"));
        }
        if length > self.authorization {
            return Err(Error::resource(
                "serialized authorization exceeds pre-admitted bound",
            ));
        }
        Ok(())
    }
    pub fn hpo_final_summary(&mut self, length: usize) -> Result<()> {
        if self.released || self.data_started || self.hpo_finalized || length == 0 {
            return Err(Error::contract(
                "one HPO final control delivery before data required",
            ));
        }
        let combined = self
            .summary
            .checked_add(length)
            .ok_or_else(|| Error::resource("HPO control tree size overflow"))?;
        let lease = self
            .lease
            .as_mut()
            .ok_or_else(|| Error::contract("HPO pending control admission required"))?;
        lease.resize(reservation_bytes(combined, self.authorization)?)?;
        self.summary = combined;
        self.hpo_finalized = true;
        Ok(())
    }
    pub fn begin_data(&mut self) -> Result<()> {
        if !self.data_started {
            self.resize(0)?;
            self.data_started = true;
        }
        Ok(())
    }
    pub fn proof(&mut self, length: usize, final_decode: bool) -> Result<()> {
        let proof = self.proof.max(length);
        let copies = if final_decode { 2 } else { 1 };
        let live = proof
            .checked_mul(copies)
            .ok_or_else(|| Error::resource("Python proof reservation overflow"))?;
        self.resize(live)?;
        self.proof = proof;
        Ok(())
    }
    fn resize(&mut self, other: usize) -> Result<()> {
        if self.released {
            return Err(Error::Closed);
        }
        let lease = self
            .lease
            .as_mut()
            .ok_or_else(|| Error::contract("control summary admission required"))?;
        lease.resize(reservation_bytes(self.summary, other)?)
    }
    pub fn reserved(&self) -> usize {
        self.lease.as_ref().map_or(0, Lease::bytes)
    }
    pub fn release(&mut self) {
        self.lease.take();
        self.released = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hpo_final_control_tree_retains_pending_and_final_until_actual_release() {
        let b = Budget::new(128 << 20);
        let mut value = ControlMemory::default();
        value.summary(&b, 1024, 8192).unwrap();
        value.hpo_final_summary(2048).unwrap();
        assert_eq!(value.reserved(), reservation_bytes(3072, 8192).unwrap());
        assert!(value.hpo_final_summary(2048).is_err());
        value.begin_data().unwrap();
        assert_eq!(value.reserved(), reservation_bytes(3072, 0).unwrap());
        value.release();
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn control_lease_survives_core_and_exports_until_explicit_release() {
        let budget = Budget::new(128 * 1024 * 1024);
        let mut control = ControlMemory::default();
        control.summary(&budget, 1024, 16384).unwrap();
        assert_eq!(budget.used(), reservation_bytes(1024, 16384).unwrap());
        assert!(control.authorization(16385).is_err());
        control.begin_data().unwrap();
        control.proof(4096, false).unwrap();
        assert_eq!(control.reserved(), reservation_bytes(1024, 4096).unwrap());
        control.proof(4096, true).unwrap();
        assert_eq!(budget.used(), reservation_bytes(1024, 8192).unwrap());
        control.release();
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn rejects_inflated_controls_before_publication_and_growth_is_atomic() {
        let budget = Budget::new(64 * 1024 * 1024);
        let mut control = ControlMemory::default();
        assert!(matches!(
            control.summary(&budget, 2 * 1024 * 1024, 8192),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(budget.used(), 0);
        control.summary(&budget, 1024, 8192).unwrap();
        let before = budget.used();
        assert!(control.proof(1024 * 1024, true).is_err());
        assert_eq!(budget.used(), before);
        assert!(reservation_bytes(usize::MAX, 1).is_err());
        control.release();
        assert_eq!(budget.used(), 0);
    }
}
