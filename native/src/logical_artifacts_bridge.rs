//! The sole Python adapter for the generic original-byte provider.
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    logical_artifacts::{LogicalArtifactProvider, METADATA_LIMIT},
};
use pyo3::{prelude::*, types::PyBytes};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(crate) struct PythonArtifactProvider {
    object: Py<PyAny>,
    active: AtomicBool,
    failed: AtomicBool,
}
struct Callback<'a>(&'a PythonArtifactProvider);
impl Drop for Callback<'_> {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}
impl PythonArtifactProvider {
    pub(crate) fn retain_on_error(&self, py: Python<'_>, error: PyErr) -> PyErr {
        // A failed constructor has no returned native owner. Its exception must
        // retain the actual Python registry, including any uncertain loan.
        let _ = error.value(py).setattr(
            "_native_original_provider_cleanup_owner",
            self.object.bind(py),
        );
        error
    }
    pub(crate) fn new(object: Py<PyAny>) -> Arc<Self> {
        Arc::new(Self {
            object,
            active: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        })
    }
    pub(crate) fn check_reentry(&self) -> Result<()> {
        if self.active.load(Ordering::Acquire) {
            self.failed.store(true, Ordering::Release);
            return Err(Error::contract(
                "original provider cannot reenter native session",
            ));
        }
        Ok(())
    }
    fn enter(&self, control: &Control) -> Result<Callback<'_>> {
        control.check()?;
        self.check_reentry()?;
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        if self.active.swap(true, Ordering::AcqRel) {
            return Err(Error::contract("original provider callback reentry"));
        }
        Ok(Callback(self))
    }
    fn error(&self, error: impl std::fmt::Display) -> Error {
        self.failed.store(true, Ordering::Release);
        Error::contract(format!("original provider callback: {error}"))
    }
    fn metadata(
        &self,
        method: &str,
        root: Option<&str>,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<BudgetVec<u8>> {
        let _callback = self.enter(control)?;
        // Fixed metadata maximum is admitted BEFORE Python can allocate/copy.
        let _admission = budget.reserve(METADATA_LIMIT)?;
        Python::attach(|py| {
            let result = if let Some(root) = root {
                self.object.bind(py).call_method1(method, (root,))
            } else {
                self.object.bind(py).call_method0(method)
            }
            .map_err(|e| self.error(e))?;
            if !result.is_exact_instance_of::<PyBytes>() {
                return Err(self.error("exact metadata bytes required"));
            }
            let raw = result
                .cast::<PyBytes>()
                .map_err(|e| self.error(e))?
                .as_bytes();
            if raw.len() > METADATA_LIMIT {
                return Err(self.error("metadata cap exceeded"));
            }
            let mut destination = BudgetVec::with_capacity(budget, raw.len())?;
            destination.extend_from_slice(raw)?;
            control.check()?;
            if self.failed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            Ok(destination)
        })
    }
}
impl LogicalArtifactProvider for PythonArtifactProvider {
    fn registry(&self, budget: &Arc<Budget>, control: &Control) -> Result<BudgetVec<u8>> {
        self.metadata("registry", None, budget, control)
    }
    fn describe(
        &self,
        root: &str,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<BudgetVec<u8>> {
        self.metadata("describe", Some(root), budget, control)
    }
    fn read(
        &self,
        root: &str,
        name: &str,
        length: usize,
        sha256: &str,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<BudgetVec<u8>> {
        let _callback = self.enter(control)?;
        // The sole Rust destination exists before borrow/data callbacks. There
        // is no extract::<Vec<u8>>() intermediate or uncharged Python copy.
        let mut destination = BudgetVec::with_capacity(budget, length)?;
        Python::attach(|py| {
            let loan = self
                .object
                .bind(py)
                .call_method1("borrow", (root, name))
                .map_err(|e| self.error(e))?;
            let result = (|| {
                let value = loan.call_method0("data").map_err(|e| self.error(e))?;
                if !value.is_exact_instance_of::<PyBytes>() {
                    return Err(self.error("exact original PyBytes required"));
                }
                let raw = value
                    .cast::<PyBytes>()
                    .map_err(|e| self.error(e))?
                    .as_bytes();
                if raw.len() != length || format!("{:x}", Sha256::digest(raw)) != sha256 {
                    return Err(self.error("original length/hash differs"));
                }
                destination.extend_from_slice(raw)?;
                control.check()?;
                if self.failed.load(Ordering::Acquire) {
                    return Err(Error::Closed);
                }
                Ok(())
            })();
            // Explicit close runs on EVERY admitted loan path, before returning
            // the Rust bytes. The concrete Python registry retains uncertainty.
            let close = loan.call_method0("close").map_err(|e| self.error(e));
            match (result, close) {
                (Err(primary), Err(cleanup)) => Err(Error::Cleanup {
                    primary: Box::new(primary),
                    cleanup: cleanup.to_string(),
                }),
                (Err(primary), _) => Err(primary),
                (_, Err(cleanup)) => Err(cleanup),
                (Ok(()), Ok(_)) => Ok(destination),
            }
        })
    }
    fn recheck(&self, root: &str, control: &Control) -> Result<()> {
        let _callback = self.enter(control)?;
        Python::attach(|py| {
            let value = self
                .object
                .bind(py)
                .call_method1("recheck", (root,))
                .map_err(|e| self.error(e))?;
            if !value.is_none() {
                return Err(self.error("original recheck must return None"));
            }
            if self.failed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            control.check()
        })
    }
}
