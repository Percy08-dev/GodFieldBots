use crate::json::ValueExt;
/// Packed original-byte contract accepted after independent reviews.
pub const CARD_INSTANCE_GROUP_PACK_CAPABILITY: &str = "provider1;proof2;plan4";
// Explicit packed variants reviewed independently; Source admission remains separate.
pub const CARD_INSTANCE_PACKED_FORMAL_CAPABILITY: &str =
    "run4-formal-packed-parent-v1;declaration2;stop2";
pub const CARD_INSTANCE_PACKED_FINAL_CAPABILITY: &str =
    "final64-packed;provider1;request2;summary2;lease1";
pub mod budget;
mod card_instance_fixture;
pub mod control_memory;
pub mod controls;
mod episode_pool;
pub mod error;
pub mod feature;
pub mod index;
pub mod journal;
pub mod json;
mod python_formatter;
mod logical_artifacts;
mod logical_artifacts_bridge;
mod numeric_io;
pub mod parallel;
pub mod pipeline;
pub mod projection;
pub mod player_runtime;
pub mod retirement;
pub mod session;
pub mod source;
pub mod stack;
pub mod structured_controls;
pub mod structured_journal;
pub mod structured_session;
pub mod structured_source;
pub mod structured_staging;
pub mod structured_tokens;
mod structured_tree_dataset;
pub mod telemetry;
pub(crate) mod typed_journal;
pub mod validation;

/// Diagnostic controls-only owner. This deliberately does not activate reader2.
#[pyclass]
struct StructuredParentControls {
    core: Mutex<Option<structured_session::ParentControls>>,
    retirement: Mutex<Option<retirement::Retirement<structured_session::ParentControls>>>,
    budget: Arc<budget::Budget>,
    control: budget::Control,
    python: Mutex<control_memory::ControlMemory>,
    caller: Mutex<budget::Lease>,
}
fn retire_structured_controls(core: structured_session::ParentControls) -> error::Result<()> {
    drop(core);
    Ok(())
}
#[pymethods]
impl StructuredParentControls {
    #[new]
    #[pyo3(signature=(directory,settings,deadline,cancel_handle=None))]
    fn new(
        py: Python<'_>,
        directory: &str,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        if settings.len() > 65536 {
            return Err(python_error(Error::resource(
                "oversized structured diagnostic settings",
            )));
        }
        let control = match cancel_handle {
            Some(c) => c.control.at_deadline(deadline),
            None => budget::Control::new(deadline),
        }
        .map_err(python_error)?;
        let (core, retirement, budget) = py
            .detach(|| {
                control.check()?;
                let initial = budget::Budget::new(1048576);
                let settings = json::parse(settings, &initial, &control)?;
                settings
                    .root
                    .closed(&["memory_bytes", "max_json_bytes"], &[])?;
                let memory = settings
                    .root
                    .require("memory_bytes")?
                    .integer(1, 12i128 << 30)? as usize;
                let max_json = settings
                    .root
                    .require("max_json_bytes")?
                    .integer(1, 64i128 << 20)? as usize;
                let budget = budget::Budget::new(memory);
                let retirement =
                    retirement::Retirement::start(&budget, retire_structured_controls)?;
                let core = structured_session::ParentControls::open(
                    std::path::Path::new(directory),
                    budget.clone(),
                    control.clone(),
                    max_json,
                )?;
                Ok((core, retirement, budget))
            })
            .map_err(python_error)?;
        let caller = budget.reserve(0).map_err(python_error)?;
        Ok(Self {
            core: Mutex::new(Some(core)),
            retirement: Mutex::new(Some(retirement)),
            budget,
            control,
            python: Mutex::new(control_memory::ControlMemory::default()),
            caller: Mutex::new(caller),
        })
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.control.check()?;
                self.caller
                    .lock()
                    .map_err(|_| Error::contract("structured caller lease poisoned"))?
                    .resize(retained_python_bytes)?;
                let core = self
                    .core
                    .lock()
                    .map_err(|_| Error::contract("structured controls poisoned"))?;
                let core = core.as_ref().ok_or(Error::Closed)?;
                core.sources.recheck()?;
                let bytes = json::canonical(&core.summary.root, &self.budget, &self.control)?;
                self.python
                    .lock()
                    .map_err(|_| Error::contract("structured controls lease poisoned"))?
                    .summary(&self.budget, bytes.len(), 0)?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn recheck(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        py.detach(|| {
            self.control.check()?;
            self.caller
                .lock()
                .map_err(|_| Error::contract("structured caller lease poisoned"))?
                .resize(retained_python_bytes)?;
            let core = self
                .core
                .lock()
                .map_err(|_| Error::contract("structured controls poisoned"))?;
            core.as_ref().ok_or(Error::Closed)?.sources.recheck()
        })
        .map_err(python_error)
    }
    fn used_bytes(&self) -> usize {
        self.budget.used()
    }
    fn peak_bytes(&self) -> usize {
        self.budget.peak()
    }
    fn control_reserved_bytes(&self) -> usize {
        self.python.lock().map(|c| c.reserved()).unwrap_or(0)
    }
    fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            control: self.control.clone(),
            stats: Arc::new(telemetry::Telemetry::default()),
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut core = self
                .core
                .lock()
                .map_err(|_| Error::contract("structured controls poisoned"))?;
            let Some(owned) = core.take() else {
                return Ok(());
            };
            let result = owned.sources.recheck();
            let retirement = self
                .retirement
                .lock()
                .map_err(|_| Error::contract("structured retirement poisoned"))?
                .take()
                .ok_or(Error::Closed)?;
            let retired = retirement.retire(Some(owned));
            self.python
                .lock()
                .map_err(|_| Error::contract("structured controls lease poisoned"))?
                .release();
            self.caller
                .lock()
                .map_err(|_| Error::contract("structured caller lease poisoned"))?
                .resize(0)?;
            result.and(retired)
        })
        .map_err(python_error)
    }
}
impl Drop for StructuredParentControls {
    fn drop(&mut self) {
        if let Ok(core) = self.core.get_mut() {
            if let Some(owned) = core.take() {
                if let Ok(retirement) = self.retirement.get_mut() {
                    if let Some(retirement) = retirement.take() {
                        let _ = retirement.retire(Some(owned));
                    }
                }
            }
        }
    }
}

/// Diagnostic J5/raw-token owner. This deliberately does not activate reader2.
#[pyclass]
struct StructuredJournalTokens {
    provider: Option<Arc<logical_artifacts_bridge::PythonArtifactProvider>>,
    core: Mutex<Option<structured_journal::Journal>>,
    retirement: Mutex<Option<retirement::Retirement<structured_journal::Journal>>>,
    budget: Arc<budget::Budget>,
    control: budget::Control,
    python: Mutex<control_memory::ControlMemory>,
    caller: Mutex<budget::Lease>,
    corrected: bool,
    phase: Mutex<JournalRawPhase>,
}
fn retire_structured_journal(core: structured_journal::Journal) -> error::Result<()> {
    drop(core);
    Ok(())
}
// This phase belongs only to the additive corrected entry. Engineering J5
// keeps its historical refusal/export/close behavior.
#[derive(Debug)]
enum JournalRawPhase {
    Active,
    Exported,
    Publishing,
    Failed(Error),
    Closed(Option<Error>),
    Uncertain(Error),
}
impl StructuredJournalTokens {
    fn open_shared(
        py: Python<'_>,
        directory: &str,
        journal_file: &str,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
        corrected: bool,
    ) -> PyResult<Self> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        if settings.len() > 65536 {
            return Err(python_error(Error::resource(
                "oversized structured diagnostic settings",
            )));
        }
        let control = match cancel_handle {
            Some(c) => c.control.at_deadline(deadline),
            None => budget::Control::new(deadline),
        }
        .map_err(python_error)?;
        let (core, retirement, budget) = py
            .detach(|| {
                control.check()?;
                let initial = budget::Budget::new(1048576);
                let settings = json::parse(settings, &initial, &control)?;
                settings
                    .root
                    .closed(&["memory_bytes", "max_json_bytes", "max_record_bytes"], &[])?;
                let memory = settings
                    .root
                    .require("memory_bytes")?
                    .integer(1, 12i128 << 30)? as usize;
                let max_json = settings
                    .root
                    .require("max_json_bytes")?
                    .integer(1, 64i128 << 20)? as usize;
                let max_record = settings
                    .root
                    .require("max_record_bytes")?
                    .integer(1, max_json as i128)? as usize;
                let budget = budget::Budget::new(memory);
                let retirement = retirement::Retirement::start(&budget, retire_structured_journal)?;
                let open = if corrected {
                    structured_journal::Journal::open_corrected
                } else {
                    structured_journal::Journal::open
                };
                let core = open(
                    std::path::Path::new(directory),
                    journal_file,
                    budget.clone(),
                    control.clone(),
                    max_json,
                    max_record,
                )?;
                Ok((core, retirement, budget))
            })
            .map_err(python_error)?;
        let caller = budget.reserve(0).map_err(python_error)?;
        Ok(Self {
            provider: None,
            core: Mutex::new(Some(core)),
            retirement: Mutex::new(Some(retirement)),
            budget,
            control,
            python: Mutex::new(control_memory::ControlMemory::default()),
            caller: Mutex::new(caller),
            corrected,
            phase: Mutex::new(JournalRawPhase::Active),
        })
    }
    fn raw_phase(&self) -> error::Result<std::sync::MutexGuard<'_, JournalRawPhase>> {
        if let Some(provider) = &self.provider {
            provider.check_reentry()?;
        }
        self.phase
            .lock()
            .map_err(|_| Error::contract("corrected RAW phase poisoned"))
    }
    fn admitted_phase(phase: &JournalRawPhase, export: bool) -> error::Result<()> {
        match phase {
            JournalRawPhase::Active => Ok(()),
            JournalRawPhase::Exported if !export => Ok(()),
            JournalRawPhase::Failed(error)
            | JournalRawPhase::Uncertain(error)
            | JournalRawPhase::Closed(Some(error)) => Err(error.clone()),
            _ => Err(Error::Closed),
        }
    }
    fn corrected_diagnostic(&self, retained: usize) -> error::Result<budget::BudgetVec<u8>> {
        let mut phase = self.raw_phase()?;
        Self::admitted_phase(&phase, true)?;
        // Latch BEFORE quota, source/hash, decode or export work. Publication is
        // completed separately under the GIL; no partially published bytes escape.
        *phase = JournalRawPhase::Publishing;
        let result: error::Result<budget::BudgetVec<u8>> = (|| {
            self.control.check()?;
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained)?;
            let core = self.core.lock().map_err(|_| Error::Closed)?;
            let core = core.as_ref().ok_or(Error::Closed)?;
            core.sources.recheck()?;
            let bytes = core.diagnostic()?;
            self.python
                .lock()
                .map_err(|_| Error::Closed)?
                .summary(&self.budget, bytes.len(), 0)?;
            Ok(bytes)
        })();
        if let Err(error) = &result {
            *phase = JournalRawPhase::Failed(error.clone());
        }
        result
    }
    fn raw_publication_guard(&self) -> error::Result<()> {
        let mut phase = self.raw_phase()?;
        if !matches!(*phase, JournalRawPhase::Publishing) {
            return Err(Error::Closed);
        }
        let result = self.control.check();
        if let Err(error) = &result {
            *phase = JournalRawPhase::Failed(error.clone());
        }
        result
    }
    fn raw_final_source_guard(&self) -> error::Result<()> {
        let mut phase = self.raw_phase()?;
        if !matches!(*phase, JournalRawPhase::Publishing) {
            return Err(Error::Closed);
        }
        let result = (|| {
            self.control.check()?;
            let core = self.core.lock().map_err(|_| Error::Closed)?;
            core.as_ref().ok_or(Error::Closed)?.sources.recheck()?;
            self.control.check()
        })();
        if let Err(error) = &result {
            *phase = JournalRawPhase::Failed(error.clone());
        }
        // Keep Publishing across detached I/O and its GIL reacquisition.
        result
    }
    fn finish_raw_publication(&self, success: bool) -> error::Result<()> {
        let mut phase = self.raw_phase()?;
        if !matches!(*phase, JournalRawPhase::Publishing) {
            return Err(Error::Closed);
        }
        if !success {
            *phase =
                JournalRawPhase::Failed(Error::resource("corrected RAW Python publication failed"));
            return Ok(());
        }
        // Called with the GIL AFTER the final detached Source guard. The final
        // GIL reacquisition can itself expire or cancel an otherwise valid copy.
        let result = self.control.check();
        *phase = match &result {
            Ok(()) => JournalRawPhase::Exported,
            Err(error) => JournalRawPhase::Failed(error.clone()),
        };
        result
    }
    fn corrected_recheck(&self, retained: usize) -> error::Result<()> {
        let mut phase = self.raw_phase()?;
        Self::admitted_phase(&phase, false)?;
        let exported = matches!(*phase, JournalRawPhase::Exported);
        *phase = JournalRawPhase::Failed(Error::Closed);
        let result = (|| {
            self.control.check()?;
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained)?;
            let core = self.core.lock().map_err(|_| Error::Closed)?;
            core.as_ref().ok_or(Error::Closed)?.sources.recheck()
        })();
        *phase = match &result {
            Ok(()) if exported => JournalRawPhase::Exported,
            Ok(()) => JournalRawPhase::Active,
            Err(error) => JournalRawPhase::Failed(error.clone()),
        };
        result
    }
    fn corrected_close(&self) -> error::Result<()> {
        let mut phase = self.raw_phase()?;
        let primary = match &*phase {
            JournalRawPhase::Closed(None) => return Ok(()),
            JournalRawPhase::Closed(Some(error)) | JournalRawPhase::Uncertain(error) => {
                return Err(error.clone());
            }
            JournalRawPhase::Failed(error) => Some(error.clone()),
            // The same object may be reached while another Python thread is
            // publishing. It must finish or fail before cleanup can transfer it.
            JournalRawPhase::Publishing => {
                return Err(Error::contract("corrected RAW publication in progress"));
            }
            _ => None,
        };
        *phase = JournalRawPhase::Failed(primary.clone().unwrap_or(Error::Closed));
        let result = (|| {
            // Acquire EVERY fallible ownership lock before taking the core.
            // Poison/absent cleanup owners remain reachable; an empty retry is
            // never interpreted as confirmed retirement.
            let mut core = self.core.lock().map_err(|_| Error::Closed)?;
            let mut worker = self.retirement.lock().map_err(|_| Error::Closed)?;
            let mut python = self.python.lock().map_err(|_| Error::Closed)?;
            let mut caller = self.caller.lock().map_err(|_| Error::Closed)?;
            let owned = core.as_ref().ok_or(Error::Closed)?;
            let checked = owned.sources.recheck();
            if worker.is_none() {
                return Err(Error::contract(
                    "corrected RAW retirement owner unavailable",
                ));
            }
            let retirement = worker.take().unwrap();
            let retired = retirement.retire(core.take());
            if let Err(cleanup) = retired {
                let error = match primary.clone().or_else(|| checked.err()) {
                    Some(primary) => Error::Cleanup {
                        primary: Box::new(primary),
                        cleanup: format!("{cleanup:?}"),
                    },
                    None => cleanup,
                };
                *phase = JournalRawPhase::Uncertain(error.clone());
                return Err(error);
            }
            python.release();
            caller.resize(0)?;
            let failure = primary.clone().or_else(|| checked.err());
            *phase = JournalRawPhase::Closed(failure.clone());
            failure.map_or(Ok(()), Err)
        })();
        if let Err(error) = &result {
            if !matches!(
                *phase,
                JournalRawPhase::Closed(_) | JournalRawPhase::Uncertain(_)
            ) {
                *phase = JournalRawPhase::Failed(primary.unwrap_or_else(|| error.clone()));
            }
        }
        result
    }
}
#[pymethods]
impl StructuredJournalTokens {
    #[staticmethod]
    #[pyo3(signature=(logical_root,episode_id,journal_file,provider,settings,deadline,cancel_handle=None))]
    fn card_instance_artifacts(
        py: Python<'_>,
        logical_root: &str,
        episode_id: &str,
        journal_file: &str,
        provider: Py<PyAny>,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        let provider = logical_artifacts_bridge::PythonArtifactProvider::new(provider);
        (|| {
            use logical_artifacts::LogicalArtifactProvider;
            use sha2::Digest;
            stack::ensure_minimum_stack().map_err(python_error)?;
            if settings.len() > 65536
                || logical_root.len() > 65536
                || episode_id.len() > 4096
                || !["player-1.jsonl", "player-2.jsonl"].contains(&journal_file)
            {
                return Err(python_error(Error::contract(
                    "bounded registered own RAW episode required",
                )));
            }
            logical_artifacts::member_name(episode_id).map_err(python_error)?;
            if episode_id.contains('/') {
                return Err(python_error(Error::contract(
                    "own episode basename required",
                )));
            }
            let control = match cancel_handle {
                Some(c) => c.control.at_deadline(deadline),
                None => Control::new(deadline),
            }
            .map_err(python_error)?;
            let wire = StructuredSessionBridge::owned_wire(&Budget::new(1 << 20), settings)
                .map_err(python_error)?;
            let (core, retirement, budget) = py
                .detach(|| {
                    control.check()?;
                    let initial = Budget::new(1 << 20);
                    let settings = json::parse(&wire, &initial, &control)?;
                    settings
                        .root
                        .closed(&["memory_bytes", "max_json_bytes", "max_record_bytes"], &[])?;
                    let memory = settings
                        .root
                        .require("memory_bytes")?
                        .integer(1, 12i128 << 30)? as usize;
                    let max_json = settings
                        .root
                        .require("max_json_bytes")?
                        .integer(1, 64i128 << 20)? as usize;
                    let max_record = settings
                        .root
                        .require("max_record_bytes")?
                        .integer(1, max_json as i128)?
                        as usize;
                    let budget = Budget::new(memory);
                    let raw = provider.registry(&budget, &control)?;
                    let digest = format!("{:x}", sha2::Sha256::digest(&*raw));
                    let registry = logical_artifacts::Registry::open(
                        provider.clone(),
                        &digest,
                        &budget,
                        &control,
                        max_json,
                    )?;
                    if !registry
                        .roots
                        .contains_key(std::path::Path::new(logical_root))
                    {
                        return Err(Error::contract(
                            "RAW root absent from original provider registry",
                        ));
                    }
                    let retirement =
                        retirement::Retirement::start(&budget, retire_structured_journal)?;
                    let directory = std::path::Path::new(logical_root).join(episode_id);
                    let core = structured_journal::Journal::open_corrected_artifacts(
                        &directory,
                        journal_file,
                        budget.clone(),
                        control.clone(),
                        max_json,
                        max_record,
                        registry,
                    )?;
                    let viewer = if journal_file == "player-1.jsonl" {
                        1
                    } else {
                        2
                    };
                    if core
                        .header_value()
                        .require("viewer")?
                        .integer(viewer, viewer)?
                        != viewer
                        || core.header_value().require("episode_id")?.string()? != episode_id
                    {
                        return Err(Error::contract("RAW registered own viewer differs"));
                    }
                    core.sources.recheck()?;
                    Ok((core, retirement, budget))
                })
                .map_err(python_error)?;
            let caller = budget.reserve(0).map_err(python_error)?;
            Ok(Self {
                provider: Some(provider.clone()),
                core: Mutex::new(Some(core)),
                retirement: Mutex::new(Some(retirement)),
                budget,
                control,
                python: Mutex::new(control_memory::ControlMemory::default()),
                caller: Mutex::new(caller),
                corrected: true,
                phase: Mutex::new(JournalRawPhase::Active),
            })
        })()
        .map_err(|error| provider.retain_on_error(py, error))
    }
    #[new]
    #[pyo3(signature=(directory,journal_file,settings,deadline,cancel_handle=None))]
    fn new(
        py: Python<'_>,
        directory: &str,
        journal_file: &str,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        Self::open_shared(
            py,
            directory,
            journal_file,
            settings,
            deadline,
            cancel_handle,
            false,
        )
    }
    #[classmethod]
    #[pyo3(signature=(directory,journal_file,settings,deadline,cancel_handle=None))]
    fn card_instance(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        directory: &str,
        journal_file: &str,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        Self::open_shared(
            py,
            directory,
            journal_file,
            settings,
            deadline,
            cancel_handle,
            true,
        )
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn diagnostic<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if self.corrected {
            let bytes = py
                .detach(|| self.corrected_diagnostic(retained_python_bytes))
                .map_err(python_error)?;
            self.raw_publication_guard().map_err(python_error)?;
            let result = PyBytes::new_with(py, bytes.len(), |output| {
                output.copy_from_slice(&bytes);
                Ok(())
            });
            // Allocation failure exposes no bytes and permanently prevents a
            // second operational attempt. The actual original owner is retained.
            match result {
                Ok(value) => {
                    py.detach(|| self.raw_final_source_guard())
                        .map_err(python_error)?;
                    self.finish_raw_publication(true).map_err(python_error)?;
                    return Ok(value);
                }
                Err(error) => {
                    let _ = self.finish_raw_publication(false);
                    return Err(error);
                }
            }
        }
        let bytes = py
            .detach(|| {
                self.control.check()?;
                self.caller
                    .lock()
                    .map_err(|_| Error::contract("structured caller lease poisoned"))?
                    .resize(retained_python_bytes)?;
                let core = self
                    .core
                    .lock()
                    .map_err(|_| Error::contract("structured controls poisoned"))?;
                let core = core.as_ref().ok_or(Error::Closed)?;
                core.sources.recheck()?;
                let bytes = core.diagnostic()?;
                self.python
                    .lock()
                    .map_err(|_| Error::contract("structured controls lease poisoned"))?
                    .summary(&self.budget, bytes.len(), 0)?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn recheck(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        if self.corrected {
            return py
                .detach(|| self.corrected_recheck(retained_python_bytes))
                .map_err(python_error);
        }
        py.detach(|| {
            self.control.check()?;
            self.caller
                .lock()
                .map_err(|_| Error::contract("structured caller lease poisoned"))?
                .resize(retained_python_bytes)?;
            let core = self
                .core
                .lock()
                .map_err(|_| Error::contract("structured controls poisoned"))?;
            core.as_ref().ok_or(Error::Closed)?.sources.recheck()
        })
        .map_err(python_error)
    }
    fn used_bytes(&self) -> PyResult<usize> {
        if let Some(provider) = &self.provider {
            provider.check_reentry().map_err(python_error)?;
        }
        Ok(self.budget.used())
    }
    fn peak_bytes(&self) -> PyResult<usize> {
        if let Some(provider) = &self.provider {
            provider.check_reentry().map_err(python_error)?;
        }
        Ok(self.budget.peak())
    }
    fn control_reserved_bytes(&self) -> PyResult<usize> {
        if let Some(provider) = &self.provider {
            provider.check_reentry().map_err(python_error)?;
        }
        Ok(self.python.lock().map(|c| c.reserved()).unwrap_or(0))
    }
    fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            control: self.control.clone(),
            stats: Arc::new(telemetry::Telemetry::default()),
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        if self.corrected {
            return py.detach(|| self.corrected_close()).map_err(python_error);
        }
        py.detach(|| {
            let mut core = self
                .core
                .lock()
                .map_err(|_| Error::contract("structured controls poisoned"))?;
            let Some(owned) = core.take() else {
                return Ok(());
            };
            let result = owned.sources.recheck();
            let retirement = self
                .retirement
                .lock()
                .map_err(|_| Error::contract("structured retirement poisoned"))?
                .take()
                .ok_or(Error::Closed)?;
            let retired = retirement.retire(Some(owned));
            self.python
                .lock()
                .map_err(|_| Error::contract("structured controls lease poisoned"))?
                .release();
            self.caller
                .lock()
                .map_err(|_| Error::contract("structured caller lease poisoned"))?
                .resize(0)?;
            result.and(retired)
        })
        .map_err(python_error)
    }
}
impl Drop for StructuredJournalTokens {
    fn drop(&mut self) {
        if self.corrected {
            let core = self.core.get_mut().unwrap_or_else(|p| p.into_inner());
            let worker = self.retirement.get_mut().unwrap_or_else(|p| p.into_inner());
            if let Some(owned) = core.take() {
                if let Some(worker) = worker.take() {
                    let _ = worker.retire(Some(owned));
                } else {
                    // An uncertain owner is quarantined, never dropped on the caller stack.
                    std::mem::forget(owned);
                }
            }
            return;
        }
        if let Ok(core) = self.core.get_mut() {
            if let Some(owned) = core.take() {
                if let Ok(retirement) = self.retirement.get_mut() {
                    if let Some(retirement) = retirement.take() {
                        let _ = retirement.retire(Some(owned));
                    }
                }
            }
        }
    }
}

fn structured_settings(
    settings: &[u8],
    control: &Control,
) -> error::Result<(usize, usize, usize, usize, usize)> {
    structured_settings_context(settings, control, false)
}
fn structured_settings_context(
    settings: &[u8],
    control: &Control,
    hpo: bool,
) -> error::Result<(usize, usize, usize, usize, usize)> {
    let initial = Budget::new(1048576);
    let settings = json::parse(settings, &initial, &control)?;
    let v = &settings.root;
    v.closed(
        &[
            "memory_bytes",
            "scratch_bytes",
            "max_source_file_bytes",
            "max_record_bytes",
            "chunk_target_bytes",
        ],
        &[],
    )?;
    let limit = if hpo { i64::MAX as i128 } else { 12i128 << 30 };
    let memory = v.require("memory_bytes")?.integer(1, limit)? as usize;
    let scratch = v.require("scratch_bytes")?.integer(1, limit)? as usize;
    let source = v
        .require("max_source_file_bytes")?
        .integer(1, 64i128 << 20)? as usize;
    let record = v.require("max_record_bytes")?.integer(1, source as i128)? as usize;
    let chunk = v.require("chunk_target_bytes")?.integer(1, 64i128 << 20)? as usize;
    Ok((memory, scratch, source, record, chunk))
}

enum StructuredBridgeCore {
    Engineering(structured_session::CohortSession),
    Corrected(structured_session::CardInstanceNativeSession),
}
impl StructuredBridgeCore {
    fn corrected(&self) -> bool {
        matches!(self, Self::Corrected(_))
    }
    fn hpo(&self) -> bool {
        matches!(self,Self::Corrected(value) if value.is_hpo())
    }
    fn summary(&self) -> error::Result<budget::BudgetVec<u8>> {
        match self {
            Self::Engineering(v) => v.summary(),
            Self::Corrected(v) => v.control_summary(),
        }
    }
    fn authorization_max_bytes(&self) -> error::Result<usize> {
        match self {
            Self::Engineering(v) => v
                .controls
                .summary
                .root
                .require("authorization_max_bytes")?
                .integer(1, 64i128 << 20)
                .map(|v| v as usize),
            Self::Corrected(v) => v.authorization_max_bytes(),
        }
    }
    fn original_config_inputs(&self, index: usize) -> error::Result<budget::BudgetVec<u8>> {
        match self {
            Self::Engineering(v) => v.original_config_inputs(index),
            Self::Corrected(v) => v.original_config_inputs(index),
        }
    }
    fn authorize(&mut self, bytes: &[u8]) -> error::Result<()> {
        match self {
            Self::Engineering(v) => v.authorize(bytes),
            Self::Corrected(v) => v.authorize(bytes),
        }
    }
    fn authorize_hpo(&mut self, bytes: &[u8]) -> error::Result<budget::BudgetVec<u8>> {
        match self {
            Self::Corrected(value) => value.authorize_hpo(bytes),
            _ => Err(Error::contract(
                "HPO authorization requires corrected same session",
            )),
        }
    }
    fn export(
        &mut self,
    ) -> error::Result<(
        budget::BudgetVec<u8>,
        structured_staging::Banks,
        Option<(Arc<()>, budget::Lease)>,
    )> {
        match self {
            Self::Engineering(v) => {
                let (m, b) = v.export()?;
                Ok((m, b, None))
            }
            Self::Corrected(v) => {
                let (m, b, g, l) = v.export_cohort()?.into_parts();
                Ok((m, b, Some((g, l))))
            }
        }
    }
    fn export_recorded(
        &mut self,
        index: usize,
    ) -> error::Result<(
        budget::BudgetVec<u8>,
        structured_staging::Banks,
        Option<(Arc<()>, budget::Lease)>,
    )> {
        match self {
            Self::Corrected(v) => {
                let (m, b, g, l) = v.export_recorded_cohort(index)?.into_parts();
                Ok((m, b, Some((g, l))))
            }
            _ => Err(Error::contract(
                "recorded export requires corrected session",
            )),
        }
    }
    fn export_selected(
        &mut self,
        recorded: Option<usize>,
    ) -> error::Result<(
        budget::BudgetVec<u8>,
        structured_staging::Banks,
        Option<(Arc<()>, budget::Lease)>,
    )> {
        match recorded {
            Some(index) => self.export_recorded(index),
            None => self.export(),
        }
    }
    fn fail_export(&mut self) {
        match self {
            Self::Engineering(v) => v.fail_export(),
            Self::Corrected(v) => v.fail_admitted(),
        }
    }
    fn retire_payload(&mut self) -> error::Result<()> {
        match self {
            Self::Engineering(v) => v.retire_payload(),
            Self::Corrected(v) => v.retire_payload(),
        }
    }
    fn begin_update(&mut self) -> error::Result<()> {
        match self {
            Self::Engineering(v) => v.begin_update(),
            Self::Corrected(v) => v.begin_update(),
        }
    }
    fn recheck(&self) -> error::Result<()> {
        match self {
            Self::Engineering(v) => v.recheck(),
            Self::Corrected(v) => v.recheck(),
        }
    }
    fn finalize(&mut self) -> error::Result<budget::BudgetVec<u8>> {
        match self {
            Self::Engineering(v) => v.finalize(),
            _ => Err(Error::contract(
                "corrected finalization uses explicit entry",
            )),
        }
    }
    fn close(&mut self) -> error::Result<()> {
        match self {
            Self::Engineering(v) => v.close(),
            Self::Corrected(v) => v.close(),
        }
    }
    fn is_closed(&self) -> bool {
        match self {
            Self::Engineering(v) => v.is_closed(),
            Self::Corrected(v) => v.is_closed(),
        }
    }
    fn preflight(&self, op: structured_session::CardInstanceOperation) -> error::Result<()> {
        match self {
            Self::Corrected(v) => v.preflight(op),
            _ => Ok(()),
        }
    }
}
/// Candidate structured Source/typed-bank path; markers remain disabled until acceptance.
struct StructuredSessionBridge {
    provider: Option<Arc<logical_artifacts_bridge::PythonArtifactProvider>>,
    final_provider: Mutex<Option<Arc<logical_artifacts_bridge::PythonArtifactProvider>>>,
    core: Mutex<Option<StructuredBridgeCore>>,
    retirement: Mutex<Option<retirement::Retirement<structured_session::CohortSession>>>,
    budget: Arc<Budget>,
    control: Control,
    python: Mutex<control_memory::ControlMemory>,
    input_queries: Mutex<std::collections::BTreeMap<usize, budget::Lease>>,
    caller: Mutex<budget::Lease>,
    payload: Mutex<Option<Arc<ExportLease>>>,
}
fn retire_structured_session(core: structured_session::CohortSession) -> error::Result<()> {
    drop(core);
    Ok(())
}
impl StructuredSessionBridge {
    fn callback_guard(&self) -> PyResult<()> {
        self.provider
            .as_ref()
            .map_or(Ok(()), |p| p.check_reentry())
            .map_err(python_error)?;
        self.final_provider
            .lock()
            .map_err(|_| python_error(Error::Closed))?
            .as_ref()
            .map_or(Ok(()), |p| p.check_reentry())
            .map_err(python_error)
    }
    fn release_final_consumer(&self, require_released: bool) -> error::Result<()> {
        let final_packet = {
            let core = self.core.lock().map_err(|_| Error::Closed)?;
            matches!(core.as_ref(),Some(StructuredBridgeCore::Corrected(v)) if v.final_has_packet() || v.original_has_packet())
        };
        if final_packet {
            release_final_packet_consumer(&self.payload, require_released)?;
        }
        Ok(())
    }

    fn owned_wire(b: &Arc<Budget>, bytes: &[u8]) -> error::Result<budget::BudgetVec<u8>> {
        let mut result = budget::BudgetVec::with_capacity(b, bytes.len())?;
        for value in bytes {
            result.push(*value)?;
        }
        Ok(result)
    }

    fn mutating<T>(
        &self,
        op: structured_session::CardInstanceOperation,
        work: impl FnOnce() -> error::Result<T>,
    ) -> error::Result<T> {
        let corrected = {
            let core = self.core.lock().map_err(|_| Error::Closed)?;
            let core = core.as_ref().ok_or(Error::Closed)?;
            if core.corrected() {
                core.preflight(op)?;
            }
            core.corrected()
        };
        let result = work();
        if result.is_err() && corrected {
            if let Ok(mut core) = self.core.lock() {
                if let Some(core) = core.as_mut() {
                    core.fail_export();
                }
            }
        }
        result
    }
    fn new(
        py: Python<'_>,
        corrected: bool,
        plan: &[u8],
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        Self::new_context(py, corrected, plan, settings, deadline, cancel_handle, None)
    }
    fn new_context(
        py: Python<'_>,
        corrected: bool,
        plan: &[u8],
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
        provider: Option<Arc<logical_artifacts_bridge::PythonArtifactProvider>>,
    ) -> PyResult<Self> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        if settings.len() > 65536 || plan.len() > 65536 {
            return Err(python_error(Error::resource("structured request wire cap")));
        }
        let wire_budget = Budget::new(1048576);
        let owned_plan = if corrected {
            Some(Self::owned_wire(&wire_budget, plan).map_err(python_error)?)
        } else {
            None
        };
        let owned_settings = if corrected {
            Some(Self::owned_wire(&wire_budget, settings).map_err(python_error)?)
        } else {
            None
        };
        let plan = owned_plan.as_deref().unwrap_or(plan);
        let settings = owned_settings.as_deref().unwrap_or(settings);
        let control = match cancel_handle {
            Some(c) => c.control.at_deadline(deadline),
            None => Control::new(deadline),
        }
        .map_err(python_error)?;
        let (core, retirement, budget) = py
            .detach(|| {
                control.check()?;
                let hpo = if corrected {
                    let parsed = json::parse(plan, &wire_budget, &control)?;
                    matches!(parsed.root.get("schema_version").and_then(json::Value::as_i64), Some(5 | 6))
                } else {
                    false
                };
                let (memory, scratch, source, record, chunk) =
                    structured_settings_context(settings, &control, hpo)?;
                let budget = Budget::new(memory);
                let (core, retirement) = if corrected {
                    (
                        StructuredBridgeCore::Corrected(if let Some(provider) = &provider {
                            structured_session::CardInstanceNativeSession::open4(
                                plan,
                                provider.clone(),
                                budget.clone(),
                                control.clone(),
                                source,
                                record,
                                scratch,
                                chunk,
                            )?
                        } else {
                            structured_session::CardInstanceNativeSession::open(
                                plan,
                                budget.clone(),
                                control.clone(),
                                source,
                                record,
                                scratch,
                                chunk,
                            )?
                        }),
                        None,
                    )
                } else {
                    let retirement =
                        retirement::Retirement::start(&budget, retire_structured_session)?;
                    let core = structured_session::CohortSession::open(
                        plan,
                        budget.clone(),
                        control.clone(),
                        source,
                        record,
                        scratch,
                        chunk,
                    )?;
                    (StructuredBridgeCore::Engineering(core), Some(retirement))
                };
                Ok((core, retirement, budget))
            })
            .map_err(python_error)?;
        let caller = budget.reserve(0).map_err(python_error)?;
        Ok(Self {
            provider,
            final_provider: Mutex::new(None),
            core: Mutex::new(Some(core)),
            retirement: Mutex::new(retirement),
            budget,
            control,
            python: Mutex::new(control_memory::ControlMemory::default()),
            input_queries: Mutex::new(std::collections::BTreeMap::new()),
            caller: Mutex::new(caller),
            payload: Mutex::new(None),
        })
    }
    fn control_summary<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.control.check()?;
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let core = self.core.lock().map_err(|_| Error::Closed)?;
                let core = core.as_ref().ok_or(Error::Closed)?;
                let bytes = core.summary()?;
                let auth = core.authorization_max_bytes()?;
                self.python.lock().map_err(|_| Error::Closed)?.summary(
                    &self.budget,
                    bytes.len(),
                    auth,
                )?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn original_config_inputs<'py>(
        &self,
        py: Python<'py>,
        parent_index: usize,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let core = self.core.lock().map_err(|_| Error::Closed)?;
                let core = core.as_ref().ok_or(Error::Closed)?;
                let mut queries = self.input_queries.lock().map_err(|_| Error::Closed)?;
                if queries.contains_key(&parent_index) {
                    return Err(Error::contract(
                        "original input query already delivered for parent",
                    ));
                }
                let bytes = core.original_config_inputs(parent_index)?;
                let lease = self
                    .budget
                    .reserve(control_memory::reservation_bytes(bytes.len(), 512)?)?;
                queries.insert(parent_index, lease);
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn authorize(
        &self,
        py: Python<'_>,
        encoded: &[u8],
        retained_python_bytes: usize,
    ) -> PyResult<()> {
        let corrected = self
            .core
            .lock()
            .map_err(|_| python_error(Error::Closed))?
            .as_ref()
            .ok_or_else(|| python_error(Error::Closed))?
            .corrected();
        let owned = if corrected {
            Some(
                self.mutating(structured_session::CardInstanceOperation::Authorize, || {
                    self.python
                        .lock()
                        .map_err(|_| Error::Closed)?
                        .authorization(encoded.len())?;
                    Self::owned_wire(&self.budget, encoded)
                })
                .map_err(python_error)?,
            )
        } else {
            None
        };
        let encoded = owned.as_deref().unwrap_or(encoded);
        py.detach(|| {
            self.mutating(structured_session::CardInstanceOperation::Authorize, || {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                self.python
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .authorization(encoded.len())?;
                self.core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .authorize(encoded)
            })
        })
        .map_err(python_error)
    }
    fn authorize_hpo<'py>(
        &self,
        py: Python<'py>,
        encoded: &[u8],
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let owned = self
            .mutating(structured_session::CardInstanceOperation::Authorize, || {
                self.python
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .authorization(encoded.len())?;
                Self::owned_wire(&self.budget, encoded)
            })
            .map_err(python_error)?;
        let bytes = py
            .detach(|| {
                self.mutating(structured_session::CardInstanceOperation::Authorize, || {
                    self.caller
                        .lock()
                        .map_err(|_| Error::Closed)?
                        .resize(retained_python_bytes)?;
                    let bytes = self
                        .core
                        .lock()
                        .map_err(|_| Error::Closed)?
                        .as_mut()
                        .ok_or(Error::Closed)?
                        .authorize_hpo(&owned)?;
                    self.python
                        .lock()
                        .map_err(|_| Error::Closed)?
                        .hpo_final_summary(bytes.len())?;
                    Ok(bytes)
                })
            })
            .map_err(python_error)?;
        if let Err(error) = self.control.check() {
            if let Ok(mut core) = self.core.lock() {
                if let Some(core) = core.as_mut() {
                    core.fail_export();
                }
            }
            return Err(python_error(error));
        }
        Ok(PyBytes::new(py, &bytes))
    }
    fn export_selected<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
        recorded: Option<usize>,
    ) -> PyResult<(Py<PyBytes>, Py<PyTuple>, Py<PyTuple>, Py<PyTuple>)> {
        if recorded.is_some() {
            self.release_final_consumer(true).map_err(python_error)?;
        }
        let (metadata, banks, card_guard) = py
            .detach(|| {
                self.mutating(
                    match recorded {
                        Some(index) => structured_session::CardInstanceOperation::Recorded(index),
                        None => structured_session::CardInstanceOperation::Export,
                    },
                    || {
                        self.caller
                            .lock()
                            .map_err(|_| Error::Closed)?
                            .resize(retained_python_bytes)?;
                        if self
                            .core
                            .lock()
                            .map_err(|_| Error::Closed)?
                            .as_ref()
                            .ok_or(Error::Closed)?
                            .hpo()
                        {
                            self.python
                                .lock()
                                .map_err(|_| Error::Closed)?
                                .begin_data()?;
                        }
                        self.core
                            .lock()
                            .map_err(|_| Error::Closed)?
                            .as_mut()
                            .ok_or(Error::Closed)?
                            .export_selected(recorded)
                    },
                )
            })
            .map_err(python_error)?;
        let result = (|| {
            let (packet, consumer) =
                publish_raw_banks(py, &self.budget, &metadata, banks, card_guard)?;
            *self
                .payload
                .lock()
                .map_err(|_| python_error(Error::Closed))? = Some(consumer);
            Ok(packet)
        })();
        if result.is_err() {
            if let Ok(mut core) = self.core.lock() {
                if let Some(core) = core.as_mut() {
                    core.fail_export();
                }
            }
        }
        result
    }
    fn retire_payload(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        py.detach(|| {
            let corrected = self
                .core
                .lock()
                .map_err(|_| Error::Closed)?
                .as_ref()
                .ok_or(Error::Closed)?
                .corrected();
            if corrected {
                let payload = self.payload.lock().map_err(|_| Error::Closed)?;
                let consumer = payload
                    .as_ref()
                    .ok_or_else(|| Error::contract("no exported payload ownership"))?;
                if Arc::strong_count(consumer) != 1 {
                    return Err(Error::contract(
                        "raw payload views must retire before decoder reconstruction",
                    ));
                }
            }
            self.mutating(structured_session::CardInstanceOperation::Retire, || {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let mut payload = self.payload.lock().map_err(|_| Error::Closed)?;
                let consumer = payload
                    .as_ref()
                    .ok_or_else(|| Error::contract("no exported payload ownership"))?;
                if Arc::strong_count(consumer) != 1 {
                    return Err(Error::contract(
                        "raw payload views must retire before decoder reconstruction",
                    ));
                }
                let corrected = self
                    .core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_ref()
                    .ok_or(Error::Closed)?
                    .corrected();
                if corrected {
                    payload.take();
                }
                self.core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .retire_payload()?;
                payload.take();
                Ok(())
            })
        })
        .map_err(python_error)
    }
    fn begin_update(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        py.detach(|| {
            self.mutating(structured_session::CardInstanceOperation::Begin, || {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                self.core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .begin_update()
            })
        })
        .map_err(python_error)
    }
    fn recheck(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        self.release_final_consumer(false).map_err(python_error)?;
        py.detach(|| {
            let mut core = self.core.lock().map_err(|_| Error::Closed)?;
            let core = core.as_mut().ok_or(Error::Closed)?;
            if let StructuredBridgeCore::Corrected(v) = core {
                if v.final_loan_open() {
                    v.final_recheck_preconditions()?;
                    let result = (|| {
                        self.caller
                            .lock()
                            .map_err(|_| Error::Closed)?
                            .resize(retained_python_bytes)?;
                        v.recheck_final_validation()
                    })();
                    if let Err(e) = &result {
                        v.fail_final_validation(e.clone());
                    }
                    return result;
                }
            }
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained_python_bytes)?;
            core.recheck()
        })
        .map_err(python_error)
    }
    fn finalize<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let bytes = self
                    .core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .finalize()?;
                self.python
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .proof(bytes.len(), true)?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn bind_numeric(&self, py: Python<'_>, encoded: &[u8], retained: usize) -> PyResult<()> {
        let encoded = self
            .mutating(structured_session::CardInstanceOperation::Bind, || {
                if encoded.len() > 64 * 1024 * 1024 {
                    return Err(Error::resource("numeric binding wire cap"));
                }
                Self::owned_wire(&self.budget, encoded)
            })
            .map_err(python_error)?;
        py.detach(|| {
            self.mutating(structured_session::CardInstanceOperation::Bind, || {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained)?;
                let mut core = self.core.lock().map_err(|_| Error::Closed)?;
                match core.as_mut().ok_or(Error::Closed)? {
                    StructuredBridgeCore::Corrected(v) => v.bind_numeric(&encoded),
                    _ => Err(Error::contract(
                        "numeric4 binding requires corrected session",
                    )),
                }
            })
        })
        .map_err(python_error)
    }
    fn finish_corrected<'py>(
        &self,
        py: Python<'py>,
        retained: usize,
        prepare: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.mutating(
                    if prepare {
                        structured_session::CardInstanceOperation::Prepare
                    } else {
                        structured_session::CardInstanceOperation::FinalizeRead
                    },
                    || {
                        self.caller
                            .lock()
                            .map_err(|_| Error::Closed)?
                            .resize(retained)?;
                        let mut core = self.core.lock().map_err(|_| Error::Closed)?;
                        let bytes = match core.as_mut().ok_or(Error::Closed)? {
                            StructuredBridgeCore::Corrected(v) => {
                                if prepare {
                                    v.prepare_commit()?;
                                } else {
                                    v.finalize_read()?;
                                }
                                v.take_receipt()?
                            }
                            _ => {
                                return Err(Error::contract(
                                    "corrected finalization requires corrected session",
                                ));
                            }
                        };
                        self.python
                            .lock()
                            .map_err(|_| Error::Closed)?
                            .proof(bytes.len(), true)?;
                        Ok(bytes)
                    },
                )
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn used_bytes(&self) -> usize {
        self.budget.used()
    }
    fn peak_bytes(&self) -> usize {
        self.budget.peak()
    }
    fn control_reserved_bytes(&self) -> usize {
        self.python.lock().map(|v| v.reserved()).unwrap_or(0)
            + self
                .input_queries
                .lock()
                .map(|v| v.values().map(|lease| lease.bytes()).sum::<usize>())
                .unwrap_or(0)
    }
    fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            control: self.control.clone(),
            stats: Arc::new(telemetry::Telemetry::default()),
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.release_final_consumer(true).map_err(python_error)?;
        py.detach(|| {
            let mut core = self.core.lock().map_err(|_| Error::Closed)?;
            let Some(owned) = core.as_mut() else {
                return Ok(());
            };
            let result = owned.close();
            if !owned.is_closed() {
                return result;
            }
            let owned = core.take().ok_or(Error::Closed)?;
            let retired = match owned {
                StructuredBridgeCore::Engineering(owned) => self
                    .retirement
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .take()
                    .ok_or(Error::Closed)?
                    .retire(Some(owned)),
                StructuredBridgeCore::Corrected(_) => Ok(()),
            };
            self.payload.lock().map_err(|_| Error::Closed)?.take();
            self.python.lock().map_err(|_| Error::Closed)?.release();
            self.input_queries
                .lock()
                .map_err(|_| Error::Closed)?
                .clear();
            self.caller.lock().map_err(|_| Error::Closed)?.resize(0)?;
            result.and(retired)
        })
        .map_err(python_error)
    }
}
impl Drop for StructuredSessionBridge {
    fn drop(&mut self) {
        if let Ok(core) = self.core.get_mut() {
            if let Some(owned) = core.take() {
                match owned {
                    StructuredBridgeCore::Engineering(owned) => {
                        if let Ok(retirement) = self.retirement.get_mut() {
                            if let Some(retirement) = retirement.take() {
                                let _ = retirement.retire(Some(owned));
                            }
                        }
                    }
                    StructuredBridgeCore::Corrected(mut owned) => owned.retire_on_drop(),
                }
            }
        }
    }
}

#[pyclass]
struct StructuredNativeSession {
    bridge: StructuredSessionBridge,
}
#[pyclass]
struct CardInstanceNativeSession {
    bridge: StructuredSessionBridge,
}
macro_rules! session_methods {
    ($class:ident,$corrected:expr,{$($extra:tt)*})=>{
        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature=(plan,settings,deadline,cancel_handle=None))]
            fn new(py:Python<'_>,plan:&[u8],settings:&[u8],deadline:f64,cancel_handle:Option<PyRef<'_,CancelHandle>>)->PyResult<Self>{
                Ok(Self{bridge:StructuredSessionBridge::new(py,$corrected,plan,settings,deadline,cancel_handle)?})
            }
            #[pyo3(signature=(retained_python_bytes=0))]
            fn control_summary<'py>(&self,py:Python<'py>,retained_python_bytes:usize)->PyResult<Bound<'py,PyBytes>>{self.bridge.callback_guard()?;self.bridge.control_summary(py,retained_python_bytes)}
            #[pyo3(signature=(parent_index,retained_python_bytes=0))]
            fn original_config_inputs<'py>(&self,py:Python<'py>,parent_index:usize,retained_python_bytes:usize)->PyResult<Bound<'py,PyBytes>>{self.bridge.callback_guard()?;self.bridge.original_config_inputs(py,parent_index,retained_python_bytes)}
            #[pyo3(signature=(encoded,retained_python_bytes=0))]
            fn authorize(&self,py:Python<'_>,encoded:&[u8],retained_python_bytes:usize)->PyResult<()>{self.bridge.callback_guard()?;self.bridge.authorize(py,encoded,retained_python_bytes)}
            #[pyo3(signature=(retained_python_bytes=0))]
            fn export_cohort<'py>(&self,py:Python<'py>,retained_python_bytes:usize)->PyResult<(Py<PyBytes>,Py<PyTuple>,Py<PyTuple>,Py<PyTuple>)>{self.bridge.callback_guard()?;self.bridge.export_selected(py,retained_python_bytes,None)}
            #[pyo3(signature=(retained_python_bytes=0))]
            fn retire_payload(&self,py:Python<'_>,retained_python_bytes:usize)->PyResult<()>{self.bridge.callback_guard()?;self.bridge.retire_payload(py,retained_python_bytes)}
            #[pyo3(signature=(retained_python_bytes=0))]
            fn begin_update(&self,py:Python<'_>,retained_python_bytes:usize)->PyResult<()>{self.bridge.callback_guard()?;self.bridge.begin_update(py,retained_python_bytes)}
            #[pyo3(signature=(retained_python_bytes=0))]
            fn recheck(&self,py:Python<'_>,retained_python_bytes:usize)->PyResult<()>{self.bridge.callback_guard()?;self.bridge.recheck(py,retained_python_bytes)}
            fn used_bytes(&self)->PyResult<usize>{self.bridge.callback_guard()?;Ok(self.bridge.used_bytes())}
            fn peak_bytes(&self)->PyResult<usize>{self.bridge.callback_guard()?;Ok(self.bridge.peak_bytes())}
            fn control_reserved_bytes(&self)->PyResult<usize>{self.bridge.callback_guard()?;Ok(self.bridge.control_reserved_bytes())}
            fn cancel_handle(&self)->CancelHandle{self.bridge.cancel_handle()}
            fn close(&self,py:Python<'_>)->PyResult<()>{self.bridge.callback_guard()?;self.bridge.close(py)}
            $($extra)*
        }
    }
}
session_methods!(StructuredNativeSession, false, {
    #[pyo3(signature=(retained_python_bytes=0))]
    fn finalize<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.bridge.finalize(py, retained_python_bytes)
    }
});
session_methods!(CardInstanceNativeSession, true, {
    fn numeric_used_bytes(&self) -> Option<usize> {
        self.bridge.budget.numeric_used()
    }
    #[pyo3(signature=(encoded,retained_python_bytes=0))]
    fn authorize_hpo<'py>(
        &self,
        py: Python<'py>,
        encoded: &[u8],
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.bridge.callback_guard()?;
        self.bridge
            .authorize_hpo(py, encoded, retained_python_bytes)
    }
    #[staticmethod]
    #[pyo3(signature=(plan,provider,settings,deadline,cancel_handle=None))]
    fn from_artifacts(
        py: Python<'_>,
        plan: &[u8],
        provider: Py<PyAny>,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        let provider = logical_artifacts_bridge::PythonArtifactProvider::new(provider);
        StructuredSessionBridge::new_context(
            py,
            true,
            plan,
            settings,
            deadline,
            cancel_handle,
            Some(provider.clone()),
        )
        .map(|bridge| Self { bridge })
        .map_err(|error| provider.retain_on_error(py, error))
    }

    #[pyo3(signature=(request,retained_python_bytes=0,artifact_provider=None))]
    fn inspect_final_validation<'py>(
        &self,
        py: Python<'py>,
        request: &[u8],
        retained_python_bytes: usize,
        artifact_provider: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.bridge.callback_guard()?;
        // Check pure mode/phase before any caller/wire allocation or poison.
        {
            let core = self
                .bridge
                .core
                .lock()
                .map_err(|_| python_error(Error::Closed))?;
            match core.as_ref().ok_or_else(|| python_error(Error::Closed))? {
                StructuredBridgeCore::Corrected(v) => {
                    v.final_parent_preconditions().map_err(python_error)?
                }
                _ => return Err(python_error(Error::contract("corrected parent required"))),
            }
        }
        // Reserve/bound/copy the Python borrow while the GIL is held. Only
        // this owned wire crosses detach, matching the shared corrected ABI.
        let raw = (|| -> error::Result<_> {
            let mut core = self.bridge.core.lock().map_err(|_| Error::Closed)?;
            let StructuredBridgeCore::Corrected(v) = core.as_mut().ok_or(Error::Closed)? else {
                return Err(Error::Closed);
            };
            let result = (|| {
                v.final_request_bound(request.len())?;
                self.bridge
                    .caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                StructuredSessionBridge::owned_wire(&self.bridge.budget, request)
            })();
            if let Err(e) = &result {
                v.fail_final_validation(e.clone());
            }
            result
        })()
        .map_err(python_error)?;
        let provider = artifact_provider.map(logical_artifacts_bridge::PythonArtifactProvider::new);
        if let Some(provider) = &provider {
            provider.check_reentry().map_err(python_error)?;
            let mut slot = self
                .bridge
                .final_provider
                .lock()
                .map_err(|_| python_error(Error::Closed))?;
            if slot.is_some() {
                return Err(python_error(Error::contract(
                    "final provider has one lifetime",
                )));
            }
            *slot = Some(provider.clone());
        }
        let result = py
            .detach(|| {
                let mut core = self.bridge.core.lock().map_err(|_| Error::Closed)?;
                let StructuredBridgeCore::Corrected(v) = core.as_mut().ok_or(Error::Closed)? else {
                    return Err(Error::Closed);
                };
                v.inspect_final_validation_artifacts(
                    &raw,
                    provider
                        .clone()
                        .map(|p| p as Arc<dyn logical_artifacts::LogicalArtifactProvider>),
                )
            })
            .map_err(|error| {
                let error = python_error(error);
                match &provider {
                    Some(p) => p.retain_on_error(py, error),
                    None => error,
                }
            })?;
        Ok(PyBytes::new(py, &result))
    }
    #[pyo3(signature=(episode_index,retained_python_bytes=0))]
    fn export_final_validation<'py>(
        &self,
        py: Python<'py>,
        episode_index: usize,
        retained_python_bytes: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyTuple>, Py<PyTuple>, Py<PyTuple>)> {
        self.bridge.callback_guard()?;
        self.bridge
            .release_final_consumer(true)
            .map_err(python_error)?;
        let payload = py
            .detach(|| {
                let mut core = self.bridge.core.lock().map_err(|_| Error::Closed)?;
                let StructuredBridgeCore::Corrected(v) = core.as_mut().ok_or(Error::Closed)? else {
                    return Err(Error::Closed);
                };
                v.final_export_preconditions(episode_index)?;
                let result = (|| {
                    self.bridge
                        .caller
                        .lock()
                        .map_err(|_| Error::Closed)?
                        .resize(retained_python_bytes)?;
                    v.export_final_validation(episode_index)
                })();
                if let Err(e) = &result {
                    v.fail_final_validation(e.clone());
                }
                result
            })
            .map_err(python_error)?;
        let (metadata, banks, guard, lease) = payload.into_parts();
        let result = publish_raw_banks(
            py,
            &self.bridge.budget,
            &metadata,
            banks,
            Some((guard, lease)),
        )
        .and_then(|(packet, consumer)| {
            *self
                .bridge
                .payload
                .lock()
                .map_err(|_| python_error(Error::Closed))? = Some(consumer);
            Ok(packet)
        });
        if result.is_err() {
            if let Ok(mut core) = self.bridge.core.lock() {
                if let Some(StructuredBridgeCore::Corrected(v)) = core.as_mut() {
                    v.fail_final_validation(Error::resource(
                        "final64 Python payload publication failed",
                    ));
                }
            }
        }
        result
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn recheck_final_validation(
        &self,
        py: Python<'_>,
        retained_python_bytes: usize,
    ) -> PyResult<()> {
        self.bridge.callback_guard()?;
        self.bridge
            .release_final_consumer(false)
            .map_err(python_error)?;
        py.detach(|| {
            let mut core = self.bridge.core.lock().map_err(|_| Error::Closed)?;
            let StructuredBridgeCore::Corrected(v) = core.as_mut().ok_or(Error::Closed)? else {
                return Err(Error::Closed);
            };
            v.final_recheck_preconditions()?;
            let result = (|| {
                self.bridge
                    .caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                v.recheck_final_validation()
            })();
            if let Err(e) = &result {
                v.fail_final_validation(e.clone());
            }
            result
        })
        .map_err(python_error)
    }
    fn close_final_validation(&self, py: Python<'_>) -> PyResult<()> {
        self.bridge.callback_guard()?;
        self.bridge
            .release_final_consumer(true)
            .map_err(python_error)?;
        py.detach(|| {
            let mut core = self.bridge.core.lock().map_err(|_| Error::Closed)?;
            let StructuredBridgeCore::Corrected(v) = core.as_mut().ok_or(Error::Closed)? else {
                return Err(Error::Closed);
            };
            v.close_final_validation()
        })
        .map_err(python_error)?;
        self.bridge
            .final_provider
            .lock()
            .map_err(|_| python_error(Error::Closed))?
            .take();
        Ok(())
    }

    #[pyo3(signature=(collection_index,episode_index,retained_python_bytes=0))]
    fn export_recorded_episode_payload<'py>(
        &self,
        py: Python<'py>,
        collection_index: usize,
        episode_index: usize,
        retained_python_bytes: usize,
    ) -> PyResult<RawTokenPayload> {
        self.bridge.callback_guard()?;
        self.bridge
            .release_final_consumer(true)
            .map_err(python_error)?;
        let payload = py
            .detach(|| {
                let mut core = self.bridge.core.lock().map_err(|_| Error::Closed)?;
                let StructuredBridgeCore::Corrected(value) = core.as_mut().ok_or(Error::Closed)?
                else {
                    return Err(Error::Closed);
                };
                value.recorded_episode_preconditions(collection_index, episode_index)?;
                let result = (|| {
                    self.bridge
                        .caller
                        .lock()
                        .map_err(|_| Error::Closed)?
                        .resize(retained_python_bytes)?;
                    value.export_recorded_episode_payload(collection_index, episode_index)
                })();
                if result.is_err() {
                    value.fail_admitted();
                }
                result
            })
            .map_err(python_error)?;
        let (metadata, banks, guard, lease) = payload.into_parts();
        let result = publish_raw_banks(
            py,
            &self.bridge.budget,
            &metadata,
            banks,
            Some((guard, lease)),
        )
        .and_then(|(packet, consumer)| {
            *self
                .bridge
                .payload
                .lock()
                .map_err(|_| python_error(Error::Closed))? = Some(consumer);
            Ok(packet)
        });
        if result.is_err() {
            if let Ok(mut core) = self.bridge.core.lock() {
                if let Some(StructuredBridgeCore::Corrected(v)) = core.as_mut() {
                    v.fail_admitted();
                }
            }
        }
        result
    }
    #[pyo3(signature=(collection_index,retained_python_bytes=0))]
    fn export_recorded_cohort<'py>(
        &self,
        py: Python<'py>,
        collection_index: usize,
        retained_python_bytes: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyTuple>, Py<PyTuple>, Py<PyTuple>)> {
        self.bridge.callback_guard()?;
        self.bridge
            .export_selected(py, retained_python_bytes, Some(collection_index))
    }
    #[pyo3(signature=(encoded,retained_python_bytes=0))]
    fn bind_numeric(
        &self,
        py: Python<'_>,
        encoded: &[u8],
        retained_python_bytes: usize,
    ) -> PyResult<()> {
        self.bridge.callback_guard()?;
        self.bridge.bind_numeric(py, encoded, retained_python_bytes)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn prepare_commit<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.bridge.callback_guard()?;
        self.bridge
            .finish_corrected(py, retained_python_bytes, true)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn finalize_read<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.bridge.callback_guard()?;
        self.bridge
            .finish_corrected(py, retained_python_bytes, false)
    }
});

/// Structural component tokens. Never a Source owner or fitting permission.
#[pyclass]
struct CardInstanceFixtureTokens {
    owner: Mutex<card_instance_fixture::FixtureTokens>,
    budget: Arc<Budget>,
    control: Control,
    caller: Mutex<budget::Lease>,
    payload: Mutex<Option<Arc<ExportLease>>>,
}
impl CardInstanceFixtureTokens {
    fn from_parts(
        owner: card_instance_fixture::FixtureTokens,
        budget: Arc<Budget>,
        control: Control,
        caller: budget::Lease,
    ) -> Self {
        Self {
            owner: Mutex::new(owner),
            budget,
            control,
            caller: Mutex::new(caller),
            payload: Mutex::new(None),
        }
    }
}
#[pymethods]
impl CardInstanceFixtureTokens {
    #[new]
    #[pyo3(signature=(envelope,columns,settings,deadline,cancel_handle=None,*,aggregate_window=128,retained_python_bytes=0))]
    fn new(
        py: Python<'_>,
        envelope: &Bound<'_, PyAny>,
        columns: &Bound<'_, PyAny>,
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
        aggregate_window: usize,
        retained_python_bytes: usize,
    ) -> PyResult<Self> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        if settings.len() > 65536 {
            return Err(python_error(Error::resource("fixture settings wire cap")));
        }
        let envelope = envelope.cast_exact::<PyBytes>()?;
        let columns = columns.cast_exact::<PyTuple>()?;
        if columns.len() > 1024 {
            return Err(python_error(Error::resource("fixture decision capacity")));
        }
        let control = match cancel_handle {
            Some(c) => c.control.at_deadline(deadline),
            None => Control::new(deadline),
        }
        .map_err(python_error)?;
        let (memory, scratch, source, record, chunk) =
            structured_settings(settings, &control).map_err(python_error)?;
        let budget = Budget::new(memory);
        let borrowed = budget
            .reserve(65536 + 128 * columns.len())
            .map_err(python_error)?;
        let owners = columns
            .iter()
            .map(|value| value.cast_exact::<PyBytes>().cloned().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let columns = owners
            .iter()
            .map(|value| value.as_bytes())
            .collect::<Vec<_>>();
        let envelope = envelope.as_bytes();
        let caller = budget
            .reserve(retained_python_bytes)
            .map_err(python_error)?;
        let mut owner =
            card_instance_fixture::FixtureTokens::empty(budget.clone(), control.clone())
                .map_err(python_error)?;
        let result = py.detach(|| {
            owner.populate(
                envelope,
                &columns,
                source,
                record,
                scratch,
                chunk,
                aggregate_window,
            )
        });
        drop(borrowed);
        if let Err(primary) = result {
            let cleanup = py.detach(|| owner.close());
            if let Err(cleanup) = cleanup {
                let error = python_error(Error::Cleanup {
                    primary: Box::new(primary),
                    cleanup: format!("{cleanup:?}"),
                });
                // No partial constructor owner disappears behind an exception.
                // The caller can retry close; discarding it quarantines an
                // unconfirmed spool instead of deleting foreign evidence.
                match Py::new(py, Self::from_parts(owner, budget, control, caller)) {
                    Ok(cleanup_owner) => {
                        // Attaching diagnostic ownership cannot replace the
                        // initiating cancellation/validation error. If Python
                        // cannot attach it, Drop quarantines the bounded spool.
                        let _ = error
                            .value(py)
                            .setattr("_native_fixture_cleanup_owner", cleanup_owner);
                    }
                    Err(_) => {
                        let _ = error.value(py).call_method1("add_note", ("fixture cleanup owner allocation failed; unconfirmed spool quarantined",));
                    }
                }
                return Err(error);
            }
            return Err(python_error(primary));
        }
        Ok(Self::from_parts(owner, budget, control, caller))
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn export_tokens(
        &self,
        py: Python<'_>,
        retained_python_bytes: usize,
    ) -> PyResult<RawTokenPayload> {
        let (metadata, banks) = py
            .detach(|| {
                let mut owner = self.owner.lock().map_err(|_| Error::Closed)?;
                owner.export_preconditions()?;
                let quota = self
                    .caller
                    .lock()
                    .map_err(|_| Error::Closed)
                    .and_then(|mut lease| lease.resize(retained_python_bytes));
                if let Err(error) = quota {
                    owner.fail();
                    return Err(error);
                }
                owner.export()
            })
            .map_err(python_error)?;
        let result = (|| {
            self.control.check().map_err(python_error)?;
            let (packet, consumer) = publish_raw_banks(py, &self.budget, &metadata, banks, None)?;
            self.control.check().map_err(python_error)?;
            *self
                .payload
                .lock()
                .map_err(|_| python_error(Error::Closed))? = Some(consumer);
            Ok(packet)
        })();
        if result.is_err() {
            if let Ok(mut owner) = self.owner.lock() {
                owner.fail();
            }
        }
        result
    }
    fn used_bytes(&self) -> usize {
        self.budget.used()
    }
    fn peak_bytes(&self) -> usize {
        self.budget.peak()
    }
    fn control_reserved_bytes(&self) -> usize {
        self.payload
            .lock()
            .map(|value| value.as_ref().map(|v| v._metadata.bytes()).unwrap_or(0))
            .unwrap_or(0)
    }
    fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            control: self.control.clone(),
            stats: Arc::new(telemetry::Telemetry::default()),
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.owner.lock().map_err(|_| Error::Closed)?.close()?;
            self.payload.lock().map_err(|_| Error::Closed)?.take();
            self.caller.lock().map_err(|_| Error::Closed)?.resize(0)
        })
        .map_err(python_error)
    }
}

/// One frozen TreeDataset2 worklist; raw selected inputs only, never fitting authority.
#[pyclass]
struct TreeDatasetSession {
    core: Mutex<Option<structured_tree_dataset::TreeDataset>>,
    retirement: Mutex<Option<retirement::Retirement<structured_tree_dataset::TreeDataset>>>,
    budget: Arc<Budget>,
    control: Control,
    python: Mutex<control_memory::ControlMemory>,
    current_python: Mutex<control_memory::ControlMemory>,
    caller: Mutex<budget::Lease>,
    payload: Mutex<Option<Arc<ExportLease>>>,
}
fn retire_tree_dataset(core: structured_tree_dataset::TreeDataset) -> error::Result<()> {
    drop(core);
    Ok(())
}
#[pymethods]
impl TreeDatasetSession {
    #[new]
    #[pyo3(signature=(plan,settings,deadline,cancel_handle=None))]
    fn new(
        py: Python<'_>,
        plan: &[u8],
        settings: &[u8],
        deadline: f64,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        if settings.len() > 65536 || plan.len() > 65536 {
            return Err(python_error(Error::resource("structured request wire cap")));
        }
        let control = match cancel_handle {
            Some(c) => c.control.at_deadline(deadline),
            None => Control::new(deadline),
        }
        .map_err(python_error)?;
        let (core, retirement, budget) = py
            .detach(|| {
                control.check()?;
                let initial = Budget::new(1048576);
                let settings = json::parse(settings, &initial, &control)?;
                let v = &settings.root;
                v.closed(
                    &[
                        "memory_bytes",
                        "scratch_bytes",
                        "max_source_file_bytes",
                        "max_record_bytes",
                        "chunk_target_bytes",
                    ],
                    &[],
                )?;
                let memory = v.require("memory_bytes")?.integer(1, 12i128 << 30)? as usize;
                let scratch = v.require("scratch_bytes")?.integer(1, 12i128 << 30)? as usize;
                let source = v
                    .require("max_source_file_bytes")?
                    .integer(1, 64i128 << 20)? as usize;
                let record = v.require("max_record_bytes")?.integer(1, source as i128)? as usize;
                let chunk = v.require("chunk_target_bytes")?.integer(1, 64i128 << 20)? as usize;
                let budget = Budget::new(memory);
                let retirement = retirement::Retirement::start(&budget, retire_tree_dataset)?;
                let core = structured_tree_dataset::TreeDataset::open(
                    plan,
                    budget.clone(),
                    control.clone(),
                    source,
                    record,
                    scratch,
                    chunk,
                )?;
                Ok((core, retirement, budget))
            })
            .map_err(python_error)?;
        let caller = budget.reserve(0).map_err(python_error)?;
        Ok(Self {
            core: Mutex::new(Some(core)),
            retirement: Mutex::new(Some(retirement)),
            budget,
            control,
            python: Mutex::new(control_memory::ControlMemory::default()),
            current_python: Mutex::new(control_memory::ControlMemory::default()),
            caller: Mutex::new(caller),
            payload: Mutex::new(None),
        })
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn control_summary<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.control.check()?;
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let core = self.core.lock().map_err(|_| Error::Closed)?;
                let core = core.as_ref().ok_or(Error::Closed)?;
                let bytes = core.control_summary()?;
                let auth = core.control_authorization_limit()?;
                self.python.lock().map_err(|_| Error::Closed)?.summary(
                    &self.budget,
                    bytes.len(),
                    auth,
                )?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(encoded,retained_python_bytes=0))]
    fn authorize_controls(
        &self,
        py: Python<'_>,
        encoded: &[u8],
        retained_python_bytes: usize,
    ) -> PyResult<()> {
        py.detach(|| {
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained_python_bytes)?;
            self.python
                .lock()
                .map_err(|_| Error::Closed)?
                .authorization(encoded.len())?;
            self.core
                .lock()
                .map_err(|_| Error::Closed)?
                .as_mut()
                .ok_or(Error::Closed)?
                .authorize_controls(encoded)
        })
        .map_err(python_error)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn export_current<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyTuple>, Py<PyTuple>)> {
        let (metadata, banks) = py
            .detach(|| {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                self.core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .export_current()
            })
            .map_err(python_error)?;
        let result = (|| {
            let structured_staging::Banks {
                bytes,
                integers,
                floats,
                _metadata,
            } = banks;
            let consumer = Arc::new(ExportLease {
                _consumer: self.budget.reserve(0).map_err(python_error)?,
                _metadata: self
                    .budget
                    .reserve(
                        control_memory::reservation_bytes(
                            metadata.len(),
                            1024 * (bytes.len() + integers.len()),
                        )
                        .map_err(python_error)?,
                    )
                    .map_err(python_error)?,
                _index: Some(_metadata),
                _card_guard: None,
                stats: Arc::new(telemetry::Telemetry::default()),
            });
            let bytes = bytes
                .into_iter()
                .map(|v| byte_array(py, v, &consumer))
                .collect::<PyResult<Vec<_>>>()?;
            let integers = integers
                .into_iter()
                .map(|v| integer_array(py, v, &consumer))
                .collect::<PyResult<Vec<_>>>()?;
            if !floats.is_empty() {
                return Err(python_error(Error::contract(
                    "TreeDataset raw inputs must have no float bank",
                )));
            }
            *self
                .payload
                .lock()
                .map_err(|_| python_error(Error::Closed))? = Some(consumer);
            Ok((
                PyBytes::new(py, &metadata).unbind(),
                PyTuple::new(py, bytes)?.unbind(),
                PyTuple::new(py, integers)?.unbind(),
            ))
        })();
        if result.is_err() {
            if let Ok(mut core) = self.core.lock() {
                if let Some(core) = core.as_mut() {
                    core.fail_export();
                }
            }
        }
        result
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn retire_current(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        py.detach(|| {
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained_python_bytes)?;
            let mut payload = self.payload.lock().map_err(|_| Error::Closed)?;
            let consumer = payload
                .as_ref()
                .ok_or_else(|| Error::contract("no exported payload ownership"))?;
            if Arc::strong_count(consumer) != 1 {
                return Err(Error::contract(
                    "raw payload views must retire before decoder reconstruction",
                ));
            }
            self.core
                .lock()
                .map_err(|_| Error::Closed)?
                .as_mut()
                .ok_or(Error::Closed)?
                .retire_current()?;
            payload.take();
            let mut current = self.current_python.lock().map_err(|_| Error::Closed)?;
            current.release();
            *current = control_memory::ControlMemory::default();
            Ok(())
        })
        .map_err(python_error)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn stage_next(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<bool> {
        py.detach(|| {
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained_python_bytes)?;
            self.core
                .lock()
                .map_err(|_| Error::Closed)?
                .as_mut()
                .ok_or(Error::Closed)?
                .stage_next()
        })
        .map_err(python_error)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn collection_summary<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let mut core = self.core.lock().map_err(|_| Error::Closed)?;
                let core = core.as_mut().ok_or(Error::Closed)?;
                let bytes = core.collection_summary()?;
                let auth = core.collection_authorization_limit()?;
                self.current_python
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .summary(&self.budget, bytes.len(), auth)?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(encoded,retained_python_bytes=0))]
    fn authorize_collection(
        &self,
        py: Python<'_>,
        encoded: &[u8],
        retained_python_bytes: usize,
    ) -> PyResult<()> {
        py.detach(|| {
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained_python_bytes)?;
            self.current_python
                .lock()
                .map_err(|_| Error::Closed)?
                .authorization(encoded.len())?;
            self.core
                .lock()
                .map_err(|_| Error::Closed)?
                .as_mut()
                .ok_or(Error::Closed)?
                .authorize_collection(encoded)
        })
        .map_err(python_error)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn verify_complete<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let bytes = self
                    .core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .verify_complete()?;
                self.python
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .proof(bytes.len(), true)?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn recheck(&self, py: Python<'_>, retained_python_bytes: usize) -> PyResult<()> {
        py.detach(|| {
            self.caller
                .lock()
                .map_err(|_| Error::Closed)?
                .resize(retained_python_bytes)?;
            self.core
                .lock()
                .map_err(|_| Error::Closed)?
                .as_mut()
                .ok_or(Error::Closed)?
                .recheck()
        })
        .map_err(python_error)
    }
    #[pyo3(signature=(retained_python_bytes=0))]
    fn finalize<'py>(
        &self,
        py: Python<'py>,
        retained_python_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.caller
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .resize(retained_python_bytes)?;
                let bytes = self
                    .core
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .as_mut()
                    .ok_or(Error::Closed)?
                    .finalize()?;
                self.python
                    .lock()
                    .map_err(|_| Error::Closed)?
                    .proof(bytes.len(), true)?;
                Ok(bytes)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn used_bytes(&self) -> usize {
        self.budget.used()
    }
    fn peak_bytes(&self) -> usize {
        self.budget.peak()
    }
    fn control_reserved_bytes(&self) -> usize {
        self.python.lock().map(|v| v.reserved()).unwrap_or(0)
            + self
                .current_python
                .lock()
                .map(|v| v.reserved())
                .unwrap_or(0)
    }
    fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            control: self.control.clone(),
            stats: Arc::new(telemetry::Telemetry::default()),
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut core = self.core.lock().map_err(|_| Error::Closed)?;
            let Some(owned) = core.as_mut() else {
                return Ok(());
            };
            let result = owned.close();
            if !owned.is_closed() {
                return result;
            }
            let owned = core.take().ok_or(Error::Closed)?;
            let retirement = self
                .retirement
                .lock()
                .map_err(|_| Error::Closed)?
                .take()
                .ok_or(Error::Closed)?;
            let retired = retirement.retire(Some(owned));
            self.payload.lock().map_err(|_| Error::Closed)?.take();
            self.python.lock().map_err(|_| Error::Closed)?.release();
            self.current_python
                .lock()
                .map_err(|_| Error::Closed)?
                .release();
            self.caller.lock().map_err(|_| Error::Closed)?.resize(0)?;
            result.and(retired)
        })
        .map_err(python_error)
    }
}
impl Drop for TreeDatasetSession {
    fn drop(&mut self) {
        if let Ok(core) = self.core.get_mut() {
            if let Some(owned) = core.take() {
                if let Ok(retirement) = self.retirement.get_mut() {
                    if let Some(retirement) = retirement.take() {
                        let _ = retirement.retire(Some(owned));
                    }
                }
            }
        }
    }
}

#[cfg(feature = "allocator-mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use crate::budget::{Budget, Control};
use crate::error::Error;
use ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::{PyAny, PyInt, PyList, PyTuple};
use std::io::{Seek, SeekFrom};
use std::sync::{Arc, Mutex};

#[pyclass(frozen)]
struct FloatOwner {
    data: Array2<f64>,
    _lease: crate::budget::Lease,
    _consumer: Option<Arc<ExportLease>>,
}
#[pymethods]
impl FloatOwner {
    #[getter]
    fn pointer(&self) -> usize {
        self.data.as_ptr() as usize
    }
    #[getter]
    fn capacity_bytes(&self) -> usize {
        self._lease.bytes()
    }
}

fn publish_float<'py>(
    py: Python<'py>,
    data: Array2<f64>,
    lease: crate::budget::Lease,
    stats: Option<&telemetry::Telemetry>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let owner = Bound::new(
        py,
        FloatOwner {
            data,
            _lease: lease,
            _consumer: None,
        },
    )?;
    // SAFETY: NumPy's actual base is the frozen owner containing the buffer and
    // quota lease. It exposes no reallocation/mutation API; NumPy keeps it alive.
    let array =
        unsafe { PyArray2::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    if let Some(stats) = stats {
        stats
            .export(
                owner.borrow().data.as_ptr() as usize,
                array.data() as usize,
                owner.borrow().data.len() * 8,
            )
            .map_err(python_error)?;
    }
    Ok(array)
}

#[pyclass(frozen)]
struct ScalarOwner {
    data: Array1<f64>,
    _lease: crate::budget::Lease,
    _consumer: Arc<ExportLease>,
}
#[pyclass(frozen)]
struct Float32Owner {
    data: Array1<f32>,
    _lease: crate::budget::Lease,
    _consumer: Arc<ExportLease>,
}
#[pyclass(frozen)]
struct IntegerOwner {
    data: Array1<i64>,
    _lease: crate::budget::Lease,
    _consumer: Arc<ExportLease>,
}
#[pyclass(frozen)]
struct ByteOwner {
    data: Array1<u8>,
    _lease: crate::budget::Lease,
    _consumer: Arc<ExportLease>,
}
#[pyclass(frozen)]
struct OffsetOwner {
    data: Array1<i32>,
    _lease: crate::budget::Lease,
    _consumer: Arc<ExportLease>,
}
struct ExportLease {
    _consumer: budget::Lease,
    _metadata: budget::Lease,
    _index: Option<budget::Lease>,
    _card_guard: Option<(Arc<()>, budget::Lease)>,
    stats: Arc<telemetry::Telemetry>,
}
type RawTokenPayload = (Py<PyBytes>, Py<PyTuple>, Py<PyTuple>, Py<PyTuple>);
fn publish_raw_banks(
    py: Python<'_>,
    budget: &Arc<Budget>,
    metadata: &[u8],
    banks: structured_staging::Banks,
    card_guard: Option<(Arc<()>, budget::Lease)>,
) -> PyResult<(RawTokenPayload, Arc<ExportLease>)> {
    let structured_staging::Banks {
        bytes,
        integers,
        floats,
        _metadata,
    } = banks;
    let consumer = Arc::new(ExportLease {
        _consumer: budget.reserve(0).map_err(python_error)?,
        _metadata: budget
            .reserve(
                control_memory::reservation_bytes(
                    metadata.len(),
                    1024 * (bytes.len() + integers.len() + floats.len()),
                )
                .map_err(python_error)?,
            )
            .map_err(python_error)?,
        _index: Some(_metadata),
        _card_guard: card_guard,
        stats: Arc::new(telemetry::Telemetry::default()),
    });
    let bytes = bytes
        .into_iter()
        .map(|v| byte_array(py, v, &consumer))
        .collect::<PyResult<Vec<_>>>()?;
    let integers = integers
        .into_iter()
        .map(|v| integer_array(py, v, &consumer))
        .collect::<PyResult<Vec<_>>>()?;
    let floats = floats
        .into_iter()
        .map(|v| float32_array(py, v, &consumer))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        (
            PyBytes::new(py, metadata).unbind(),
            PyTuple::new(py, bytes)?.unbind(),
            PyTuple::new(py, integers)?.unbind(),
            PyTuple::new(py, floats)?.unbind(),
        ),
        consumer,
    ))
}

fn float_array<'py>(
    py: Python<'py>,
    values: budget::BudgetVec<f64>,
    rows: usize,
    width: usize,
    consumer: &Arc<ExportLease>,
) -> PyResult<Py<PyArray2<f64>>> {
    let (values, lease) = values.into_parts();
    let data = Array2::from_shape_vec((rows, width), values)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("native array shape mismatch"))?;
    let owner = Bound::new(
        py,
        FloatOwner {
            data,
            _lease: lease,
            _consumer: Some(consumer.clone()),
        },
    )?;
    // SAFETY: immutable actual base owns buffer+capacity+consumer leases.
    let array =
        unsafe { PyArray2::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    consumer
        .stats
        .export(
            owner.borrow().data.as_ptr() as usize,
            array.data() as usize,
            owner.borrow().data.len() * 8,
        )
        .map_err(python_error)?;
    Ok(array.unbind())
}
fn scalar_array<'py>(
    py: Python<'py>,
    values: budget::BudgetVec<f64>,
    consumer: &Arc<ExportLease>,
) -> PyResult<Py<PyArray1<f64>>> {
    let (values, lease) = values.into_parts();
    let owner = Bound::new(
        py,
        ScalarOwner {
            data: Array1::from_vec(values),
            _lease: lease,
            _consumer: consumer.clone(),
        },
    )?;
    // SAFETY: immutable actual base owns buffer+capacity+consumer leases.
    let array =
        unsafe { PyArray1::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    consumer
        .stats
        .export(
            owner.borrow().data.as_ptr() as usize,
            array.data() as usize,
            owner.borrow().data.len() * 8,
        )
        .map_err(python_error)?;
    Ok(array.unbind())
}
fn float32_array<'py>(
    py: Python<'py>,
    values: budget::BudgetVec<f32>,
    consumer: &Arc<ExportLease>,
) -> PyResult<Py<PyArray1<f32>>> {
    let (values, lease) = values.into_parts();
    let owner = Bound::new(
        py,
        Float32Owner {
            data: Array1::from_vec(values),
            _lease: lease,
            _consumer: consumer.clone(),
        },
    )?;
    // SAFETY: the actual immutable base owns FP32 storage and both leases.
    let array =
        unsafe { PyArray1::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    consumer
        .stats
        .export(
            owner.borrow().data.as_ptr() as usize,
            array.data() as usize,
            owner.borrow().data.len() * 4,
        )
        .map_err(python_error)?;
    Ok(array.unbind())
}
fn integer_array<'py>(
    py: Python<'py>,
    values: budget::BudgetVec<i64>,
    consumer: &Arc<ExportLease>,
) -> PyResult<Py<PyArray1<i64>>> {
    let (values, lease) = values.into_parts();
    let owner = Bound::new(
        py,
        IntegerOwner {
            data: Array1::from_vec(values),
            _lease: lease,
            _consumer: consumer.clone(),
        },
    )?;
    // SAFETY: immutable actual base owns buffer+capacity+consumer leases.
    let array =
        unsafe { PyArray1::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    consumer
        .stats
        .export(
            owner.borrow().data.as_ptr() as usize,
            array.data() as usize,
            owner.borrow().data.len() * 8,
        )
        .map_err(python_error)?;
    Ok(array.unbind())
}
fn byte_array<'py>(
    py: Python<'py>,
    values: budget::BudgetVec<u8>,
    consumer: &Arc<ExportLease>,
) -> PyResult<Py<PyArray1<u8>>> {
    let (values, lease) = values.into_parts();
    let owner = Bound::new(
        py,
        ByteOwner {
            data: Array1::from_vec(values),
            _lease: lease,
            _consumer: consumer.clone(),
        },
    )?;
    // SAFETY: actual frozen base owns the bytes and both producer/consumer leases.
    let array =
        unsafe { PyArray1::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    consumer
        .stats
        .export(
            owner.borrow().data.as_ptr() as usize,
            array.data() as usize,
            owner.borrow().data.len(),
        )
        .map_err(python_error)?;
    Ok(array.unbind())
}
fn offset_array<'py>(
    py: Python<'py>,
    values: budget::BudgetVec<i32>,
    consumer: &Arc<ExportLease>,
) -> PyResult<Py<PyArray1<i32>>> {
    let (values, lease) = values.into_parts();
    let owner = Bound::new(
        py,
        OffsetOwner {
            data: Array1::from_vec(values),
            _lease: lease,
            _consumer: consumer.clone(),
        },
    )?;
    // SAFETY: actual frozen base owns the offsets and both producer/consumer leases.
    let array =
        unsafe { PyArray1::borrow_from_array(&owner.borrow().data, owner.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    consumer
        .stats
        .export(
            owner.borrow().data.as_ptr() as usize,
            array.data() as usize,
            owner.borrow().data.len() * 4,
        )
        .map_err(python_error)?;
    Ok(array.unbind())
}
fn utf8_column<'py>(
    py: Python<'py>,
    column: index::Utf8Column,
    consumer: &Arc<ExportLease>,
) -> PyResult<Bound<'py, PyAny>> {
    let data = byte_array(py, column.data, consumer)?
        .into_bound(py)
        .into_any();
    let offsets = offset_array(py, column.offsets, consumer)?
        .into_bound(py)
        .into_any();
    Ok(PyTuple::new(py, [data, offsets])?.into_any())
}
fn index_columns<'py>(
    py: Python<'py>,
    columns: Option<index::Columns>,
    consumer: &Arc<ExportLease>,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(columns) = columns else {
        return Ok(py.None().into_bound(py));
    };
    let [run, parent, batch, split, opponent, manifest, game, dataset] = columns.strings;
    Ok(PyTuple::new(
        py,
        [
            utf8_column(py, run, consumer)?,
            utf8_column(py, parent, consumer)?,
            utf8_column(py, batch, consumer)?,
            integer_array(py, columns.generation, consumer)?
                .into_bound(py)
                .into_any(),
            utf8_column(py, split, consumer)?,
            integer_array(py, columns.viewer, consumer)?
                .into_bound(py)
                .into_any(),
            utf8_column(py, opponent, consumer)?,
            utf8_column(py, manifest, consumer)?,
            utf8_column(py, game, consumer)?,
            integer_array(py, columns.decision, consumer)?
                .into_bound(py)
                .into_any(),
            scalar_array(py, columns.returns, consumer)?
                .into_bound(py)
                .into_any(),
            utf8_column(py, dataset, consumer)?,
        ],
    )?
    .into_any())
}

fn publish_batch(
    py: Python<'_>,
    batch: pipeline::Batch,
    online: bool,
    proof: Option<budget::BudgetVec<u8>>,
    stats: &Arc<telemetry::Telemetry>,
) -> PyResult<Py<PyTuple>> {
    let metadata_lease = batch
        .features
        .budget()
        .reserve(batch.metadata.len() + proof.as_ref().map_or(0, |p| p.len()) + 1024)
        .map_err(python_error)?;
    let consumer = Arc::new(ExportLease {
        _consumer: batch.consumer,
        _metadata: metadata_lease,
        _index: batch.index_consumer,
        _card_guard: None,
        stats: stats.clone(),
    });
    stats.output_copy(batch.metadata.len() + proof.as_ref().map_or(0, |proof| proof.len()));
    let decisions = batch.selected.len();
    let proof = match proof {
        Some(proof) => PyBytes::new(py, &proof).into_any(),
        None => py.None().into_bound(py),
    };
    let layout = if online { "all_candidates" } else { "selected" }
        .into_pyobject(py)?
        .into_any();
    let items = [
        float_array(py, batch.features, batch.candidates, batch.width, &consumer)?
            .into_bound(py)
            .into_any(),
        float_array(py, batch.state_features, decisions, batch.width, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.episode_offsets, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.candidate_offsets, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.candidate_counts, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.selected, &consumer)?
            .into_bound(py)
            .into_any(),
        scalar_array(py, batch.old_logprob, &consumer)?
            .into_bound(py)
            .into_any(),
        scalar_array(py, batch.old_value, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.old_logprob_present, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.old_value_present, &consumer)?
            .into_bound(py)
            .into_any(),
        float_array(py, batch.terminal_rewards, batch.episodes, 2, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.winner, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.learner_viewer, &consumer)?
            .into_bound(py)
            .into_any(),
        integer_array(py, batch.split, &consumer)?
            .into_bound(py)
            .into_any(),
        layout,
        PyBytes::new(py, &batch.metadata).into_any(),
        index_columns(py, batch.index_columns, &consumer)?,
        proof,
    ];
    Ok(PyTuple::new(py, items)?.unbind())
}

#[pyclass(frozen)]
struct CancelHandle {
    control: Control,
    stats: Arc<telemetry::Telemetry>,
}
#[pymethods]
impl CancelHandle {
    #[new]
    fn new(deadline: f64) -> PyResult<Self> {
        let stats = Arc::new(telemetry::Telemetry::default());
        stats.control(0, 0);
        Ok(Self {
            control: Control::new(deadline).map_err(python_error)?,
            stats,
        })
    }
    fn cancel(&self) {
        self.stats.control(0, 0);
        self.control.cancel();
    }
}

#[pyclass(frozen)]
struct StagingLease {
    leases: Mutex<Option<(budget::Lease, budget::Lease)>>,
}
#[pymethods]
impl StagingLease {
    fn release(&self) -> PyResult<()> {
        self.leases
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("staging lease panicked"))?
            .take();
        Ok(())
    }
}

#[pyclass(frozen)]
struct NativeSession {
    state: Mutex<OwnedState>,
    budget: Arc<Budget>,
    control: Control,
    wall_deadline: f64,
    stats: Arc<telemetry::Telemetry>,
    controls: Mutex<control_memory::ControlMemory>,
}
enum OwnedState {
    Live {
        core: session::Session,
        retirement: retirement::Retirement<session::Session>,
    },
    Retired,
}
impl OwnedState {
    fn retire(self) -> error::Result<()> {
        match self {
            Self::Live { core, retirement } => retirement.retire(Some(core)),
            Self::Retired => Ok(()),
        }
    }
}
fn retire_core(mut core: session::Session) -> error::Result<()> {
    let result = core.close();
    drop(core);
    result
}
#[pymethods]
impl NativeSession {
    #[new]
    #[pyo3(signature=(plan,settings,deadline,wall_deadline=None,cancel_handle=None))]
    fn new(
        py: Python<'_>,
        plan: &[u8],
        settings: &[u8],
        deadline: f64,
        wall_deadline: Option<f64>,
        cancel_handle: Option<PyRef<'_, CancelHandle>>,
    ) -> PyResult<Self> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        let (shared_control, stats) = if let Some(handle) = cancel_handle {
            (Some(handle.control.clone()), handle.stats.clone())
        } else {
            (None, Arc::new(telemetry::Telemetry::default()))
        };
        stats.control(plan.len() + settings.len(), 0);
        let preflight_started = std::time::Instant::now();
        let (state, retirement) = py
            .detach(|| {
                let control = match shared_control {
                    Some(control) => control.at_deadline(deadline)?,
                    None => Control::new(deadline)?,
                };
                let wall = wall_deadline.unwrap_or(deadline);
                if wall < deadline {
                    return Err(Error::contract(
                        "native wall deadline precedes work deadline",
                    ));
                }
                let settings = session::Settings::decode(settings, &control)?;
                let budget = Budget::new(settings.memory);
                let retirement = retirement::Retirement::start(&budget, retire_core)?;
                let core =
                    session::Session::open(plan, settings, budget, control, wall, stats.clone())?;
                Ok((core, retirement))
            })
            .map_err(python_error)?;
        stats.preflight(preflight_started);
        Ok(Self {
            budget: state.budget.clone(),
            control: state.control.clone(),
            wall_deadline: wall_deadline.unwrap_or(deadline),
            state: Mutex::new(OwnedState::Live {
                core: state,
                retirement,
            }),
            stats,
            controls: Mutex::new(control_memory::ControlMemory::default()),
        })
    }
    fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            control: self.control.clone(),
            stats: self.stats.clone(),
        }
    }
    fn used_bytes(&self) -> usize {
        self.stats.control(0, 0);
        self.budget.used()
    }
    fn peak_bytes(&self) -> usize {
        self.stats.control(0, 0);
        self.budget.peak()
    }
    fn reserve_staging(
        &self,
        py: Python<'_>,
        memory_bytes: usize,
        scratch_bytes: usize,
    ) -> PyResult<StagingLease> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, 0);
        let leases = py
            .detach(|| {
                self.with_state(|s| {
                    s.check()?;
                    let stage = s.stage.as_ref().ok_or_else(|| {
                        Error::contract("native staging requires complete source proof")
                    })?;
                    Ok((
                        s.budget.reserve(memory_bytes)?,
                        stage.scratch.reserve(scratch_bytes)?,
                    ))
                })
            })
            .map_err(python_error)?;
        Ok(StagingLease {
            leases: Mutex::new(Some(leases)),
        })
    }
    fn source_hash(&self, py: Python<'_>, path: String) -> PyResult<String> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, path.len());
        self.stats.output_copy(64);
        py.detach(|| {
            self.with_state(|s| {
                s.check()?;
                s.sources
                    .checked_digest(std::path::Path::new(&path), &s.control)?
                    .map(str::to_owned)
                    .ok_or_else(|| Error::contract("source was not read"))
            })
        })
        .map_err(python_error)
    }
    fn export_staging<'py>(
        &self,
        py: Python<'py>,
        path: String,
        offset: usize,
        rows: usize,
        columns: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, path.len());
        let (data, lease) = py
            .detach(|| {
                self.with_state(|s| {
                    s.check()?;
                    let count = rows
                        .checked_mul(columns)
                        .ok_or_else(|| Error::resource("staging array size overflow"))?;
                    let length = count
                        .checked_mul(8)
                        .and_then(|n| n.checked_add(offset))
                        .ok_or_else(|| Error::resource("staging array size overflow"))?;
                    let mut file = std::fs::File::open(&path)?;
                    if file.metadata()?.len() != length as u64 {
                        return Err(Error::contract("staging array byte length mismatch"));
                    }
                    let mut values = budget::BudgetVec::with_capacity(&s.budget, count)?;
                    file.seek(SeekFrom::Start(offset as u64))?;
                    let _buffer = s.budget.reserve(32768)?;
                    let mut file = std::io::BufReader::with_capacity(32768, file);
                    numeric_io::read_f64(&mut file, &mut values, count, &s.control)?;
                    self.stats.stage_read(count * 8);
                    let (values, lease) = values.into_parts();
                    let data = Array2::from_shape_vec((rows, columns), values)
                        .map_err(|_| Error::contract("staging array shape mismatch"))?;
                    Ok((data, lease))
                })
            })
            .map_err(python_error)?;
        let array = publish_float(py, data, lease, Some(&self.stats))?;
        self.control.check().map_err(python_error)?;
        Ok(array)
    }
    fn control_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, 0);
        let (encoded, authorization_bound) = py
            .detach(|| self.with_state(|s| s.summary()))
            .map_err(python_error)?;
        self.controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .summary(&self.budget, encoded.len(), authorization_bound)
            .map_err(python_error)?;
        let _copy = self.budget.reserve(encoded.len()).map_err(python_error)?;
        let published = PyBytes::new(py, &encoded);
        self.stats.output_copy(encoded.len());
        self.control.check().map_err(python_error)?;
        Ok(published)
    }
    fn authorize(&self, py: Python<'_>, authorization: &[u8]) -> PyResult<()> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(authorization.len(), 0);
        self.controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .authorization(authorization.len())
            .map_err(python_error)?;
        py.detach(|| self.with_state(|s| s.authorize(authorization)))
            .map_err(python_error)
    }
    fn proof<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, 0);
        let encoded = py
            .detach(|| self.with_state(|s| s.proof()))
            .map_err(python_error)?;
        self.controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .proof(encoded.len(), false)
            .map_err(python_error)?;
        let _copy = self.budget.reserve(encoded.len()).map_err(python_error)?;
        let published = PyBytes::new(py, &encoded);
        self.stats.output_copy(encoded.len());
        self.control.check().map_err(python_error)?;
        Ok(published)
    }
    fn next_batch(
        &self,
        py: Python<'_>,
        retained_python_bytes: &Bound<'_, PyAny>,
        consumer_layout: &Bound<'_, PyTuple>,
    ) -> PyResult<Option<Py<PyTuple>>> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.data_call();
        let _timer = self.stats.data_timer();
        self.controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .begin_data()
            .map_err(python_error)?;
        let retained = size_argument(retained_python_bytes)?;
        if consumer_layout.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "consumer layout requires four nonnegative integer coefficients",
            ));
        }
        let mut coefficients = [0usize; 4];
        for (i, value) in consumer_layout.iter().enumerate() {
            coefficients[i] = size_argument(&value)?;
        }
        let layout = pipeline::ConsumerLayout {
            base: coefficients[0],
            decision: coefficients[1],
            candidate: coefficients[2],
            episode: coefficients[3],
        };
        let (batch, online, proof) = py
            .detach(|| {
                self.with_state(|s| {
                    let first = s.stage.is_none();
                    let started = std::time::Instant::now();
                    let result = s.next_batch(retained, layout);
                    if first {
                        self.stats.scan(started);
                    }
                    let batch = result?;
                    let proof = if batch.is_some() && !s.proof_delivered {
                        let proof = s.proof()?;
                        self.controls
                            .lock()
                            .map_err(|_| Error::contract("control reservation poisoned"))?
                            .proof(proof.len(), false)?;
                        s.proof_delivered = true;
                        Some(proof)
                    } else {
                        None
                    };
                    Ok((batch, s.online, proof))
                })
            })
            .map_err(python_error)?;
        let result = batch
            .map(|batch| publish_batch(py, batch, online, proof, &self.stats))
            .transpose()
            .and_then(|value| {
                self.control.check().map_err(python_error)?;
                Ok(value)
            });
        if result.is_err() {
            let _ = self.close_state();
        }
        if let Ok(value) = &result {
            self.stats.delivered(value.is_some());
        }
        result
    }
    fn finalize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, 0);
        let _timer = self.stats.cleanup_timer();
        let encoded = py
            .detach(|| self.with_state(|s| s.finalize()))
            .map_err(python_error)?;
        self.controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .proof(encoded.len(), true)
            .map_err(python_error)?;
        py.detach(|| self.close_state()).map_err(python_error)?;
        let _copy = self.budget.reserve(encoded.len()).map_err(python_error)?;
        let published = PyBytes::new(py, &encoded);
        self.stats.output_copy(encoded.len());
        if budget::monotonic() >= self.wall_deadline {
            return Err(python_error(Error::Deadline));
        }
        Ok(published)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        stack::ensure_minimum_stack().map_err(python_error)?;
        self.stats.control(0, 0);
        let _timer = self.stats.cleanup_timer();
        py.detach(|| self.close_state()).map_err(python_error)
    }
    /// A completed operation hands Python control results to its caller. This
    /// lifecycle call is forbidden while native work could still be admitted.
    fn release_controls(&self) -> PyResult<()> {
        self.stats.control(0, 0);
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(*state, OwnedState::Retired) {
            return Err(python_error(Error::contract(
                "cannot release controls before native retirement",
            )));
        }
        self.controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .release();
        Ok(())
    }
    fn control_reserved_bytes(&self) -> PyResult<usize> {
        Ok(self
            .controls
            .lock()
            .map_err(|_| python_error(Error::contract("control reservation poisoned")))?
            .reserved())
    }
    fn telemetry<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.stats.control(0, 0);
        let (reallocations, reallocated, extended) = self.budget.copy_counters();
        let base = self.stats.json(reallocations, reallocated, extended);
        let encoded = format!(
            "{},\"json_estimated_peak_bytes\":{}}}",
            &base[..base.len() - 1],
            self.budget.json_estimated_peak()
        );
        let output = PyBytes::new(py, encoded.as_bytes());
        self.stats.output_copy(encoded.len());
        output
    }
}
fn size_argument(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nonnegative integer capacity required",
        ));
    }
    value.extract::<usize>()
}
impl NativeSession {
    fn with_state<T>(
        &self,
        action: impl FnOnce(&mut session::Session) -> error::Result<T>,
    ) -> error::Result<T> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                let owned = std::mem::replace(&mut *state, OwnedState::Retired);
                drop(state);
                let primary = Error::contract("native session panicked");
                return match owned.retire() {
                    Ok(()) => Err(primary),
                    Err(cleanup) => Err(Error::Cleanup {
                        primary: Box::new(primary),
                        cleanup: cleanup.to_string(),
                    }),
                };
            }
        };
        let result = match &mut *state {
            OwnedState::Live { core, .. } => action(core),
            OwnedState::Retired => Err(Error::Closed),
        };
        match result {
            Ok(value) => Ok(value),
            Err(primary) => {
                let owned = std::mem::replace(&mut *state, OwnedState::Retired);
                drop(state);
                match owned.retire() {
                    Ok(()) => Err(primary),
                    Err(cleanup) => Err(Error::Cleanup {
                        primary: Box::new(primary),
                        cleanup: cleanup.to_string(),
                    }),
                }
            }
        }
    }
    fn close_state(&self) -> error::Result<()> {
        self.control.cancel();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let owned = std::mem::replace(&mut *state, OwnedState::Retired);
        drop(state);
        owned.retire()
    }
}
impl Drop for NativeSession {
    fn drop(&mut self) {
        self.control.cancel();
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let owned = std::mem::replace(state, OwnedState::Retired);
        let _ = owned.retire();
    }
}

fn python_error(error: Error) -> PyErr {
    use pyo3::exceptions::{
        PyMemoryError, PyOSError, PyRuntimeError, PyTimeoutError, PyValueError,
    };
    match error {
        Error::Contract(message) => PyValueError::new_err(message),
        Error::ResourceLimit(message) => PyMemoryError::new_err(message),
        Error::Deadline => PyTimeoutError::new_err(error.to_string()),
        Error::Io(message) => PyOSError::new_err(message),
        Error::Cleanup { primary, cleanup } => {
            let original = python_error(*primary);
            Python::attach(|py| {
                let _ = original.value(py).call_method1(
                    "add_note",
                    (format!("native retirement cleanup failed: {cleanup}"),),
                );
            });
            original
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[pyfunction]
fn canonical_json<'py>(
    py: Python<'py>,
    encoded: &[u8],
    memory_bytes: usize,
    deadline: f64,
) -> PyResult<Bound<'py, PyBytes>> {
    stack::ensure_minimum_stack().map_err(python_error)?;
    let budget = Budget::new(memory_bytes);
    let control = Control::new(deadline).map_err(python_error)?;
    let result = py
        .detach(|| {
            let document = json::parse(encoded, &budget, &control)?;
            let result = json::canonical(&document.root, &budget, &control)?;
            Ok(result)
        })
        .map_err(python_error)?;
    let _copy = budget.reserve(result.len()).map_err(python_error)?;
    let published = PyBytes::new(py, &result);
    control.check().map_err(python_error)?;
    Ok(published)
}

#[pyfunction]
fn feature_names<'py>(
    py: Python<'py>,
    config: &[u8],
    memory_bytes: usize,
    deadline: f64,
) -> PyResult<Bound<'py, PyList>> {
    stack::ensure_minimum_stack().map_err(python_error)?;
    let control = Control::new(deadline).map_err(python_error)?;
    let names = py
        .detach(|| {
            let budget = Budget::new(memory_bytes);
            let control = Control::new(deadline)?;
            let config = json::parse(config, &budget, &control)?;
            let config = feature::FeatureConfig::from_value(&config.root)?;
            feature::feature_names_controlled(&config, &budget, &control)
        })
        .map_err(python_error)?;
    let result = PyList::empty(py);
    for name in names.iter() {
        control.check().map_err(python_error)?;
        result.append(name)?;
    }
    control.check().map_err(python_error)?;
    Ok(result)
}

#[pyfunction]
fn validate_input(
    py: Python<'_>,
    encoded: &[u8],
    mode: Option<bool>,
    memory_bytes: usize,
    deadline: f64,
) -> PyResult<()> {
    stack::ensure_minimum_stack().map_err(python_error)?;
    py.detach(|| {
        let budget = Budget::new(memory_bytes);
        let control = Control::new(deadline)?;
        let document = json::parse(encoded, &budget, &control)?;
        validation::validate_worker_input_controlled(&document.root, mode, &control)
    })
    .map_err(python_error)
}

#[pyfunction]
fn encode_features<'py>(
    py: Python<'py>,
    input: &[u8],
    actions: &[u8],
    config: &[u8],
    memory_bytes: usize,
    deadline: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    stack::ensure_minimum_stack().map_err(python_error)?;
    let (data, lease) = py
        .detach(|| {
            let budget = Budget::new(memory_bytes);
            let control = Control::new(deadline)?;
            let input = json::parse(input, &budget, &control)?;
            let actions = json::parse(actions, &budget, &control)?;
            let config = json::parse(config, &budget, &control)?;
            validation::validate_worker_input_controlled(&input.root, None, &control)?;
            let config = feature::FeatureConfig::from_value(&config.root)?;
            let actions = actions.root.array()?;
            let width = config.width();
            let length = width
                .checked_mul(actions.len())
                .ok_or_else(|| Error::resource("feature capacity overflow"))?;
            let mut values = budget::BudgetVec::with_capacity(&budget, length)?;
            for _ in 0..length {
                values.push(0.0)?;
            }
            let prepared = feature::PreparedDecision::new(&input.root, &config, &budget, &control)?;
            for (index, action) in actions.iter().enumerate() {
                control.check()?;
                prepared.encode(
                    action,
                    &mut values[index * width..(index + 1) * width],
                    &control,
                )?;
            }
            let (values, lease) = values.into_parts();
            let data = Array2::from_shape_vec((actions.len(), width), values)
                .map_err(|_| Error::contract("feature shape mismatch"))?;
            Ok((data, lease))
        })
        .map_err(python_error)?;
    let array = publish_float(py, data, lease, None)?;
    Control::new(deadline)
        .map_err(python_error)?
        .check()
        .map_err(python_error)?;
    Ok(array)
}

#[pyfunction]
fn build_info() -> Vec<(&'static str, &'static str)> {
    vec![
        ("version", env!("CARGO_PKG_VERSION")),
        ("rustc", env!("GFB_NATIVE_RUSTC")),
        ("target", env!("GFB_NATIVE_TARGET")),
        ("profile", env!("GFB_NATIVE_PROFILE")),
        ("parser_contract", json::PARSER_CONTRACT),
        ("json_memory_policy", json::MEMORY_POLICY),
        ("rustflags", env!("GFB_NATIVE_RUSTFLAGS")),
        ("cargo_lock_sha256", env!("GFB_NATIVE_CARGO_LOCK_SHA256")),
        ("native_source_sha256", env!("GFB_NATIVE_SOURCE_SHA256")),
        ("encoder_version", "participant-action-v1"),
        ("recipe_identity", "native-preprocessing-v1"),
        ("abi", "cp314-gil"),
        (
            "allocator",
            if cfg!(feature = "allocator-mimalloc") {
                "mimalloc"
            } else {
                "system"
            },
        ),
        ("typed_reader_contract", "typed-borrowed-v1"),
        ("episode_pipeline_contract", "owned-complete-episode-v1"),
        ("ppo_policy_models", "mlp,gru,transformer"),
        (
            "structured_preprocessing_contract",
            "participant-structured-v1;journal=5;rollout=3;generation=2;model=2",
        ),
        (
            "structured_default_signature",
            "4b4ae064f08192a322de83e24e4a93ef4191c7d73396997d59cd7710cd05cb04",
        ),
        (
            "card_instance_contract",
            "participant-structured-v2;journal6;rollout4;generation3;model3",
        ),
        (
            "card_instance_signature",
            "ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f",
        ),
        ("card_instance_models", "mlp,gru,transformer"),
        ("card_instance_guardian_acted", "card-or-guardian-v1"),
        ("card_instance_concealed_draw", "raw-preserved;feature-omitted-v1"),
        ("card_instance_defense_priority", "owner-defense-priority-v2"),
        ("card_instance_refresh", "baseline1;options1;identity1"),
        ("card_instance_refresh_opponents", "mixed16-v1;ordinary1;hpo2"),
        ("card_instance_hpo", structured_controls::hpo::CAPABILITY),
        (
            "card_instance_hpo_update_diagnostics",
            structured_controls::hpo::UPDATE_DIAGNOSTICS_CAPABILITY,
        ),
        ("card_instance_hpo_checkpoint_evaluation", "hpo-checkpoint-evaluation-v1"),
        ("card_instance_hpo_update_storage", "hpo-update-zstd-v1"),
        ("card_instance_hpo_resident", structured_controls::resident::CAPABILITY),
        (
            "card_instance_group_pack",
            CARD_INSTANCE_GROUP_PACK_CAPABILITY,
        ),
        ("card_instance_pack_checksum", "pack-checksum-v1;custody2"),
        (
            "card_instance_packed_formal_controls",
            CARD_INSTANCE_PACKED_FORMAL_CAPABILITY,
        ),
        (
            "card_instance_packed_final_validation",
            CARD_INSTANCE_PACKED_FINAL_CAPABILITY,
        ),
        (
            "card_instance_formal_controls",
            "run3-formal-parent-v1;declaration1;stop1",
        ),
        (
            "card_instance_final_validation",
            "collection5-final64;journal6;codec2",
        ),
        (
            "card_instance_original_history",
            "journal6;codec2;episode-dto1;recorded16",
        ),
        (
            "card_instance_fixture_tokens",
            "component-envelope1;participant-structured-v2;codec2",
        ),
        (
            "card_instance_tree_raw",
            "journal6;lightgbm3;inference4;codec2",
        ),
    ]
}

#[pyfunction]
fn player_search_v5<'py>(py: Python<'py>, encoded: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let result = py.detach(|| player_runtime::search_diagnostic(encoded)).map_err(python_error)?;
    Ok(PyBytes::new(py, &result))
}
#[pyfunction]
fn player_select_v5<'py>(py: Python<'py>, encoded: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let result = py.detach(|| player_runtime::select_diagnostic(encoded)).map_err(python_error)?;
    Ok(PyBytes::new(py, &result))
}
#[pyfunction]
fn player_rng_v5<'py>(py: Python<'py>, encoded: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let result = py.detach(|| player_runtime::rng_diagnostic(encoded)).map_err(python_error)?;
    Ok(PyBytes::new(py, &result))
}
#[pymodule]
fn gf_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<player_runtime::batch::PlayerInferenceDispatcher>()?;
    module.add_class::<player_runtime::scheduler::PlayerRuntime>()?;
    module.add_class::<player_runtime::game::storage::StorageBundle>()?;
    module.add_class::<player_runtime::game::storage::RecordingAccount>()?;
    module.add_class::<player_runtime::game::storage::RecordingLease>()?;
    module.add_class::<player_runtime::training::PlayerPpoEpisode>()?;
    module.add_class::<player_runtime::training::PlayerPpoBatch>()?;
    module.add_function(wrap_pyfunction!(player_runtime::training::build_player_ppo_batch, module)?)?;
    module.add_function(wrap_pyfunction!(player_search_v5, module)?)?;
    module.add_function(wrap_pyfunction!(player_select_v5, module)?)?;
    module.add_function(wrap_pyfunction!(player_rng_v5, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(feature_names, module)?)?;
    module.add_function(wrap_pyfunction!(validate_input, module)?)?;
    module.add_function(wrap_pyfunction!(encode_features, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_class::<FloatOwner>()?;
    module.add_class::<ScalarOwner>()?;
    module.add_class::<IntegerOwner>()?;
    module.add_class::<Float32Owner>()?;
    module.add_class::<ByteOwner>()?;
    module.add_class::<OffsetOwner>()?;
    module.add_class::<CancelHandle>()?;
    module.add_class::<StagingLease>()?;
    module.add_class::<NativeSession>()?;
    module.add_class::<StructuredParentControls>()?;
    module.add_class::<StructuredJournalTokens>()?;
    module.add_class::<StructuredNativeSession>()?;
    module.add_class::<CardInstanceNativeSession>()?;
    module.add_class::<CardInstanceFixtureTokens>()?;
    module.add_class::<TreeDatasetSession>()?;
    Ok(())
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    fn owned(fail_close: bool) -> (NativeSession, Arc<Budget>) {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let budget = Budget::new(64 * 1024 * 1024);
                let retirement = retirement::Retirement::start(&budget, retire_core).unwrap();
                let core = session::Session::test_core(&budget, 127, fail_close);
                let native = NativeSession {
                    control: core.control.clone(),
                    wall_deadline: budget::monotonic() + 60.,
                    stats: Arc::new(telemetry::Telemetry::default()),
                    budget: budget.clone(),
                    state: Mutex::new(OwnedState::Live { core, retirement }),
                    controls: Mutex::new(control_memory::ControlMemory::default()),
                };
                (native, budget)
            })
            .unwrap()
            .join()
            .unwrap()
    }
    #[test]
    fn actual_session_gc_drop_is_safe_on_small_caller_stacks() {
        for bytes in [1024 * 1024, 2 * 1024 * 1024] {
            let (native, budget) = owned(false);
            std::thread::Builder::new()
                .stack_size(bytes)
                .spawn(move || drop(native))
                .unwrap()
                .join()
                .unwrap();
            assert_eq!(budget.used(), 0);
        }
    }
    #[test]
    fn close_failure_retires_graph_and_preserves_first_error() {
        let (native, budget) = owned(true);
        std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(move || {
                let result: error::Result<()> =
                    native.with_state(|_| Err(Error::contract("first operation failed")));
                assert!(matches!(result, Err(Error::Cleanup { .. })));
                assert_eq!(budget.used(), 0);
                drop(native);
            })
            .unwrap()
            .join()
            .unwrap();
    }
    #[test]
    fn poisoned_actual_session_gc_also_retires_on_safe_stack() {
        let (native, budget) = owned(false);
        let native = Arc::new(native);
        let other = native.clone();
        assert!(
            std::thread::Builder::new()
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let _lock = other.state.lock().unwrap();
                    panic!("injected caller panic");
                })
                .unwrap()
                .join()
                .is_err()
        );
        std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(move || drop(native))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(budget.used(), 0);
    }
}

/// The bridge itself retains even a zero-bank packet's quota until the next
/// boundary. NumPy base owners share this SAME existing ExportLease Arc.
fn release_final_packet_consumer(
    payload: &Mutex<Option<Arc<ExportLease>>>,
    required: bool,
) -> error::Result<()> {
    let mut payload = payload.lock().map_err(|_| Error::Closed)?;
    if let Some(consumer) = payload.as_ref() {
        if Arc::strong_count(consumer) != 1 {
            if required {
                return Err(Error::contract(
                    "final64 packet array views must retire before next export/close",
                ));
            }
        } else {
            payload.take();
        }
    }
    Ok(())
}
#[cfg(test)]
mod final64_export_lease_tests {
    use super::*;
    #[test]
    fn card_instance_final64_shared_base_owner_and_zero_bank_packet_keep_quota_until_release() {
        for has_view in [false, true] {
            let b = Budget::new(65536);
            let guard = Arc::new(());
            let weak = Arc::downgrade(&guard);
            let consumer = Arc::new(ExportLease {
                _consumer: b.reserve(4096).unwrap(),
                _metadata: b.reserve(2048).unwrap(),
                _index: None,
                _card_guard: Some((guard, b.reserve(1024).unwrap())),
                stats: Arc::new(telemetry::Telemetry::default()),
            });
            let view = if has_view {
                Some(consumer.clone())
            } else {
                None
            };
            let payload = Mutex::new(Some(consumer));
            assert_eq!(b.used(), 7168);
            assert!(weak.upgrade().is_some());
            if has_view {
                assert!(release_final_packet_consumer(&payload, true).is_err());
                release_final_packet_consumer(&payload, false).unwrap();
                assert_eq!(b.used(), 7168);
                assert!(weak.upgrade().is_some());
            }
            drop(view);
            release_final_packet_consumer(&payload, true).unwrap();
            assert_eq!(b.used(), 0);
            assert!(weak.upgrade().is_none());
            release_final_packet_consumer(&payload, true).unwrap();
        }
    }
}

#[cfg(test)]
mod corrected_journal_owner_tests {
    use super::*;
    fn owner(
        failed_retirement: bool,
    ) -> (StructuredJournalTokens, std::path::PathBuf, Arc<Budget>) {
        let budget = Budget::new(256 << 20);
        let control = Control::new(budget::monotonic() + 60.).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
            &budget,
            &control,
        )
        .unwrap();
        let case = &fixture.root["journals"][0];
        let path = std::env::temp_dir().join(format!(
            "gf-j6-owner-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        let mut bytes = Vec::new();
        for row in case["rows"].array().unwrap() {
            bytes.extend_from_slice(&json::canonical(row, &budget, &control).unwrap());
            bytes.push(b'\n');
        }
        std::fs::write(path.join("player-1.jsonl"), bytes).unwrap();
        let hex = case["bytes_hex"].string().unwrap();
        let columns = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        std::fs::write(
            path.join(case["rows"][2]["table"]["file"].string().unwrap()),
            columns,
        )
        .unwrap();
        fn failed(core: structured_journal::Journal) -> error::Result<()> {
            drop(core);
            Err(Error::contract("injected retirement uncertainty"))
        }
        let worker = retirement::Retirement::start(
            &budget,
            if failed_retirement {
                failed
            } else {
                retire_structured_journal
            },
        )
        .unwrap();
        let journal = structured_journal::Journal::open_corrected(
            &path,
            "player-1.jsonl",
            budget.clone(),
            control.clone(),
            16 << 20,
            4 << 20,
        )
        .unwrap();
        let value = StructuredJournalTokens {
            provider: None,
            core: Mutex::new(Some(journal)),
            retirement: Mutex::new(Some(worker)),
            budget: budget.clone(),
            control,
            python: Mutex::new(control_memory::ControlMemory::default()),
            caller: Mutex::new(budget.reserve(0).unwrap()),
            corrected: true,
            phase: Mutex::new(JournalRawPhase::Active),
        };
        (value, path, budget)
    }
    #[test]
    fn corrected_raw_gil_reacquisition_cancel_after_final_source_guard_cannot_publish() {
        let (owner, path, b) = owner(false);
        let bytes = owner.corrected_diagnostic(0).unwrap();
        drop(bytes);
        owner.raw_final_source_guard().unwrap();
        assert!(matches!(
            *owner.phase.lock().unwrap(),
            JournalRawPhase::Publishing
        ));
        owner.control.cancel();
        assert_eq!(
            owner.finish_raw_publication(true).unwrap_err(),
            Error::Cancelled
        );
        assert_eq!(owner.corrected_close().unwrap_err(), Error::Cancelled);
        assert_eq!(b.used(), 0);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn corrected_raw_late_cancel_or_original_mutation_cannot_publish() {
        for cancel in [false, true] {
            let (owner, path, b) = owner(false);
            let bytes = owner.corrected_diagnostic(0).unwrap();
            drop(bytes);
            if cancel {
                owner.control.cancel();
            } else {
                std::fs::write(
                    path.join("player-1.jsonl"),
                    b"changed after detached export",
                )
                .unwrap();
            }
            let error = owner.raw_final_source_guard().unwrap_err();
            if cancel {
                assert_eq!(error, Error::Cancelled);
            }
            assert_eq!(owner.corrected_recheck(0).unwrap_err(), error);
            assert!(owner.finish_raw_publication(true).is_err());
            assert_eq!(owner.corrected_close().unwrap_err(), error);
            assert_eq!(b.used(), 0);
            std::fs::remove_dir_all(path).unwrap();
        }
    }
    #[test]
    fn corrected_raw_global_copy_overlap_failure_exposes_no_packet_and_stays_failed() {
        let (owner, path, b) = owner(false);
        let caller = (256usize << 20) - b.used() - 65536;
        // Caller ownership consumes the SAME quota before diagnostic JSON and
        // its Python control/copy reservation. No actual huge allocation occurs.
        let error = owner
            .corrected_diagnostic(caller)
            .err()
            .expect("copy overlap must refuse before publication");
        assert!(matches!(error, Error::ResourceLimit(_)));
        assert!(!matches!(
            *owner.phase.lock().unwrap(),
            JournalRawPhase::Exported
        ));
        assert_eq!(owner.corrected_diagnostic(0).err().unwrap(), error);
        assert_eq!(owner.corrected_close().unwrap_err(), error);
        assert_eq!(b.used(), 0);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn corrected_raw_quota_failure_is_sticky_before_diagnostic_and_recheck() {
        for diagnostic in [false, true] {
            let (owner, path, b) = owner(false);
            let error = if diagnostic {
                owner.corrected_diagnostic(usize::MAX).err().unwrap()
            } else {
                owner.corrected_recheck(usize::MAX).unwrap_err()
            };
            assert!(matches!(error, Error::ResourceLimit(_)));
            assert_eq!(owner.corrected_recheck(0).unwrap_err(), error);
            assert_eq!(owner.corrected_diagnostic(0).err().unwrap(), error);
            assert_eq!(owner.corrected_close().unwrap_err(), error);
            assert_eq!(b.used(), 0);
            assert_eq!(owner.corrected_close().unwrap_err(), error);
            std::fs::remove_dir_all(path).unwrap();
        }
    }
    #[test]
    fn corrected_raw_publication_failure_has_no_late_success_or_operational_retry() {
        let (owner, path, b) = owner(false);
        let bytes = owner.corrected_diagnostic(0).unwrap();
        drop(bytes);
        assert!(owner.corrected_recheck(0).is_err());
        assert!(owner.corrected_close().is_err()); // publication owns its one slot
        owner.finish_raw_publication(false).unwrap();
        assert!(owner.finish_raw_publication(true).is_err());
        assert!(owner.corrected_recheck(0).is_err());
        assert!(owner.corrected_close().is_err());
        assert_eq!(b.used(), 0);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn corrected_raw_retirement_error_stays_uncertain_across_empty_close_retry() {
        let (owner, path, b) = owner(true);
        let bytes = owner.corrected_diagnostic(0).unwrap();
        drop(bytes);
        owner.raw_final_source_guard().unwrap();
        owner.finish_raw_publication(true).unwrap();
        let first = owner.corrected_close().unwrap_err();
        assert_eq!(first, Error::contract("injected retirement uncertainty"));
        assert!(owner.core.lock().unwrap().is_none());
        assert!(b.used() > 0); // conservative control lease remains until actual owner release
        assert_eq!(owner.corrected_close().unwrap_err(), first);
        assert_eq!(owner.corrected_recheck(0).unwrap_err(), first);
        drop(owner);
        assert_eq!(b.used(), 0);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn corrected_raw_poisoned_lock_keeps_core_reachable_and_gc_joins_real_worker() {
        let (owner, path, b) = owner(false);
        let owner = Arc::new(owner);
        let other = owner.clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = other.core.lock().unwrap();
                panic!("injected owner lock poison");
            })
            .join()
            .is_err()
        );
        assert!(owner.corrected_close().is_err());
        assert!(owner.core.lock().err().unwrap().into_inner().is_some());
        assert!(owner.corrected_close().is_err());
        drop(owner);
        assert_eq!(b.used(), 0);
        std::fs::remove_dir_all(path).unwrap();
    }
}
