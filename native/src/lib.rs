pub mod budget;
pub mod control_memory;
pub mod controls;
#[cfg(feature = "decode-profile")]
pub mod decode_profile;
pub mod error;
pub mod feature;
pub mod index;
pub mod journal;
pub mod json;
pub mod lookup;
pub mod parallel;
pub mod pipeline;
pub mod projection;
pub mod retirement;
pub mod session;
pub mod source;
pub mod stack;
pub mod telemetry;
pub mod validation;

use crate::budget::{Budget, Control};
use crate::error::Error;
use ndarray::{Array1, Array2};
use numpy::PyArray1;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::{PyAny, PyInt, PyList, PyTuple};
use std::io::{Read, Seek, SeekFrom};
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
    stats: Arc<telemetry::Telemetry>,
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
                    for index in 0..count {
                        if index % 4096 == 0 {
                            s.control.check()?;
                        }
                        let mut bytes = [0; 8];
                        file.read_exact(&mut bytes)?;
                        values.push(f64::from_le_bytes(bytes))?;
                    }
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
        let encoded = self.stats.json(reallocations, reallocated, extended);
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
            let actions = actions.root.as_array()?;
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
        (
            "cargo_features",
            if cfg!(feature = "decode-profile") {
                "decode-profile"
            } else {
                "default"
            },
        ),
        ("decode_profile_capture", {
            #[cfg(feature = "decode-profile")]
            {
                crate::decode_profile::capture_identity()
            }
            #[cfg(not(feature = "decode-profile"))]
            {
                "not_compiled"
            }
        }),
        ("rustflags", env!("GFB_NATIVE_RUSTFLAGS")),
        ("cargo_lock_sha256", env!("GFB_NATIVE_CARGO_LOCK_SHA256")),
        ("native_source_sha256", env!("GFB_NATIVE_SOURCE_SHA256")),
        ("encoder_version", "participant-action-v1"),
        ("recipe_identity", "native-preprocessing-v1"),
        ("abi", "cp314-gil"),
    ]
}

#[pymodule]
fn gf_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(feature_names, module)?)?;
    module.add_function(wrap_pyfunction!(validate_input, module)?)?;
    module.add_function(wrap_pyfunction!(encode_features, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_class::<FloatOwner>()?;
    module.add_class::<ScalarOwner>()?;
    module.add_class::<IntegerOwner>()?;
    module.add_class::<ByteOwner>()?;
    module.add_class::<OffsetOwner>()?;
    module.add_class::<CancelHandle>()?;
    module.add_class::<StagingLease>()?;
    module.add_class::<NativeSession>()?;
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
                let core = session::Session::test_core(&budget, 990, fail_close);
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
