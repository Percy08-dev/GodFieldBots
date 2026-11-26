//! Typed, accepted learner trajectories and same-process PPO operands.

mod packing;

use super::{
    ensure, error,
    game::{
        GameSpec,
        storage::{Account, Reservation},
    },
    numeric::{Buffer, ServingRow},
    requests::wait_duration,
    search::Table,
};
use crate::{
    error::{Error, Result},
    json::{Value, ValueExt},
    python_error,
};
use ndarray::{Array1, ArrayD, IxDyn};
use numpy::{Element, PyArray};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use std::{
    collections::HashSet,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const FRAME_FIELDS: [&str; 13] = [
    "state_numerics",
    "state_present",
    "state_categories",
    "state_category_present",
    "opponent_learned",
    "card_categories",
    "card_numerics",
    "card_present",
    "card_mask",
    "history",
    "interval_numerics",
    "interval_present",
    "interval_categories",
];
const CANDIDATE_FIELDS: [&str; 6] = [
    "kinds", "targets", "numerics", "present", "offsets", "slots",
];
const CARD_NUMERICS: usize = 6;
const CARD_CATEGORIES: usize = 5;
const OPPONENT_LEARNED: usize = 4;
const EVENT_START: usize = 10;

fn capacity_bytes<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::resource("PPO capacity overflow"))
}
fn allocate<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::resource("PPO allocation failed"))?;
    ensure(
        values.capacity() == count,
        "unexpected PPO allocator capacity",
    )?;
    Ok(values)
}

struct ReservedVec<T> {
    data: Vec<T>,
    lease: Option<Reservation>,
}
impl<T> ReservedVec<T> {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            lease: None,
        }
    }
    fn reserve_next(&mut self, account: &Arc<Account>) -> Result<()> {
        if self.data.len() < self.data.capacity() {
            return Ok(());
        }
        let count = self
            .data
            .capacity()
            .max(2)
            .checked_mul(2)
            .ok_or_else(|| Error::resource("PPO trajectory capacity overflow"))?;
        let lease = account.reserve_training(capacity_bytes::<T>(count)?)?;
        let mut replacement = allocate(count)?;
        replacement.append(&mut self.data);
        self.data = replacement;
        self.lease = Some(lease);
        Ok(())
    }
    fn bytes(&self) -> usize {
        self.lease.as_ref().map_or(0, Reservation::bytes)
    }
}

enum TokenArray {
    Long(ArrayD<i64>),
    Byte(ArrayD<u8>),
}
impl TokenArray {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Long(data) => data.shape(),
            Self::Byte(data) => data.shape(),
        }
    }
    fn longs(&self) -> &[i64] {
        match self {
            Self::Long(data) => data.as_slice().expect("owned contiguous long tokens"),
            Self::Byte(_) => unreachable!("long token field"),
        }
    }
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Byte(data) => data.as_slice().expect("owned contiguous byte tokens"),
            Self::Long(_) => unreachable!("byte token field"),
        }
    }
    fn export<'py>(&self, base: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Long(data) => view(data, base),
            Self::Byte(data) => view(data, base),
        }
    }
}
fn view<'py, T: Element, D: ndarray::Dimension>(
    array: &ndarray::Array<T, D>,
    base: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: every supplied base owns the immutable, fixed-address array.
    // NumPy holds that base (and its admission) for all derived views.
    let array = unsafe { PyArray::borrow_from_array(array, base.clone()) };
    array.call_method1("setflags", (false,))?;
    Ok(array.into_any())
}
fn fill<T>(value: &Value, out: &mut Vec<T>, scalar: &impl Fn(&Value) -> Result<T>) -> Result<()> {
    if let Some(items) = value.as_array() {
        for item in items {
            fill(item, out, scalar)?;
        }
    } else {
        out.push(scalar(value)?);
    }
    Ok(())
}
fn shape_array<T>(dims: &[usize], values: Vec<T>) -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(dims), values).expect("checked contiguous token shape")
}
fn frame_column(name: &str, value: &Value, dims: &[usize]) -> Result<TokenArray> {
    let count = dims.iter().product();
    if name.contains("present") || name.ends_with("mask") {
        let mut data = allocate(count)?;
        fill(value, &mut data, &|v| Ok(u8::from(v.boolean()?)))?;
        ensure(data.len() == count, "invalid PPO byte token shape")?;
        Ok(TokenArray::Byte(shape_array(dims, data)))
    } else {
        let mut data = allocate(count)?;
        fill(value, &mut data, &|v| {
            Ok(v.integer(i64::MIN as i128, i64::MAX as i128)? as i64)
        })?;
        ensure(data.len() == count, "invalid PPO int64 token shape")?;
        Ok(TokenArray::Long(shape_array(dims, data)))
    }
}
fn longs(dims: &[usize], values: impl ExactSizeIterator<Item = i64>) -> Result<TokenArray> {
    let mut data = allocate(values.len())?;
    data.extend(values);
    Ok(TokenArray::Long(shape_array(dims, data)))
}
fn bytes(dims: &[usize], values: impl Iterator<Item = u8>) -> Result<TokenArray> {
    let mut data = allocate(dims.iter().product())?;
    data.extend(values);
    Ok(TokenArray::Byte(shape_array(dims, data)))
}

pub(super) struct RowOperands {
    frame: Vec<TokenArray>,
    candidates: Vec<TokenArray>,
    events: usize,
    event_widths: [usize; 3],
    card_order_index: usize,
    count: usize,
    frame_hash: [u8; 64],
    table_hash: [u8; 64],
    revision: u64,
    lease: Reservation,
}
impl RowOperands {
    pub(super) fn build(
        tokens: &Value,
        table: &Table,
        serving: &ServingRow,
        card_order_index: usize,
        account: &Arc<Account>,
    ) -> Result<Self> {
        let frame_dims = FRAME_FIELDS.map(|name| serving.fields[name].shape());
        let frame_bytes = FRAME_FIELDS
            .iter()
            .zip(frame_dims)
            .map(|(name, dims)| {
                dims.iter().product::<usize>()
                    * if name.contains("present") || name.ends_with("mask") {
                        1
                    } else {
                        8
                    }
            })
            .sum::<usize>();
        let n = serving.candidates;
        let candidate_bytes = n * (8 * 6 + 4) + (table.offsets.len() + table.slots.len()) * 8;
        let metadata_bytes =
            capacity_bytes::<TokenArray>(FRAME_FIELDS.len() + CANDIDATE_FIELDS.len())?;
        let lease = account.reserve_training(frame_bytes + candidate_bytes + metadata_bytes)?;
        let mut frame = allocate(FRAME_FIELDS.len())?;
        for (name, dims) in FRAME_FIELDS.iter().zip(frame_dims) {
            let dims = if dims.first() == Some(&0) { &[0] } else { dims };
            frame.push(frame_column(name, tokens.require(name)?, dims)?);
        }
        let mut candidates = allocate(CANDIDATE_FIELDS.len())?;
        for name in ["kinds", "targets"] {
            let Buffer::Long(values, _) = &serving.fields[name] else {
                unreachable!("serving categories")
            };
            candidates.push(longs(&[n], values.iter().copied())?);
        }
        let mut numerics = allocate(n * 4)?;
        numerics.extend(table.numerics.iter().flatten().copied());
        candidates.push(TokenArray::Long(shape_array(&[n, 4], numerics)));
        candidates.push(bytes(
            &[n, 4],
            table.presence.iter().flatten().map(|&v| u8::from(v)),
        )?);
        candidates.push(longs(&[n + 1], table.offsets.iter().map(|&v| v as i64))?);
        candidates.push(longs(
            &[table.slots.len()],
            table.slots.iter().map(|&v| v as i64),
        )?);
        Ok(Self {
            frame,
            candidates,
            events: serving.events,
            event_widths: std::array::from_fn(|i| frame_dims[EVENT_START + i][1]),
            card_order_index,
            count: n,
            frame_hash: table
                .frame_hash
                .as_bytes()
                .try_into()
                .map_err(|_| error("PPO frame digest width"))?,
            table_hash: table
                .table_hash
                .as_bytes()
                .try_into()
                .map_err(|_| error("PPO table digest width"))?,
            revision: serving.revision,
            lease,
        })
    }
}

pub(super) struct Trajectory {
    account: Arc<Account>,
    rows: ReservedVec<RowOperands>,
    selected: ReservedVec<i64>,
    old_logprob: ReservedVec<f32>,
    old_value: ReservedVec<f32>,
}
impl Trajectory {
    pub(super) fn new(account: Arc<Account>) -> Self {
        Self {
            account,
            rows: ReservedVec::new(),
            selected: ReservedVec::new(),
            old_logprob: ReservedVec::new(),
            old_value: ReservedVec::new(),
        }
    }
    pub(super) fn account(&self) -> &Arc<Account> {
        &self.account
    }
    pub(super) fn reserve_next(&mut self) -> Result<()> {
        self.rows.reserve_next(&self.account)?;
        self.selected.reserve_next(&self.account)?;
        self.old_logprob.reserve_next(&self.account)?;
        self.old_value.reserve_next(&self.account)?;
        Ok(())
    }
    pub(super) fn accept(
        &mut self,
        row: RowOperands,
        decision: u64,
        selected: usize,
        logprob: f64,
        value: f64,
    ) {
        // Capacity was reserved before submitting the simulator action.
        debug_assert_eq!(self.rows.data.len() as u64, decision);
        self.rows.data.push(row);
        self.selected.data.push(selected as i64);
        self.old_logprob.data.push(logprob as f32);
        self.old_value.data.push(value as f32);
    }
    pub(super) fn finish(
        self,
        spec: &GameSpec,
        game: &str,
        outcome: &Value,
    ) -> Result<Arc<Episode>> {
        wait_duration(spec.deadline)?;
        let viewer = spec.learner_viewer.expect("explicit learner trajectory");
        let policy = &spec.policies[usize::from(viewer - 1)];
        let checkpoint = policy.checkpoint.as_ref().expect("learner MLP checkpoint");
        let outcome = if outcome == "draw" {
            Outcome::Draw
        } else {
            Outcome::Win(outcome["win"]["winner"].integer(1, 2)? as u8)
        };
        let bytes = size_of::<Episode>()
            + spec.run.len()
            + spec.trial.len()
            + spec.episode.len()
            + game.len()
            + checkpoint.len();
        let lease = self.account.reserve_training(bytes)?;
        let resident_bytes = lease.bytes()
            + self.rows.bytes()
            + self.selected.bytes()
            + self.old_logprob.bytes()
            + self.old_value.bytes()
            + self
                .rows
                .data
                .iter()
                .map(|row| row.lease.bytes())
                .sum::<usize>();
        Ok(Arc::new(Episode {
            run: spec.run.clone(),
            trial: spec.trial.clone(),
            episode: spec.episode.clone(),
            game: game.to_owned(),
            viewer,
            checkpoint: checkpoint.clone(),
            epoch: policy.epoch,
            positions: policy
                .limits
                .as_ref()
                .expect("learner MLP limits")
                .max_positions,
            deadline: spec.deadline,
            outcome,
            resident_bytes,
            claimed: AtomicBool::new(false),
            account: self.account,
            rows: self.rows,
            selected: Array1::from_vec(self.selected.data),
            old_logprob: Array1::from_vec(self.old_logprob.data),
            old_value: Array1::from_vec(self.old_value.data),
            _leases: [
                self.selected.lease,
                self.old_logprob.lease,
                self.old_value.lease,
            ],
            _lease: lease,
        }))
    }
}
#[derive(Copy, Clone)]
enum Outcome {
    Draw,
    Win(u8),
}
impl Outcome {
    fn reward(self, learner: u8) -> f64 {
        match self {
            Self::Draw => 0.0,
            Self::Win(winner) if winner == learner => 1.0,
            Self::Win(_) => -1.0,
        }
    }
}
pub(crate) struct Episode {
    run: String,
    trial: String,
    episode: String,
    game: String,
    viewer: u8,
    checkpoint: String,
    epoch: u64,
    positions: usize,
    deadline: f64,
    outcome: Outcome,
    resident_bytes: usize,
    claimed: AtomicBool,
    account: Arc<Account>,
    rows: ReservedVec<RowOperands>,
    selected: Array1<i64>,
    old_logprob: Array1<f32>,
    old_value: Array1<f32>,
    _leases: [Option<Reservation>; 3],
    _lease: Reservation,
}
impl Episode {
    fn count(&self) -> usize {
        self.rows.data.len()
    }

    pub(crate) fn require_identity(
        &self, run: &str, trial: &str, episode: &str, game: &str,
        viewer: u8, checkpoint: &str, epoch: u64, decisions: usize, winner: Option<u8>,
    ) -> Result<()> {
        wait_duration(self.deadline)?;
        let actual_winner = match self.outcome {
            Outcome::Draw => None,
            Outcome::Win(winner) => Some(winner),
        };
        ensure(
            self.run == run
                && self.trial == trial
                && self.episode == episode
                && self.game == game
                && self.viewer == viewer
                && self.checkpoint == checkpoint
                && self.epoch == epoch
                && self.count() == decisions
                && actual_winner == winner,
            "native learner/Journal trajectory identity differs",
        )
    }

    pub(crate) fn require_row(
        &self, decision: usize, revision: u64, frame_hash: &str, table_hash: &str,
        candidates: usize, selected: i64, old_logprob: f64, old_value: f64,
    ) -> Result<()> {
        wait_duration(self.deadline)?;
        let row = self.rows.data.get(decision)
            .ok_or_else(|| error("native learner/Journal decision missing"))?;
        ensure(
            row.revision == revision
                && row.frame_hash == frame_hash.as_bytes()
                && row.table_hash == table_hash.as_bytes()
                && row.count == candidates
                && self.selected[decision] == selected
                && f64::from(self.old_logprob[decision]).to_bits() == old_logprob.to_bits()
                && f64::from(self.old_value[decision]).to_bits() == old_value.to_bits(),
            "native learner/Journal decision differs",
        )
    }
}

#[pyclass(frozen)]
pub struct PlayerPpoEpisode {
    inner: Arc<Episode>,
}
impl PlayerPpoEpisode {
    pub(crate) fn new(inner: Arc<Episode>) -> Self {
        Self { inner }
    }
}
#[pymethods]
impl PlayerPpoEpisode {
    #[pyo3(signature = (run, trial, episode, game, viewer, checkpoint, epoch, decisions, winner))]
    fn require_identity(
        &self, run: &str, trial: &str, episode: &str, game: &str,
        viewer: u8, checkpoint: &str, epoch: u64, decisions: usize, winner: Option<u8>,
    ) -> PyResult<()> {
        self.inner.require_identity(run, trial, episode, game, viewer, checkpoint, epoch, decisions, winner)
            .map_err(python_error)
    }

    fn require_row(
        &self, decision: usize, revision: u64, frame_hash: &str, table_hash: &str,
        candidates: usize, selected: i64, old_logprob: f64, old_value: f64,
    ) -> PyResult<()> {
        self.inner.require_row(decision, revision, frame_hash, table_hash, candidates, selected, old_logprob, old_value)
            .map_err(python_error)
    }
}

fn gae(values: &[f32], reward: f64, lambda: f64, advantages: &mut [f64], targets: &mut [f32]) {
    let mut carry = 0.0;
    let mut next = 0.0;
    for (i, &value) in values.iter().enumerate().rev() {
        let value = f64::from(value);
        let step_reward = if i + 1 == values.len() { reward } else { 0.0 };
        carry = step_reward + next - value + lambda * carry;
        advantages[i] = carry;
        targets[i] = (value + carry) as f32;
        next = value;
    }
}
fn population<I: Iterator<Item = f64>>(values: impl Fn() -> I) -> Option<(f64, f64)> {
    let (sum, count) = values().fold((0.0, 0usize), |(sum, count), value| {
        (sum + value, count + 1)
    });
    if count == 0 {
        return None;
    }
    let mean = sum / count as f64;
    let variance = values()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    Some((mean, variance.sqrt() + 1e-8))
}

type Location = (usize, usize);

struct Targets {
    advantage: Array1<f32>,
    target: Array1<f32>,
}
#[pyclass(frozen)]
pub struct PlayerPpoBatch {
    episodes: Vec<Arc<Episode>>,
    targets: Vec<Targets>,
    selected: Option<Vec<Location>>,
    deadline: f64,
    resident_bytes: usize,
    _lease: Reservation,
}
impl PlayerPpoBatch {
    fn build(
        episodes: Vec<Arc<Episode>>,
        lambda: f64,
        mut selected: Option<Vec<Location>>,
        deadline: f64,
    ) -> Result<Self> {
        wait_duration(deadline)?;
        ensure(
            lambda.is_finite() && (0.0..=1.0).contains(&lambda),
            "invalid PPO GAE lambda",
        )?;
        let first = episodes
            .first()
            .ok_or_else(|| error("PPO collection requires episodes"))?;
        let mut names = HashSet::new();
        let mut games = HashSet::new();
        for episode in &episodes {
            ensure(
                Arc::ptr_eq(&episode.account, &first.account)
                    && episode.run == first.run
                    && episode.trial == first.trial
                    && episode.checkpoint == first.checkpoint
                    && episode.epoch == first.epoch
                    && episode.positions == first.positions
                    && episode.deadline.to_bits() == deadline.to_bits()
                    && names.insert(&episode.episode)
                    && games.insert(&episode.game),
                "foreign, duplicate or expired PPO episode capability",
            )?;
        }
        let count = episodes.iter().map(|ep| ep.count()).sum::<usize>();
        if let Some(locations) = selected.as_mut() {
            locations.sort_unstable();
            ensure(
                locations.windows(2).all(|pair| pair[0] != pair[1])
                    && locations
                        .iter()
                        .all(|&(e, p)| episodes.get(e).is_some_and(|ep| p < ep.count())),
                "PPO selection differs from collected decisions",
            )?;
        }
        let resident_allocation = size_of::<Self>()
            + capacity_bytes::<Arc<Episode>>(episodes.capacity())?
            + capacity_bytes::<Targets>(episodes.len())?
            + capacity_bytes::<f32>(count)? * 2
            + capacity_bytes::<Location>(selected.as_ref().map_or(0, Vec::capacity))?;
        let lease = first.account.reserve_training(resident_allocation)?;
        let _scratch = first.account.reserve_training(
            capacity_bytes::<f64>(count)? + capacity_bytes::<usize>(episodes.len() + 1)?,
        )?;
        let mut offsets = allocate(episodes.len() + 1)?;
        offsets.push(0);
        let mut raw = allocate(count)?;
        raw.resize(count, 0.0);
        let mut targets = allocate(episodes.len())?;
        for ep in &episodes {
            wait_duration(deadline)?;
            let mut target = allocate(ep.count())?;
            target.resize(ep.count(), 0.0);
            let mut advantage = allocate(ep.count())?;
            advantage.resize(ep.count(), 0.0);
            let start = *offsets.last().expect("initial offset");
            let end = start + ep.count();
            gae(
                ep.old_value.as_slice().expect("contiguous value column"),
                ep.outcome.reward(ep.viewer),
                lambda,
                &mut raw[start..end],
                &mut target,
            );
            offsets.push(end);
            targets.push(Targets {
                advantage: Array1::from_vec(advantage),
                target: Array1::from_vec(target),
            });
        }
        if let Some(locations) = &selected {
            if let Some((mean, scale)) =
                population(|| locations.iter().map(|&(e, p)| raw[offsets[e] + p]))
            {
                for &(e, p) in locations {
                    targets[e].advantage[p] = ((raw[offsets[e] + p] - mean) / scale) as f32;
                }
            }
        } else if let Some((mean, scale)) = population(|| raw.iter().copied()) {
            for (e, ep) in targets.iter_mut().enumerate() {
                for (value, &source) in ep
                    .advantage
                    .iter_mut()
                    .zip(&raw[offsets[e]..offsets[e + 1]])
                {
                    *value = ((source - mean) / scale) as f32;
                }
            }
        }
        wait_duration(deadline)?;
        for episode in &episodes {
            ensure(
                !episode.claimed.swap(true, Ordering::AcqRel),
                "PPO transaction already consumed",
            )?;
        }
        let resident_bytes =
            resident_allocation + episodes.iter().map(|ep| ep.resident_bytes).sum::<usize>();
        Ok(Self {
            episodes,
            targets,
            selected,
            deadline,
            resident_bytes,
            _lease: lease,
        })
    }
    fn row(&self, (episode, decision): Location) -> Result<(&Episode, &RowOperands)> {
        let episode = self
            .episodes
            .get(episode)
            .ok_or_else(|| error("unknown PPO episode"))?;
        let row = episode
            .rows
            .data
            .get(decision)
            .ok_or_else(|| error("unknown PPO decision"))?;
        Ok((episode, row))
    }
}
#[pymethods]
impl PlayerPpoBatch {
    #[getter]
    fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    fn row_arrays<'py>(
        slf: Bound<'py, Self>,
        episode: usize,
        decision: usize,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        wait_duration(slf.get().deadline).map_err(python_error)?;
        let (_, row) = slf.get().row((episode, decision)).map_err(python_error)?;
        let frame = PyDict::new(slf.py());
        for (name, data) in FRAME_FIELDS.iter().zip(&row.frame) {
            frame.set_item(name, data.export(slf.as_any())?)?;
        }
        let candidates = PyDict::new(slf.py());
        for (name, data) in CANDIDATE_FIELDS.iter().zip(&row.candidates) {
            candidates.set_item(name, data.export(slf.as_any())?)?;
        }
        Ok((frame, candidates))
    }

    fn episode_arrays<'py>(slf: Bound<'py, Self>, episode: usize) -> PyResult<Bound<'py, PyDict>> {
        let batch = slf.get();
        wait_duration(batch.deadline).map_err(python_error)?;
        let ep = batch
            .episodes
            .get(episode)
            .ok_or_else(|| python_error(error("unknown PPO episode")))?;
        let out = PyDict::new(slf.py());
        out.set_item("selected", view(&ep.selected, slf.as_any())?)?;
        for (name, column) in [
            ("old_logprob", &ep.old_logprob),
            ("old_value", &ep.old_value),
            ("advantage", &batch.targets[episode].advantage),
            ("target", &batch.targets[episode].target),
        ] {
            out.set_item(name, view(column, slf.as_any())?)?;
        }
        Ok(out)
    }

    /// Check the actual host recording and numeric caps without building pads.
    /// This checks one full group, or its retained candidates followed by
    /// synchronously transferred frame chunks. Actual reservation still gates
    /// every pack allocation.
    fn _pack_fits(&self, parts: &Bound<'_, PyList>, deadline: f64) -> PyResult<bool> {
        ensure(
            deadline.to_bits() == self.deadline.to_bits(),
            "PPO group deadline differs from collection",
        )
        .map_err(python_error)?;
        wait_duration(deadline).map_err(python_error)?;
        let mut peak = packing::PackingPeak::default();
        for item in parts.iter() {
            let pair = item.cast_into::<PyTuple>()?;
            ensure(pair.len() == 2, "invalid PPO packing part").map_err(python_error)?;
            let locations = pair.get_item(0)?.cast_into::<PyList>()?;
            let part = pair.get_item(1)?;
            let part = packing::Part::parse(part.extract::<&str>()?).map_err(python_error)?;
            let locations_iter = locations.iter().map(|item| {
                item.extract::<Location>()
                    .map_err(|_| error("invalid PPO group location"))
            });
            // The exact Python list also used by pack extracts to a Vec with
            // capacity equal to the input length on the configured allocator.
            let plan = packing::GroupPlan::new(self, locations_iter, locations.len(), part)
                .map_err(python_error)?;
            peak.include(&plan, part).map_err(python_error)?;
        }
        let required = peak.finish().map_err(python_error)?;
        wait_duration(deadline).map_err(python_error)?;
        Ok(self.episodes[0].account.training_fits(required))
    }

    #[pyo3(signature = (locations, deadline, part="all"))]
    fn pack<'py>(
        &self,
        py: Python<'py>,
        locations: Vec<Location>,
        deadline: f64,
        part: &str,
    ) -> PyResult<packing::Output<'py>> {
        ensure(
            deadline.to_bits() == self.deadline.to_bits(),
            "PPO group deadline differs from collection",
        )
        .map_err(python_error)?;
        let part = packing::Part::parse(part).map_err(python_error)?;
        let packed = py
            .detach(|| packing::pack(self, locations, part))
            .map_err(python_error)?;
        wait_duration(deadline).map_err(python_error)?;
        packing::export(py, packed)
    }
}

/// The trusted Python collection/Source owner calls this once after matching
/// every terminal episode to the committed Journal and verifying the Pack seal.
/// This internal bridge checks native ownership and builds operands.
#[pyfunction(name = "_build_player_ppo_batch")]
pub fn build_player_ppo_batch(
    py: Python<'_>,
    episodes: Vec<Py<PlayerPpoEpisode>>,
    gae_lambda: f64,
    selected: Option<Vec<Location>>,
    deadline: f64,
) -> PyResult<PlayerPpoBatch> {
    let episodes = episodes
        .iter()
        .map(|episode| episode.bind(py).get().inner.clone())
        .collect();
    let batch = py
        .detach(|| PlayerPpoBatch::build(episodes, gae_lambda, selected, deadline))
        .map_err(python_error)?;
    wait_duration(deadline).map_err(python_error)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_reverse_gae_precedes_subset_population_normalization() {
        let mut a = [0.; 3];
        let mut target = [0.; 3];
        gae(&[0.25, -0.5, 0.5], 1.0, 0.5, &mut a, &mut target);
        assert_eq!(a, [-0.125, 1.25, 0.5]);
        assert_eq!(target, [0.125, 0.75, 1.0]);
        let (mean, scale) = population(|| [a[0], a[2]].into_iter()).unwrap();
        assert_eq!(mean, 0.1875);
        assert_eq!(scale, 0.3125 + 1e-8);
        assert!(((a[0] - mean) / scale + 1.0).abs() < 1e-6);
        assert!(((a[2] - mean) / scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn one_selected_decision_and_empty_trajectory_are_well_defined() {
        let mut a = [0.];
        let mut target = [0.];
        gae(&[0.25], -1.0, 0.95, &mut a, &mut target);
        assert_eq!(target, [-1.0]);
        assert_eq!(population(|| a.into_iter()).unwrap(), (-1.25, 1e-8));
        gae(&[], 0.0, 0.95, &mut [], &mut []);
        assert!(population(std::iter::empty).is_none());
    }
}
