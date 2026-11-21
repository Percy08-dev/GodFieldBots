//! Padding for one admitted group. Operands share one slab per execution dtype.

use super::{
    CARD_CATEGORIES, CARD_NUMERICS, EVENT_START, FRAME_FIELDS, Location, OPPONENT_LEARNED,
    PlayerPpoBatch, Reservation, RowOperands, TokenArray, allocate, capacity_bytes, ensure, view,
    wait_duration,
};
use crate::error::{Error, Result};
use ndarray::Array1;
use pyo3::{prelude::*, types::PyDict};
use std::mem::size_of;

const FRAME_ADDONS: usize = 3;
const FRAME_OPERANDS: usize = FRAME_FIELDS.len() + FRAME_ADDONS + 1;
const CANDIDATE_OPERANDS: usize = 8;
const MAX_OPERANDS: usize = FRAME_OPERANDS + CANDIDATE_OPERANDS;
const PAD_SLOT: i64 = 18;

#[derive(Clone, Copy)]
pub(super) enum Part {
    All,
    Frames,
    Candidates,
}
impl Part {
    pub(super) fn parse(name: &str) -> Result<Self> {
        match name {
            "all" => Ok(Self::All),
            "frames" => Ok(Self::Frames),
            "candidates" => Ok(Self::Candidates),
            _ => Err(Error::contract("unknown PPO group part")),
        }
    }
    fn frames(self) -> usize {
        if matches!(self, Self::All | Self::Frames) {
            FRAME_OPERANDS
        } else {
            0
        }
    }
    fn candidates(self) -> usize {
        if matches!(self, Self::All | Self::Candidates) {
            CANDIDATE_OPERANDS
        } else {
            0
        }
    }
}

#[derive(Clone, Copy)]
enum Dtype {
    Long = 0,
    Float = 1,
    Bool = 2,
}
#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    dtype: Dtype,
    start: usize,
    dims: [usize; 3],
    rank: usize,
}
impl Layout {
    const EMPTY: Self = Self {
        name: "",
        dtype: Dtype::Bool,
        start: 0,
        dims: [0; 3],
        rank: 0,
    };

    fn shape(&self) -> &[usize] {
        &self.dims[..self.rank]
    }
    fn row_start(&self, index: usize) -> usize {
        self.start + index * self.shape()[1..].iter().product::<usize>()
    }
}
struct Layouts {
    fields: [Layout; MAX_OPERANDS],
    len: usize,
    counts: [usize; 3],
}
impl Layouts {
    fn fields(&self) -> &[Layout] {
        &self.fields[..self.len]
    }

    fn insert(&mut self, name: &'static str, dtype: Dtype, shape: &[usize]) -> Result<()> {
        let index = dtype as usize;
        let start = self.counts[index];
        self.counts[index] = shape
            .iter()
            .try_fold(1usize, |count, width| count.checked_mul(*width))
            .and_then(|size| start.checked_add(size))
            .ok_or_else(|| Error::resource("PPO packed shape overflow"))?;
        let mut dims = [0; 3];
        dims[..shape.len()].copy_from_slice(shape);
        self.fields[self.len] = Layout {
            name,
            dtype,
            start,
            dims,
            rank: shape.len(),
        };
        self.len += 1;
        Ok(())
    }
    fn with_group(
        first: &RowOperands,
        rows: usize,
        events: usize,
        candidates: usize,
        positions: usize,
        part: Part,
    ) -> Result<Self> {
        let mut output = Self {
            fields: [Layout::EMPTY; MAX_OPERANDS],
            len: 0,
            counts: [0; 3],
        };
        if part.frames() != 0 {
            for (index, (&name, array)) in FRAME_FIELDS.iter().zip(&first.frame).enumerate() {
                let dtype = match name {
                    "state_categories"
                    | "opponent_learned"
                    | "card_categories"
                    | "interval_categories" => Dtype::Long,
                    "card_mask" => Dtype::Bool,
                    _ => Dtype::Float,
                };
                let mut shape = [rows, 0, 0];
                let rank = if index >= EVENT_START {
                    shape[1] = events;
                    shape[2] = first.event_widths[index - EVENT_START];
                    3
                } else {
                    let dims = array.shape();
                    shape[1..1 + dims.len()].copy_from_slice(dims);
                    1 + dims.len()
                };
                output.insert(name, dtype, &shape[..rank])?;
            }
            output.insert(
                "opponent_mask",
                Dtype::Float,
                &[rows, first.frame[OPPONENT_LEARNED].shape()[0]],
            )?;
            output.insert(
                "card_category_suffix",
                Dtype::Long,
                &[
                    rows,
                    first.frame[CARD_CATEGORIES].shape()[0],
                    first.frame[CARD_CATEGORIES].shape()[1] - 2,
                ],
            )?;
            output.insert(
                "card_order",
                Dtype::Float,
                &[rows, first.frame[CARD_NUMERICS].shape()[0]],
            )?;
            output.insert("event_lengths", Dtype::Long, &[rows])?;
        }
        if part.candidates() != 0 {
            output.insert("kinds", Dtype::Long, &[rows, candidates])?;
            output.insert("targets", Dtype::Long, &[rows, candidates])?;
            output.insert("numerics", Dtype::Float, &[rows, candidates, 4])?;
            output.insert("present", Dtype::Float, &[rows, candidates, 4])?;
            output.insert("references", Dtype::Long, &[rows, candidates, positions])?;
            output.insert("mask", Dtype::Bool, &[rows, candidates])?;
            output.insert("selected", Dtype::Long, &[rows])?;
            output.insert("vectors", Dtype::Float, &[3, rows])?;
        }
        Ok(output)
    }
}

pub(super) struct GroupPlan {
    layouts: Layouts,
    positions: usize,
    candidates: usize,
    length_count: usize,
    scratch: usize,
    slab_bytes: usize,
}

impl GroupPlan {
    pub(super) fn new(
        batch: &PlayerPpoBatch,
        mut locations: impl ExactSizeIterator<Item = Result<Location>>,
        location_capacity: usize,
        part: Part,
    ) -> Result<Self> {
        wait_duration(batch.deadline)?;
        let n = locations.len();
        let first = locations
            .next()
            .transpose()?
            .ok_or_else(|| Error::contract("empty PPO group"))?;
        let (episode, row) = batch.row(first)?;
        let positions = episode.positions;
        let length_count = if part.frames() != 0 { n } else { 0 };
        let scratch = total_bytes([
            capacity_bytes::<usize>(length_count)?,
            capacity_bytes::<Location>(location_capacity)?,
        ])?;
        let mut events = 0;
        let mut candidates = 0;
        for location in std::iter::once(Ok(first)).chain(locations) {
            let location = location?;
            let (_, row) = batch.row(location)?;
            ensure(
                batch
                    .selected
                    .as_ref()
                    .is_none_or(|s| s.binary_search(&location).is_ok()),
                "PPO group includes an unused decision",
            )?;
            events = events.max(row.events);
            candidates = candidates.max(row.count);
        }
        let layouts = Layouts::with_group(row, n, events, candidates, positions, part)?;
        let [long_count, float_count, bool_count] = layouts.counts;
        let slab_bytes = total_bytes([
            size_of::<Slabs>(),
            capacity_bytes::<i64>(long_count)?,
            capacity_bytes::<f32>(float_count)?,
            capacity_bytes::<bool>(bool_count)?,
        ])?;
        Ok(Self {
            layouts,
            positions,
            candidates,
            length_count,
            scratch,
            slab_bytes,
        })
    }
}

fn total_bytes(sizes: impl IntoIterator<Item = usize>) -> Result<usize> {
    sizes
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::resource("PPO packed capacity overflow"))
}

#[derive(Default)]
pub(super) struct PackingPeak {
    last: Option<Part>,
    held_candidates: usize,
    bytes: usize,
}
impl PackingPeak {
    pub(super) fn include(&mut self, plan: &GroupPlan, part: Part) -> Result<()> {
        ensure(
            matches!(
                (self.last, part),
                (None, Part::All | Part::Candidates)
                    | (Some(Part::Candidates | Part::Frames), Part::Frames)
            ),
            "invalid PPO packing sequence",
        )?;
        self.bytes = self.bytes.max(total_bytes([
            self.held_candidates,
            plan.slab_bytes,
            plan.scratch,
        ])?);
        if matches!(part, Part::Candidates) {
            self.held_candidates = plan.slab_bytes;
        }
        // The candidate window keeps its host backing. Synchronous CUDA
        // transfers retire each staged frame's host backing before the next.
        self.last = Some(part);
        Ok(())
    }
    pub(super) fn finish(self) -> Result<usize> {
        ensure(
            matches!(self.last, Some(Part::All | Part::Frames)),
            "incomplete PPO packing sequence",
        )?;
        Ok(self.bytes)
    }
}
#[pyclass(frozen)]
struct Slabs {
    longs: Array1<i64>,
    floats: Array1<f32>,
    bools: Array1<bool>,
    _lease: Reservation,
}
impl Slabs {
    fn copy(&mut self, layout: &Layout, index: usize, data: &TokenArray) {
        let start = layout.row_start(index);
        match (layout.dtype, data) {
            (Dtype::Long, TokenArray::Long(_)) => {
                let data = data.longs();
                self.longs.as_slice_mut().unwrap()[start..start + data.len()].copy_from_slice(data);
            }
            (Dtype::Float, TokenArray::Long(_)) => {
                let data = data.longs();
                for (to, &from) in self.floats.as_slice_mut().unwrap()[start..start + data.len()]
                    .iter_mut()
                    .zip(data)
                {
                    *to = from as f32;
                }
            }
            (Dtype::Float, TokenArray::Byte(_)) => {
                let data = data.bytes();
                for (to, &from) in self.floats.as_slice_mut().unwrap()[start..start + data.len()]
                    .iter_mut()
                    .zip(data)
                {
                    *to = f32::from(from);
                }
            }
            (Dtype::Bool, TokenArray::Byte(_)) => {
                let data = data.bytes();
                for (to, &from) in self.bools.as_slice_mut().unwrap()[start..start + data.len()]
                    .iter_mut()
                    .zip(data)
                {
                    *to = from != 0;
                }
            }
            _ => unreachable!("typed PPO slab matches pinned field"),
        }
    }
}
pub(super) struct Packed {
    slabs: Slabs,
    layouts: Layouts,
    event_lengths: Vec<usize>,
    _lease: Reservation,
}
pub(super) fn pack(batch: &PlayerPpoBatch, locations: Vec<Location>, part: Part) -> Result<Packed> {
    let n = locations.len();
    let GroupPlan {
        layouts,
        positions,
        candidates,
        length_count,
        scratch,
        slab_bytes,
    } = GroupPlan::new(
        batch,
        locations.iter().copied().map(Ok),
        locations.capacity(),
        part,
    )?;
    let account = &batch.episodes[0].account;
    let scratch = account.reserve_training(scratch)?;
    let [long_count, float_count, bool_count] = layouts.counts;
    let lease = account.reserve_training(slab_bytes)?;
    let mut longs = allocate(long_count)?;
    longs.resize(long_count, 0);
    let mut floats = allocate(float_count)?;
    floats.resize(float_count, 0.0);
    let mut bools = allocate(bool_count)?;
    bools.resize(bool_count, false);
    let mut slabs = Slabs {
        longs: Array1::from_vec(longs),
        floats: Array1::from_vec(floats),
        bools: Array1::from_vec(bools),
        _lease: lease,
    };
    let (frame_fields, candidate_fields) = layouts.fields().split_at(part.frames());
    let mut event_lengths = allocate(length_count)?;
    for (i, location @ (ep_index, decision)) in locations.into_iter().enumerate() {
        wait_duration(batch.deadline)?;
        let (ep, row) = batch.row(location)?;
        if !frame_fields.is_empty() {
            let [
                opponent_mask,
                card_category_suffix,
                card_order,
                event_lengths_layout,
            ] = &frame_fields[FRAME_FIELDS.len()..]
            else {
                unreachable!("fixed frame layout")
            };
            for (spec, data) in frame_fields.iter().zip(&row.frame) {
                slabs.copy(spec, i, data);
            }
            let mut start = opponent_mask.row_start(i);
            for &value in row.frame[OPPONENT_LEARNED].longs() {
                slabs.floats[start] = if value == 0 { 0.0 } else { 1.0 };
                start += 1;
            }
            let mut start = card_category_suffix.row_start(i);
            let card_cats = &row.frame[CARD_CATEGORIES];
            for card in card_cats.longs().chunks_exact(card_cats.shape()[1]) {
                let data = &card[2..];
                slabs.longs.as_slice_mut().unwrap()[start..start + data.len()]
                    .copy_from_slice(data);
                start += data.len();
            }
            let mut start = card_order.row_start(i);
            let card_nums = &row.frame[CARD_NUMERICS];
            for card in card_nums.longs().chunks_exact(card_nums.shape()[1]) {
                slabs.floats[start] = card[row.card_order_index] as f32;
                start += 1;
            }
            event_lengths.push(row.events);
            slabs.longs[event_lengths_layout.start + i] = row.events as i64;
        }
        if !candidate_fields.is_empty() {
            let [
                kinds,
                targets,
                numerics,
                present,
                references,
                mask,
                selected,
                vectors,
            ] = candidate_fields
            else {
                unreachable!("fixed candidate layout")
            };
            for (spec, data) in [kinds, targets, numerics, present]
                .into_iter()
                .zip(&row.candidates)
            {
                slabs.copy(spec, i, data);
            }
            let start = references.row_start(i);
            let refs =
                &mut slabs.longs.as_slice_mut().unwrap()[start..start + candidates * positions];
            refs.fill(PAD_SLOT);
            let offsets = row.candidates[4].longs();
            let slots = row.candidates[5].longs();
            for (candidate, pair) in offsets.windows(2).enumerate() {
                let data = &slots[pair[0] as usize..pair[1] as usize];
                let start = candidate * positions;
                refs[start..start + data.len()].copy_from_slice(data);
            }
            let start = mask.row_start(i);
            slabs.bools.as_slice_mut().unwrap()[start..start + row.count].fill(true);
            slabs.longs[selected.start + i] = ep.selected[decision];
            slabs.floats[vectors.start + i] = ep.old_logprob[decision];
            slabs.floats[vectors.start + n + i] = batch.targets[ep_index].advantage[decision];
            slabs.floats[vectors.start + 2 * n + i] = batch.targets[ep_index].target[decision];
        }
    }
    Ok(Packed {
        slabs,
        layouts,
        event_lengths,
        _lease: scratch,
    })
}

pub(super) type Output<'py> = (Vec<Bound<'py, PyAny>>, Bound<'py, PyDict>, Vec<usize>);

pub(super) fn export<'py>(py: Python<'py>, packed: Packed) -> PyResult<Output<'py>> {
    let Packed {
        slabs,
        layouts,
        event_lengths,
        _lease,
    } = packed;
    let owner = Bound::new(py, slabs)?;
    let storage = owner.get();
    let arrays = vec![
        view(&storage.longs, owner.as_any())?,
        view(&storage.floats, owner.as_any())?,
        view(&storage.bools, owner.as_any())?,
    ];
    let result = PyDict::new(py);
    for layout in layouts.fields() {
        result.set_item(
            layout.name,
            (layout.dtype as usize, layout.start, layout.shape().to_vec()),
        )?;
    }
    Ok((arrays, result, event_lengths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        budget::{Budget, Control, monotonic},
        json::Value,
        player_runtime::{
            Config, Input,
            game::{GameSpec, storage::Mailbox},
            numeric::ServingRow,
            training::Trajectory,
        },
    };
    use serde_json::json;

    #[test]
    fn acting_tokens_ragged_padding_partial_selection_and_native_lifetime() {
        let originals: Value = serde_json::from_slice(include_bytes!(
            "../../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let episode = originals["episodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|episode| episode["decisions"].as_array().unwrap().len() >= 3)
            .unwrap();
        let decisions = episode["decisions"].as_array().unwrap();
        let config = Config::parse(&episode["actions"], &episode["rules"]).unwrap();
        let viewer = decisions[0]["frame"]["actor"].as_u64().unwrap();
        let policy = json!({
            "kind":"mlp", "seed":17, "checkpoint":"a".repeat(64), "epoch":0,
            "limits":{"max_decisions":64,"max_episode_events":65536, "max_interval_events":65536,
                "max_candidates":65536,"max_positions":18,"aggregate_window":128},
        });
        let deadline = monotonic() + 30.0;
        let spec: GameSpec = serde_json::from_value(json!({
            "episode":"endpoint-test","run":"run","trial":"batch","server":{},"actions":episode["actions"],
            "rules":episode["rules"],"numeric_costs":episode["numeric_costs"],
            "policies":[policy.clone(),policy],"max_decisions":64,"deadline":deadline,
            "cleanup_deadline":deadline,"learner_viewer":viewer,
        })).unwrap();
        let bank = Mailbox::new(64 * 1024 * 1024, 1).unwrap();
        let mut trajectory = Trajectory::new(bank.account().account);
        let budget = Budget::new(64 * 1024 * 1024);
        let control = Control::new(deadline).unwrap();
        let mut events = Vec::new();
        for (i, decision) in decisions.iter().take(3).enumerate() {
            events.extend(
                decision["delta"]["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .cloned(),
            );
            let frame = &decision["frame"];
            let input = Input::parse(frame, &json!(events), &config, &budget, &control).unwrap();
            let table = serde_json::from_value(decision["generated"]["table"].clone()).unwrap();
            trajectory.reserve_next().unwrap();
            let (serving, row) = ServingRow::build_training(
                frame,
                &events,
                &input,
                &config,
                &table,
                128,
                18,
                &budget,
                &control,
                trajectory.account(),
            )
            .unwrap();
            assert_eq!(row.frame[EVENT_START].shape()[0], serving.events);
            assert_eq!(row.candidates[4].longs(), serving.candidate_offsets);
            assert_eq!(
                row.candidates[5].longs(),
                table.slots.iter().map(|&v| v as i64).collect::<Vec<_>>()
            );
            trajectory.accept(
                row,
                i as u64,
                0,
                -(i as f64),
                f64::from([0.25, -0.5, 0.5][i]),
            );
        }
        let finished = trajectory
            .finish(&spec, "game", &json!({"win":{"winner":viewer}}))
            .unwrap();
        let evidence = super::super::PlayerPpoEpisode::new(finished.clone());
        let reserved = bank.used();
        let checkpoint = "a".repeat(64);
        evidence
            .require_identity(
                "run",
                "batch",
                "endpoint-test",
                "game",
                viewer as u8,
                &checkpoint,
                0,
                3,
                Some(viewer as u8),
            )
            .unwrap();
        assert!(
            evidence
                .require_identity(
                    "run",
                    "batch",
                    "endpoint-test",
                    "game",
                    (3 - viewer) as u8,
                    &checkpoint,
                    0,
                    3,
                    Some(viewer as u8)
                )
                .is_err()
        );
        assert!(
            evidence
                .require_identity(
                    "run",
                    "batch",
                    "endpoint-test",
                    "game",
                    viewer as u8,
                    &checkpoint,
                    0,
                    2,
                    Some(viewer as u8)
                )
                .is_err()
        );
        assert!(
            evidence
                .require_identity(
                    "run",
                    "batch",
                    "endpoint-test",
                    "game",
                    viewer as u8,
                    &checkpoint,
                    0,
                    3,
                    None
                )
                .is_err()
        );
        let row = &finished.rows.data[0];
        let frame_hash = decisions[0]["frame"]["frame_hash"].as_str().unwrap();
        let table_hash = decisions[0]["generated"]["table"]["table_hash"]
            .as_str()
            .unwrap();
        evidence
            .require_row(
                0,
                row.revision,
                frame_hash,
                table_hash,
                row.count,
                0,
                -0.0,
                0.25,
            )
            .unwrap();
        assert!(
            evidence
                .require_row(
                    0,
                    row.revision,
                    frame_hash,
                    table_hash,
                    row.count,
                    0,
                    0.0,
                    0.25
                )
                .is_err()
        );
        assert!(
            evidence
                .require_row(
                    0,
                    row.revision,
                    frame_hash,
                    frame_hash,
                    row.count,
                    0,
                    -0.0,
                    0.25
                )
                .is_err()
        );
        assert_eq!(bank.used(), reserved);
        drop(evidence);
        let batch =
            PlayerPpoBatch::build(vec![finished], 0.5, Some(vec![(0, 2), (0, 0)]), deadline)
                .unwrap();
        assert_eq!(bank.used(), batch.resident_bytes);
        assert_eq!(
            batch.targets[0].target.as_slice(),
            Some(&[0.125, 0.75, 1.0][..])
        );
        let normalized = (0.3125_f64 / (0.3125_f64 + 1e-8)) as f32;
        assert_eq!(
            batch.targets[0].advantage.as_slice(),
            Some(&[-normalized, 0.0, normalized][..])
        );
        assert!(pack(&batch, vec![(0, 1)], Part::All).is_err());

        let packed = pack(&batch, vec![(0, 2), (0, 0)], Part::All).unwrap();
        let slots = packed
            .layouts
            .fields()
            .iter()
            .find(|field| field.name == "references")
            .unwrap();
        let mask = packed
            .layouts
            .fields()
            .iter()
            .find(|field| field.name == "mask")
            .unwrap();
        for (i, position) in [2, 0].into_iter().enumerate() {
            let (_, row) = batch.row((0, position)).unwrap();
            assert_eq!(packed.event_lengths[i], row.events);
            let span = slots.dims[1] * slots.dims[2];
            let refs = &packed.slabs.longs.as_slice().unwrap()
                [slots.row_start(i)..slots.row_start(i) + span];
            let mut expected = vec![18; span];
            let offsets = row.candidates[4].longs();
            for (index, offset) in offsets.windows(2).enumerate() {
                let raw = &row.candidates[5].longs()[offset[0] as usize..offset[1] as usize];
                expected[index * 18..index * 18 + raw.len()].copy_from_slice(raw);
            }
            assert_eq!(refs, expected);
            let start = mask.row_start(i);
            assert!(
                packed.slabs.bools.as_slice().unwrap()[start..start + row.count]
                    .iter()
                    .all(|&v| v)
            );
            assert!(
                packed.slabs.bools.as_slice().unwrap()[start + row.count..start + mask.dims[1]]
                    .iter()
                    .all(|&v| !v)
            );
        }

        let frames = pack(&batch, vec![(0, 2)], Part::Frames).unwrap();
        let candidates = pack(&batch, vec![(0, 2)], Part::Candidates).unwrap();
        assert_eq!(frames.layouts.fields().len(), FRAME_OPERANDS);
        assert_eq!(candidates.layouts.fields().len(), CANDIDATE_OPERANDS);
        assert!(candidates.event_lengths.is_empty());
        assert!(bank.used() > batch.resident_bytes);
        let plan = GroupPlan::new(&batch, [(0, 2)].into_iter().map(Ok), 1, Part::Frames).unwrap();
        assert_eq!(plan.scratch, frames._lease.bytes());
        assert_eq!(plan.slab_bytes, frames.slabs._lease.bytes());
        let candidate =
            GroupPlan::new(&batch, [(0, 2)].into_iter().map(Ok), 1, Part::Candidates).unwrap();
        let mut peak = PackingPeak::default();
        peak.include(&candidate, Part::Candidates).unwrap();
        peak.include(&plan, Part::Frames).unwrap();
        let required = (candidate.slab_bytes + plan.scratch + plan.slab_bytes)
            .max(candidate.scratch + candidate.slab_bytes);
        assert_eq!(peak.bytes, required);
        peak.include(&plan, Part::Frames).unwrap();
        assert_eq!(peak.bytes, required);
        drop(frames);
        drop(candidates);
        drop(packed);
        assert_eq!(bank.used(), batch.resident_bytes);
        let account = &batch.episodes[0].account;
        let exact = account
            .reserve_training(64 * 1024 * 1024 - bank.used() - required)
            .unwrap();
        assert!(account.training_fits(required));
        assert!(!account.training_fits(required + 1));
        drop(exact);
        drop(batch);
        assert_eq!(bank.used(), 0);
    }
}
