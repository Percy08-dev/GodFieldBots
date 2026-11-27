//! Final-dtype serving operands. Native token kernels; no Python tokenizer or JSON transport.
use super::{Config, Input, ensure, error, game::storage::Account, search::Table, training::RowOperands};
use crate::{
    budget::{Budget, Control},
    error::Result,
    json::{Value, ValueExt},
    structured_tokens::{self, Recipe},
};
use ndarray::{ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDict};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

pub enum Buffer {
    Float(Vec<f32>, Vec<usize>),
    Long(Vec<i64>, Vec<usize>),
    Bool(Vec<bool>, Vec<usize>),
}
impl Buffer {
    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Float(_, shape) | Self::Long(_, shape) | Self::Bool(_, shape) => shape,
        }
    }
    fn export(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        macro_rules! array {
            ($v:expr,$s:expr) => {
                ArrayD::from_shape_vec(IxDyn(&$s), $v)
                    .expect("owned checked shape")
                    .into_pyarray(py)
                    .into_any()
                    .unbind()
            };
        }
        Ok(match self {
            Self::Float(v, s) => array!(v, s),
            Self::Long(v, s) => array!(v, s),
            Self::Bool(v, s) => array!(v, s),
        })
    }
    pub fn bytes(&self) -> usize {
        match self {
            Self::Float(v, _) => v.len() * 4,
            Self::Long(v, _) => v.len() * 8,
            Self::Bool(v, _) => v.len(),
        }
    }
}
pub struct ServingRow {
    pub actor: u8,
    pub revision: u64,
    pub fields: BTreeMap<String, Buffer>,
    pub candidates: usize,
    pub events: usize,
    pub event_start: usize,
    pub event_end: usize,
    pub candidate_offsets: Vec<i64>,
    pub assembly_seconds: f64,
}
fn shape(v: &Value) -> Vec<usize> {
    match v.as_array() {
        Some(a) if !a.is_empty() => {
            let mut s = vec![a.len()];
            s.extend(shape(&a[0]));
            s
        }
        Some(_) => vec![0],
        None => vec![],
    }
}
fn flatten(v: &Value, out: &mut Vec<i64>) -> Result<()> {
    if let Some(a) = v.as_array() {
        for v in a {
            flatten(v, out)?;
        }
    } else {
        out.push(if let Some(b) = v.as_bool() {
            i64::from(b)
        } else {
            v.integer(i64::MIN as i128, i64::MAX as i128)? as i64
        });
    }
    Ok(())
}
fn operand(v: &Value, long: bool, boolean: bool, dims: Option<Vec<usize>>) -> Result<Buffer> {
    let s = dims.unwrap_or_else(|| shape(v));
    let mut values = vec![];
    flatten(v, &mut values)?;
    ensure(
        s.iter().product::<usize>() == values.len(),
        "serving operand shape mismatch",
    )?;
    Ok(if long {
        Buffer::Long(values, s)
    } else if boolean {
        Buffer::Bool(values.into_iter().map(|v| v != 0).collect(), s)
    } else {
        Buffer::Float(values.into_iter().map(|v| v as f32).collect(), s)
    })
}
impl ServingRow {
    /// Frame/events and table are the actor's validated own view and typed generated table.
    pub fn build(
        frame: &Value,
        events: &[Value],
        input: &Input,
        config: &Config,
        table: &Table,
        window: usize,
        max_positions: usize,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        Self::build_context(
            frame,
            events,
            input,
            config,
            table,
            window,
            max_positions,
            budget,
            control,
            false,
            None,
        ).map(|(serving, _)| serving)
    }
    /// Native actor only: its incremental History owns original validation.
    pub(crate) fn build_owned(
        frame: &Value,
        events: &[Value],
        input: &Input,
        config: &Config,
        table: &Table,
        window: usize,
        max_positions: usize,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        Self::build_context(
            frame,
            events,
            input,
            config,
            table,
            window,
            max_positions,
            budget,
            control,
            true,
            None,
        ).map(|(serving, _)| serving)
    }
    pub(super) fn build_training(
        frame: &Value,
        events: &[Value],
        input: &Input,
        config: &Config,
        table: &Table,
        window: usize,
        max_positions: usize,
        budget: &Arc<Budget>,
        control: &Control,
        account: &Arc<Account>,
    ) -> Result<(Self, RowOperands)> {
        let (serving, training) = Self::build_context(frame, events, input, config, table,
            window, max_positions, budget, control, true, Some(account))?;
        Ok((serving, training.expect("explicit learner row")))
    }
    fn build_context(
        frame: &Value,
        events: &[Value],
        input: &Input,
        config: &Config,
        table: &Table,
        window: usize,
        max_positions: usize,
        budget: &Arc<Budget>,
        control: &Control,
        owned: bool,
        training: Option<&Arc<Account>>,
    ) -> Result<(Self, Option<RowOperands>)> {
        let at = Instant::now();
        ensure(
            (1..=65536).contains(&window) && (1..=128).contains(&max_positions),
            "serving model geometry outside corrected capacity",
        )?;
        ensure(
            table.frame_hash == input.frame_hash
                && frame["frame_hash"].as_str() == Some(input.frame_hash.as_str())
                && table.action_contract == config.action_contract,
            "serving table own binding mismatch",
        )?;
        let recipe = Recipe::corrected(window, budget, control)?;
        let tokens = if owned {
            structured_tokens::frame_tokens_owned(frame, events, &recipe, budget, control)?
        } else {
            structured_tokens::frame_tokens_for_profile(
                frame, events, &recipe, budget, control, config.wire_profile(),
            )?
        };
        let t = &tokens.value;
        let layout: Value =
            serde_json::from_slice(crate::structured_controls::CARD_INSTANCE_RECIPE).unwrap();
        let mut fields = BTreeMap::new();
        let count = t.require("interval_numerics")?.array()?.len();
        ensure(
            count <= input.boundary - input.interval_start,
            "interval token count exceeds raw interval",
        )?;
        for name in [
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
        ] {
            let long = matches!(
                name,
                "state_categories" | "opponent_learned" | "card_categories" | "interval_categories"
            );
            let dims = if name.starts_with("interval_") {
                ensure(
                    t.require(name)?.array()?.len() == count,
                    "interval token row mismatch",
                )?;
                Some(vec![
                    count,
                    layout[if name == "interval_categories" {
                        "event_categories"
                    } else {
                        "event_numeric"
                    }]
                    .array()?
                    .len(),
                ])
            } else {
                None
            };
            fields.insert(
                name.into(),
                operand(t.require(name)?, long, name == "card_mask", dims)?,
            );
        }
        let opponent = t["opponent_learned"].array()?;
        fields.insert(
            "opponent_mask".into(),
            Buffer::Float(
                opponent
                    .iter()
                    .map(|v| if v.as_i64() == Some(0) { 0. } else { 1. })
                    .collect(),
                vec![opponent.len()],
            ),
        );
        let cats = t["card_categories"].array()?;
        fields.insert(
            "card_category_suffix".into(),
            Buffer::Long(
                cats.iter()
                    .flat_map(|r| {
                        r.as_array().unwrap()[2..]
                            .iter()
                            .map(|v| v.as_i64().unwrap())
                    })
                    .collect(),
                vec![18, cats[0].array()?.len() - 2],
            ),
        );
        let order = layout["card_numeric"]
            .array()?
            .iter()
            .position(|v| v == "possession_position")
            .ok_or_else(|| error("corrected possession order field absent"))?;
        fields.insert(
            "card_order".into(),
            Buffer::Float(
                t["card_numerics"]
                    .array()?
                    .iter()
                    .map(|r| r[order].as_i64().unwrap() as f32)
                    .collect(),
                vec![18],
            ),
        );
        let n = table.kinds.len();
        ensure(
            n > 0 && table.offsets.len() == n + 1,
            "empty or malformed generated table",
        )?;
        let category = |field: &[String], vocab: &Value| -> Result<Vec<i64>> {
            field
                .iter()
                .map(|s| {
                    vocab
                        .array()?
                        .iter()
                        .position(|v| v.as_str() == Some(s))
                        .map(|p| p as i64 + 2)
                        .ok_or_else(|| error("unknown serving candidate category"))
                })
                .collect()
        };
        fields.insert(
            "kinds".into(),
            Buffer::Long(
                category(&table.kinds, &layout["candidate_categories"][0])?,
                vec![n],
            ),
        );
        fields.insert(
            "targets".into(),
            Buffer::Long(
                category(&table.targets, &layout["candidate_categories"][1])?,
                vec![n],
            ),
        );
        fields.insert(
            "numerics".into(),
            Buffer::Float(
                table.numerics.iter().flatten().map(|v| *v as f32).collect(),
                vec![n, 4],
            ),
        );
        fields.insert(
            "present".into(),
            Buffer::Float(
                table
                    .presence
                    .iter()
                    .flatten()
                    .map(|v| if *v { 1. } else { 0. })
                    .collect(),
                vec![n, 4],
            ),
        );
        let mut refs = vec![18i64; n * max_positions];
        for (i, pair) in table.offsets.windows(2).enumerate() {
            let slots = &table.slots[pair[0]..pair[1]];
            ensure(
                slots.len() <= max_positions,
                "candidate exceeds model position capacity",
            )?;
            for (j, slot) in slots.iter().enumerate() {
                refs[i * max_positions + j] = *slot as i64;
            }
        }
        fields.insert(
            "references".into(),
            Buffer::Long(refs, vec![n, max_positions]),
        );
        let mut serving = Self {
            actor: input.actor,
            revision: input.revision,
            event_start: input.interval_start,
            event_end: input.boundary,
            fields,
            candidates: n,
            events: count,
            candidate_offsets: table.offsets.iter().map(|v| *v as i64).collect(),
            assembly_seconds: 0.,
        };
        let row = training.map(|account| RowOperands::build(t, table, &serving, order, account))
            .transpose()?;
        serving.assembly_seconds = at.elapsed().as_secs_f64();
        Ok((serving, row))
    }
    pub fn bytes(&self) -> usize {
        self.fields.values().map(Buffer::bytes).sum::<usize>() + self.candidate_offsets.len() * 8
    }
    pub fn export(self, py: Python<'_>) -> PyResult<(Py<PyDict>, Py<PyAny>)> {
        let out = PyDict::new(py);
        for (name, buffer) in self.fields {
            out.set_item(name, buffer.export(py)?)?;
        }
        Ok((
            out.unbind(),
            self.candidate_offsets.into_pyarray(py).into_any().unbind(),
        ))
    }
}
