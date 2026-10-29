//! Arrow-compatible column buffers. Recorded returns stay raw until Python checks.
use crate::json::ValueExt;
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    journal::string,
    json::{self, Value},
    pipeline::Episode,
    projection::EpisodeLinks,
    session::Collection,
    source::Sources,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub struct IndexInfo {
    strings: [String; 8],
    generation: i64,
    viewer: i64,
    reward: f64,
    _metadata: crate::budget::Lease,
}
impl IndexInfo {
    pub fn new(
        collection: &Collection,
        links: &EpisodeLinks<'_>,
        header: &Value,
        reward: f64,
        dataset: &Value,
        sources: &Sources,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        let m = &collection.manifest.root;
        let link = m.require("ppo")?;
        let viewer = header.require("viewer")?.integer(1, 2)? as i64;
        let policies = links.spec.require("policies")?.array()?;
        let values = [
            string(m, "run_id")?,
            string(link, "parent_run_id")?,
            string(link, "batch_id")?,
            string(m, "split")?,
            policies[(2 - viewer) as usize].string()?,
            sources.digest(&collection.directory.join("manifest.json"), control)?,
            string(header, "game_id")?,
            string(dataset, "content_sha256")?,
        ];
        let metadata_bytes = values
            .iter()
            .try_fold(8 * size_of::<String>(), |total, text| {
                total
                    .checked_add(text.len())
                    .ok_or_else(|| Error::resource("index metadata size overflow"))
            })?;
        let metadata = budget.reserve(metadata_bytes)?;
        Ok(Self {
            _metadata: metadata,
            strings: [
                (values[0]).to_owned(),
                (values[1]).to_owned(),
                (values[2]).to_owned(),
                (values[3]).to_owned(),
                (values[4]).to_owned(),
                (values[5]).to_owned(),
                (values[6]).to_owned(),
                (values[7]).to_owned(),
            ],
            generation: link.require("collection_generation")?.integer(0, 1000)? as i64,
            viewer,
            reward,
        })
    }
    pub fn utf8_bytes(&self, decisions: usize) -> Result<usize> {
        let sum = self.strings.iter().try_fold(0usize, |n, s| {
            n.checked_add(s.as_bytes().len())
                .ok_or_else(|| Error::resource("index UTF8 capacity overflow"))
        })?;
        sum.checked_mul(decisions)
            .ok_or_else(|| Error::resource("index UTF8 capacity overflow"))
    }
    pub fn producer_bytes(&self, decisions: usize) -> Result<usize> {
        self.utf8_bytes(decisions)?
            .checked_add(
                (decisions + 1)
                    .checked_mul(32)
                    .ok_or_else(|| Error::resource("index offset overflow"))?,
            )
            .and_then(|n| decisions.checked_mul(32).and_then(|v| n.checked_add(v)))
            .ok_or_else(|| Error::resource("index capacity overflow"))
    }
    fn row(&self, decision: usize) -> Result<Value> {
        let text = |i: usize| json::text(self.strings[i].as_str());
        Ok(json::object([
            ("run_id", text(0)),
            ("parent_run_id", text(1)),
            ("batch_id", text(2)),
            ("generation", json::integer(self.generation as i128)),
            ("split", text(3)),
            ("learner_viewer", json::integer(self.viewer as i128)),
            ("opponent", text(4)),
            ("source_manifest_sha256", text(5)),
            ("game_id", text(6)),
            ("decision", json::integer(decision as i128)),
            ("return", json::float(self.reward)?),
            ("dataset_content_sha256", text(7)),
        ]))
    }
}
pub fn content_digest(
    episodes: &[Episode],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<(String, usize)> {
    let mut hash = Sha256::new();
    hash.update(b"[");
    let mut count = 0usize;
    let mut scratch = 0usize;
    for episode in episodes {
        let info = episode
            .index
            .as_ref()
            .ok_or_else(|| Error::contract("index metadata missing"))?;
        for decision in 0..episode.decisions {
            control.check()?;
            if count > 0 {
                hash.update(b",");
            }
            let row = info.row(decision)?;
            let encoded = json::canonical(&row, budget, control)?;
            hash.update(&*encoded);
            count += 1;
            scratch = encoded
                .len()
                .checked_mul(4)
                .and_then(|n| n.checked_add(4096))
                .and_then(|n| scratch.checked_add(n))
                .ok_or_else(|| Error::resource("index output scratch capacity overflow"))?;
        }
    }
    hash.update(b"]");
    Ok((format!("{:x}", hash.finalize()), scratch))
}
pub struct Utf8Column {
    pub data: BudgetVec<u8>,
    pub offsets: BudgetVec<i32>,
}
pub struct Columns {
    pub strings: [Utf8Column; 8],
    pub generation: BudgetVec<i64>,
    pub viewer: BudgetVec<i64>,
    pub decision: BudgetVec<i64>,
    pub returns: BudgetVec<f64>,
}
impl Columns {
    pub fn new(episodes: &[Episode], budget: &Arc<Budget>, control: &Control) -> Result<Self> {
        let decisions = episodes.iter().try_fold(0usize, |n, e| {
            n.checked_add(e.decisions)
                .ok_or_else(|| Error::resource("index row count overflow"))
        })?;
        fn text_column(
            episodes: &[Episode],
            column: usize,
            decisions: usize,
            budget: &Arc<Budget>,
            control: &Control,
        ) -> Result<Utf8Column> {
            let mut bytes = 0usize;
            for episode in episodes {
                let info = episode
                    .index
                    .as_ref()
                    .ok_or_else(|| Error::contract("index metadata missing"))?;
                bytes = info.strings[column]
                    .as_bytes()
                    .len()
                    .checked_mul(episode.decisions)
                    .and_then(|n| bytes.checked_add(n))
                    .ok_or_else(|| Error::resource("index UTF8 capacity overflow"))?;
            }
            if bytes > i32::MAX as usize {
                return Err(Error::resource(
                    "complete index episode exceeds Arrow UTF8 offset capacity",
                ));
            }
            let mut data = BudgetVec::with_capacity(budget, bytes)?;
            let mut offsets = BudgetVec::with_capacity(budget, decisions + 1)?;
            offsets.push(0)?;
            for episode in episodes {
                let info = episode
                    .index
                    .as_ref()
                    .ok_or_else(|| Error::contract("index metadata missing"))?;
                for _ in 0..episode.decisions {
                    control.check()?;
                    data.extend_from_slice(info.strings[column].as_bytes())?;
                    offsets.push(data.len() as i32)?;
                }
            }
            Ok(Utf8Column { data, offsets })
        }
        let strings = [
            text_column(episodes, 0, decisions, budget, control)?,
            text_column(episodes, 1, decisions, budget, control)?,
            text_column(episodes, 2, decisions, budget, control)?,
            text_column(episodes, 3, decisions, budget, control)?,
            text_column(episodes, 4, decisions, budget, control)?,
            text_column(episodes, 5, decisions, budget, control)?,
            text_column(episodes, 6, decisions, budget, control)?,
            text_column(episodes, 7, decisions, budget, control)?,
        ];
        let mut output = Self {
            strings,
            generation: BudgetVec::with_capacity(budget, decisions)?,
            viewer: BudgetVec::with_capacity(budget, decisions)?,
            decision: BudgetVec::with_capacity(budget, decisions)?,
            returns: BudgetVec::with_capacity(budget, decisions)?,
        };
        for episode in episodes {
            let info = episode
                .index
                .as_ref()
                .ok_or_else(|| Error::contract("index metadata missing"))?;
            for decision in 0..episode.decisions {
                control.check()?;
                output.generation.push(info.generation)?;
                output.viewer.push(info.viewer)?;
                output.decision.push(decision as i64)?;
                output.returns.push(info.reward)?;
            }
        }
        Ok(output)
    }
}
