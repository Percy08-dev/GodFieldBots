//! Original-history recorded16 projection on the SAME retained parent Sources.
//! No parent baseline, numeric permit, scheduler or qualification is created.
use super::*;
use crate::structured_controls::digest;
use crate::structured_source::CollectionLimits;
use crate::structured_staging::Stage;

pub(super) struct State {
    retirement: Option<crate::retirement::Retirement<Stage>>,
    packet: std::sync::Weak<()>,
    uncertain: bool,
}
impl State {
    pub(super) fn new() -> Self {
        Self {
            retirement: None,
            packet: std::sync::Weak::new(),
            uncertain: false,
        }
    }
}
impl CardInstanceNativeSession {
    pub(crate) fn original_has_packet(&self) -> bool {
        self.original_episode.packet.upgrade().is_some()
    }
    pub(super) fn original_episode_idle(&self) -> Result<()> {
        if self.original_has_packet()
            || self.original_episode.retirement.is_some()
            || self.original_episode.uncertain
        {
            return Err(Error::contract(
                "original-history packet/retirement remains live",
            ));
        }
        Ok(())
    }
    pub(super) fn original_episode_retired(&self) -> bool {
        !self.original_has_packet()
            && self.original_episode.retirement.is_none()
            && !self.original_episode.uncertain
    }
    pub(crate) fn recorded_episode_preconditions(
        &self,
        collection: usize,
        episode: usize,
    ) -> Result<()> {
        self.final_slot_operation(CardInstanceOperation::Recorded(collection))?;
        self.original_episode_idle()?;
        let core = self.core()?;
        if core.inputs.format.hpo() {
            return Err(Error::contract(
                "HPO payload uses its guarded cohort profile, not legacy recorded16 exports",
            ));
        }
        core.recorded_export_preconditions(collection)?;
        if episode >= 16 || self.payload.upgrade().is_some() || core.inputs.stage.is_some() {
            return Err(Error::contract(
                "recorded original16 ordinal/views/Stage differ",
            ));
        }
        Ok(())
    }
    pub(crate) fn export_recorded_episode_payload(
        &mut self,
        collection: usize,
        episode: usize,
    ) -> Result<CardInstancePayload> {
        self.recorded_episode_preconditions(collection, episode)?;
        self.core_mut()?.phase = Phase::Failed;
        let result = self.export_original_inner(collection, episode);
        if result.is_ok() {
            self.core_mut()?.phase = Phase::Authorized;
            if let Err(error) =
                self.final_slot_operation(CardInstanceOperation::Recorded(collection))
            {
                self.core_mut()?.phase = Phase::Failed;
                return Err(error);
            }
        }
        result
    }
    fn export_original_inner(
        &mut self,
        collection: usize,
        episode: usize,
    ) -> Result<CardInstancePayload> {
        let core = self.core.as_mut().ok_or(Error::Closed)?;
        core.inputs.recheck()?;
        let i = &mut core.inputs;
        let b = &i.budget;
        let c = &i.control;
        let source = &i.collections[collection];
        if source.digests.len() != 16 {
            return Err(Error::contract(
                "whole original sixteen selected roles required",
            ));
        }
        let m = &source.manifest.root;
        let spec = m
            .require("schedule")?
            .array()?
            .iter()
            .flat_map(|g| {
                g.get("episodes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .nth(episode)
            .ok_or(Error::Closed)?;
        let name = spec.require("episode_id")?.string()?;
        let viewer = spec.require("learner_viewer")?.integer(1, 2)?;
        let journal = source
            .directory
            .join(name)
            .join(format!("player-{viewer}.jsonl"));
        let length = source
            .sources
            .iter()
            .find(|s| s.owns_original(&journal))
            .ok_or(Error::Closed)?
            .original_len(&journal)?;
        let _copies = json::admit(
            length
                .checked_mul(2)
                .and_then(|n| n.checked_add(65536))
                .ok_or_else(|| Error::resource("recorded episode copy overflow"))?,
            b,
        )?;
        self.original_episode.retirement =
            Some(crate::retirement::Retirement::start(b, |stage: Stage| {
                drop(stage);
                Ok(())
            })?);
        i.stage = Some(Stage::new(
            b.clone(),
            i.scratch_bytes,
            i.chunk_bytes,
            c.clone(),
        )?);
        let source = &mut i.collections[collection];
        let mut body = source.restage_original_episode(
            episode,
            i.stage.as_mut().unwrap(),
            &CollectionLimits {
                source_bytes: i.source_bytes,
                record_bytes: i.record_bytes,
            },
            b,
            c,
        )?;
        let spec = source
            .manifest
            .root
            .require("schedule")?
            .array()?
            .iter()
            .flat_map(|g| {
                g.get("episodes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .nth(episode)
            .ok_or(Error::Closed)?;
        let link = source.manifest.root.require("linkage")?;
        for (key, value) in [
            ("schema_version", json::integer(1)),
            ("kind", json::text("card-instance-recorded-episode-payload")),
            (
                "summary_sha256",
                json::text(&digest(&core.controls.summary.root, b, c)?),
            ),
            (
                "parent_run_sha256",
                json::text(
                    i.parents[0]
                        .sources
                        .verified_sha(&i.parents[0].directory.join("run.json"))?,
                ),
            ),
            ("collection_index", json::integer(collection as i128)),
            (
                "collection_generation",
                json::copy(link.require("collection_generation")?, c)?,
            ),
            (
                "collection_manifest_sha256",
                json::copy(source.proof.root.require("manifest_sha256")?, c)?,
            ),
            (
                "source_set_sha256",
                json::copy(source.proof.root.require("source_set_sha256")?, c)?,
            ),
            ("episode_index", json::integer(episode as i128)),
            ("episode_id", json::copy(spec.require("episode_id")?, c)?),
            ("viewer", json::copy(spec.require("learner_viewer")?, c)?),
            ("episode_digest", json::text(&source.digests[episode])),
            ("arrays", i.stage.as_ref().unwrap().descriptors()?),
        ] {
            body.as_object_mut().unwrap().insert(key.into(), value);
        }
        // Five shared BODY keys + thirteen envelope keys = EXACT eighteen.
        if body.object()?.len() != 18 {
            return Err(Error::contract(
                "recorded episode packet exact eighteen fields required",
            ));
        }
        let metadata = json::canonical(&body, b, c)?;
        if metadata.len() > i.record_bytes {
            return Err(Error::resource("recorded episode metadata record cap"));
        }
        let banks = i.stage.as_mut().unwrap().export()?;
        i.stage.as_mut().unwrap().close()?;
        if let Err(error) = self
            .original_episode
            .retirement
            .take()
            .ok_or(Error::Closed)?
            .retire(i.stage.take())
        {
            self.original_episode.uncertain = true;
            return Err(error);
        }
        i.recheck()?;
        let lease = b.reserve(1024)?;
        let guard = Arc::new(());
        self.original_episode.packet = Arc::downgrade(&guard);
        Ok(CardInstancePayload {
            metadata,
            banks,
            _guard: guard,
            _lease: lease,
        })
    }
    pub(super) fn close_original_episode(&mut self) -> Result<()> {
        if self.original_episode.uncertain {
            return Err(Error::contract(
                "recorded episode retirement remains unconfirmed",
            ));
        }
        if self.original_has_packet() {
            return Err(Error::contract("recorded episode views remain live"));
        }
        if self.original_episode.retirement.is_none() {
            return Ok(());
        }
        let core = self.core.as_mut().ok_or(Error::Closed)?;
        if let Some(stage) = core.inputs.stage.as_mut() {
            stage.close()?;
        }
        if let Err(error) = self
            .original_episode
            .retirement
            .take()
            .unwrap()
            .retire(core.inputs.stage.take())
        {
            self.original_episode.uncertain = true;
            return Err(error);
        }
        Ok(())
    }
}

impl CardInstanceNativeSession {
    pub(super) fn retire_original_on_drop(&mut self) {
        if self.close_original_episode().is_err() {
            if let Some(core) = self.core.as_mut() {
                if let Some(stage) = core.inputs.stage.take() {
                    std::mem::forget(stage);
                }
            }
            if let Some(retirement) = self.original_episode.retirement.take() {
                let _ = retirement.retire(None);
            }
        }
    }
}
