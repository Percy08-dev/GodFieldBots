//! Structural component-envelope tokens only. Python owns all external membership
//! and original-file/fit authority. This module never opens original sources.
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
    retirement::Retirement,
    structured_controls::checksum,
    structured_journal::{self, card_instance},
    structured_staging::{Banks, Stage},
    structured_tokens::Recipe,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
const LIMIT: usize = 64 * 1024 * 1024;
const ORIGIN: &str = "declared-component-fixture";
const LABEL: &str = "card-instance-declared-terminal-v1";
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::contract(message))
    }
}
fn name(value: &Value) -> Result<&str> {
    let text = value.string()?;
    require(
        !text.is_empty()
            && text.len() <= 101
            && text.as_bytes()[0].is_ascii_alphanumeric()
            && text
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        "closed fixture identifier required",
    )?;
    Ok(text)
}
fn label(outcome: &Value, viewer: i128) -> Result<f64> {
    if outcome.require("kind")?.string()? == "draw" {
        outcome.closed(&["kind"], &[])?;
        Ok(0.)
    } else {
        outcome.closed(&["kind", "winner"], &[])?;
        require(
            outcome.require("kind")?.string()? == "win",
            "declared fixture outcome required",
        )?;
        Ok(if outcome.require("winner")?.integer(1, 2)? == viewer {
            1.
        } else {
            -1.
        })
    }
}
struct Core {
    envelope: Option<Document>,
    metadata: Option<Document>,
    stage: Option<Stage>,
    _owner: Lease,
}
fn retire(core: Core) -> Result<()> {
    // Reportable spool close MUST have happened before this handoff.
    require(
        core.stage.is_none(),
        "fixture core retired with reachable spool",
    )?;
    drop(core);
    Ok(())
}
pub(crate) struct FixtureTokens {
    core: Option<Core>,
    retirement: Option<Retirement<Core>>,
    budget: Arc<Budget>,
    control: Control,
    failed: bool,
    exported: bool,
    closed: bool,
}
impl FixtureTokens {
    pub(crate) fn empty(budget: Arc<Budget>, control: Control) -> Result<Self> {
        let owner = budget.reserve(65536)?;
        let retirement = Retirement::start(&budget, retire)?;
        Ok(Self {
            core: Some(Core {
                envelope: None,
                metadata: None,
                stage: None,
                _owner: owner,
            }),
            retirement: Some(retirement),
            budget,
            control,
            failed: true,
            exported: false,
            closed: false,
        })
    }
    pub(crate) fn populate(
        &mut self,
        envelope: &[u8],
        columns: &[&[u8]],
        source: usize,
        record: usize,
        scratch: usize,
        chunk: usize,
        window: usize,
    ) -> Result<()> {
        require(
            self.failed
                && !self.closed
                && !self.exported
                && self.core.as_ref().is_some_and(|c| c.envelope.is_none()),
            "one fixture construction required",
        )?;
        self.control.check()?;
        require(
            source > 0
                && source <= LIMIT
                && record > 0
                && record <= source
                && scratch > 0
                && scratch <= 12usize << 30
                && chunk > 0
                && chunk <= LIMIT,
            "fixture resource settings bounds",
        )?;
        require(
            envelope.len() <= source && envelope.len() <= record,
            "fixture envelope one-record/source bound",
        )?;
        require(columns.len() <= 1024, "fixture decision capacity")?;
        let total = columns.iter().try_fold(envelope.len(), |n, v| {
            require(
                v.len() <= source && v.len() <= LIMIT,
                "fixture columns source bound",
            )?;
            n.checked_add(v.len())
                .ok_or_else(|| Error::resource("fixture input sum overflow"))
        })?;
        let _inputs = self.budget.reserve(
            total
                .checked_add(65536 + 128 * columns.len())
                .ok_or_else(|| Error::resource("fixture input metadata overflow"))?,
        )?;
        let core = self.core.as_mut().ok_or(Error::Closed)?;
        core.envelope = Some(json::parse(envelope, &self.budget, &self.control)?);
        let raw = &core.envelope.as_ref().unwrap().root;
        raw.closed(
            &[
                "schema_version",
                "kind",
                "fixture_id",
                "declaration_sha256",
                "population_id",
                "cohort_id",
                "episode",
                "viewer",
                "split",
                "generation",
                "roles",
                "selected_policy_id",
                "outcome",
                "outcome_origin",
                "events",
                "decisions",
            ],
            &[],
        )?;
        require(
            raw.require("schema_version")?.integer(1, 1)? == 1
                && raw.require("kind")?.string()? == "card-instance-component-original"
                && raw.require("outcome_origin")?.string()? == ORIGIN,
            "component envelope1 origin required",
        )?;
        for key in ["fixture_id", "population_id", "cohort_id", "episode"] {
            name(raw.require(key)?)?;
        }
        checksum(raw.require("declaration_sha256")?)?;
        let selected = checksum(raw.require("selected_policy_id")?)?;
        let viewer = raw.require("viewer")?.integer(1, 2)?;
        let generation = raw.require("generation")?.integer(0, 2)?;
        require(
            raw.require("split")?.string()?
                == if generation == 2 {
                    "validation"
                } else {
                    "train"
                },
            "fixture generation/split mismatch",
        )?;
        let roles = raw.require("roles")?.array()?;
        require(roles.len() == 2, "both fixture roles required")?;
        for (index, role) in roles.iter().enumerate() {
            role.closed(
                &["participant", "role", "policy_generation", "policy_id"],
                &[],
            )?;
            let actor = (index + 1) as i128;
            let own = actor == viewer;
            let g = role
                .require("policy_generation")?
                .integer(0, if own { 2 } else { 1 })?;
            let id = checksum(role.require("policy_id")?)?;
            require(
                role.require("participant")?.integer(actor, actor)? == actor
                    && role.require("role")?.string()?
                        == if own {
                            "declared-selected"
                        } else {
                            "declared-opponent"
                        },
                "fixture role/viewer mismatch",
            )?;
            if own {
                require(
                    g == generation && id == selected,
                    "fixture selected policy/generation mismatch",
                )?;
            }
        }
        let reward = label(raw.require("outcome")?, viewer)?;
        let events = raw.require("events")?.array()?;
        let decisions = raw.require("decisions")?.array()?;
        require(
            events.len() <= 262144 && decisions.len() <= 1024 && columns.len() == decisions.len(),
            "fixture ordered events/decisions/columns mismatch",
        )?;
        let binding = card_instance::component_binding(&self.budget, &self.control)?;
        let recipe = Recipe::corrected(window, &self.budget, &self.control)?;
        let fixed_signature = Recipe::corrected(128, &self.budget, &self.control)?.signature;
        let mut prefixes = BudgetVec::new(&self.budget);
        prefixes.push(structured_journal::event_prefix_initial(
            viewer,
            raw.require("episode")?,
            Some(binding.require("action_contract")?),
            &self.budget,
            &self.control,
        )?)?;
        for event in events {
            card_instance::event(event, &recipe, &self.budget, &self.control)?;
            prefixes.push(structured_journal::event_prefix_next(
                prefixes.last().unwrap(),
                event,
                &self.budget,
                &self.control,
            )?)?;
        }
        let _metadata = self.budget.reserve(65536 + 16384 * decisions.len())?;
        core.stage = Some(Stage::new(
            self.budget.clone(),
            scratch,
            chunk,
            self.control.clone(),
        )?);
        let stage = core.stage.as_mut().unwrap();
        let mut frames = Vec::with_capacity(decisions.len());
        let (mut previous_end, mut previous_revision) = (0usize, None);
        for (position, (decision, payload)) in decisions.iter().zip(columns).enumerate() {
            self.control.check()?;
            decision.closed(&["frame", "columns", "selected_index"], &[])?;
            let frame = decision.require("frame")?;
            card_instance::frame(frame, &self.budget, &self.control)?;
            let start = frame
                .require("interval_start")?
                .integer(0, events.len() as i128)? as usize;
            let end = frame
                .require("boundary")?
                .require("end")?
                .integer(start as i128, events.len() as i128)? as usize;
            let revision = frame.require("revision")?.integer(0, u64::MAX as i128)?;
            require(
                frame.require("actor")?.integer(1, 2)? == viewer
                    && frame.require("episode")? == raw.require("episode")?
                    && start == previous_end
                    && previous_revision.is_none_or(|v| revision > v)
                    && frame.require("interval_start_hash")?.string()?
                        == structured_journal::event_prefix_text(&prefixes[start])
                    && frame
                        .require("boundary")?
                        .require("prefix_hash")?
                        .string()?
                        == structured_journal::event_prefix_text(&prefixes[end]),
                "fixture own frame/revision/event prefix continuity mismatch",
            )?;
            let descriptor = decision.require("columns")?;
            require(
                descriptor.require("file")?.string()?
                    == format!("{}-d{position:04}.bin", name(raw.require("episode")?)?),
                "fixture ordered original column filename mismatch",
            )?;
            let table = card_instance::component_columns(
                payload,
                descriptor,
                frame,
                &self.budget,
                &self.control,
            )?;
            let index = decision
                .require("selected_index")?
                .integer(0, table.value.require("kinds")?.array()?.len() as i128 - 1)?;
            let staged = stage.fixture_frame(0, position, frame, &table.value, events, &recipe)?;
            let mut entry = json::object([
                ("decision", json::integer(position as i128)),
                ("selected_index", json::integer(index)),
                (
                    "columns_file",
                    json::copy(descriptor.require("file")?, &self.control)?,
                ),
                (
                    "columns_sha256",
                    json::copy(descriptor.require("sha256")?, &self.control)?,
                ),
            ]);
            for key in [
                "revision",
                "interval_start",
                "interval_start_hash",
                "boundary",
                "frame_hash",
                "table_hash",
                "token_prefix",
                "event_token_prefix",
            ] {
                entry
                    .as_object_mut()
                    .unwrap()
                    .insert(key.into(), json::copy(staged.require(key)?, &self.control)?);
            }
            frames.push(entry);
            previous_end = end;
            previous_revision = Some(revision);
        }
        require(
            previous_end == events.len(),
            "fixture contains future/unbound own event prefix",
        )?;
        let boundary = json::object([
            ("end", json::integer(events.len() as i128)),
            (
                "prefix_hash",
                json::text(&structured_journal::event_prefix_text(
                    prefixes.last().unwrap(),
                )),
            ),
        ]);
        let mut metadata = json::object([
            ("schema_version", json::integer(1)),
            ("kind", json::text("card-instance-component-token-payload")),
            (
                "envelope_sha256",
                json::text(&format!("{:x}", Sha256::digest(envelope))),
            ),
            ("label_recipe", json::text(LABEL)),
            ("label", json::float(reward)?),
            ("input_binding", binding),
            ("layout_signature", json::text(&fixed_signature)),
            ("aggregate_window", json::integer(window as i128)),
            ("token_signature", json::text(&recipe.signature)),
            ("boundary", boundary),
            ("decisions", json::array(frames)),
            ("arrays", stage.descriptors()?),
        ]);
        for key in [
            "fixture_id",
            "declaration_sha256",
            "population_id",
            "cohort_id",
            "episode",
            "viewer",
            "split",
            "generation",
            "roles",
            "selected_policy_id",
            "outcome",
            "outcome_origin",
        ] {
            metadata
                .as_object_mut()
                .unwrap()
                .insert(key.into(), json::copy(raw.require(key)?, &self.control)?);
        }
        let encoded = json::canonical(&metadata, &self.budget, &self.control)?;
        require(
            encoded.len() <= LIMIT,
            "fixture token metadata exceeds64MiB",
        )?;
        core.metadata = Some(json::parse(&encoded, &self.budget, &self.control)?);
        self.control.check()?;
        self.failed = false;
        Ok(())
    }
    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }
    pub(crate) fn export_preconditions(&self) -> Result<()> {
        require(
            !self.failed && !self.exported && !self.closed && self.core.is_some(),
            "fixture token export is closed or failed",
        )
    }
    pub(crate) fn export(&mut self) -> Result<(BudgetVec<u8>, Banks)> {
        self.export_preconditions()?;
        self.failed = true;
        self.control.check()?;
        let core = self.core.as_mut().ok_or(Error::Closed)?;
        let bytes = json::canonical(
            &core.metadata.as_ref().ok_or(Error::Closed)?.root,
            &self.budget,
            &self.control,
        )?;
        let banks = core.stage.as_mut().ok_or(Error::Closed)?.export()?;
        // Close/join are reportable and complete before any FFI arrays escape.
        self.retire_core()?;
        self.exported = true;
        Ok((bytes, banks))
    }
    fn retire_core(&mut self) -> Result<()> {
        if let Some(core) = self.core.as_mut() {
            if let Some(stage) = core.stage.as_mut() {
                stage.close_fixture()?;
            }
            core.stage.take();
        }
        let core = self.core.take();
        if let Some(retirement) = self.retirement.take() {
            retirement.retire(core)?;
        }
        Ok(())
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.failed = true;
        self.retire_core()?;
        self.closed = true;
        Ok(())
    }
}
impl Drop for FixtureTokens {
    fn drop(&mut self) {
        if self.close().is_err() {
            // A caller discarding its reachable cleanup owner cannot authorize
            // deletion of replaced/foreign evidence. Quarantine that bounded
            // spool owner; still join retirement of the parsed recursive core.
            if let Some(core) = self.core.as_mut() {
                if let Some(stage) = core.stage.take() {
                    std::mem::forget(stage);
                }
            }
            let _ = self.retire_core();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Arc<Budget>, Control, Document) {
        let b = Budget::new(1024 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 120.).unwrap();
        let doc = json::parse(
            include_bytes!("../tests/fixtures/card_instance_component_tokens_python_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        (b, c, doc)
    }
    fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
    fn payloads(case: &Value) -> Vec<Vec<u8>> {
        case["columns_hex"]
            .array()
            .unwrap()
            .iter()
            .map(|v| bytes(v.string().unwrap()))
            .collect()
    }
    fn start(case: &Value) -> (Arc<Budget>, FixtureTokens) {
        let b = Budget::new(512 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 60.).unwrap();
        let mut owner = FixtureTokens::empty(b.clone(), c).unwrap();
        let payloads = payloads(case);
        owner
            .populate(
                case["envelope_utf8"].string().unwrap().as_bytes(),
                &payloads.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                LIMIT,
                LIMIT,
                LIMIT,
                65536,
                2,
            )
            .unwrap();
        (b, owner)
    }
    fn reseal(frame: &mut Value, b: &Arc<Budget>, c: &Control) {
        let mut binding = frame.clone();
        binding.as_object_mut().unwrap().remove("frame_hash");
        let contract = card_instance::component_binding(b, c).unwrap()["action_contract"].clone();
        for key in ["input_recipe", "policy_input_version"] {
            binding
                .as_object_mut()
                .unwrap()
                .insert(key.into(), contract[key].clone());
        }
        binding
            .as_object_mut()
            .unwrap()
            .insert("action_contract".into(), contract);
        frame.as_object_mut().unwrap().insert(
            "frame_hash".into(),
            json::text(&crate::structured_controls::digest(&binding, b, c).unwrap()),
        );
    }
    #[test]
    fn card_instance_fixture_python_all_arrays_metadata_order_offsets_and_labels_are_exact() {
        let (_, _, all) = fixture();
        for case in all.root["cases"].array().unwrap() {
            let (b, mut owner) = start(case);
            let (metadata, banks) = owner.export().unwrap();
            assert!(owner.core.is_none() && owner.retirement.is_none());
            let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
            let actual = json::parse(&metadata, &b, &c).unwrap();
            assert_eq!(actual.root, case["metadata"], "{}", case["label"]);
            for (dtype, expected) in case["banks_hex"].object().unwrap() {
                let observed = match dtype.as_str() {
                    "uint8" => banks.bytes.iter().map(|v| v.to_vec()).collect::<Vec<_>>(),
                    "int64" => banks
                        .integers
                        .iter()
                        .map(|v| v.iter().flat_map(|n| n.to_le_bytes()).collect())
                        .collect(),
                    "float32" => banks
                        .floats
                        .iter()
                        .map(|v| v.iter().flat_map(|n| n.to_le_bytes()).collect())
                        .collect(),
                    _ => panic!("unknown oracle dtype"),
                };
                let expected = expected
                    .array()
                    .unwrap()
                    .iter()
                    .map(|v| bytes(v.string().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(observed, expected, "{} {dtype}", case["label"]);
            }
            assert!(owner.export().is_err());
            owner.close().unwrap();
            assert!(b.used() > 0); // Returned banks/metadata retain their own RAM leases.
            drop(actual);
            drop(metadata);
            drop(banks);
            drop(owner);
            assert_eq!(b.used(), 0);
        }
    }
    #[test]
    fn card_instance_fixture_closed_malformed_context_roles_prefix_and_codec_refuse() {
        let (oracle_b, c, all) = fixture();
        let case = &all.root["cases"][0];
        let payloads = payloads(case);
        for mutation in 0..25 {
            let mut envelope = case["envelope"].clone();
            match mutation {
                0 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("schema_version".into(), json::integer(6));
                }
                1 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("kind".into(), json::text("card-instance-source-controls"));
                }
                2 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("kind".into(), json::text("header"));
                }
                3 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("observation".into(), json::object([]));
                }
                4 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("outcome_origin".into(), json::text("observed-natural"));
                }
                5 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("fixture_id".into(), json::text("../foreign"));
                }
                6 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("declaration_sha256".into(), json::text("bad"));
                }
                7 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("viewer".into(), Value::Bool(true));
                }
                8 => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("split".into(), json::text("validation"));
                }
                9 => {
                    envelope["roles"][0]
                        .as_object_mut()
                        .unwrap()
                        .insert("participant".into(), json::integer(2));
                }
                10 => {
                    envelope["roles"][0]
                        .as_object_mut()
                        .unwrap()
                        .insert("role".into(), json::text("declared-opponent"));
                }
                11 => {
                    envelope["roles"][0]
                        .as_object_mut()
                        .unwrap()
                        .insert("policy_generation".into(), json::integer(1));
                }
                12 => {
                    envelope["roles"][0]
                        .as_object_mut()
                        .unwrap()
                        .insert("policy_id".into(), json::text(&"f".repeat(64)));
                }
                13 => {
                    envelope["outcome"]
                        .as_object_mut()
                        .unwrap()
                        .insert("winner".into(), json::integer(0));
                }
                14 => {
                    envelope["decisions"][0]
                        .as_object_mut()
                        .unwrap()
                        .insert("selected_index".into(), json::integer(65536));
                }
                15 => {
                    envelope["decisions"][0]["columns"]
                        .as_object_mut()
                        .unwrap()
                        .insert("purpose".into(), json::text("qualified_game"));
                }
                16 => {
                    envelope["decisions"][0]["columns"]
                        .as_object_mut()
                        .unwrap()
                        .insert("schema_version".into(), json::integer(1));
                }
                17 => {
                    envelope["decisions"][0]["columns"]
                        .as_object_mut()
                        .unwrap()
                        .insert("codebook_sha256".into(), json::text(&"0".repeat(64)));
                }
                18 => {
                    envelope["decisions"][0]["columns"]
                        .as_object_mut()
                        .unwrap()
                        .insert("file".into(), json::text("foreign.bin"));
                }
                19 => {
                    envelope["decisions"][0]["columns"]
                        .as_object_mut()
                        .unwrap()
                        .insert("synthetic_capacity".into(), Value::Bool(true));
                }
                20 => {
                    envelope["decisions"][0]["frame"]["boundary"]
                        .as_object_mut()
                        .unwrap()
                        .insert("prefix_hash".into(), json::text(&"0".repeat(64)));
                    reseal(&mut envelope["decisions"][0]["frame"], &oracle_b, &c);
                }
                21 => {
                    envelope["decisions"][0]["frame"]
                        .as_object_mut()
                        .unwrap()
                        .insert("interval_start_hash".into(), json::text(&"0".repeat(64)));
                    reseal(&mut envelope["decisions"][0]["frame"], &oracle_b, &c);
                }
                22 => {
                    let event = envelope["events"][0].clone();
                    envelope["events"].as_array_mut().unwrap().push(event);
                }
                23 => {
                    envelope["decisions"][0]["columns"]
                        .as_object_mut()
                        .unwrap()
                        .remove("codebook_sha256");
                }
                _ => {
                    envelope
                        .as_object_mut()
                        .unwrap()
                        .insert("GameSeed".into(), json::integer(17));
                }
            }
            let raw = json::canonical(&envelope, &oracle_b, &c).unwrap();
            let b = Budget::new(512 * 1024 * 1024);
            let mut owner = FixtureTokens::empty(b.clone(), c.clone()).unwrap();
            assert!(
                owner
                    .populate(
                        &raw,
                        &payloads.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                        LIMIT,
                        LIMIT,
                        LIMIT,
                        65536,
                        2
                    )
                    .is_err(),
                "mutation {mutation}"
            );
            assert!(owner.export().is_err());
            owner.close().unwrap();
            drop(owner);
            assert_eq!(b.used(), 0, "mutation {mutation}");
        }
    }
    #[test]
    fn card_instance_fixture_record_source_budget_and_cancel_are_terminal() {
        let (_, _, all) = fixture();
        let case = &all.root["cases"][0];
        let payloads = payloads(case);
        let raw = case["envelope_utf8"].string().unwrap().as_bytes();
        for mode in 0..5 {
            let b = Budget::new(if mode == 3 {
                9 * 1024 * 1024
            } else {
                512 * 1024 * 1024
            });
            let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
            let mut owner = FixtureTokens::empty(b.clone(), c.clone()).unwrap();
            if mode == 4 {
                c.cancel();
            }
            let input = if mode == 2 {
                vec![]
            } else {
                payloads.iter().map(Vec::as_slice).collect()
            };
            assert!(
                owner
                    .populate(
                        raw,
                        &input,
                        if mode == 0 { 1 } else { LIMIT },
                        if mode <= 1 { 1 } else { LIMIT },
                        LIMIT,
                        65536,
                        2
                    )
                    .is_err()
            );
            assert!(owner.export().is_err());
            owner.close().unwrap();
            drop(owner);
            assert_eq!(b.used(), 0);
        }
        let (b, mut owner) = start(case);
        owner.control.cancel();
        assert_eq!(owner.export().err().unwrap(), Error::Cancelled);
        assert!(owner.export().is_err());
        owner.close().unwrap();
        drop(owner);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn card_instance_fixture_partial_constructor_stage_has_confirmed_cleanup() {
        let (oracle, c, all) = fixture();
        let case = all.root["cases"]
            .array()
            .unwrap()
            .iter()
            .find(|v| v["envelope"]["decisions"].array().unwrap().len() == 3)
            .unwrap();
        let mut envelope = case["envelope"].clone();
        envelope["decisions"][1]
            .as_object_mut()
            .unwrap()
            .insert("selected_index".into(), json::integer(999999));
        let raw = json::canonical(&envelope, &oracle, &c).unwrap();
        let payloads = payloads(case);
        let b = Budget::new(512 * 1024 * 1024);
        let mut owner = FixtureTokens::empty(b.clone(), c).unwrap();
        assert!(
            owner
                .populate(
                    &raw,
                    &payloads.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    LIMIT,
                    LIMIT,
                    LIMIT,
                    65536,
                    2
                )
                .is_err()
        );
        let directory = owner
            .core
            .as_ref()
            .unwrap()
            .stage
            .as_ref()
            .unwrap()
            .fixture_test_directory()
            .to_owned();
        assert!(directory.exists());
        assert!(owner.export().is_err());
        owner.close().unwrap();
        assert!(!directory.exists());
        assert!(owner.retirement.is_none());
        drop(owner);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn card_instance_fixture_failed_export_preserves_foreign_evidence_and_repair_cannot_retry() {
        let (_, _, all) = fixture();
        let (b, mut owner) = start(&all.root["cases"][0]);
        let directory = owner
            .core
            .as_ref()
            .unwrap()
            .stage
            .as_ref()
            .unwrap()
            .fixture_test_directory()
            .to_owned();
        let foreign = directory.join("foreign-evidence");
        std::fs::write(&foreign, b"preserve").unwrap();
        assert!(owner.export().is_err());
        assert_eq!(std::fs::read(&foreign).unwrap(), b"preserve");
        match owner.close().unwrap_err() {
            Error::Cleanup { cleanup, .. } => {
                assert!(cleanup.contains(directory.to_str().unwrap()))
            }
            error => panic!("retained cleanup location absent: {error:?}"),
        }
        assert!(owner.core.is_some() && owner.retirement.is_some());
        std::fs::remove_file(&foreign).unwrap();
        assert!(owner.export().is_err());
        owner.close().unwrap();
        assert!(!directory.exists());
        drop(owner);
        assert_eq!(b.used(), 0);
    }
}
