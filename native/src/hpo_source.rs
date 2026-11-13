//! HPO facts and authorization on the existing OriginalInputs owner.
//! Pending facts never contain a final Proof3 or update correspondence.
use super::*;
use crate::structured_controls::hpo as metadata;

pub(super) fn original_facts_equal(
    actual: &Value,
    saved: &Value,
    packed: bool,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<bool> {
    saved.require("schema_version")?.integer(3, 3)?;
    require(
        field(saved, "kind")?
            == if packed {
                "card-instance-hpo-packed-original-source-proof"
            } else {
                "card-instance-hpo-original-source-proof"
            },
        "explicit HPO original proof3 required",
    )?;
    // Project only the unchanged original/physical evidence to the shared pure
    // custody comparator. This is no old-format loader or Source permission.
    let mut facts = json::copy(saved, c)?;
    facts.as_object_mut().unwrap().remove("selection");
    facts
        .as_object_mut()
        .unwrap()
        .remove("expected_optimizer_step_delta");
    facts["schema_version"] = json::integer(if packed { 2 } else { 1 });
    facts["kind"] = json::text(if packed {
        "card-instance-packed-original-source-proof"
    } else {
        "card-instance-original-source-proof"
    });
    if packed {
        crate::logical_artifacts::same_proof(actual, &facts, b, c)
    } else {
        Ok(actual == &facts)
    }
}

pub(super) fn pending(inputs: &mut OriginalInputs) -> Result<Controls> {
    inputs.recheck()?;
    require(
        inputs.format.hpo() && inputs.parents.len() == 1,
        "one explicit HPO original parent required",
    )?;
    let b = &inputs.budget;
    let c = &inputs.control;
    let parent = &inputs.parents[0];
    let state = &parent.state.root;
    let records = parent.summary.root.require("generations")?.array()?;
    let mut copy_bytes = json::canonical(&parent.summary.root, b, c)?.len() + 65536;
    let control_bytes = json::canonical(state.require("identity")?, b, c)?.len()
        + json::canonical(state.require("execution")?, b, c)?.len()
        + 65536;
    for source in inputs.collections.iter() {
        let proof_bytes = json::canonical(&source.proof.root, b, c)?.len();
        copy_bytes = copy_bytes
            .checked_add(json::canonical(&source.manifest.root, b, c)?.len())
            .and_then(|n| {
                n.checked_add(
                    control_bytes + source.digests.len() * 256 + source.learner_counts.len() * 32,
                )
            })
            .and_then(|n| n.checked_add(proof_bytes))
            .ok_or_else(|| Error::resource("HPO pending control-copy admission overflow"))?;
    }
    let _copies = json::admit(copy_bytes, b)?;
    let mut immutable = json::object([]);
    for number in 0..records.len() {
        let relative = format!("generations/{number:06}.json");
        immutable.as_object_mut().unwrap().insert(
            relative.clone(),
            json::text(
                parent
                    .sources
                    .verified_sha(&parent.directory.join(relative))?,
            ),
        );
    }
    let ack = json::object([
        (
            "run_sha256",
            json::text(
                parent
                    .sources
                    .verified_sha(&parent.directory.join("run.json"))?,
            ),
        ),
        ("config_sha256", state.require("config_sha256")?.clone()),
        ("resume_identity", state.require("resume_identity")?.clone()),
        (
            "generation_files_sha256",
            json::text(&digest(&immutable, b, c)?),
        ),
        (
            "registry_sha256",
            json::text(
                parent
                    .sources
                    .verified_sha(&parent.directory.join("artifact-registry.json"))?,
            ),
        ),
    ]);
    let mut collections = Vec::new();
    let mut declarations = Vec::new();
    let mut authorization_bound = 65536usize;
    for (index, source) in inputs.collections.iter().enumerate() {
        c.check()?;
        let manifest = &source.manifest.root;
        let link = manifest.require("linkage")?;
        let remaining = integer(
            link.require("selection_context")?,
            "remaining",
            0,
            i64::MAX as i128,
        )?;
        authorization_bound = authorization_bound
            .checked_add(65536)
            .and_then(|n| n.checked_add(source.learner_counts.len().checked_mul(128)?))
            .and_then(|n| n.checked_add(source.accepted_learner.min(remaining).checked_mul(16)?))
            .ok_or_else(|| Error::resource("HPO authorization bound overflow"))?;
        let generation = integer(link, "collection_generation", 0, records.len() as i128 - 1)?;
        let record = &records[generation];
        let declaration = json::copy(parent.summary.root.require("semantics")?, c)?;
        let mut files = json::object([]);
        let generation_name = format!("generations/{generation:06}.json");
        files.as_object_mut().unwrap().insert(
            generation_name.clone(),
            json::text(
                parent
                    .sources
                    .verified_sha(&parent.directory.join(generation_name))?,
            ),
        );
        files.as_object_mut().unwrap().insert(
            format!("artifacts/{generation:06}/inference.json"),
            record.require("inference_sha256")?.clone(),
        );
        let committed = inputs
            .committed
            .iter()
            .find(|row| row.collection_index == index);
        let mut control = json::object([
            ("run_id", state.require("run_id")?.clone()),
            ("config_sha256", state.require("config_sha256")?.clone()),
            ("identity", state.require("identity")?.clone()),
            ("execution", state.require("execution")?.clone()),
            ("plan", link.clone()),
            ("files", files),
            (
                "source_set_sha256",
                source.proof.root.require("source_set_sha256")?.clone(),
            ),
            ("recipe", json::text("card-instance-hpo-source-controls-v1")),
            ("hpo_spec_sha256", state.require("hpo_spec_sha256")?.clone()),
        ]);
        if inputs.format.packed() {
            let original = committed
                .map(|row| row.update.root.require("source_proof"))
                .transpose()?
                .unwrap_or(&source.proof.root);
            for key in ["storage", "custody"] {
                control
                    .as_object_mut()
                    .unwrap()
                    .insert(key.into(), original.require(key)?.clone());
            }
        }
        let numeric = if let Some(row) = committed {
            json::object([
                (
                    "directory",
                    json::text(
                        owned_relative(&parent.directory, row.update.root.require("numeric")?)?
                            .to_str()
                            .ok_or_else(|| Error::contract("HPO UTF-8 numeric path required"))?,
                    ),
                ),
                ("sha256", row.update.root.require("numeric_sha256")?.clone()),
                ("files", row.update.root.require("numeric_files")?.clone()),
            ])
        } else {
            Value::Null
        };
        collections.push(json::object([
            ("collection_index", json::integer(index as i128)),
            (
                "directory",
                json::text(
                    source
                        .directory
                        .to_str()
                        .ok_or_else(|| Error::contract("HPO UTF-8 collection path required"))?,
                ),
            ),
            ("generation", json::integer(generation as i128)),
            (
                "manifest_sha256",
                source.proof.root.require("manifest_sha256")?.clone(),
            ),
            ("config_sha256", manifest.require("config_hash")?.clone()),
            ("linkage", link.clone()),
            ("schedule_sha256", json::text(&source.schedule_sha256)),
            ("bindings_sha256", json::text(&source.bindings_sha256)),
            ("original_proof", source.proof.root.clone()),
            ("semantics", declaration.clone()),
            (
                "learner_label",
                json::text(
                    source
                        .learner_label
                        .as_deref()
                        .ok_or_else(|| Error::contract("HPO original learner identity missing"))?,
                ),
            ),
            (
                "learner_decisions",
                json::integer(source.accepted_learner as i128),
            ),
            (
                "episode_counts",
                json::array(
                    source
                        .learner_counts
                        .iter()
                        .map(|v| json::integer(*v as i128)),
                ),
            ),
            (
                "episode_digests",
                json::array(source.digests.iter().map(|v| json::text(v))),
            ),
            ("control", control),
            ("numeric", numeric),
        ]));
        declarations.push(Some(doc(&declaration, b, c)?));
    }
    let authorization = doc(
        &json::object([("schema_version", json::integer(5)), ("parent", ack)]),
        b,
        c,
    )?;
    let mut summary = json::object([
        ("schema_version", json::integer(5)),
        ("kind", json::text("card-instance-hpo-pending-controls")),
        ("reader_contract", json::text("participant-structured-v2")),
        (
            "mode",
            json::text(match inputs.mode {
                ScanMode::Historical => "historical_parent",
                ScanMode::Live { .. } => "live_cohort",
            }),
        ),
        ("parent", parent.summary.root.clone()),
        ("collections", json::array(collections)),
        (
            "authorization_max_bytes",
            json::integer(authorization_bound.min(64usize << 20) as i128),
        ),
        ("qualified", Value::Bool(false)),
        ("parent_qualification", json::text("not_performed")),
    ]);
    for key in [
        "action_contract",
        "purpose",
        "synthetic_capacity",
        "rule_signature",
        "layout_signature",
    ] {
        summary
            .as_object_mut()
            .unwrap()
            .insert(key.into(), state.require(key)?.clone());
    }
    if inputs.format.packed() {
        for key in ["storage_binding", "artifact_registry_sha256"] {
            summary
                .as_object_mut()
                .unwrap()
                .insert(key.into(), inputs._plan.root.require(key)?.clone());
        }
    }
    inputs.recheck()?;
    let fingerprint = digest(&summary, b, c)?;
    Ok(Controls {
        summary: doc(&summary, b, c)?,
        authorization,
        declarations,
        correspondences: (0..inputs.collections.len()).map(|_| None).collect(),
        fingerprint,
    })
}

pub(super) fn authorize(
    inputs: &OriginalInputs,
    pending: &Controls,
    request: &Value,
) -> Result<(Controls, Vec<Document>)> {
    inputs.recheck()?;
    require(
        inputs.format.hpo()
            && field(&pending.summary.root, "kind")? == "card-instance-hpo-pending-controls",
        "HPO pending original controls required",
    )?;
    let b = &inputs.budget;
    let c = &inputs.control;
    request.closed(
        &["schema_version", "pending_sha256", "parent", "collections"],
        &[],
    )?;
    request.require("schema_version")?.integer(5, 5)?;
    require(
        field(request, "pending_sha256")? == pending.fingerprint,
        "stale HPO pending original fingerprint",
    )?;
    require(
        request.require("parent")? == pending.authorization.root.require("parent")?,
        "HPO actual sealed parent authorization differs",
    )?;
    let supplied = request.require("collections")?.array()?;
    let originals = pending.summary.root.require("collections")?.array()?;
    require(
        supplied.len() == originals.len() && supplied.len() == inputs.collections.len(),
        "HPO authorization must preserve every original collection",
    )?;
    let pending_bytes = json::canonical(&pending.summary.root, b, c)?.len();
    let bytes = json::canonical(request, b, c)?
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(pending_bytes.checked_mul(2)?))
        .ok_or_else(|| Error::resource("HPO final control-copy bound overflow"))?;
    let _copies = json::admit(bytes, b)?;
    let recipe = inputs.parents[0]
        .state
        .root
        .require("effective_config")?
        .require("hpo_spec")?
        .require("training_recipe")?;
    metadata::training(recipe)?;
    let mut collections = Vec::new();
    let mut proofs = Vec::new();
    let mut correspondences = Vec::new();
    for (index, (expected, raw)) in originals.iter().zip(supplied).enumerate() {
        c.check()?;
        raw.closed(
            &[
                "collection_index",
                "selection",
                "training_recipe",
                "expected_optimizer_step_delta",
                "expected_correspondence",
            ],
            &[],
        )?;
        raw.require("collection_index")?
            .integer(index as i128, index as i128)?;
        require(
            raw.require("training_recipe")? == recipe,
            "HPO full recipe authorization differs from actual original parent",
        )?;
        let selected = raw.require("selection")?;
        let source = &inputs.collections[index];
        let used = metadata::selection(
            selected,
            &source.learner_counts,
            expected.require("linkage")?.require("selection_context")?,
            b,
            c,
        )?;
        let delta = metadata::optimizer_delta(recipe, used)?;
        raw.require("expected_optimizer_step_delta")?
            .integer(delta, delta)?;
        if let Some(saved) = inputs
            .committed
            .iter()
            .find(|row| row.collection_index == index)
        {
            require(
                saved
                    .update
                    .root
                    .require("source_proof")?
                    .require("selection")?
                    == selected
                    && saved
                        .update
                        .root
                        .require("expected_optimizer_step_delta")?
                        .integer(delta, delta)?
                        == delta,
                "HPO selected endpoints differ from immutable committed evidence",
            )?;
        }
        let mut control = json::copy(expected.require("control")?, c)?;
        control
            .as_object_mut()
            .unwrap()
            .insert("selection".into(), json::copy(selected, c)?);
        let fingerprint = digest(&control, b, c)?;
        let mut correspondence = cohort_binding(
            &fingerprint,
            expected.require("semantics")?,
            &source.digests,
            b,
            c,
        )?;
        correspondence["recipe"] = json::text(if inputs.format.packed() {
            "card-instance-hpo-packed-source-correspondence-v1"
        } else {
            "card-instance-hpo-source-correspondence-v1"
        });
        correspondence
            .as_object_mut()
            .unwrap()
            .remove("binding_sha256");
        correspondence["binding_sha256"] = json::text(&digest(&correspondence, b, c)?);
        require(
            raw.require("expected_correspondence")? == &correspondence,
            "HPO final original/selection correspondence differs from independently checked Python expectation",
        )?;
        let mut proof = json::copy(&source.proof.root, c)?;
        proof["schema_version"] = json::integer(3);
        proof["kind"] = json::text(if inputs.format.packed() {
            "card-instance-hpo-packed-original-source-proof"
        } else {
            "card-instance-hpo-original-source-proof"
        });
        proof
            .as_object_mut()
            .unwrap()
            .insert("selection".into(), json::copy(selected, c)?);
        proof
            .as_object_mut()
            .unwrap()
            .insert("expected_optimizer_step_delta".into(), json::integer(delta));
        let mut item = json::copy(expected, c)?;
        for key in [
            "original_proof",
            "episode_counts",
            "episode_digests",
            "control",
        ] {
            item.as_object_mut().unwrap().remove(key);
        }
        item.as_object_mut()
            .unwrap()
            .insert("source_proof".into(), proof.clone());
        item.as_object_mut()
            .unwrap()
            .insert("selection".into(), json::copy(selected, c)?);
        item.as_object_mut()
            .unwrap()
            .insert("correspondence".into(), correspondence.clone());
        proofs.push(doc(&proof, b, c)?);
        correspondences.push(Some(doc(&correspondence, b, c)?));
        collections.push(item);
    }
    let mut summary = json::copy(&pending.summary.root, c)?;
    summary["kind"] = json::text("card-instance-hpo-source-controls");
    summary["collections"] = json::array(collections);
    summary
        .as_object_mut()
        .unwrap()
        .insert("pending_sha256".into(), json::text(&pending.fingerprint));
    let fingerprint = digest(&summary, b, c)?;
    let declarations = pending
        .declarations
        .iter()
        .map(|entry| {
            entry
                .as_ref()
                .map(|value| doc(&value.root, b, c))
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;
    let final_controls = Controls {
        summary: doc(&summary, b, c)?,
        authorization: doc(request, b, c)?,
        declarations,
        correspondences,
        fingerprint,
    };
    inputs.recheck()?;
    Ok((final_controls, proofs))
}
