//! Closed final64 collection loan on the SAME retained corrected parent.
//! Structural evidence only. No scheduler, eligibility, numeric or update authority.
use super::*;
use crate::structured_controls::{checksum, digest};
use crate::structured_source::{CollectionLimits, CollectionScan, ScanMode};
use crate::structured_staging::Stage;

fn require(ok: bool, text: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::contract(text))
    }
}
fn field<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v.require(k)?.string()
}
fn nonnegative(v: &Value) -> Result<f64> {
    let n = v.number()?;
    require(
        n.is_finite() && n >= 0.,
        "finite nonnegative final64 clock required",
    )?;
    Ok(n)
}
fn copy(v: &Value, c: &Control) -> Result<Value> {
    json::copy(v, c)
}

/// Historical clocks describe the original execution, never the active Control.
pub(crate) fn invocation(value: &Value, run: &Value) -> Result<f64> {
    value.closed(
        &[
            "started_unix",
            "started_monotonic",
            "caller_deadline_monotonic",
            "admission_deadline_monotonic",
            "execution_deadline_monotonic",
            "wall_deadline_monotonic",
        ],
        &[],
    )?;
    for (key, n) in [
        ("wall_seconds", 7200),
        ("admission_seconds", 5400),
        ("execution_seconds", 7140),
        ("finalization_seconds", 60),
    ] {
        run.require(key)?.integer(n, n)?;
    }
    nonnegative(value.require("started_unix")?)?;
    use crate::structured_controls::formal::arithmetic;
    let start = value.require("started_monotonic")?;
    let caller = value.require("caller_deadline_monotonic")?;
    nonnegative(start)?;
    nonnegative(caller)?;
    let min = |a: Value, b: Value| -> Result<Value> {
        Ok(if a.number()? <= b.number()? { a } else { b })
    };
    let wall = min(
        caller.clone(),
        arithmetic(start, &json::integer(7200), false)?,
    )?;
    let execution = min(
        arithmetic(start, &json::integer(7140), false)?,
        arithmetic(&wall, &json::integer(60), true)?,
    )?;
    require(
        execution.number()? > start.number()?,
        "final64 invocation cannot preserve finalization reserve",
    )?;
    let admission = min(
        arithmetic(start, &json::integer(5400), false)?,
        execution.clone(),
    )?;
    for (key, expected) in [
        ("admission_deadline_monotonic", admission),
        ("execution_deadline_monotonic", execution),
        ("wall_deadline_monotonic", wall.clone()),
    ] {
        let actual = value.require(key)?;
        let integer = |v: &Value| v.as_number().is_some_and(json::integer_token);
        require(
            integer(actual) == integer(&expected)
                && if integer(actual) {
                    actual.integer(i128::MIN, i128::MAX)?
                        == expected.integer(i128::MIN, i128::MAX)?
                } else {
                    actual.number()? == expected.number()?
                },
            "final64 original invocation arithmetic/type differs",
        )?;
    }
    Ok(arithmetic(&wall, start, true)?.number()?)
}

pub(crate) fn manifest_controls(
    m: &Value,
    parent_root: &Path,
    state: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    require(
        ["formal-parent-v1", "formal-packed-parent-v1"].contains(&field(state, "run_variant")?)
            && field(state, "status")? == "complete",
        "complete retained formal parent required for final64",
    )?;
    let updates = state.require("updates")?.integer(1, 1000)?;
    let link = m.require("linkage")?;
    link.closed(
        &[
            "parent_run_id",
            "parent_run_path",
            "batch_id",
            "final_generation",
            "final_inference_sha256",
            "policy_label",
            "learner_seats",
            "schedule_digest",
            "policy_registry_digest",
        ],
        &[],
    )?;
    require(
        link.require("parent_run_id")? == state.require("run_id")?
            && field(link, "parent_run_path")? == parent_root.to_str().ok_or(Error::Closed)?
            && field(link, "batch_id")? == "validation-final"
            && link.require("final_generation")?.integer(0, 1000)? == updates,
        "final64 original parent/generation linkage differs",
    )?;
    checksum(link.require("final_inference_sha256")?)?;
    require(
        !field(link, "policy_label")?.is_empty()
            && field(link, "schedule_digest")? == digest(m.require("schedule")?, b, c)?
            && field(link, "policy_registry_digest")?
                == digest(m.require("policy_bindings")?, b, c)?,
        "final64 original registry/schedule linkage differs",
    )?;
    let cfg = m.require("config")?;
    let original = state.require("effective_config")?;
    let _projection_copy = json::admit(
        json::canonical(original, b, c)?
            .len()
            .checked_add(65536)
            .ok_or_else(|| Error::resource("final64 Config projection copy overflow"))?,
        b,
    )?;
    let mut projected = copy(original, c)?;
    require(
        field(original.require("run")?, "split")? == "train",
        "final64 original training split required",
    )?;
    projected
        .get_mut("run")
        .ok_or(Error::Closed)?
        .as_object_mut()
        .ok_or(Error::Closed)?
        .insert("split".into(), json::text("validation"));
    require(
        &projected == cfg,
        "final64 may project ONLY original run.split",
    )?;
    let projection = m.require("config_projection")?;
    projection.closed(
        &[
            "recipe",
            "parent_config_sha256",
            "parent_config_inputs_sha256",
            "projected_config_sha256",
        ],
        &[],
    )?;
    require(
        field(projection, "recipe")? == "run-split-train-to-validation-v1"
            && field(projection, "parent_config_sha256")? == digest(original, b, c)?
            && field(projection, "parent_config_inputs_sha256")?
                == digest(state.require("config_inputs")?, b, c)?
            && field(projection, "projected_config_sha256")? == digest(cfg, b, c)?,
        "final64 original projection/input origins differ",
    )?;
    let duration = invocation(m.require("validation_invocation")?, cfg.require("run")?)?;
    require(
        nonnegative(m.require("elapsed_seconds")?)? <= duration,
        "final64 elapsed exceeds recorded invocation",
    )?;
    Ok(())
}

fn manifest_kind(request: &Value) -> Result<bool> {
    let mode = request.require("manifest")?;
    match field(mode, "kind")? {
        "canonical" | "packed-canonical" => {
            mode.closed(&["kind"], &[])?;
            Ok(false)
        }
        "issued-pending" | "packed-issued-pending" => {
            mode.closed(&["kind", "evidence"], &[])?;
            Ok(true)
        }
        _ => Err(Error::contract(
            "closed canonical/issued-pending final64 manifest required",
        )),
    }
}
fn pending_evidence(
    request: &Value,
    root: &Path,
    registry: Option<&crate::logical_artifacts::Registry>,
) -> Result<()> {
    if let Some(registry) = registry {
        let source = registry
            .roots
            .get(root)
            .ok_or_else(|| Error::contract("final registry has no SAME original root"))?;
        let custody = source.description.root.require("custody")?;
        let pending = manifest_kind(request)?;
        if pending {
            require(
                request.require("manifest")?.require("evidence")? == custody,
                "pending final custody differs from actual provider originals",
            )?;
        }
        let rows = custody.require("files")?.array()?;
        let locator = rows
            .iter()
            .find(|v| v.get("role").and_then(Value::as_str) == Some("locator"))
            .ok_or(Error::Closed)?;
        let descriptor = rows
            .iter()
            .find(|v| v.get("role").and_then(Value::as_str) == Some("descriptor"))
            .ok_or(Error::Closed)?;
        require(
            Path::new(field(locator, "path")?)
                == root.join(if pending {
                    "collection.locator.json.tmp"
                } else {
                    "collection.locator.json"
                })
                && Path::new(field(descriptor, "path")?)
                    .file_name()
                    .and_then(|v| v.to_str())
                    == Some(if pending {
                        "collection.pending.json"
                    } else {
                        "collection.published.json"
                    }),
            "packed final original locator/descriptor subtype differs",
        )?;
        if pending {
            for path in [
                root.join("collection.locator.json"),
                Path::new(field(descriptor, "path")?).with_file_name("collection.published.json"),
            ] {
                match fs::symlink_metadata(path) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    _ => {
                        return Err(Error::contract(
                            "pending final public originals must remain absent",
                        ));
                    }
                }
            }
        }
        return Ok(());
    }
    if !manifest_kind(request)? {
        return Ok(());
    }
    match fs::symlink_metadata(root.join("manifest.json")) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => {
            return Err(Error::contract(
                "canonical final64 manifest must remain absent",
            ));
        }
    }
    let ev = request.require("manifest")?.require("evidence")?;
    ev.closed(
        &["directory_identity", "file_identity", "bytes", "sha256"],
        &[],
    )?;
    let dir = ev.require("directory_identity")?;
    dir.closed(&["device", "inode"], &[])?;
    let info = fs::symlink_metadata(root)?;
    require(
        info.is_dir()
            && !info.file_type().is_symlink()
            && dir.require("device")?.integer(0, u64::MAX as i128)? == info.dev() as i128
            && dir.require("inode")?.integer(0, u64::MAX as i128)? == info.ino() as i128,
        "pending final64 directory identity differs",
    )?;
    let file = ev.require("file_identity")?;
    file.closed(
        &["device", "inode", "mode", "size", "mtime_ns", "ctime_ns"],
        &[],
    )?;
    let path = root.join("manifest.json.tmp");
    // A new independent read FD: never a publisher descriptor or caller-selected name.
    let fd = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&path)?;
    let info = fd.metadata()?;
    require(
        info.is_file() && Identity::of(&info) == Identity::of(&fs::symlink_metadata(path)?),
        "pending final64 FD/path identity differs",
    )?;
    for (k, v) in [
        ("device", info.dev() as i128),
        ("inode", info.ino() as i128),
        ("mode", info.mode() as i128),
        ("size", info.len() as i128),
        (
            "mtime_ns",
            info.mtime() as i128 * 1_000_000_000 + info.mtime_nsec() as i128,
        ),
        (
            "ctime_ns",
            info.ctime() as i128 * 1_000_000_000 + info.ctime_nsec() as i128,
        ),
    ] {
        require(
            file.require(k)?.integer(i128::MIN, i128::MAX)? == v,
            "pending final64 original file stat differs",
        )?;
    }
    require(
        ev.require("bytes")? == file.require("size")?,
        "pending final64 byte count differs",
    )?;
    checksum(ev.require("sha256")?)?;
    // The Sources read below hashes/checks this SAME evidence before decode.
    Ok(())
}

pub(super) struct Slot {
    request: Option<Document>,
    registry: Option<Arc<crate::logical_artifacts::Registry>>,
    scan: Option<CollectionScan>,
    summary: Option<Document>,
    stage: Option<Stage>,
    retirement: Option<crate::retirement::Retirement<Stage>>,
    packet: std::sync::Weak<()>,
    next: usize,
    failure: Option<Error>,
    closed: bool,
    uncertain: bool,
    python_control: Option<Lease>,
    working: bool,
}
impl Slot {
    fn empty() -> Self {
        Self {
            request: None,
            registry: None,
            scan: None,
            summary: None,
            stage: None,
            retirement: None,
            packet: std::sync::Weak::new(),
            next: 0,
            failure: None,
            closed: false,
            uncertain: false,
            python_control: None,
            working: false,
        }
    }
    fn active(&self) -> Result<()> {
        if let Some(e) = &self.failure {
            return Err(e.clone());
        }
        require(
            !self.closed && !self.working && self.scan.is_some(),
            "active retained final64 slot required",
        )
    }
    fn remember<T>(&mut self, result: Result<T>) -> Result<T> {
        match result {
            Ok(v) => {
                self.working = false;
                Ok(v)
            }
            Err(e) => {
                if self.failure.is_none() {
                    self.failure = Some(e.clone())
                }
                Err(self.failure.as_ref().unwrap().clone())
            }
        }
    }
    fn recheck(&self, core: &CohortSession) -> Result<()> {
        core.check()?;
        core.inputs.recheck()?;
        let request = &self.request.as_ref().ok_or(Error::Closed)?.root;
        let scan = self.scan.as_ref().ok_or(Error::Closed)?;
        pending_evidence(request, &scan.directory, self.registry.as_deref())?;
        for s in scan.sources.iter() {
            s.recheck()?;
            s.final_snapshots()?;
        }
        if self.registry.is_none() && manifest_kind(request)? {
            require(
                scan.proof.root.require("manifest_sha256")?
                    == request
                        .require("manifest")?
                        .require("evidence")?
                        .require("sha256")?,
                "pending final64 original bytes/hash differ",
            )?;
        }
        core.inputs.control.check()
    }
    fn retire_stage(&mut self) -> Result<()> {
        // A consumed failed retirement is permanently uncertain, never repaired by emptiness.
        if self.uncertain {
            return Err(Error::contract("final64 retirement remains unconfirmed"));
        }
        if let Some(stage) = self.stage.as_mut() {
            // Reportable cleanup while the SAME owner remains reachable. The
            // worker only drops a spool whose disk cleanup already succeeded.
            stage.close()?;
        }
        if let Some(retirement) = self.retirement.take() {
            if let Err(e) = retirement.retire(self.stage.take()) {
                self.uncertain = true;
                return Err(e);
            }
        } else if self.stage.is_some() {
            return Err(Error::contract("final64 Stage has no retirement owner"));
        }
        Ok(())
    }
}

impl CardInstanceNativeSession {
    pub(crate) fn final_parent_preconditions(&self) -> Result<()> {
        let core = self.core()?;
        require(
            !core.inputs.format.hpo(),
            "HPO exploratory source cannot enter the legacy final64 profile",
        )?;
        core.transition_owner_check()?;
        require(
            core.phase == Phase::Authorized
                && matches!(core.inputs.mode, ScanMode::Historical)
                && core.inputs.parents.len() == 1,
            "final64 requires SAME authorized historical parent",
        )?;
        require(
            self.final_slot.is_none() && self.payload.upgrade().is_none(),
            "final64 slot has one lifetime/no outstanding parent payload",
        )
    }
    pub(crate) fn inspect_final_validation(&mut self, request: &[u8]) -> Result<BudgetVec<u8>> {
        self.inspect_final_validation_artifacts(request, None)
    }
    pub(crate) fn inspect_final_validation_artifacts(
        &mut self,
        request: &[u8],
        provider: Option<Arc<dyn crate::logical_artifacts::LogicalArtifactProvider>>,
    ) -> Result<BudgetVec<u8>> {
        self.final_parent_preconditions()?;
        self.final_slot = Some(Slot::empty()); // Reachable BEFORE any admitted work.
        self.final_slot.as_mut().unwrap().working = true;
        let result = self.inspect_final_inner(request, provider);
        self.remember_final(result)
    }
    fn inspect_final_inner(
        &mut self,
        raw: &[u8],
        provider: Option<Arc<dyn crate::logical_artifacts::LogicalArtifactProvider>>,
    ) -> Result<BudgetVec<u8>> {
        let core = self.core.as_ref().ok_or(Error::Closed)?;
        let i = &core.inputs;
        let b = &i.budget;
        let c = &i.control;
        if raw.len() > i.record_bytes || raw.len() > i.source_bytes {
            return Err(Error::resource("final64 request record/source cap"));
        }
        i.recheck()?;
        let request = json::parse(raw, b, c)?;
        let r = &request.root;
        let packed = i.format.packed();
        r.closed(
            &[
                "schema_version",
                "kind",
                "parent_run_sha256",
                "directory",
                "manifest",
                "expected_plan",
                "returned_report",
            ]
            .into_iter()
            .chain(if packed {
                vec!["storage_binding", "artifact_registry_sha256"]
            } else {
                vec![]
            })
            .collect::<Vec<_>>()
            .as_slice(),
            &[],
        )?;
        r.require("schema_version")?
            .integer(if packed { 2 } else { 1 }, if packed { 2 } else { 1 })?;
        require(
            field(r, "kind")?
                == if packed {
                    "card-instance-packed-final-validation-request"
                } else {
                    "card-instance-final-validation-request"
                },
            "closed final64 request required",
        )?;
        require(
            packed == provider.is_some(),
            "packed final request requires explicit final-only provider; ordinary requires None",
        )?;
        if packed {
            require(
                r.require("storage_binding")? == &crate::logical_artifacts::binding(),
                "final storage binding differs",
            )?;
        }
        let p = &i.parents[0];
        require(
            r.require("parent_run_sha256")?.string()?
                == p.sources.verified_sha(&p.directory.join("run.json"))?,
            "final64 retained parent run differs",
        )?;
        let root = PathBuf::from(field(r, "directory")?);
        require(
            root.is_absolute()
                && !root.starts_with(&p.directory)
                && !p.directory.starts_with(&root),
            "final64 collection must be outside parent originals",
        )?;
        require(
            fs::canonicalize(&root)? == root,
            "final64 original directory must be canonical/nonsymlink",
        )?;
        let pending = manifest_kind(r)?;
        require(
            field(r.require("manifest")?, "kind")?
                == match (packed, pending) {
                    (true, true) => "packed-issued-pending",
                    (true, false) => "packed-canonical",
                    (false, true) => "issued-pending",
                    (false, false) => "canonical",
                },
            "final request manifest subtype differs",
        )?;
        let registry = if let Some(provider) = provider {
            let registry = crate::logical_artifacts::Registry::open(
                provider,
                checksum(r.require("artifact_registry_sha256")?)?,
                b,
                c,
                i.source_bytes,
            )?;
            require(
                registry.roots.len() == 1 && registry.roots.contains_key(&root),
                "final registry must contain exactly SAME admitted root",
            )?;
            Some(registry)
        } else {
            None
        };
        self.final_slot.as_mut().ok_or(Error::Closed)?.registry = registry.clone();
        require(
            pending || r.require("returned_report")?.is_null(),
            "canonical final64 cannot claim returned live report",
        )?;
        require(
            !pending || !r.require("returned_report")?.is_null(),
            "pending final64 requires actual returned report evidence",
        )?;
        pending_evidence(r, &root, registry.as_deref())?;
        let expected = r.require("expected_plan")?;
        expected.closed(
            &[
                "linkage",
                "config_projection",
                "validation_invocation",
                "schedule",
                "policy_bindings",
            ],
            &[],
        )?;
        // Independent source read with full admission before JSON parsing; no publisher FD.
        let mut evidence = Sources::from_registry(
            &root,
            b.clone(),
            c.clone(),
            i.source_bytes,
            registry.clone(),
        )?;
        let control_name = if pending && !packed {
            "manifest.json.tmp"
        } else {
            "manifest.json"
        };
        if (if packed {
            evidence.file_len(&root.join(control_name))? as u64
        } else {
            fs::symlink_metadata(root.join(control_name))?.len()
        }) > i.source_bytes as u64
        {
            return Err(Error::resource(
                "final64 manifest source cap before hashing/decode",
            ));
        }
        let hash = evidence.hash(&root.join(control_name))?;
        if pending && !packed {
            require(
                hash == checksum(
                    r.require("manifest")?
                        .require("evidence")?
                        .require("sha256")?,
                )?,
                "pending final64 original hash differs before decode",
            )?;
        }
        let slot = self.final_slot.as_mut().ok_or(Error::Closed)?;
        slot.request = Some(request);
        let limits = CollectionLimits {
            source_bytes: i.source_bytes,
            record_bytes: i.record_bytes,
        };
        let scan = if let Some(registry) = registry {
            crate::structured_source::inspect_final64_artifacts(
                &root,
                (&p.directory, &p.state.root),
                &limits,
                b.clone(),
                c.clone(),
                registry,
            )?
        } else {
            crate::structured_source::inspect_final64_collection(
                &root,
                (&p.directory, &p.state.root),
                &limits,
                b.clone(),
                c.clone(),
                pending,
            )?
        };
        slot.scan = Some(scan);
        let scan = slot.scan.as_mut().unwrap();
        scan.sources.push(evidence)?;
        let r = &slot.request.as_ref().unwrap().root;
        let expected = r.require("expected_plan")?;
        let m = &scan.manifest.root;
        for k in [
            "linkage",
            "config_projection",
            "validation_invocation",
            "schedule",
            "policy_bindings",
        ] {
            require(
                m.require(k)? == expected.require(k)?,
                "final64 original differs from frozen expected plan",
            )?;
        }
        let final_generation = p
            .summary
            .root
            .require("generations")?
            .array()?
            .last()
            .ok_or(Error::Closed)?;
        let link = m.require("linkage")?;
        require(
            link.require("final_inference_sha256")?
                == final_generation.require("inference_sha256")?
                && link.require("final_generation")? == final_generation.require("generation")?,
            "final64 final artifact differs from retained parent",
        )?;
        let label = field(link, "policy_label")?;
        require(
            scan.learner_label.as_deref() == Some(label),
            "final64 selected label differs",
        )?;
        let options = m
            .require("policy_bindings")?
            .require(label)?
            .require("policy_options")?;
        require(
            options.require("expected_manifest_digest")?
                == link.require("final_inference_sha256")?,
            "final64 neural final artifact differs",
        )?;
        let report = scan
            .sources
            .iter_mut()
            .find(|s| s.owns_original(&root.join("report.json")))
            .ok_or(Error::Closed)?
            .json(&root.join("report.json"))?;
        report_order(m, &report.root, r.require("returned_report")?)?;
        let mut summary_bound = json::canonical(&scan.manifest.root, b, c)?
            .len()
            .checked_add(json::canonical(&scan.episodes.root, b, c)?.len())
            .and_then(|n| n.checked_add(65536))
            .ok_or_else(|| Error::resource("final64 summary copy overflow"))?;
        if packed {
            let description = &slot
                .registry
                .as_ref()
                .ok_or(Error::Closed)?
                .roots
                .get(&root)
                .ok_or(Error::Closed)?
                .description
                .root;
            let extra = json::canonical(description, b, c)?
                .len()
                .checked_add(json::canonical(&scan.proof.root, b, c)?.len())
                .and_then(|v| v.checked_add(65536))
                .ok_or_else(|| Error::resource("packed final summary bound overflow"))?;
            summary_bound = summary_bound
                .checked_add(extra)
                .ok_or_else(|| Error::resource("packed final summary bound overflow"))?;
        }
        let _summary_copy = json::admit(summary_bound, b)?;
        let mut summary = json::object([
            ("schema_version", json::integer(if packed { 2 } else { 1 })),
            (
                "kind",
                json::text(if packed {
                    "card-instance-packed-final-validation-summary"
                } else {
                    "card-instance-final-validation-summary"
                }),
            ),
            (
                "parent_run_sha256",
                copy(r.require("parent_run_sha256")?, c)?,
            ),
            ("directory", json::text(root.to_str().ok_or(Error::Closed)?)),
            (
                "manifest_kind",
                copy(r.require("manifest")?.require("kind")?, c)?,
            ),
            (
                "manifest_sha256",
                copy(scan.proof.root.require("manifest_sha256")?, c)?,
            ),
            (
                "source_set_sha256",
                copy(scan.proof.root.require("source_set_sha256")?, c)?,
            ),
            ("games", json::integer(64)),
            (
                "selected_decisions",
                json::integer(scan.accepted_learner as i128),
            ),
            (
                "zero_endpoint_games",
                json::integer(
                    scan.episodes
                        .root
                        .array()?
                        .iter()
                        .filter(|v| v.get("decisions").and_then(Value::as_i64) == Some(0))
                        .count() as i128,
                ),
            ),
            ("episodes", copy(&scan.episodes.root, c)?),
            ("qualified", Value::Bool(false)),
            ("parent_qualification", json::text("not_performed")),
        ]);
        for k in [
            "linkage",
            "config_projection",
            "validation_invocation",
            "elapsed_seconds",
        ] {
            summary
                .as_object_mut()
                .unwrap()
                .insert(k.into(), copy(m.require(k)?, c)?);
        }
        if packed {
            summary.as_object_mut().unwrap().insert(
                "storage_binding".into(),
                copy(r.require("storage_binding")?, c)?,
            );
            summary
                .as_object_mut()
                .unwrap()
                .insert("source_proof".into(), copy(&scan.proof.root, c)?);
        }
        let encoded = json::canonical(&summary, b, c)?;
        if encoded.len() > i.record_bytes {
            return Err(Error::resource("final64 summary record cap"));
        }
        slot.python_control =
            Some(b.reserve(crate::control_memory::reservation_bytes(encoded.len(), 0)?)?);
        slot.summary = Some(json::parse(&encoded, b, c)?);
        slot.recheck(core)?;
        Ok(encoded)
    }
    pub(crate) fn recheck_final_validation(&mut self) -> Result<()> {
        let core = self.core.as_ref().ok_or(Error::Closed)?;
        let slot = self.final_slot.as_mut().ok_or(Error::Closed)?;
        slot.active()?;
        slot.working = true;
        let result = slot.recheck(core);
        self.remember_final(result)
    }
    pub(crate) fn export_final_validation(&mut self, index: usize) -> Result<CardInstancePayload> {
        self.final_export_preconditions(index)?;
        self.final_slot.as_mut().unwrap().working = true;
        let result = self.export_final_inner(index);
        self.remember_final(result)
    }
    fn export_final_inner(&mut self, index: usize) -> Result<CardInstancePayload> {
        let core = self.core.as_ref().ok_or(Error::Closed)?;
        let i = &core.inputs;
        let b = &i.budget;
        let c = &i.control;
        let slot = self.final_slot.as_mut().ok_or(Error::Closed)?;
        slot.recheck(core)?;
        let scan = slot.scan.as_ref().ok_or(Error::Closed)?;
        let info = &scan.episodes.root.array()?[index];
        let journal = scan
            .directory
            .join(field(info, "episode_id")?)
            .join(format!(
                "player-{}.jsonl",
                info.require("viewer")?.integer(1, 2)?
            ));
        let original_len = scan
            .sources
            .iter()
            .find(|s| s.owns_original(&journal))
            .ok_or(Error::Closed)?
            .original_len(&journal)?;
        let copy_bound = original_len
            .checked_mul(2)
            .and_then(|n| n.checked_add(65536))
            .ok_or_else(|| Error::resource("final64 packet DTO copy overflow"))?;
        let _packet_copy = json::admit(copy_bound, b)?;
        slot.retirement = Some(crate::retirement::Retirement::start(b, |s: Stage| {
            drop(s);
            Ok(())
        })?);
        slot.stage = Some(Stage::new(
            b.clone(),
            i.scratch_bytes,
            i.chunk_bytes,
            c.clone(),
        )?);
        let scan = slot.scan.as_mut().ok_or(Error::Closed)?;
        let stage = slot.stage.as_mut().unwrap();
        let mut packet = scan.restage_final64(
            index,
            stage,
            &CollectionLimits {
                source_bytes: i.source_bytes,
                record_bytes: i.record_bytes,
            },
            b,
            c,
        )?;
        let info = &scan.episodes.root.array()?[index];
        for (k, v) in [
            ("schema_version", json::integer(1)),
            (
                "kind",
                json::text("card-instance-final-validation-episode-payload"),
            ),
            (
                "summary_sha256",
                json::text(&digest(
                    &slot.summary.as_ref().ok_or(Error::Closed)?.root,
                    b,
                    c,
                )?),
            ),
            ("episode_index", json::integer(index as i128)),
            ("episode_id", copy(info.require("episode_id")?, c)?),
            ("viewer", copy(info.require("viewer")?, c)?),
            ("episode_digest", json::text(&scan.digests[index])),
            ("arrays", stage.descriptors()?),
        ] {
            packet.as_object_mut().unwrap().insert(k.into(), v);
        }
        let metadata = json::canonical(&packet, b, c)?;
        if metadata.len() > i.record_bytes {
            return Err(Error::resource("final64 payload metadata record cap"));
        }
        let banks = stage.export()?;
        slot.retire_stage()?; // Join precedes any successful Python array return.
        slot.recheck(core)?;
        let lease = b.reserve(1024)?;
        let guard = Arc::new(());
        slot.packet = Arc::downgrade(&guard);
        slot.next += 1;
        Ok(CardInstancePayload {
            metadata,
            banks,
            _guard: guard,
            _lease: lease,
        })
    }
    fn remember_final<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            if let Some(core) = self.core.as_mut() {
                core.phase = Phase::Failed;
            }
        }
        self.final_slot.as_mut().unwrap().remember(result)
    }
    pub(crate) fn fail_final_validation(&mut self, error: Error) {
        if let Some(core) = self.core.as_mut() {
            core.phase = Phase::Failed;
        }
        let slot = self.final_slot.get_or_insert_with(Slot::empty);
        if slot.failure.is_none() {
            slot.failure = Some(error)
        }
    }
    pub(crate) fn close_final_validation(&mut self) -> Result<()> {
        let Some(slot) = self.final_slot.as_mut() else {
            return Ok(());
        };
        if slot.closed {
            return Ok(());
        }
        if slot.packet.upgrade().is_some() {
            return Err(Error::contract("final64 packet views remain live"));
        }
        if slot.working && slot.failure.is_none() {
            slot.failure = Some(Error::contract("interrupted final64 admitted transition"));
            if let Some(core) = self.core.as_mut() {
                core.phase = Phase::Failed;
            }
        }
        slot.working = true;
        let guard = if slot.failure.is_none() {
            self.core
                .as_ref()
                .ok_or(Error::Closed)
                .and_then(|core| slot.recheck(core))
        } else {
            Ok(())
        };
        if let Err(e) = guard {
            if let Some(core) = self.core.as_mut() {
                core.phase = Phase::Failed;
            }
            if slot.failure.is_none() {
                slot.failure = Some(e)
            }
        }
        if let Err(e) = slot.retire_stage() {
            if let Some(core) = self.core.as_mut() {
                core.phase = Phase::Failed;
            }
            return slot.remember(Err(e));
        }
        slot.scan = None;
        slot.summary = None;
        slot.request = None;
        slot.registry = None;
        slot.python_control = None;
        slot.closed = true;
        match &slot.failure {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }
    pub(super) fn final_slot_retired(&self) -> bool {
        self.final_slot
            .as_ref()
            .is_none_or(|s| s.closed && !s.uncertain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_source::{Controls, OriginalInputs, doc};
    use crate::structured_staging::Dtype;

    /// Real structural C5/J6 originals under an UNADMITTED parent primitive.
    /// No Run3/generations/actual runtime or public session admission is fabricated.
    fn setup() -> (PathBuf, CardInstanceNativeSession, Value, Document) {
        let base = std::env::temp_dir().join(format!(
            "gf-unadmitted-final64-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let parent = base.join("parent");
        let originals = base.join("originals");
        fs::create_dir_all(parent.join("artifacts/000000")).unwrap();
        fs::create_dir(parent.join("unadmitted-originals")).unwrap();
        fs::create_dir(&originals).unwrap();
        fs::write(
            parent.join("original-control.bin"),
            b"UNADMITTED parent primitive, not Run3",
        )
        .unwrap();
        fs::write(
            parent.join("run.json"),
            b"UNADMITTED retained-byte primitive, NOT a Run3",
        )
        .unwrap();
        let b = Budget::new(2 * 1024 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 120.).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/card_instance_final64_originals.json"),
            &b,
            &c,
        )
        .unwrap();
        for (name, hex) in fixture.root["files"].object().unwrap() {
            let hex = hex.string().unwrap();
            let mut raw = BudgetVec::with_capacity(&b, hex.len() / 2).unwrap();
            for i in (0..hex.len()).step_by(2) {
                raw.push(u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .unwrap();
            }
            let path = originals.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, &*raw).unwrap();
        }
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(originals.join("manifest.json")).unwrap()).unwrap();
        // Instantiate ONLY this controlled fixture's declared parent path.
        manifest["linkage"]["parent_run_path"] = json::text(parent.to_str().unwrap());
        fs::write(
            originals.join("manifest.json"),
            &*json::canonical(&manifest, &b, &c).unwrap(),
        )
        .unwrap();
        let state = &fixture.root["state"];
        let run_sha = format!(
            "{:x}",
            Sha256::digest(b"UNADMITTED retained-byte primitive, NOT a Run3")
        );
        let mut inputs = OriginalInputs::unadmitted_component_lifecycle(
            &parent,
            state,
            &manifest,
            &Value::Null,
            &Value::Null,
            &Value::Null,
            Stage::new(b.clone(), 1024 * 1024, 1024 * 1024, c.clone()).unwrap(),
            b.clone(),
            c.clone(),
        )
        .unwrap();
        inputs.stage.take().unwrap().close().unwrap();
        inputs.mode = ScanMode::Historical;
        inputs.scratch_bytes = 64 * 1024 * 1024;
        inputs.parents[0]
            .sources
            .hash(&parent.join("run.json"))
            .unwrap();
        inputs.parents[0].summary = doc(
            &json::object([(
                "generations",
                json::array([json::object([
                    ("generation", json::integer(1)),
                    (
                        "inference_sha256",
                        manifest["linkage"]["final_inference_sha256"].clone(),
                    ),
                ])]),
            )]),
            &b,
            &c,
        )
        .unwrap();
        let empty = || doc(&Value::Null, &b, &c).unwrap();
        let controls = Controls {
            summary: empty(),
            authorization: empty(),
            declarations: vec![],
            correspondences: vec![],
            fingerprint: "UNADMITTED-NOT-Source-authority".into(),
        };
        let core = CohortSession {
            inputs,
            controls,
            phase: Phase::Authorized,
            numeric_permit: None,
            numeric: None,
            expected_arrays: None,
            recorded_episodes: None,
            numeric_guard: None,
        };
        let retirement = crate::retirement::Retirement::start(&b, |core: CohortSession| {
            drop(core);
            Ok(())
        })
        .unwrap();
        let session = CardInstanceNativeSession {
            core: Some(core),
            retirement: Some(retirement),
            budget: b.clone(),
            payload: std::sync::Weak::new(),
            receipt: None,
            final_slot: None,
            original_episode: super::original_episode::State::new(),
        };
        let expected = json::object(
            [
                "linkage",
                "config_projection",
                "validation_invocation",
                "schedule",
                "policy_bindings",
            ]
            .into_iter()
            .map(|k| (k, manifest[k].clone())),
        );
        let request = json::object([
            ("schema_version", json::integer(1)),
            ("kind", json::text("card-instance-final-validation-request")),
            ("parent_run_sha256", json::text(&run_sha)),
            ("directory", json::text(originals.to_str().unwrap())),
            (
                "manifest",
                json::object([("kind", json::text("canonical"))]),
            ),
            ("expected_plan", expected),
            ("returned_report", Value::Null),
        ]);
        (base, session, request, fixture)
    }
    // Explicit synthetic storage metadata over the SAME unadmitted component
    // logical originals. This never admits a Python formal parent or Source.
    struct PackedDiagnostic {
        root: String,
        registry: Vec<u8>,
        describe: Vec<u8>,
        files: std::collections::BTreeMap<String, Vec<u8>>,
    }
    impl crate::logical_artifacts::LogicalArtifactProvider for PackedDiagnostic {
        fn registry(&self, b: &Arc<Budget>, _: &Control) -> Result<BudgetVec<u8>> {
            let mut v = BudgetVec::with_capacity(b, self.registry.len())?;
            v.extend_from_slice(&self.registry)?;
            Ok(v)
        }
        fn describe(&self, root: &str, b: &Arc<Budget>, _: &Control) -> Result<BudgetVec<u8>> {
            require(root == self.root, "diagnostic root differs")?;
            let mut v = BudgetVec::with_capacity(b, self.describe.len())?;
            v.extend_from_slice(&self.describe)?;
            Ok(v)
        }
        fn read(
            &self,
            root: &str,
            name: &str,
            length: usize,
            sha: &str,
            b: &Arc<Budget>,
            _: &Control,
        ) -> Result<BudgetVec<u8>> {
            require(root == self.root, "diagnostic root differs")?;
            let raw = self.files.get(name).ok_or(Error::Closed)?;
            require(
                raw.len() == length && format!("{:x}", Sha256::digest(raw)) == sha,
                "diagnostic original differs",
            )?;
            let mut v = BudgetVec::with_capacity(b, raw.len())?;
            v.extend_from_slice(raw)?;
            Ok(v)
        }
        fn recheck(&self, root: &str, c: &Control) -> Result<()> {
            c.check()?;
            require(root == self.root, "diagnostic root differs")
        }
    }
    fn packed_diagnostic(
        session: &mut CardInstanceNativeSession,
        request: &mut Value,
        fixture: &Document,
    ) -> Arc<PackedDiagnostic> {
        let b = session.budget.clone();
        let c = session.core().unwrap().inputs.control.clone();
        let root = PathBuf::from(request["directory"].string().unwrap());
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let mut state = session.core().unwrap().inputs.parents[0].state.root.clone();
        state["run_variant"] = json::text("formal-packed-parent-v1");
        state.as_object_mut().unwrap().insert(
            "storage_binding".into(),
            crate::logical_artifacts::binding(),
        );
        let storage_config = json::object([
            ("format", json::text("sqlite-group-pack-v1")),
            ("db_root", json::text("/UNADMITTED/db")),
            ("ram_root", json::text("/UNADMITTED/ram")),
            ("budget_mib", json::integer(2048)),
            ("queue_mib", json::integer(512)),
            ("max_pack_mib", json::integer(256)),
        ]);
        state["effective_config"]
            .as_object_mut()
            .unwrap()
            .insert("storage".into(), storage_config.clone());
        manifest["config"]
            .as_object_mut()
            .unwrap()
            .insert("storage".into(), storage_config);
        manifest["config_hash"] = json::text(&digest(&manifest["config"], &b, &c).unwrap());
        manifest["config_projection"]["parent_config_sha256"] =
            json::text(&digest(&state["effective_config"], &b, &c).unwrap());
        manifest["config_projection"]["projected_config_sha256"] = manifest["config_hash"].clone();
        request["expected_plan"]["config_projection"] = manifest["config_projection"].clone();
        let core = session.core.as_mut().unwrap();
        core.inputs.parents[0].state = doc(&state, &b, &c).unwrap();
        core.inputs.format = crate::structured_controls::SourceFormat::PackedCorrected;
        let mut files = std::collections::BTreeMap::new();
        for name in fixture.root["files"].object().unwrap().keys() {
            files.insert(name.clone(), fs::read(root.join(name)).unwrap());
        }
        files.insert(
            "manifest.json".into(),
            json::canonical(&manifest, &b, &c).unwrap().to_vec(),
        );
        let sha = "a".repeat(64);
        let storage = json::object([
            ("format", json::text("sqlite-group-pack-v1")),
            ("locator_sha256", json::text(&sha)),
            ("descriptor_sha256", json::text(&sha)),
            ("database_sha256", json::text(&sha)),
            ("seal_id", json::text("UNADMITTED-synthetic-storage")),
            ("logical_sha256", json::text(&sha)),
        ]);
        let physical = root.join("UNADMITTED-physical");
        let mut custody_files = vec![];
        for (index, role, path) in [
            (1, "locator", root.join("collection.locator.json")),
            (2, "descriptor", physical.join("collection.published.json")),
            (3, "database", physical.join("collection.sqlite3")),
        ] {
            custody_files.push(json::object([
                ("role", json::text(role)),
                ("path", json::text(path.to_str().unwrap())),
                ("sha256", json::text(&sha)),
                (
                    "identity",
                    json::array([1, index, 33152, 1, 1, 1].map(json::integer)),
                ),
            ]));
        }
        custody_files.sort_by_key(|v| v["path"].string().unwrap().to_owned());
        let mut directories = [root.clone(), physical];
        directories.sort();
        let mut custody = json::object([
            ("recipe", json::text("packed-original-custody-v1")),
            ("files", json::array(custody_files)),
            (
                "directories",
                json::array(directories.iter().map(|path| {
                    json::object([
                        ("path", json::text(path.to_str().unwrap())),
                        ("device", json::integer(1)),
                        ("inode", json::integer(1)),
                    ])
                })),
            ),
        ]);
        let fingerprint = digest(&custody, &b, &c).unwrap();
        custody
            .as_object_mut()
            .unwrap()
            .insert("fingerprint".into(), json::text(&fingerprint));
        let mut groups = vec![];
        let mut episodes = vec![];
        let mut membership = std::collections::BTreeMap::new();
        for (schedule, health) in manifest["schedule"]
            .array()
            .unwrap()
            .iter()
            .zip(manifest["groups"].array().unwrap())
        {
            let group = schedule["episodes"][0]["episode_id"].clone();
            groups.push(json::object([
                ("group_id", group.clone()),
                ("record", health.clone()),
            ]));
            for spec in schedule["episodes"].array().unwrap() {
                let name = spec["episode_id"].string().unwrap();
                let result = manifest["episodes"]
                    .array()
                    .unwrap()
                    .iter()
                    .find(|v| v["episode_id"] == spec["episode_id"])
                    .unwrap();
                episodes.push(json::object([
                    ("episode_id", spec["episode_id"].clone()),
                    ("group_id", group.clone()),
                    ("game_id", result["game_id"].clone()),
                    (
                        "record",
                        json::object([
                            ("end_kind", result["end_kind"].clone()),
                            ("failure", result["failure"].clone()),
                        ]),
                    ),
                ]));
                membership.insert(name.to_owned(), group.clone());
            }
        }
        groups.sort_by_key(|v| v["group_id"].string().unwrap().to_owned());
        episodes.sort_by_key(|v| v["episode_id"].string().unwrap().to_owned());
        let rows = files.iter().map(|(name, raw)| {
            let member_sha = format!("{:x}", Sha256::digest(raw));
            let group = membership.get(name.split('/').next().unwrap());
            let identity = json::object([
                ("format", json::text("packed-member-v1")),
                ("database_sha256", json::text(&sha)),
                ("group_id", group.cloned().unwrap_or(Value::Null)),
                (
                    "pack_sha256",
                    if group.is_some() {
                        json::text(&sha)
                    } else {
                        Value::Null
                    },
                ),
                ("name", json::text(name)),
                (
                    "offset",
                    if group.is_some() {
                        json::integer(0)
                    } else {
                        Value::Null
                    },
                ),
                ("length", json::integer(raw.len() as i128)),
                ("sha256", json::text(&member_sha)),
            ]);
            json::object([
                ("name", json::text(name)),
                ("bytes", json::integer(raw.len() as i128)),
                ("sha256", json::text(&member_sha)),
                ("identity", identity),
            ])
        });
        let describe = json::object([
            ("format", json::text("python-original-bytes-v1")),
            ("root", json::text(root.to_str().unwrap())),
            ("storage", storage.clone()),
            ("custody", custody.clone()),
            ("files", json::array(rows)),
            (
                "index",
                json::object([
                    ("groups", json::array(groups)),
                    ("episodes", json::array(episodes)),
                ]),
            ),
        ]);
        let registry = json::object([
            ("format", json::text("python-original-bytes-v1")),
            (
                "roots",
                json::array([json::object([
                    ("root", json::text(root.to_str().unwrap())),
                    ("storage", storage),
                    ("custody", custody),
                ])]),
            ),
        ]);
        let registry = json::canonical(&registry, &b, &c).unwrap().to_vec();
        request["schema_version"] = json::integer(2);
        request["kind"] = json::text("card-instance-packed-final-validation-request");
        request["manifest"]["kind"] = json::text("packed-canonical");
        request.as_object_mut().unwrap().insert(
            "storage_binding".into(),
            crate::logical_artifacts::binding(),
        );
        request.as_object_mut().unwrap().insert(
            "artifact_registry_sha256".into(),
            json::text(&format!("{:x}", Sha256::digest(&registry))),
        );
        Arc::new(PackedDiagnostic {
            root: root.to_str().unwrap().to_owned(),
            registry,
            describe: json::canonical(&describe, &b, &c).unwrap().to_vec(),
            files,
        })
    }
    #[test]
    #[ignore = "old rules-v1 captured input unsupported by current-v2; requires newly generated current reference"]
    fn packed_final64_unadmitted_component_four_method_packets_preserve_original_own_values() {
        let (base, mut session, mut request, fixture) = setup();
        // A second independent UNADMITTED reference primitive is test-only.
        // Production final validation still borrows its SAME actual parent.
        let (reference_base, mut reference, reference_request, _) = setup();
        reference
            .inspect_final_validation(&encoded(&reference, &reference_request))
            .unwrap();
        let provider = packed_diagnostic(&mut session, &mut request, &fixture);
        let raw = encoded(&session, &request);
        let summary = session
            .inspect_final_validation_artifacts(&raw, Some(provider))
            .unwrap();
        let b = session.budget.clone();
        let c = session.core().unwrap().inputs.control.clone();
        let summary = json::parse(&summary, &b, &c).unwrap();
        assert_eq!(summary.root["schema_version"], json::integer(2));
        assert_eq!(summary.root["qualified"], Value::Bool(false));
        assert_eq!(
            summary.root["parent_qualification"],
            json::text("not_performed")
        );
        assert_eq!(summary.root["zero_endpoint_games"], json::integer(16));
        for ordinal in 0..64 {
            session.recheck_final_validation().unwrap();
            let packet = session.export_final_validation(ordinal).unwrap();
            let meta = json::parse(&packet.metadata, &b, &c).unwrap();
            let original = reference.export_final_validation(ordinal).unwrap();
            let original_meta = json::parse(&original.metadata, &b, &c).unwrap();
            for key in [
                "schema_version",
                "kind",
                "episode_index",
                "episode_id",
                "viewer",
                "episode_digest",
                "binding",
                "endpoint",
                "bank",
                "decisions",
                "arrays",
                "correspondence",
            ] {
                assert_eq!(
                    meta.root[key], original_meta.root[key],
                    "original packet field {ordinal}/{key}"
                );
            }
            for descriptor in meta.root["arrays"].array().unwrap() {
                assert_eq!(
                    bytes(&packet.banks, descriptor),
                    bytes(&original.banks, descriptor),
                    "exact original own numeric/DTO bytes {ordinal}/{}",
                    descriptor["key"]
                );
            }
            drop(original);
            assert_eq!(meta.root["schema_version"], json::integer(1));
            assert_eq!(
                meta.root["episode_digest"],
                fixture.root["oracle"][ordinal]["digest"]
            );
            assert_eq!(
                meta.root["viewer"],
                fixture.root["oracle"][ordinal]["viewer"]
            );
            assert_eq!(
                meta.root["endpoint"]["outcome"],
                fixture.root["oracle"][ordinal]["outcome"]
            );
            assert!(session.close_final_validation().is_err());
            drop(packet);
        }
        session.close_final_validation().unwrap();
        session.close().unwrap();
        reference.close_final_validation().unwrap();
        reference.close().unwrap();
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(reference_base).unwrap();
    }
    fn encoded(s: &CardInstanceNativeSession, v: &Value) -> BudgetVec<u8> {
        json::canonical(v, &s.budget, &s.core().unwrap().inputs.control).unwrap()
    }
    fn bytes(bank: &crate::structured_staging::Banks, desc: &Value) -> Vec<u8> {
        let index = desc["bank_index"].as_u64().unwrap() as usize;
        let start = desc["element_offset"].as_u64().unwrap() as usize;
        let count = desc["element_count"].as_u64().unwrap() as usize;
        match desc["dtype"].as_str().unwrap() {
            "uint8" => bank.bytes[index][start..start + count].to_vec(),
            "int64" => bank.integers[index][start..start + count]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            "float32" => bank.floats[index][start..start + count]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            _ => panic!(),
        }
    }
    #[test]
    #[ignore = "old rules-v1 captured input unsupported by current-v2; requires newly generated current reference"]
    fn card_instance_final64_all_originals_raw_dto_order_zero_and_lifetimes_match_python() {
        let (base, mut session, request, fixture) = setup();
        // Persisted invocation ended long ago; the fresh active Control admits it.
        assert_eq!(
            request["expected_plan"]["validation_invocation"]["started_monotonic"].as_f64(),
            Some(10.)
        );
        assert!(
            session.core().unwrap().inputs.parents[0]
                .summary
                .root
                .get("run_sha256")
                .is_none()
        );
        let summary = session
            .inspect_final_validation(&encoded(&session, &request))
            .unwrap();
        let c = session.core().unwrap().inputs.control.clone();
        let b = session.budget.clone();
        let info = json::parse(&summary, &b, &c).unwrap();
        assert_eq!(info.root["games"], json::integer(64));
        assert_eq!(info.root["zero_endpoint_games"], json::integer(16));
        assert_eq!(info.root["qualified"], Value::Bool(false));
        for index in 0..64 {
            let packet = session.export_final_validation(index).unwrap();
            let meta = json::parse(&packet.metadata, &b, &c).unwrap();
            let oracle = &fixture.root["oracle"][index];
            assert_eq!(meta.root["episode_digest"], oracle["digest"]);
            assert_eq!(meta.root["viewer"], oracle["viewer"]);
            assert_eq!(meta.root["endpoint"]["outcome"], oracle["outcome"]);
            for desc in meta.root["arrays"].array().unwrap() {
                let key = desc["key"].as_str().unwrap();
                if key.starts_with("dto.") || key.ends_with(".payload") {
                    continue;
                }
                let split = key.split('.').collect::<Vec<_>>();
                let position = split[3].parse::<usize>().unwrap();
                let name = split.last().unwrap();
                let tokens = &oracle["tokens"][position];
                let value = if key.contains(".candidate.") {
                    &tokens["candidate"][name]
                } else {
                    &tokens["frame"][name]
                };
                let mut reference =
                    Stage::new(b.clone(), 1024 * 1024, 1024 * 1024, c.clone()).unwrap();
                let dtype = if desc["dtype"] == "uint8" {
                    Dtype::U8
                } else {
                    Dtype::I64
                };
                reference.array("reference", dtype, value).unwrap();
                let descriptor = reference.descriptors().unwrap();
                let banks = reference.export_retiring_disk().unwrap();
                assert_eq!(
                    bytes(&packet.banks, desc),
                    bytes(&banks, &descriptor[0]),
                    "raw parity {index}/{key}"
                );
                assert_eq!(desc["shape"], descriptor[0]["shape"]);
            }
            assert!(session.export_final_validation(index).is_err());
            assert!(session.close_final_validation().is_err()); // view guard, not an admitted failure
            if let Ok(directory) = std::env::var("GFB_FINAL64_TEST_EXPORT_DIR") {
                let target = PathBuf::from(directory);
                fs::create_dir_all(&target).unwrap();
                fs::write(target.join("summary.json"), &*summary).unwrap();
                let mut dump = json::object([("metadata", meta.root.clone())]);
                let arrays = meta.root["arrays"]
                    .array()
                    .unwrap()
                    .iter()
                    .map(|d| {
                        (
                            d["key"].string().unwrap().to_owned(),
                            json::text(
                                &bytes(&packet.banks, d)
                                    .iter()
                                    .map(|v| format!("{v:02x}"))
                                    .collect::<String>(),
                            ),
                        )
                    })
                    .collect();
                dump.as_object_mut()
                    .unwrap()
                    .insert("array_hex".into(), Value::Object(arrays));
                fs::write(
                    target.join(format!("episode-{index:02}.json")),
                    &*json::canonical(&dump, &b, &c).unwrap(),
                )
                .unwrap();
            }
            drop(packet);
            session.recheck_final_validation().unwrap();
        }
        assert!(
            session
                .final_slot_operation(CardInstanceOperation::Recorded(0))
                .is_ok()
        );
        assert!(
            session
                .final_slot_operation(CardInstanceOperation::FinalizeRead)
                .is_err()
        );
        session.close_final_validation().unwrap();
        session.close_final_validation().unwrap();
        assert!(session.recheck_final_validation().is_err());
        session.close().unwrap();
        drop(session);
        drop(fixture);
        drop(info);
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    #[ignore = "old rules-v1 captured input unsupported by current-v2; requires newly generated current reference"]
    fn card_instance_final64_original_mutation_repair_never_revives_and_cleanup_works() {
        let (base, mut s, r, _fixture) = setup();
        s.inspect_final_validation(&encoded(&s, &r)).unwrap();
        let path = base.join("originals/report.json");
        let before = fs::read(&path).unwrap();
        fs::write(&path, b"changed").unwrap();
        assert!(s.recheck_final_validation().is_err());
        fs::write(&path, before).unwrap();
        assert!(s.export_final_validation(0).is_err());
        assert!(s.close_final_validation().is_err());
        assert!(s.final_slot_retired());
        assert!(s.close_final_validation().is_ok());
        s.close().unwrap();
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn card_instance_final64_closed_request_and_plan_failures_are_sticky() {
        let (base, mut s, mut r, _fixture) = setup();
        r["expected_plan"]["linkage"]["final_generation"] = json::integer(2);
        assert!(s.inspect_final_validation(&encoded(&s, &r)).is_err());
        r["expected_plan"]["linkage"]["final_generation"] = json::integer(1);
        assert!(s.inspect_final_validation(&encoded(&s, &r)).is_err());
        assert!(s.export_final_validation(0).is_err());
        assert!(s.close_final_validation().is_err());
        s.close().unwrap();
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn card_instance_final64_recorded_clock_projection_and_linkage_mutations_reject() {
        let (base, s, r, fixture) = setup();
        let b = &s.budget;
        let c = &s.core().unwrap().inputs.control;
        let path = base.join("originals/manifest.json");
        let raw = fs::read(path).unwrap();
        let original = json::parse(&raw, b, c).unwrap();
        let state = &fixture.root["state"];
        let parent = base.join("parent");
        manifest_controls(&original.root, &parent, state, b, c).unwrap();
        for change in 0..8 {
            let mut m = original.root.clone();
            match change {
                0 => m["linkage"]["collection_generation"] = json::integer(0),
                1 => m["linkage"]["final_generation"] = json::integer(0),
                2 => m["config"]["run"]["execution_seconds"] = json::integer(1),
                3 => m["validation_invocation"]["execution_deadline_monotonic"] = json::integer(1),
                4 => {
                    m["config_projection"]["parent_config_inputs_sha256"] =
                        json::text(&"0".repeat(64))
                }
                5 => m["elapsed_seconds"] = json::float(7201.).unwrap(),
                6 => m["config"]["extra"] = Value::Null,
                _ => m["linkage"]["parent_run_path"] = json::text("/foreign"),
            };
            assert!(
                manifest_controls(&m, &parent, state, b, c).is_err(),
                "mutation {change}"
            );
        }
        drop(r);
        drop(s);
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    #[ignore = "old rules-v1 captured input unsupported by current-v2; requires newly generated current reference"]
    fn card_instance_final64_pending_inode_hash_and_canonical_absence_are_exact() {
        let (base, mut s, mut r, _fixture) = setup();
        let root = base.join("originals");
        fs::rename(root.join("manifest.json"), root.join("manifest.json.tmp")).unwrap();
        let metadata = fs::metadata(root.join("manifest.json.tmp")).unwrap();
        let dir = fs::metadata(&root).unwrap();
        let mut file = json::object([]);
        for (k, v) in [
            ("device", metadata.dev() as i128),
            ("inode", metadata.ino() as i128),
            ("mode", metadata.mode() as i128),
            ("size", metadata.len() as i128),
            (
                "mtime_ns",
                metadata.mtime() as i128 * 1_000_000_000 + metadata.mtime_nsec() as i128,
            ),
            (
                "ctime_ns",
                metadata.ctime() as i128 * 1_000_000_000 + metadata.ctime_nsec() as i128,
            ),
        ] {
            file.as_object_mut()
                .unwrap()
                .insert(k.into(), json::integer(v));
        }
        let sha = format!(
            "{:x}",
            Sha256::digest(fs::read(root.join("manifest.json.tmp")).unwrap())
        );
        r["manifest"] = json::object([
            ("kind", json::text("issued-pending")),
            (
                "evidence",
                json::object([
                    (
                        "directory_identity",
                        json::object([
                            ("device", json::integer(dir.dev() as i128)),
                            ("inode", json::integer(dir.ino() as i128)),
                        ]),
                    ),
                    ("file_identity", file),
                    ("bytes", json::integer(metadata.len() as i128)),
                    ("sha256", json::text(&sha)),
                ]),
            ),
        ]);
        r["returned_report"] =
            serde_json::from_slice(&fs::read(root.join("report.json")).unwrap()).unwrap();
        r["returned_report"]["wall_seconds"] = json::float(0.3).unwrap();
        s.inspect_final_validation(&encoded(&s, &r)).unwrap();
        fs::write(root.join("manifest.json"), b"foreign").unwrap();
        assert!(s.recheck_final_validation().is_err());
        fs::remove_file(root.join("manifest.json")).unwrap();
        assert!(s.export_final_validation(0).is_err());
        assert!(s.close_final_validation().is_err());
        s.close().unwrap();
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    #[ignore = "old rules-v1 captured input unsupported by current-v2; requires newly generated current reference"]
    fn card_instance_final64_request_admission_budget_cancel_and_no_revival() {
        let (base, mut s, r, _fixture) = setup();
        assert!(s.final_request_bound(64 * 1024 * 1024 + 1).is_err());
        s.inspect_final_validation(&encoded(&s, &r)).unwrap();
        assert!(s.export_final_validation(1).is_err()); // pure ordinal refusal
        assert!(s.core().unwrap().phase == Phase::Authorized);
        let budget = s.budget.clone();
        let held = budget
            .reserve(budget.limit() - budget.used() - 1024)
            .unwrap();
        assert!(s.export_final_validation(0).is_err());
        drop(held);
        assert!(s.export_final_validation(0).is_err()); // late capacity cannot revive
        assert!(s.close_final_validation().is_err());
        assert!(s.final_slot_retired());
        assert!(s.finalize_read().is_err());
        s.close().unwrap();
        fs::remove_dir_all(base).unwrap();

        let (base, mut s, r, _fixture) = setup();
        s.inspect_final_validation(&encoded(&s, &r)).unwrap();
        s.core().unwrap().inputs.control.cancel();
        assert!(matches!(
            s.export_final_validation(0),
            Err(Error::Cancelled)
        ));
        assert!(matches!(s.close_final_validation(), Err(Error::Cancelled)));
        assert!(s.final_slot_retired());
        assert!(s.close().is_err());
        assert!(s.is_closed());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn card_instance_final64_failed_stage_keeps_owner_and_foreign_evidence_until_cleanup() {
        let b = Budget::new(64 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let mut slot = Slot::empty();
        slot.retirement = Some(
            crate::retirement::Retirement::start(&b, |s: Stage| {
                drop(s);
                Ok(())
            })
            .unwrap(),
        );
        slot.stage = Some(Stage::new(b.clone(), 1024 * 1024, 1024 * 1024, c.clone()).unwrap());
        let root = slot.stage.as_ref().unwrap().unadmitted_fault_directory();
        fs::write(root.join("foreign"), b"preserve").unwrap();
        assert!(slot.retire_stage().is_err());
        assert!(slot.stage.is_some() && slot.retirement.is_some());
        assert_eq!(fs::read(root.join("foreign")).unwrap(), b"preserve");
        fs::remove_file(root.join("foreign")).unwrap();
        slot.retire_stage().unwrap();
        assert!(!root.exists());
        slot.retire_stage().unwrap();
        assert!(slot.stage.is_none() && slot.retirement.is_none());
        let mut uncertain = Slot::empty();
        uncertain.uncertain = true;
        assert!(uncertain.retire_stage().is_err());
        assert!(uncertain.retire_stage().is_err());
    }

    #[test]
    fn card_instance_final64_python_integer_float_and_min_tie_clock_literals_are_exact() {
        let b = Budget::new(8 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/card_instance_final64_invocations_python.json"),
            &b,
            &c,
        )
        .unwrap();
        for case in fixture.root["cases"].array().unwrap() {
            assert_eq!(
                invocation(&case["invocation"], &fixture.root["run"]).unwrap(),
                case["duration"].number().unwrap()
            );
            let mut changed = case["invocation"].clone();
            let old = &changed["execution_deadline_monotonic"];
            changed["execution_deadline_monotonic"] =
                if old.as_number().is_some_and(json::integer_token) {
                    json::float(old.number().unwrap()).unwrap()
                } else {
                    json::integer(old.number().unwrap() as i128)
                };
            // Values that are integral retain Python's exact numeric type.
            if changed["execution_deadline_monotonic"].number().unwrap()
                == case["invocation"]["execution_deadline_monotonic"]
                    .number()
                    .unwrap()
            {
                assert!(invocation(&changed, &fixture.root["run"]).is_err());
            }
        }
    }
}
fn report_order(m: &Value, r: &Value, returned: &Value) -> Result<()> {
    let duration = invocation(
        m.require("validation_invocation")?,
        m.require("config")?.require("run")?,
    )?;
    require(
        r.get("healthy") == Some(&Value::Bool(true))
            && r.get("comparison_valid") == Some(&Value::Bool(true)),
        "healthy final64 original report required",
    )?;
    let wall = nonnegative(r.require("wall_seconds")?)?;
    let elapsed = nonnegative(m.require("elapsed_seconds")?)?;
    require(
        wall <= elapsed && elapsed <= duration,
        "final64 persisted wall/elapsed order differs",
    )?;
    if !returned.is_null() {
        require(
            returned.object()?.len() == r.object()?.len(),
            "final64 returned report fieldset differs",
        )?;
        for (k, v) in r.object()? {
            if k != "wall_seconds" {
                require(
                    returned.require(k)? == v,
                    "final64 returned original report differs",
                )?;
            }
        }
        let final_wall = nonnegative(returned.require("wall_seconds")?)?;
        require(
            elapsed <= final_wall && final_wall <= duration,
            "final64 returned wall/elapsed order differs",
        )?;
    }
    Ok(())
}

impl CardInstanceNativeSession {
    pub(crate) fn final_slot_operation(&self, op: CardInstanceOperation) -> Result<()> {
        if let Some(slot) = &self.final_slot {
            if slot.closed {
                return Ok(());
            }
            slot.active()?;
            if matches!(op, CardInstanceOperation::Recorded(_))
                && slot.next == 64
                && slot.packet.upgrade().is_none()
                && slot.stage.is_none()
            {
                return slot.recheck(self.core()?);
            }
            return Err(Error::contract(
                "final64 collection loan excludes this parent operation",
            ));
        }
        Ok(())
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.retire_stage().is_err() {
            // Discarding the reachable cleanup owner cannot authorize deletion
            // of foreign evidence or stale descriptor retries. Same Stage
            // quarantine policy as the existing fixture core.
            if let Some(stage) = self.stage.take() {
                std::mem::forget(stage);
            }
            if let Some(retirement) = self.retirement.take() {
                let _ = retirement.retire(None);
            }
        }
    }
}

impl CardInstanceNativeSession {
    pub(crate) fn final_request_bound(&self, length: usize) -> Result<()> {
        let i = &self.core()?.inputs;
        if length > i.record_bytes || length > i.source_bytes {
            return Err(Error::resource(
                "final64 request record/source bound before copy",
            ));
        }
        Ok(())
    }
    pub(crate) fn final_export_preconditions(&self, index: usize) -> Result<()> {
        let slot = self.final_slot.as_ref().ok_or(Error::Closed)?;
        slot.active()?;
        require(
            index < 64 && index == slot.next && slot.packet.upgrade().is_none(),
            "final64 ordered ordinal or outstanding packet differs",
        )
    }
}

impl CardInstanceNativeSession {
    pub(crate) fn final_loan_open(&self) -> bool {
        self.final_slot.as_ref().is_some_and(|slot| !slot.closed)
    }
    pub(crate) fn final_recheck_preconditions(&self) -> Result<()> {
        self.final_slot.as_ref().ok_or(Error::Closed)?.active()
    }
    pub(crate) fn final_has_packet(&self) -> bool {
        self.final_slot
            .as_ref()
            .is_some_and(|slot| slot.packet.upgrade().is_some())
    }
}
