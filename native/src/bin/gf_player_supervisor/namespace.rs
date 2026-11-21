//! SAME passwd-user scheduling namespace. Durable observations never recreate GpuJob.
use crate::{Result, invalid, process::Retirement, records::*};
use serde_json::{Value, json};
use std::{
    ffi::CStr,
    fs::{self, File, OpenOptions},
    os::{
        fd::AsRawFd,
        unix::{
            ffi::OsStrExt,
            fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
        },
    },
    path::{Path, PathBuf},
};
const RESTART: &str = "I confirm the environment restarted and all previous GPU jobs terminated.";
const RELEASE: &str =
    "I authorize GPU admission to resume without confirming termination of the previous job.";
fn legacy_admission(v: &Value, ticket: &str) -> Result<()> {
    record(
        v,
        "hpo-gpu-admission",
        1,
        &["ticket", "context", "boot_id", "owner_pid", "admitted_unix"],
    )?;
    hex(&v["ticket"], 32)?;
    if v["ticket"] != ticket {
        return Err(invalid("legacy ticket mismatch"));
    }
    context(&v["context"], false)?;
    text(&v["boot_id"])?;
    positive(&v["owner_pid"])?;
    observation_time(&v["admitted_unix"])
}
fn legacy_termination(v: &Value, a: &Value) -> Result<()> {
    let shared = v["schema_version"].as_u64() == Some(2);
    let mut fields = vec![
        "ticket",
        "admission_sha256",
        "context_sha256",
        "boot_id",
        "confirmed_unix",
        "resources",
    ];
    if shared {
        fields.push("services");
    }
    record(
        v,
        "hpo-gpu-termination",
        if shared { 2 } else { 1 },
        &fields,
    )?;
    if a["schema_version"] != 1
        || v["ticket"] != a["ticket"]
        || v["admission_sha256"] != digest(a)
        || v["context_sha256"] != digest(&a["context"])
        || v["boot_id"] != a["boot_id"]
    {
        return Err(invalid("legacy termination mismatch"));
    }
    observation_time(&v["confirmed_unix"])?;
    let mut keys = std::collections::BTreeSet::new();
    for r in v["resources"]
        .as_array()
        .ok_or_else(|| invalid("legacy actual resources required"))?
    {
        object(r, &["kind", "key", "pid", "outcome"])?;
        let kind = text(&r["kind"])?;
        let outcome = text(&r["outcome"])?;
        if !["worker", "learner"].contains(&kind)
            || !["reaped", "never_started"].contains(&outcome)
            || !keys.insert((kind, uint(&r["key"])?))
        {
            return Err(invalid("invalid legacy resource history"));
        }
        if !r["pid"].is_null() {
            positive(&r["pid"])?;
        }
        if (outcome == "reaped") == r["pid"].is_null() {
            return Err(invalid("legacy resource outcome/PID mismatch"));
        }
    }
    if shared {
        let services = v["services"]
            .as_array()
            .ok_or_else(|| invalid("legacy shared service required"))?;
        if services.len() != 1 {
            return Err(invalid("legacy exact service count required"));
        }
        let s = &services[0];
        object(
            s,
            &[
                "key",
                "service_id",
                "broker_pid",
                "broker_outcome",
                "gpu_pid",
                "gpu_outcome",
                "gpu_exitcode",
                "registration_sha256",
                "retirement_sha256",
            ],
        )?;
        hex(&s["service_id"], 32)?;
        if s["key"] != 1 {
            return Err(invalid("legacy service key mismatch"));
        }
        for k in ["registration_sha256", "retirement_sha256"] {
            hex(&s[k], 64)?;
        }
        if s["broker_outcome"] == "never_started" && s["gpu_outcome"] == "never_started" {
            if !["broker_pid", "gpu_pid", "gpu_exitcode"]
                .iter()
                .all(|k| s[k].is_null())
            {
                return Err(invalid("legacy never-started service mismatch"));
            }
        } else {
            for prefix in ["broker", "gpu"] {
                positive(&s[format!("{prefix}_pid")])?;
                if s[format!("{prefix}_outcome")] != "reaped" {
                    return Err(invalid("legacy actual composite reap required"));
                }
            }
            s["gpu_exitcode"]
                .as_i64()
                .ok_or_else(|| invalid("legacy actual exit code required"))?;
        }
    }
    Ok(())
}
fn recovery(v: &Value, a: &Value, restart: bool) -> Result<()> {
    let keys = if restart {
        vec![
            "ticket",
            "admission_sha256",
            "admitted_boot_id",
            "current_boot_id",
            "operator",
            "confirmation",
            "confirmed_unix",
        ]
    } else {
        vec![
            "ticket",
            "admission_sha256",
            "admitted_boot_id",
            "current_boot_id",
            "operator",
            "reason",
            "confirmation",
            "released_unix",
        ]
    };
    record(
        v,
        if restart {
            "hpo-gpu-environment-restart"
        } else {
            "hpo-gpu-admission-release"
        },
        1,
        &keys,
    )?;
    if v["ticket"] != a["ticket"]
        || v["admission_sha256"] != digest(a)
        || v["admitted_boot_id"] != a["boot_id"]
        || v["confirmation"] != if restart { RESTART } else { RELEASE }
    {
        return Err(invalid("operator recovery binding mismatch"));
    }
    text(&v["current_boot_id"])?;
    text(&v["operator"])?;
    if !restart && text(&v["reason"])?.trim().is_empty() {
        return Err(invalid("operator release reason required"));
    }
    observation_time(
        &v[if restart {
            "confirmed_unix"
        } else {
            "released_unix"
        }],
    )
}
pub(crate) struct Namespace {
    root: PathBuf,
    _lock: File,
    pub admission: Option<Value>,
}
impl Namespace {
    pub fn acquire() -> Result<Self> {
        // This supervisor is single-threaded. Copy passwd data before another libc call.
        let entry = unsafe { libc::getpwuid(libc::getuid()) };
        if entry.is_null() {
            return Err(invalid("passwd user home unavailable"));
        }
        let home = unsafe { CStr::from_ptr((*entry).pw_dir) }
            .to_bytes()
            .to_vec();
        let root = PathBuf::from(std::ffi::OsStr::from_bytes(&home))
            .join(".local/state/godfieldbots/hpo-gpu-admission-v1");
        Self::open(&root)
    }
    #[cfg(test)]
    pub fn private_test(root: &Path) -> Result<Self> {
        Self::open(root)
    }
    fn open(root: &Path) -> Result<Self> {
        ensure_private(root)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(root.join("lock"))?;
        let info = lock.metadata()?;
        if !info.is_file() || info.uid() != unsafe { libc::getuid() } || info.nlink() != 1 {
            return Err(invalid("invalid SAME namespace lock"));
        }
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(invalid("GPU admission busy"));
        }
        let jobs = root.join("jobs");
        if !jobs.exists() {
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            builder.create(&jobs)?;
            sync_dir(root)?;
        }
        private_directory(&jobs)?;
        let owner = Self {
            root: root.to_path_buf(),
            _lock: lock,
            admission: None,
        };
        owner.scan()?;
        Ok(owner)
    }
    fn scan(&self) -> Result<()> {
        for entry in fs::read_dir(&self.root)? {
            if !["lock", "current.json", "jobs"]
                .contains(&entry?.file_name().to_string_lossy().as_ref())
            {
                return Err(invalid("uncertain GPU namespace entry"));
            }
        }
        let mut jobs = std::collections::BTreeMap::new();
        let mut unresolved = false;
        for entry in fs::read_dir(self.root.join("jobs"))? {
            let entry = entry?;
            let path = entry.path();
            let ticket = entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("invalid GPU ticket"))?;
            hex(&json!(ticket), 32)?;
            private_directory(&path)?;
            let names = fs::read_dir(&path)?
                .map(|e| e.map(|p| p.file_name().to_string_lossy().into_owned()))
                .collect::<std::io::Result<std::collections::BTreeSet<_>>>()?;
            let allowed = std::collections::BTreeSet::from([
                "admission.json".to_owned(),
                "termination.json".to_owned(),
                "restart.json".to_owned(),
                "release.json".to_owned(),
            ]);
            if !names.contains("admission.json") || !names.is_subset(&allowed) || names.len() > 2 {
                return Err(invalid("incomplete/conflicting GPU records"));
            }
            let (admission, _) = read_owned(&path.join("admission.json"), None)?;
            match admission["schema_version"].as_u64() {
                Some(1) => legacy_admission(&admission, &ticket)?,
                Some(2) => global_admission(&admission, &ticket)?,
                _ => return Err(invalid("unknown GPU admission schema")),
            }
            let mut resolved = false;
            for name in ["termination.json", "restart.json", "release.json"] {
                if names.contains(name) {
                    let (receipt, _) = read_owned(&path.join(name), None)?;
                    match name {
                        "termination.json" => {
                            if admission["schema_version"] == 1 {
                                legacy_termination(&receipt, &admission)?;
                            } else {
                                global_termination(&receipt, &admission)?;
                            }
                        }
                        "restart.json" => recovery(&receipt, &admission, true)?,
                        _ => recovery(&receipt, &admission, false)?,
                    }
                    resync(&path.join(name))?;
                    resolved = true;
                }
            }
            unresolved |= !resolved;
            jobs.insert(ticket, admission);
        }
        let current = self.root.join("current.json");
        if current.exists() || current.is_symlink() {
            let (pointer, _) = read_owned(&current, None)?;
            object(&pointer, &["schema_version", "ticket", "admission_sha256"])?;
            if pointer["schema_version"] != 1 {
                return Err(invalid("invalid GPU current index schema"));
            }
            let ticket = hex(&pointer["ticket"], 32)?;
            hex(&pointer["admission_sha256"], 64)?;
            if jobs
                .get(ticket)
                .is_none_or(|a| pointer["admission_sha256"] != digest(a))
            {
                return Err(invalid("GPU current index identity mismatch"));
            }
            resync(&current)?;
        } else if !jobs.is_empty() && !unresolved {
            return Err(invalid("missing GPU current index"));
        }
        if unresolved {
            return Err(invalid(
                "RECOVERY_REQUIRED: unresolved original GPU admission",
            ));
        }
        Ok(())
    }
    pub fn admit(&mut self, local: &Value) -> Result<Value> {
        if self.admission.is_some() {
            return Err(invalid("namespace already admitted"));
        }
        let ticket = hex(&local["global_ticket"], 32)?;
        let admission = json!({"schema_version":2,"kind":"rust-player-gpu-admission-v1","ticket":ticket,"context":local["context"],"boot_id":boot_id()?,"owner_pid":std::process::id(),"admitted_unix":unix_ns()?/1_000_000_000,"host_admission":local});
        global_admission(&admission, ticket)?;
        let directory = self.root.join("jobs").join(ticket);
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(&directory)?;
        sync_dir(directory.parent().unwrap())?;
        // Retain the pending original before any fallible durable publication.
        self.admission = Some(admission.clone());
        write(&directory.join("admission.json"), &admission, true)?;
        write(
            &self.root.join("current.json"),
            &json!({"schema_version":1,"ticket":ticket,"admission_sha256":digest(&admission)}),
            false,
        )?;
        Ok(admission)
    }
    pub fn termination_path(&self) -> Result<PathBuf> {
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("no original namespace admission"))?;
        Ok(self
            .root
            .join("jobs")
            .join(text(&admission["ticket"])?)
            .join("termination.json"))
    }
    pub fn confirm(&self, exit_record: &Value, actual: &Retirement) -> Result<Value> {
        let a = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("no original global admission"))?;
        if !actual.descendants_reaped
            || exit_record["host_original"] != serde_json::to_value(&actual.origin).unwrap()
            || exit_record["wait"] != serde_json::to_value(&actual.wait).unwrap()
            || exit_record["descendants"] != serde_json::to_value(&actual.descendants).unwrap()
            || exit_record["signals"] != serde_json::to_value(&actual.signals).unwrap()
        {
            return Err(invalid(
                "physical confirmation requires SAME actual retirement",
            ));
        }
        let receipt = json!({"schema_version":3,"kind":"rust-player-gpu-termination-v1","ticket":a["ticket"],"admission_sha256":digest(a),"context_sha256":digest(&a["context"]),"boot_id":a["boot_id"],"confirmed_unix":unix_ns()?/1_000_000_000,"exit_sha256":digest(exit_record),"exit_record":exit_record});
        global_termination(&receipt, a)?;
        let directory = self.root.join("jobs").join(text(&a["ticket"])?);
        let (current, _) = read_owned(&directory.join("admission.json"), None)?;
        let (pointer, _) = read_owned(&self.root.join("current.json"), None)?;
        if current != *a
            || pointer
                != json!({"schema_version":1,"ticket":a["ticket"],"admission_sha256":digest(a)})
        {
            return Err(invalid("original global admission changed"));
        }
        write(&directory.join("termination.json"), &receipt, true)?;
        Ok(receipt)
    }
}
