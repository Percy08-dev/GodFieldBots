//! Standalone original-host supervisor. This target never imports gf_native/PyO3.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io,
    os::unix::process::ExitStatusExt,
    process::ExitStatus,
    time::{Duration, Instant},
};

#[path = "gf_player_supervisor/process.rs"]
mod process;
use process::*;

#[path = "gf_player_supervisor/namespace.rs"]
mod namespace;
#[path = "gf_player_supervisor/records.rs"]
mod records;

#[path = "gf_player_supervisor/packet.rs"]
mod packet;

#[path = "gf_player_supervisor/session.rs"]
mod session;

#[cfg(feature = "allocator-mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

type Result<T> = io::Result<T>;
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn build_info() -> Value {
    json!({"version":env!("CARGO_PKG_VERSION"),"rustc":env!("GFB_NATIVE_RUSTC"),
        "target":env!("GFB_NATIVE_TARGET"),"profile":env!("GFB_NATIVE_PROFILE"),
        "rustflags":env!("GFB_NATIVE_RUSTFLAGS"),"cargo_lock_sha256":env!("GFB_NATIVE_CARGO_LOCK_SHA256"),
        "native_source_sha256":env!("GFB_NATIVE_SOURCE_SHA256"),"abi":"cp314-gil",
        "allocator":if cfg!(feature="allocator-mimalloc") {"mimalloc"} else {"system"}})
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WaitRecord {
    exit_code: Option<i32>,
    signal: Option<i32>,
    core_dumped: bool,
}
fn wait_record(status: ExitStatus) -> WaitRecord {
    WaitRecord {
        exit_code: status.code(),
        signal: status.signal(),
        core_dumped: status.core_dumped(),
    }
}

fn monotonic() -> Result<f64> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(value.tv_sec as f64 + value.tv_nsec as f64 * 1e-9)
}
fn deadline_from_bits(bits: &str) -> Result<(f64, Instant)> {
    if bits.len() != 16
        || !bits
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("invalid original monotonic deadline bits"));
    }
    let original =
        f64::from_bits(u64::from_str_radix(bits, 16).map_err(|_| invalid("invalid deadline"))?);
    if !original.is_finite() || original <= 0.0 {
        return Err(invalid("nonfinite/nonpositive deadline"));
    }
    let sampled = Instant::now();
    let remaining = original - monotonic()?;
    let duration = Duration::try_from_secs_f64(remaining)
        .map_err(|_| invalid("expired/unrepresentable deadline"))?;
    let deadline = sampled
        .checked_add(duration)
        .ok_or_else(|| invalid("unrepresentable original deadline"))?;
    Ok((original, deadline))
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--build-info" {
        println!(
            "{}",
            json!({"schema_version":1,"kind":"rust-player-supervisor-build-v1","build":build_info()})
        );
        return;
    }
    if args.len() == 2 && args[0] == "--launch-fd" {
        let result = (|| -> Result<()> {
            let fd: i32 = args[1]
                .to_str()
                .ok_or_else(|| invalid("invalid launch FD"))?
                .parse()
                .map_err(|_| invalid("invalid launch FD"))?;
            if fd < 3 {
                return Err(invalid("original private launch FD required"));
            }
            session::Supervisor::new(session::LaunchOwner::take(fd)?)?.run()
        })();
        if result.is_ok() {
            return;
        }
        eprintln!("rust-player supervision failed; inspect retained protocol records");
        std::process::exit(1);
    }
    eprintln!("rust-player supervisor requires its authenticated fixed host bootstrap");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    // A WNOHANG=0 result must never turn a live descendant into a reap receipt.
    #[test]
    fn actual_child_wait_preserves_signal_and_does_not_infer_exit_from_eof() {
        let _serial = PROCESS_TEST.lock().unwrap();
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("exec sleep 30")
            .spawn()
            .unwrap();
        assert!(child.try_wait().unwrap().is_none());
        unsafe {
            libc::kill(child.id() as i32, libc::SIGTERM);
        }
        let status = child.wait().unwrap();
        let actual = wait_record(status);
        assert_eq!(actual.signal, Some(libc::SIGTERM));
        assert_eq!(actual.exit_code, None);
        assert_eq!(status.signal(), actual.signal);
    }

    #[test]
    fn original_monotonic_deadline_bits_reject_nonfinite_and_overflow() {
        assert!(deadline_from_bits("7ff0000000000000").is_err());
        assert!(deadline_from_bits("7ff8000000000000").is_err());
        assert!(deadline_from_bits("bff0000000000000").is_err());
        assert!(deadline_from_bits("7fefffffffffffff").is_err());
        let now = monotonic().unwrap();
        let bits = format!("{:016x}", (now + 1.0).to_bits());
        let (original, instant) = deadline_from_bits(&bits).unwrap();
        assert_eq!(original.to_bits(), (now + 1.0).to_bits());
        assert!(instant > std::time::Instant::now());
    }
    // Real C descendants exercise Linux adoption, not a PID-only process double.
    fn c_fixture() -> std::path::PathBuf {
        static BINARY: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        BINARY
            .get_or_init(|| {
                let directory =
                    std::env::temp_dir().join(format!("task5b-c-fixture-{}", std::process::id()));
                std::fs::create_dir_all(&directory).unwrap();
                let source = directory.join("host.c");
                std::fs::write(
                    &source,
                    r#"#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int main(int argc,char**argv) {
 int mode=argc>1?atoi(argv[1]):0;
 if(mode==2) {signal(SIGTERM,SIG_IGN); for(;;)pause();}
 if(mode==5)return 7;
 pid_t original=getpid();
 sigset_t blocked;sigemptyset(&blocked);sigaddset(&blocked,SIGTERM);
 if(mode==4||mode==6)sigprocmask(SIG_BLOCK,&blocked,0);
 pid_t child=fork();
 if(child<0)return 90;
 if(child==0) {
  if(mode==4||mode==6) {int received=0;sigwait(&blocked,&received);
   if(mode==6){setsid();struct timespec escape_delay={0,300000000};nanosleep(&escape_delay,0);}
   _exit(getpgid(original)==original?19:81);
  }
  if(mode==3)setsid();
  if(mode==1) {signal(SIGTERM,SIG_IGN);for(;;)pause();}
  struct timespec delay={0,50000000};nanosleep(&delay,0);_exit(19);
 }
 return 7;
}"#,
                )
                .unwrap();
                let binary = directory.join("host");
                assert!(
                    std::process::Command::new("cc")
                        .args(["-O2", "-o"])
                        .arg(&binary)
                        .arg(&source)
                        .status()
                        .unwrap()
                        .success()
                );
                binary
            })
            .clone()
    }
    fn cleanup_clock() -> CleanupClock {
        let now = monotonic().unwrap();
        CleanupClock::new(now + 0.1, now + 0.25, now + 1.0).unwrap()
    }
    static PROCESS_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn original_wait_and_positive_adopted_c_child_reap_are_separate_facts() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("0");
        let mut owner = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let outcome = owner.retire(cleanup_clock()).unwrap();
        assert_eq!(outcome.wait.exit_code, Some(7));
        assert_eq!(outcome.descendants.len(), 1);
        assert_eq!(outcome.descendants[0].wait.exit_code, Some(19));
        assert!(outcome.descendants_reaped);
        assert!(outcome.signals.is_empty());
    }
    #[test]
    fn actual_live_adopted_c_child_requires_term_then_kill_and_positive_wait() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("1");
        let mut owner = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let outcome = owner.retire(cleanup_clock()).unwrap();
        assert_eq!(outcome.wait.exit_code, Some(7));
        assert_eq!(outcome.descendants[0].wait.signal, Some(libc::SIGKILL));
        assert_eq!(
            outcome
                .signals
                .iter()
                .map(|item| item.signal)
                .collect::<Vec<_>>(),
            vec![libc::SIGTERM, libc::SIGKILL]
        );
        assert!(outcome.descendants_reaped);
    }
    #[test]
    fn original_sigkill_wait_cannot_be_replaced_by_control_eof() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("2");
        let mut owner = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        assert!(!owner.observe_exit().unwrap());
        let outcome = owner.retire(cleanup_clock()).unwrap();
        assert_eq!(outcome.wait.signal, Some(libc::SIGKILL));
        assert!(outcome.descendants_reaped);
    }
    #[test]
    fn adopted_outgroup_child_is_reaped_but_never_positive_containment() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("3");
        let mut owner = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let outcome = owner.retire(cleanup_clock()).unwrap();
        assert!(!outcome.descendants_reaped);
        assert_eq!(outcome.descendants.len(), 1);
        assert!(outcome.signals.is_empty());
    }

    #[test]
    fn same_ofd_lock_survives_close_only_transfer_but_unlock_is_never_reacquired() {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        use std::os::unix::fs::MetadataExt;
        let path = std::env::temp_dir().join(format!("task5b-parent-lock-{}", std::process::id()));
        let original = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(original.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let metadata = original.metadata().unwrap();
        let duplicate = unsafe {
            OwnedFd::from_raw_fd(libc::fcntl(original.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3))
        };
        let held = HeldLock::adopt(duplicate, "parent", metadata.dev(), metadata.ino()).unwrap();
        drop(original);
        held.recheck().unwrap();
        assert_eq!(
            unsafe { libc::flock(held.fd.as_raw_fd(), libc::LOCK_UN) },
            0
        );
        assert!(held.recheck().is_err());
        let competitor = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(competitor.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        std::fs::remove_file(path).unwrap();
    }

    fn python() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(".venv/bin/python")
    }
    fn local_fixture(host: &HostIdentity, ticket: &str) -> Value {
        let now = monotonic().unwrap();
        let mut deadlines = json!({});
        for (index, key) in records::DEADLINES.iter().enumerate() {
            deadlines[key] = json!(format!("{:016x}", (now + index as f64 + 1.0).to_bits()));
        }
        json!({"schema_version":1,"kind":"rust-player-admission-v1","supervision_id":"1".repeat(32),"invocation_id":"2".repeat(32),"nonce":"3".repeat(32),
            "supervisor":{"pid":std::process::id(),"start_ticks":start_ticks(std::process::id()).unwrap()},"host":host,
            "context":{"campaign_id":"campaign","operation_id":"operation","kind":"training","run_id":"run","evaluation_id":null},
            "budget":{"managed_ram_bytes":1,"managed_cuda_bytes":1,"game_concurrency":1,"player_worker_threads":1,"max_batch_size":1,"storage_bytes":1,"source_workspace_bytes":1},
            "operation_id":"operation","run_id":"run","config_sha256":"a".repeat(64),"runtime_binding_sha256":"b".repeat(64),"deadlines":deadlines,
            "producer":{"schema_version":1,"kind":"rust-player-producer-v1","python_source_digest":"c".repeat(64),"python_runtime_digest":"d".repeat(64),"extension_sha256":"e".repeat(64),"build":build_info(),"supervisor_sha256":"f".repeat(64),"sampling_runtime_sha256":"a".repeat(64)},"global_ticket":ticket})
    }
    fn exit_fixture(local: &Value, actual: &Retirement) -> Value {
        json!({"schema_version":1,"kind":"rust-player-exit-v1","supervision_id":local["supervision_id"],"invocation_id":local["invocation_id"],"nonce":local["nonce"],
            "admission_sha256":records::digest(local),"prepared_sha256":null,"host_original":actual.origin,"wait":actual.wait,
            "descendants":actual.descendants,"signals":actual.signals,"descendants_reaped":actual.descendants_reaped,"cooperative":false,"finished_unix_ns":records::unix_ns().unwrap()})
    }
    const LEGACY_HELPER: &str = r#"
import sys
from pathlib import Path
from gf_training import hpo_gpu_lease as gpu
from tests.test_hpo_gpu_lease import spawn_worker
gpu._GLOBAL_ROOT=Path(sys.argv[1])
mode=sys.argv[2]
if mode=='busy':
 try: gpu.admit_gpu_job(gpu.JobContext('legacy','operation','training','run',None),global_cuda_bytes=1,max_workers=1)
 except gpu.AdmissionBusy: sys.exit(0)
 raise RuntimeError('namespace split while Rust owner held flock')
with gpu.admit_gpu_job(gpu.JobContext('legacy','operation','training','run',None),global_cuda_bytes=1,max_workers=1) as job:
 owner,lease,process=spawn_worker(job,residency=1)
 process.wait(timeout=5);lease.reaped(process)
 if mode!='orphan': job.confirm_termination()
"#;
    #[test]
    fn namespace_real_legacy_new_contention_confirmation_and_orphan_quarantine() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let root = std::env::temp_dir().join(format!(
            "task5b-namespace-{}",
            records::random_id().unwrap()
        ));
        let run = |mode: &str| {
            std::process::Command::new(python())
                .args(["-c", LEGACY_HELPER])
                .arg(&root)
                .arg(mode)
                .current_dir(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .unwrap(),
                )
                .status()
                .unwrap()
        };
        assert!(run("confirm").success()); // actual old resource ledger/receipt, not a fabricated list
        let mut namespace = namespace::Namespace::private_test(&root).unwrap();
        assert!(run("busy").success());
        let mut command = std::process::Command::new(c_fixture());
        command.arg("0");
        let mut host = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let local = local_fixture(&host.identity, &records::random_id().unwrap());
        namespace.admit(&local).unwrap();
        let actual = host.retire(cleanup_clock()).unwrap();
        let exit = exit_fixture(&local, &actual);
        namespace.confirm(&exit, &actual).unwrap();
        drop(namespace);
        assert!(run("confirm").success()); // old reader consumes actual Rust admission2/termination3
        assert!(run("orphan").success());
        assert!(namespace::Namespace::private_test(&root).is_err()); // dead PID cannot clear it
    }
    #[test]
    fn canonical_new_protocol_hashes_match_python_for_embedded_and_observed_scalars() {
        let _serial = PROCESS_TEST.lock().unwrap();
        let value = json!({"schema_version":1,"nullable":null,"bits":"3ff0000000000000","integer":18446744073709551615u64,
            "observed_unix":records::unix_ns().unwrap() as f64 / 1e9,"nested":{"truth":true,"text":"資料😀"}});
        let expected = records::digest(&value);
        let encoded = serde_json::to_string(&value).unwrap();
        let output=std::process::Command::new(python()).args(["-c","import sys,json;from gf_common.domain import canonical,digest;v=json.loads(sys.argv[1]);print(digest(v));print(canonical(v))"]).arg(&encoded).output().unwrap();
        assert!(output.status.success());
        let text = String::from_utf8(output.stdout).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some(expected.as_str()));
        assert_eq!(lines.next(), Some(encoded.as_str()));
    }

    #[test]
    fn kernel_seqpacket_credentials_nonce_truncation_and_fd_count_are_checked() {
        use std::os::fd::AsRawFd;
        let _serial = PROCESS_TEST.lock().unwrap();
        let (mut owner, peer) = packet::Channel::pair(std::process::id(), "1", "2", "3").unwrap();
        let frame = json!({"v":1,"kind":"ready","supervision_id":"1","invocation_id":"2","nonce":"wrong","seq":0});
        packet::send_raw(peer.as_raw_fd(), &serde_json::to_vec(&frame).unwrap(), &[]).unwrap();
        assert!(owner.receive().is_err());
        packet::send_raw(peer.as_raw_fd(), &vec![b' '; 65537], &[]).unwrap();
        assert!(owner.receive().is_err());
        packet::send_raw(peer.as_raw_fd(), b"{", &[]).unwrap();
        assert!(owner.receive().is_err());
        let proper = json!({"v":1,"kind":"ready","supervision_id":"1","invocation_id":"2","nonce":"3","seq":0});
        packet::send_raw(peer.as_raw_fd(), &serde_json::to_vec(&proper).unwrap(), &[]).unwrap();
        let actual = owner.receive().unwrap().unwrap();
        assert_eq!(actual.value, proper);
        assert!(actual.fds.is_empty());
        // A process inheriting the endpoint still cannot impersonate original child credentials.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            let bytes = b"{}";
            unsafe {
                libc::send(peer.as_raw_fd(), bytes.as_ptr().cast(), bytes.len(), 0);
                libc::_exit(0);
            }
        }
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(owner.receive().is_err());
    }

    fn run_host_cases(cases: &[&str]) {
        use std::os::fd::{AsRawFd, IntoRawFd};
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
        let _serial = PROCESS_TEST.lock().unwrap();
        let output=std::process::Command::new(python()).args(["-c","import json;from gf_training.native_backend import binary_identity;print(json.dumps(binary_identity()))"]).output().unwrap();
        assert!(output.status.success());
        let identity: Value = serde_json::from_slice(&output.stdout).unwrap();
        let mut build = json!({});
        for key in records::BUILD {
            build[key] = identity["build"][key].clone();
        }
        assert_eq!(
            build,
            build_info(),
            "actual loaded extension must match standalone source/profile before host tests"
        );
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        for &mode in cases {
            let ordinary = ["ordinary-normal", "cuda-numeric", "cpu-source", "cpu-common"].contains(&mode);
            let runtime_mode = if ordinary { "ordinary" } else { "hpo" };
            let root = std::env::temp_dir().join(format!(
                "{}-{mode}-{}",
                if mode == "cpu-common" { "task5c1b-common" } else { "task5b-session" }, records::random_id().unwrap()
            ));
            records::ensure_private(&root).unwrap();
            let parent = root.join("parent");
            std::fs::create_dir(&parent).unwrap();
            let namespace_root = root.join("global");
            let now = monotonic().unwrap();
            let bits = |n: f64| format!("{:016x}", n.to_bits());
            let deadlines = if mode == "cpu-common" {
                json!({"admission_bits":bits(now+480.0),"execution_bits":bits(now+540.0),"term_bits":bits(now+570.0),"kill_bits":bits(now+585.0),"finish_bits":bits(now+600.0)})
            } else if mode == "cuda-numeric" || mode == "cpu-source" {
                json!({"admission_bits":bits(now+30.0),"execution_bits":bits(now+80.0),"term_bits":bits(now+90.0),"kill_bits":bits(now+100.0),"finish_bits":bits(now+120.0)})
            } else {
                json!({"admission_bits":bits(now+1.5),"execution_bits":bits(now+2.0),"term_bits":bits(now+2.5),"kill_bits":bits(now+3.0),"finish_bits":bits(now+5.0)})
            };
            let context = if ordinary {
                Value::Null
            } else {
                json!({"campaign_id":"campaign","operation_id":"operation","kind":"training","run_id":"run","evaluation_id":null})
            };
            let operation_id = if ["cuda-numeric", "cpu-source", "cpu-common"].contains(&mode) {
                format!("{mode}-{}", records::random_id().unwrap())
            } else {
                "operation".to_owned()
            };
            let mut operation = json!({"schema_version":1,"kind":"rust-player-operation-v1","operation_id":operation_id,"run_id":"run","mode":runtime_mode,"operation_type":if ordinary {"ordinary-train"} else {"hpo-train"},"config_path":root.join("config.json"),"config_sha256":"a".repeat(64),"runtime_binding_sha256":"b".repeat(64),"deadlines":deadlines,"parent_directory":parent,"expected_context":context});
            if mode == "cpu-source" || mode == "cpu-common" {
                let helper = if mode == "cpu-common" { "prepare_cpu_common" } else { "prepare_cpu_source" };
                let code = format!("import json,sys;from pathlib import Path;from tests.test_rust_player_spine import {helper};print(json.dumps({helper}(Path(sys.argv[1]),Path(sys.argv[2]))))");
                let output = std::process::Command::new(python()).args(["-c", &code])
                    .arg(&root).arg(std::env::current_exe().unwrap()).current_dir(repo).output().unwrap();
                assert!(output.status.success(), "CPU Source setup: {}", String::from_utf8_lossy(&output.stderr));
                let facts: Value = serde_json::from_slice(&output.stdout).unwrap();
                operation["config_sha256"] = facts["config_sha256"].clone();
                operation["runtime_binding_sha256"] = facts["runtime_binding_sha256"].clone();
                operation["schema_version"] = json!(2);
                operation["kind"] = json!("rust-player-operation-v2");
                operation["execution_request"] = json!({"mode":"fresh","pause_after_update":null});
                if mode == "cpu-common" { operation["config_path"] = json!(parent.join("config.json")); }
            }
            let operation_path = root.join("operation.json");
            let operation_sha = records::write(&operation_path, &operation, true).unwrap();
            let campaign = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(if mode == "cpu-common" { parent.join(".lock") } else { root.join("campaign.lock") })
                .unwrap();
            assert_eq!(
                unsafe { libc::flock(campaign.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
                0
            );
            let metadata = campaign.metadata().unwrap();
            let mapping = if ordinary && mode != "cpu-common" {
                drop(campaign);
                json!([])
            } else {
                json!([{"lock_kind":if mode=="cpu-common"{"parent"}else{"campaign"},"fd":campaign.into_raw_fd(),"device":metadata.dev(),"inode":metadata.ino()}])
            };
            let launch = json!({"schema_version":1,"kind":"rust-player-launch-v1","supervision_id":records::random_id().unwrap(),"python_executable":python(),"operation_path":operation_path,"operation_sha256":operation_sha,"record_directory":root,"mode":runtime_mode,"session_deadlines":{"finish_bits":bits(now+if mode=="session-expire"{5.3}else if mode=="cpu-common"{600.0}else if mode=="cuda-numeric" || mode=="cpu-source"{120.0}else{10.0})},"clock_binding":{"boot_id":records::boot_id().unwrap(),"monotonic_before_bits":bits(now),"monotonic_after_bits":bits(now),"unix_anchor_bits":bits(records::unix_ns().unwrap() as f64/1e9)},"binding":{"python_abi":identity["python_abi"],"extension_sha256":identity["extension_sha256"],"build":build},"locks":mapping});
            let launch_path = root.join("launch.json");
            records::write(&launch_path, &launch, true).unwrap();
            let launch_file = std::fs::File::open(&launch_path).unwrap();
            let mut supervisor = session::Supervisor::new(
                session::LaunchOwner::take(launch_file.into_raw_fd()).unwrap(),
            )
            .unwrap();
            supervisor.test_command = Some(vec![
                python().to_string_lossy().into_owned(),
                repo.join("tests/test_rust_player_supervisor.py")
                    .to_string_lossy()
                    .into_owned(),
                "_host".to_owned(),
                mode.to_owned(),
            ]);
            supervisor.test_namespace = Some(namespace_root.clone());
            let result = supervisor.run();
            if ["cold-expire", "session-expire"].contains(&mode) {
                let cutoff = if mode == "cold-expire" {
                    now + 5.0
                } else {
                    now + 5.3
                };
                assert!(
                    monotonic().unwrap() >= cutoff,
                    "old milestones cut short hard-cutoff authority"
                );
            }
            // TEST-ONLY reap after a hard-cutoff uncertainty. No production grace or
            // successful receipt is minted from this separate fixture-cleanup wait.
            let mut fixture_pids = std::collections::BTreeSet::new();
            for entry in std::fs::read_dir(&root).unwrap() {
                let entry = entry.unwrap();
                if entry.file_name().to_string_lossy().contains("-started") {
                    if let Ok(pid) = std::fs::read_to_string(entry.path())
                        .unwrap()
                        .parse::<i32>()
                    {
                        fixture_pids.insert(pid);
                    }
                }
            }
            for pid in fixture_pids {
                let cleanup_finish = std::time::Instant::now() + std::time::Duration::from_secs(1);
                loop {
                    let mut status = 0;
                    let outcome = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
                    if outcome == pid {
                        assert_eq!(libc::WTERMSIG(status), libc::SIGKILL);
                        break;
                    }
                    if outcome == -1 {
                        assert_eq!(
                            std::io::Error::last_os_error().raw_os_error(),
                            Some(libc::ECHILD)
                        );
                        break;
                    }
                    assert!(
                        std::time::Instant::now() < cleanup_finish,
                        "fixture child survived hard containment"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
            assert_eq!(
                result.is_ok(),
                [
                    "normal",
                    "ordinary-normal",
                    "cold-delay",
                    "two-operation",
                    "legitimate-stop",
                    "cpu-common"
                ]
                .contains(&mode),
                "mode={mode}: {result:?}"
            );
            let exits = std::fs::read_dir(&root)
                .unwrap()
                .filter_map(|p| {
                    let p = p.unwrap().path();
                    p.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with("exit-")
                        .then_some(p)
                })
                .collect::<Vec<_>>();
            if mode == "handshake-exit" || mode == "lost-transfer-ack" {
                assert!(exits.is_empty());
                assert!(!namespace_root.exists());
                if mode == "lost-transfer-ack" {
                    assert!(!root.join("component-prepared.json").exists());
                    assert!(!root.join("component-finalizer-output.json").exists());
                    assert!(!root.join("cold-started").exists());
                    assert!(!std::fs::read_dir(&root).unwrap().any(|entry| {
                        entry
                            .unwrap()
                            .file_name()
                            .to_string_lossy()
                            .starts_with("admission-")
                    }));
                }
            } else if mode == "exit-write-fail" {
                assert!(exits.is_empty());
                assert!(namespace::Namespace::private_test(&namespace_root).is_err());
            } else {
                assert_eq!(
                    exits.len(),
                    if mode == "two-operation" { 2 } else { 1 },
                    "{mode}"
                );
                if mode == "two-operation" {
                    let first = records::read_owned(&exits[0], Some(&root)).unwrap().0;
                    let second = records::read_owned(&exits[1], Some(&root)).unwrap().0;
                    assert!(first["cooperative"] == true && second["cooperative"] == true);
                    assert_ne!(
                        first["host_original"]["pid"],
                        second["host_original"]["pid"]
                    );
                    assert!(root.join("second-component-finalizer-output.json").exists());
                }
                let (exit, _) = records::read_owned(&exits[0], Some(&root)).unwrap();
                assert_eq!(
                    exit["cooperative"],
                    json!(
                        [
                            "normal",
                            "ordinary-normal",
                            "cold-fail",
                            "prepared-tamper",
                            "cold-delay",
                            "cold-expire",
                            "two-operation",
                            "same-operation",
                            "unfinalized-bind",
                            "session-expire",
                            "legitimate-stop",
                            "cpu-common"
                        ]
                        .contains(&mode)
                    )
                );
                assert_eq!(exit["descendants_reaped"], true);
                if ordinary {
                    assert!(!namespace_root.exists());
                } else {
                    assert!(namespace::Namespace::private_test(&namespace_root).is_ok());
                }
                if mode == "control-eof-kill" {
                    assert_eq!(exit["wait"]["signal"], 9);
                }
            }
            if [
                "exit-before-prepare",
                "wrong-nonce",
                "control-eof-kill",
                "prepared-nonzero",
                "prepared-tamper",
                "exit-write-fail",
                "duplicate-stop-ack",
                "unsolicited-stop-ack",
                "handshake-exit",
            ]
            .contains(&mode)
            {
                assert!(
                    !root.join("cold-started").exists(),
                    "{mode}: forbidden continuation"
                );
            }
            assert_eq!(
                root.join("component-finalizer-output.json").exists(),
                [
                    "normal",
                    "ordinary-normal",
                    "cold-delay",
                    "two-operation",
                    "same-operation",
                    "session-expire",
                    "legitimate-stop"
                ]
                .contains(&mode)
            );
            if mode == "cuda-numeric" {
                let result = root.join("cuda-numerical-component/result.json");
                assert!(
                    result.exists(),
                    "actual CUDA numerical component did not finish"
                );
                let output = std::process::Command::new(python())
                    .args(["-c", "import sys;from pathlib import Path;from tests.test_rust_player_learning import audit_cuda_saved_pairs_cpu;audit_cuda_saved_pairs_cpu(Path(sys.argv[1]))"])
                    .arg(&result).current_dir(repo).env("CUDA_VISIBLE_DEVICES", "").output().unwrap();
                assert!(
                    output.status.success(),
                    "fresh CPU-only audit failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                eprintln!("CUDA numerical component artifacts: {}", result.display());
            }
            if mode == "cpu-source" {
                assert!(root.join("cpu-source-result.json").exists(), "actual CPU Source component did not finish: {}", root.display());
                assert!(!root.join("component-prepared.json").exists());
                assert!(!root.join("cold-started").exists());
                eprintln!("CPU Source component artifacts: {}", root.display());
            }
            if mode == "cpu-common" {
                let entries = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
                assert_eq!(entries.iter().filter(|p|p.file_name().unwrap().to_string_lossy().starts_with("completion-")).count(),1);
                assert_eq!(entries.iter().filter(|p|p.file_name().unwrap().to_string_lossy().starts_with("cold-audit-")).count(),1);
                assert!(root.join("parent/generations/000001.json").exists());
                eprintln!("COMMON PPO CPU artifacts: {}",root.display());
            }
        }
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn actual_host_prepared_original_reap_cold_pair_and_fault_windows() {
        run_host_cases(&[
            "ordinary-normal",
            "normal",
            "exit-before-prepare",
            "lost-transfer-ack",
            "wrong-nonce",
            "control-eof-kill",
            "prepared-nonzero",
            "cold-fail",
            "prepared-tamper",
            "exit-write-fail",
            "handshake-exit",
        ]);
    }
    #[test]
    #[ignore = "requires Root's explicit CUDA numerical component budget"]
    fn local_cuda_numerical_component_and_fresh_cpu_saved_state_audit() {
        assert_eq!(
            std::env::var("GFB_RUN_RUST_PLAYER_CUDA_NUMERIC").as_deref(),
            Ok("1")
        );
        run_host_cases(&["cuda-numeric"]);
    }
    #[test]
    #[ignore = "requires matched release extension and approved 16-game CPU component budget"]
    fn actual_cpu_native_writer_source_persisted_component() {
        run_host_cases(&["cpu-source"]);
    }
    #[test]
    #[ignore = "requires matched runtime and approved 600s one-update ordinary CPU scope"]
    fn actual_common_ppo_one_update_and_independent_cold_cpu_audit() {
        run_host_cases(&["cpu-common"]);
    }
    #[test]
    #[ignore = "requires Root-approved full Model4 pause/resume component scope and matching runtime"]
    fn actual_hpo_model4_pause_resume_and_cold_parent_component() {
        let output = std::process::Command::new(python())
            .args(["-m", "tests.test_rust_player_spine", "_prepare_hpo_exec"])
            .arg(std::env::current_exe().unwrap())
            .current_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
            .status().unwrap();
        assert!(output.success(), "actual management→exec HPO component failed");
    }
    #[test]
    #[ignore = "private inherited-launch child, only management→exec"]
    fn actual_hpo_resume_inherited_child() {
        let fd: i32 = std::env::var("GFB_HPO_COMPONENT_LAUNCH_FD").expect("actual inherited launch FD").parse().unwrap();
        let launch = session::LaunchOwner::take(fd).unwrap();
        let root = std::path::PathBuf::from(launch.value["record_directory"].as_str().unwrap());
        let mut supervisor = session::Supervisor::new(launch).unwrap();
        supervisor.test_command = Some(vec![python().to_string_lossy().into_owned(),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("tests/test_rust_player_supervisor.py").to_string_lossy().into_owned(),
            "_host".to_owned(), "hpo-resume".to_owned()]);
        // REAL default shared GPU namespace; no isolated test namespace override.
        let result = supervisor.run();
        assert!(result.is_ok(), "actual HPO component {}: {result:?}", root.display());
        let paths = std::fs::read_dir(&root).unwrap().map(|e|e.unwrap().path()).collect::<Vec<_>>();
        for prefix in ["admission-", "exit-", "completion-", "cold-audit-"] {
            assert_eq!(paths.iter().filter(|p|p.file_name().unwrap().to_string_lossy().starts_with(prefix)).count(),2,"{prefix}");
        }
        assert_eq!(paths.iter().filter(|p|p.file_name().unwrap().to_string_lossy().starts_with("parent-metadata-")).count(),1);
        eprintln!("HPO ACTUAL PAUSE/RESUME COMPONENT ARTIFACTS {}", root.display());
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn fix1_delayed_cold_audit_uses_remaining_original_finish() {
        run_host_cases(&["cold-delay"]);
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn fix1_two_operations_cross_previous_finish_under_original_session() {
        run_host_cases(&["two-operation"]);
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn fix1_duplicate_stop_ack_is_ineligible() {
        run_host_cases(&["duplicate-stop-ack"]);
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn fix1_unsolicited_stop_ack_is_ineligible() {
        run_host_cases(&["unsolicited-stop-ack"]);
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn fix1_legitimate_stop_ack_and_unfinalized_same_op_limits() {
        run_host_cases(&["legitimate-stop", "same-operation", "unfinalized-bind"]);
    }
    #[test]
    #[ignore = "requires matched release extension"]
    fn fix1_original_cold_and_session_hard_cutoffs_never_renew_wait() {
        run_host_cases(&["cold-expire", "session-expire"]);
    }
    #[test]
    fn fix1_original_leader_stays_kernel_backed_when_adopted_child_receives_term() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("4");
        let mut host = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let actual = host.retire(cleanup_clock()).unwrap();
        assert_eq!(actual.wait.exit_code, Some(7));
        assert_eq!(
            actual.descendants[0].wait.exit_code,
            Some(19),
            "child observed original PID/PGID lifetime at actual group TERM"
        );
    }
    #[test]
    fn fix1_explicit_inherited_sigchld_ignore_is_normalized_before_spawn() {
        let _serial = PROCESS_TEST.lock().unwrap();
        let binary = c_fixture();
        enable_subreaper().unwrap();
        unsafe {
            libc::signal(libc::SIGCHLD, libc::SIG_IGN);
        }
        let mut command = std::process::Command::new(binary);
        command.arg("5");
        let result = OwnedHost::spawn(&mut command, cleanup_clock().finish)
            .and_then(|mut host| host.retire(cleanup_clock()));
        unsafe {
            libc::signal(libc::SIGCHLD, libc::SIG_DFL);
        }
        let actual = result
            .expect("supervisor must preserve actual wait under inherited explicit SIGCHLD ignore");
        assert_eq!(actual.wait.exit_code, Some(7));
    }
    #[test]
    fn fix1_repeated_wnowait_keeps_pid_pgid_until_actual_original_consume() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("4");
        let mut host = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let finish = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !host.observe_exit().unwrap() {
            assert!(std::time::Instant::now() < finish);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        for _ in 0..8 {
            assert!(host.observe_exit().unwrap());
            assert_eq!(
                unsafe { libc::getpgid(host.pid() as i32) },
                host.identity.pgid
            );
        }
        let actual = host.retire(cleanup_clock()).unwrap();
        assert_eq!(actual.wait.exit_code, Some(7));
        assert_eq!(actual.descendants[0].wait.exit_code, Some(19));
        assert!(host.observe_exit().is_err());
        assert!(host.retire(cleanup_clock()).is_err());
        assert_eq!(unsafe { libc::getpgid(host.pid() as i32) }, -1);
    }
    #[test]
    fn fix1_escaped_child_cannot_unback_group_signals_while_leader_pinned() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("6");
        let mut host = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let original_group = host.identity.pgid;
        let actual = host.retire(cleanup_clock()).unwrap();
        assert!(!actual.descendants_reaped);
        assert_eq!(actual.descendants[0].wait.exit_code, Some(19));
        assert_ne!(actual.descendants[0].pgid, original_group);
        assert_eq!(
            actual
                .signals
                .iter()
                .map(|s| (s.pgid, s.signal))
                .collect::<Vec<_>>(),
            vec![
                (original_group, libc::SIGTERM),
                (original_group, libc::SIGKILL)
            ]
        );
    }
    #[test]
    fn fix1_lost_original_wait_owner_disables_all_later_group_authority() {
        let _serial = PROCESS_TEST.lock().unwrap();
        enable_subreaper().unwrap();
        let mut command = std::process::Command::new(c_fixture());
        command.arg("5");
        let mut host = OwnedHost::spawn(&mut command, cleanup_clock().finish).unwrap();
        let mut status = 0;
        assert_eq!(
            unsafe { libc::waitpid(host.pid() as i32, &mut status, 0) },
            host.pid() as i32
        );
        let error = host.observe_exit().unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
        assert!(host.retire(cleanup_clock()).is_err());
    }
}
