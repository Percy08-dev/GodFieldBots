//! Fixed application bootstrap, original lock custody and sequential cold reentry.
use crate::{
    Result, build_info, invalid, monotonic,
    namespace::Namespace,
    packet::{self, Channel, Packet},
    process::*,
    records::*,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::{
            fs::{FileExt, MetadataExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

pub(crate) struct LaunchOwner {
    pub file: File,
    pub value: Value,
    device: u64,
    inode: u64,
    sha: String,
}
impl LaunchOwner {
    pub fn take(fd: RawFd) -> Result<Self> {
        let file = unsafe { File::from_raw_fd(fd) };
        let info = file.metadata()?;
        if !info.is_file()
            || info.uid() != unsafe { libc::getuid() }
            || info.len() > packet::LIMIT as u64
        {
            return Err(invalid("owned compact regular launch descriptor required"));
        }
        let access = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
        let required =
            libc::F_SEAL_SEAL | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_WRITE;
        if access < 0
            || access & libc::O_ACCMODE != libc::O_RDONLY
                && (seals < 0 || seals & required != required)
        {
            return Err(invalid("launch must be readonly or sealed"));
        }
        let mut bytes = vec![0; info.len() as usize];
        file.read_exact_at(&mut bytes, 0)?;
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|_| invalid("invalid fixed launch envelope"))?;
        let owner = Self {
            file,
            value,
            device: info.dev(),
            inode: info.ino(),
            sha: bytes_sha(&bytes),
        };
        owner.recheck()?;
        Ok(owner)
    }
    fn recheck(&self) -> Result<()> {
        let info = self.file.metadata()?;
        if (info.dev(), info.ino()) != (self.device, self.inode)
            || info.len() > packet::LIMIT as u64
        {
            return Err(invalid("original launch descriptor changed"));
        }
        let mut bytes = vec![0; info.len() as usize];
        self.file.read_exact_at(&mut bytes, 0)?;
        if bytes_sha(&bytes) != self.sha {
            return Err(invalid("original launch bytes changed"));
        }
        Ok(())
    }
}
#[derive(Clone)]
struct Bound {
    value: Value,
    path: PathBuf,
    sha: String,
    times: [f64; 5],
    clock: CleanupClock,
}
#[derive(Clone)]
struct Prepared {
    value: Value,
    path: PathBuf,
    sha: String,
}
#[derive(Clone)]
struct Cold {
    prepared: Prepared,
    exit: Value,
    exit_path: PathBuf,
    exit_sha: String,
    admission: Value,
    physical: Option<(Value, PathBuf, String)>,
}
#[derive(Clone)]
struct FailedReturn {
    failure: Prepared,
    exit: Value,
    exit_path: PathBuf,
    exit_sha: String,
    admission: Value,
    physical: Value,
    physical_path: PathBuf,
    physical_sha: String,
}
#[derive(Clone, Copy, PartialEq)]
enum ControlPhase {
    Training,
    Cold,
    FailedReturn,
    Session,
}
impl ControlPhase {
    fn clock(self, bound: &Bound, session_cutoff: Instant) -> CleanupClock {
        match self {
            Self::Training => bound.clock,
            Self::Cold | Self::FailedReturn => CleanupClock::hard_cutoff(bound.clock.finish),
            Self::Session => CleanupClock::hard_cutoff(session_cutoff),
        }
    }
}
pub(crate) struct Supervisor {
    launch: LaunchOwner,
    root: PathBuf,
    mode: String,
    session_finish: f64,
    performance: Option<crate::PerformanceDiagnostics>,
    session_cutoff: Instant,
    locks: Vec<HeldLock>,
    bound: Bound,
    campaign: Option<CampaignBinding>,
    seen: BTreeMap<String, String>,
    #[cfg(test)]
    pub test_command: Option<Vec<String>>,
    #[cfg(test)]
    pub test_namespace: Option<PathBuf>,
    #[cfg(test)]
    pub test_terminal_barrier: bool,
}
impl Supervisor {
    pub fn new(launch_owner: LaunchOwner) -> Result<Self> {
        let performance = crate::PerformanceDiagnostics::from_environment()?;
        let finish = launch(&launch_owner.value)?;
        let session_cutoff = crate::deadline_from_bits(text(
            &launch_owner.value["session_deadlines"]["finish_bits"],
        )?)?
        .1;
        let root = PathBuf::from(text(&launch_owner.value["record_directory"])?);
        ensure_private(&root)?;
        let mode = text(&launch_owner.value["mode"])?.to_owned();
        let bound = Self::load_bound(
            &root,
            &mode,
            finish,
            Path::new(text(&launch_owner.value["operation_path"])?),
            text(&launch_owner.value["operation_sha256"])?,
        )?;
        let mut locks = Vec::new();
        let mut fd_seen = std::collections::BTreeSet::from([launch_owner.file.as_raw_fd()]);
        for item in launch_owner.value["locks"].as_array().unwrap() {
            let fd: i32 = positive(&item["fd"])?
                .try_into()
                .map_err(|_| invalid("invalid inherited custody FD"))?;
            if !fd_seen.insert(fd) {
                return Err(invalid("duplicate inherited descriptor"));
            }
            let descriptor = unsafe { OwnedFd::from_raw_fd(fd) };
            locks.push(HeldLock::adopt(
                descriptor,
                text(&item["lock_kind"])?,
                uint(&item["device"])?,
                positive(&item["inode"])?,
            )?);
        }
        let seen = BTreeMap::from([(
            text(&bound.value["operation_id"])?.to_owned(),
            bound.sha.clone(),
        )]);
        Ok(Self {
            launch: launch_owner,
            root,
            mode,
            session_finish: finish,
            performance,
            session_cutoff,
            locks,
            campaign: None,
            bound,
            seen,
            #[cfg(test)]
            test_command: None,
            #[cfg(test)]
            test_namespace: None,
            #[cfg(test)]
            test_terminal_barrier: false,
        })
    }
    fn load_bound(root: &Path, mode: &str, finish: f64, path: &Path, sha: &str) -> Result<Bound> {
        let (value, actual) = read_owned(path, Some(root))?;
        if actual != sha {
            return Err(invalid("operation original hash mismatch"));
        }
        let times = operation(&value, mode, finish)?;
        let clock = CleanupClock::new(times[2], times[3], times[4])?;
        Ok(Bound {
            value,
            path: path.to_owned(),
            sha: sha.to_owned(),
            times,
            clock,
        })
    }
    fn check_locks(&self, parent: bool) -> Result<()> {
        self.launch.recheck()?;
        if let Some(campaign) = &self.campaign {
            campaign.recheck(&self.launch.value, &self.locks)?;
        }
        for lock in &self.locks {
            lock.recheck()?;
        }
        if parent {
            let lock = self
                .locks
                .iter()
                .find(|l| l.kind == "parent")
                .ok_or_else(|| invalid("actual parent lock required before admission"))?;
            let path = Path::new(text(&self.bound.value["parent_directory"])?).join(".lock");
            let info = std::fs::symlink_metadata(path)?;
            if !info.is_file() || (info.dev(), info.ino()) != (lock.device, lock.inode) {
                return Err(invalid("parent FD differs from actual bound run lock"));
            }
            if self.mode == "hpo" && !self.locks.iter().any(|l| l.kind == "campaign") {
                return Err(invalid("actual campaign lock required"));
            }
        }
        Ok(())
    }
    fn command(&self, control: RawFd, phase: &str) -> Command {
        #[cfg(test)]
        if let Some(command) = &self.test_command {
            let mut cmd = Command::new(&command[0]);
            cmd.args(&command[1..]);
            cmd.args([
                control.to_string(),
                self.launch.file.as_raw_fd().to_string(),
                phase.to_owned(),
            ]);
            return cmd;
        }
        let mut command = Command::new(self.launch.value["python_executable"].as_str().unwrap());
        command.args([
            "-I",
            "-m",
            "gf_cli.rust_player_host",
            "--control-fd",
            &control.to_string(),
            "--launch-fd",
            &self.launch.file.as_raw_fd().to_string(),
            "--phase",
            phase,
        ]);
        command.stdin(Stdio::null());
        command
    }
    fn spawn(&self, phase: &str) -> Result<(OwnedHost, Channel)> {
        self.check_locks(false)?;
        let invocation = random_id()?;
        let nonce = random_id()?;
        let (mut channel, peer) = Channel::pair(
            0,
            text(&self.launch.value["supervision_id"])?,
            &invocation,
            &nonce,
        )?;
        let mut command = self.command(peer.as_raw_fd(), phase);
        let mut fds = vec![peer.as_raw_fd(), self.launch.file.as_raw_fd()];
        fds.extend(self.locks.iter().map(|l| l.fd.as_raw_fd()));
        unsafe {
            command.pre_exec(move || {
                for fd in &fds {
                    let flags = libc::fcntl(*fd, libc::F_GETFD);
                    if flags < 0 || libc::fcntl(*fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        if let Some(d) = &self.performance {
            d.phase(json!({"run_id":self.bound.value["run_id"],"operation_id":self.bound.value["operation_id"],"invocation_id":invocation,"batch_id":null,"phase":if phase=="cold"{"cold"}else{"startup"}}),
                "phase_start",crate::diagnostic_monotonic_ns(),json!({"operation_sha256":self.bound.sha,"prepared_sha256":null,"exit_sha256":null,"completion_sha256":null}));
        }
        let host = OwnedHost::spawn(&mut command, self.bound.clock.finish)?;
        channel.peer_pid = host.pid();
        drop(peer);
        Ok((host, channel))
    }
    fn await_packet(
        channel: &mut Channel,
        host: &mut OwnedHost,
        deadline: Instant,
    ) -> Result<Packet> {
        loop {
            if let Some(packet) = channel.receive()? {
                return Ok(packet);
            }
            if host.observe_exit()? {
                return Err(invalid("original host exited during control handshake"));
            }
            if Instant::now() >= deadline {
                return Err(invalid("original control deadline expired"));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn lock_mapping(&self) -> Value {
        json!(self.locks.iter().map(|l|json!({"lock_kind":l.kind,"fd":l.fd.as_raw_fd(),"device":l.device,"inode":l.inode})).collect::<Vec<_>>())
    }
    fn handshake(
        &self,
        host: &mut OwnedHost,
        channel: &mut Channel,
        cold: Option<&Cold>,
        failed: Option<&FailedReturn>,
    ) -> Result<u64> {
        channel.send("hello",0,json!({"supervisor_pid":std::process::id(),"host_pid":host.identity.pid,"host_start_ticks":host.identity.start_ticks,"build":build_info()}))?;
        let packet = Self::await_packet(channel, host, self.bound.clock.finish)?;
        let frame = &packet.value;
        packet::fields(
            frame,
            &[
                "pid",
                "start_ticks",
                "python_abi",
                "extension_sha256",
                "build",
            ],
        )?;
        if !packet.fds.is_empty()
            || frame["kind"] != "ready"
            || frame["seq"] != 0
            || frame["pid"] != host.identity.pid
            || frame["start_ticks"] != host.identity.start_ticks
            || frame["python_abi"] != self.launch.value["binding"]["python_abi"]
            || frame["extension_sha256"] != self.launch.value["binding"]["extension_sha256"]
            || frame["build"] != build_info()
        {
            return Err(invalid("original host Python/native handshake mismatch"));
        }
        self.check_locks(false)?;
        if let Some(failed) = failed {
            self.check_locks(true)?;
            for (path, value, sha) in [
                (
                    &failed.failure.path,
                    &failed.failure.value,
                    &failed.failure.sha,
                ),
                (&failed.exit_path, &failed.exit, &failed.exit_sha),
                (
                    &failed.physical_path,
                    &failed.physical,
                    &failed.physical_sha,
                ),
            ] {
                let (actual, checksum) = read_campaign_control(path, None)?;
                if actual != *value || checksum != *sha {
                    return Err(invalid("failed return immutable evidence changed"));
                }
            }
            channel.send("campaign_return", 1, json!({"operation_path":self.bound.path,
                "operation_sha256":self.bound.sha,"failure_path":failed.failure.path,
                "failure_sha256":failed.failure.sha,"exit_path":failed.exit_path,"exit_sha256":failed.exit_sha,
                "admission":failed.admission,"physical_path":failed.physical_path,"physical_sha256":failed.physical_sha,
                "locks":self.lock_mapping(),"deadlines":self.bound.value["deadlines"]}))?;
            let reply = Self::await_packet(channel, host, self.bound.clock.finish)?;
            packet::fields(
                &reply.value,
                &["failure_sha256", "exit_sha256", "physical_sha256"],
            )?;
            if !reply.fds.is_empty()
                || reply.value["kind"] != "campaign_return_ready"
                || reply.value["seq"] != 1
                || reply.value["failure_sha256"] != failed.failure.sha
                || reply.value["exit_sha256"] != failed.exit_sha
                || reply.value["physical_sha256"] != failed.physical_sha
            {
                return Err(invalid("failed return exact evidence ACK differs"));
            }
            Ok(2)
        } else if let Some(cold) = cold {
            self.check_locks(true)?;
            let (previous, sha) = read_owned(&cold.prepared.path, Some(&self.root))?;
            let (previous_exit, exit_sha) = read_owned(&cold.exit_path, Some(&self.root))?;
            if previous != cold.prepared.value
                || sha != cold.prepared.sha
                || previous_exit != cold.exit
                || exit_sha != cold.exit_sha
            {
                return Err(invalid("cold immutable pair changed after original reap"));
            }
            channel.send("cold",1,json!({"operation_path":self.bound.path,"operation_sha256":self.bound.sha,"prepared_path":cold.prepared.path,"prepared_sha256":cold.prepared.sha,"exit_path":cold.exit_path,"exit_sha256":cold.exit_sha,"admission":cold.admission,"locks":self.lock_mapping(),"deadlines":self.bound.value["deadlines"]}))?;
            let response = Self::await_packet(channel, host, self.bound.clock.finish)?;
            packet::fields(&response.value, &["prepared_sha256", "exit_sha256"])?;
            if !response.fds.is_empty()
                || response.value["kind"] != "cold_ready"
                || response.value["seq"] != 1
                || response.value["prepared_sha256"] != cold.prepared.sha
                || response.value["exit_sha256"] != cold.exit_sha
            {
                return Err(invalid("cold exact pair acknowledgement mismatch"));
            }
            Ok(2)
        } else {
            Ok(1)
        }
    }
    fn prepare(&self, frame: &Value, admission: &Value) -> Result<Prepared> {
        self.check_locks(true)?;
        let path = PathBuf::from(text(&frame["prepared_path"])?);
        let (value, sha) = read_owned(&path, Some(&self.root))?;
        if sha != text(&frame["prepared_sha256"])? || sha != digest(&value) {
            return Err(invalid("prepared original/canonical hash mismatch"));
        }
        prepared(&value, admission, &self.bound.value)?;
        let close_path = Path::new(text(&value["close_ack_path"])?);
        let (close, close_sha) = read_owned(close_path, Some(&self.root))?;
        if close_sha != text(&value["close_ack_sha256"])? {
            return Err(invalid("close acknowledgement original hash mismatch"));
        }
        close_ack(&close, &value)?;
        let servers_path = Path::new(text(&close["servers_path"])?);
        let (servers, servers_sha) = read_owned(servers_path, Some(&self.root))?;
        if servers_sha != text(&close["servers_sha256"])? {
            return Err(invalid("server retirement original hash mismatch"));
        }
        record(
            &servers,
            "rust-player-server-retirement-v1",
            1,
            &["supervision_id", "invocation_id", "operation_id", "servers"],
        )?;
        for key in ["supervision_id", "invocation_id", "operation_id"] {
            if servers[key] != value[key] {
                return Err(invalid("server retirement identity mismatch"));
            }
        }
        let entries = servers["servers"]
            .as_array()
            .ok_or_else(|| invalid("actual server retirement entries required"))?;
        if entries.len() as u64 != uint(&close["server_count"])? {
            return Err(invalid("actual server retirement count mismatch"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for server in entries {
            object(
                server,
                &["pid", "start_ticks", "exit_code", "signal", "logs_closed"],
            )?;
            if !seen.insert(positive(&server["pid"])?) || server["logs_closed"] != true {
                return Err(invalid("server log/PID retirement unconfirmed"));
            }
            positive(&server["start_ticks"])?;
            wait(
                &json!({"exit_code":server["exit_code"],"signal":server["signal"],"core_dumped":false}),
            )?;
        }
        resync(servers_path)?;
        resync(close_path)?;
        resync(&path)?;
        Ok(Prepared { value, path, sha })
    }
    pub fn run(mut self) -> Result<()> {
        enable_subreaper()?;
        self.campaign =
            CampaignBinding::from_initial(&self.launch.value, &self.bound.value, &self.locks)?;
        let mut cold: Option<Cold> = None;
        let mut failed: Option<FailedReturn> = None;
        loop {
            if monotonic()? >= self.session_finish {
                return Err(invalid("original session deadline expired"));
            }
            if let Some(previous) = &cold {
                let (prepared, prepared_sha) =
                    read_owned(&previous.prepared.path, Some(&self.root))?;
                let (exit, exit_sha) = read_owned(&previous.exit_path, Some(&self.root))?;
                if prepared != previous.prepared.value
                    || prepared_sha != previous.prepared.sha
                    || exit != previous.exit
                    || exit_sha != previous.exit_sha
                {
                    return Err(invalid("immutable pair changed before continuation"));
                }
            }
            let (mut host, mut channel) = self.spawn(if failed.is_some() {
                "campaign-return"
            } else if cold.is_some() {
                "cold"
            } else {
                "host"
            })?;
            let mut phase = if failed.is_some() {
                ControlPhase::FailedReturn
            } else if cold.is_some() {
                ControlPhase::Cold
            } else {
                if self.bound.value["operation_type"] == "campaign-continue"
                    && self.campaign.is_some()
                {
                    ControlPhase::Session
                } else {
                    ControlPhase::Training
                }
            };
            let mut protocol_failed = false;
            let mut admission: Option<Value> = None;
            let mut namespace: Option<Namespace> = None;
            let mut durable = true;
            let mut prepared_result: Option<Prepared> = None;
            let mut failure_result: Option<Prepared> = None;
            let mut control_completed: Option<String> = None;
            let mut disposition_ack: Option<Value> = None;
            let mut finalized_path: Option<PathBuf> = None;
            let mut finalized = false;
            let mut finalized_sha: Option<String> = None;
            let handshake = self.handshake(&mut host, &mut channel, cold.as_ref(), failed.as_ref());
            if let Some(d) = &self.performance {
                d.phase(json!({"run_id":self.bound.value["run_id"],"operation_id":self.bound.value["operation_id"],"invocation_id":channel.invocation,"batch_id":null,"phase":if cold.is_some(){"cold"}else{"startup"}}),
                    "phase_end",crate::diagnostic_monotonic_ns(),json!({"operation_sha256":self.bound.sha,"prepared_sha256":null,"exit_sha256":null,"completion_sha256":null}));
            }
            let mut server_seq = match handshake {
                Ok(seq) => seq,
                Err(_) => {
                    protocol_failed = true;
                    0
                }
            };
            let mut host_seq = if failed.is_some() { 2 } else { 1 };
            #[cfg(test)]
            let mut terminal_barrier_used = false;
            let mut stopped = false;
            let mut pending_stop: Option<u64> = None;
            let control_result = (|| -> Result<()> {
                while !protocol_failed && prepared_result.is_none() {
                    let authority = phase.clock(&self.bound, self.session_cutoff);
                    if Instant::now() >= authority.finish {
                        protocol_failed = true;
                        break;
                    }
                    let now = monotonic()?;
                    if phase == ControlPhase::Training && now >= self.bound.times[1] && !stopped {
                        if channel
                            .send("stop", server_seq, json!({"reason":"deadline"}))
                            .is_err()
                        {
                            protocol_failed = true;
                            break;
                        }
                        pending_stop = Some(server_seq);
                        server_seq += 1;
                        stopped = true;
                    }
                    if phase == ControlPhase::Training && Instant::now() >= self.bound.clock.term {
                        break;
                    }
                    let packet = match channel.receive() {
                        Ok(Some(p)) => p,
                        Ok(None) => {
                            #[cfg(test)]
                            if self.test_terminal_barrier
                                && !terminal_barrier_used
                                && (failure_result.is_some() || control_completed.is_some())
                            {
                                terminal_barrier_used = true;
                                write(
                                    &self.root.join("terminal-eagain.json"),
                                    &json!({"invocation_id":channel.invocation,"host_pid":host.pid()}),
                                    true,
                                )?;
                                // Private deterministic EAGAIN→send/exit→WNOWAIT
                                // barrier, bounded by this SAME original authority.
                                let cutoff = if phase == ControlPhase::Training {
                                    authority.finish.min(self.bound.clock.term)
                                } else {
                                    authority.finish
                                };
                                while !host.observe_exit()? {
                                    if Instant::now() >= cutoff {
                                        return Err(invalid(
                                            "original private barrier deadline expired",
                                        ));
                                    }
                                    std::thread::sleep(Duration::from_millis(2));
                                }
                                write(
                                    &self.root.join("terminal-observed.json"),
                                    &json!({"invocation_id":channel.invocation,"host_pid":host.pid(),"observed_wnowait":true}),
                                    true,
                                )?;
                            }
                            if host.observe_exit()? {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(2));
                            continue;
                        }
                        Err(e) => {
                            if !((finalized
                                || failure_result.is_some()
                                || control_completed.is_some())
                                && e.kind() == io::ErrorKind::UnexpectedEof)
                            {
                                protocol_failed = true;
                            }
                            break;
                        }
                    };
                    let frame = &packet.value;
                    let kind = text(&frame["kind"])?;
                    if kind == "stop_ack" {
                        if !settle_pending_stop_ack(&packet, &mut pending_stop) {
                            protocol_failed = true;
                            break;
                        }
                        continue;
                    }
                    if failure_result.is_some() || control_completed.is_some() {
                        return Err(invalid("packet after accepted terminal arm"));
                    }
                    if frame["seq"] != host_seq {
                        protocol_failed = true;
                        break;
                    }
                    let seq = host_seq;
                    host_seq += 1;
                    let result = (|| -> Result<()> {
                        match kind {
                            "bind_operation" => {
                                packet::fields(
                                    frame,
                                    &[
                                        "operation_path",
                                        "operation_sha256",
                                        "operation_id",
                                        "run_id",
                                        "config_sha256",
                                        "runtime_binding_sha256",
                                        "deadlines",
                                    ],
                                )?;
                                if admission.is_some()
                                    || cold.is_some() && !finalized
                                    || failed.is_some() && phase != ControlPhase::Session
                                    || self.locks.iter().any(|l| l.kind == "parent")
                                    || !packet.fds.is_empty()
                                {
                                    return Err(invalid(
                                        "operation overlaps active/unfinalized original",
                                    ));
                                }
                                if let Some(campaign) = &self.campaign {
                                    campaign.recheck(&self.launch.value, &self.locks)?;
                                    if now >= bits(&campaign.clock["soft_dispatch_finish_bits"])? {
                                        return Err(invalid(
                                            "original campaign soft dispatch boundary reached",
                                        ));
                                    }
                                }
                                let next = Self::load_bound(
                                    &self.root,
                                    &self.mode,
                                    self.session_finish,
                                    Path::new(text(&frame["operation_path"])?),
                                    text(&frame["operation_sha256"])?,
                                )?;
                                for key in [
                                    "operation_id",
                                    "run_id",
                                    "config_sha256",
                                    "runtime_binding_sha256",
                                    "deadlines",
                                ] {
                                    if frame[key] != next.value[key] {
                                        return Err(invalid(
                                            "operation bind differs from immutable original",
                                        ));
                                    }
                                }
                                let id = text(&next.value["operation_id"])?;
                                if self
                                    .seen
                                    .get(id)
                                    .is_some_and(|sha| self.campaign.is_some() || sha != &next.sha)
                                {
                                    return Err(invalid(
                                        "same operation cannot reset original deadlines",
                                    ));
                                }
                                channel.send("operation_bound",seq,json!({"operation_id":next.value["operation_id"],"operation_sha256":next.sha}))?;
                                self.seen.insert(id.to_owned(), next.sha.clone());
                                self.bound = next;
                                cold = None;
                                failed = None;
                                finalized = false;
                                finalized_sha = None;
                                finalized_path = None;
                                disposition_ack = None;
                                phase = ControlPhase::Training;
                                stopped = false;
                                pending_stop = None;
                            }
                            "lock" => {
                                packet::fields(frame, &["lock_kind", "device", "inode"])?;
                                if packet.fds.len() != 1
                                    || admission.is_some()
                                    || cold.is_some() && !finalized
                                    || failed.is_some()
                                    || self.bound.value["run_id"].is_null()
                                {
                                    return Err(invalid("unexpected custody transfer"));
                                }
                                let kind = text(&frame["lock_kind"])?;
                                if self.locks.iter().any(|l| l.kind == kind) {
                                    return Err(invalid("duplicate live custody"));
                                }
                                let fd = packet.fds.into_iter().next().unwrap();
                                self.locks.push(HeldLock::adopt(
                                    fd,
                                    kind,
                                    uint(&frame["device"])?,
                                    positive(&frame["inode"])?,
                                )?);
                                channel.send("lock_ack",seq,json!({"lock_kind":frame["lock_kind"],"device":frame["device"],"inode":frame["inode"]}))?;
                            }
                            "admit" => {
                                packet::fields(
                                    frame,
                                    &[
                                        "context",
                                        "budget",
                                        "producer",
                                        "operation_id",
                                        "run_id",
                                        "config_sha256",
                                        "runtime_binding_sha256",
                                    ],
                                )?;
                                if !packet.fds.is_empty()
                                    || admission.is_some()
                                    || cold.is_some() && !finalized
                                    || phase != ControlPhase::Training
                                    || failed.is_some()
                                    || now >= self.bound.times[0]
                                    || self.bound.value["run_id"].is_null()
                                {
                                    return Err(invalid(
                                        "admission outside original operation phase/deadline",
                                    ));
                                }
                                self.check_locks(true)?;
                                for key in [
                                    "operation_id",
                                    "run_id",
                                    "config_sha256",
                                    "runtime_binding_sha256",
                                ] {
                                    if frame[key] != self.bound.value[key] {
                                        return Err(invalid(
                                            "admission immutable operation mismatch",
                                        ));
                                    }
                                }
                                if frame["context"] != self.bound.value["expected_context"] {
                                    return Err(invalid("admission exact context mismatch"));
                                }
                                producer(&frame["producer"])?;
                                if frame["producer"]["build"] != build_info()
                                    || frame["producer"]["extension_sha256"]
                                        != self.launch.value["binding"]["extension_sha256"]
                                    || frame["producer"]["supervisor_sha256"]
                                        != bytes_sha(&std::fs::read("/proc/self/exe")?)
                                {
                                    return Err(invalid(
                                        "admission producer executable/extension mismatch",
                                    ));
                                }
                                let ticket = if self.mode == "hpo" {
                                    Some(random_id()?)
                                } else {
                                    None
                                };
                                let local = json!({"schema_version":1,"kind":"rust-player-admission-v1","supervision_id":channel.supervision,"invocation_id":channel.invocation,"nonce":channel.nonce,"supervisor":{"pid":std::process::id(),"start_ticks":start_ticks(std::process::id())?},"host":host.identity,"context":frame["context"],"budget":frame["budget"],"operation_id":frame["operation_id"],"run_id":frame["run_id"],"config_sha256":frame["config_sha256"],"runtime_binding_sha256":frame["runtime_binding_sha256"],"deadlines":self.bound.value["deadlines"],"producer":frame["producer"],"global_ticket":ticket});
                                crate::records::local(&local)?;
                                admission = Some(local.clone());
                                durable = false;
                                if self.mode == "hpo" {
                                    #[cfg(test)]
                                    let owner = if let Some(root) = &self.test_namespace {
                                        Namespace::private_test(root)?
                                    } else {
                                        Namespace::acquire()?
                                    };
                                    #[cfg(not(test))]
                                    let owner = Namespace::acquire()?;
                                    namespace = Some(owner);
                                    namespace.as_mut().unwrap().admit(&local)?;
                                }
                                let path = self
                                    .root
                                    .join(format!("admission-{}.json", channel.invocation));
                                let sha = write(&path, &local, true)?;
                                durable = true;
                                channel.send("admitted",seq,json!({"admission_path":path,"admission_sha256":sha,"global_ticket":ticket}))?;
                            }
                            "training_failed" => {
                                packet::fields(frame, &["failure_path", "failure_sha256"])?;
                                if !packet.fds.is_empty()
                                    || phase != ControlPhase::Training
                                    || prepared_result.is_some()
                                    || failure_result.is_some()
                                {
                                    return Err(invalid(
                                        "duplicate/unexpected training terminal arm",
                                    ));
                                }
                                let local = admission
                                    .as_ref()
                                    .ok_or_else(|| invalid("failure before actual admission"))?;
                                let path = self
                                    .root
                                    .join(format!("failure-{}.json", channel.invocation));
                                if frame["failure_path"] != path.to_string_lossy().as_ref() {
                                    return Err(invalid("noncanonical training failure path"));
                                }
                                let (value, sha) = read_campaign_control(&path, Some(&self.root))?;
                                training_failure(&value, local, &self.bound.sha)?;
                                if frame["failure_sha256"] != sha {
                                    return Err(invalid("failure original hash differs"));
                                }
                                resync(&path)?;
                                channel.send(
                                    "training_failed_ack",
                                    seq,
                                    json!({"failure_sha256":sha}),
                                )?;
                                failure_result = Some(Prepared { value, path, sha });
                            }
                            "control_complete" => {
                                packet::fields(
                                    frame,
                                    &["operation_sha256", "result_path", "result_sha256"],
                                )?;
                                let campaign = self.campaign.as_ref().ok_or_else(|| {
                                    invalid("control completion needs initial campaign purpose")
                                })?;
                                if !packet.fds.is_empty()
                                    || phase != ControlPhase::Session
                                    || admission.is_some()
                                    || self.locks.iter().any(|l| l.kind == "parent")
                                    || (self.bound.value["operation_type"] != "campaign-continue"
                                        && (failed.is_none() || disposition_ack.is_none()))
                                {
                                    return Err(invalid(
                                        "control completion outside actual parent-free Session",
                                    ));
                                }
                                campaign.recheck(&self.launch.value, &self.locks)?;
                                let (value, sha) = read_campaign_control(
                                    Path::new(text(&frame["result_path"])?),
                                    Some(&self.root),
                                )?;
                                record(
                                    &value,
                                    "rust-player-control-completion-v1",
                                    1,
                                    &[
                                        "supervision_id",
                                        "operation_sha256",
                                        "session_clock_sha256",
                                        "status",
                                    ],
                                )?;
                                if frame["operation_sha256"] != self.bound.sha
                                    || frame["result_sha256"] != sha
                                    || value["operation_sha256"] != self.bound.sha
                                    || value["supervision_id"] != channel.supervision
                                    || value["session_clock_sha256"] != campaign.clock_sha
                                    || !["complete", "session_boundary", "no_work"]
                                        .contains(&text(&value["status"])?)
                                {
                                    return Err(invalid(
                                        "control completion original binding differs",
                                    ));
                                }
                                resync(Path::new(text(&frame["result_path"])?))?;
                                channel.send("control_completed",seq,json!({"operation_sha256":self.bound.sha,"result_path":frame["result_path"],"result_sha256":sha}))?;
                                control_completed = Some(sha);
                            }
                            "prepared" => {
                                packet::fields(frame, &["prepared_path", "prepared_sha256"])?;
                                if !packet.fds.is_empty() || failure_result.is_some() {
                                    return Err(invalid("unexpected prepared FD/terminal arm"));
                                }
                                let local = admission
                                    .as_ref()
                                    .ok_or_else(|| invalid("prepared before actual admission"))?;
                                let original = self.prepare(frame, local)?;
                                channel.send(
                                    "prepared_ack",
                                    seq,
                                    json!({"prepared_sha256":original.sha}),
                                )?;
                                prepared_result = Some(original);
                            }
                            "finalized" => {
                                packet::fields(
                                    frame,
                                    &[
                                        "prepared_sha256",
                                        "exit_sha256",
                                        "completion_path",
                                        "completion_sha256",
                                    ],
                                )?;
                                if !packet.fds.is_empty()
                                    || admission.is_some()
                                    || phase == ControlPhase::Training
                                {
                                    return Err(invalid("finalization overlaps training"));
                                }
                                let previous =
                                    cold.as_ref().ok_or_else(|| invalid("no exact cold pair"))?;
                                if frame["prepared_sha256"] != previous.prepared.sha
                                    || frame["exit_sha256"] != previous.exit_sha
                                {
                                    return Err(invalid("cold pair differs"));
                                }
                                self.check_locks(true)?;
                                let path = Path::new(text(&frame["completion_path"])?);
                                let (completion, sha) = read_owned(path, Some(&self.root))?;
                                let initial = initial_hpo_pause(&self.bound.value);
                                let (version, kind) = if initial {
                                    (2, "rust-player-initial-pause-completion-v2")
                                } else {
                                    (1, "rust-player-training-completion-v1")
                                };
                                if sha != text(&frame["completion_sha256"])?
                                    || completion["schema_version"] != version
                                    || completion["kind"] != kind
                                    || completion["prepared_sha256"] != previous.prepared.sha
                                    || completion["exit_sha256"] != previous.exit_sha
                                    || initial
                                        && (completion["state"] != "paused"
                                            || completion["committed_update"] != 0
                                            || completion["eligible"] != false
                                            || !completion["source_manifest_sha256"].is_null())
                                {
                                    return Err(invalid("cold finalizer output/pair mismatch"));
                                }
                                if finalized_sha.as_ref().is_some_and(|old| old != &sha) {
                                    return Err(invalid("competing finalizer output differs"));
                                }
                                resync(path)?;
                                finalized = true;
                                finalized_sha = Some(sha.clone());
                                finalized_path = Some(path.to_path_buf());
                                channel.send(
                                    "finalized_ack",
                                    seq,
                                    json!({"completion_sha256":sha}),
                                )?;
                            }
                            "campaign_disposition" => {
                                packet::fields(frame, &["disposition_path", "disposition_sha256"])?;
                                if !packet.fds.is_empty()
                                    || disposition_ack.is_some()
                                    || admission.is_some()
                                {
                                    return Err(invalid("unexpected campaign disposition"));
                                }
                                self.check_locks(true)?;
                                let campaign = self
                                    .campaign
                                    .as_ref()
                                    .ok_or_else(|| invalid("initial campaign purpose required"))?;
                                campaign.recheck(&self.launch.value, &self.locks)?;
                                let (value, sha) = campaign.disposition(
                                    frame,
                                    &self.bound,
                                    failed.as_ref(),
                                    cold.as_ref(),
                                    finalized,
                                    finalized_path.as_deref(),
                                    finalized_sha.as_deref(),
                                )?;
                                let ack = json!({"schema_version":1,"kind":"rust-player-campaign-disposition-ack-v1",
                                    "supervision_id":channel.supervision,"invocation_id":value["invocation_id"],
                                    "control_invocation_id":channel.invocation,"operation_sha256":self.bound.sha,
                                    "disposition_sha256":sha,"evidence_sha256":digest(&value["evidence"])});
                                let ack_path = self.root.join(format!(
                                    "disposition-ack-{}.json",
                                    text(&value["invocation_id"])?
                                ));
                                let ack_sha = write(&ack_path, &ack, true)?;
                                disposition_ack = Some(ack);
                                channel.send("campaign_disposition_ack",seq,json!({"disposition_sha256":sha,"ack_path":ack_path,"ack_sha256":ack_sha}))?;
                            }
                            "release_parent" => {
                                packet::fields(frame, &["device", "inode"])?;
                                if !packet.fds.is_empty()
                                    || !(finalized && phase == ControlPhase::Cold
                                        || failed.is_some()
                                            && disposition_ack.is_some()
                                            && phase == ControlPhase::FailedReturn)
                                    || admission.is_some()
                                {
                                    return Err(invalid(
                                        "parent release before exact finalization",
                                    ));
                                }
                                let index = self
                                    .locks
                                    .iter()
                                    .position(|l| l.kind == "parent")
                                    .ok_or_else(|| invalid("no original parent custody"))?;
                                let held = &self.locks[index];
                                held.recheck()?;
                                if frame["device"] != held.device || frame["inode"] != held.inode {
                                    return Err(invalid("foreign parent release"));
                                }
                                if let (Some(campaign), Some(cold)) = (&self.campaign, &cold) {
                                    if finalized
                                        && cold.prepared.value["state"] != "complete"
                                        && disposition_ack.is_none()
                                    {
                                        campaign.recheck(&self.launch.value, &self.locks)?;
                                        campaign.checkpoint_pause(
                                            &self.bound,
                                            cold,
                                            finalized_sha.as_deref().ok_or_else(|| {
                                                invalid("original acknowledged completion required")
                                            })?,
                                        )?;
                                    }
                                }
                                self.locks.remove(index);
                                channel.send(
                                    "parent_released",
                                    seq,
                                    json!({"device":frame["device"],"inode":frame["inode"]}),
                                )?;
                                phase = ControlPhase::Session;
                            }
                            "stop" => {
                                packet::fields(frame, &["reason"])?;
                                text(&frame["reason"])?;
                                if !packet.fds.is_empty() {
                                    return Err(invalid("stop with FD"));
                                }
                                stopped = true;
                                channel.send("stop_ack", seq, json!({}))?;
                            }
                            _ => return Err(invalid("unknown/out-of-order host control kind")),
                        }
                        Ok(())
                    })();
                    if result.is_err() {
                        protocol_failed = true;
                        break;
                    }
                }
                Ok(())
            })();
            if control_result.is_err() {
                protocol_failed = true;
            }
            // Never lose the original Child on a protocol/durable failure. One actual retirement owner.
            let actual = host.retire(phase.clock(&self.bound, self.session_cutoff))?;
            // EAGAIN before WNOWAIT observation is not terminal-channel proof:
            // the original may have queued a post-terminal error then exited.
            // After actual original+descendant waits, require a genuine EOF with
            // no queued packet/error. Never dispatch late commands or add a wait.
            if !terminal_channel_quiet(&mut channel, &mut pending_stop) {
                protocol_failed = true;
            }
            if !actual.descendants_reaped || !durable {
                return Err(invalid(
                    "unconfirmed containment/durability retains quarantine",
                ));
            }
            if let Some(local) = admission {
                let cooperative = !protocol_failed
                    && prepared_result.is_some()
                    && actual.wait.exit_code == Some(0)
                    && actual.signals.is_empty();
                let exit_record = json!({"schema_version":1,"kind":"rust-player-exit-v1","supervision_id":channel.supervision,"invocation_id":channel.invocation,"nonce":channel.nonce,"admission_sha256":digest(&local),"prepared_sha256":prepared_result.as_ref().map(|p|p.sha.as_str()),"host_original":actual.origin,"wait":actual.wait,"descendants":actual.descendants,"signals":actual.signals,"descendants_reaped":actual.descendants_reaped,"cooperative":cooperative,"finished_unix_ns":unix_ns()?});
                exit(&exit_record, &local)?;
                let exit_path = self.root.join(format!("exit-{}.json", channel.invocation));
                let exit_sha = write(&exit_path, &exit_record, true)?;
                if let Some(d) = &self.performance {
                    d.phase(json!({"run_id":local["run_id"],"operation_id":local["operation_id"],"invocation_id":channel.invocation,"batch_id":null,"phase":"retirement"}),
                        "host_reaped",actual.host_reaped_monotonic_ns,json!({"operation_sha256":self.bound.sha,"prepared_sha256":prepared_result.as_ref().map(|p|p.sha.as_str()),"exit_sha256":exit_sha,"completion_sha256":null}));
                }
                let mut physical_result = None;
                if let Some(owner) = namespace.take() {
                    let physical = owner.confirm(&exit_record, &actual)?;
                    physical_result = Some((physical, owner.termination_path()?));
                    drop(owner);
                }
                let eligible_failure = !protocol_failed
                    && durable
                    && failure_result.is_some()
                    && self.campaign.is_some()
                    && actual.wait.signal.is_none()
                    && actual.signals.is_empty()
                    && actual.descendants.iter().all(|d| d.wait.signal.is_none())
                    && Instant::now() < self.bound.clock.finish;
                if eligible_failure {
                    self.check_locks(true)?;
                    let (physical, physical_path) = physical_result.ok_or_else(|| {
                        invalid("failure return requires SAME actual namespace retirement")
                    })?;
                    let physical_sha = digest(&physical);
                    failed = Some(FailedReturn {
                        failure: failure_result.unwrap(),
                        exit: exit_record,
                        exit_path,
                        exit_sha,
                        admission: local,
                        physical,
                        physical_path,
                        physical_sha,
                    });
                    cold = None;
                    continue;
                }
                if !cooperative {
                    return Err(invalid(
                        "physical cleanup confirmed; training result remains ineligible",
                    ));
                }
                self.check_locks(true)?;
                cold = Some(Cold {
                    prepared: prepared_result.unwrap(),
                    exit: exit_record,
                    exit_path,
                    exit_sha,
                    admission: local,
                    physical: physical_result.map(|(value, path)| {
                        let sha = digest(&value);
                        (value, path, sha)
                    }),
                });
            } else if let Some(completion_sha) = control_completed {
                if protocol_failed
                    || actual.wait.exit_code != Some(0)
                    || !actual.signals.is_empty()
                    || actual
                        .descendants
                        .iter()
                        .any(|d| d.wait.exit_code != Some(0))
                {
                    return Err(invalid("control completion original wait failed"));
                }
                let value = json!({"schema_version":1,"kind":"rust-player-control-exit-v1",
                    "supervision_id":channel.supervision,"operation_sha256":self.bound.sha,
                    "control_completion_sha256":completion_sha,"host_original":actual.origin,
                    "wait":actual.wait,"descendants":actual.descendants,"descendants_reaped":actual.descendants_reaped,
                    "signals":actual.signals});
                write(
                    &self
                        .root
                        .join(format!("control-exit-{}.json", channel.invocation)),
                    &value,
                    true,
                )?;
                return Ok(());
            } else if finalized
                && phase != ControlPhase::Training
                && !protocol_failed
                && actual.wait.exit_code == Some(0)
                && actual.signals.is_empty()
            {
                return Ok(());
            } else {
                return Err(invalid(
                    "host ended without admitted prepared operation or cold finalization",
                ));
            }
        }
    }
}

// Authenticated INITIAL purpose is retained independently of the CURRENT operation.
struct CampaignBinding {
    directory: PathBuf,
    root: PathBuf,
    bootstrap: Value,
    bootstrap_path: PathBuf,
    bootstrap_sha: String,
    clock: Value,
    clock_sha: String,
}

#[cfg(test)]
mod checkpoint_pause_tests {
    use super::*;

    #[test]
    fn exact_pause_closes_against_the_original_operation_and_retirement() {
        let root = std::env::temp_dir().join(format!("checkpoint-pause-{}", random_id().unwrap()));
        let awaiting_directory = root.join("awaiting");
        let pause_directory = root.join("checkpoint-pauses");
        std::fs::create_dir_all(&awaiting_directory).unwrap();
        std::fs::create_dir(&pause_directory).unwrap();
        let bound = Bound {
            value: json!({"operation_id":"slice", "operation_type":"hpo-train",
                "execution_request":{"mode":"fresh","pause_after_update":0}}),
            path: root.join("operation-slice.json"),
            sha: "a".repeat(64),
            times: [1.; 5],
            clock: CleanupClock::hard_cutoff(Instant::now()),
        };
        let awaiting = json!({"operation_sha256":bound.sha, "run_id":"run"});
        let awaiting_sha = write(&awaiting_directory.join("slice.json"), &awaiting, true).unwrap();
        let physical = json!({"kind":"test physical fixture"});
        let physical_path = root.join("physical.json");
        let physical_sha = write(&physical_path, &physical, true).unwrap();
        let cold = Cold {
            prepared: Prepared {
                value: json!({"state":"paused","committed_update":0}),
                path: root.join("prepared.json"),
                sha: "b".repeat(64),
            },
            exit: Value::Null,
            exit_path: root.join("exit.json"),
            exit_sha: "c".repeat(64),
            admission: json!({"run_id":"run", "invocation_id":"invocation"}),
            physical: Some((physical, physical_path, physical_sha.clone())),
        };
        let campaign = CampaignBinding {
            directory: root.clone(),
            root: root.clone(),
            bootstrap: Value::Null,
            bootstrap_path: root.join("bootstrap.json"),
            bootstrap_sha: String::new(),
            clock: Value::Null,
            clock_sha: String::new(),
        };
        let completion_sha = "d".repeat(64);
        assert!(campaign.checkpoint_pause(&bound, &cold, &completion_sha).is_err());
        let pause = json!({"schema_version":1,"kind":"rust-player-campaign-checkpoint-pause-v1",
            "awaiting_sha256":awaiting_sha,"invocation_id":"invocation","generation":0,
            "completion_sha256":completion_sha,"physical_sha256":physical_sha});
        write(&pause_directory.join("slice.json"), &pause, true).unwrap();
        assert!(campaign.checkpoint_pause(&bound, &cold, &completion_sha).is_ok());
        assert!(campaign.checkpoint_pause(&bound, &cold, &"e".repeat(64)).is_err());
        let mut wrong = bound.clone();
        wrong.value["execution_request"]["pause_after_update"] = json!(1);
        assert!(campaign.checkpoint_pause(&wrong, &cold, &completion_sha).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

impl CampaignBinding {
    fn from_initial(launch: &Value, initial: &Value, locks: &[HeldLock]) -> Result<Option<Self>> {
        if initial["operation_type"] != "campaign-continue" {
            return Ok(None);
        }
        if initial["mode"] != "hpo"
            || !initial["run_id"].is_null()
            || locks.iter().any(|l| l.kind == "parent")
        {
            return Err(invalid("initial campaign control has no parent/admission"));
        }
        let path = PathBuf::from(text(&initial["config_path"])?);
        let (bootstrap, bootstrap_sha) = read_campaign_control(&path, None)?;
        record(
            &bootstrap,
            "rust-player-campaign-bootstrap-v1",
            1,
            &[
                "supervision_id",
                "campaign_directory",
                "mode",
                "loaded_campaign_path",
                "loaded_campaign_sha256",
                "seed_settings_path",
                "seed_settings_sha256",
                "progress",
                "session_clock_path",
                "session_clock_sha256",
                "cursor_path",
            ],
        )?;
        if bootstrap_sha != text(&initial["config_sha256"])?
            || bootstrap["supervision_id"] != launch["supervision_id"]
            || !["fresh", "resume"].contains(&text(&bootstrap["mode"])?)
            || !["auto", "plain", "off"].contains(&text(&bootstrap["progress"])?)
        {
            return Err(invalid("initial campaign bootstrap binding differs"));
        }
        let directory = PathBuf::from(text(&bootstrap["campaign_directory"])?);
        let root = directory
            .join("supervision")
            .join(hex(&launch["supervision_id"], 32)?);
        if path != root.join("bootstrap.json")
            || Path::new(text(&launch["record_directory"])?) != root
            || Path::new(text(&bootstrap["cursor_path"])?)
                != directory.join("supervision/current.json")
        {
            return Err(invalid("canonical actual campaign control root required"));
        }
        for (name, file) in [
            ("loaded_campaign", "loaded-campaign.json"),
            ("seed_settings", "seed-settings.json"),
            ("session_clock", "session-clock.json"),
        ] {
            if Path::new(text(&bootstrap[format!("{name}_path")])?) != root.join(file) {
                return Err(invalid("campaign bootstrap canonical path differs"));
            }
            hex(&bootstrap[format!("{name}_sha256")], 64)?;
        }
        let (clock, clock_sha) =
            read_campaign_control(&root.join("session-clock.json"), Some(&root))?;
        record(
            &clock,
            "rust-player-campaign-clock-v1",
            1,
            &[
                "supervision_id",
                "boot_id",
                "monotonic_before_bits",
                "monotonic_after_bits",
                "unix_anchor_bits",
                "soft_dispatch_finish_bits",
                "hard_session_finish_bits",
                "max_operation_wall_seconds",
                "mapping_recipe",
            ],
        )?;
        let wall = clock["max_operation_wall_seconds"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| invalid("maximum declared total operation wall required"))?;
        let soft = bits(&clock["soft_dispatch_finish_bits"])?;
        let hard = bits(&clock["hard_session_finish_bits"])?;
        if clock_sha != bootstrap["session_clock_sha256"]
            || clock["supervision_id"] != launch["supervision_id"]
            || clock["mapping_recipe"] != "soft-dispatch-plus-max-operation-wall-v1"
            || hard.to_bits() != (soft + wall).to_bits()
            || soft <= bits(&clock["monotonic_after_bits"])?
            || clock["hard_session_finish_bits"] != launch["session_deadlines"]["finish_bits"]
        {
            return Err(invalid("original campaign clock mapping differs"));
        }
        for key in [
            "boot_id",
            "monotonic_before_bits",
            "monotonic_after_bits",
            "unix_anchor_bits",
        ] {
            if clock[key] != launch["clock_binding"][key] {
                return Err(invalid("campaign original launch clock differs"));
            }
        }
        let owner = Self {
            directory,
            root,
            bootstrap,
            bootstrap_path: path,
            bootstrap_sha,
            clock,
            clock_sha,
        };
        owner.recheck(launch, locks)?;
        Ok(Some(owner))
    }
    fn recheck(&self, launch: &Value, locks: &[HeldLock]) -> Result<()> {
        let (initial, sha) = read_campaign_control(
            Path::new(text(&launch["operation_path"])?),
            Some(&self.root),
        )?;
        if sha != launch["operation_sha256"]
            || initial["operation_type"] != "campaign-continue"
            || initial["config_path"] != self.bootstrap_path.to_string_lossy().as_ref()
            || initial["config_sha256"] != self.bootstrap_sha
        {
            return Err(invalid("authenticated INITIAL campaign launch changed"));
        }
        let (value, sha) = read_campaign_control(&self.bootstrap_path, Some(&self.root))?;
        let (clock, clock_sha) =
            read_campaign_control(&self.root.join("session-clock.json"), Some(&self.root))?;
        if value != self.bootstrap
            || sha != self.bootstrap_sha
            || clock != self.clock
            || clock_sha != self.clock_sha
        {
            return Err(invalid(
                "immutable original campaign bootstrap/clock changed",
            ));
        }
        let (loaded, loaded_sha) =
            read_campaign_control(&self.root.join("loaded-campaign.json"), Some(&self.root))?;
        record(
            &loaded,
            "rust-player-loaded-campaign-v1",
            1,
            &[
                "campaign_effective",
                "campaign_sha256",
                "base_effective",
                "base_sha256",
                "inputs",
            ],
        )?;
        if loaded_sha != self.bootstrap["loaded_campaign_sha256"]
            || !loaded["campaign_effective"].is_object()
            || !loaded["base_effective"].is_object()
            || !loaded["inputs"].is_array()
            || loaded["base_sha256"] != python_digest(&loaded["base_effective"])?
            || loaded["campaign_sha256"]
                != python_digest(
                    &json!({"campaign":loaded["campaign_effective"],"base":loaded["base_sha256"]}),
                )?
        {
            return Err(invalid("original campaign CONFIG DATA hash differs"));
        }
        let (seeds, seeds_sha) =
            read_campaign_control(&self.root.join("seed-settings.json"), Some(&self.root))?;
        record(&seeds, "rust-player-seed-settings-v1", 1, &["sources"])?;
        if seeds_sha != self.bootstrap["seed_settings_sha256"] {
            return Err(invalid("original seed-settings snapshot changed"));
        }
        for source in seeds["sources"]
            .as_array()
            .ok_or_else(|| invalid("seed provenance rows required"))?
        {
            object(source, &["path", "sha256", "settings", "origin"])?;
            text(&source["path"])?;
            hex(&source["sha256"], 64)?;
            object(&source["settings"], &["model", "training", "identity"])?;
            if !source["settings"]["model"].is_object()
                || !source["settings"]["training"].is_object()
            {
                return Err(invalid("decoded seed settings CONFIG required"));
            }
            hex(&source["settings"]["identity"], 64)?;
            object(
                &source["origin"],
                &["campaign_id", "study_identity", "proposal_sha256"],
            )?;
            text(&source["origin"]["campaign_id"])?;
            text(&source["origin"]["study_identity"])?;
            text(&source["origin"]["proposal_sha256"])?;
        }
        let held = locks
            .iter()
            .find(|l| l.kind == "campaign")
            .ok_or_else(|| invalid("retained actual campaign OFD required"))?;
        held.recheck()?;
        let info = std::fs::symlink_metadata(self.directory.join("lock"))?;
        if !info.is_file() || (info.dev(), info.ino()) != (held.device, held.inode) {
            return Err(invalid("campaign namespace lock differs from original OFD"));
        }
        Ok(())
    }
    fn checkpoint_pause(&self, bound: &Bound, cold: &Cold, completion_sha: &str) -> Result<()> {
        let operation = text(&bound.value["operation_id"])?;
        safe_campaign_component(operation)?;
        let name = format!("{operation}.json");
        let path = self.root.join("checkpoint-pauses").join(&name);
        let (pause, _) = read_campaign_control(&path, Some(&self.root))?;
        record(
            &pause,
            "rust-player-campaign-checkpoint-pause-v1",
            1,
            &[
                "awaiting_sha256",
                "invocation_id",
                "generation",
                "completion_sha256",
                "physical_sha256",
            ],
        )?;
        let (awaiting, awaiting_sha) =
            read_campaign_control(&self.root.join("awaiting").join(name), Some(&self.root))?;
        let (physical, physical_path, physical_sha) = cold.physical.as_ref().ok_or_else(|| {
            invalid("checkpoint pause requires actual retired training namespace")
        })?;
        if bound.value["operation_type"] != "hpo-train"
            || cold.prepared.value["state"] != "paused"
            || pause["generation"] != cold.prepared.value["committed_update"]
            || pause["generation"] != bound.value["execution_request"]["pause_after_update"]
            || pause["invocation_id"] != cold.admission["invocation_id"]
            || pause["completion_sha256"] != completion_sha
            || pause["physical_sha256"] != *physical_sha
            || pause["awaiting_sha256"] != awaiting_sha
            || awaiting["operation_sha256"] != bound.sha
            || awaiting["run_id"] != cold.admission["run_id"]
        {
            return Err(invalid("checkpoint pause differs from actual awaiting, completion or retirement"));
        }
        let (actual, sha) = read_campaign_control(physical_path, None)?;
        if actual != *physical || sha != *physical_sha {
            return Err(invalid("checkpoint pause physical retirement changed"));
        }
        resync(&path)
    }

    #[allow(clippy::too_many_arguments)]
    fn disposition(
        &self,
        frame: &Value,
        bound: &Bound,
        failed: Option<&FailedReturn>,
        cold: Option<&Cold>,
        finalized: bool,
        completion_path: Option<&Path>,
        completion_sha: Option<&str>,
    ) -> Result<(Value, String)> {
        let operation_id = text(&bound.value["operation_id"])?;
        safe_campaign_component(operation_id)?;
        let path = self
            .root
            .join("terminal")
            .join(format!("{operation_id}.json"));
        if frame["disposition_path"] != path.to_string_lossy().as_ref() {
            return Err(invalid("canonical terminal disposition path required"));
        }
        let (value, sha) = read_campaign_control(&path, Some(&self.root))?;
        record(
            &value,
            "rust-player-campaign-terminal-disposition-v1",
            1,
            &[
                "campaign_id",
                "proposal_no",
                "proposal_sha256",
                "learning_seed",
                "supervision_id",
                "session_clock_sha256",
                "awaiting_path",
                "awaiting_sha256",
                "operation_path",
                "operation_sha256",
                "invocation_id",
                "classification",
                "evidence",
                "proposal_failure_path",
                "proposal_failure_sha256",
                "tell_intent_path",
                "tell_intent_sha256",
                "tell_receipt_path",
                "tell_receipt_sha256",
            ],
        )?;
        let admission = if let Some(failed) = failed {
            &failed.admission
        } else {
            &cold
                .ok_or_else(|| invalid("no actual return pair"))?
                .admission
        };
        if sha != frame["disposition_sha256"]
            || value["supervision_id"] != self.bootstrap["supervision_id"]
            || value["session_clock_sha256"] != self.clock_sha
            || value["operation_sha256"] != bound.sha
            || value["operation_path"] != bound.path.to_string_lossy().as_ref()
            || value["invocation_id"] != admission["invocation_id"]
            || value["campaign_id"] != admission["context"]["campaign_id"]
        {
            return Err(invalid(
                "terminal disposition actual operation/session/invocation differs",
            ));
        }
        let awaiting_path = self
            .root
            .join("awaiting")
            .join(format!("{operation_id}.json"));
        if value["awaiting_path"] != awaiting_path.to_string_lossy().as_ref() {
            return Err(invalid("canonical awaiting path required"));
        }
        let (awaiting, awaiting_sha) = read_campaign_control(&awaiting_path, Some(&self.root))?;
        record(
            &awaiting,
            "rust-player-training-awaiting-v1",
            1,
            &[
                "campaign_id",
                "proposal_no",
                "learning_seed",
                "run_id",
                "run_path",
                "config_path",
                "config_sha256",
                "runtime_binding_sha256",
                "operation_path",
                "operation_sha256",
                "supervision_id",
                "session_clock_sha256",
                "created_unix",
            ],
        )?;
        if awaiting_sha != value["awaiting_sha256"] {
            return Err(invalid("awaiting original hash differs"));
        }
        for key in [
            "campaign_id",
            "proposal_no",
            "learning_seed",
            "operation_path",
            "operation_sha256",
            "supervision_id",
            "session_clock_sha256",
        ] {
            if awaiting[key] != value[key] {
                return Err(invalid("disposition awaiting identity differs"));
            }
        }
        for key in [
            "run_id",
            "config_path",
            "config_sha256",
            "runtime_binding_sha256",
        ] {
            if awaiting[key] != bound.value[key] {
                return Err(invalid("awaiting actual bound operation differs"));
            }
        }
        if awaiting["run_path"] != bound.value["parent_directory"] {
            return Err(invalid("awaiting actual parent differs"));
        }
        uint(&value["learning_seed"])?;
        observation_time(&awaiting["created_unix"])?;
        let evidence = &value["evidence"];
        if let Some(failed) = failed {
            if value["classification"] != "training_failed" {
                return Err(invalid("actual failed return classification required"));
            }
            object(
                evidence,
                &[
                    "failure_path",
                    "failure_sha256",
                    "exit_path",
                    "exit_sha256",
                    "physical_path",
                    "physical_sha256",
                ],
            )?;
            for (name, path, sha, original) in [
                (
                    "failure",
                    &failed.failure.path,
                    &failed.failure.sha,
                    &failed.failure.value,
                ),
                ("exit", &failed.exit_path, &failed.exit_sha, &failed.exit),
                (
                    "physical",
                    &failed.physical_path,
                    &failed.physical_sha,
                    &failed.physical,
                ),
            ] {
                if evidence[format!("{name}_path")] != path.to_string_lossy().as_ref()
                    || evidence[format!("{name}_sha256")] != *sha
                {
                    return Err(invalid(
                        "terminal failed evidence differs from SAME original return",
                    ));
                }
                let (current, checksum) = read_campaign_control(path, None)?;
                if current != *original || checksum != *sha {
                    return Err(invalid("terminal failed original evidence changed"));
                }
                resync(path)?;
            }
        } else {
            let cold = cold.ok_or_else(|| invalid("actual finalized incomplete Cold required"))?;
            let path = completion_path
                .ok_or_else(|| invalid("actual acknowledged completion path required"))?;
            let sha = completion_sha
                .ok_or_else(|| invalid("actual acknowledged completion hash required"))?;
            let (completion, actual) = read_campaign_control(path, Some(&self.root))?;
            record(
                &completion,
                "rust-player-training-completion-v1",
                1,
                &[
                    "supervision_id",
                    "invocation_id",
                    "operation_id",
                    "run_id",
                    "config_sha256",
                    "runtime_binding_sha256",
                    "operation_sha256",
                    "prepared_sha256",
                    "exit_sha256",
                    "state_sha256",
                    "generation_sha256",
                    "source_manifest_sha256",
                    "state",
                    "committed_update",
                    "audit_path",
                    "audit_sha256",
                    "eligible",
                    "parent_metadata_validation",
                ],
            )?;
            for key in [
                "supervision_id",
                "invocation_id",
                "operation_id",
                "run_id",
                "config_sha256",
                "runtime_binding_sha256",
            ] {
                if completion[key] != admission[key] {
                    return Err(invalid("incomplete completion actual admission differs"));
                }
            }
            let (physical, physical_path, physical_sha) = cold
                .physical
                .as_ref()
                .ok_or_else(|| invalid("SAME actual Cold namespace retirement required"))?;
            object(
                evidence,
                &[
                    "completion_path",
                    "completion_sha256",
                    "physical_path",
                    "physical_sha256",
                ],
            )?;
            if !finalized
                || value["classification"] != "healthy_incomplete"
                || actual != sha
                || !["paused", "incomplete"].contains(&text(&cold.prepared.value["state"])?)
                || completion["state"] != cold.prepared.value["state"]
                || completion["eligible"] != false
                || completion["operation_sha256"] != bound.sha
                || completion["invocation_id"] != admission["invocation_id"]
                || completion["prepared_sha256"] != cold.prepared.sha
                || completion["exit_sha256"] != cold.exit_sha
                || evidence["completion_path"] != path.to_string_lossy().as_ref()
                || evidence["completion_sha256"] != sha
                || evidence["physical_path"] != physical_path.to_string_lossy().as_ref()
                || evidence["physical_sha256"] != *physical_sha
            {
                return Err(invalid(
                    "disposition requires genuine finalized ineligible incomplete Cold",
                ));
            }
            let (actual, sha) = read_campaign_control(physical_path, None)?;
            if actual != *physical || sha != *physical_sha {
                return Err(invalid("Cold physical original changed"));
            }
            resync(path)?;
            resync(physical_path)?;
        }
        let number = uint(&value["proposal_no"])?;
        hex(&value["proposal_sha256"], 64)?;
        let mut legacy = Vec::new();
        for (name, file) in [
            ("proposal_failure", "failure.json"),
            ("tell_intent", "tell.intent.json"),
            ("tell_receipt", "tell.receipt.json"),
        ] {
            let expected = self.directory.join(format!("proposals/{number:06}/{file}"));
            if value[format!("{name}_path")] != expected.to_string_lossy().as_ref() {
                return Err(invalid(
                    "canonical actual proposal FAIL evidence path required",
                ));
            }
            let (record, sha) = read_campaign_legacy(&self.directory, number, file)?;
            if value[format!("{name}_sha256")] != sha
                || record["proposal_sha256"] != value["proposal_sha256"]
            {
                return Err(invalid("exact actual proposal FAIL evidence differs"));
            }
            legacy.push(record);
        }
        record(
            &legacy[0],
            "hpo-proposal-failure",
            1,
            &["proposal_sha256", "reason", "boundary"],
        )?;
        text(&legacy[0]["reason"])?;
        text(&legacy[0]["boundary"])?;
        for tell in &legacy[1..] {
            record(
                tell,
                "hpo-tell",
                1,
                &[
                    "proposal_sha256",
                    "trial_number",
                    "params",
                    "distributions",
                    "state",
                    "value",
                    "evaluation_receipt_sha256",
                    "reason",
                ],
            )?;
            if tell["state"] != "FAIL"
                || !tell["value"].is_null()
                || !tell["evaluation_receipt_sha256"].is_null()
                || tell["reason"] != legacy[0]["reason"]
                || !tell["params"].is_object()
                || !tell["distributions"].is_object()
            {
                return Err(invalid(
                    "exact durable FAIL/value=None transaction required",
                ));
            }
            if !tell["trial_number"].is_null() {
                uint(&tell["trial_number"])?;
            }
        }
        if legacy[1] != legacy[2] {
            return Err(invalid("FAIL tell intent and receipt differ"));
        }
        let proposal_path = self
            .directory
            .join(format!("proposals/{number:06}/proposal.json"));
        let (_, proposal_sha) = read_owned(&proposal_path, Some(&self.directory))?;
        if proposal_sha != value["proposal_sha256"] {
            return Err(invalid("terminal actual proposal original differs"));
        }
        resync(&awaiting_path)?;
        resync(&bound.path)?;
        resync(&path)?;
        Ok((value, sha))
    }
}
fn safe_campaign_component(value: &str) -> Result<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        return Err(invalid(
            "safe generated campaign operation identity required",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod entry_component_tests {
    use super::*;
    #[test]
    fn generated_component_contract_has_no_invented_smaller_filename_limit() {
        assert!(safe_campaign_component(&"a".repeat(200)).is_ok());
        assert!(safe_campaign_component("../escape").is_err());
        assert!(safe_campaign_component("x/y").is_err());
    }
}

fn settle_pending_stop_ack(packet: &Packet, pending_stop: &mut Option<u64>) -> bool {
    // Channel.receive has already authenticated actual peer credentials/header.
    if packet::fields(&packet.value, &[]).is_err()
        || !packet.fds.is_empty()
        || packet.value["kind"] != "stop_ack"
        || pending_stop.is_none()
        || *pending_stop != packet.value["seq"].as_u64()
    {
        return false;
    }
    *pending_stop = None;
    true
}

fn terminal_channel_quiet(channel: &mut Channel, pending_stop: &mut Option<u64>) -> bool {
    // Settle at most the ONE already-issued obligation, never a late command or
    // arbitrary discard/resync. A legitimate ACK may cross accepted Prepared.
    if pending_stop.is_some() {
        match channel.receive() {
            Ok(Some(packet)) if settle_pending_stop_ack(&packet, pending_stop) => {}
            _ => return false,
        }
    }
    matches!(channel.receive(), Err(error) if error.kind() == io::ErrorKind::UnexpectedEof)
}

#[cfg(test)]
mod entry_terminal_quiet_tests {
    use super::*;
    #[test]
    fn positive_terminal_quiet_requires_actual_eof_not_empty_or_late_input() {
        let (mut channel, peer) = Channel::pair(
            std::process::id(),
            &"a".repeat(32),
            &"b".repeat(32),
            &"c".repeat(32),
        )
        .unwrap();
        assert!(!terminal_channel_quiet(&mut channel, &mut None));
        packet::send_raw(peer.as_raw_fd(), b"{", &[]).unwrap();
        assert!(!terminal_channel_quiet(&mut channel, &mut None));
        assert!(!terminal_channel_quiet(&mut channel, &mut None));
        drop(peer);
        assert!(terminal_channel_quiet(&mut channel, &mut None));
    }
}

#[cfg(test)]
mod entry_pending_stop_tests {
    use super::*;
    #[test]
    fn only_actual_pending_stop_ack_may_precede_real_terminal_eof() {
        for variant in [
            "valid",
            "duplicate",
            "unsolicited",
            "wrong_seq",
            "wrong_nonce",
            "wrong_peer",
            "extra_fields",
            "attached_fd",
            "extra_after_ack",
            "zero_rights_after_ack",
            "missing_ack",
            "open_after_ack",
        ] {
            let (mut channel, peer) = Channel::pair(
                std::process::id(),
                &"a".repeat(32),
                &"b".repeat(32),
                &"c".repeat(32),
            )
            .unwrap();
            let mut pending = if variant == "unsolicited" {
                None
            } else {
                Some(9)
            };
            let mut ack = json!({"v":1,"kind":"stop_ack","supervision_id":"a".repeat(32),"invocation_id":"b".repeat(32),"nonce":"c".repeat(32),"seq":9});
            if variant == "wrong_seq" {
                ack["seq"] = json!(8);
            }
            if variant == "wrong_nonce" {
                ack["nonce"] = json!("d".repeat(32));
            }
            if variant == "extra_fields" {
                ack["extra"] = json!(true);
            }
            let fd = std::fs::File::open("/dev/null").unwrap();
            let fds = if variant == "attached_fd" {
                vec![fd.as_raw_fd()]
            } else {
                vec![]
            };
            if variant != "missing_ack" {
                packet::send_raw(peer.as_raw_fd(), &serde_json::to_vec(&ack).unwrap(), &fds)
                    .unwrap();
            }
            if variant == "duplicate" {
                packet::send_raw(peer.as_raw_fd(), &serde_json::to_vec(&ack).unwrap(), &[])
                    .unwrap();
            }
            if variant == "extra_after_ack" {
                packet::send_raw(peer.as_raw_fd(), b"{", &[]).unwrap();
            }
            if variant == "zero_rights_after_ack" {
                packet::send_raw(peer.as_raw_fd(), b"", &[fd.as_raw_fd()]).unwrap();
            }
            if variant == "wrong_peer" {
                channel.peer_pid += 1;
            }
            let keep_open = variant == "open_after_ack";
            let held = if keep_open {
                Some(peer)
            } else {
                drop(peer);
                None
            };
            assert_eq!(
                terminal_channel_quiet(&mut channel, &mut pending),
                variant == "valid",
                "{variant}"
            );
            drop(held);
        }
    }
}
