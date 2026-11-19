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
}
#[derive(Clone, Copy, PartialEq)]
enum ControlPhase {
    Training,
    Cold,
    Session,
}
impl ControlPhase {
    fn clock(self, bound: &Bound, session_cutoff: Instant) -> CleanupClock {
        match self {
            Self::Training => bound.clock,
            Self::Cold => CleanupClock::hard_cutoff(bound.clock.finish),
            Self::Session => CleanupClock::hard_cutoff(session_cutoff),
        }
    }
}
pub(crate) struct Supervisor {
    launch: LaunchOwner,
    root: PathBuf,
    mode: String,
    session_finish: f64,
    session_cutoff: Instant,
    locks: Vec<HeldLock>,
    bound: Bound,
    seen: BTreeMap<String, String>,
    #[cfg(test)]
    pub test_command: Option<Vec<String>>,
    #[cfg(test)]
    pub test_namespace: Option<PathBuf>,
}
impl Supervisor {
    pub fn new(launch_owner: LaunchOwner) -> Result<Self> {
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
            session_cutoff,
            locks,
            bound,
            seen,
            #[cfg(test)]
            test_command: None,
            #[cfg(test)]
            test_namespace: None,
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
        if let Some(cold) = cold {
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
        prepared(&value, admission)?;
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
        let mut cold: Option<Cold> = None;
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
            let (mut host, mut channel) =
                self.spawn(if cold.is_some() { "cold" } else { "host" })?;
            let mut phase = if cold.is_some() {
                ControlPhase::Cold
            } else {
                ControlPhase::Training
            };
            let mut protocol_failed = false;
            let mut admission: Option<Value> = None;
            let mut namespace: Option<Namespace> = None;
            let mut durable = true;
            let mut prepared_result: Option<Prepared> = None;
            let mut finalized = false;
            let mut finalized_sha: Option<String> = None;
            let handshake = self.handshake(&mut host, &mut channel, cold.as_ref());
            let mut server_seq = match handshake {
                Ok(seq) => seq,
                Err(_) => {
                    protocol_failed = true;
                    0
                }
            };
            let mut host_seq = 1;
            let mut stopped = false;
            let mut pending_stop: Option<u64> = None;
            let control_result = (|| -> Result<()> {
                while !protocol_failed && prepared_result.is_none() {
                    let authority = phase.clock(&self.bound, self.session_cutoff);
                    if Instant::now() >= authority.finish {
                        protocol_failed = true;
                        break;
                    }
                    if host.observe_exit()? {
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
                            std::thread::sleep(Duration::from_millis(2));
                            continue;
                        }
                        Err(e) => {
                            if !(finalized && e.kind() == io::ErrorKind::UnexpectedEof) {
                                protocol_failed = true;
                            }
                            break;
                        }
                    };
                    let frame = &packet.value;
                    let kind = text(&frame["kind"])?;
                    if kind == "stop_ack" {
                        if packet::fields(frame, &[]).is_err()
                            || !packet.fds.is_empty()
                            || pending_stop != frame["seq"].as_u64()
                        {
                            protocol_failed = true;
                            break;
                        }
                        pending_stop = None;
                        continue;
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
                                    || self.locks.iter().any(|l| l.kind == "parent")
                                    || !packet.fds.is_empty()
                                {
                                    return Err(invalid(
                                        "operation overlaps active/unfinalized original",
                                    ));
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
                                if self.seen.get(id).is_some_and(|sha| sha != &next.sha) {
                                    return Err(invalid(
                                        "same operation cannot reset original deadlines",
                                    ));
                                }
                                channel.send("operation_bound",seq,json!({"operation_id":next.value["operation_id"],"operation_sha256":next.sha}))?;
                                self.seen.insert(id.to_owned(), next.sha.clone());
                                self.bound = next;
                                phase = ControlPhase::Training;
                                stopped = false;
                                pending_stop = None;
                            }
                            "lock" => {
                                packet::fields(frame, &["lock_kind", "device", "inode"])?;
                                if packet.fds.len() != 1
                                    || admission.is_some()
                                    || cold.is_some() && !finalized
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
                            "prepared" => {
                                packet::fields(frame, &["prepared_path", "prepared_sha256"])?;
                                if !packet.fds.is_empty() {
                                    return Err(invalid("unexpected prepared FD"));
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
                                if sha != text(&frame["completion_sha256"])?
                                    || completion["schema_version"] != 1
                                    || completion["kind"] != "rust-player-training-completion-v1"
                                    || completion["prepared_sha256"] != previous.prepared.sha
                                    || completion["exit_sha256"] != previous.exit_sha
                                {
                                    return Err(invalid("cold finalizer output/pair mismatch"));
                                }
                                if finalized_sha.as_ref().is_some_and(|old| old != &sha) {
                                    return Err(invalid("competing finalizer output differs"));
                                }
                                resync(path)?;
                                finalized = true;
                                finalized_sha = Some(sha.clone());
                                channel.send(
                                    "finalized_ack",
                                    seq,
                                    json!({"completion_sha256":sha}),
                                )?;
                            }
                            "release_parent" => {
                                packet::fields(frame, &["device", "inode"])?;
                                if !packet.fds.is_empty()
                                    || !finalized
                                    || admission.is_some()
                                    || phase != ControlPhase::Cold
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
                if let Some(owner) = namespace.take() {
                    owner.confirm(&exit_record, &actual)?;
                    drop(owner);
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
                });
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
