//! Original Child/group custody and positive Linux subreaper waits.
use crate::{Result, WaitRecord, invalid, monotonic, wait_record};
use serde::Serialize;
use std::{
    fs, io,
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            process::CommandExt,
        },
    },
    process::{Child, Command},
    time::{Duration, Instant},
};

pub(crate) fn enable_subreaper() -> Result<()> {
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(crate) fn start_ticks(pid: u32) -> Result<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let fields = stat
        .rsplit_once(')')
        .ok_or_else(|| invalid("invalid process identity"))?
        .1;
    fields
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| invalid("missing process start"))?
        .parse()
        .map_err(|_| invalid("invalid process start"))
}
#[derive(Debug, Serialize, Clone)]
pub(crate) struct HostIdentity {
    pub pid: u32,
    pub start_ticks: u64,
    pub pgid: i32,
}
#[derive(Serialize)]
pub(crate) struct Descendant {
    pub pid: u32,
    pub start_ticks: u64,
    pub pgid: i32,
    pub wait: WaitRecord,
}
#[derive(Serialize)]
pub(crate) struct GroupSignal {
    pub pgid: i32,
    pub signal: i32,
    pub delivered: bool,
}
#[derive(Serialize)]
pub(crate) struct Retirement {
    #[serde(skip)]
    pub origin: HostIdentity,
    pub wait: WaitRecord,
    pub descendants: Vec<Descendant>,
    pub signals: Vec<GroupSignal>,
    pub descendants_reaped: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct CleanupClock {
    pub term: Instant,
    pub kill: Instant,
    pub finish: Instant,
    hard: bool,
}
impl CleanupClock {
    pub fn hard_cutoff(finish: Instant) -> Self {
        Self {
            term: finish,
            kill: finish,
            finish,
            hard: true,
        }
    }
    pub fn new(term: f64, kill: f64, finish: f64) -> Result<Self> {
        if !(term.is_finite() && term <= kill && kill < finish) {
            return Err(invalid("invalid cleanup milestones"));
        }
        let now = Instant::now();
        let mono = monotonic()?;
        let checked = |deadline: f64| -> Result<Instant> {
            let duration = Duration::try_from_secs_f64((deadline - mono).max(0.0))
                .map_err(|_| invalid("unsupported cleanup duration"))?;
            now.checked_add(duration)
                .ok_or_else(|| invalid("unsupported cleanup clock"))
        };
        Ok(Self {
            term: checked(term)?,
            kill: checked(kill)?,
            finish: checked(finish)?,
            hard: false,
        })
    }
}
fn normalize_sigchld() -> Result<()> {
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = libc::SIG_DFL;
    action.sa_flags = 0; // In particular, never inherit SA_NOCLDWAIT/SIG_IGN.
    unsafe {
        libc::sigemptyset(&mut action.sa_mask);
    }
    if unsafe { libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn observe_pid(pid: u32) -> Result<bool> {
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    if unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    match unsafe { info.si_pid() } {
        0 => Ok(false),
        found if found == pid as i32 => Ok(true),
        _ => Err(invalid("targeted original wait identity mismatch")),
    }
}

pub(crate) struct OwnedHost {
    child: Child,
    pub identity: HostIdentity,
    leader_retained: bool,
}
impl OwnedHost {
    pub fn spawn(command: &mut Command, deadline: Instant) -> Result<Self> {
        normalize_sigchld()?;
        // SAFETY: only async-signal-safe setsid runs between fork and exec.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn()?;
        let pid = child.id();
        let identity = (|| {
            let pgid = unsafe { libc::getpgid(pid as i32) };
            if pgid != pid as i32 || pgid <= 1 || pgid == unsafe { libc::getpgrp() } {
                return Err(invalid("host group isolation failed"));
            }
            Ok(HostIdentity {
                pid,
                start_ticks: start_ticks(pid)?,
                pgid,
            })
        })();
        match identity {
            Ok(identity) => Ok(Self {
                child,
                identity,
                leader_retained: true,
            }),
            Err(error) => {
                // No group authority was issued. Target only the original returned Child.
                let _ = child.kill();
                while Instant::now() < deadline {
                    match observe_pid(pid) {
                        Ok(true) => {
                            let _ = child.wait();
                            return Err(error);
                        }
                        Err(wait_error) => return Err(wait_error),
                        Ok(false) => std::thread::sleep(Duration::from_millis(2)),
                    }
                }
                if matches!(observe_pid(pid), Ok(true)) {
                    let _ = child.wait();
                }
                Err(invalid(
                    "setup child retirement unconfirmed at original deadline",
                ))
            }
        }
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn observe_exit(&mut self) -> Result<bool> {
        if !self.leader_retained {
            return Err(invalid("original leader wait ownership ended"));
        }
        match observe_pid(self.pid()) {
            Ok(exited) => Ok(exited),
            Err(error) => {
                self.leader_retained = false;
                Err(error)
            }
        }
    }
    fn signal_group(&mut self, signal: i32) -> Result<GroupSignal> {
        // Even a zombie remains the exclusive, unreaped original session leader.
        // This observation MUST NOT consume it. ECHILD permanently removes authority.
        self.observe_exit()?;
        let pgid = self.identity.pgid;
        if self.identity.pid != self.pid()
            || pgid != self.pid() as i32
            || pgid <= 1
            || pgid == unsafe { libc::getpgrp() }
        {
            return Err(invalid("unsafe original host group"));
        }
        let delivered = unsafe { libc::kill(-pgid, signal) } == 0;
        if !delivered && io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
            return Err(io::Error::last_os_error());
        }
        Ok(GroupSignal {
            pgid,
            signal,
            delivered,
        })
    }
    fn adopted_pids(&self) -> Result<Vec<u32>> {
        fs::read_to_string(format!("/proc/self/task/{}/children", std::process::id()))?
            .split_whitespace()
            .map(|pid| pid.parse().map_err(|_| invalid("unknown adopted child")))
            .filter(|pid| !matches!(pid, Ok(found) if *found == self.pid()))
            .collect()
    }
    fn drain_adopted(
        &mut self,
        descendants: &mut Vec<Descendant>,
        containment: &mut bool,
    ) -> Result<bool> {
        for pid in self.adopted_pids()? {
            // Target each adopted child: P_ALL would repeatedly select the pinned leader.
            if !observe_pid(pid)? {
                let pgid = unsafe { libc::getpgid(pid as i32) };
                if pgid < 0 {
                    return Err(io::Error::last_os_error());
                }
                if pgid != self.identity.pgid {
                    *containment = false;
                }
                continue;
            }
            let ticks = start_ticks(pid)?;
            let pgid = unsafe { libc::getpgid(pid as i32) };
            if pgid < 0 {
                return Err(io::Error::last_os_error());
            }
            if pgid != self.identity.pgid {
                *containment = false;
            }
            let mut status = 0;
            if unsafe { libc::waitpid(pid as i32, &mut status, libc::WNOHANG) } != pid as i32 {
                return Err(invalid("targeted adopted wait remains unconfirmed"));
            }
            use std::os::unix::process::ExitStatusExt;
            descendants.push(Descendant {
                pid,
                start_ticks: ticks,
                pgid,
                wait: wait_record(std::process::ExitStatus::from_raw(status)),
            });
        }
        Ok(self.adopted_pids()?.is_empty())
    }
    fn consume_original(&mut self) -> Result<WaitRecord> {
        if !self.observe_exit()? {
            return Err(invalid("live original cannot be consumed"));
        }
        // No later code path may signal the group, including wait errors.
        self.leader_retained = false;
        Ok(wait_record(self.child.wait()?))
    }
    fn final_exhaustion() -> Result<()> {
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_ALL,
                0,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD) {
            return Ok(());
        }
        Err(invalid("final adopted-child exhaustion unconfirmed"))
    }
    fn end_signal_authority(&mut self) {
        let waitable = self.leader_retained && matches!(observe_pid(self.pid()), Ok(true));
        self.leader_retained = false;
        if waitable {
            let _ = self.child.wait();
        } // Already waitable, never wait on live original.
    }
    pub fn retire(&mut self, clock: CleanupClock) -> Result<Retirement> {
        let mut descendants = Vec::new();
        let mut signals = Vec::new();
        let mut term = false;
        let mut kill = false;
        let mut containment = true;
        let result = (|| {
            loop {
                let original_exited = self.observe_exit()?;
                let drained = self.drain_adopted(&mut descendants, &mut containment)?;
                if original_exited && drained {
                    let wait = self.consume_original()?;
                    Self::final_exhaustion()?;
                    return Ok(Retirement {
                        origin: self.identity.clone(),
                        wait,
                        descendants,
                        signals,
                        descendants_reaped: containment,
                    });
                }
                let now = Instant::now();
                if now >= clock.finish {
                    if clock.hard {
                        // Contain at the ORIGINAL cutoff, then observe only nonblocking facts.
                        // No renewed TERM/KILL/wait allowance belongs to cold/session control.
                        signals.push(self.signal_group(libc::SIGKILL)?);
                        let exited = self.observe_exit()?;
                        let drained = self.drain_adopted(&mut descendants, &mut containment)?;
                        if exited && drained {
                            let wait = self.consume_original()?;
                            Self::final_exhaustion()?;
                            return Ok(Retirement {
                                origin: self.identity.clone(),
                                wait,
                                descendants,
                                signals,
                                descendants_reaped: containment,
                            });
                        }
                    }
                    return Err(invalid("original descendant reap deadline expired"));
                }
                for (milestone, signal, sent) in [
                    (clock.term, libc::SIGTERM, &mut term),
                    (clock.kill, libc::SIGKILL, &mut kill),
                ] {
                    if !*sent && now >= milestone {
                        signals.push(self.signal_group(signal)?);
                        *sent = true;
                    }
                }
                std::thread::sleep(
                    Duration::from_millis(2)
                        .min(clock.finish.saturating_duration_since(Instant::now())),
                );
            }
        })();
        if result.is_err() {
            self.end_signal_authority();
        }
        result
    }
}

pub(crate) struct HeldLock {
    pub fd: OwnedFd,
    pub kind: String,
    pub device: u64,
    pub inode: u64,
}
impl HeldLock {
    pub fn adopt(fd: OwnedFd, kind: &str, device: u64, inode: u64) -> Result<Self> {
        if !["parent", "campaign"].contains(&kind) {
            return Err(invalid("unknown custody kind"));
        }
        let held = Self {
            fd,
            kind: kind.to_owned(),
            device,
            inode,
        };
        held.recheck()?;
        Ok(held)
    }
    pub fn recheck(&self) -> Result<()> {
        let fd = self.fd.as_raw_fd();
        let path = format!("/proc/self/fd/{fd}");
        let info = fs::metadata(&path)?;
        if !info.is_file()
            || info.uid() != unsafe { libc::getuid() }
            || info.nlink() != 1
            || info.dev() != self.device
            || info.ino() != self.inode
        {
            return Err(invalid("original custody inode/owner changed"));
        }
        let state = fs::read_to_string(format!("/proc/self/fdinfo/{fd}"))?;
        if !state.lines().any(|line| {
            line.starts_with("lock:")
                && line.split_whitespace().any(|s| s == "FLOCK")
                && line.split_whitespace().any(|s| s == "WRITE")
        }) {
            return Err(invalid("original OFD no longer holds exclusive flock"));
        }
        let competitor = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(path)?;
        let result = unsafe { libc::flock(competitor.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            return Err(invalid("original custody lost exclusive exclusion"));
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(error);
        }
        Ok(())
    }
}
