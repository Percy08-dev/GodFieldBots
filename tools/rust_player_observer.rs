//! Standalone Linux process diagnostics. Build with rustc, never Cargo.
//! This tool owns only its original Child; it never signals or reaps descendants.
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_int;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Seek, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const CPU_NS: u64 = 100_000_000;
const PSS_NS: u64 = 1_000_000_000;
#[repr(C)]
struct Timespec {
    seconds: i64,
    nanos: i64,
}
// Linux libc sigaction ABI: sigset_t is 1024 bits. No signal handler is installed.
#[repr(C)]
struct Sigaction {
    handler: usize,
    mask: [u64; 16],
    flags: c_int,
    restorer: usize,
}
unsafe extern "C" {
    fn clock_gettime(clock: c_int, time: *mut Timespec) -> c_int;
    fn sysconf(name: c_int) -> i64;
    fn sigaction(signal: c_int, action: *const Sigaction, previous: *mut Sigaction) -> c_int;
}
fn monotonic() -> io::Result<u64> {
    let mut time = Timespec {
        seconds: 0,
        nanos: 0,
    };
    // SAFETY: clock_gettime writes the valid, aligned Timespec; CLOCK_MONOTONIC=1.
    if unsafe { clock_gettime(1, &mut time) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if time.seconds < 0 || !(0..1_000_000_000).contains(&time.nanos) {
        return Err(io::Error::other("invalid CLOCK_MONOTONIC reading"));
    }
    (time.seconds as u64)
        .checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(time.nanos as u64))
        .ok_or_else(|| io::Error::other("CLOCK_MONOTONIC exceeds u64 nanoseconds"))
}
fn retain_child_status() -> io::Result<()> {
    let action = Sigaction {
        handler: 0,
        mask: [0; 16],
        flags: 0,
        restorer: 0,
    };
    // SAFETY: SIGCHLD=17 on supported Linux targets, SIG_DFL=0, zero flags clear
    // inherited SA_NOCLDWAIT. This standalone executable has no competing reaper.
    if unsafe { sigaction(17, &action, std::ptr::null_mut()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn strings(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|s| quote(s))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn num(value: Option<u64>) -> String {
    value.map_or_else(|| "null".into(), |n| n.to_string())
}
fn text(value: Option<String>) -> String {
    value.map_or_else(|| "null".into(), |s| quote(&s))
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn parse_u64(value: Option<&&str>) -> io::Result<u64> {
    value
        .ok_or_else(|| invalid("short /proc stat"))?
        .parse()
        .map_err(|_| invalid("invalid /proc stat integer"))
}
#[derive(Clone, Debug)]
struct Stat {
    pid: u32,
    ppid: u32,
    start: u64,
    state: String,
    user: u64,
    system: u64,
    rss: Option<u64>,
    threads: u64,
}
fn parse_stat(raw: &str, page: u64) -> io::Result<Stat> {
    let left = raw.find('(').ok_or_else(|| invalid("missing stat comm"))?;
    let right = raw
        .rfind(')')
        .ok_or_else(|| invalid("missing stat comm end"))?;
    if right <= left {
        return Err(invalid("invalid stat comm"));
    }
    let pid = raw[..left]
        .trim()
        .parse::<u32>()
        .map_err(|_| invalid("invalid stat pid"))?;
    let fields: Vec<_> = raw[right + 1..].split_whitespace().collect();
    let ppid = u32::try_from(parse_u64(fields.get(1))?).map_err(|_| invalid("ppid overflow"))?;
    let rss_pages = fields
        .get(21)
        .ok_or_else(|| invalid("missing RSS"))?
        .parse::<i64>()
        .map_err(|_| invalid("invalid RSS"))?;
    Ok(Stat {
        pid,
        ppid,
        start: parse_u64(fields.get(19))?,
        state: fields
            .first()
            .ok_or_else(|| invalid("missing state"))?
            .to_string(),
        user: parse_u64(fields.get(11))?,
        system: parse_u64(fields.get(12))?,
        rss: u64::try_from(rss_pages)
            .ok()
            .and_then(|n| n.checked_mul(page)),
        threads: parse_u64(fields.get(17))?,
    })
}
fn read_stat(pid: u32, page: u64) -> io::Result<Stat> {
    parse_stat(&fs::read_to_string(format!("/proc/{pid}/stat"))?, page)
}
type Owned = BTreeMap<u32, u64>;
fn discover(
    table: &HashMap<u32, Stat>,
    owned: &mut Owned,
    unavailable: impl Fn(u32) -> bool,
    mut confirm_parent: impl FnMut(u32) -> Option<u64>,
) -> usize {
    owned.retain(|pid, start| {
        table
            .get(pid)
            .map_or_else(|| unavailable(*pid), |s| s.start == *start)
    });
    let mut added = 0;
    loop {
        let next: Vec<_> = table
            .values()
            .filter(|s| {
                !owned.contains_key(&s.pid)
                    && owned.get(&s.ppid).is_some_and(|parent_start| {
                        // The table is not atomic: bind this NEW edge to a
                        // positively read parent AFTER candidate observation.
                        *parent_start <= s.start && confirm_parent(s.ppid) == Some(*parent_start)
                    })
            })
            .map(|s| (s.pid, s.start))
            .collect();
        if next.is_empty() {
            break;
        }
        added += next.len();
        owned.extend(next);
    }
    added
}
fn process_table(page: u64) -> io::Result<HashMap<u32, Stat>> {
    let mut table = HashMap::new();
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        if let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        {
            // Unowned metadata/races are transient discovery input, never output.
            if let Ok(stat) = read_stat(pid, page) {
                table.insert(pid, stat);
            }
        }
    }
    Ok(table)
}
fn optional<T>(result: io::Result<T>, label: &str, errors: &mut Vec<String>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(format!("{label}: {error}"));
            None
        }
    }
}
fn task_json(pid: u32) -> io::Result<String> {
    let mut tasks = BTreeMap::new();
    for entry in fs::read_dir(format!("/proc/{pid}/task"))? {
        let entry = entry?;
        let tid = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| invalid("invalid TID"))?;
        let name = fs::read_to_string(entry.path().join("comm"))?;
        tasks.insert(tid, name.strip_suffix('\n').unwrap_or(&name).to_string());
    }
    Ok(format!(
        "[{}]",
        tasks
            .into_iter()
            .map(|(tid, name)| format!("{{\"tid\":{tid},\"name\":{}}}", quote(&name)))
            .collect::<Vec<_>>()
            .join(",")
    ))
}
fn argv_json(pid: u32) -> io::Result<String> {
    let bytes = fs::read(format!("/proc/{pid}/cmdline"))?;
    let mut parts: Vec<_> = bytes.split(|b| *b == 0).collect();
    if parts.last() == Some(&&[][..]) {
        parts.pop();
    }
    let values: Result<Vec<_>, _> = parts
        .into_iter()
        .map(|s| std::str::from_utf8(s).map(str::to_string))
        .collect();
    Ok(strings(
        &values.map_err(|_| invalid("non-UTF8 command line"))?,
    ))
}
fn process_json(pid: u32, start: u64, page: u64, errors: &mut Vec<String>) -> Option<String> {
    let label = format!("process {pid}/{start}");
    let stat = optional(read_stat(pid, page), &label, errors)?;
    if stat.start != start {
        errors.push(format!("{label}: identity changed"));
        return None;
    }
    let exe = optional(
        fs::read_link(format!("/proc/{pid}/exe")).and_then(|p| {
            p.into_os_string()
                .into_string()
                .map_err(|_| invalid("non-UTF8 executable"))
        }),
        &format!("{label} executable"),
        errors,
    );
    let argv =
        optional(argv_json(pid), &format!("{label} argv"), errors).unwrap_or_else(|| "null".into());
    let tasks = optional(task_json(pid), &format!("{label} tasks"), errors)
        .unwrap_or_else(|| "null".into());
    match read_stat(pid, page) {
        Ok(after) if after.start == start => {}
        _ => {
            errors.push(format!("{label}: identity unavailable after reads"));
            return None;
        }
    }
    Some(format!(
        "{{\"pid\":{pid},\"start_ticks\":{start},\"ppid\":{},\"state\":{},\"user_ticks\":{},\"system_ticks\":{},\"rss_bytes\":{},\"threads\":{},\"executable\":{},\"argv\":{argv},\"tasks\":{tasks}}}",
        stat.ppid,
        quote(&stat.state),
        stat.user,
        stat.system,
        num(stat.rss),
        stat.threads,
        text(exe)
    ))
}
fn parse_pss(raw: &str) -> io::Result<u64> {
    let line = raw
        .lines()
        .find(|s| s.starts_with("Pss:"))
        .ok_or_else(|| invalid("missing Pss field"))?;
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() != 3 || fields[2] != "kB" {
        return Err(invalid("invalid Pss units"));
    }
    fields[1]
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| invalid("invalid/overflowing Pss"))
}
fn checked_identity_read(
    expected: u64,
    mut identity: impl FnMut() -> io::Result<u64>,
    read: impl FnOnce() -> io::Result<u64>,
) -> io::Result<u64> {
    if identity()? != expected {
        return Err(invalid("identity changed before PSS"));
    }
    let value = read()?;
    if identity()? != expected {
        return Err(invalid("identity changed after PSS"));
    }
    Ok(value)
}
fn identity_pss(pid: u32, start: u64, page: u64) -> io::Result<u64> {
    checked_identity_read(
        start,
        || read_stat(pid, page).map(|stat| stat.start),
        || parse_pss(&fs::read_to_string(format!("/proc/{pid}/smaps_rollup"))?),
    )
}

fn total_pss(values: &[Option<u64>]) -> Option<u64> {
    values
        .iter()
        .try_fold(0u64, |sum, value| sum.checked_add((*value)?))
}
fn write_row(writer: &mut impl Write, row: &str) -> io::Result<()> {
    writer.write_all(row.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}
#[derive(Default)]
struct Coverage {
    process: u64,
    sweeps: u64,
    complete: u64,
    partial: u64,
    observed: u64,
}
struct Sampler {
    process: File,
    pss: File,
    owned: Owned,
    page: u64,
    coverage: Coverage,
}
impl Sampler {
    fn sample_processes(&mut self) -> io::Result<()> {
        let stamp = monotonic()?;
        let table = process_table(self.page)?;
        self.coverage.observed += discover(
            &table,
            &mut self.owned,
            |pid| Path::new(&format!("/proc/{pid}")).exists(),
            |pid| read_stat(pid, self.page).ok().map(|stat| stat.start),
        ) as u64;
        let mut errors = Vec::new();
        let processes: Vec<_> = self
            .owned
            .iter()
            .filter_map(|(pid, start)| process_json(*pid, *start, self.page, &mut errors))
            .collect();
        write_row(
            &mut self.process,
            &format!(
                "{{\"monotonic_ns\":{stamp},\"processes\":[{}],\"errors\":{}}}",
                processes.join(","),
                strings(&errors)
            ),
        )?;
        self.coverage.process += 1;
        Ok(())
    }
    fn sample_pss(&mut self) -> io::Result<()> {
        let start = monotonic()?;
        let mut processes = Vec::new();
        let mut values = Vec::new();
        let mut errors = Vec::new();
        for (pid, ticks) in &self.owned {
            let before = monotonic()?;
            let reading = identity_pss(*pid, *ticks, self.page);
            let after = monotonic()?;
            let (value, error) = match reading {
                Ok(value) => (Some(value), None),
                Err(error) => {
                    let message = format!("process {pid}/{ticks} PSS: {error}");
                    errors.push(message.clone());
                    (None, Some(message))
                }
            };
            values.push(value);
            processes.push(format!("{{\"pid\":{pid},\"start_ticks\":{ticks},\"start_monotonic_ns\":{before},\"finish_monotonic_ns\":{after},\"pss_bytes\":{},\"error\":{}}}", num(value), text(error)));
        }
        let total = total_pss(&values);
        if total.is_none() && errors.is_empty() {
            errors.push("PSS sum overflow".into());
        }
        let complete = total.is_some();
        let finish = monotonic()?;
        write_row(
            &mut self.pss,
            &format!(
                "{{\"start_monotonic_ns\":{start},\"finish_monotonic_ns\":{finish},\"complete\":{complete},\"total_pss_bytes\":{},\"processes\":[{}],\"errors\":{}}}",
                num(total),
                processes.join(","),
                strings(&errors)
            ),
        )?;
        self.coverage.sweeps += 1;
        if complete {
            self.coverage.complete += 1;
        } else {
            self.coverage.partial += 1;
        }
        Ok(())
    }
    fn run(&mut self, stop: mpsc::Receiver<()>, command_start: u64) -> io::Result<()> {
        let mut next_cpu = command_start
            .checked_add(CPU_NS)
            .ok_or_else(|| invalid("clock overflow"))?;
        let mut next_pss = command_start
            .checked_add(PSS_NS)
            .ok_or_else(|| invalid("clock overflow"))?;
        loop {
            let now = monotonic()?;
            let delay = next_cpu.min(next_pss).saturating_sub(now);
            match stop.recv_timeout(Duration::from_nanos(delay)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.sample_processes()?;
                    self.sample_pss()?; // Shutdown observations remain raw, never auto-eligible.
                    self.process.flush()?;
                    self.pss.flush()?;
                    return Ok(());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            let now = monotonic()?;
            if now >= next_cpu {
                self.sample_processes()?;
                next_cpu = monotonic()?
                    .checked_add(CPU_NS)
                    .ok_or_else(|| invalid("clock overflow"))?;
            }
            if now >= next_pss {
                self.sample_pss()?;
                next_pss = monotonic()?
                    .checked_add(PSS_NS)
                    .ok_or_else(|| invalid("clock overflow"))?;
            }
        }
    }
}
// Read only our known raw output, streaming a fixed scalar prefix. This avoids
// keeping unbounded historical sweeps in memory and filters by the ORIGINAL wait
// interval even when sampler shutdown or main-thread scheduling is delayed.
fn eligible_peak(
    mut reader: impl BufRead,
    command_start: u64,
    command_finish: u64,
) -> io::Result<(Option<u64>, u64)> {
    let mut peak = None;
    let mut count = 0;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if !line.ends_with('\n') {
            return Err(invalid("incomplete PSS row"));
        }
        let prefix = line
            .split_once(",\"processes\":")
            .ok_or_else(|| invalid("incomplete PSS row"))?
            .0;
        let parts: Vec<_> = prefix.split(',').collect();
        if parts.len() != 4 {
            return Err(invalid("invalid PSS scalar prefix"));
        }
        let number = |part: &str| -> io::Result<u64> {
            part.rsplit_once(':')
                .ok_or_else(|| invalid("invalid PSS scalar"))?
                .1
                .parse()
                .map_err(|_| invalid("invalid PSS scalar number"))
        };
        let start = number(parts[0])?;
        let finish = number(parts[1])?;
        let complete = parts[2] == "\"complete\":true";
        // Empty complete shutdown sweeps may sum to zero but never qualify.
        let nonempty = !line.contains(",\"processes\":[],");
        if complete
            && nonempty
            && start >= command_start
            && finish <= command_finish
            && finish >= start
        {
            let value = number(parts[3])?;
            peak = Some(peak.map_or(value, |old: u64| old.max(value)));
            count += 1;
        }
    }
    Ok((peak, count))
}
fn replay_original_pss(
    file: &mut File,
    published: &Path,
    start: u64,
    finish: u64,
) -> io::Result<(Option<u64>, u64)> {
    // The retained original open description prevents its inode from being
    // recycled. Inspect the published path without following it; never open a
    // replacement file, even if it contains plausible JSON from another source.
    let original = file.metadata()?;
    let verify_published = || -> io::Result<()> {
        let current = fs::symlink_metadata(published)?;
        if !current.is_file() || current.dev() != original.dev() || current.ino() != original.ino()
        {
            return Err(invalid(
                "published raw PSS path no longer identifies original file",
            ));
        }
        Ok(())
    };
    verify_published()?;
    file.rewind()?;
    let peak = eligible_peak(BufReader::new(file), start, finish)?;
    verify_published()?;
    Ok(peak)
}

fn wait_and_stamp(child: &mut Child) -> (io::Result<ExitStatus>, io::Result<u64>) {
    let status = child.wait();
    let finish = monotonic(); // Deliberately before sampler stop/drain/join.
    (status, finish)
}
fn wrapper_code(status: Option<&ExitStatus>, observation_failed: bool) -> i32 {
    match status {
        Some(status) => match (status.code(), status.signal()) {
            (Some(0), _) if observation_failed => 2,
            (Some(code), _) => code,
            (_, Some(signal)) => 128 + signal,
            _ => 2,
        },
        None => 2,
    }
}
struct Options {
    output: PathBuf,
    enabled: bool,
    command: Vec<String>,
}
fn options(args: Vec<String>) -> io::Result<Options> {
    if args.len() < 5 || args[1] != "--output" {
        return Err(invalid(
            "usage: observer --output ABSENT_ABSOLUTE_DIRECTORY [--no-sampling] -- COMMAND ARGS",
        ));
    }
    let output = PathBuf::from(&args[2]);
    let mut position = 3;
    let enabled = args[position] != "--no-sampling";
    if !enabled {
        position += 1;
    }
    if args.get(position).map(String::as_str) != Some("--") || args.len() <= position + 1 {
        return Err(invalid("missing -- COMMAND"));
    }
    if !output.is_absolute()
        || output
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
    {
        return Err(invalid(
            "output must be an absent absolute normalized directory",
        ));
    }
    Ok(Options {
        output,
        enabled,
        command: args[position + 1..].to_vec(),
    })
}
fn fresh_file(output: &Path, name: &str, readable: bool) -> io::Result<File> {
    OpenOptions::new()
        .read(readable)
        .write(true)
        .create_new(true)
        .custom_flags(0x20000)
        .open(output.join(name)) // Linux O_NOFOLLOW.
}
fn run(options: Options) -> io::Result<i32> {
    let observer_start = monotonic()?;
    let self_pid = std::process::id();
    // SAFETY: sysconf takes scalar constants and returns a scalar; no pointers.
    let (hz, page) = unsafe { (sysconf(2), sysconf(30)) };
    if hz <= 0 || page <= 0 {
        return Err(io::Error::other("sysconf HZ/page size unavailable"));
    }
    let page = page as u64;
    let clock_boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    let kernel = fs::read_to_string("/proc/sys/kernel/osrelease")?;
    let self_before = read_stat(self_pid, page)?;
    retain_child_status()?;
    let parent = options
        .output
        .parent()
        .ok_or_else(|| invalid("output has no parent"))?;
    if parent.canonicalize()? != parent {
        return Err(invalid("output ancestors must not contain symlinks"));
    }
    fs::create_dir(&options.output)?;
    let stdout = fresh_file(&options.output, "command.stdout.log", false)?;
    let stderr = fresh_file(&options.output, "command.stderr.log", false)?;
    let process_file = fresh_file(&options.output, "process.jsonl", false)?;
    let pss_file = fresh_file(&options.output, "pss.jsonl", true)?;
    let mut pss_replay = pss_file.try_clone()?;
    let mut summary_file = fresh_file(&options.output, "summary.json", false)?;
    let mut command = Command::new(&options.command[0]);
    command
        .args(&options.command[1..])
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    let command_start = monotonic()?;
    let spawn = command.spawn();
    let mut errors = Vec::new();
    let mut coverage = Coverage::default();
    let mut pid = None;
    let mut start_ticks = None;
    let command_finish;
    let mut status = None;
    let mut peak = None;
    let mut eligible = 0;
    match spawn {
        Err(error) => {
            command_finish = optional(monotonic(), "spawn finish clock", &mut errors);
            errors.push(format!("command spawn failed: {error}"));
        }
        Ok(mut child) => {
            pid = Some(child.id() as u64);
            // Child is retained and not yet waited: even an immediate exit keeps
            // its zombie identity available, independent of sampler scheduling.
            let identity = optional(
                read_stat(child.id(), page),
                "original child identity",
                &mut errors,
            );
            let mut owned = Owned::new();
            if let Some(stat) = identity {
                start_ticks = Some(stat.start);
                owned.insert(stat.pid, stat.start);
            }
            let (send, receive) = mpsc::channel();
            let sampler = if options.enabled && !owned.is_empty() {
                let mut sampler = Sampler {
                    process: process_file,
                    pss: pss_file,
                    owned,
                    page,
                    coverage: Coverage {
                        observed: 1,
                        ..Coverage::default()
                    },
                };
                match thread::Builder::new()
                    .name("process-observer".into())
                    .spawn(move || {
                        let result = sampler.run(receive, command_start);
                        (sampler.coverage, result)
                    }) {
                    Ok(handle) => Some(handle),
                    Err(error) => {
                        errors.push(format!("sampler start failed: {error}"));
                        None
                    }
                }
            } else {
                None
            };
            let (wait, finish) = wait_and_stamp(&mut child);
            command_finish = optional(finish, "command wait clock", &mut errors);
            status = optional(wait, "original child wait", &mut errors);
            let _ = send.send(());
            if let Some(handle) = sampler {
                match handle.join() {
                    Ok((observed, result)) => {
                        coverage = observed;
                        if let Err(error) = result {
                            errors.push(format!("sampler failed: {error}"));
                        }
                    }
                    Err(_) => errors.push("sampler panicked; observation incomplete".into()),
                }
            }
            if options.enabled {
                if let Some(finish) = command_finish {
                    match replay_original_pss(
                        &mut pss_replay,
                        &options.output.join("pss.jsonl"),
                        command_start,
                        finish,
                    ) {
                        Ok((value, count)) => {
                            peak = value;
                            eligible = count;
                        }
                        Err(error) => errors.push(format!("PSS eligibility unavailable: {error}")),
                    }
                }
            }
        }
    }
    let self_after = optional(read_stat(self_pid, page), "observer final CPU", &mut errors);
    let observer_finish = optional(monotonic(), "observer finish clock", &mut errors);
    let user = self_after
        .as_ref()
        .and_then(|s| s.user.checked_sub(self_before.user));
    let system = self_after
        .as_ref()
        .and_then(|s| s.system.checked_sub(self_before.system));
    let exit_code = status.as_ref().and_then(ExitStatus::code);
    let signal = status.as_ref().and_then(ExitStatusExt::signal);
    let observed = exit_code.is_some() || signal.is_some();
    let wall = command_finish.and_then(|n| n.checked_sub(command_start));
    let limitations = vec![
        "Diagnostic observations confer no Source, eligibility, GPU, lock or completion authority.".into(),
        "Sampling can miss short-lived processes, CPU work and memory peaks; observed ticks are not final rusage.".into(),
        "PSS is a timestamped sweep, not a simultaneous physical peak, allocator usage or reservation.".into(),
        "Unseen descendants and processes reparented before first discovery cannot be attributed.".into(),
        "Read races and unavailable fields remain explicit in raw rows; partial sweeps cannot qualify as complete peaks.".into(),
        "Tool/source/binary/compiler identity must be verified by the external manifest; no pathname verifies a hash.".into(),
        "Observer CPU/wall excludes final summary serialization and write; raw sample spans include observation work.".into(),
    ];
    let summary = format!(
        concat!(
            "{{\"schema_version\":1,\"kind\":\"rust-player-process-observation-v1\",",
            "\"command\":{{\"argv\":{},\"pid\":{},\"start_ticks\":{},\"start_monotonic_ns\":{},\"finish_monotonic_ns\":{},\"wall_ns\":{}}},",
            "\"clock\":{{\"name\":\"CLOCK_MONOTONIC\",\"boot_id\":{},\"hz\":{},\"page_bytes\":{},\"kernel\":{}}},",
            "\"observer\":{{\"pid\":{},\"start_monotonic_ns\":{},\"finish_monotonic_ns\":{},\"user_ticks\":{},\"system_ticks\":{}}},",
            "\"command_wait\":{{\"exit_code\":{},\"signal\":{},\"observed\":{}}},",
            "\"sampling\":{{\"enabled\":{},\"cpu_interval_ns\":{},\"pss_interval_ns\":{}}},",
            "\"peak_complete_pss_bytes\":{},\"coverage\":{{\"process_samples\":{},\"pss_sweeps\":{},\"complete_pss_sweeps\":{},\"partial_pss_sweeps\":{},\"peak_eligible_pss_sweeps\":{},\"observed_processes\":{}}},",
            "\"errors\":{},\"limitations\":{}}}"
        ),
        strings(&options.command),
        num(pid),
        num(start_ticks),
        command_start,
        num(command_finish),
        num(wall),
        quote(clock_boot.trim()),
        hz,
        page,
        quote(kernel.trim()),
        self_pid,
        observer_start,
        num(observer_finish),
        num(user),
        num(system),
        exit_code.map_or_else(|| "null".into(), |n| n.to_string()),
        signal.map_or_else(|| "null".into(), |n| n.to_string()),
        observed,
        options.enabled,
        CPU_NS,
        PSS_NS,
        num(peak),
        coverage.process,
        coverage.sweeps,
        coverage.complete,
        coverage.partial,
        eligible,
        coverage.observed,
        strings(&errors),
        strings(&limitations)
    );
    let result = wrapper_code(status.as_ref(), !errors.is_empty());
    if let Err(error) = write_row(&mut summary_file, &summary) {
        let _ = writeln!(
            io::stderr(),
            "observer summary write failed after actual command wait: {error}"
        );
        return Ok(wrapper_code(status.as_ref(), true));
    }
    Ok(result)
}
fn main() {
    #[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
    compile_error!("rust_player_observer requires 64-bit Linux libc ABI");
    let result = std::env::args_os()
        .map(|s| {
            s.into_string()
                .map_err(|_| invalid("non-UTF8 CLI argument"))
        })
        .collect::<io::Result<Vec<_>>>()
        .and_then(options)
        .and_then(run);
    std::process::exit(match result {
        Ok(code) => code,
        Err(error) => {
            let _ = writeln!(io::stderr(), "observer: {error}");
            2
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn fixture(pid: u32, ppid: u32, start: u64) -> Stat {
        Stat {
            pid,
            ppid,
            start,
            state: "S".into(),
            user: 11,
            system: 12,
            rss: Some(4096),
            threads: 3,
        }
    }
    #[test]
    fn stat_handles_spaces_parentheses_and_unsigned_bounds() {
        let mut fields = vec!["0"; 22];
        fields[0] = "R";
        fields[1] = "7";
        fields[11] = "11";
        fields[12] = "12";
        fields[17] = "3";
        fields[19] = "18446744073709551615";
        fields[21] = "2";
        let raw = format!("42 (a (tricky) name)) {}", fields.join(" "));
        let s = parse_stat(&raw, 4096).unwrap();
        assert_eq!(
            (s.pid, s.ppid, s.start, s.user, s.system, s.rss, s.threads),
            (42, 7, u64::MAX, 11, 12, Some(8192), 3)
        );
        assert!(parse_stat("1 (bad) R 2", 4096).is_err());
        assert_eq!(
            parse_stat(&raw.replace(" 0 2", " 0 -1"), 4096).unwrap().rss,
            None
        );
    }
    #[test]
    fn ownership_handles_ancestry_reparenting_and_pid_reuse() {
        let mut owned = Owned::from([(1, 10)]);
        let table = HashMap::from([
            (1, fixture(1, 0, 10)),
            (2, fixture(2, 1, 20)),
            (3, fixture(3, 2, 30)),
        ]);
        assert_eq!(
            discover(
                &table,
                &mut owned,
                |_| false,
                |pid| table.get(&pid).map(|s| s.start)
            ),
            2
        );
        let table = HashMap::from([(2, fixture(2, 999, 20)), (3, fixture(3, 2, 30))]);
        assert_eq!(
            discover(
                &table,
                &mut owned,
                |_| false,
                |pid| table.get(&pid).map(|s| s.start)
            ),
            0
        );
        assert_eq!(owned, Owned::from([(2, 20), (3, 30)]));
        let table = HashMap::from([(2, fixture(2, 999, 40)), (4, fixture(4, 2, 50))]);
        discover(
            &table,
            &mut owned,
            |_| false,
            |pid| table.get(&pid).map(|s| s.start),
        );
        assert!(owned.is_empty(), "recycled PID must not inherit ownership");
    }
    #[test]
    fn pss_partial_overflow_and_null_are_explicit() {
        assert_eq!(parse_pss("Rss: 9 kB\nPss: 7 kB\n").unwrap(), 7168);
        assert!(parse_pss("Pss: 18446744073709551615 kB").is_err());
        assert!(parse_pss("Pss: 9 bytes").is_err());
        assert_eq!(total_pss(&[Some(5), None]), None);
        assert_eq!(total_pss(&[Some(u64::MAX), Some(1)]), None);
        assert_eq!(num(None), "null");
        assert_eq!(num(Some(u64::MAX)), "18446744073709551615");
    }
    fn sweep(start: u64, finish: u64, complete: bool, total: &str, members: &str) -> String {
        format!(
            "{{\"start_monotonic_ns\":{start},\"finish_monotonic_ns\":{finish},\"complete\":{complete},\"total_pss_bytes\":{total},\"processes\":[{members}],\"errors\":[]}}\n"
        )
    }
    #[test]
    fn only_nonempty_complete_in_window_sweeps_qualify() {
        let raw = sweep(10, 20, true, "0", "")
            + &sweep(10, 20, false, "null", "{}")
            + &sweep(5, 15, true, "999", "{}")
            + &sweep(20, 31, true, "999", "{}")
            + &sweep(10, 20, true, "40", "{}")
            + &sweep(20, 30, true, "60", "{}");
        assert_eq!(
            eligible_peak(Cursor::new(raw), 10, 30).unwrap(),
            (Some(60), 2)
        );
        assert_eq!(
            eligible_peak(Cursor::new(sweep(10, 20, true, "0", "")), 10, 30).unwrap(),
            (None, 0)
        );
        assert_eq!(eligible_peak(Cursor::new(""), 10, 30).unwrap(), (None, 0));
        assert!(eligible_peak(Cursor::new("partial"), 10, 30).is_err());
    }
    #[test]
    fn truncated_pss_record_cannot_manufacture_an_eligible_peak() {
        let row = sweep(10, 20, true, "999", "{}");
        assert!(eligible_peak(Cursor::new(row.trim_end()), 10, 30).is_err());
        assert!(
            eligible_peak(
                Cursor::new(row.split(",\"errors\":").next().unwrap()),
                10,
                30
            )
            .is_err()
        );
    }
    #[test]
    fn json_escaping_is_valid_and_does_not_reinterpret_unicode() {
        assert_eq!(
            quote("\"\\\n\t\r\0\u{1f}雪"),
            "\"\\\"\\\\\\n\\t\\r\\u0000\\u001f雪\""
        );
    }
    struct FailedWriter {
        flush_only: bool,
    }
    impl Write for FailedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.flush_only {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("injected write failure"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("injected flush failure"))
        }
    }
    #[test]
    fn actual_wait_survives_sampler_write_and_flush_failure() {
        for flush_only in [false, true] {
            let mut child = Command::new("/bin/sh")
                .args(["-c", "exit 29"])
                .spawn()
                .unwrap();
            assert!(write_row(&mut FailedWriter { flush_only }, "diagnostic").is_err());
            let (status, finish) = wait_and_stamp(&mut child);
            assert_eq!(status.unwrap().code(), Some(29));
            assert!(finish.unwrap() <= monotonic().unwrap());
        }
    }
    #[test]
    fn observation_error_cannot_replace_actual_failure_or_forge_success() {
        assert_eq!(wrapper_code(Some(&ExitStatus::from_raw(0)), true), 2);
        assert_eq!(wrapper_code(Some(&ExitStatus::from_raw(23 << 8)), true), 23);
        assert_eq!(wrapper_code(Some(&ExitStatus::from_raw(15)), true), 143);
        assert_eq!(wrapper_code(None, false), 2);
    }
    #[test]
    fn pss_identity_is_checked_both_before_and_after_read() {
        let mut calls = 0;
        let result = checked_identity_read(
            5,
            || {
                calls += 1;
                Ok(if calls == 1 { 5 } else { 6 })
            },
            || Ok(1024),
        );
        assert!(result.unwrap_err().to_string().contains("after"));
        let result =
            checked_identity_read(5, || Ok(6), || panic!("must not read a recycled identity"));
        assert!(result.unwrap_err().to_string().contains("before"));
        assert_eq!(
            checked_identity_read(5, || Ok(5), || Ok(1024)).unwrap(),
            1024
        );
    }
    #[test]
    fn original_wait_stamp_precedes_blocked_sampler_drain() {
        let (release, blocked) = mpsc::channel();
        let drain = thread::spawn(move || {
            blocked.recv().unwrap();
            monotonic().unwrap()
        });
        let mut child = Command::new("/bin/true").spawn().unwrap();
        let (status, finish) = wait_and_stamp(&mut child);
        assert_eq!(status.unwrap().code(), Some(0));
        let finish = finish.unwrap();
        let before_release = monotonic().unwrap();
        release.send(()).unwrap();
        let after_drain = drain.join().unwrap();
        assert!(finish <= before_release && before_release <= after_drain);
    }
    #[test]
    fn sigchld_reset_clears_actual_ignored_and_nocldwait_dispositions() {
        // exec may itself clear SA_NOCLDWAIT on some kernels. Exercise the real
        // in-process disposition immediately before setup, in an isolated test
        // executable so other parallel helper tests cannot lose child statuses.
        const MARKER: &str = "GF_OBSERVER_SIGCHLD_TEST_CHILD";
        if std::env::var_os(MARKER).is_none() {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "tests::sigchld_reset_clears_actual_ignored_and_nocldwait_dispositions",
                    "--nocapture",
                ])
                .env(MARKER, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        for (handler, flags) in [(1, 0), (0, 2)] {
            let action = Sigaction {
                handler,
                mask: [0; 16],
                flags,
                restorer: 0,
            };
            let mut installed = Sigaction {
                handler: 0,
                mask: [0; 16],
                flags: 0,
                restorer: 0,
            };
            // SAFETY: isolated helper subprocess, valid Linux sigaction structs.
            assert_eq!(unsafe { sigaction(17, &action, std::ptr::null_mut()) }, 0);
            assert_eq!(
                unsafe { sigaction(17, std::ptr::null(), &mut installed) },
                0
            );
            assert_eq!(installed.handler, handler);
            assert_eq!(installed.flags & 2, flags);
            retain_child_status().unwrap();
            assert_eq!(
                unsafe { sigaction(17, std::ptr::null(), &mut installed) },
                0
            );
            assert_eq!(installed.handler, 0);
            assert_eq!(installed.flags & 2, 0);
            let mut child = Command::new("/bin/sh")
                .args(["-c", "exit 19"])
                .spawn()
                .unwrap();
            assert_eq!(child.wait().unwrap().code(), Some(19));
        }
    }
    #[test]
    fn missing_parent_identity_cannot_grant_new_ownership() {
        let mut owned = Owned::from([(101, 10)]);
        let table = HashMap::from([(202, fixture(202, 101, 50))]);
        assert_eq!(discover(&table, &mut owned, |_| true, |_| None), 0);
        assert_eq!(owned, Owned::from([(101, 10)]));
    }

    #[test]
    fn matching_but_stale_scan_parent_cannot_grant_new_ownership() {
        let mut owned = Owned::from([(101, 10)]);
        // Parent was read with its old identity before the candidate child.
        // A positive post-candidate read observes that numeric PID recycled.
        let table = HashMap::from([(101, fixture(101, 0, 10)), (202, fixture(202, 101, 50))]);
        let mut checked = Vec::new();
        let added = discover(
            &table,
            &mut owned,
            |_| true,
            |pid| {
                checked.push(pid);
                Some(40)
            },
        );
        assert_eq!(added, 0);
        assert_eq!(checked, vec![101]);
        assert_eq!(owned, Owned::from([(101, 10)]));
    }
    #[test]
    fn equal_tick_parent_and_child_bind_without_rechecking_existing_owners() {
        let mut owned = Owned::from([(101, 10), (303, 5)]);
        let table = HashMap::from([
            (101, fixture(101, 0, 10)),
            (202, fixture(202, 101, 10)),
            (303, fixture(303, 999, 5)), // already observed, now reparented
        ]);
        let mut checked = Vec::new();
        let added = discover(
            &table,
            &mut owned,
            |_| false,
            |pid| {
                checked.push(pid);
                Some(10)
            },
        );
        assert_eq!(added, 1);
        assert_eq!(checked, vec![101]);
        assert_eq!(owned, Owned::from([(101, 10), (202, 10), (303, 5)]));
    }
}
