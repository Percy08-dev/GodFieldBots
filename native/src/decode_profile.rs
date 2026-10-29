//! Diagnostic-only whole-file sampling and fixed exclusive TLS accounting.
//! None of this module is compiled into an ordinary native build.
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    marker::PhantomData,
    path::Path,
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Kind {
    Journal,
    Replay,
    Json,
}
impl Kind {
    const ALL: [Self; 3] = [Self::Journal, Self::Replay, Self::Json];
    fn name(self) -> &'static str {
        match self {
            Self::Journal => "journal",
            Self::Replay => "replay",
            Self::Json => "json",
        }
    }
}
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Category {
    Encoding,
    Framing,
    Strings,
    Numbers,
    Keys,
    Containers,
    Budget,
    Allocation,
    Checks,
    Unclassified,
}
impl Category {
    const ALL: [Self; 10] = [
        Self::Encoding,
        Self::Framing,
        Self::Strings,
        Self::Numbers,
        Self::Keys,
        Self::Containers,
        Self::Budget,
        Self::Allocation,
        Self::Checks,
        Self::Unclassified,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Encoding => "encoding",
            Self::Framing => "journal_framing",
            Self::Strings => "strings",
            Self::Numbers => "numbers",
            Self::Keys => "key_hash_duplicate_index",
            Self::Containers => "containers",
            Self::Budget => "budget_reserve_release",
            Self::Allocation => "allocation_copy",
            Self::Checks => "deadline_cancel",
            Self::Unclassified => "unclassified_parser_runtime_instrumentation",
        }
    }
}
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Count {
    Nodes,
    Strings,
    Numbers,
    Containers,
    Allocations,
    CopyBytes,
    Checks,
    Reservations,
}
impl Count {
    const ALL: [Self; 8] = [
        Self::Nodes,
        Self::Strings,
        Self::Numbers,
        Self::Containers,
        Self::Allocations,
        Self::CopyBytes,
        Self::Checks,
        Self::Reservations,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Nodes => "nodes",
            Self::Strings => "strings",
            Self::Numbers => "numbers",
            Self::Containers => "containers",
            Self::Allocations => "allocations",
            Self::CopyBytes => "copied_bytes",
            Self::Checks => "control_checks",
            Self::Reservations => "budget_reservations",
        }
    }
}

#[derive(Clone, Copy)]
struct Capture {
    enabled: bool,
    valid: bool,
}
fn capture() -> Capture {
    static CAPTURE: OnceLock<Capture> = OnceLock::new();
    *CAPTURE.get_or_init(
        || match std::env::var("GFB_NATIVE_DECODE_PROFILE").as_deref() {
            Ok("1") => Capture {
                enabled: true,
                valid: true,
            },
            Ok("0") | Err(std::env::VarError::NotPresent) => Capture {
                enabled: false,
                valid: true,
            },
            _ => Capture {
                enabled: false,
                valid: false,
            },
        },
    )
}
pub fn capture_identity() -> &'static str {
    let capture = capture();
    if !capture.valid {
        "invalid_disabled"
    } else if capture.enabled {
        "on"
    } else {
        "off"
    }
}
pub fn selected(kind: Kind, canonical_path: &Path) -> bool {
    let mut hash = Sha256::new();
    hash.update(b"gfb-native-decode-profile-v1\0");
    hash.update(kind.name().as_bytes());
    hash.update([0]);
    hash.update(canonical_path.as_os_str().as_encoded_bytes());
    let digest = hash.finalize();
    u64::from_be_bytes(digest[..8].try_into().expect("SHA256 prefix")) % 64 == 0
}

struct Local {
    active: bool,
    generation: u64,
    current: Category,
    last: Option<Instant>,
    exclusive: [u64; 10],
    counts: [u64; 8],
    calls: [u64; 10],
}
impl Local {
    const fn new() -> Self {
        Self {
            active: false,
            generation: 0,
            current: Category::Unclassified,
            last: None,
            exclusive: [0; 10],
            counts: [0; 8],
            calls: [0; 10],
        }
    }
    fn charge(&mut self, now: Instant) {
        if let Some(last) = self.last {
            self.exclusive[self.current as usize] = self.exclusive[self.current as usize]
                .saturating_add(now.duration_since(last).as_nanos() as u64);
        }
        self.last = Some(now);
    }
}
thread_local! {static LOCAL:RefCell<Local>=const{RefCell::new(Local::new())};}
pub struct Guard {
    previous: Category,
    generation: u64,
    active: bool,
    _local: PhantomData<Rc<()>>,
}
pub fn enter(category: Category) -> Guard {
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        if !local.active {
            return Guard {
                previous: category,
                generation: 0,
                active: false,
                _local: PhantomData,
            };
        }
        let now = Instant::now();
        local.charge(now);
        let previous = local.current;
        local.current = category;
        local.calls[category as usize] = local.calls[category as usize].saturating_add(1);
        Guard {
            previous,
            generation: local.generation,
            active: true,
            _local: PhantomData,
        }
    })
}
impl Drop for Guard {
    fn drop(&mut self) {
        if self.active {
            LOCAL.with(|local| {
                let mut local = local.borrow_mut();
                if local.active && local.generation == self.generation {
                    local.charge(Instant::now());
                    local.current = self.previous;
                }
            });
        }
    }
}
pub fn count(counter: Count, amount: usize) {
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        if local.active {
            local.counts[counter as usize] =
                local.counts[counter as usize].saturating_add(amount as u64);
        }
    });
}

#[derive(Default)]
struct KindCounters {
    files: AtomicU64,
    bytes: AtomicU64,
    ns: AtomicU64,
    failed: AtomicU64,
    cohort_files: AtomicU64,
    cohort_bytes: AtomicU64,
    cohort_ns: AtomicU64,
    sampled_files: AtomicU64,
    sampled_bytes: AtomicU64,
    sampled_ns: AtomicU64,
    sampled_failed: AtomicU64,
    path_unavailable: AtomicU64,
    nested_suppressed: AtomicU64,
    exclusive: [AtomicU64; 10],
    counts: [AtomicU64; 8],
    calls: [AtomicU64; 10],
}
#[derive(Default)]
pub struct DecodeCounters {
    kinds: [KindCounters; 3],
}
pub struct FileGuard<'a> {
    counters: &'a KindCounters,
    started: Instant,
    bytes: usize,
    sampled: bool,
    cohort: bool,
    success: bool,
    generation: u64,
    _local: PhantomData<Rc<()>>,
}
impl DecodeCounters {
    pub fn begin(&self, kind: Kind, canonical_path: Option<&Path>, bytes: usize) -> FileGuard<'_> {
        let counters = &self.kinds[kind as usize];
        let enabled = capture().enabled;
        let cohort = canonical_path.is_some_and(|path| selected(kind, path));
        if canonical_path.is_none() {
            counters.path_unavailable.fetch_add(1, Ordering::Relaxed);
        }
        let started = Instant::now();
        let mut generation = 0;
        let mut sampled = false;
        if enabled && cohort {
            LOCAL.with(|local| {
                let mut local = local.borrow_mut();
                if local.active {
                    counters.nested_suppressed.fetch_add(1, Ordering::Relaxed);
                } else {
                    let next = local.generation.wrapping_add(1);
                    *local = Local::new();
                    local.generation = next;
                    local.active = true;
                    local.last = Some(started);
                    generation = next;
                    sampled = true;
                }
            });
        }
        FileGuard {
            counters,
            started,
            bytes,
            sampled,
            cohort,
            success: false,
            generation,
            _local: PhantomData,
        }
    }
    pub fn json(&self) -> String {
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        let kinds=Kind::ALL.iter().map(|kind|{let counters=&self.kinds[*kind as usize];
            let exclusive=Category::ALL.iter().map(|category|format!("\"{}\":{}",category.name(),load(&counters.exclusive[*category as usize]) as f64/1e9)).collect::<Vec<_>>().join(",");
            let calls=Category::ALL.iter().map(|category|format!("\"{}\":{}",category.name(),load(&counters.calls[*category as usize]))).collect::<Vec<_>>().join(",");
            let counts=Count::ALL.iter().map(|count|format!("\"{}\":{}",count.name(),load(&counters.counts[*count as usize]))).collect::<Vec<_>>().join(",");
            format!(concat!("\"{}\":{{\"files\":{},\"input_bytes\":{},\"decode_wall_seconds\":{},\"failed_files\":{},\"cohort_files\":{},\"cohort_input_bytes\":{},\"cohort_wall_seconds\":{},\"sampled_files\":{},\"sampled_input_bytes\":{},\"sampled_decode_wall_seconds\":{},\"sampled_failed_files\":{},\"sampling_path_unavailable\":{},\"nested_sampling_suppressed\":{},\"sampled_exclusive_seconds\":{{{}}},\"sampled_category_calls\":{{{}}},\"sampled_counts\":{{{}}}}}"),kind.name(),load(&counters.files),load(&counters.bytes),load(&counters.ns) as f64/1e9,load(&counters.failed),load(&counters.cohort_files),load(&counters.cohort_bytes),load(&counters.cohort_ns) as f64/1e9,load(&counters.sampled_files),load(&counters.sampled_bytes),load(&counters.sampled_ns) as f64/1e9,load(&counters.sampled_failed),load(&counters.path_unavailable),load(&counters.nested_suppressed),exclusive,calls,counts)
        }).collect::<Vec<_>>().join(",");
        format!(
            "{{\"capture\":\"{}\",\"sample_modulus\":64,\"sampling\":\"SHA256(domain,kind,canonical_path);whole-file;no-RNG\",\"scope\":\"all-call decode excludes read/hash; cohort is fixed in capture-off/on; exclusive buckets cover sampled calls only and include hook overhead; residual is not syntax; not a partition of prior62.87s\",\"count_scope\":\"attempted parser nodes/string/number calls and array/object openings; allocation attempts after budget admission; copied bytes cover tracked buffer growth and slice extension, not assignments or allocator internals\",\"kinds\":{{{}}}}}",
            capture_identity(),
            kinds
        )
    }
}
impl FileGuard<'_> {
    pub fn success(&mut self) {
        self.success = true;
    }
}
impl Drop for FileGuard<'_> {
    fn drop(&mut self) {
        let ended = Instant::now();
        let elapsed = ended.duration_since(self.started).as_nanos() as u64;
        let counters = self.counters;
        counters.files.fetch_add(1, Ordering::Relaxed);
        counters
            .bytes
            .fetch_add(self.bytes as u64, Ordering::Relaxed);
        counters.ns.fetch_add(elapsed, Ordering::Relaxed);
        if !self.success {
            counters.failed.fetch_add(1, Ordering::Relaxed);
        }
        if self.cohort {
            counters.cohort_files.fetch_add(1, Ordering::Relaxed);
            counters
                .cohort_bytes
                .fetch_add(self.bytes as u64, Ordering::Relaxed);
            counters.cohort_ns.fetch_add(elapsed, Ordering::Relaxed);
        }
        if self.sampled {
            LOCAL.with(|local| {
                let mut local = local.borrow_mut();
                if local.active && local.generation == self.generation {
                    local.charge(ended);
                    for index in 0..10 {
                        counters.exclusive[index]
                            .fetch_add(local.exclusive[index], Ordering::Relaxed);
                        counters.calls[index].fetch_add(local.calls[index], Ordering::Relaxed);
                    }
                    for index in 0..8 {
                        counters.counts[index].fetch_add(local.counts[index], Ordering::Relaxed);
                    }
                    local.active = false;
                    local.last = None;
                }
            });
            counters.sampled_files.fetch_add(1, Ordering::Relaxed);
            counters
                .sampled_bytes
                .fetch_add(self.bytes as u64, Ordering::Relaxed);
            counters.sampled_ns.fetch_add(elapsed, Ordering::Relaxed);
            if !self.success {
                counters.sampled_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sampled_path(kind: Kind) -> std::path::PathBuf {
        (0..10000)
            .map(|i| std::path::PathBuf::from(format!("/fixed/profile-{i}")))
            .find(|path| selected(kind, path))
            .unwrap()
    }
    fn begin_for_test(counters: &DecodeCounters, kind: Kind) -> FileGuard<'_> {
        let path = sampled_path(kind);
        let counters = &counters.kinds[kind as usize];
        let started = Instant::now();
        LOCAL.with(|local| {
            let mut local = local.borrow_mut();
            let generation = local.generation.wrapping_add(1);
            *local = Local::new();
            local.active = true;
            local.generation = generation;
            local.last = Some(started);
        });
        let generation = LOCAL.with(|local| local.borrow().generation);
        assert!(selected(kind, &path));
        FileGuard {
            counters,
            started,
            bytes: 100,
            sampled: true,
            cohort: true,
            success: false,
            generation,
            _local: PhantomData,
        }
    }
    #[test]
    fn deterministic_whole_file_selection() {
        let path = sampled_path(Kind::Journal);
        for _ in 0..100 {
            assert!(selected(Kind::Journal, &path));
        }
        let forward = (0..10000)
            .filter(|i| selected(Kind::Json, Path::new(&format!("/a/{i}"))))
            .collect::<Vec<_>>();
        let mut reverse = (0..10000)
            .rev()
            .filter(|i| selected(Kind::Json, Path::new(&format!("/a/{i}"))))
            .collect::<Vec<_>>();
        reverse.reverse();
        assert_eq!(forward, reverse);
        assert!((100..220).contains(&forward.len()));
    }
    #[test]
    fn nested_exclusive_intervals_sum_to_sample_wall() {
        let counters = DecodeCounters::default();
        {
            let mut file = begin_for_test(&counters, Kind::Json);
            {
                let _outer = enter(Category::Strings);
                count(Count::Strings, 1);
                {
                    let _inner = enter(Category::Budget);
                    std::hint::black_box(17);
                }
            }
            file.success();
        }
        let c = &counters.kinds[Kind::Json as usize];
        let sum = c
            .exclusive
            .iter()
            .map(|n| n.load(Ordering::Relaxed))
            .sum::<u64>();
        assert_eq!(sum, c.sampled_ns.load(Ordering::Relaxed));
        assert_eq!(c.counts[Count::Strings as usize].load(Ordering::Relaxed), 1);
    }
    #[test]
    fn unwind_and_error_restore_tls() {
        let counters = DecodeCounters::default();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _file = begin_for_test(&counters, Kind::Replay);
            let _outer = enter(Category::Numbers);
            let _inner = enter(Category::Allocation);
            panic!("profile unwind");
        }));
        assert!(failed.is_err());
        assert!(!LOCAL.with(|local| local.borrow().active));
        {
            let _file = begin_for_test(&counters, Kind::Replay);
            let _guard = enter(Category::Checks);
        }
        assert!(!LOCAL.with(|local| local.borrow().active));
        assert_eq!(
            counters.kinds[Kind::Replay as usize]
                .sampled_failed
                .load(Ordering::Relaxed),
            2
        );
    }
}
