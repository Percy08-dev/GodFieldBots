//! Diagnostic-only System forwarding. Requested bytes count every alloc/alloc_zeroed
//! request and the NEW size of realloc requests (not net bytes or resident bytes).
//! Deallocation counts requests; allocator internals/usable sizes are unobserved.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
struct CountSystem;
#[global_allocator]
static GLOBAL: CountSystem = CountSystem;
// SAFETY: this adapter never changes a pointer/layout or takes ownership. Every
// operation forwards exactly once to System under GlobalAlloc's caller contract.
unsafe impl GlobalAlloc for CountSystem {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: unchanged layout forwarded to System; see impl contract above.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: unchanged layout forwarded to System.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        FREES.fetch_add(1, Ordering::Relaxed);
        // SAFETY: unchanged System pointer and layout forwarded to System.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        REALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n as u64, Ordering::Relaxed);
        // SAFETY: unchanged System pointer, old layout and requested size.
        unsafe { System.realloc(p, l, n) }
    }
}
pub fn snapshot() -> [u64; 4] {
    [
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
        REALLOCS.load(Ordering::Relaxed),
    ]
}
