//! Supported Linux/WSL entry-stack constraint for the bounded recursive decoder.
use crate::error::{Error, Result};

#[cfg(target_os = "linux")]
pub fn ensure_minimum_stack() -> Result<()> {
    let mut attributes = std::mem::MaybeUninit::<libc::pthread_attr_t>::uninit();
    // SAFETY: Linux fills attributes for the calling live pthread. Every
    // successful initialization is destroyed before this function returns.
    let initialized =
        unsafe { libc::pthread_getattr_np(libc::pthread_self(), attributes.as_mut_ptr()) };
    if initialized != 0 {
        return Err(Error::resource("native pthread stack capacity unavailable"));
    }
    let mut attributes = unsafe { attributes.assume_init() };
    let mut address = std::ptr::null_mut();
    let mut size = 0usize;
    let inspected = unsafe { libc::pthread_attr_getstack(&attributes, &mut address, &mut size) };
    unsafe { libc::pthread_attr_destroy(&mut attributes) };
    if inspected != 0 {
        return Err(Error::resource("native pthread stack capacity unavailable"));
    }
    let mut marker = 0u8;
    let current = std::hint::black_box(&mut marker) as *mut u8 as usize;
    let bottom = address as usize;
    let end = bottom
        .checked_add(size)
        .ok_or_else(|| Error::resource("native pthread stack bounds overflow"))?;
    // Linux/WSL x86-64 stacks grow downward. Main-thread attributes can exclude
    // guard/argument pages, so nominal 8MiB allows 64KiB of such accounting.
    if size < 8 * 1024 * 1024 - 65536
        || current < bottom
        || current > end
        || current - bottom < 7 * 1024 * 1024
    {
        return Err(Error::resource(
            "native preprocessing requires nominal 8MiB stack and 7MiB available headroom",
        ));
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
pub fn ensure_minimum_stack() -> Result<()> {
    Err(Error::resource(
        "native preprocessing stack checks support Linux/WSL only",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_small_stack_is_rejected_before_deep_work() {
        std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(|| {
                assert!(matches!(
                    ensure_minimum_stack(),
                    Err(Error::ResourceLimit(_))
                ));
            })
            .unwrap()
            .join()
            .unwrap();
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| ensure_minimum_stack().unwrap())
            .unwrap()
            .join()
            .unwrap();
    }
}
