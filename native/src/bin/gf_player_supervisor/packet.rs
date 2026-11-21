//! Linux per-message authenticated control, with ancillary custody per packet.
use crate::{
    Result, invalid,
    records::{object, text, uint},
};
use serde_json::{Value, json};
use std::{
    io, mem,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
};
pub(crate) const LIMIT: usize = 65536;
pub(crate) struct Packet {
    pub value: Value,
    pub fds: Vec<OwnedFd>,
}
pub(crate) struct Channel {
    pub fd: OwnedFd,
    pub peer_pid: u32,
    pub supervision: String,
    pub invocation: String,
    pub nonce: String,
}
impl Channel {
    pub fn pair(
        peer_pid: u32,
        supervision: &str,
        invocation: &str,
        nonce: &str,
    ) -> Result<(Self, OwnedFd)> {
        let mut sockets = [-1; 2];
        if unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
                0,
                sockets.as_mut_ptr(),
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        let (left, right) = unsafe {
            (
                OwnedFd::from_raw_fd(sockets[0]),
                OwnedFd::from_raw_fd(sockets[1]),
            )
        };
        for fd in [&left, &right] {
            let yes: libc::c_int = 1;
            if unsafe {
                libc::setsockopt(
                    fd.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_PASSCRED,
                    (&yes as *const libc::c_int).cast(),
                    mem::size_of_val(&yes) as u32,
                )
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
        }
        Ok((
            Self {
                fd: left,
                peer_pid,
                supervision: supervision.to_owned(),
                invocation: invocation.to_owned(),
                nonce: nonce.to_owned(),
            },
            right,
        ))
    }
    pub fn send(&self, kind: &str, seq: u64, payload: Value) -> Result<()> {
        let mut frame = json!({"v":1,"kind":kind,"supervision_id":self.supervision,"invocation_id":self.invocation,"nonce":self.nonce,"seq":seq});
        let map = payload
            .as_object()
            .ok_or_else(|| invalid("compact payload object required"))?;
        for (key, value) in map {
            if frame.as_object().unwrap().contains_key(key) {
                return Err(invalid("reserved control envelope key"));
            }
            frame[key] = value.clone();
        }
        let bytes = serde_json::to_vec(&frame).map_err(|_| invalid("invalid control JSON"))?;
        if bytes.len() > LIMIT {
            return Err(invalid("control packet exceeds compact limit"));
        }
        send_raw(self.fd.as_raw_fd(), &bytes, &[])
    }
    pub fn receive(&mut self) -> Result<Option<Packet>> {
        let mut bytes = vec![0u8; LIMIT];
        let mut control = [0usize; 32];
        let mut iov = libc::iovec {
            iov_base: bytes.as_mut_ptr().cast(),
            iov_len: bytes.len(),
        };
        let mut message: libc::msghdr = unsafe { mem::zeroed() };
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = mem::size_of_val(&control);
        let length = unsafe {
            libc::recvmsg(
                self.fd.as_raw_fd(),
                &mut message,
                libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC,
            )
        };
        if length == -1 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::WouldBlock {
                return Ok(None);
            }
            return Err(e);
        }
        let mut credentials = None;
        let mut fds = Vec::new();
        let mut unknown = false;
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&message);
            while !cmsg.is_null() {
                let item = &*cmsg;
                let base = libc::CMSG_LEN(0) as usize;
                if item.cmsg_len < base {
                    return Err(invalid("invalid control ancillary header"));
                }
                if item.cmsg_level == libc::SOL_SOCKET && item.cmsg_type == libc::SCM_RIGHTS {
                    let size = item.cmsg_len - base;
                    if size % mem::size_of::<RawFd>() != 0 {
                        return Err(invalid("invalid control descriptor count"));
                    }
                    let ptr = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                    for index in 0..size / mem::size_of::<RawFd>() {
                        fds.push(OwnedFd::from_raw_fd(*ptr.add(index)));
                    }
                } else if item.cmsg_level == libc::SOL_SOCKET
                    && item.cmsg_type == libc::SCM_CREDENTIALS
                {
                    if item.cmsg_len
                        != libc::CMSG_LEN(mem::size_of::<libc::ucred>() as u32) as usize
                        || credentials.is_some()
                    {
                        return Err(invalid("invalid control credentials"));
                    }
                    credentials = Some(std::ptr::read_unaligned(
                        libc::CMSG_DATA(cmsg).cast::<libc::ucred>(),
                    ));
                } else {
                    unknown = true;
                }
                cmsg = libc::CMSG_NXTHDR(&message, cmsg);
            }
        }
        if length == 0 {
            // SO_PASSCRED supplies ancillary data for an actual queued empty
            // datagram. True stream termination has no message/rights/credentials.
            // All received rights above are already OwnedFds before either arm.
            if message.msg_controllen != 0
                || credentials.is_some()
                || !fds.is_empty()
                || unknown
                || message.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0
            {
                return Err(invalid("zero-length control packet is not terminal EOF"));
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "control EOF requests cleanup",
            ));
        }
        if message.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 || unknown {
            return Err(invalid("truncated/unknown control packet"));
        }
        let credentials =
            credentials.ok_or_else(|| invalid("kernel control credentials missing"))?;
        if credentials.pid != self.peer_pid as i32 || credentials.uid != unsafe { libc::getuid() } {
            return Err(invalid("control sender is not original host"));
        }
        let value: Value = serde_json::from_slice(&bytes[..length as usize])
            .map_err(|_| invalid("invalid compact control JSON"))?;
        if value["v"] != 1
            || value["supervision_id"] != self.supervision
            || value["invocation_id"] != self.invocation
            || value["nonce"] != self.nonce
        {
            return Err(invalid("control authentication binding mismatch"));
        }
        text(&value["kind"])?;
        uint(&value["seq"])?;
        Ok(Some(Packet { value, fds }))
    }
}
pub(crate) fn fields(value: &Value, keys: &[&str]) -> Result<()> {
    let mut all = vec![
        "v",
        "kind",
        "supervision_id",
        "invocation_id",
        "nonce",
        "seq",
    ];
    all.extend(keys);
    object(value, &all)?;
    Ok(())
}
pub(crate) fn send_raw(fd: RawFd, bytes: &[u8], fds: &[RawFd]) -> Result<()> {
    let mut iov = libc::iovec {
        iov_base: bytes.as_ptr() as *mut _,
        iov_len: bytes.len(),
    };
    let mut control = [0usize; 32];
    let mut message: libc::msghdr = unsafe { mem::zeroed() };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    if !fds.is_empty() {
        let size = mem::size_of_val(fds);
        let space = unsafe { libc::CMSG_SPACE(size as u32) } as usize;
        if space > mem::size_of_val(&control) {
            return Err(invalid("too many control descriptors"));
        }
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = space;
        unsafe {
            let header = libc::CMSG_FIRSTHDR(&message);
            (*header).cmsg_level = libc::SOL_SOCKET;
            (*header).cmsg_type = libc::SCM_RIGHTS;
            (*header).cmsg_len = libc::CMSG_LEN(size as u32) as usize;
            std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(header).cast(), fds.len());
        }
    }
    let size = unsafe { libc::sendmsg(fd, &message, libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL) };
    if size < 0 {
        return Err(io::Error::last_os_error());
    }
    if size as usize != bytes.len() {
        return Err(invalid("partial control packet send"));
    }
    Ok(())
}
