"""Linux sealed array banks and invocation-local, single-use FD transfers.

No path imports, Torch dependency or persistence authority. An uncertain transfer
deliberately retains its sender's physical reservation until ACK or owned reap.
"""
from __future__ import annotations

import array
import asyncio.subprocess
import fcntl
import hashlib
import json
import mmap
import os
import socket
import subprocess
import sys
import uuid
import weakref
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gf_common.artifacts import ArtifactAdmission, ArtifactReservation
from gf_common.domain import ContractError, canonical, closed, integer

MAX_BANK_BYTES = 512 * 1024**2
MAX_ENVELOPE_BYTES = 65536
MAX_ARRAYS = 256
_DTYPES = frozenset(("|u1", "|i1", "|b1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8"))
# Linux UAPI values (some CPython builds omit these optional fcntl names).
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 8)
_SEALS = F_SEAL_WRITE | 4 | 2 | 1
_TOKEN = object()
type Array = NDArray[Any]
type Child = subprocess.Popen[Any] | asyncio.subprocess.Process


class RamBankCleanupError(RuntimeError):
    """Terminal close uncertainty; the retained owner must be quarantined."""

    def __init__(self, message: str, owner: object) -> None:
        super().__init__(message)
        self.cleanup_owner = owner


class _ReceivedFdCleanup:
    def __init__(self, fds: list[int], reservation: ArtifactReservation | None = None) -> None:
        self._fds, self.reservation = tuple(fds), reservation
        self._attempted = False

    def close(self) -> None:
        if self._attempted:
            raise RamBankCleanupError("received FD close was already attempted", self)
        self._attempted = True
        failures = []
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError as error:
                failures.append(error)
        if failures:
            raise RamBankCleanupError("received FD close is uncertain", self) from failures[0]
        if self.reservation is not None:
            self.reservation.close()
            self.reservation = None


@dataclass(frozen=True)
class ArrayDescriptor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def _layout(raw: Any, size: int, limit: int) -> tuple[ArrayDescriptor, ...]:
    integer(size, 1, limit)
    if type(raw) not in (tuple, list) or not 1 <= len(raw) <= MAX_ARRAYS:
        raise ContractError("bounded nonempty array layout required")
    result = []
    end = 0
    names: set[str] = set()
    for item in raw:
        closed(item, {"name", "dtype", "shape", "offset", "nbytes"})
        name, dtype, shape = item["name"], item["dtype"], item["shape"]
        if (type(name) is not str or not name or len(name) > 128 or name in names
                or type(dtype) is not str or dtype not in _DTYPES
                or type(shape) not in (list, tuple) or len(shape) > 8):
            raise ContractError("invalid closed bank dtype/name/shape")
        count = 1
        for dimension in shape:
            integer(dimension, 0, limit)
            count *= dimension
            if count > limit:
                raise ContractError("array shape exceeds bank bound")
        nbytes = count * np.dtype(dtype).itemsize
        offset = (end + 63) // 64 * 64
        if (type(item["offset"]) is not int or item["offset"] != offset
                or type(item["nbytes"]) is not int or item["nbytes"] != nbytes
                or offset + nbytes > size):
            raise ContractError("bank layout range/overlap mismatch")
        result.append(ArrayDescriptor(name, dtype, tuple(shape), offset, nbytes))
        names.add(name)
        end = offset + nbytes
    if size != max(1, end):
        raise ContractError("bank has unaccounted bytes")
    return tuple(result)


class RamBankBuilder:
    """One admitted physical allocation, copied once; no writable views escape."""

    def __init__(self, arrays: Mapping[str, Array], budget: ArtifactAdmission, *,
                 max_bytes: int = MAX_BANK_BYTES) -> None:
        if sys.platform != "linux":
            raise ContractError("owned RAM banks require Linux memfd sealing")
        integer(max_bytes, 1, MAX_BANK_BYTES)
        if not isinstance(arrays, Mapping) or not 1 <= len(arrays) <= MAX_ARRAYS:
            raise ContractError("bounded nonempty bank required")
        descriptors = []
        end = 0
        for name, values in arrays.items():
            if type(values) is not np.ndarray or not values.flags.c_contiguous:
                raise ContractError("bank source must be a contiguous exact ndarray")
            offset = (end + 63) // 64 * 64
            descriptors.append({"name": name, "dtype": values.dtype.str,
                "shape": values.shape, "offset": offset, "nbytes": values.nbytes})
            end = offset + values.nbytes
        size = max(1, end)
        self.descriptors = _layout(descriptors, size, max_bytes)
        self.size, self._fd = size, -1
        self._close_uncertain = False
        self._reservation: ArtifactReservation | None = budget.reserve(
            ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
            + MAX_ENVELOPE_BYTES + 1024 * len(self.descriptors))
        self._consumed = False
        try:
            self._fd = os.memfd_create("gf-owned-bank", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            os.ftruncate(self._fd, size)
            os.posix_fallocate(self._fd, 0, size)
            with mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE) as writable:
                for descriptor in self.descriptors:
                    target = np.ndarray(descriptor.shape, dtype=descriptor.dtype,
                        buffer=writable, offset=descriptor.offset)
                    try:
                        np.copyto(target, arrays[descriptor.name], casting="no")
                    finally:
                        del target
        except BaseException:
            self.close()
            raise

    def seal(self) -> ReadonlyRamBank:
        if self._consumed:
            raise ContractError("RAM builder already consumed")
        try:
            fcntl.fcntl(self._fd, F_ADD_SEALS, _SEALS)
            assert self._reservation is not None
            bank = ReadonlyRamBank(_TOKEN, self._fd, self.size, self.descriptors, self._reservation)
        except BaseException:
            self.close()
            raise
        self._consumed, self._fd, self._reservation = True, -1, None
        return bank

    def close(self) -> None:
        if self._close_uncertain:
            raise RamBankCleanupError("builder close is quarantined", self)
        if self._fd >= 0:
            fd = self._fd
            self._fd = -1
            self._consumed = True
            try:
                os.close(fd)
            except OSError as error:
                self._close_uncertain = True
                raise RamBankCleanupError("builder FD close is uncertain", self) from error
        if self._reservation is not None:
            self._reservation.close()
            self._reservation = None
        self._consumed = True


class ReadonlyRamBank:
    def __init__(self, token: object, fd: int, size: int,
                 descriptors: tuple[ArrayDescriptor, ...],
                 reservation: ArtifactReservation | None) -> None:
        if token is not _TOKEN:
            raise ContractError("RAM owner must originate at builder or FD receiver")
        if fcntl.fcntl(fd, F_GET_SEALS) & _SEALS != _SEALS:
            raise ContractError("RAM bank lacks immutable kernel seals")
        self._fd, self.size, self.descriptors = fd, size, descriptors
        self._reservation = reservation
        self._mapping = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        self._views: list[weakref.ReferenceType[Array]] = []
        self._pins = 0
        self._close_uncertain = False

    def fileno(self) -> int:
        if self._fd < 0:
            raise ContractError("RAM bank is closed")
        return self._fd

    def array(self, name: str) -> Array:
        self.fileno()
        descriptor = next((d for d in self.descriptors if d.name == name), None)
        if descriptor is None:
            raise ContractError("unknown bank array")
        # frombuffer's exported memoryview also protects the mmap from close when
        # a consumer retains a base/derived view rather than this exact ndarray.
        values = np.frombuffer(self._mapping, dtype=descriptor.dtype,
            count=descriptor.nbytes // np.dtype(descriptor.dtype).itemsize,
            offset=descriptor.offset).reshape(descriptor.shape)
        self._views = [ref for ref in self._views if ref() is not None]
        self._views.append(weakref.ref(values))
        return values

    def close(self) -> None:
        if self._close_uncertain:
            raise RamBankCleanupError("RAM bank close is quarantined", self)
        if self._fd < 0:
            return
        if self._pins or any(ref() is not None for ref in self._views):
            raise BufferError("RAM bank has surviving views or transfer leases")
        # mmap also checks exports retained through ndarray.base/memoryview.
        try:
            self._mapping.close()
        except OSError as error:
            self._close_uncertain = True
            raise RamBankCleanupError("RAM mapping close is uncertain", self) from error
        fd = self._fd
        self._fd = -1
        try:
            os.close(fd)
        except OSError as error:
            self._close_uncertain = True
            raise RamBankCleanupError("RAM bank FD close is uncertain", self) from error
        if self._reservation is not None:
            self._reservation.close()
            self._reservation = None


class TransferLease:
    """Cancellation records uncertainty; it cannot refund physical ownership."""

    def __init__(self, channel: RamBankChannel, bank: ReadonlyRamBank, envelope: dict[str, Any]) -> None:
        self._channel, self._bank, self._envelope = channel, bank, envelope
        self.released, self.cancelled = False, False
        bank._pins += 1

    def cancel(self) -> None:
        self.cancelled = True

    def release_after_reap(self, process: object) -> None:
        if (process is not self._channel._child
                or not isinstance(process, (subprocess.Popen, asyncio.subprocess.Process))):
            raise ContractError("transfer requires the actual owned child process")
        code = process.poll() if isinstance(process, subprocess.Popen) else process.returncode
        if code is None:
            raise ContractError("transfer child is not reaped")
        self._release()

    def _release(self) -> None:
        if self.released:
            raise ContractError("transfer completion is single use")
        self.released = True
        self._bank._pins -= 1
        del self._channel._pending[self._envelope["transaction"]]


class ReceivedBank:
    def __init__(self, channel: RamBankChannel, bank: ReadonlyRamBank, envelope: dict[str, Any]) -> None:
        self._channel, self.bank, self._envelope = channel, bank, envelope
        self._completed = False

    def complete(self, synchronize: Callable[[], None]) -> None:
        if self._completed or self._channel._poisoned:
            raise ContractError("bank completion is single use")
        synchronize()  # actual learner/actor owner supplies its device fence.
        self.bank.close()  # refusal retains custody; no ACK while views survive.
        message = {"kind": "complete", "invocation": self._channel.invocation,
            "transaction": self._envelope["transaction"],
            "envelope_sha256": hashlib.sha256(canonical(self._envelope).encode()).hexdigest()}
        # From the actual send attempt onward delivery can be uncertain. Never
        # emit a second ACK, even when the first syscall's outcome is unknown.
        self._completed = True
        try:
            self._channel._send(message)
        except BaseException:
            self._channel._poisoned = True
            raise


class RamBankChannel:
    """A private socketpair endpoint, not a loader for user-supplied paths."""

    def __init__(self, token: object, endpoint: socket.socket, invocation: str,
                 child: Child | None = None) -> None:
        if token is not _TOKEN:
            raise ContractError("RAM channels require a private socketpair")
        self.socket, self.invocation, self._child = endpoint, invocation, child
        self._sent, self._received = 0, 0
        self._pending: dict[int, TransferLease] = {}
        self._poisoned = False
        self._closed, self._close_uncertain = False, False

    @classmethod
    def pair(cls, *, child: Child | None = None, timeout: float = 5) -> tuple[RamBankChannel, RamBankChannel]:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        left.settimeout(timeout)
        right.settimeout(timeout)
        invocation = str(uuid.uuid4())
        return cls(_TOKEN, left, invocation, child), cls(_TOKEN, right, invocation)

    def inherited_endpoint(self) -> tuple[int, str]:
        """Pass this exact FD in pass_fds; close the parent's copy after spawn."""
        if self._sent or self._received or self._pending:
            raise ContractError("only an unused private endpoint may be inherited")
        return self.socket.fileno(), self.invocation

    @classmethod
    def _from_inherited(cls, fd: int, invocation: str) -> RamBankChannel:
        """Child bootstrap only: adopt its explicitly inherited socket, no path."""
        integer(fd, 0, 2**31 - 1)
        if str(uuid.UUID(invocation)) != invocation:
            raise ContractError("invalid inherited channel invocation")
        endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET, fileno=fd)
        try:
            if (endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_DOMAIN) != socket.AF_UNIX
                    or endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET):
                raise ContractError("inherited channel must be the private Unix seqpacket endpoint")
            endpoint.set_inheritable(False)
            endpoint.settimeout(5)
            return cls(_TOKEN, endpoint, invocation)
        except BaseException:
            try:
                endpoint.close()
            except OSError as error:
                raise RamBankCleanupError("inherited endpoint close is uncertain", endpoint) from error
            raise

    def bind_child(self, process: Child) -> None:
        """Bind the exact returned owned subprocess before any transfer starts."""
        if (self._child is not None or self._sent or self._received or self._pending
                or not isinstance(process, (subprocess.Popen, asyncio.subprocess.Process))):
            raise ContractError("channel child binding is original and single use")
        integer(process.pid, 1)
        self._child = process

    def _send(self, message: dict[str, Any], fd: int | None = None) -> None:
        if self._poisoned:
            raise ContractError("RAM channel is poisoned")
        raw = canonical(message).encode()
        if len(raw) > MAX_ENVELOPE_BYTES:
            raise ContractError("RAM envelope exceeds admission")
        ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
        if self.socket.sendmsg([raw], ancillary) != len(raw):
            raise ContractError("uncertain RAM envelope delivery")

    def send(self, bank: ReadonlyRamBank, *, purpose: str, generation: int, identity: str) -> TransferLease:
        if self._poisoned:
            raise ContractError("RAM channel is poisoned")
        integer(generation)
        if purpose not in ("batch", "model", "save") or type(identity) is not str or not identity or len(identity) > 512:
            raise ContractError("invalid RAM transfer identity/purpose")
        stat = os.fstat(bank.fileno())
        transaction = self._sent + 1
        envelope = {"kind": "bank", "invocation": self.invocation, "transaction": transaction,
            "purpose": purpose, "generation": generation, "identity": identity,
            "backing": [stat.st_dev, stat.st_ino], "size": bank.size,
            "arrays": [asdict(descriptor) for descriptor in bank.descriptors]}
        # Validate/size the envelope before acquiring custody. After this point
        # even failed/unknown send retains the lease in this channel's ledger.
        if len(canonical(envelope).encode()) > MAX_ENVELOPE_BYTES:
            raise ContractError("RAM envelope exceeds admission")
        self._sent = transaction
        lease = TransferLease(self, bank, envelope)
        self._pending[self._sent] = lease
        try:
            self._send(envelope, bank.fileno())
        except BaseException:
            self._poisoned = True
            lease.cancel()
            raise
        return lease

    @property
    def outstanding(self) -> tuple[TransferLease, ...]:
        return tuple(self._pending.values())

    def _receive(self) -> tuple[dict[str, Any], list[int]]:
        raw, ancillary, flags, _ = self.socket.recvmsg(MAX_ENVELOPE_BYTES,
            socket.CMSG_SPACE(4 * 8), socket.MSG_CMSG_CLOEXEC)
        fds: list[int] = []
        try:
            for level, kind, data in ancillary:
                if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                    raise ContractError("unexpected RAM ancillary message")
                values = array.array("i")
                values.frombytes(data[:len(data) // values.itemsize * values.itemsize])
                fds.extend(values)
            if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or not raw:
                raise ContractError("truncated RAM envelope")
            value = json.loads(raw)
            if type(value) is not dict:
                raise ContractError("invalid RAM envelope")
            return value, fds
        except BaseException:
            _ReceivedFdCleanup(fds).close()
            raise

    def receive(self, *, purpose: str, generation: int, identity: str, budget: ArtifactAdmission,
                max_bytes: int = MAX_BANK_BYTES) -> ReceivedBank:
        if self._poisoned:
            raise ContractError("RAM channel is poisoned")
        integer(max_bytes, 1, MAX_BANK_BYTES)
        # Receiver charges descriptors/decoded envelope/mapping ownership, while
        # the original physical pages stay charged once to the sender's bank.
        reservation: ArtifactReservation | None = budget.reserve(
            64 * MAX_ENVELOPE_BYTES + 1024 * MAX_ARRAYS)
        fds: list[int] = []
        try:
            envelope, fds = self._receive()
            closed(envelope, {"kind", "invocation", "transaction", "purpose", "generation", "identity", "backing", "size", "arrays"})
            if (len(fds) != 1 or envelope["kind"] != "bank"
                    or envelope["invocation"] != self.invocation
                    or type(envelope["transaction"]) is not int or envelope["transaction"] != self._received + 1
                    or type(envelope["generation"]) is not int
                    or (envelope["purpose"], envelope["generation"], envelope["identity"]) != (purpose, generation, identity)):
                raise ContractError("RAM transaction/ownership mismatch")
            descriptors = _layout(envelope["arrays"], envelope["size"], max_bytes)
            stat = os.fstat(fds[0])
            if envelope["backing"] != [stat.st_dev, stat.st_ino] or stat.st_size != envelope["size"]:
                raise ContractError("RAM original backing mismatch")
            bank = ReadonlyRamBank(_TOKEN, fds[0], stat.st_size, descriptors, reservation)
            reservation = None
            fds.clear()
            self._received += 1
            return ReceivedBank(self, bank, envelope)
        except RamBankCleanupError as error:
            if isinstance(error.cleanup_owner, _ReceivedFdCleanup):
                error.cleanup_owner.reservation = reservation
                reservation = None
            self._poisoned = True
            raise
        except BaseException:
            self._poisoned = True
            raise
        finally:
            if fds:
                cleanup = _ReceivedFdCleanup(fds, reservation)
                reservation = None
                cleanup.close()
            if reservation is not None:
                reservation.close()

    def accept_completion(self, lease: TransferLease) -> None:
        if self._poisoned:
            raise ContractError("RAM channel is poisoned")
        if lease._channel is not self or lease.released or self._pending.get(lease._envelope["transaction"]) is not lease:
            raise ContractError("completion lacks its original live transfer")
        try:
            message, fds = self._receive()
            try:
                expected = {"kind": "complete", "invocation": self.invocation,
                    "transaction": lease._envelope["transaction"],
                    "envelope_sha256": hashlib.sha256(canonical(lease._envelope).encode()).hexdigest()}
                if fds or type(message.get("transaction")) is not int or message != expected:
                    raise ContractError("RAM completion does not match actual transaction")
            finally:
                _ReceivedFdCleanup(fds).close()
            lease._release()
        except BaseException:
            self._poisoned = True
            raise

    def close(self) -> None:
        # Closing IPC never proves that a receiver released the backing.
        if self._close_uncertain:
            raise RamBankCleanupError("RAM channel close is quarantined", self)
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close()
        except OSError as error:
            self._close_uncertain = True
            raise RamBankCleanupError("RAM channel close is uncertain", self) from error
