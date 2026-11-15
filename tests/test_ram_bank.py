"""Real Linux backing, socket custody and process-lifetime checks."""
from __future__ import annotations

import array
import errno
import fcntl
import gc
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.group_pack import StorageSettings
from gf_common.ram_bank import F_GET_SEALS, F_SEAL_WRITE, RamBankBuilder, RamBankChannel
from gf_training.group_pack_resources import StorageBudget


def budget():
    return StorageBudget(StorageSettings(), time.monotonic() + 30)


def bank(account=None):
    return RamBankBuilder({"x": np.array([2, 4, 6], dtype=np.int64)}, account or budget()).seal()


def test_seal_copies_aliases_consumes_builder_and_enforces_kernel_seals():
    account = budget()
    source = np.array([2, 4, 6], dtype=np.int64)
    builder = RamBankBuilder({"x": source}, account)
    frozen = builder.seal()
    source[:] = 9
    view = frozen.array("x")
    assert view.tolist() == [2, 4, 6]
    with pytest.raises(ValueError):
        view.setflags(write=True)
    with pytest.raises(OSError):
        os.pwrite(frozen.fileno(), b"x", 0)
    with pytest.raises(OSError):
        os.ftruncate(frozen.fileno(), frozen.size + 1)
    with pytest.raises(OSError):
        os.ftruncate(frozen.fileno(), 0)
    assert fcntl.fcntl(frozen.fileno(), F_GET_SEALS) & F_SEAL_WRITE
    with pytest.raises(ContractError):
        builder.seal()
    charged = account.used
    child_view = view[1:]
    del view
    with pytest.raises(BufferError):
        frozen.close()
    assert account.used == charged
    assert child_view.tolist() == [4, 6]
    del child_view
    gc.collect()
    frozen.close()
    assert account.used == 0


@pytest.mark.parametrize("values", [
    {"x": np.array([object()], dtype=object)},
    {"x": np.array([1], dtype=">i8")},
    {"x": np.arange(12, dtype=np.int64)[::2]},
    {"x": np.ones((100,), dtype=np.float32)},
])
def test_invalid_or_overbudget_layout_fails_before_memfd_allocation(values, monkeypatch):
    account = budget()
    def forbidden(*args, **kwargs):
        pytest.fail("invalid layout reached physical allocation")
    monkeypatch.setattr(os, "memfd_create", forbidden)
    with pytest.raises((ContractError, ValueError)):
        RamBankBuilder(values, account, max_bytes=128)
    assert account.used == 0


def test_roundtrip_keeps_sender_charge_until_released_views_and_matching_ack():
    account = budget()
    frozen = bank(account)
    sender, receiver = RamBankChannel.pair()
    try:
        lease = sender.send(frozen, purpose="batch", generation=3, identity="snapshot-a")
        delivery = receiver.receive(purpose="batch", generation=3, identity="snapshot-a", budget=budget())
        view = delivery.bank.array("x")
        assert view.tolist() == [2, 4, 6]
        def failed_fence():
            raise RuntimeError("device synchronization failed")
        with pytest.raises(RuntimeError):
            delivery.complete(failed_fence)
        assert not lease.released
        with pytest.raises(BufferError):
            delivery.complete(lambda: None)
        lease.cancel()
        with pytest.raises(BufferError):
            frozen.close()
        assert account.used > 0
        del view
        gc.collect()
        delivery.complete(lambda: None)
        sender.accept_completion(lease)
        assert lease.released
        with pytest.raises(ContractError):
            sender.accept_completion(lease)
        frozen.close()
        assert account.used == 0
    finally:
        sender.close()
        receiver.close()


def test_wrong_generation_poisoned_delivery_retains_sender_until_actual_child_reap():
    frozen = bank()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child)
    try:
        lease = sender.send(frozen, purpose="model", generation=8, identity="model-a")
        with pytest.raises(ContractError):
            receiver.receive(purpose="model", generation=9, identity="model-a", budget=budget())
        lease.cancel()
        with pytest.raises(ContractError):
            lease.release_after_reap(True)
        with pytest.raises(ContractError):
            lease.release_after_reap(child)
        with pytest.raises(BufferError):
            frozen.close()
        child.terminate()
        child.wait(timeout=5)
        lease.release_after_reap(child)
        frozen.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        sender.close()
        receiver.close()


@pytest.mark.parametrize("fault", ["wrong_fd", "missing_fd", "extra_fd", "range", "replay"])
def test_external_fd_envelope_is_not_authority(fault):
    first, second = bank(), bank()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child)
    try:
        lease = sender.send(first, purpose="batch", generation=0, identity="m")
        raw, ancillary, _, _ = receiver.socket.recvmsg(65536, socket.CMSG_SPACE(8 * 4))
        imported = array.array("i")
        for _, _, data in ancillary:
            imported.frombytes(data)
        for fd in imported:
            os.close(fd)
        envelope = json.loads(raw)
        fds = [second.fileno()] if fault == "wrong_fd" else [first.fileno()]
        if fault == "missing_fd":
            fds = []
        if fault == "extra_fd":
            fds.append(second.fileno())
        if fault == "range":
            envelope["arrays"][0]["offset"] = first.size + 1
        if fault == "replay":
            envelope["transaction"] = 0
        rights = [] if not fds else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
        sender.socket.sendmsg([json.dumps(envelope).encode()], rights)
        with pytest.raises(ContractError):
            receiver.receive(purpose="batch", generation=0, identity="m", budget=budget())
        assert not lease.released
    finally:
        child.terminate()
        child.wait(timeout=5)
        for outstanding in sender.outstanding:
            outstanding.release_after_reap(child)
        sender.close()
        receiver.close()
        first.close()
        second.close()


def test_actual_exec_inherits_only_private_endpoint_and_ack_releases_before_reap():
    sender, inherited = RamBankChannel.pair()
    fd, invocation = inherited.inherited_endpoint()
    script = """
import sys, time
from gf_common.ram_bank import RamBankChannel
from gf_common.group_pack import StorageSettings
from gf_training.group_pack_resources import StorageBudget
channel = RamBankChannel._from_inherited(int(sys.argv[1]), sys.argv[2])
budget = StorageBudget(StorageSettings(), time.monotonic() + 20)
delivery = channel.receive(purpose='model', generation=4, identity='inference-only', budget=budget)
values = delivery.bank.array('x')
assert values.tolist() == [2, 4, 6]
del values
delivery.complete(lambda: None)
assert budget.used == 0
channel.close()
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(fd), invocation], pass_fds=(fd,))
    sender.bind_child(child)
    inherited.close()
    frozen = bank()
    try:
        lease = sender.send(frozen, purpose="model", generation=4, identity="inference-only")
        sender.accept_completion(lease)
        assert lease.released
        frozen.close()
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        sender.close()


def test_failed_send_and_false_completion_retain_actual_lease_until_reap():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child)
    frozen = bank()
    receiver.close()
    try:
        with pytest.raises(OSError):
            sender.send(frozen, purpose="save", generation=1, identity="resume-state")
        assert len(sender.outstanding) == 1 and sender.outstanding[0].cancelled
        with pytest.raises(BufferError):
            frozen.close()
        child.terminate()
        child.wait(timeout=5)
        sender.outstanding[0].release_after_reap(child)
        frozen.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        sender.close()


@pytest.mark.parametrize("wrong", ["generation", "purpose", "identity", "invocation", "boolean_transaction"])
def test_wrong_completion_cannot_release_original_bank(wrong):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child)
    frozen = bank()
    try:
        lease = sender.send(frozen, purpose="batch", generation=3, identity="snapshot-a")
        delivery = receiver.receive(purpose="batch", generation=3, identity="snapshot-a", budget=budget())
        if wrong == "boolean_transaction":
            delivery.complete(lambda: None)
            raw, _, _, _ = sender.socket.recvmsg(65536)
            message = json.loads(raw)
            message["transaction"] = True
        else:
            delivery.bank.close()
            message = {"kind": "complete", "invocation": sender.invocation,
                "transaction": 1, "envelope_sha256": "wrong", wrong: "wrong"}
        receiver.socket.sendmsg([json.dumps(message).encode()])
        with pytest.raises(ContractError):
            sender.accept_completion(lease)
        assert not lease.released
        child.terminate()
        child.wait(timeout=5)
        lease.release_after_reap(child)
        frozen.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        sender.close()
        receiver.close()


@pytest.mark.parametrize("builder", [False, True])
def test_uncertain_close_quarantines_charge_and_never_retries_reused_fd(monkeypatch, builder):
    account = budget()
    owner = RamBankBuilder({"x": np.array([1], dtype=np.int64)}, account)
    if not builder:
        owner = owner.seal()
    original_fd = owner._fd
    original_close = os.close
    def uncertain(fd):
        original_close(fd)
        if fd == original_fd:
            raise OSError(errno.EIO, "close failed after possible kernel release")
    monkeypatch.setattr(os, "close", uncertain)
    with pytest.raises(RuntimeError) as raised:
        owner.close()
    assert raised.value.cleanup_owner is owner
    assert account.used > 0
    reused = os.open("/dev/null", os.O_RDONLY)
    try:
        with pytest.raises(RuntimeError):
            owner.close()
        os.fstat(reused)
        assert account.used > 0
    finally:
        original_close(reused)


def test_local_oversized_envelope_refusal_preserves_next_actual_transaction():
    oversized = RamBankBuilder({"界" * 125 + f"{index:03}": np.array([index], dtype=np.int64)
        for index in range(256)}, budget()).seal()
    frozen = bank()
    sender, receiver = RamBankChannel.pair()
    try:
        with pytest.raises(ContractError):
            sender.send(oversized, purpose="batch", generation=3, identity="m")
        assert not sender.outstanding
        lease = sender.send(frozen, purpose="batch", generation=3, identity="m")
        delivery = receiver.receive(purpose="batch", generation=3, identity="m", budget=budget())
        delivery.complete(lambda: None)
        sender.accept_completion(lease)
        frozen.close()
    finally:
        oversized.close()
        sender.close()
        receiver.close()


def test_completion_send_uncertainty_is_terminal_without_duplicate_ack(monkeypatch):
    account = budget()
    frozen = bank(account)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child, timeout=0.01)
    original_send = RamBankChannel._send
    def uncertain(self, message, fd=None):
        original_send(self, message, fd)
        if self is receiver:
            raise OSError(errno.EIO, "ACK sent but completion delivery is uncertain")
    monkeypatch.setattr(RamBankChannel, "_send", uncertain)
    try:
        lease = sender.send(frozen, purpose="batch", generation=3, identity="m")
        delivery = receiver.receive(purpose="batch", generation=3, identity="m", budget=budget())
        with pytest.raises(OSError):
            delivery.complete(lambda: None)
        with pytest.raises(ContractError):
            delivery.complete(lambda: None)
        # Drain the actual ACK to model lost completion handling. A second
        # completion call must not emit another message onto this real socket.
        raw, _, _, _ = sender.socket.recvmsg(65536)
        assert json.loads(raw)["kind"] == "complete"
        sender.socket.setblocking(False)
        with pytest.raises(BlockingIOError):
            sender.socket.recvmsg(65536)
        assert not lease.released and account.used > 0
        with pytest.raises(ContractError):
            receiver.send(frozen, purpose="save", generation=3, identity="m")
        with pytest.raises(ContractError):
            receiver.receive(purpose="batch", generation=3, identity="m", budget=budget())
    finally:
        child.terminate()
        child.wait(timeout=5)
        for outstanding in sender.outstanding:
            outstanding.release_after_reap(child)
        frozen.close()
        assert account.used == 0
        sender.close()
        receiver.close()


@pytest.mark.parametrize("fault", ["timeout", "malformed_json", "wrong_ack"])
def test_completion_receive_uncertainty_poisoned_without_retry_or_further_traffic(fault):
    account = budget()
    frozen = bank(account)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child, timeout=0.01)
    try:
        lease = sender.send(frozen, purpose="batch", generation=3, identity="m")
        delivery = receiver.receive(purpose="batch", generation=3, identity="m", budget=budget())
        if fault != "timeout":
            receiver.socket.sendmsg([b"{" if fault == "malformed_json" else b'{"kind":"complete"}'])
        with pytest.raises((TimeoutError, ValueError)):
            sender.accept_completion(lease)
        # A late actual ACK cannot restart or repair the failed transaction.
        delivery.complete(lambda: None)
        with pytest.raises(ContractError):
            sender.accept_completion(lease)
        with pytest.raises(ContractError):
            sender.send(frozen, purpose="batch", generation=4, identity="m")
        with pytest.raises(ContractError):
            sender.receive(purpose="batch", generation=3, identity="m", budget=budget())
        assert not lease.released and account.used > 0
        with pytest.raises(BufferError):
            frozen.close()
        receiver.socket.setblocking(False)
        with pytest.raises(BlockingIOError):
            receiver.socket.recvmsg(65536)
    finally:
        child.terminate()
        child.wait(timeout=5)
        for outstanding in sender.outstanding:
            outstanding.release_after_reap(child)
        frozen.close()
        assert account.used == 0
        sender.close()
        receiver.close()


def test_duplicate_old_ack_poisoned_next_live_transaction_until_actual_reap():
    account = budget()
    frozen = bank(account)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sender, receiver = RamBankChannel.pair(child=child, timeout=0.01)
    try:
        first = sender.send(frozen, purpose="model", generation=3, identity="m")
        delivery = receiver.receive(purpose="model", generation=3, identity="m", budget=budget())
        delivery.complete(lambda: None)
        raw, _, _, _ = sender.socket.recvmsg(65536)
        receiver.socket.sendmsg([raw])
        receiver.socket.sendmsg([raw])
        sender.accept_completion(first)
        assert first.released
        second = sender.send(frozen, purpose="model", generation=4, identity="m")
        delivery = receiver.receive(purpose="model", generation=4, identity="m", budget=budget())
        with pytest.raises(ContractError):
            sender.accept_completion(second)
        delivery.complete(lambda: None)
        with pytest.raises(ContractError):
            sender.accept_completion(second)
        assert not second.released and account.used > 0
    finally:
        child.terminate()
        child.wait(timeout=5)
        for outstanding in sender.outstanding:
            outstanding.release_after_reap(child)
        frozen.close()
        assert account.used == 0
        sender.close()
        receiver.close()
