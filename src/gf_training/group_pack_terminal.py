"""Readonly handoff for terminal controls staged by their actual original owner."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from gf_common.domain import ContractError
from gf_training.group_pack_resources import OwnedDescriptor, remaining
from gf_training.group_pack_source import _no_symlink_path, _stat_key


class _TerminalOriginal(OwnedDescriptor):
    """A fresh readonly capability bound to an ALREADY owned original.

    Only this composite's controlled write-close handoff may change ctime.
    This object neither admits a fresh inode nor changes Source custody rules.
    """

    def __init__(self, path: Path, expected: os.stat_result, sha256: str,
                 deadline: float, *, write_handoff: bool, database_expected: bool = False) -> None:
        super().__init__(-1)
        self.path, self.expected, self.sha256 = path, expected, sha256
        self.deadline, self.write_handoff = deadline, write_handoff
        self.database_expected = database_expected
        self.identity: os.stat_result | None = None

    def open_verified(self) -> None:
        remaining(self.deadline)
        _no_symlink_path(self.path.parent)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        actual = os.fstat(self.fd)
        expected_key, actual_key = _stat_key(self.expected), _stat_key(actual)
        if (not stat.S_ISREG(actual.st_mode) or actual_key[:5] != expected_key[:5]
                or not self.write_handoff and actual_key != expected_key):
            raise ContractError("terminal original identity changed before readonly hash")
        self.identity = actual
        self.recheck()

    def check_path_identity(self) -> None:
        assert self.identity is not None
        remaining(self.deadline)
        _no_symlink_path(self.path.parent)
        if _stat_key(self.path.lstat()) != _stat_key(self.identity):
            raise ContractError("terminal readonly original path identity changed")

    def recheck(self) -> None:
        assert self.identity is not None
        self.check_path_identity()
        if _stat_key(os.fstat(self.fd)) != _stat_key(self.identity):
            raise ContractError("terminal readonly handle identity changed")
        if self.database_expected:
            self.check_path_identity()
            return
        value = hashlib.sha256()
        offset = 0
        while offset < self.identity.st_size:
            remaining(self.deadline)
            chunk = os.pread(self.fd, min(65536, self.identity.st_size - offset), offset)
            if not chunk:
                raise ContractError("terminal readonly original shortened")
            value.update(chunk)
            offset += len(chunk)
            del chunk
        if os.pread(self.fd, 1, offset) or value.hexdigest() != self.sha256:
            raise ContractError("terminal readonly original content changed")
        if _stat_key(os.fstat(self.fd)) != _stat_key(self.identity):
            raise ContractError("terminal original changed during readonly hash")
        self.check_path_identity()
