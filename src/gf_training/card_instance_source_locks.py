"""Inventory-preserving real Dataset custody; locks supply no Source permission."""
from __future__ import annotations

import fcntl
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Self

from gf_common.domain import ContractError
from gf_common.owned_completion import _close_owned_descriptor
from gf_training.lineage import RunOwnership, run_lock


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _root(path: Path) -> Path:
    lexical = Path(path).absolute()
    for item in (lexical, *lexical.parents):
        if not stat.S_ISDIR(item.lstat().st_mode):
            raise ContractError('game original roots/ancestors must be nonsymlink directories')
    return lexical.resolve(strict=True)


class ValidationManifestOwnership:
    """One actual readonly canonical-manifest flock; never a copied token value."""
    _control_name = 'manifest.json'

    def __init__(self, directory: Path, *, deadline: float) -> None:
        self.directory = _root(directory)
        self.deadline = deadline
        self._fd: int | None = None
        self._directory_fd: int | None = None
        self._snapshot: tuple[int, ...] | None = None
        self._root_identity: tuple[int, int] | None = None
        self._entered = self._active = self._closed = False
        self._failure: BaseException | None = None
        self._uncertain_releases: list[Any] = []

    def __enter__(self) -> Self:
        if self._entered or self._closed:
            raise ContractError('manifest custody has exactly one lifetime')
        self._entered = True
        try:
            if time.monotonic() >= self.deadline:
                raise TimeoutError('manifest custody original deadline')
            self._directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(self._directory_fd)
            self._root_identity = info.st_dev, info.st_ino
            self._fd = os.open(self._control_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._directory_fd)
            before = os.fstat(self._fd)
            if not stat.S_ISREG(before.st_mode):
                raise ContractError('canonical validation manifest must be regular')
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._snapshot = _identity(before)
            self._active = True
            self.require(self.directory)
            return self
        except BaseException as error:
            self._failure = error
            try:
                self.close()
            except BaseException:  # noqa: BLE001 - preserve first failure and actual cleanup capability
                vars(error)['_game_source_lock_cleanup_owner'] = self
            raise

    def require(self, directory: Path) -> None:
        if self._failure is not None:
            raise ContractError('manifest custody permanently failed') from self._failure
        if not self._active or self._closed or self._fd is None or self._directory_fd is None:
            raise ContractError('SAME active canonical manifest custody required')
        try:
            if Path(directory).resolve(strict=True) != self.directory or time.monotonic() >= self.deadline:
                raise ContractError('manifest custody directory/deadline differs')
            root = self.directory.lstat()
            held_root = os.fstat(self._directory_fd)
            if (not stat.S_ISDIR(root.st_mode) or (root.st_dev, root.st_ino) != self._root_identity
                    or (held_root.st_dev, held_root.st_ino) != self._root_identity):
                raise ContractError('manifest custody root replaced')
            path = os.stat(self._control_name, dir_fd=self._directory_fd, follow_symlinks=False)
            held = os.fstat(self._fd)
            if (not stat.S_ISREG(path.st_mode) or _identity(path) != self._snapshot
                    or _identity(held) != self._snapshot):
                raise ContractError('original canonical manifest custody changed')
        except BaseException as error:
            self._failure = error
            raise

    @property
    def inode(self) -> tuple[int, int]:
        self.require(self.directory)
        assert self._snapshot is not None
        return self._snapshot[0], self._snapshot[1]

    @property
    def cleanup_complete(self) -> bool:
        return self._closed and self._fd is None and self._directory_fd is None and not self._uncertain_releases

    def close(self) -> None:
        if self._closed:
            if self._uncertain_releases:
                raise ContractError('manifest custody close remains unconfirmed') from self._failure
            return
        failure = self._failure
        self._active = False
        for name, identity in (('_fd', self._snapshot), ('_directory_fd', self._root_identity)):
            descriptor = getattr(self, name)
            if descriptor is None:
                continue
            setattr(self, name, None)  # Never operate on a number submitted to close again.
            try:
                _close_owned_descriptor(self, descriptor, identity, name)
            except BaseException as error:  # noqa: BLE001 - try independent handle, retain first error
                failure = failure or error
        self._closed = True
        self._failure = failure
        if failure is not None:
            if not self.cleanup_complete:
                vars(failure)['_game_source_lock_cleanup_owner'] = self
            raise failure

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        original = value if isinstance(value, BaseException) else None
        if original is not None:
            self._failure = self._failure or original
        elif self._active:
            try:
                self.require(self.directory)
            except BaseException as error:  # noqa: BLE001 - final guard preserves initiating error
                self._failure = self._failure or error
        try:
            self.close()
        except BaseException as error:
            if original is None:
                raise
            vars(original)['_game_source_lock_cleanup_owner'] = self
            if error is not original:
                try:
                    original.add_note('manifest custody close: ' + str(error))
                except BaseException:  # noqa: BLE001, S110 - diagnostics cannot replace primary
                    pass


class ValidationLocatorOwnership(ValidationManifestOwnership):
    """Actual immutable public packed locator flock, without Source authority."""
    _control_name = 'collection.locator.json'


class GameSourceCustody:
    """Complete six-logical-root ordered custody, optionally borrowed as a whole."""
    def __init__(self, parents: tuple[Path, ...], validations: tuple[Path, ...], *, deadline: float,
                 ownership: GameSourceCustody | None = None, packed: bool = False) -> None:
        if type(packed) is not bool:
            raise ContractError('explicit storage custody variant required')
        self._packed = packed
        if len(parents) != 3 or len(validations) != 3:
            raise ContractError('three parent and three validation memberships required')
        self.parents = tuple(_root(path) for path in parents)
        self.validations = tuple(_root(path) for path in validations)
        self.deadline, self._borrowed = deadline, ownership
        grouped: dict[tuple[int, int], tuple[Path, str]] = {}
        for path, role in (*((path, 'parent') for path in self.parents), *((path, 'validation') for path in self.validations)):
            info = path.lstat(); identity = info.st_dev, info.st_ino
            if identity in grouped:
                previous, previous_role = grouped[identity]
                if previous_role != role:
                    raise ContractError('parent and validation physical root cannot overlap')
                if os.fsencode(path) < os.fsencode(previous):
                    grouped[identity] = path, role
            else:
                grouped[identity] = path, role
        self._ordered = tuple(sorted(grouped.values(), key=lambda row: os.fsencode(row[0])))
        self._entries: dict[Path, tuple[Any, Any]] = {}
        self._physical: dict[tuple[int, int], Any] = {}
        self._entered = self._active = self._closed = False
        self._failure: BaseException | None = None
        self._uncertain: set[Path] = set()
        self._gate = threading.Lock()

    def __enter__(self) -> Self:
        with self._gate:
            if self._entered or self._closed:
                raise ContractError('complete game custody has one lifetime')
            self._entered = True
            if self._borrowed is not None:
                if type(self._borrowed) is not GameSourceCustody or self._borrowed._ordered != self._ordered or self._borrowed._packed != self._packed:
                    raise ContractError('only SAME complete ordered custody set can be borrowed')
                self._borrowed.require()
                self._active = True
                return self
            try:
                for path, role in self._ordered:
                    if time.monotonic() >= self.deadline:
                        raise TimeoutError('complete game custody deadline')
                    owner = (run_lock(path) if role == 'parent' else
                        (ValidationLocatorOwnership if self._packed else ValidationManifestOwnership)(path, deadline=self.deadline))
                    token = owner.__enter__()
                    self._entries[path] = owner, token
                    info = path.lstat()
                    self._physical[info.st_dev, info.st_ino] = token
                self._active = True
                return self
            except BaseException as error:
                self._failure = error
                try:
                    self.close()
                except BaseException:  # noqa: BLE001 - actual remaining custody owner stays reachable
                    vars(error)['_game_source_lock_cleanup_owner'] = self
                raise

    def require(self) -> None:
        if self._failure is not None or not self._active or self._closed:
            raise ContractError('SAME complete active game custody required') from self._failure
        if time.monotonic() >= self.deadline:
            raise TimeoutError('complete game custody original deadline')
        if self._borrowed is not None:
            self._borrowed.require()
            return
        from gf_training.lineage import _OWNED_LOCKS
        for path, role in self._ordered:
            _, token = self._entries[path]
            if role == 'validation':
                token.require(path)
            elif _OWNED_LOCKS.get(path) != token.token:
                raise ContractError('actual original parent lock no longer held')

    def ownership_for(self, path: Path) -> RunOwnership | ValidationManifestOwnership | ValidationLocatorOwnership:
        self.require()
        if self._borrowed is not None:
            return self._borrowed.ownership_for(path)
        canonical = _root(path); info = canonical.lstat()
        token = self._physical.get((info.st_dev, info.st_ino))
        if token is None:
            raise ContractError('root not in actual complete game custody')
        return token

    @property
    def cleanup_complete(self) -> bool:
        return self._closed and not self._entries and not self._uncertain

    def close(self) -> None:
        if self._closed:
            return
        self._active = False
        if self._borrowed is not None:
            self._closed = True
            return
        failure = self._failure
        for path, _role in reversed(self._ordered):
            if path in self._uncertain:
                continue
            entry = self._entries.pop(path, None)
            if entry is None:
                continue
            owner, _token = entry
            try:
                owner.__exit__(type(failure) if failure is not None else None, failure, None)
                if type(owner) is ValidationManifestOwnership and not owner.cleanup_complete:
                    raise ContractError('canonical manifest lock close remains unconfirmed')
            except BaseException as error:  # noqa: BLE001 - never retry an uncertain consumed close
                # A Source identity/guard error can be reported AFTER both
                # independent manifest descriptors retired successfully.
                if type(owner) is not ValidationManifestOwnership or not owner.cleanup_complete:
                    self._entries[path] = entry
                    self._uncertain.add(path)
                failure = failure or error
        self._closed = not self._entries
        self._failure = failure
        if failure is not None:
            if not self.cleanup_complete:
                vars(failure)['_game_source_lock_cleanup_owner'] = self
            raise failure

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        original = value if isinstance(value, BaseException) else None
        self._failure = self._failure or original
        try:
            self.close()
        except BaseException:
            if original is None:
                raise
            vars(original)['_game_source_lock_cleanup_owner'] = self
