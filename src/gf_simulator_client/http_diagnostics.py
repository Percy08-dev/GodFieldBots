"""Own the private Simulator HTTP capture for one managed seat-swap pair."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

from gf_common.domain import ContractError, Json

if TYPE_CHECKING:
    from gf_common.hpo_storage import HPODiagnosticsConfig

_BLOCK_BYTES = 1024 * 1024
# The pinned logger retains at most 64 KiB of each body before JSON escaping.
# Its local HTTP records also include headers and route metadata.
_MAX_RECORD_BYTES = _BLOCK_BYTES


@dataclass
class HttpFacts:
    """Client-side floors and sticky transport faults, outside game data."""

    responses: int = 0
    error_responses: int = 0
    transport_errors: int = 0

    def observe_response(self, status: int) -> None:
        self.responses += 1
        self.error_responses += 400 <= status <= 599


def _check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("original pair cleanup deadline exhausted")


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _part(value: str) -> str:
    if not value or "\0" in value or value in (".", "..") or Path(value).name != value:
        raise ContractError("unsafe HTTP pair storage component")
    return value


def _scope(root: Path, operation: str, collection: str) -> Path:
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ContractError("symlink in HTTP diagnostic root ancestry")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    operation_root = root / operation
    directory = operation_root / collection
    for parent in (operation_root, directory):
        parent.mkdir(mode=0o700, exist_ok=True)
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077):
            raise ContractError("HTTP diagnostic scope must be an owned private directory")
    return directory


def _scan(source: BinaryIO, *, deadline: float, pending_limit: int) -> tuple[int, int]:
    pending: dict[str, tuple[str | None, int | None]] = {}
    responses = errors = 0
    while line := source.readline(_MAX_RECORD_BYTES + 1):
        _check(deadline)
        if len(line) > _MAX_RECORD_BYTES or not line.endswith(b"\n"):
            raise ValueError("overlong or incomplete HTTP JSONL record")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError("invalid HTTP JSONL object")
        request_id = row.get("request_id")
        game, player = row["game_id"], row["player_id"]
        if (type(request_id) is not str or not request_id or
                game is not None and type(game) is not str or
                player is not None and type(player) is not int):
            raise ValueError("invalid HTTP JSONL correlation fields")
        if row.get("kind") == "request":
            if (request_id in pending or type(row.get("method")) is not str
                    or not row["method"] or type(row.get("path")) is not str
                    or not row["path"].startswith("/") or len(pending) >= pending_limit):
                raise ValueError("invalid request or missing HTTP JSONL responses")
            pending[request_id] = (game, player)
        elif row.get("kind") == "response":
            status, duration = row.get("status"), row.get("duration_ms")
            if (pending.pop(request_id, None) != (game, player)
                    or type(status) is not int or not 100 <= status <= 599
                    or type(duration) is not int or duration < 0):
                raise ValueError("unmatched or invalid HTTP JSONL response")
            responses += 1
            errors += 400 <= status <= 599
        else:
            raise ValueError("invalid HTTP JSONL record kind")
    if pending:
        raise ValueError("missing HTTP JSONL responses")
    return responses, errors


class HttpPairCapture:
    def __init__(self, ram_root: Path, archive_root: Path, *,
                 operation_id: str, collection_id: str, pair_id: str,
                 retention: Literal["errors", "all"], compression: Literal["zstd", "none"],
                 episode_count: int = 2) -> None:
        self.ram_root, self.archive_root = ram_root, archive_root
        self.operation_id, self.collection_id, self.pair_id = map(
            _part, (operation_id, collection_id, pair_id))
        self.retention, self.compression = retention, compression
        self.episode_count = episode_count
        self.directory: Path | None = None
        self._report: Json | None = None

    @classmethod
    def configured(cls, config: HPODiagnosticsConfig, *, operation_id: str,
                   collection_id: str, pair_id: str, episode_count: int) -> HttpPairCapture:
        return cls(Path(config.http_log_ram_root), Path(config.http_log_archive_root),
                   operation_id=operation_id, collection_id=collection_id, pair_id=pair_id,
                   retention=config.http_log_retention, compression=config.http_log_compression,
                   episode_count=episode_count)

    def prepare(self) -> Path:
        parent = _scope(self.ram_root, self.operation_id, self.collection_id)
        self.directory = Path(tempfile.mkdtemp(prefix=self.pair_id + "-", dir=parent))
        # The Simulator opens the existing file in append mode.
        fd = os.open(self.directory / "http.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        return self.directory

    def _archive(self, source: BinaryIO, deadline: float) -> tuple[Path, int]:
        assert self.directory is not None
        parent = _scope(self.archive_root, self.operation_id, self.collection_id)
        directory = parent / self.directory.name
        directory.mkdir(mode=0o700, exist_ok=False)
        path = directory / ("http.jsonl.zst" if self.compression == "zstd" else "http.jsonl")
        fd, name = tempfile.mkstemp(dir=directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as output:
                if self.compression == "zstd":
                    from gf_common.hpo_storage import require_zstd

                    zstd = require_zstd()
                    try:
                        with zstd.ZstdFile(output, "wb", level=3) as encoded:
                            self._copy(source, encoded.write, deadline)
                    except zstd.ZstdError as error:
                        raise ValueError("HTTP zstd archive failed: " + str(error)) from error
                else:
                    self._copy(source, output.write, deadline)
                _check(deadline)
                output.flush()
                size = output.tell()
                os.fsync(output.fileno())
            _check(deadline)
            temporary.replace(path)
            # Another pair may have created an as-yet-unsynced ancestor. Keep
            # the original until the full entry chain on this mount is durable.
            device = directory.stat().st_dev
            for ancestor in (directory, *directory.parents):
                _check(deadline)
                if ancestor.lstat().st_dev != device:
                    break
                _sync_dir(ancestor)
            return path, size
        except BaseException as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup:
                error.add_note("HTTP archive temporary cleanup failed: " + str(cleanup))
            raise

    @staticmethod
    def _copy(source: BinaryIO, write: Callable[[bytes], object], deadline: float) -> None:
        while chunk := source.read(_BLOCK_BYTES):
            _check(deadline)
            write(chunk)

    def _failure(self, report: Json) -> Json:
        if self.directory is not None:
            try:
                target = self.directory / "diagnostic.json"
                with tempfile.NamedTemporaryFile(mode="wb", dir=self.directory, delete=False) as output:
                    temporary = Path(output.name)
                    try:
                        output.write(json.dumps(report, allow_nan=False).encode() + b"\n")
                        output.flush()
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
            except OSError as error:
                report["metadata_error"] = str(error)
        return report

    def settle(self, *, stopped: bool, deadline: float, abnormal: bool = False,
               responses: int = 0, error_responses: int = 0, transport_errors: int = 0) -> Json:
        if self._report is not None:
            return self._report
        start = time.monotonic()
        if self.directory is None:
            return {"status": "capture_failed", "error": "HTTP pair staging was not created",
                    "staging_root": str(self.ram_root)}
        path = self.directory / "http.jsonl"
        report: Json = {"status": "stop_unconfirmed", "residual": str(path),
                        "observed_responses": responses, "observed_error_responses": error_responses,
                        "transport_errors": transport_errors, "abnormal": abnormal}
        if not stopped:
            return self._failure(report)
        try:
            report["status"] = "capture_failed"
            _check(deadline)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("HTTP capture is not a regular file")
                report["raw_bytes"] = info.st_size
                # Each actor issues one request at a time. A prior failed transport
                # can leave the server finishing that request concurrently.
                errors = 0
                try:
                    recorded, errors = _scan(source, deadline=deadline,
                        pending_limit=2 * self.episode_count + transport_errors)
                    report["logged_responses"], report["logged_error_responses"] = recorded, errors
                    floor = max(responses, int(not abnormal and not transport_errors))
                    if recorded < floor or errors < error_responses:
                        raise ValueError("HTTP JSONL is below the client response floor")
                except (ValueError, TypeError, KeyError, RecursionError) as error:
                    # The stopped writer's original bytes can still explain an
                    # incomplete response, logger fault or interrupted line.
                    report["capture_error"] = str(error)
                report["scan_seconds"] = time.monotonic() - start
                if self.retention == "all" or abnormal or transport_errors or errors or "capture_error" in report:
                    report["status"] = "archive_failed"
                    source.seek(0)
                    archive_started = time.monotonic()
                    archive, size = self._archive(source, deadline)
                    report.update(archive=str(archive), archive_bytes=size,
                                  archive_seconds=time.monotonic() - archive_started)
            report["status"] = "cleanup_failed"
            _check(deadline)
            (self.directory / "diagnostic.json").unlink(missing_ok=True)
            path.unlink()
            report["residual"] = str(self.directory)
            self.directory.rmdir()
            report.pop("residual")
            report["status"] = "archived" if "archive" in report else "discarded"
        except (OSError, ValueError) as error:
            report["error"] = str(error)
        report["settle_seconds"] = time.monotonic() - start
        self._report = self._failure(report) if "error" in report else report
        return self._report
