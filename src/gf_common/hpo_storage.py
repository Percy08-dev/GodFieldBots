"""Shared validation for the optional sealed HPO storage settings."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from gf_common.domain import ContractError, closed


@dataclass(frozen=True)
class HPODiagnosticsConfig:
    http_log_ram_root: str
    http_log_archive_root: str
    http_log_retention: Literal['errors', 'all'] = 'errors'
    http_log_compression: Literal['zstd', 'none'] = 'zstd'
    update_compression: Literal['zstd', 'none'] = 'zstd'


HPO_STORAGE_EXTENSIONS = {'campaign_dir', 'hpo_diagnostics'}
GF_DATA_ROOT = Path('/mnt/wsl/gf_data')
WINDOWS_D_ROOT = Path('/mnt/d')


def require_zstd() -> ModuleType:
    """Select the optional Python codec only when its format is requested."""
    try:
        from compression import zstd
    except ImportError as error:
        raise ContractError('HPO zstd storage requires Python compression.zstd') from error
    return zstd


def _absolute_path(value: Any) -> Path:
    if type(value) is not str or '\x00' in value or value.startswith('//'):
        raise ContractError('absolute HPO storage path required')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ContractError('absolute normalized HPO storage path required')
    return path


def hpo_diagnostics(storage: Mapping[str, Any]) -> HPODiagnosticsConfig | None:
    if 'hpo_diagnostics' not in storage:
        return None
    raw = closed(storage['hpo_diagnostics'], set(HPODiagnosticsConfig.__dataclass_fields__))
    ram = _absolute_path(raw['http_log_ram_root'])
    archive = _absolute_path(raw['http_log_archive_root'])
    if ram.is_relative_to(archive) or archive.is_relative_to(ram):
        raise ContractError('HPO HTTP staging and archive roots must be separate')
    if raw['http_log_retention'] not in ('all', 'errors'):
        raise ContractError('HPO http_log_retention must be errors or all')
    for key in ('http_log_compression', 'update_compression'):
        if raw[key] not in ('zstd', 'none'):
            raise ContractError(f'HPO {key} must be zstd or none')
    return HPODiagnosticsConfig(**raw)


def check_storage_extensions(storage: Mapping[str, Any]) -> None:
    if 'campaign_dir' in storage:
        campaign = _absolute_path(storage['campaign_dir'])
        parent = _absolute_path(storage['parent_root'])
        if campaign == parent or not campaign.is_relative_to(parent):
            raise ContractError('HPO campaign_dir must be beneath parent_root')
    hpo_diagnostics(storage)


def safe_storage_path(path: Path) -> None:
    if not path.is_absolute() or '..' in path.parts:
        raise ContractError('absolute HPO owned storage path required')
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ContractError('symlink in HPO owned storage ancestry')


@dataclass(frozen=True)
class Mount:
    mount_id: int
    device: str
    root: str
    mount_point: str
    fs_type: str
    source: str
    options: tuple[str, ...]


def _unescape(value: str) -> str:
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)


def mount_facts(text: str) -> tuple[Mount, ...]:
    result = []
    for line in text.splitlines():
        try:
            before, after = line.split(' - ', 1)
            fields, filesystem = before.split(), after.split()
            if len(fields) < 6 or len(filesystem) != 3:
                raise ValueError
            mount = Mount(int(fields[0]), fields[2], _unescape(fields[3]), _unescape(fields[4]),
                filesystem[0], _unescape(filesystem[1]),
                tuple(_unescape(value) for value in (fields[5] + ',' + filesystem[2]).split(',')))
            if not Path(mount.mount_point).is_absolute():
                raise ValueError
            result.append(mount)
        except (ValueError, IndexError) as error:
            raise ContractError('unreadable actual Linux mount identity') from error
    if not result:
        raise ContractError('actual Linux mount table required')
    return tuple(result)


def read_mounts() -> tuple[Mount, ...]:
    return mount_facts(Path('/proc/self/mountinfo').read_text())


def windows_drive(mount: Mount) -> str | None:
    if mount.fs_type not in ('9p', 'drvfs', 'fuseblk'):
        return None
    match = re.match(r'^([A-Za-z]):[\\/]', mount.source)
    if match is not None:
        return match[1].upper()
    if mount.fs_type in ('9p', 'drvfs'):
        for option in mount.options:
            match = re.search(r'(?:^|;)path=([A-Za-z]):[\\/]', option)
            if match is not None and ('drvfs' in option or mount.fs_type == 'drvfs'):
                return match[1].upper()
    return None


def existing_ancestor(path: Path) -> Path:
    safe_storage_path(path)
    while not path.exists():
        path = path.parent
    return path


def _mount(path: Path, mounts: tuple[Mount, ...]) -> Mount:
    actual = existing_ancestor(path)
    candidates = [mount for mount in mounts if actual.is_relative_to(Path(mount.mount_point))]
    if not candidates:
        raise ContractError('destination has no actual mounted filesystem identity')
    return max(candidates, key=lambda mount: (len(Path(mount.mount_point).parts), mount.mount_id))


def destination_mount(path: Path, mounts: tuple[Mount, ...], *,
                       fs_type: str | None = None, file: bool = False, writable: bool) -> Mount:
    existing = existing_ancestor(path)
    if not existing.is_dir() and (not file or path != existing or not existing.is_file()):
        raise ContractError(f'HPO destination is not a usable path: {path}')
    if file and path == existing and not existing.is_file():
        raise ContractError(f'HPO SQLite destination must be a regular file: {path}')
    mount = _mount(path, mounts)
    if path.is_relative_to(GF_DATA_ROOT):
        distro = next((row for row in mounts if row.mount_point == '/'), None)
        if (Path(mount.mount_point) != GF_DATA_ROOT or mount.fs_type != 'ext4'
                or distro is not None and mount.device == distro.device):
            raise ContractError(f'HPO gf_data requires the dedicated ext4 mount: {path}')
    if path.is_relative_to(WINDOWS_D_ROOT) and windows_drive(mount) != 'D':
        raise ContractError(f'configured D-drive destination requires the actual D mount: {path}')
    if fs_type is not None and mount.fs_type != fs_type:
        raise ContractError(f'HPO destination must be actual {fs_type}: {path}')
    if writable:
        parent = existing_ancestor(path.parent) if file else existing
        if ('ro' in mount.options or os.statvfs(existing).f_flag & os.ST_RDONLY
                or not os.access(parent, os.W_OK | os.X_OK)
                or file and existing == path and not os.access(path, os.W_OK)):
            raise ContractError(f'HPO destination is read-only or not writable: {path}')
    return mount


def check_sqlite_mounts(database: Path, mounts: tuple[Mount, ...], *,
                        fs_type: str | None = None, writable: bool, same_device: str | None = None) -> None:
    """SQLite can write beside the database or open an existing auxiliary file."""
    for suffix in ('', '-wal', '-shm', '-journal'):
        path = database.with_name(database.name + suffix)
        if path.exists() or path.is_symlink() or not suffix:
            actual = destination_mount(path, mounts, fs_type=fs_type, file=True, writable=writable)
            if same_device is not None and actual.device != same_device:
                raise ContractError(f'SQLite files must share the originals filesystem: {path}')


def check_collection_destinations(output: Path, originals: Path, *, catalog_root: Path) -> None:
    """Admit the exact collection controls, Pack directory and SQLite catalog."""
    mounts = read_mounts()
    destination_mount(output, mounts, fs_type='ext4', writable=True)
    destination_mount(originals, mounts, writable=True)
    check_sqlite_mounts(catalog_root / 'catalog.sqlite3', mounts, writable=True)


def check_staged_collection_destinations(work: Path, output: Path, *,
                                         catalog_root: Path, originals: Path | None = None) -> tuple[Mount, Mount]:
    """Preflight the dedicated durable filesystem and private collection tmpfs."""
    mounts = read_mounts()
    distro = _mount(Path('/'), mounts)
    temporary = destination_mount(work, mounts, fs_type='tmpfs', writable=True)
    controls = destination_mount(output, mounts, fs_type='ext4', writable=True)
    durable = destination_mount(catalog_root, mounts, fs_type='ext4', writable=True)
    if distro.device in (controls.device, durable.device):
        raise ContractError('staged collection controls and originals require dedicated ext4')
    if originals is not None and (
            destination_mount(originals, mounts, fs_type='ext4', writable=True).device != durable.device):
        raise ContractError('staged collection and catalog must share the same ext4 filesystem')
    check_sqlite_mounts(catalog_root / 'catalog.sqlite3', mounts, fs_type='ext4',
                        writable=True, same_device=durable.device)
    return temporary, durable
