"""Concrete corrected original Source owner over retained DB members and custody."""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.group_pack import MAX_JSON, parse_json, safe_name
from gf_common.group_pack_game import checked_storage_binding
from gf_training.group_pack_pool import OriginalArtifactReader, PackedArtifactPool
from gf_training.group_pack_resources import close_resources, retain_cleanup
from gf_training.group_pack_source import OriginalFile, PackedSource, _no_symlink_path, _stat_key
from gf_training.structured_source import _SourceOwner


class PackedSourceOwner(_SourceOwner):
    def __init__(self, directory: Path, settings: Any, deadline: float, *, storage_binding: Json,
                 max_owned_bytes: int, extra_owned_bytes: int = 0,
                 storage_limit: int | None = None, pool: PackedArtifactPool | None = None,
                 expected_descriptor_root: Path | None = None) -> None:
        checked_storage_binding(storage_binding)
        self.root, self.settings, self.deadline = directory.absolute(), settings, deadline
        self.limit = min(max_owned_bytes, settings.memory_budget_mib * 1024**2,
                         storage_limit or max_owned_bytes)
        self.extra_owned_bytes = extra_owned_bytes
        self.used = 0
        self.files: Json = {}
        self.inventories: list[Any] = []
        self._cleanup_children: list[Any] = []
        self._closed = False
        self._owns_pool = pool is None
        self.pool = pool or PackedArtifactPool(max_owned_bytes=self.limit, deadline=deadline)
        if type(self.pool) is not PackedArtifactPool or self.pool.budget.limit > self.limit:
            raise ContractError('same bounded concrete original pool required')
        self.locator: OriginalFile | None = None
        self.artifacts: PackedSource | None = None
        self._charge_bytes = 0
        try:
            self.charge(extra_owned_bytes + 65536)
            self.directory_identity = _no_symlink_path(self.root).stat()
            self.locator = OriginalFile(self.root / 'collection.locator.json', deadline)
            self.charge(32 * self.locator.identity.st_size + 65536)
            value = closed(parse_json(self.locator.read(MAX_JSON)),
                {'format', 'schema_version', 'descriptor', 'descriptor_sha256'})
            if (value['format'] != 'sqlite-group-pack-locator-v1'
                    or type(value['schema_version']) is not int or value['schema_version'] != 1
                    or type(value['descriptor']) is not str or not Path(value['descriptor']).is_absolute()):
                raise ContractError('explicit original collection locator required')
            descriptor = Path(value['descriptor'])
            if expected_descriptor_root is not None and descriptor.parent != expected_descriptor_root.absolute():
                raise ContractError('packed locator targets foreign parent batch root')
            self.artifacts = PackedSource(descriptor, deadline=deadline,
                max_owned_bytes=self.limit, pool=self.pool)
            originals = self.artifacts._originals
            if originals.descriptor_file is None or originals.descriptor_file.sha256 != value['descriptor_sha256']:
                raise ContractError('original locator/descriptor hash mismatch')
            self.reader = OriginalArtifactReader(self.artifacts, self.read_raw, self.charge, self.root)
            self._entries = {m['name']: m for m in self.artifacts._index['members']}
            self._entries.update({c['name']: c for c in self.artifacts._index['controls']})
            self.finalize()
        except BaseException as error:
            retain_cleanup(self, error)
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- actual owner retained on failure.
                error.add_note('packed Source cleanup: ' + str(cleanup))
            raise

    def check(self) -> None:
        super().check()
        if self._closed:
            raise RuntimeError('packed original Source closed')

    def charge(self, size: int) -> None:
        self.check()
        self.pool.budget.acquire(size, self.deadline)
        self._charge_bytes += size
        self.used += size

    def _safe(self, path: Path) -> None:
        if not path.is_relative_to(self.root):
            raise ContractError('foreign logical source path')
        safe_name(path.relative_to(self.root).as_posix())

    def contains(self, path: Path) -> bool:
        self._safe(path)
        return path.relative_to(self.root).as_posix() in self._entries

    def member_identity(self, name: str) -> Json:
        assert self.artifacts is not None
        entry = self._entries[name]
        group = entry.get('group_id')
        return {'format': 'packed-member-v1', 'database_sha256': self.storage['database_sha256'],
            'group_id': group, 'pack_sha256': self.artifacts._groups[group]['sha256'] if group is not None else None,
            'name': name, 'offset': entry.get('offset'), 'length': entry.get('length', entry.get('bytes')),
            'sha256': entry['sha256']}

    def read(self, path: Path) -> bytes:
        """Generic original read retains its existing JSON/decode allowance."""
        return self._read_original(path, decode=True)

    def read_raw(self, path: Path) -> bytes:
        """Journal/column bytes only; their parser admits its own live graphs."""
        return self._read_original(path, decode=False)

    def _read_original(self, path: Path, *, decode: bool) -> bytes:
        self.check()
        self._safe(path)
        name = path.relative_to(self.root).as_posix()
        entry = self._entries[name]
        size = entry.get('length', entry.get('bytes'))
        if size > self.settings.max_source_file_mib * 1024**2:
            raise ContractError('bounded original logical file required')
        # Each read's immutable raw bytes/aliases remain conservatively held
        # through actual graph retirement or close. Never deduplicate a second
        # live read/parse by filename; only stable evidence metadata is shared.
        self.charge((32 if decode else 2) * size + 65536)
        assert self.artifacts is not None
        if name in self.artifacts._member_groups:
            with self.artifacts.borrow_group(self.artifacts._member_groups[name]) as loan:
                raw = loan.read(name)
        else:
            raw = self.artifacts.read(name)
        self._remember_entry(name)
        return raw

    def _remember_entry(self, name: str) -> Json:
        if name not in self.files:
            # Reserve the actual stable identity/evidence graph BEFORE creating
            # it. The constructor already admitted and validated the complete
            # index; this additional map has its own retained metadata charge.
            self.charge(2048 + 32 * len(name))
            identity = self.member_identity(name)
            self.files[name] = {'sha256': identity['sha256'], 'bytes': identity['length'], 'identity': identity}
        return self.files[name]

    def remember_inventory_entry(self, name: str, size: int, checksum: str) -> None:
        self.check()
        self._safe(self.root / name)
        integer(size, 0)
        if size > self.settings.max_source_file_mib * 1024**2:
            raise ContractError('bounded original logical file required')
        original = self._entries.get(name)
        if (original is None or original.get('length', original.get('bytes')) != size
                or original['sha256'] != checksum):
            raise ContractError('original packed inventory hash/size/name mismatch')
        # This is the SAME constructor-validated DB/Pack/member index, not fresh
        # SQL or caller-supplied evidence. Finalize still checks actual originals.
        self._remember_entry(name)

    def remember(self, path: Path, identity: Any, checksum: str) -> None:
        self.check()
        self._safe(path)
        name = path.relative_to(self.root).as_posix()
        expected = self._remember_entry(name)
        if canonical(identity) != canonical(expected['identity']) or checksum != expected['sha256']:
            raise ContractError('packed logical identity differs from original member')

    def _inventory_names(self, pattern: str, *, retain: bool) -> tuple[str, ...]:
        self.check()
        assert self.artifacts is not None
        names = tuple(name for name in self.artifacts.names()
            if pattern == '**/*' or fnmatch.fnmatchcase(name, pattern))
        if retain:
            self.charge(512 + sum(512 + 4 * len(name) for name in names))
        return names

    def validate_index(self, manifest: Json) -> None:
        assert self.artifacts is not None
        index = self.artifacts._index
        groups = {row['group_id']: row for row in index['groups']}
        episodes = {row['episode_id']: row for row in index['episodes']}
        expected_episodes = {row['episode_id']: row for row in manifest['episodes']}
        expected_groups = {group['episodes'][0]['episode_id'] for group in manifest['schedule']}
        if set(groups) != expected_groups or set(episodes) != set(expected_episodes):
            raise ContractError('packed group/episode index differs from original complete schedule')
        episode_groups = {}
        for schedule, health in zip(manifest['schedule'], manifest['groups'], strict=True):
            group_id = schedule['episodes'][0]['episode_id']
            if canonical(groups[group_id]['record']) != canonical(health):
                raise ContractError('packed group index differs from original group health')
            for spec in schedule['episodes']:
                name = spec['episode_id']
                original = expected_episodes[name]
                row = episodes[name]
                if (row['group_id'] != group_id or row['game_id'] != original['game_id']
                        or canonical(row['record']) != canonical({key: original[key] for key in ('end_kind', 'failure')})):
                    raise ContractError('packed episode index differs from original result')
                episode_groups[name] = group_id
        for row in index['members']:
            prefix = row['name'].split('/')[0]
            if prefix in episode_groups and row['group_id'] != episode_groups[prefix]:
                raise ContractError('packed own episode member belongs to a different group')

    def retire_decoded(self) -> None:
        """Drop per-batch decode allowances after its graph has explicitly retired."""
        retained = 65536 + 1024 * len(self.files) + sum(len(str(k)) * 32 for k in self.files)
        release = max(0, self._charge_bytes - retained)
        self.pool.budget.release(release)
        self._charge_bytes -= release

    def borrow_group(self, group_id: str) -> Any:
        assert self.artifacts is not None
        return self.artifacts.borrow_group(group_id)

    @property
    def storage(self) -> Json:
        self.check()
        assert self.artifacts is not None and self.locator is not None
        originals = self.artifacts._originals
        descriptor = originals.descriptor
        assert originals.descriptor_file is not None and originals.database_file is not None
        return {'format': 'sqlite-group-pack-v1', 'locator_sha256': self.locator.sha256,
            'descriptor_sha256': originals.descriptor_file.sha256,
            'database_sha256': originals.database_file.sha256,
            'seal_id': descriptor['seal_id'], 'logical_sha256': originals.validated['seal']['logical_sha256']}

    @property
    def custody(self) -> Json:
        self.check()
        assert self.artifacts is not None and self.locator is not None
        originals = self.artifacts._originals
        files: list[Json] = [{'role': role, 'path': str(file.path), 'sha256': file.sha256,
            'identity': list(_stat_key(file.identity))} for role, file in (
                ('locator', self.locator), ('descriptor', originals.descriptor_file),
                ('database', originals.database_file), ('predecessor', originals.predecessor_file)) if file is not None]
        directories: list[Json] = [{'path': str(path), 'device': value.st_dev, 'inode': value.st_ino}
            for path, value in ((self.root, self.directory_identity), (originals.directory, originals.directory_identity))]
        content = {'recipe': 'packed-original-custody-v1', 'files': sorted(files, key=lambda row: row['path']),
            'directories': sorted(directories, key=lambda row: row['path'])}
        return {**content, 'fingerprint': digest(content)}

    def require_custody(self, custody: Json) -> None:
        self.finalize()
        if canonical(custody) != canonical(self.custody):
            raise ContractError('persisted packed original custody changed')

    def finalize(self) -> None:
        self.check()
        if self.locator is not None:
            self.locator.recheck()
        current = _no_symlink_path(self.root).stat()
        if (current.st_dev, current.st_ino) != (self.directory_identity.st_dev, self.directory_identity.st_ino):
            raise ContractError('original packed envelope directory changed')
        seen = False
        with __import__('os').scandir(self.root) as entries:
            for entry in entries:
                self.check()
                if seen or entry.name != 'collection.locator.json' or not entry.is_file(follow_symlinks=False):
                    raise ContractError('original packed envelope namespace changed')
                seen = True
        if not seen:
            raise ContractError('original packed envelope locator missing')
        if self.artifacts is not None:
            self.artifacts.recheck()

    def close(self) -> None:
        if self._closed:
            return
        self.pool.require_terminal_close()
        if self.artifacts is not None and self.artifacts._loans:
            raise RuntimeError('cannot close a Source with live group loans')
        close_resources(self, [self.artifacts, self.locator])
        getattr(self, '_entries', {}).clear()
        self.files.clear()
        self.inventories.clear()
        self.pool.budget.release(self._charge_bytes)
        self._charge_bytes = 0
        if self._owns_pool:
            self.pool.close()
        self._closed = True


class PackedBatchCustody:
    """SAME original batch retained through callback, projection and precommit."""
    def __init__(self, owner: PackedSourceOwner) -> None:
        if type(owner) is not PackedSourceOwner:
            raise TypeError('concrete original Source owner required')
        self.owner = owner
        self.scans: list[Any] = []
        self.retired = False
        self.frozen_proof: bytes | None = None

    def borrow(self, scan: Any) -> PackedSourceOwner:
        if self.retired:
            raise ContractError('packed batch custody already retired')
        if scan not in self.scans:
            self.scans.append(scan)
        if scan._packed_pool is not None and scan._packed_pool is not self.owner.pool:
            raise ContractError('scan cannot borrow a second packed session pool')
        scan._packed_pool = self.owner.pool
        scan._owns_packed_pool = False
        scan.admit_pool_controls()
        scan._packed_owners[self.owner.root] = self.owner
        scan._packed_borrows[self.owner.root] = self
        return self.owner

    def release(self, scan: Any) -> None:
        scan._packed_owners.pop(self.owner.root, None)
        scan._packed_borrows.pop(self.owner.root, None)
        if scan in self.scans:
            self.scans.remove(scan)

    def prepare(self, proof: Json) -> None:
        if self.retired:
            raise ContractError('packed custody cannot prepare twice')
        self.owner.pool.require_terminal_close()
        self.owner.finalize()
        if canonical(proof['custody']) != canonical(self.owner.custody):
            raise ContractError('prepared proof differs from SAME original custody')
        frozen = canonical(proof).encode()
        # Close every historical original admitted by these SAME scans before
        # the batch's pool/FD retirement. Physical lineage graphs remain charged.
        for scan in tuple(self.scans):
            scan.close_packed_originals(exclude=self.owner.root)
        self.owner.close()
        for scan in tuple(self.scans):
            self.release(scan)
        self.frozen_proof = frozen
        self.retired = True

    def close(self) -> None:
        if self.retired:
            return
        self.owner.pool.require_terminal_close()
        primary: BaseException | None = None
        try:
            self.owner.finalize()
        except BaseException as error:  # noqa: BLE001 -- recheck cannot suppress actual FD retirement.
            primary = error
        cleanup_error: BaseException | None = None
        for scan in tuple(self.scans):
            try:
                scan.close_packed_originals(exclude=self.owner.root, recheck=False)
            except BaseException as cleanup:  # noqa: BLE001 -- attempt independent scans and the actual batch.
                cleanup_error = cleanup_error or cleanup
        try:
            self.owner.close()
        except BaseException as cleanup:  # noqa: BLE001 -- retain the SAME actual cleanup owner.
            cleanup_error = cleanup_error or cleanup
        if cleanup_error is not None:
            failure = primary or cleanup_error
            failure.__dict__['cleanup_owner'] = self
            if primary is not None:
                primary.add_note('packed original close: ' + str(cleanup_error))
                raise primary from cleanup_error
            raise failure
        for scan in tuple(self.scans):
            self.release(scan)
        self.retired = True
        if primary is not None:
            raise primary
