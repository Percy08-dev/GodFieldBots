"""Explicit bounded readonly storage audit; this never grants Source authority."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from gf_common.domain import Json, canonical, closed
from gf_common.domain import digest as object_digest
from gf_common.group_pack import MAX_JSON, MIB, canonical_json, digest, parse_json, safe_name
from gf_common.group_pack_game import checked_storage_custody, same_storage_custody
from gf_training.group_pack_db import is_digest, validate_database
from gf_training.group_pack_resources import ByteBudget, close_resources, remaining, retain_cleanup
from gf_training.group_pack_source import OriginalFile, PackedOriginals, _no_symlink_path, _stat_key

POLICY = 'pack-checksum-v1'


class StorageAuditError(RuntimeError):
    """Report plus the actual cleanup owner, including uncertain close state."""
    def __init__(self, result: Json, owner: Any, error: BaseException) -> None:
        super().__init__(str(error))
        self.result, self.cleanup_owner, self.error = result, owner, error


class _Audit:
    def __init__(self, target: Path, kind: str, memory: int, wall: float) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + wall
        self.budget = ByteBudget(memory)
        self._cleanup_children: list[Any] = []
        self.owners: list[Any] = []
        self.errors: list[BaseException] = []
        self.pending_charges: list[int] = []
        self.scanned: dict[tuple[int, int], Json] = {}
        self.completed_files: list[OriginalFile] = []
        self.completed_absences: list[Path] = []
        self.completed_namespaces: list[tuple[Path, tuple[int, int], set[str]]] = []
        self.result: Json = {'schema_version': 1, 'kind': 'group-pack-storage-audit',
            'target': {'kind': kind, 'path': str(target.absolute())}, 'policy': POLICY,
            'scope': 'diagnostic-storage-only', 'passed': False, 'complete': False,
            'counts': {'references': 0, 'unique_databases': 0, 'packs': 0, 'members': 0, 'controls': 0},
            'references': [], 'failures': [], 'elapsed_seconds': 0.0,
            'limits': {'memory_budget_bytes': memory, 'wall_seconds': wall},
            'cleanup_confirmed': False}

    def check(self) -> None:
        remaining(self.deadline)

    def close(self) -> None:
        close_resources(self, self.owners)
        self.owners.clear()
        for amount in self.pending_charges:
            self.budget.release(amount)
        self.pending_charges.clear()

    def _retire(self, owner: Any) -> None:
        if isinstance(owner, PackedOriginals):
            files = [file for file in (owner.database_file, owner.descriptor_file, owner.predecessor_file)
                     if file is not None]
            self.budget.acquire(4096*len(files), self.deadline)
            self.completed_files.extend(files)
            self.completed_namespaces.append((owner.directory,
                (owner.directory_identity.st_dev, owner.directory_identity.st_ino), set(owner.namespace)))
        else:
            self.budget.acquire(4096, self.deadline)
            self.completed_files.append(owner)
        owner.close()
        self.owners.remove(owner)
        self.check()  # A successful close after deadline is still a failed audit.

    def check_completed(self) -> None:
        """Final pathname checks bind the stat6 of originals retired after verified reads."""
        for file in self.completed_files:
            self.check()
            file.check_path_identity()
        for path in self.completed_absences:
            self.check()
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ValueError('audit completed deletion staging entry reappeared')
        for path, identity, expected in self.completed_namespaces:
            self.check()
            _no_symlink_path(path)
            info = path.stat()
            if (info.st_dev, info.st_ino) != identity:
                raise ValueError('audit original namespace identity changed')
            with os.scandir(path) as entries:
                seen = set()
                for entry in entries:
                    self.check()
                    if entry.name not in expected or not entry.is_file(follow_symlinks=False):
                        raise ValueError('audit original namespace changed')
                    seen.add(entry.name)
            if seen != expected:
                raise ValueError('audit original namespace changed')

    def file(self, path: Path, expected: str | None = None, *, json_value: bool = True,
             retain_value: bool = False) -> Any:
        self.check()
        # Admit metadata BEFORE OriginalFile hashes/reads it. Identity checks on
        # its actual FD detect growth/replacement across this initial size check.
        size = path.lstat().st_size
        if json_value and size > 64*MIB:
            raise ValueError('audit control exceeds metadata admission')
        charge = (32 * size if json_value else 65536) + 65536
        self.budget.acquire(charge, self.deadline)
        retired = False
        try:
            # Preown the actual object before __init__ can open/hash and fail.
            original = OriginalFile.__new__(OriginalFile)
            self.owners.append(original)
            OriginalFile.__init__(original, path, self.deadline)
            if original.identity.st_size != size:
                raise ValueError('audit control changed during admission')
            if expected is not None and original.sha256 != expected:
                raise ValueError('audit original control checksum differs')
            value = parse_json(original.read(64*MIB), limit=64*MIB) if json_value else None
            original.recheck()
            self._retire(original)
            retired = True
            return value
        except BaseException as error:
            retain_cleanup(self, error)
            raise
        finally:
            if not retired:
                # An uncertain constructor/close keeps this reservation with
                # its actual owner; only confirmed cleanup refunds it.
                self.pending_charges.append(charge)
            elif not retain_value:
                self.budget.release(charge)

    def reference(self, path: Path, *, expected_storage: Json | None = None, expected_custody: Json | None = None,
                  collection: Json | None = None, parent: Json | None = None) -> None:
        self.check()
        # Report records and scan receipts stay accounted after owners retire.
        self.budget.acquire(65536 + 128 * len(str(path)), self.deadline)
        pointer: OriginalFile | None = None
        originals: PackedOriginals | None = None
        try:
            size = path.lstat().st_size
            if size > MAX_JSON:
                raise ValueError('audit locator/descriptor exceeds metadata admission')
            pointer_charge = 32 * size + 65536
            self.budget.acquire(pointer_charge, self.deadline)
            pointer = OriginalFile(path, self.deadline)
            self.owners.append(pointer)
            if pointer.identity.st_size != size:
                raise ValueError('audit pointer changed during admission')
            value = parse_json(pointer.read(MAX_JSON))
            if isinstance(value, dict) and value.get('format') == 'sqlite-group-pack-locator-v1':
                closed(value, {'format', 'schema_version', 'descriptor', 'descriptor_sha256'})
                if (type(value['schema_version']) is not int or value['schema_version'] != 1
                        or type(value['descriptor']) is not str or not Path(value['descriptor']).is_absolute()):
                    raise ValueError('audit requires exact locator1')
                descriptor_path = Path(value['descriptor'])
            else:
                descriptor_path = path
            originals = PackedOriginals(descriptor_path, deadline=self.deadline,
                max_owned_bytes=self.budget.limit-self.budget.used, allow_pending=True)
            self.owners.append(originals)
            originals_charge = originals.budget.used
            self.budget.acquire(originals_charge, self.deadline)
            descriptor = originals.descriptor
            database = originals.database_file
            assert database is not None and originals.descriptor_file is not None
            if descriptor_path != path and originals.descriptor_file.sha256 != value['descriptor_sha256']:
                raise ValueError('audit locator descriptor checksum differs')
            if collection is not None and descriptor['publication'] != 'published':
                raise ValueError('Dataset4 requires published storage references')
            storage = {'format': 'sqlite-group-pack-v1', 'locator_sha256': pointer.sha256,
                'descriptor_sha256': originals.descriptor_file.sha256,
                'database_sha256': database.sha256, 'seal_id': descriptor['seal_id'],
                'logical_sha256': originals.validated['seal']['logical_sha256']}
            if expected_storage is not None:
                assert expected_custody is not None
                files: Any = [{'role': role, 'path': str(file.path), 'sha256': file.sha256,
                          'identity': list(_stat_key(file.identity))} for role, file in (
                    ('locator', pointer), ('descriptor', originals.descriptor_file),
                    ('database', database), ('predecessor', originals.predecessor_file)) if file is not None]
                dirs = [(path.parent, path.parent.stat()), (originals.directory, originals.directory_identity)]
                custody: Any = {'recipe': 'packed-original-custody-v2', 'files': sorted(files, key=lambda row: row['path']),
                    'directories': sorted([{'path': str(p.absolute()), 'device': s.st_dev, 'inode': s.st_ino}
                        for p, s in dirs], key=lambda row: row['path'])}
                custody['fingerprint'] = object_digest(custody)
                checked_storage_custody(expected_storage, expected_custody)
                if not same_storage_custody(expected_storage, expected_custody, storage, custody):
                    raise ValueError('audit Dataset4 original storage/custody differs')
            key = (database.identity.st_dev, database.identity.st_ino)
            observed: Json = {'database_bytes': database.identity.st_size,
                'database_sha256': None, 'run_id': originals.validated['seal']['run_id'],
                'seal_id': originals.validated['seal']['seal_id']}
            expected = {k: descriptor[k] for k in observed}
            reference = {'path': str(path.absolute()), 'publication': descriptor['publication'],
                'expected': expected, 'observed': observed, 'deduplicated': key in self.scanned,
                'complete': False}
            self.result['references'].append(reference)
            self.result['counts']['references'] += 1
            if key in self.scanned:
                receipt = self.scanned[key]
                if receipt['identity'] != _stat_key(database.identity) or receipt['expected'] != expected:
                    raise ValueError('conflicting expectations for SAME database inode')
                observed['database_sha256'] = receipt['sha256']
            else:
                observed['database_sha256'] = database.hash()
                if observed != expected:
                    raise ValueError('audit database size/checksum/run/seal differs')
                available = self.budget.limit-self.budget.used
                # The existing admitted metadata remains live. Full verification's
                # second bounded graph and codec must fit the remaining account.
                assert originals.db is not None
                validated = validate_database(originals.db, deadline=self.deadline,
                    max_pack_bytes=256*MIB, max_owned_bytes=available)
                index = validated['index']
                for name, count in (('packs', len(index['groups'])), ('members', len(index['members'])),
                                    ('controls', len(index['controls']))):
                    self.result['counts'][name] += count
                del validated, index
                self.scanned[key] = {'identity': _stat_key(database.identity),
                    'expected': expected, 'sha256': observed['database_sha256']}
                self.result['counts']['unique_databases'] += 1
            if observed != expected:
                raise ValueError('audit reference expected identity differs')
            if collection is not None:
                assert parent is not None
                self._collection(originals, collection, parent)
            originals.recheck()
            pointer.recheck()
            self.check()
            reference['complete'] = True
            self._retire(originals)
            self.budget.release(originals_charge)
            self._retire(pointer)
            self.budget.release(pointer_charge)
        except BaseException as error:
            retain_cleanup(self, error)
            raise

    def _collection(self, originals: PackedOriginals, row: Json, parent: Json) -> None:
        entry = next((c for c in originals.validated['index']['controls'] if c['name'] == 'manifest.json'), None)
        if entry is None:
            raise ValueError('audit Dataset4 collection manifest missing')
        charge = 32*entry['bytes']+65536
        self.budget.acquire(charge, self.deadline)
        try:
            self._collection_contents(originals, row, parent)
        finally:
            self.budget.release(charge)

    def _collection_contents(self, originals: PackedOriginals, row: Json, parent: Json) -> None:
        db = originals.db
        assert db is not None
        control = db.execute("SELECT data FROM controls WHERE name='manifest.json'").fetchone()
        if control is None or digest(control[0]) != row['manifest_sha256']:
            raise ValueError('audit Dataset4 collection manifest differs')
        manifest = parse_json(control[0], limit=64*MIB)
        link = manifest['linkage']
        if (manifest['collection_role'] != row['collection_role']
                or object_digest(manifest['schedule']) != row['schedule_sha256']
                or object_digest(manifest['policy_bindings']) != row['bindings_sha256']
                or link['parent_run_id'] != parent['run_id']
                or Path(link['parent_run_path']) != Path(parent['directory'])
                or link['final_generation' if row['split'] == 'validation' else 'collection_generation'] != row['generation']):
            raise ValueError('audit Dataset4 parent/collection correspondence differs')
        if row['numeric'] is not None:
            numeric = row['numeric']
            self.file(Path(numeric['directory'])/'rollout.json', numeric['sha256'])
            for name, sha in numeric['files'].items():
                self.file(Path(numeric['directory'])/safe_name(name), sha, json_value=False)

    def namespace(self, path: Path, names: set[str]) -> None:
        self.check()
        _no_symlink_path(path)
        info = path.stat()
        self.budget.acquire(65536 + sum(128*len(name) for name in names), self.deadline)
        self.completed_namespaces.append((path, (info.st_dev, info.st_ino), names))
        with os.scandir(path) as entries:
            seen = set()
            for entry in entries:
                self.check()
                if entry.name not in names or not entry.is_file(follow_symlinks=False):
                    raise ValueError('audit artifact retention namespace differs')
                seen.add(entry.name)
        if seen != names:
            raise ValueError('audit artifact retention namespace differs')

    def generation_artifacts(self, directory: Path, record: Json, declared: Json,
                             receipt: Json, *, required: bool) -> None:
        """Storage receipt checks only; never instantiate a model or qualification."""
        from gf_models.structured_sealed import basename, pruning_basename
        from gf_training.structured_lineage import RECEIPT_FIELDS
        closed(receipt, RECEIPT_FIELDS)
        files = record['artifact_files']
        if (type(files) is not dict or not {'inference.json', 'resume.json'}.issubset(files)
                or files['inference.json'] != declared['inference_sha256']
                or files['resume.json'] != declared['resume_sha256']
                or canonical(receipt['original_files']) != canonical(files)
                or receipt['inference_sha256'] != declared['inference_sha256']):
            raise ValueError('audit original artifact registry/map differs')
        for name, sha in files.items():
            basename(name)
            if not is_digest(sha):
                raise ValueError('audit original artifact checksum invalid')
        inference = self.file(directory/'inference.json', declared['inference_sha256'], retain_value=True)
        if receipt['status'] == 'retained':
            if receipt['deletion_intent'] is not None or receipt['deletion_receipt'] is not None:
                raise ValueError('audit retained artifact has deletion evidence')
            self.namespace(directory, set(files))
            for name, sha in files.items():
                if name != 'inference.json':
                    self.file(directory/name, sha, json_value=False)
            return
        if required or receipt['status'] != 'deleted':
            raise ValueError('audit required artifact retention or completed deletion missing')
        intent = closed(receipt['deletion_intent'], {'files', 'staging_basename'})
        targets = {name: sha for name, sha in files.items() if name != 'inference.json'}
        if (canonical(intent['files']) != canonical(targets)
                or canonical(receipt['deletion_receipt']) != canonical({'files': targets})):
            raise ValueError('audit confirmed deletion receipt differs')
        weights = inference['weights']['files']
        if (type(weights) is not dict or not weights
                or any(name not in files or files[name] != sha for name, sha in weights.items())):
            raise ValueError('audit pruned original weights map differs')
        states = set(files)-{'inference.json', 'resume.json'}-set(weights)
        count = len(states)//2
        if not count or states != {f'resume-{i:06}.{suffix}' for i in range(count) for suffix in ('npz', 'json')}:
            raise ValueError('audit pruned original resume shard map differs')
        staging = directory.parent/pruning_basename(intent['staging_basename'])
        try:
            staging.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError('audit completed deletion staging entry exists')
        self.budget.acquire(4096, self.deadline)
        self.completed_absences.append(staging)
        self.namespace(directory, {'inference.json'})

    def dataset(self, path: Path) -> None:
        from gf_common.card_instance_game_tree import packed_manifest
        from gf_training.lineage import checkpoint_path, generation_path
        value = self.file(path, retain_value=True)
        # Reserve the admitted validated Dataset graph throughout the serial scan.
        size = len(canonical_json(value, limit=self.budget.limit))
        self.budget.acquire(64*size+65536, self.deadline)
        dataset = packed_manifest(value)
        for parent in dataset['parents']:
            root = Path(parent['directory'])
            state = self.file(root/'run.json', parent['run_sha256'], retain_value=True)
            if (state['run_id'] != parent['run_id'] or state['schema_version'] != 4
                    or state['resume_identity'] != parent['resume_identity']
                    or state['config_sha256'] != parent['config_sha256']
                    or object_digest(state['config_inputs']) != parent['config_inputs_sha256']
                    or state['formal_declaration']['sha256'] != parent['formal_declaration_sha256']
                    or state['seed'] != parent['seed'] or state['updates'] != parent['final_generation']
                    or state['generation_sha256'] != parent['final_generation_sha256']
                    or state['storage_binding'] != dataset['storage_binding']
                    or parent['generations'][-1]['generation_sha256'] != parent['final_generation_sha256']
                    or parent['generations'][-1]['inference_sha256'] != parent['final_inference_sha256']):
                raise ValueError('audit Dataset4 parent control correspondence differs')
            registry = closed(self.file(root/'artifact-registry.json', parent['registry_sha256'], retain_value=True),
                {'schema_version', 'generations'})
            if (type(registry['schema_version']) is not int or registry['schema_version'] != 3
                    or type(registry['generations']) is not dict
                    or set(registry['generations']) != {str(i) for i in range(parent['final_generation']+1)}):
                raise ValueError('audit artifact registry complete generation inventory differs')
            for generation in parent['generations']:
                number = generation['generation']
                record = self.file(generation_path(root, number), generation['generation_sha256'], retain_value=True)
                if (object_digest(record['artifact_files']) != generation['artifact_files_sha256']
                        or record['generation'] != number
                        or any(record[key] != generation[key] for key in ('update_sha256', 'inference_sha256', 'resume_sha256'))):
                    raise ValueError('audit Dataset4 artifact aggregate differs')
                if generation['update_sha256'] is not None:
                    self.file(root/'updates'/f'{number:06}.json', generation['update_sha256'])
                self.generation_artifacts(checkpoint_path(root, number), record, generation,
                    registry['generations'][str(number)],
                    required=number == 0 or number % 10 == 0 or number == parent['final_generation'])
        for row in dataset['collections']:
            self.reference(Path(row['directory'])/'collection.locator.json',
                expected_storage=row['storage'], expected_custody=row['custody'], collection=row,
                parent=dataset['parents'][row['parent_index']])


def verify_storage(*, locator: Path | None = None, dataset: Path | None = None,
                   memory_budget_mib: float = 2048, wall_seconds: float = 7200) -> Json:
    """Audit exact requested storage. Failure retains its report and actual owners."""
    if (locator is None) == (dataset is None):
        raise ValueError('exactly one locator or Dataset4 target required')
    for value in (memory_budget_mib, wall_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or value > 2**53:
            raise ValueError('audit limits must be strictly positive finite values')
    target = locator if locator is not None else dataset
    assert target is not None
    audit = _Audit(target, 'locator' if locator is not None else 'dataset4',
        int(memory_budget_mib*MIB), float(wall_seconds))
    primary: BaseException | None = None
    try:
        audit.budget.acquire(65536, audit.deadline)
        if locator is not None:
            audit.reference(locator)
        else:
            audit.dataset(target)
        audit.check()
        audit.result['complete'] = True
    except BaseException as error:  # noqa: BLE001 -- cancellation also produces a failed diagnostic.
        primary = error
        audit.errors.append(error)
    try:
        audit.close()
        audit.result['cleanup_confirmed'] = True
        audit.check_completed()
        audit.check()
    except BaseException as error:  # noqa: BLE001 -- actual uncertain owner retained in StorageAuditError.
        if primary is not None and primary is not error:
            primary.add_note('audit cleanup/final check: ' + str(error))
        audit.errors.append(error)
        primary = primary or error
    audit.result['elapsed_seconds'] = time.monotonic()-audit.started
    audit.result['passed'] = primary is None and audit.result['cleanup_confirmed']
    if primary is not None:
        audit.result['complete'] = False
        audit.result['failures'].extend({'error': type(error).__name__, 'message': str(error)}
            for error in audit.errors)
        raise StorageAuditError(audit.result, audit, primary) from primary
    return audit.result
