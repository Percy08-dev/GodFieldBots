"""Physical final envelopes only. These synthetic controls confer no Source rights."""
import asyncio
import json
import time

import pytest

from gf_common.domain import ContractError
from gf_common.group_pack import StorageSettings
from gf_training.card_instance_collection import CollectionPublication
from gf_training.group_pack_collection import GroupPackCollection
from gf_training.group_pack_publication import GroupPackCollectionPublication
from tests.test_group_pack_game_source import owner as public_source


async def staged(root):
    collection = GroupPackCollection(StorageSettings(db_root=root / 'db', ram_root=root / 'ram'),
        'unadmitted-final-physical', deadline=time.monotonic() + 30)
    await collection.start()
    envelope = root / 'envelope'
    envelope.mkdir()
    group = GroupPackCollectionPublication(collection, envelope)
    collection.freeze_storage_metrics()
    report = {'healthy': True, 'wall_seconds': 0., 'scope': 'UNADMITTED physical test only'}
    manifest = {'schema_version': 5, 'collection_role': 'ppo-final-validation', 'files': [dict(row) for row in collection.inventory],
        'elapsed_seconds': 0., 'validation_invocation': {'started_unix': time.time(), 'started_monotonic': time.monotonic(),
            'caller_deadline_monotonic': collection.deadline, 'admission_deadline_monotonic': collection.deadline,
            'execution_deadline_monotonic': collection.deadline, 'wall_deadline_monotonic': collection.deadline}}
    await group.prepare(report, manifest)
    publication = CollectionPublication(envelope, work_deadline=collection.deadline,
        cleanup_deadline=collection.deadline, group_pack=group)
    return publication, report, manifest


def test_pending_final_locator_and_original_control_exist_once_without_public_authority(tmp_path):
    publication, report, manifest = asyncio.run(staged(tmp_path))
    try:
        publication.begin_final_validation(report, manifest)
        assert publication.final_pending
        assert publication.pending_manifest() == 'manifest.json'
        assert not (publication.directory / 'collection.locator.json').exists()
        pointer = publication.directory / 'collection.locator.json.tmp'
        value = json.loads(pointer.read_bytes())
        assert value['descriptor'] == str(publication.group_pack.pending.path)
        with pytest.raises((FileNotFoundError, ContractError)):
            public_source(publication.directory)
        with pytest.raises(ContractError):
            publication.finish_final_validation()
        assert not (publication.directory / 'collection.locator.json').exists()
    finally:
        publication.revoke()
    assert pointer.read_bytes().startswith(b'!')
    assert publication.group_pack.pending.path.read_bytes().startswith(b'!')
    assert publication.cleanup_complete


def test_final_revocation_invalidates_private_locator_before_descriptor_or_close(tmp_path, monkeypatch):
    publication, report, manifest = asyncio.run(staged(tmp_path))
    publication.begin_final_validation(report, manifest)
    pending = publication.group_pack.pending
    close = pending.close
    pointer = publication.directory / 'collection.locator.json.tmp'
    def checked_close():
        assert pointer.read_bytes().startswith(b'!')
        assert pending.path.read_bytes().startswith(b'!')
        close()
    monkeypatch.setattr(pending, 'close', checked_close)
    publication.revoke()
    assert publication.cleanup_complete and not publication.committed


def test_failed_pending_publication_retires_actual_collection_metadata_account(tmp_path):
    publication, report, manifest = asyncio.run(staged(tmp_path))
    publication.begin_final_validation(report, manifest)
    assert publication.group_pack.collection.budget.used > 0
    publication.revoke()
    assert publication.cleanup_complete
    assert publication.group_pack.collection.budget.used == 0


def test_outer_context_failure_revokes_already_linked_owned_pointer_before_healthy_return(tmp_path):
    from gf_training.card_instance_final_validation import (
        GameFinalValidationContext,
        _IssuedFinalValidationCollection,
    )
    from tests.test_group_pack_runtime_review import prepared, prepared_manifest
    collection, group = asyncio.run(prepared(tmp_path))
    publication = CollectionPublication(group.envelope, work_deadline=collection.deadline,
        cleanup_deadline=collection.deadline, group_pack=group)
    publication.commit({'healthy': True}, prepared_manifest(group))
    assert publication.committed
    # Cleanup-only context primitive. No Source or actual formal parent exists.
    context = GameFinalValidationContext.__new__(GameFinalValidationContext)
    context._failure, context._source, context._closed = None, None, False
    context._parent = object()
    context._issued = _IssuedFinalValidationCollection(context, publication)
    first = OSError('outer borrow retirement failed before healthy return')
    context._record_failure(first)
    assert (group.envelope/'collection.locator.json').read_bytes().startswith(b'!')
    assert context._failure is first
