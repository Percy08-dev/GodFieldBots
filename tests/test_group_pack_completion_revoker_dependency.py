"""The last actual revoker depends on earlier close stages becoming certain."""
import os
import stat
import time

import pytest

from gf_common.domain import ContractError
from gf_models.structured_tree_pairing import (
    TREE2_RECORDS,
    TREE3_GAME_RECORDS,
    TREE3_PACKED_GAME_RECORDS,
    TREE3_RECORDS,
)
from gf_training.card_instance_formal import FormalCompletion
from gf_training.group_pack_resources import OwnedDescriptor
from tests.test_group_pack_formal_completion_handoff import controls
from tests.test_group_pack_tree_completion_handoff import primitive


@pytest.mark.parametrize('context', ['formal-old', 'formal-packed', TREE2_RECORDS, TREE3_RECORDS, TREE3_GAME_RECORDS, TREE3_PACKED_GAME_RECORDS])
@pytest.mark.parametrize('stage', ['directory', 'auxiliary'])
@pytest.mark.parametrize('after', [False, True])
def test_prior_close_failure_preserves_revoker_across_actual_original_move(tmp_path, monkeypatch, context, stage, after):
    formal = isinstance(context, str)
    if formal:
        directory = tmp_path/'formal'
        state = controls(directory, context == 'formal-packed')
        publication = FormalCompletion()
        publication.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
        name = 'run.json'
    else:
        publication, state = primitive(tmp_path/'tree', context)
        directory, name = publication.directory, 'training.json'
    auxiliary = None
    if stage == 'auxiliary':
        auxiliary = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        publication._opened_stages.append(OwnedDescriptor(auxiliary))
    closing = os.close
    first = OSError('earlier close uncertainty')
    attempted = []
    moved = directory/'moved-controlled-original.json'
    foreign = []
    def fault(fd):
        if attempted and fd == attempted[0]:
            pytest.fail('uncertain raw descriptor retried')
        target = fd == auxiliary if auxiliary is not None else stat.S_ISDIR(os.fstat(fd).st_mode)
        if target and not attempted:
            attempted.append(fd)
            if after:
                closing(fd)
            path = directory/name
            raw = path.read_bytes()
            os.replace(path, moved)
            path.write_bytes(raw)
            foreign.append(path.stat())
            raise first
        return closing(fd)
    monkeypatch.setattr(os, 'close', fault)
    try:
        with pytest.raises(OSError) as caught:
            publication.finish() if formal else publication.commit(state)
        assert caught.value is first
        if formal:
            assert publication.fd is not None
            with pytest.raises((RuntimeError, OSError, ContractError)):
                publication.revoke()
        assert publication.revoked and not publication.committed
        assert moved.read_bytes().startswith(b'!')
        assert not (directory/name).read_bytes().startswith(b'!')
        info = (directory/name).stat()
        assert (info.st_dev, info.st_ino) == (foreign[0].st_dev, foreign[0].st_ino)
    finally:
        monkeypatch.setattr(os, 'close', closing)
        if publication.fd is not None:
            try:
                publication.revoke()
            except (RuntimeError, OSError, ContractError):
                pass  # Actual uncertain stages remain quarantined after known-inode invalidation.
        if attempted and not after:
            closing(attempted[0])  # Controlled test disposal of the deliberately uncertain live descriptor.
