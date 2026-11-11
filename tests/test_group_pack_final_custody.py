"""Canonical locator flock custody supplies no Source or fitting permission."""
import os
import time

import pytest

from gf_common.domain import ContractError
from gf_training.card_instance_source_locks import (
    ValidationLocatorOwnership,
    ValidationManifestOwnership,
)
from tests.test_group_pack_game_source import packed


def test_public_packed_custody_locks_the_actual_locator_and_rejects_replacement(tmp_path):
    root, _ = packed(tmp_path)
    assert not (root / 'manifest.json').exists()
    lock = ValidationLocatorOwnership(root, deadline=time.monotonic() + 30)
    with pytest.raises(ContractError, match='custody'), lock:
        lock.require(root)
        original = (root / 'collection.locator.json').stat()
        assert lock.inode == (original.st_dev, original.st_ino)
        with pytest.raises(BlockingIOError), ValidationLocatorOwnership(root, deadline=time.monotonic() + 30):
            pytest.fail('second actual locator flock was admitted')
        replacement = root / 'replacement'
        replacement.write_bytes((root / 'collection.locator.json').read_bytes())
        os.replace(replacement, root / 'collection.locator.json')
        with pytest.raises(ContractError, match='custody'):
            lock.require(root)
    assert lock.cleanup_complete
    with pytest.raises(FileNotFoundError), ValidationManifestOwnership(root, deadline=time.monotonic() + 30):
        pytest.fail('old manifest custody must retain its old literal name')
