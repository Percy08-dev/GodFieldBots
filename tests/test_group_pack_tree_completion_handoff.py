"""Output-control primitives only, without fitted pair or Source permission."""
import os
import time

import pytest

from gf_common.domain import ContractError
from gf_models.structured_tree_pairing import (
    TREE2_RECORDS,
    TREE3_GAME_RECORDS,
    TREE3_PACKED_GAME_RECORDS,
    TREE3_RECORDS,
)
from gf_training.structured_lineage import _ChainScan, _write
from gf_training.structured_tree_fit import _CompletionPublication


def primitive(root, context):
    root.mkdir()
    publisher = _ChainScan(directory=root, max_owned_bytes=1024**2, deadline=time.monotonic()+30)
    _write(root/'training.json', {'status': 'running', 'scope': 'UNADMITTED control only'}, scan=publisher)
    info = root.stat()
    publisher._tree_fit_directory_identity = info.st_dev, info.st_ino
    publisher._tree_fit_ancestors = {}
    owner = _CompletionPublication(root, publisher, time.monotonic()+35, 1024**2, _context=context)
    # Resource-only sentinel. This invalid record can never load as a fitted pair.
    owner._postcheck = lambda: None
    return owner, {'status': 'complete', 'scope': 'UNADMITTED no pair or original Sources'}


@pytest.mark.parametrize('context', [TREE2_RECORDS, TREE3_RECORDS, TREE3_GAME_RECORDS, TREE3_PACKED_GAME_RECORDS])
def test_tree_writer_closes_before_mutable_replace_for_every_exact_context(tmp_path, monkeypatch, context):
    owner, complete = primitive(tmp_path/'output', context)
    replace = os.replace
    def checked(source, target, **options):
        expected = os.stat(source, dir_fd=options['src_dir_fd'], follow_symlinks=False)
        for name in os.listdir('/proc/self/fd'):
            try:
                info = os.fstat(int(name))
            except OSError:
                continue
            assert (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino)
        return replace(source, target, **options)
    monkeypatch.setattr(os, 'replace', checked)
    try:
        owner.commit(complete)
        assert owner.committed and owner.fd is None and owner.directory_fd is None
    finally:
        owner.revoke()


def test_tree_retained_close_has_no_second_ctime_handoff(tmp_path, monkeypatch):
    owner, complete = primitive(tmp_path/'output', TREE3_PACKED_GAME_RECORDS)
    injected = []
    # Inject at the actual retained close through the existing owner release seam.
    release = owner._release
    def release_then_mutate():
        release()
        if not injected:
            path = owner.directory/'training.json'
            path.chmod(path.stat().st_mode)
            injected.append(True)
    monkeypatch.setattr(owner, '_release', release_then_mutate)
    with pytest.raises(ContractError):
        owner.commit(complete)
    assert injected and owner.revoked and not owner.committed
    assert (owner.directory/'training.json').read_bytes().startswith(b'!')
