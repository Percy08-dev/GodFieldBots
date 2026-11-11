"""Actual legacy Dataset exit over a real FD cleanup primitive, without fit inputs."""
import asyncio
import os
import time

import pytest

from gf_common.preprocessing import PreparationConfig
from gf_models.config import TrainingConfig
from gf_models.structured_tree_config import ConfiguredTreeModelV2
from gf_training.structured_tree_data import VerifiedTreeDataset
from gf_training.structured_tree_fit import _TreeFitRun


def legacy_primitive(tmp_path, phase):
    first = OSError('legacy exit '+str(phase)+' failure')
    class Index:
        def finalize(self):
            pass
    class Source:
        def __init__(self):
            self.fd = os.open('/dev/null', os.O_RDONLY)
            self._closed = False
            self.exits = 0
        def close(self):
            if not self._closed:
                fd, self.fd = self.fd, None
                os.close(fd)
                self._closed = True
        def __exit__(self, *args):
            self.exits += 1
            if phase == 'before':
                raise first
            self.close()
            if phase == 'after':
                raise first
    source = Source()
    dataset = VerifiedTreeDataset.__new__(VerifiedTreeDataset)
    dataset.source, dataset.index = source, Index()
    dataset.manifest = {'scope': 'UNADMITTED no original Source or row inputs'}
    dataset._active, dataset._completed, dataset._finalized = True, False, True
    dataset.deadline = time.monotonic()+30
    dataset.settings = PreparationConfig()
    actual_exit = dataset.__exit__
    def exit_and_clear(*args):
        try:
            return actual_exit(*args)
        finally:
            dataset.source = None
    dataset.__exit__ = exit_and_clear
    owner = _TreeFitRun(dataset, tmp_path, ConfiguredTreeModelV2(), TrainingConfig(), 17)
    return owner, source, first


def test_legacy_normal_exit_confirms_actual_retained_source_once(tmp_path):
    owner, source, _ = legacy_primitive(tmp_path, None)
    try:
        owner._release_dataset(via_exit=True)
        assert owner._dataset_release_complete and source._closed and source.fd is None
        assert owner._dataset_release_source is source
        owner._release_dataset(via_exit=True)
        assert source.exits == 1
    finally:
        source.close()


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_legacy_exit_failure_never_retries_or_infers_success_from_cleared_source(tmp_path, phase):
    owner, source, first = legacy_primitive(tmp_path, phase)
    try:
        with pytest.raises(OSError) as error:
            owner._release_dataset(via_exit=True)
        assert error.value is first and owner._dataset_release_source is source
        assert not owner._dataset_release_complete and not owner.closed
        assert owner.dataset.source is None
        with pytest.raises(OSError) as repeated:
            owner._release_dataset(via_exit=True)
        assert repeated.value is first and source.exits == 1
        with pytest.raises(OSError) as closed:
            asyncio.run(owner.close())
        assert closed.value is first and not owner.closed and source.exits == 1
        assert source._closed  # Independent certain physical cleanup may finish.
    finally:
        source.close()
