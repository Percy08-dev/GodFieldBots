"""Fixture-only Dataset3 originals/recipe owner; no qualified-game fit authority."""
from __future__ import annotations

import math
import stat
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Self

from gf_common import card_instance_fixture as fixture
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_models.structured_sealed import SealedScan
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV3,
    TreeEncoderConfigV3,
    TreeResourcesV2,
)
from gf_models.structured_tree_encoder import PreparedTreeFrameV3, _owned
from gf_training.card_instance_tree_fixture import (
    FixtureSelectedInput,
    TreeFixtureSource,
    _ancestors,
)
from gf_training.structured_lineage import _ChainScan, _write
from gf_training.structured_tree_data import _check_output_ancestors, _outside_output

FIELDS = {'schema_version', 'kind', 'selection', 'label_recipe', 'origin', 'source', 'source_set_sha256', 'content_sha256'}
SELECTION = 'all-declared-component-selected-endpoints-v1'


def _manifest(source: TreeFixtureSource) -> Json:
    proof = source.verify_complete()
    value = {'schema_version': 3, 'kind': 'card-instance-fixture-fixed-sources',
        'selection': SELECTION, 'label_recipe': fixture.LABEL_RECIPE, 'origin': fixture.origin(),
        'source': proof, 'source_set_sha256': digest(proof)}
    return {**value, 'content_sha256': digest(value)}


def _consume(source: TreeFixtureSource) -> None:
    import numpy as np
    model = ConfiguredTreeModelV3(encoder=source.encoder, resources=TreeResourcesV2(max_owned_bytes=source.limit))
    for sample in source.selected_inputs():
        prepared = PreparedTreeFrameV3.create(sample.tokens, model, extra_owned_bytes=source.retained_bytes)
        blocks = prepared.iter_chunks(sample.selected_tokens())
        _, row = next(blocks)
        if row.shape != (1, 2810) or row.dtype != np.float32 or not np.isfinite(row).all():
            raise ContractError('fixture selected Tree3 row shape/dtype/finite mismatch')
        try:
            next(blocks)
        except StopIteration:
            pass
        else:
            raise ContractError('fixture selected projection must contain exactly one row')
        del prepared, blocks, row, sample


def build_fixture_dataset(path: Path, expected_digest: str, output: Path, settings: PreparationConfig, *,
                          max_owned_bytes: int, deadline: float, encoder: TreeEncoderConfigV3 | None = None,
                          max_scratch_bytes: int | None = None) -> Json:
    """Publish recipe-only fixture Dataset3 while retaining the SAME originals."""
    started = time.monotonic()
    settings = preparation_config(asdict(settings))
    scratch = settings.scratch_budget_mib*1024**2 if max_scratch_bytes is None else integer(
        max_scratch_bytes, 1, settings.scratch_budget_mib*1024**2)
    with TreeFixtureSource(path, expected_digest, settings, max_owned_bytes=max_owned_bytes,
                           deadline=deadline, encoder=encoder) as source:
        output, ancestors = _outside_output(output, list(source.original_roots))
        _check_output_ancestors(ancestors)
        _consume(source)
        manifest = _manifest(source)
        receipt = {'schema_version': 1, 'kind': 'card-instance-fixture-dataset-preparation',
            'dataset_content_sha256': manifest['content_sha256'], 'seconds': time.monotonic()-started,
            'scope': 'actual fixture inspection and selected-row preparation before publication; excludes PPO/games/fitting'}
        payload = sum(len(canonical(value).encode())+1 for value in (manifest, receipt))
        if payload > scratch or any(len(canonical(value).encode())+1 > min(64, settings.max_source_file_mib)*1024**2
                                    for value in (manifest, receipt)):
            raise ContractError('fixture Dataset3 output scratch/file allowance exceeded')
        source.reserve_sink(32*payload + 131072 + _owned((manifest, receipt)))
        remaining = source.limit-source.retained_bytes
        if remaining <= 0:
            raise ContractError('fixture Dataset3 joint publication budget exhausted')
        scan = _ChainScan(directory=output, max_owned_bytes=remaining, deadline=source.deadline)
        source.guard(); _check_output_ancestors(ancestors)
        output.mkdir(mode=0o700, exist_ok=False)
        _write(output/'manifest.json', manifest, scan=scan, immutable=True)
        _write(output/'preparation.json', receipt, scan=scan, immutable=True)
        source.verify_complete(); scan.finalize(); _check_output_ancestors(ancestors)
        source.finalize(); source.close()  # All reportable original guards precede return/publication completion.
        return manifest


class VerifiedFixtureTreeDataset3:
    """Active fixture-only Dataset3 index and original Source, retained through preparation."""

    def __init__(self, path: Path, expected_digest: str, settings: PreparationConfig, *,
                 max_owned_bytes: int, deadline: float) -> None:
        self.path = Path(path).absolute(); self.expected_digest = fixture.sha256(expected_digest)
        self.settings = preparation_config(asdict(settings)); self.limit = max_owned_bytes; self.deadline = deadline
        integer(max_owned_bytes, 1, 12*1024**3)
        if (type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0
                or self.settings.backend != 'python' or self.settings.reader != 'participant-structured-v2'):
            raise ContractError('fixture Dataset3 requires finite deadline and corrected serial Python reader')
        self.deadline = min(deadline, time.monotonic()+self.settings.wall_seconds-self.settings.cleanup_seconds)
        self._manifest: Json = {}; self._index: SealedScan | None = None; self._source: TreeFixtureSource | None = None
        self._ancestors: dict[Path, tuple[int, int]] = {}
        self._active = self._completed = self._finalized = self._closed = False

    def _live(self) -> TreeFixtureSource:
        if not self._active or self._closed or self._source is None:
            raise ContractError('active concrete VerifiedFixtureTreeDataset3 required')
        return self._source

    @property
    def manifest(self) -> Json:
        self._live()
        return clone(self._manifest)

    @property
    def source(self) -> TreeFixtureSource:
        """The SAME active originals owner, never reconstructed from a summary."""
        return self._live()

    def __enter__(self) -> Self:
        if self._active or self._closed:
            raise ContractError('fixture Dataset3 lifetime cannot be reopened')
        self._index = SealedScan(max_owned_bytes=min(self.limit, self.settings.memory_budget_mib*1024**2), deadline=self.deadline)
        self._ancestors = _ancestors(self.path.parent)
        self._index.inventory(self.path.parent, {self.path.name, 'preparation.json'})
        for path, cap in ((self.path, self.settings.max_source_file_mib),
                          (self.path.parent/'preparation.json', self.settings.max_record_mib)):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= min(64, cap)*1024**2:
                raise ContractError('fixture Dataset3 index file/record limit exceeded')
        self._manifest = self._index.object(self.path, self.expected_digest)
        raw = closed(self._manifest, FIELDS)
        if (type(raw['schema_version']) is not int or raw['schema_version'] != 3
                or raw['kind'] != 'card-instance-fixture-fixed-sources' or raw['selection'] != SELECTION
                or raw['label_recipe'] != fixture.LABEL_RECIPE or canonical(raw['origin']) != canonical(fixture.origin())
                or digest(raw['source']) != raw['source_set_sha256']
                or digest({k: v for k, v in raw.items() if k != 'content_sha256'}) != raw['content_sha256']):
            raise ContractError('fixture-only Dataset3 kind/origin/content mismatch')
        receipt = self._index.object(self.path.parent/'preparation.json')
        closed(receipt, {'schema_version', 'kind', 'dataset_content_sha256', 'seconds', 'scope'})
        if (type(receipt['schema_version']) is not int or receipt['schema_version'] != 1
                or receipt['kind'] != 'card-instance-fixture-dataset-preparation'
                or receipt['dataset_content_sha256'] != raw['content_sha256']
                or type(receipt['seconds']) not in (int, float) or not math.isfinite(receipt['seconds'])
                or receipt['seconds'] < 0 or type(receipt['scope']) is not str or not receipt['scope']):
            raise ContractError('fixture Dataset3 measured preparation receipt mismatch')
        source = raw['source']
        encoder = TreeEncoderConfigV3(**closed(source['tree_projection']['encoder'], set(TreeEncoderConfigV3.__dataclass_fields__)))
        self._source = TreeFixtureSource(Path(source['source_directory'])/'manifest.json', source['manifest_sha256'], self.settings,
            max_owned_bytes=self.limit, deadline=self.deadline, encoder=encoder, extra_owned_bytes=self._index.retained)
        try:
            self._source.__enter__()
            if self.path.is_relative_to(self._source.path.parent) or any(self.path.is_relative_to(root) for root in self._source.original_roots):
                raise ContractError('fixture Dataset3 must remain outside all originals')
            self._active = True
            return self
        except BaseException as error:
            if not self._source._closed:
                try:
                    self._source.close(); self._source.release()
                except BaseException as cleanup:  # noqa: BLE001 - retain reportable original cleanup owner
                    error.add_note('fixture Dataset3 failed entry cleanup: '+str(cleanup))
                    vars(error)['_fixture_dataset_cleanup_owner'] = self
            raise

    def selected_inputs(self) -> Iterator[tuple[str, FixtureSelectedInput]]:
        source = self._live()
        if self._completed:
            raise ContractError('one verified fixture Dataset3 worklist required')
        for sample in source.selected_inputs():
            yield sample.split, sample
            self._live()
            del sample
        self._completed = True
        self.verify_complete()

    def reserve_sink(self, total_bytes: int) -> None:
        source = self._live()
        integer(total_bytes, 0, 2**63-1)
        # The fixture Source uses cumulative admission: keep its peak reservation
        # when a fit releases packing scratch, rather than relaxing original limits.
        source.reserve_sink(max(total_bytes, source._sink))

    def retained_without_sink(self) -> int:
        return self._live().retained_without_sink()

    def verify_complete(self) -> None:
        source = self._live()
        if not self._completed:
            raise ContractError('complete fixture Dataset3 original selected worklist required')
        if canonical(_manifest(source)) != canonical(self._manifest):
            raise ContractError('fixture Dataset3 originals/recipe/full membership changed')
        assert self._index is not None
        _check_output_ancestors(self._ancestors)
        self._index.finalize()

    def finalize(self) -> None:
        self.verify_complete()
        self._live().finalize()
        self._finalized = True

    def close(self) -> None:
        if self._closed:
            return
        source = self._live()
        assert self._index is not None
        _check_output_ancestors(self._ancestors)
        self._index.finalize(); source.close(); self._index.finalize()
        _check_output_ancestors(self._ancestors)
        self._closed = True; self._active = False

    def release(self) -> None:
        if not self._closed:
            raise ContractError('fixture Dataset3 reportable close must precede memory-only release')
        if self._source is not None:
            self._source.release()
        if self._index is not None:
            self._index.files.clear(); self._index.inventories.clear()
        self._manifest.clear()

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        failure = value if isinstance(value, BaseException) else None
        try:
            if failure is None and not self._closed:
                self.finalize()
        except BaseException as error:  # noqa: BLE001 - preserve initiating failure through original guards
            failure = error
        try:
            self.close()
        except BaseException as error:  # noqa: BLE001 - preserve initiating failure through original close
            if failure is None:
                failure = error
            else:
                failure.add_note('fixture Dataset3 close: '+str(error))
        if self._closed:
            self.release()
        elif failure is not None:
            vars(failure)['_fixture_dataset_cleanup_owner'] = self
        if value is None and failure is not None:
            raise failure


def inspect_fixture_dataset(path: Path, *, expected_digest: str, settings: PreparationConfig,
                            max_owned_bytes: int, deadline: float) -> Json:
    with VerifiedFixtureTreeDataset3(path, expected_digest, settings, max_owned_bytes=max_owned_bytes,
                                     deadline=deadline) as owner:
        for _, sample in owner.selected_inputs():
            del sample
        return owner.manifest
