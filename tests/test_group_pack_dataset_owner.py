"""Concrete Dataset4 refusal tests, without manufactured eligible parents."""
import hashlib
import time

import pytest

from gf_common.domain import ContractError, canonical
from gf_common.preprocessing import PreparationConfig
from gf_models.structured_tree_config import ConfiguredTreeModelV3
from gf_models.structured_tree_pairing import TREE3_GAME_RECORDS, TREE3_PACKED_GAME_RECORDS
from gf_training.card_instance_game_tree_data import (
    VerifiedGameTreeDataset3,
    VerifiedGameTreeDataset4,
)
from gf_training.structured_tree_fit import _PreparedTreeFit
from tests.test_game_tree_dataset import dto
from tests.test_game_tree_records import request
from tests.test_group_pack_dataset_contract import packed_dto


@pytest.mark.parametrize('reader,value', [(VerifiedGameTreeDataset4, dto), (VerifiedGameTreeDataset3, packed_dto),
    (VerifiedGameTreeDataset4, packed_dto)])
def test_unadmitted_recipe_indexes_never_supply_actual_original_sources(tmp_path, reader, value):
    raw = value()
    path = tmp_path / 'manifest.json'
    encoded = (canonical(raw)+'\n').encode()
    path.write_bytes(encoded)
    (tmp_path / 'preparation.json').write_text(canonical({'schema_version': 1,
        'kind': 'card-instance-game-dataset-preparation', 'dataset_content_sha256': raw['content_sha256'],
        'seconds': 0., 'scope': 'actual original inspection and selected Tree projection; excludes shared PPO/final-validation generation and fitting'})+'\n')
    with pytest.raises(ContractError), reader(path, hashlib.sha256(encoded).hexdigest(), PreparationConfig(reader='participant-structured-v2'),
        deadline=time.monotonic()+30):
        pytest.fail('unadmitted recipe index granted Source permission')


def test_packed_fit_requires_exact_dataset4_and_old_game_context_requires_exact_dataset3():
    model = ConfiguredTreeModelV3()
    fit = request()['fit_settings']
    class Arbitrary(VerifiedGameTreeDataset4):
        pass
    for dataset, context in [(object.__new__(VerifiedGameTreeDataset3), TREE3_PACKED_GAME_RECORDS),
        (object.__new__(VerifiedGameTreeDataset4), TREE3_GAME_RECORDS),
        (object.__new__(Arbitrary), TREE3_PACKED_GAME_RECORDS)]:
        with pytest.raises(ContractError, match='concrete'):
            _PreparedTreeFit(dataset, model, fit, _context=context)


def test_schema_selector_preserves_source_file_cap_before_metadata_read(tmp_path, monkeypatch):
    from gf_training import card_instance_game_tree_data as module
    path = tmp_path/'manifest.json'
    path.write_bytes(b' '*(2*1024**2))
    def forbidden(*args, **kwargs):
        pytest.fail('schema selector parsed metadata beyond configured source-file cap')
    monkeypatch.setattr(module.SealedScan, 'object', forbidden)
    settings = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=8192,
        max_source_file_mib=1, max_record_mib=1)
    with pytest.raises(ContractError):
        module.inspect_game_tree_dataset(path, expected_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
            settings=settings, deadline=time.monotonic()+30)


def test_selector_routes_only_integer_schema_and_leaves_full_body_validation_to_reader(tmp_path, monkeypatch):
    from gf_training import card_instance_game_tree_data as module
    raw = b'{"schema_version":4,"scope":"UNADMITTED routing only"}\n'
    path = tmp_path/'manifest.json'
    path.write_bytes(raw)
    settings = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=8192)
    monkeypatch.setattr(module.game, 'packed_manifest', lambda *args: pytest.fail('full validator clone in selector'))
    result = module.verified_game_dataset_type(path, hashlib.sha256(raw).hexdigest(), settings,
        max_owned_bytes=8*1024**3, deadline=time.monotonic()+30)
    assert result is VerifiedGameTreeDataset4


def test_inspect_routing_delay_cannot_restart_original_preprocessing_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gf_training import card_instance_game_tree_data as module
    clock = [1000.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    settings = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=8192,
        wall_seconds=100, cleanup_seconds=5)
    seen = []
    def select(path, expected, actual_settings, *, max_owned_bytes, deadline):
        seen.append(deadline)
        assert actual_settings is settings
        clock[0] += 20
        return VerifiedGameTreeDataset3
    class StopBeforeSources(RuntimeError):
        pass
    def entering(dataset):
        seen.append(dataset.deadline)
        raise StopBeforeSources()
    monkeypatch.setattr(module, 'verified_game_dataset_type', select)
    monkeypatch.setattr(VerifiedGameTreeDataset3, '__enter__', entering)
    with pytest.raises(StopBeforeSources):
        module.inspect_game_tree_dataset(tmp_path/'manifest.json', expected_digest='a'*64,
            settings=settings, deadline=2000.)
    assert seen == [1095.,1095.]


@pytest.mark.parametrize('dataset_type', [VerifiedGameTreeDataset3, VerifiedGameTreeDataset4])
def test_fit_routing_delay_keeps_original_training_and_preprocessing_deadline(tmp_path, monkeypatch, dataset_type):
    import asyncio
    import time as clock_module

    from gf_models.config import TrainingConfig
    from gf_training import card_instance_game_tree_data as module
    from gf_training import structured_tree_fit as fitting
    clock = [1000.]
    monkeypatch.setattr(clock_module, 'monotonic', lambda: clock[0])
    settings = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=8192,
        wall_seconds=100, cleanup_seconds=5)
    seen = []
    def select(path, expected, actual_settings, *, max_owned_bytes, deadline):
        assert actual_settings is settings
        seen.append(deadline)
        clock[0] += 20
        return dataset_type
    async def stop_before_sources(fit):
        assert not fit.dataset._active and fit.dataset._source is None
        return {'scope': 'UNADMITTED routing only', 'deadlines': [fit.deadline,fit.dataset.deadline], 'started': fit.started}
    monkeypatch.setattr(module, 'verified_game_dataset_type', select)
    monkeypatch.setattr(fitting._TreeFitRun, 'execute', stop_before_sources)
    result = asyncio.run(fitting.fit_game_dataset(tmp_path/'manifest.json', expected_digest='a'*64,
        directory=tmp_path/'output', model=ConfiguredTreeModelV3(), settings=settings,
        training=TrainingConfig(), seed=17, deadline=2000.))
    assert seen == [1095.] and result['deadlines'] == [1095.,1095.] and result['started'] == 1000.
    assert not (tmp_path/'output').exists()


def test_build_parent_selector_consumes_original_absolute_preparation_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gf_models.structured_tree_config import TreeEncoderConfigV3
    from gf_training import card_instance_game_tree_data as module
    clock = [1000.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    settings = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=8192,
        wall_seconds=100, cleanup_seconds=5)
    parents = tuple(tmp_path/f'p{i}' for i in range(3))
    validations = tuple(tmp_path/f'v{i}' for i in range(3))
    for path in (*parents,*validations):
        path.mkdir()
    seen = []
    def select(paths, *, max_owned_bytes, deadline):
        seen.append(deadline)
        clock[0] += 20
        return False
    class StopBeforeSources(RuntimeError):
        pass
    def entering(source):
        seen.append(source.deadline)
        raise StopBeforeSources()
    monkeypatch.setattr(module, '_packed_parent_variant', select)
    monkeypatch.setattr(module._GameTreeOriginals, '__enter__', entering)
    with pytest.raises(StopBeforeSources):
        module.build_game_tree_dataset(parents,validations,tmp_path/'output',settings,
            encoder=TreeEncoderConfigV3(), deadline=2000.)
    assert seen == [1095.,1095.] and not (tmp_path/'output').exists()
