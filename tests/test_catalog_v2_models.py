"""Old gameplay/catalog identities fail without changing inference JSON schemas."""
import json
from dataclasses import asdict

import pytest

from gf_common.domain import ActionConfig, ContractError, canonical, digest
from gf_models.config import MLPModelConfig
from tests.conftest import fixture
from tests.test_lightgbm_native import toy_artifact as toy_artifact  # noqa: PLC0414 - pytest fixture export
from tests.test_lightgbm_native import toy_data as toy_data  # noqa: PLC0414 - pytest fixture export

OLD_REVISION = '860d8b92db4ce82012e040510f221b52241f28aa'


@pytest.mark.parametrize('identity', ['revision', 'catalog'])
def test_old_lightgbm_schema2_identity_rejects_before_native_loading(toy_artifact, monkeypatch, identity):
    import gf_models.lightgbm.model as adapter
    path, _, _, _ = toy_artifact
    manifest = json.loads((path / 'inference.json').read_text())
    assert manifest['schema_version'] == 2
    fields = set(manifest)
    if identity == 'revision':
        manifest['simulator_contract_revision'] = OLD_REVISION
    else:
        manifest['catalog'] = fixture('info_public.json')['card_catalog']
    (path / 'inference.json').write_text(canonical(manifest))
    monkeypatch.setattr(adapter, 'runtime', lambda: pytest.fail('old identity must reject before native load'))
    with pytest.raises(ContractError, match='semantics'):
        adapter.load_artifact(path, allow_toy=True)
    assert set(manifest) == fields


@pytest.mark.parametrize('identity', ['revision', 'catalog'])
def test_old_mlp_schema1_identity_rejects_before_native_loading(tmp_path, monkeypatch, identity):
    pytest.importorskip('torch')
    import gf_models.mlp.model as adapter
    config = MLPModelConfig()
    path = tmp_path / 'component-artifact'
    adapter.save_artifact(adapter.make_network(config, 17), path, config, digest(asdict(ActionConfig())))
    manifest = json.loads((path / 'inference.json').read_text())
    assert manifest['schema_version'] == 1
    fields = set(manifest)
    if identity == 'revision':
        manifest['simulator_contract_revision'] = OLD_REVISION
    else:
        manifest['catalog'] = fixture('info_public.json')['card_catalog']
    (path / 'inference.json').write_text(canonical(manifest))
    monkeypatch.setattr(adapter, 'make_network', lambda *args: pytest.fail('old identity must reject before network load'))
    with pytest.raises(ContractError, match='semantics'):
        adapter.load_artifact(path)
    assert set(manifest) == fields
