"""Current catalog pin changes semantics without reinterpreting historical bytes."""
import asyncio
import copy

import pytest

from gf_common.domain import CATALOG_META, ContractError, UnsupportedContract, digest
from gf_common.validation import catalog
from gf_models.config import EncoderConfig
from gf_models.encoder import signature
from tests.conftest import fixture
from tests.test_http import client_for, info, response


def test_v2_exact_public_contract_and_historical_v1_is_unchanged_and_ineligible():
    old, current = fixture('catalog_public.json'), fixture('catalog_public_v2.json')
    assert old['catalog_version'] == 'public-ja-v1'
    assert old['content_sha256'] == '4bb3ac87e4567798767f7c74a9f9280208adf931277c6afc7deeb865ef4d0673'
    assert digest({key: value for key, value in old.items() if key != 'content_sha256'}) == old['content_sha256']
    assert current['catalog_version'] == 'public-ja-v2'
    assert current['content_sha256'] == CATALOG_META['content_sha256'] == '481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d'
    assert len(catalog(current)) == 296
    changes = [(left, right) for left, right in zip(old['definitions'], current['definitions'], strict=True) if left != right]
    assert len(changes) == 1 and changes[0][0]['definition_id'] == 'magical-stick'
    left, right = (copy.deepcopy(item) for item in changes[0])
    assert right.pop('effect_text') == '併用する奇跡の実支払MPを除いた残りのMPの2倍を攻撃値とし、MPをすべて消費する'
    left.pop('effect_text')
    assert left == right
    with pytest.raises(UnsupportedContract, match='catalog metadata'):
        catalog(old)
    relabeled = copy.deepcopy(old)
    relabeled.update(CATALOG_META)
    with pytest.raises(ContractError, match='catalog hash'):
        catalog(relabeled)


@pytest.mark.parametrize('bad_info,bad_catalog', [(True, False), (False, True), (False, False)])
def test_http_discovers_pinned_v2_endpoint_and_rejects_old_info_or_payload(bad_info, bad_catalog):
    async def run():
        gate = info()
        paths = []
        def handler(request):
            paths.append(request.url.path)
            if request.url.path == '/debug/v1/test-server/info':
                return response(gate)
            if request.url.path == '/v1/info':
                return response(fixture('info_public.json' if bad_info else 'info_public_v2.json'))
            assert request.url.path == '/v1/card-catalogs/public-ja-v2'
            return response(fixture('catalog_public.json' if bad_catalog else 'catalog_public_v2.json'))
        client = client_for(handler, gate)
        try:
            await client.attest()
            if bad_info or bad_catalog:
                with pytest.raises(UnsupportedContract):
                    await client.public_catalog()
            else:
                assert len(await client.public_catalog()) == 296
            assert '/v1/card-catalogs/public-ja-v1' not in paths
            assert paths == ['/debug/v1/test-server/info', '/v1/info'] + ([] if bad_info else ['/v1/card-catalogs/public-ja-v2'])
        finally:
            await client.close()
    asyncio.run(run())


def test_encoder_signature_preserves_both_pinned_feature_recipes():
    # Pinned encoder-v1 feature identities; catalog text is not a feature.
    assert signature(EncoderConfig()) == 'b6045e464c1488807c76fc5a298008e692bca603e922b7d604e0e023b2c2de14'
    assert signature(EncoderConfig(history="public_aggregate")) == 'befca464acc87b893d1b5fadf47db3bbebbd93dfa206545a216f9fa312d3dedf'


def test_catalog_text_correction_preserves_numeric_perceived_features(make_input):
    from dataclasses import replace

    import numpy as np

    from gf_models.encoder import encode
    value = make_input(('magical-stick',))
    old_definition = next(card for card in fixture('catalog_public.json')['definitions'] if card['definition_id'] == 'magical-stick')
    old_text = replace(value, catalog={'magical-stick': old_definition})
    action = {'type': 'play_sequence', 'ordered_elements': [{'source': 'hand_artifact', 'instance_id': 1}], 'target': 2}
    for config in (EncoderConfig(), EncoderConfig(history='public_aggregate')):
        np.testing.assert_array_equal(encode(value, action, config), encode(old_text, action, config))
    value.observation['own']['status']['dream'] = True
    # The policy still sees only its own perceived definition; no actual identity is inferred.
    assert set(value.catalog) == {'magical-stick'}
    assert 'attack' not in value.catalog['magical-stick']
