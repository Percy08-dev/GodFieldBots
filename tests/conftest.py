from __future__ import annotations

import pytest

from gf_common.validation import catalog
from tests.input_support import fixture as fixture  # noqa: PLC0414 - compatibility re-export
from tests.input_support import input_factory
from tests.input_support import public_options as public_options  # noqa: PLC0414 - compatibility re-export


@pytest.fixture
def definitions():
    return catalog(fixture("catalog_public_v2.json"))


@pytest.fixture
def make_input(definitions):
    return input_factory(definitions)


@pytest.fixture(scope="session")
def _ppo_source_template(tmp_path_factory):
    pytest.importorskip("torch")
    from tests.shared_ppo_sources import generate_sources
    return generate_sources(tmp_path_factory.mktemp("shared-ppo-sources"))


@pytest.fixture
def frozen_sources(tmp_path, _ppo_source_template):
    return _ppo_source_template.clone(tmp_path / "frozen-sources")
