"""HPO capability admission; structural support never grants Source authority."""
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig


def test_missing_hpo_capability_refuses_before_any_native_session():
    from gf_training.native_backend import hpo_reader_settings
    module = SimpleNamespace(build_info=lambda: [('card_instance_contract',
        'participant-structured-v2;journal6;rollout4;generation3;model3')])
    with pytest.raises(ContractError, match='HPO'):
        hpo_reader_settings(PreparationConfig(backend='rust', reader='participant-structured-v2'), module)
