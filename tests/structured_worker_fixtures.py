"""Importable protocol2 test factories; never close over parent controls."""
from __future__ import annotations

import os
import sys
import time

from gf_common.domain import ContractError
from gf_common.structured import Selection
from gf_models.structured_baselines import StructuredBaseline


def clean_probe(model_id, options, action_config_hash):
    assert not any(key.startswith("PRIVATE_") for key in os.environ)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert sys.argv == ["gf-structured-policy-worker"]
    assert "torch" not in sys.modules
    assert options is None and len(action_config_hash) == 64
    return StructuredBaseline(model_id)


class BadHash(StructuredBaseline):
    def select_frame(self, frame, bank, table):
        return Selection("0" * 64, 0, self.version)


def bad_hash(model_id, options, action_config_hash):
    return BadHash(model_id)


class Slow(StructuredBaseline):
    def select_frame(self, frame, bank, table):
        time.sleep(2)
        raise ContractError("unreachable after parent timeout")


def slow(model_id, options, action_config_hash):
    return Slow(model_id)
