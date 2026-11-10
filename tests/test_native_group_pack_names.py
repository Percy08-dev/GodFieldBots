from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pytest

from gf_common.domain import ContractError


def test_frozen_unicode_c_ranges_match_every_supported_python_codepoint():
    path = Path(__file__).parents[1] / 'native/src/logical_name_ranges.rs'
    source = path.read_text()
    assert 'Unicode16.0.0' in source
    rows = [(int(a), int(b)) for a, b in re.findall(r'\((\d+), (\d+)\)', source)]
    assert len(rows) == 736
    serialized = json.dumps({'category': 'C', 'ranges': rows, 'unicode_version': '16.0.0'},
        sort_keys=True, separators=(',', ':')).encode() + b'\n'
    assert hashlib.sha256(serialized).hexdigest() == '3bc5dc3aa584949c7a8f7918872e7dd25ddb4c4df2bb3cbaa44c6a1f4c2632d9'
    assert unicodedata.unidata_version == '16.0.0'
    index = 0
    for value in range(0x110000):
        while index < len(rows) and rows[index][1] < value:
            index += 1
        actual = index < len(rows) and rows[index][0] <= value <= rows[index][1]
        assert actual == unicodedata.category(chr(value)).startswith('C'), hex(value)


def test_ucd_mismatch_refuses_before_any_provider_ownership(monkeypatch):
    from gf_training.native_group_pack import OriginalArtifactProvider
    monkeypatch.setattr(unicodedata, 'unidata_version', 'future')
    with pytest.raises(ContractError, match='Unicode16'):
        OriginalArtifactProvider((object(),))
