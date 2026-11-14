"""Native reader support is explicit; private builds never replace installed ELF."""
from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from tests.test_hpo_native import candidate  # noqa: F401 - private /tmp loader fixture
from tests.test_refresh_runtime import refresh_journal


@pytest.mark.parametrize('marker', [None, '', 'baseline1;options1', 'baseline2;options1;identity1'])
def test_native_refresh_refuses_old_or_mismatched_capability(marker):
    from gf_training.native_backend import require_native_refresh_baseline
    module = SimpleNamespace(build_info=lambda: [] if marker is None else [('card_instance_refresh', marker)])
    with pytest.raises(ContractError, match='refresh'):
        require_native_refresh_baseline(module)


def test_native_refresh_accepts_exact_additive_marker():
    from gf_training.native_backend import require_native_refresh_baseline
    require_native_refresh_baseline(SimpleNamespace(build_info=lambda: [('card_instance_refresh', 'baseline1;options1;identity1')]))


@pytest.mark.parametrize('threshold', [12, 2**130])
@pytest.mark.parametrize('mutation', [None, 'missing', 'threshold', 'label', 'legacy_key'])
def test_private_native_reads_refresh_journal_and_rejects_tampering(candidate, tmp_path, mutation, threshold):  # noqa: F811
    from gf_training.native_backend import require_native_refresh_baseline
    require_native_refresh_baseline(candidate)
    journal = refresh_journal(tmp_path, threshold)
    journal.finish('incomplete', None)
    rows = [json.loads(line) for line in journal.final_path.read_text().splitlines()]
    if mutation is not None:
        header = copy.deepcopy(rows[0])
        if mutation == 'missing':
            del header['baseline_options']
        elif mutation == 'threshold':
            header['baseline_options']['attack_threshold'] += 1
        elif mutation == 'label':
            header['policy'] += '0'
        else:
            header.update(model_id='heuristic', policy='heuristic-card-instance-v2')
        rows[0] = header
        journal.final_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    settings = json.dumps({'memory_bytes': 512*1024**2, 'max_json_bytes': 64*1024**2, 'max_record_bytes': 8*1024**2}).encode()
    reader = None
    try:
        if mutation is None:
            reader = candidate.StructuredJournalTokens.card_instance(str(tmp_path), journal.final_path.name, settings, time.monotonic() + 30)
            result = json.loads(reader.diagnostic())
            assert result['header']['model_id'] == 'heuristic_refresh'
            assert result['header']['baseline_options']['attack_threshold'] == threshold
        else:
            with pytest.raises((ValueError, RuntimeError)):
                reader = candidate.StructuredJournalTokens.card_instance(str(tmp_path), journal.final_path.name, settings, time.monotonic() + 30)
                reader.diagnostic()
    finally:
        if reader is not None:
            reader.close()
