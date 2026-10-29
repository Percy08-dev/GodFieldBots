"""Strict source flags and pre-publication Python control-memory admission."""
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_common.domain import ContractError, digest
from gf_training.native_authorization import NativePreparation
from gf_training.ppo import ingest_collection
from gf_training.ppo_run import train_ppo
from gf_training.preparation import ResourceLimit
from tests.test_native_ppo import native_configuration, online_context
from tests.test_ppo_lineage import synthetic_collector

pytest.importorskip('gf_native')


class Checked(RuntimeError):
    pass


@pytest.mark.parametrize('flag', [0, 1, 0.0, 1.0])
def test_source_mode_rejects_numeric_alias_with_coherent_hashes(tmp_path, make_input, flag):
    pytest.importorskip('torch')
    loaded = native_configuration(tmp_path, max_updates=1)
    collect = synthetic_collector(make_input, both_viewers=True)

    async def corrupt(plan):
        await collect(plan)
        path = plan.output / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['effective_config']['actions']['use_action_options'] = flag
        manifest['config_sha256'] = digest(manifest['effective_config'])
        action_hash = digest(manifest['effective_config']['actions'])
        for journal in plan.output.glob('*/player-*.jsonl'):
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            records[0].update(use_action_options=bool(flag), action_config_hash=action_hash)
            for record in records:
                if record['kind'] == 'decision':
                    record['candidates']['config_hash'] = action_hash
                    if not flag:
                        record['input']['action_options'] = None
            journal.write_text(''.join(json.dumps(record) + '\n' for record in records))
        path.write_text(json.dumps(manifest))
        assert digest(manifest['effective_config']) == manifest['config_sha256']
        with pytest.raises(ContractError, match='ActionOptions'):
            ingest_collection(plan.output, loaded.config.model, plan.linkage,
                              deadline=plan.execution_deadline, online_context=online_context(plan))
        owner = await NativePreparation.create_async(
            {'kind': 'online', 'encoder': asdict(loaded.config.model.encoder),
             'parent_path': str(Path(plan.linkage['parent_run_path']).resolve()),
             'collection_path': str(plan.output.resolve())}, loaded.config.preprocessing,
            deadline=plan.execution_deadline, online_context=online_context(plan))
        async with owner:
            with pytest.raises(ContractError, match='boolean'):
                await owner.anext_batch(0, (0, 0, 0, 0))
            await owner.aabort()
        raise Checked('checked before coordinator ingestion or optimizer')

    with pytest.raises(Checked):
        asyncio.run(train_ppo(loaded, corrupt, tmp_path / 'parent-"\\雪'))


def test_authorization_underestimate_rejects_without_widening():
    owner = NativePreparation.__new__(NativePreparation)
    owner.summary = {'authorization_wire_bound_bytes': 8}
    with pytest.raises(ResourceLimit, match='pre-admitted bound'):
        owner._authorization_bytes({'long': 'authorization'})
    assert owner.summary['authorization_wire_bound_bytes'] == 8


def test_large_parent_controls_reject_before_python_summary_decode(tmp_path, make_input):
    pytest.importorskip('torch')
    loaded = native_configuration(tmp_path, max_updates=1)
    collect = synthetic_collector(make_input, both_viewers=True)

    async def oversized(plan):
        await collect(plan)
        parent = Path(plan.linkage['parent_run_path'])
        record_path = parent / 'generations' / '000000.json'
        record = json.loads(record_path.read_text())
        record['parent_controls']['unused_padding'] = 'x' * (2 * 1024 * 1024)
        record_path.write_text(json.dumps(record))
        checksum = hashlib.file_digest(record_path.open('rb'), 'sha256').hexdigest()
        latest = parent / 'latest.json'
        latest.write_text(json.dumps({'generation': 0, 'sha256': checksum}))
        state_path = parent / 'run.json'
        state = json.loads(state_path.read_text())
        state['generation_sha256'] = checksum
        state_path.write_text(json.dumps(state))
        with pytest.raises(ResourceLimit, match='memory budget'):
            await NativePreparation.create_async(
                {'kind': 'online', 'encoder': asdict(loaded.config.model.encoder),
                 'parent_path': str(parent.resolve()), 'collection_path': str(plan.output.resolve())},
                loaded.config.preprocessing, deadline=plan.execution_deadline,
                online_context=online_context(plan))
        raise Checked('large controls rejected before Python summary publication')

    with pytest.raises(Checked):
        asyncio.run(train_ppo(loaded, oversized, tmp_path / 'parent'))
