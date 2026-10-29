"""Header identity drives participant selection and canonical Replay lookup."""
import asyncio
import json

import pytest

from gf_common.domain import ContractError
from gf_training.lineage import read_json
from gf_training.ppo import ingest_collection
from gf_training.ppo_run import train_ppo
from tests.test_native_ppo import native_configuration, online_context
from tests.test_ppo_lineage import synthetic_collector

pytest.importorskip('gf_native')


@pytest.mark.parametrize('change', ['unknown_episode', 'swap_journals', 'swap_both', 'swap_directories'])
def test_source_header_episode_and_viewer_reconciliation(tmp_path, make_input, monkeypatch, change):
    pytest.importorskip('torch')
    import gf_training.ppo_run as coordinator

    loaded = native_configuration(tmp_path, max_updates=1)
    collector = synthetic_collector(make_input, both_viewers=True)
    updates = []
    request = coordinator.LearningProcess.request

    async def watch_request(self, message, deadline):
        if message['command'] == 'update':
            updates.append(message)
        return await request(self, message, deadline)

    monkeypatch.setattr(coordinator.LearningProcess, 'request', watch_request)

    async def corrupt(plan):
        result = await collector(plan)
        episode = plan.groups[-1].episodes[0]  # Draw, same generation on both seats.
        assert episode.policies[0] == episode.policies[1]
        directory = plan.output / episode.episode_id
        first, second = directory / 'player-1.jsonl', directory / 'player-2.jsonl'
        if change == 'unknown_episode':
            records = [json.loads(line) for line in first.read_text().splitlines()]
            records[0]['episode_id'] = 'not-in-the-manifest'
            first.write_text(''.join(json.dumps(row) + '\n' for row in records))
        elif change == 'swap_directories':
            other = plan.output / plan.groups[-2].episodes[0].episode_id
            for name in ('player-1.jsonl', 'player-2.jsonl', 'replay-player-1.json', 'replay-player-2.json'):
                first, second = directory / name, other / name
                before, after = first.read_bytes(), second.read_bytes()
                first.write_bytes(after)
                second.write_bytes(before)
        else:
            before, after = first.read_bytes(), second.read_bytes()
            first.write_bytes(after)
            second.write_bytes(before)
            if change == 'swap_both':
                first, second = directory / 'replay-player-1.json', directory / 'replay-player-2.json'
                before, after = first.read_bytes(), second.read_bytes()
                first.write_bytes(after)
                second.write_bytes(before)
        if change == 'swap_journals':
            reference, proof = ingest_collection(plan.output, loaded.config.model, plan.linkage,
                                                 deadline=plan.execution_deadline,
                                                 online_context=online_context(plan))
            assert reference and proof['decisions'] == 16
        else:
            with pytest.raises(ContractError):
                ingest_collection(plan.output, loaded.config.model, plan.linkage,
                                  deadline=plan.execution_deadline, online_context=online_context(plan))
        return result

    output = tmp_path / 'parent'
    if change == 'swap_journals':
        assert asyncio.run(train_ppo(loaded, corrupt, output))['status'] == 'complete'
        assert len(updates) == 1
    else:
        with pytest.raises(ContractError):
            asyncio.run(train_ppo(loaded, corrupt, output))
        assert not updates
        assert read_json(output / 'run.json')['status'] == 'failed'
