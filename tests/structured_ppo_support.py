"""Predetermined public HTTP components; no executable Simulator or engine copy."""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, replace

import httpx

from gf_cli.config import LoadedConfig, RunConfigV2, load_config
from gf_common.action_contract import engineering_contract
from gf_common.domain import digest
from gf_common.preprocessing import PreparationConfig
from gf_common.structured import ActionConfigV2
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import model_defaults_v2
from gf_simulator_client.http import AttestedClient, seed_fingerprint
from gf_simulator_client.process import server_effective
from gf_training.collection import FinalizedCollection
from tests.episode_support import PublicScript
from tests.ppo_support import synthetic_server_identity


def configuration(tmp_path, *, kind='mlp', learning_device='cpu', inference_device='cpu', max_updates=2, hand=None):
    tmp_path.mkdir(exist_ok=True)
    loaded = load_config('configs/experiments/ppo-cpu.toml', {'server': synthetic_server_identity(tmp_path)})
    model = model_defaults_v2(kind)
    model = replace(model, inference=replace(model.inference, device=inference_device))
    if hand is not None:
        model = replace(model, hand=replace(model.hand, kind=hand))
    config = replace(loaded.config, actions=ActionConfigV2(action_contract=engineering_contract()), model=model,
        run=RunConfigV2(split='train'), training=PPOTrainingConfig(seeds=(17,), device=learning_device, max_updates=max_updates),
        preprocessing=PreparationConfig(backend='python', reader='structured-v1', memory_budget_mib=2048,
                                       cache_budget_mib=512, chunk_target_mib=16))
    return LoadedConfig(config, asdict(config), digest(asdict(config)), ())


def component_http(monkeypatch, loaded, *, mode='natural', source_ops=None, validation_context=None, outcome='draw', actors=(2,1,2,1)):
    import gf_cli
    from gf_models.structured_baselines import make_structured_policy
    from gf_simulator_client import structured_environment
    from gf_training.structured_journal import StructuredJournal
    # Only this test-owned subclass uses private bodies and fixed I/O doubles.
    # Public/direct Environment guards remain entirely unpatched.
    by_url, scripts, workers, seen_ledgers = {}, [], [], []
    serial = itertools.count()
    class Server:
        def __init__(self, binary, config, seed, count, directory, *args, **kwargs):
            self.expected = server_effective(config, count)
            self.script = PublicScript(mode=mode,outcome=outcome,actors=actors)
            self.script.info.update(self.expected, seed_fingerprint=seed_fingerprint(seed))
            self.url = f'http://component-{next(serial)}'
            by_url[self.url] = self.script
            scripts.append(self.script)
        async def start(self):
            return {'base_url': self.url, 'test_server_info': self.script.info}
        async def stop(self, timeout):
            return {'exit_code': 0, 'shutdown': 'stdin_eof', 'log_error': None}
    def client(url, **kwargs):
        return AttestedClient(url, transport=httpx.MockTransport(by_url[url].handle), **kwargs)
    original_worker = structured_environment.StructuredParticipantWorker
    class Worker(original_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)
    class FakeEnvironment(structured_environment.StructuredSimulatorEnvironment):
        def __init__(self, settings, ledger):
            if validation_context is None:
                assert ledger.learner is not None and ledger.learner._cleanup_owner.process.returncode is None
            else:
                assert ledger is validation_context.ledger and ledger.learner is None and not ledger.updating
            seen_ledgers.append(ledger)
            self._initialize_structured(settings, make_structured_policy, StructuredJournal, ledger=ledger)
        def _new_server(self, *args, **kwargs):
            return Server(*args, **kwargs)
        def _new_client(self, *args, **kwargs):
            return client(*args, **kwargs)
        async def run_group(self, group):
            return await self._run_group_body(group)
        async def _episode(self, client, definitions, group, spec):
            return await self._episode_body(client, definitions, group, spec)
    monkeypatch.setattr(structured_environment, 'StructuredParticipantWorker', Worker)
    async def collector(plan, ledger):
        before = time.monotonic()
        directory, report = await gf_cli._run_experiment(loaded, 'collect', collection=plan, ledger=ledger,
                                                      _structured_factory=FakeEnvironment, _structured_source_ops=source_ops,
                                                      _structured_validation_context=validation_context)
        assert time.monotonic() >= before and not ledger.workers
        assert (ledger.learner is not None) == (validation_context is None)
        return FinalizedCollection(directory, report)
    return collector, scripts, workers, seen_ledgers


def clone_component_parent(original, target):
    """Reseal a TEST-ONLY copy's paths and actual source-file fingerprints.

    The copied records remain engineering components. This is not a migration
    or qualification API, and no production source reader trusts this helper.
    """
    import json
    import shutil

    from gf_common.structured import action_config_v2
    from gf_models.arrays import sha256
    from gf_training.journal import atomic_json
    from gf_training.structured_source import inspect_component_collection
    shutil.copytree(original, target)
    state = json.loads((target / 'run.json').read_text())
    actions = action_config_v2(state['effective_config']['actions'])
    settings = PreparationConfig(**state['effective_config']['preprocessing'])
    def paths(value):
        if isinstance(value, str) and value.startswith(str(original) + '/'):
            return str(target) + value[len(str(original)):]
        if isinstance(value, dict):
            return {key: paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        return value
    parent = None
    for number in range(state['updates'] + 1):
        generation = target / 'generations' / f'{number:06}.json'
        record = json.loads(generation.read_text())
        record['parent_sha256'] = parent
        if number:
            update_path = target / 'updates' / f'{number:06}.json'
            update = paths(json.loads(update_path.read_text()))
            collection = target / update['collection']
            manifest = paths(json.loads((collection / 'manifest.json').read_text()))
            manifest['ppo']['parent_run_path'] = str(target)
            atomic_json(collection / 'manifest.json', manifest)
            source = inspect_component_collection(collection, actions=actions, settings=settings, expected_split='train')
            update.update(parent_sha256=parent, collection_manifest_sha256=sha256(collection / 'manifest.json'), source_proof=source.proof)
            atomic_json(update_path, update)
            record['update_sha256'] = sha256(update_path)
        atomic_json(generation, record)
        parent = sha256(generation)
    state['generation_sha256'] = parent
    atomic_json(target / 'run.json', state)
    atomic_json(target / 'latest.json', {'generation': state['updates'], 'sha256': parent})
    return target
