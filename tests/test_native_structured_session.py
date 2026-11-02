"""Candidate live Session over original FakeHTTP16 cohort; no real I/O authority."""
import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.domain import canonical
from gf_training.native_structured_session import NativeStructuredSource
from gf_training.structured_ppo import _train_component_ppo
from tests.structured_ppo_support import component_http, configuration

native = pytest.importorskip('gf_native')
pytestmark = pytest.mark.skipif(not hasattr(native, 'StructuredNativeSession'), reason='isolated B2 candidate required')


class ExportProbeComplete(RuntimeError):
    pass


def arrays_from_payload(value):
    encoded, bytes_, integers, floats = value
    assert type(bytes_) is type(integers) is type(floats) is tuple
    metadata = json.loads(encoded)
    banks = {'uint8': bytes_, 'int64': integers, 'float32': floats}
    arrays = {}
    for descriptor in metadata['arrays']:
        bank = banks[descriptor['dtype']][descriptor['bank_index']]
        assert str(bank.dtype) == descriptor['dtype'] and not bank.flags.writeable
        offset, count = descriptor['element_offset'], descriptor['element_count']
        assert 0 <= offset <= offset + count <= len(bank)
        view = bank[offset:offset + count].reshape(tuple(descriptor['shape']))
        assert not view.flags.writeable
        arrays[descriptor['key']] = view
    assert all(isinstance(bank.base, native.Float32Owner) for bank in floats)
    return metadata, arrays


def test_live_original16_scan_summary_sealed_schedule_authorize_typed_export(tmp_path, monkeypatch):
    import gf_training.structured_ppo as module
    loaded = configuration(tmp_path)
    collector, _scripts, workers, _ledgers = component_http(monkeypatch, loaded)
    observed = []
    @contextmanager
    def probe(plan, numeric, config, learner_label, generation, deadline, *, expected_manifest_sha256, extra_owned_bytes=0):
        # The source was produced by the unchanged Python collection path. This
        # candidate is its independent raw-input consumer, not a marker bypass.
        from gf_training.structured_source import (
            component_episode_records,
            inspect_component_collection,
        )
        reference = inspect_component_collection(plan.output, actions=config.actions, settings=config.preprocessing,
            expected_split='train', deadline=deadline)
        original_records = component_episode_records(reference, learner_seats=plan.linkage['learner_seats'])
        original_count = sum(len(record.frames) for record in original_records)
        assert len(original_records) == 16 and original_count == 32
        del original_records, reference
        original_plan = {'schema_version': 2, 'mode': 'live_cohort',
            'parents': [{'directory': str(Path(plan.linkage['parent_run_path'])), 'require_complete': False}],
            'collections': [{'directory': str(plan.output), 'parent_index': 0}], 'cohort_collection_index': 0}
        preparation = replace(config.preprocessing, backend='rust')
        # Configured scratch keeps its original meaning; the admission reserve
        # is NOT repurposed as a scratch allowance.
        with NativeStructuredSource(original_plan, preparation,
                max_owned_bytes=min(config.training.max_owned_bytes, preparation.memory_budget_mib * 1024**2),
                deadline=deadline, extra_owned_bytes=extra_owned_bytes, native=native) as owner:
            assert owner.summary is not None
            assert owner.summary['collections'][0]['manifest_sha256'] == expected_manifest_sha256
            count, proof, files = owner.write_numeric(numeric)
            assert count == original_count == owner.summary['collections'][0]['learner_decisions']
            assert proof['games'] == 16 and proof['qualified'] is False and proof['parent_qualification'] == 'not_performed'
            assert 'rollout.json' in files
            owner.begin_update()
            observed.append((owner.summary['control_fingerprint'], count, len(workers)))
            raise ExportProbeComplete('native live16 export probe complete before Adam')
        # Stop this first bounded fixture BEFORE Adam. Preserve it as failed /
        # quarantined; it is never resumed, repaired or claimed as eligible.
        yield  # pragma: no cover - the intentional pre-Adam probe never yields
    monkeypatch.setattr(module, 'project', probe)
    with pytest.raises(ExportProbeComplete, match='before Adam'):
        asyncio.run(_train_component_ppo(loaded, collector, tmp_path / 'parent'))
    assert observed and observed[0][2] == 32
    assert all(worker.process.poll() is not None for worker in workers)
    state = json.loads((tmp_path / 'parent/run.json').read_text())
    assert state['status'] == 'failed' and state['quarantine'] and state['updates'] == 0
    assert not (tmp_path / 'parent/generations/000001.json').exists()


def test_native_existing_common_loop_two_original16_batches_and_control_commit(tmp_path, monkeypatch):
    from dataclasses import asdict

    from gf_cli.config import LoadedConfig
    from gf_common.domain import digest
    from gf_training.structured_ppo import _NativeComponentOps, _train_native_component_ppo
    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    operations = _NativeComponentOps(native)
    from gf_training.ppo_run import owned_bytes
    original_project = operations.project
    allowances = []
    @contextmanager
    def measured_project(*args, **kwargs):
        # The real coordinator measures all current original files immediately
        # before projection. No admission-headroom or configured-scratch proxy.
        expected = config.training.max_owned_bytes - owned_bytes(tmp_path / 'parent')
        assert kwargs['scratch_allowance_bytes'] == max(0, expected)
        allowances.append(kwargs['scratch_allowance_bytes'])
        with original_project(*args, **kwargs) as value:
            yield value
    monkeypatch.setattr(operations, 'project', measured_project)
    collector, _scripts, workers, ledgers = component_http(monkeypatch, loaded, source_ops=operations)
    result = asyncio.run(_train_native_component_ppo(loaded, collector, tmp_path / 'parent', operations=operations))
    assert result['status'] == 'complete' and result['updates'] == 2 and result['accepted_decisions'] == 64
    assert len(workers) == 64 and all(worker.closed and worker.process.poll() is not None for worker in workers)
    assert all(ledger is ledgers[0] for ledger in ledgers)
    assert not ledgers[0].workers and ledgers[0].learner is None and not ledgers[0].updating
    assert len(allowances) == 2 and 0 < allowances[1] < allowances[0] < config.training.max_owned_bytes
    for number in (1, 2):
        update = json.loads((tmp_path / 'parent/updates' / f'{number:06}.json').read_text())
        assert update['actual_games'] == update['planned_games'] == 16
        assert update['accepted_decisions'] == 32 and len(update['learner_seats']) == 16
        assert update['source_proof']['qualified'] is False and update['source_proof']['parent_qualification'] == 'not_performed'


def test_native_exact_continuous_two_vs_one_boundary_pause_resume(tmp_path, monkeypatch):
    from dataclasses import asdict

    from gf_cli.config import LoadedConfig
    from gf_common.domain import digest
    from gf_models.sharded_arrays import read_bundle
    from gf_training.structured_ppo import _NativeComponentOps, _train_native_component_ppo
    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    def run(directory, **settings):
        operations = _NativeComponentOps(native)
        collector, _scripts, workers, ledgers = component_http(monkeypatch, loaded, source_ops=operations)
        result = asyncio.run(_train_native_component_ppo(loaded, collector, directory, operations=operations, **settings))
        assert all(worker.closed and worker.process.poll() is not None for worker in workers)
        assert not ledgers[0].workers and ledgers[0].learner is None
        return result
    continuous = run(tmp_path / 'continuous')
    paused = run(tmp_path / 'resumed', pause_after_update=1)
    assert paused['status'] == 'paused' and paused['updates'] == 1
    resumed = run(tmp_path / 'resumed', mode='resume')
    assert continuous['status'] == resumed['status'] == 'complete'
    assert continuous['updates'] == resumed['updates'] == 2
    assert continuous['accepted_decisions'] == resumed['accepted_decisions'] == 64
    for manifest_name, bundle_key in [('inference.json', 'weights'), ('resume.json', 'state')]:
        left = tmp_path / 'continuous/artifacts/000002'
        right = tmp_path / 'resumed/artifacts/000002'
        original = json.loads((left / manifest_name).read_text())
        restored = json.loads((right / manifest_name).read_text())
        arrays1, arrays2 = read_bundle(left, original[bundle_key]), read_bundle(right, restored[bundle_key])
        assert arrays1.keys() == arrays2.keys()
        for key in arrays1:
            assert arrays1[key].dtype == arrays2[key].dtype and arrays1[key].shape == arrays2[key].shape
            assert arrays1[key].tobytes() == arrays2[key].tobytes(), key
        if manifest_name == 'resume.json':
            assert canonical(original['parameter_states']) == canonical(restored['parameter_states'])
            assert canonical(original['shuffle_rng']) == canonical(restored['shuffle_rng'])
    for number in (0, 1, 2):
        left = json.loads((tmp_path / 'continuous/generations' / f'{number:06}.json').read_text())
        right = json.loads((tmp_path / 'resumed/generations' / f'{number:06}.json').read_text())
        assert canonical(left['parent_controls']) == canonical(right['parent_controls'])
        assert left['inference_sha256'] == right['inference_sha256']
    for batch in (0, 1):
        left = json.loads((tmp_path / 'continuous/collections' / f'batch-{batch:06}/manifest.json').read_text())
        right = json.loads((tmp_path / 'resumed/collections' / f'batch-{batch:06}/manifest.json').read_text())
        assert canonical(left['schedule']) == canonical(right['schedule'])


def test_actual_structured_marker_reader_settings_and_default_native_dispatch(tmp_path, monkeypatch):
    from dataclasses import asdict

    from gf_cli.config import LoadedConfig
    from gf_common.domain import digest
    from gf_common.structured_tokens import signature
    from gf_training.native_backend import STRUCTURED_CONTRACT, binary_identity, reader_settings
    from gf_training.structured_ppo import _NativeComponentOps

    information = dict(native.build_info())
    assert information['structured_preprocessing_contract'] == STRUCTURED_CONTRACT
    assert information['structured_default_signature'] == signature(128)
    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    assert reader_settings(config.preprocessing, native) == asdict(config.preprocessing)
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    observed = []
    class DispatchObserved(RuntimeError):
        pass
    def check_selected(self, current):
        assert self.native is native and self.binary == binary_identity(native)
        assert self._implementation == 'native-structured-v2'
        assert current is config
        observed.append(self.binary)
        raise DispatchObserved('real marker-selected default native operations')
    async def no_collection(*args):
        pytest.fail('dispatch-only test must finish before collection')
    monkeypatch.setattr(_NativeComponentOps, 'validate_config', check_selected)
    # No explicit candidate override and no build_info/reader_settings patch.
    with pytest.raises(DispatchObserved, match='marker-selected default'):
        asyncio.run(_train_component_ppo(loaded, no_collection, tmp_path / 'parent'))
    assert len(observed) == 1 and not (tmp_path / 'parent').exists()


def test_native_late_original_mutation_after_publication_quarantines_before_adam(tmp_path, monkeypatch):
    from dataclasses import asdict

    from gf_cli.config import LoadedConfig
    from gf_common.domain import digest
    from gf_training.structured_learning_process import StructuredLearningProcess
    from gf_training.structured_ppo import _NativeComponentOps, _train_native_component_ppo

    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    operations = _NativeComponentOps(native)
    collector, _scripts, workers, ledgers = component_http(monkeypatch, loaded, source_ops=operations)
    children = []
    make_child = operations.make_child
    def capture_child(*args, **kwargs):
        child, ledger = make_child(*args, **kwargs)
        children.append(child)
        return child, ledger
    monkeypatch.setattr(operations, 'make_child', capture_child)
    async def no_adam(*args, **kwargs):
        pytest.fail('mutated original Source must reject before Adam')
    monkeypatch.setattr(StructuredLearningProcess, 'update', no_adam)
    before_update = NativeStructuredSource.begin_update
    mutations = []
    def mutate_original_then_check(self):
        assert self._correspondence_done and self._published is not None
        numeric = tmp_path / 'parent/numeric/batch-000000/rollout.json'
        assert numeric.is_file()  # successful publish/retire/decoder preceded injection
        report = Path(self.plan['collections'][self.plan['cohort_collection_index']]['directory']) / 'report.json'
        original = report.read_bytes()
        report.write_bytes(original + b'\n')  # valid JSON, different ORIGINAL snapshot/hash
        mutations.append(report)
        before_update(self)  # unchanged real sealed + native before-Adam recheck
    monkeypatch.setattr(NativeStructuredSource, 'begin_update', mutate_original_then_check)
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(_train_native_component_ppo(loaded, collector, tmp_path / 'parent', operations=operations))
    state = json.loads((tmp_path / 'parent/run.json').read_text())
    assert state['status'] == 'failed' and state['quarantine'] is True
    assert state['updates'] == state['optimizer_steps'] == state['accepted_decisions'] == 0
    assert len(mutations) == 1 and len(workers) == 32
    assert all(worker.closed and worker.process.poll() is not None for worker in workers)
    assert len(children) == 1 and children[0].process.returncode == 0
    assert children[0].close_task.done() and children[0].health['exit_code'] == 0
    assert ledgers and all(ledger is ledgers[0] for ledger in ledgers)
    assert not ledgers[0].workers and ledgers[0].learner is None and not ledgers[0].updating
    assert not (tmp_path / 'parent/generations/000001.json').exists()
    assert not (tmp_path / 'parent/updates/000001.json').exists()
    assert not (tmp_path / 'parent/artifacts/000001').exists()
