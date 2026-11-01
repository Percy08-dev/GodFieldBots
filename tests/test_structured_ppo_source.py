"""Original Source/numeric correspondence, not parent qualification or gameplay."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace

import pytest

from gf_common.domain import (
    ContractError,
    EpisodeGroup,
    EpisodeSpec,
    RuntimePolicyBinding,
    canonical,
    digest,
)
from gf_common.preprocessing import PreparationConfig
from gf_models.arrays import atomic_json, sha256
from gf_models.structured_artifacts import semantics
from gf_models.structured_config import model_config_v2
from gf_training.collection import CollectionPlan
from gf_training.structured_ppo_source import component_batch_source
from gf_training.structured_rollout import (
    EpisodeRecords,
    StructuredRollout,
    read_rollout,
    write_rollout,
)
from tests.structured_source_support import collection_fixture

SETTINGS = PreparationConfig(reader='structured-v1', memory_budget_mib=512)


def fixture(tmp_path, make_input, **options):
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4, neural=True, games=16, **options)
    bindings = {label: RuntimePolicyBinding(**value) for label, value in manifest['policy_bindings'].items()}
    label = next(label for label, binding in bindings.items() if binding.model_id == 'mlp')
    model = model_config_v2(bindings[label].policy_options['model'])
    groups = tuple(EpisodeGroup(group['seed'], tuple(EpisodeSpec(item['episode_id'], tuple(item['policies']),
        tuple(item['sampling_seeds']), item['pair_id'], item['learner_viewer']) for item in group['episodes']), group['ancestry'])
        for group in manifest['schedule'])
    plan = CollectionPlan(groups, bindings, directory, manifest['ppo'], time.monotonic() + 60, time.monotonic() + 65)
    return plan, actions, model, label, semantics(model)


def source(values, *, settings=SETTINGS):
    plan, actions, model, label, declaration = values
    return component_batch_source(plan, actions=actions, model=model, learner_label=label,
        inference_digest='a' * 64, declaration=declaration, settings=settings, deadline=plan.execution_deadline)


def test_original16_roles_roundtrip_reuses_exact_records_and_tokenizes_once(tmp_path, make_input, monkeypatch):
    import gf_training.structured_rollout as module
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        calls = []
        actual = module.tokenize_frame
        def counted(frame, *args, **kwargs):
            calls.append(frame)
            return actual(frame, *args, **kwargs)
        monkeypatch.setattr(module, 'tokenize_frame', counted)
        loaded = original.read_numeric(tmp_path / 'numeric', sha256(tmp_path / 'numeric/rollout.json'))
        assert len(loaded.episodes) == 16 and len(loaded) == 16
        assert all(episode.records is record for episode, record in zip(loaded.episodes, original.records, strict=True))
        assert len(calls) == 16
        assert all(frame is record.frames[0] for frame, record in zip(calls, original.records, strict=True))
        assert original.proof['accepted_decisions'] == 32
        assert original.proof['qualified'] is False and original.proof['parent_qualification'] == 'not_performed'
        assert manifest['decision_count'] == 16


@pytest.mark.parametrize('mutation', ['bank', 'frame', 'table', 'selection', 'closure'])
def test_selfconsistent_rehashed_numeric_cannot_replace_original_source(tmp_path, make_input, mutation):
    from gf_common.structured import CandidateTable, DecisionFrame, EventBank, EventDelta, Selection
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        record = original.records[0]
        bank, frames, tables, selections = record.bank, record.frames, record.tables, record.selections
        end_kind, outcome, reason = record.end_kind, record.outcome, record.reason
        if mutation == 'bank':
            new = EventBank(bank.actor, bank.episode)
            new.append(EventDelta(bank.actor, bank.episode, 0, new.boundary.prefix_hash,
                [{'type': 'action_accepted', 'player': bank.actor}]))
            bank = new.view()  # terminal tail differs; original frame still sees empty prefix.
        elif mutation == 'frame':
            frame = frames[0]
            payload = json.loads(frame.payload)
            payload['observation']['own']['hp'] -= 1
            raw = canonical(payload).encode()
            frame = DecisionFrame(frame.actor, frame.episode, frame.revision, frame.interval_start,
                frame.boundary, frame.interval_start_hash, raw, digest({'actor': frame.actor, 'episode': frame.episode,
                    'revision': frame.revision, 'interval_start': frame.interval_start, 'boundary': asdict(frame.boundary),
                    'interval_start_hash': frame.interval_start_hash, 'payload': payload}))
            frames = (frame,)
            actions = [tables[0].to_action(index, record.frames[0]) for index in range(len(tables[0].kinds))]
            table = CandidateTable.from_actions(frame, actions, action_contract=record.action_contract,
                purpose='component', synthetic_capacity=False)
            tables = (table,)
            selections = (Selection(table.table_hash, selections[0].index, selections[0].policy_version, selections[0].logprob, selections[0].value),)
        elif mutation == 'table':
            actions = [tables[0].to_action(index, frames[0]) for index in reversed(range(len(tables[0].kinds)))]
            table = CandidateTable.from_actions(frames[0], actions, action_contract=record.action_contract,
                purpose='component', synthetic_capacity=False)
            tables = (table,)
            selections = (Selection(table.table_hash, 0, selections[0].policy_version, selections[0].logprob, selections[0].value),)
        elif mutation == 'selection':
            selections = (replace(selections[0], value=.5),)
        else:
            end_kind, outcome, reason = 'truncated', None, 'exact interrupted reason'
        changed = EpisodeRecords(bank, frames, tables, selections, end_kind, outcome, reason,
            action_contract=record.action_contract, purpose='component', synthetic_capacity=False)
        rollout = StructuredRollout((changed, *original.records[1:]), original.model, digest(asdict(original.actions)),
            action_contract=record.action_contract, purpose='component', synthetic_capacity=False)
        write_rollout(tmp_path / 'different', rollout)
        with pytest.raises(ContractError, match='ORIGINAL participant Source'):
            original.read_numeric(tmp_path / 'different', sha256(tmp_path / 'different/rollout.json'))


@pytest.mark.parametrize('all_zero', [False, True])
def test_zero_and_interrupted_records_preserve_all16_roles_without_targets(tmp_path, make_input, all_zero):
    values = fixture(tmp_path, make_input, decisions=False if all_zero else (False,) + (True,) * 15,
                     end_kind=('natural',) + ('cancelled',) * 15)
    with source(values) as original:
        original.write_numeric(tmp_path / 'numeric')
        loaded = original.read_numeric(tmp_path / 'numeric', sha256(tmp_path / 'numeric/rollout.json'))
        assert len(loaded.episodes) == 16 and not loaded.episodes[0].records.frames
        assert loaded.episodes[0].records.outcome == 'draw'
        assert all(episode.target is None for episode in loaded.episodes)
        assert all(episode.records.reason == 'cancelled' and episode.records.outcome is None for episode in loaded.episodes[1:])
        with pytest.raises(ContractError):
            loaded.require_learning()


def test_historical_recorded_runtime_has_no_current_queries_and_active_remains_strict(tmp_path, make_input, monkeypatch):
    import gf_models.structured_artifacts as artifacts
    import gf_models.torch_runtime as runtime
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        original.write_numeric(tmp_path / 'numeric')
    path = tmp_path / 'numeric/rollout.json'
    manifest = json.loads(path.read_text())
    manifest['runtime'] = {'torch': 'historical-torch', 'numpy': 'historical-numpy', 'python': 'historical-python', 'gil_enabled': True}
    atomic_json(path, manifest)
    declaration = {**values[-1], 'runtime': manifest['runtime']}
    def forbidden(*args, **kwargs):
        raise AssertionError('current runtime/hardware/producer query forbidden')
    monkeypatch.setattr(artifacts, 'runtime_metadata', forbidden)
    monkeypatch.setattr(artifacts, 'producer_identity', forbidden)
    monkeypatch.setattr(runtime, 'hardware', forbidden)
    with source((*values[:-1], declaration)) as original:
        assert len(original.read_numeric(path.parent, sha256(path))) == 16
    monkeypatch.undo()
    with pytest.raises(ContractError, match='runtime mismatch'):
        read_rollout(path.parent, sha256(path), values[2], action_contract=values[1].action_contract,
                     purpose='component', synthetic_capacity=False)


@pytest.mark.parametrize('mutation', ['during_decode', 'after_publication', 'late_inventory'])
def test_original_snapshot_is_not_rebased_after_decode_or_publication(tmp_path, make_input, monkeypatch, mutation):
    import gf_training.structured_rollout as module
    values = fixture(tmp_path, make_input)
    original_read = module.read_bundle
    with pytest.raises(ContractError, match='original source|inventory|size/type mismatch'), source(values) as original:
        original.write_numeric(tmp_path / 'numeric')
        if mutation == 'during_decode':
            def late(*args, **kwargs):
                arrays = original_read(*args, **kwargs)
                (values[0].output / 'report.json').write_text('changed')
                return arrays
            monkeypatch.setattr(module, 'read_bundle', late)
            original.read_numeric(tmp_path / 'numeric', sha256(tmp_path / 'numeric/rollout.json'))
        elif mutation == 'after_publication':
            (values[0].output / 'report.json').write_text('changed')
        else:
            (values[0].output / 'episode-0/late.columns').write_bytes(b'late foreign file')


@pytest.mark.parametrize('mutation', ['role', 'seed', 'binding', 'generation'])
def test_original_collection_must_match_bounded_frozen_plan(tmp_path, make_input, mutation):
    values = fixture(tmp_path, make_input)
    plan, *rest = values
    if mutation in ('role', 'seed'):
        groups = list(plan.groups)
        group = groups[0]
        specs = list(group.episodes)
        specs[0] = replace(specs[0], learner_viewer=2) if mutation == 'role' else replace(specs[0], sampling_seeds=(42, 17))
        groups[0] = replace(group, episodes=tuple(specs))
        plan = replace(plan, groups=tuple(groups))
    elif mutation == 'binding':
        bindings = copy.deepcopy(plan.bindings)
        bindings[values[3]].policy_options['inference']['candidate_chunk'] += 1
        plan = replace(plan, bindings=bindings)
    else:
        plan = replace(plan, linkage={**plan.linkage, 'collection_generation': 1})
    with pytest.raises(ContractError, match='frozen'), source((plan, *rest)):
        pass


def test_joint_source_decode_reservation_rejects_before_array_allocation(tmp_path, make_input, monkeypatch):
    import gf_training.structured_rollout as module
    from gf_models.sharded_arrays import inspect_bundle
    from gf_training.structured_rollout import _read_recorded_rollout
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        descriptors = inspect_bundle(tmp_path / 'numeric', manifest['data'])
        required = module._read_reservation(manifest, descriptors)
        extra = original._snapshot.resident_bytes
        def forbidden(*args, **kwargs):
            raise AssertionError('decoded arrays allocated before joint admission')
        monkeypatch.setattr(module, 'read_bundle', forbidden)
        with pytest.raises(ContractError, match='joint.*reservation'):
            _read_recorded_rollout(tmp_path / 'numeric', sha256(tmp_path / 'numeric/rollout.json'), values[2],
                action_contract=values[1].action_contract, declaration=values[-1], expected_records=original.records,
                max_owned_bytes=required + extra - 1, extra_owned_bytes=extra, check=lambda: None, reserve=lambda _: None)


def test_raw_source_match_does_not_authorize_changed_stored_tokens(tmp_path, make_input):
    from gf_models.sharded_arrays import read_bundle, write_bundle
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        arrays = {name: value.copy() for name, value in read_bundle(tmp_path / 'numeric', manifest['data']).items()}
        arrays['e.0.f.0.state_numerics'][0] += 1
        changed = {**manifest, 'data': write_bundle(tmp_path / 'different', 'rollout', arrays)}
        atomic_json(tmp_path / 'different/rollout.json', changed)
        with pytest.raises(ContractError, match='stored tokens differ'):
            original.read_numeric(tmp_path / 'different', sha256(tmp_path / 'different/rollout.json'))


def test_shared_source_budget_deduction_requires_actual_object_identity(tmp_path, make_input):
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        equal_copy = copy.deepcopy(original.records)
        assert equal_copy == original.records and equal_copy[0] is not original.records[0]
        exact_extra = original._snapshot.extra_bytes_for(original.records)
        copied_extra = original._snapshot.extra_bytes_for(equal_copy)
        # deepcopy legitimately shares immutable bytes/strings. Its distinct
        # dataclass containers must still be charged despite equal content.
        distinct_containers = sum(sys.getsizeof(record) + sys.getsizeof(record.__dict__) for record in original.records)
        assert copied_extra >= exact_extra + distinct_containers


def test_historical_correspondence_in_process_without_torch_or_current_metadata(tmp_path, make_input):
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        original.write_numeric(tmp_path / 'numeric')
    declaration = tmp_path / 'declaration.json'
    atomic_json(declaration, values[-1])
    code = '''
import importlib.abc, importlib.metadata, json, sys, time
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import forbidden')
sys.meta_path.insert(0, NoTorch())
def forbidden(*args, **kwargs):
    raise AssertionError('current installed metadata forbidden')
importlib.metadata.version = forbidden
from gf_common.domain import EpisodeGroup, EpisodeSpec, RuntimePolicyBinding
from gf_common.structured import action_config_v2
from gf_common.preprocessing import PreparationConfig
from gf_models.structured_config import model_config_v2
from gf_models.arrays import sha256
from gf_training.collection import CollectionPlan
from gf_training.structured_ppo_source import component_batch_source
directory, numeric, declaration_path = map(Path, sys.argv[1:])
manifest = json.loads((directory / 'manifest.json').read_text())
bindings = {key: RuntimePolicyBinding(**value) for key,value in manifest['policy_bindings'].items()}
label = next(key for key,value in bindings.items() if value.model_id == 'mlp')
model = model_config_v2(bindings[label].policy_options['model'])
actions = action_config_v2(manifest['effective_config']['actions'])
groups = tuple(EpisodeGroup(group['seed'], tuple(EpisodeSpec(item['episode_id'], tuple(item['policies']), tuple(item['sampling_seeds']), item['pair_id'], item['learner_viewer']) for item in group['episodes']), group['ancestry']) for group in manifest['schedule'])
plan = CollectionPlan(groups, bindings, directory, manifest['ppo'], time.monotonic()+60, time.monotonic()+65)
with component_batch_source(plan, actions=actions, model=model, learner_label=label, inference_digest='a'*64,
    declaration=json.loads(declaration_path.read_text()), settings=PreparationConfig(reader='structured-v1', memory_budget_mib=512), deadline=plan.execution_deadline) as source:
    assert len(source.read_numeric(numeric, sha256(numeric/'rollout.json'))) == 16
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(values[0].output), str(tmp_path / 'numeric'), str(declaration)],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_encoded_npz_scratch_is_admitted_before_any_archive_open(tmp_path, make_input, monkeypatch):
    import zipfile

    import gf_training.structured_rollout as module
    from gf_models.sharded_arrays import validate_bundle_metadata
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        shard = manifest['data']['shards'][0]
        path = tmp_path / 'numeric' / shard['file']
        path.write_bytes(b'padded-prefix' * 100000 + path.read_bytes())
        shard['metadata']['sha256'] = sha256(path)
        sidecar = path.with_suffix('.json')
        atomic_json(sidecar, shard['metadata'])
        manifest['data']['files'][path.name] = shard['metadata']['sha256']
        manifest['data']['files'][sidecar.name] = sha256(sidecar)
        atomic_json(tmp_path / 'numeric/rollout.json', manifest)
        # This is a valid concatenated ZIP with tiny decoded tensors.
        assert len(original.read_numeric(path.parent, sha256(path.parent / 'rollout.json'))) == 16
        descriptors = validate_bundle_metadata(manifest['data'])
        base = module._read_reservation(manifest, descriptors) + 512 * len(manifest['data']['files'])
        encoded = 32 * max((path.parent / item['file']).stat().st_size for item in manifest['data']['shards'])
        extra = original._snapshot.resident_bytes
        def forbidden(*args, **kwargs):
            raise AssertionError('archive opened before encoded scratch admission')
        monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
        with pytest.raises(ContractError, match='encoded archive.*reservation'):
            module._read_recorded_rollout(path.parent, sha256(path.parent / 'rollout.json'), values[2],
                action_contract=values[1].action_contract, declaration=values[-1], expected_records=original.records,
                max_owned_bytes=extra + base + encoded - 1, extra_owned_bytes=extra, check=lambda: None, reserve=lambda _: None)


def test_archive_uses_admitted_fd_and_rejects_path_replacement_without_rebasing(tmp_path, make_input, monkeypatch):
    import os
    import zipfile
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        name = manifest['data']['shards'][0]['file']
        path = tmp_path / 'numeric' / name
        original_archive = zipfile.ZipFile
        seen = []
        def archive(held, *args, **kwargs):
            assert hasattr(held, 'fileno')
            if not seen:
                seen.append(os.fstat(held.fileno()).st_ino)
                replacement = path.with_suffix('.foreign')
                replacement.write_bytes(b'foreign replacement survives')
                os.replace(replacement, path)
                assert path.stat().st_ino != seen[0]
            return original_archive(held, *args, **kwargs)
        monkeypatch.setattr(zipfile, 'ZipFile', archive)
        with pytest.raises(ContractError, match='original admitted snapshot'):
            original.read_numeric(path.parent, sha256(path.parent / 'rollout.json'))
        assert seen and path.read_bytes() == b'foreign replacement survives'


def test_source_reserves_union_only_and_distinct_equal_containers_still_cost_bytes(tmp_path, make_input, monkeypatch):
    from gf_training.preparation import ResourceLimit
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        snapshot = original._snapshot
        extra = snapshot.additional_bytes(original.records)
        copied = copy.deepcopy(original.records)
        assert copied == original.records and snapshot.additional_bytes(copied) > extra
        scratch = 1024**2 + 1024 * len(snapshot._owner.files)
        cap = snapshot.resident_bytes + extra + scratch
        monkeypatch.setattr(type(snapshot), 'max_owned_bytes', property(lambda _: cap))
        snapshot.reserve(extra)  # exact identity-union cap admits its shared records.
        with pytest.raises(ResourceLimit, match='joint Source'):
            snapshot.reserve(snapshot.additional_bytes(copied))
        assert snapshot._external_bytes == extra


def test_writer_callback_subtracts_shared_rollout_owners_at_exact_joint_cap(tmp_path, make_input, monkeypatch):
    import gf_training.structured_ppo_source as module
    from gf_training.structured_rollout import _owned_bytes
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        snapshot = original._snapshot
        called = []
        def writer(directory, rollout, **kwargs):
            owners = (rollout.model, rollout.episodes, rollout.locations, rollout.action_config_hash,
                      rollout.policy_version, rollout.input_binding)
            # Form the union by visiting existing Source first; equality alone
            # never removes the copied containers tested above.
            additional = snapshot.additional_bytes(owners)
            shared = _owned_bytes(owners) - additional
            peak = rollout.resident_bytes + 4 * 1024**2
            expected_external = peak - shared
            scratch = 1024**2 + 1024 * len(snapshot._owner.files)
            cap = snapshot.resident_bytes + expected_external + scratch
            monkeypatch.setattr(type(snapshot), 'max_owned_bytes', property(lambda _: cap))
            kwargs['reserve'](peak)
            assert snapshot._external_bytes == expected_external
            assert kwargs['extra_owned_bytes'] == snapshot.extra_bytes_for(owners)
            called.append(1)
            return {}
        monkeypatch.setattr(module, '_write_rollout', writer)
        original.write_numeric(tmp_path / 'not-published')
        assert called == [1]


@pytest.mark.parametrize('file_kind', ['sidecar', 'manifest'])
def test_actual_padded_json_bytes_pre_admitted_before_parse_and_valid_with_adequate_cap(tmp_path, make_input, monkeypatch, file_kind):
    import gf_models.sharded_arrays as shards
    import gf_training.structured_rollout as module
    values = fixture(tmp_path, make_input)
    settings = replace(SETTINGS, memory_budget_mib=1024)
    with source(values, settings=settings) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        if file_kind == 'sidecar':
            path = tmp_path / 'numeric' / manifest['data']['shards'][0]['file'].replace('.npz', '.json')
            path.write_bytes(b' ' * 20_998_719 + path.read_bytes())
            manifest['data']['files'][path.name] = sha256(path)
            atomic_json(tmp_path / 'numeric/rollout.json', manifest)
        else:
            path = tmp_path / 'numeric/rollout.json'
            path.write_bytes(b' ' * 20_998_719 + path.read_bytes())
        numeric_digest = sha256(tmp_path / 'numeric/rollout.json')
        assert len(original.read_numeric(tmp_path / 'numeric', numeric_digest)) == 16
        loads, seen = shards.json.loads, []
        def guarded(raw, *args, **kwargs):
            if len(raw) >= 20_998_719:
                seen.append(1)
                raise AssertionError('padded JSON parsed before caller joint admission')
            return loads(raw, *args, **kwargs)
        monkeypatch.setattr(shards.json, 'loads', guarded)
        with pytest.raises(ContractError, match='JSON reservation|encoded archive'):
            module._read_recorded_rollout(tmp_path / 'numeric', numeric_digest, values[2],
                action_contract=values[1].action_contract, declaration=values[-1], expected_records=original.records,
                max_owned_bytes=17_232_556, extra_owned_bytes=original._snapshot.resident_bytes,
                check=lambda: None, reserve=lambda _: None)
        assert seen == []


@pytest.mark.parametrize('kind', ['error', 'cancel'])
@pytest.mark.parametrize('recheck_failure', ['mutation', 'deadline'])
def test_source_recheck_preserves_initiating_error_and_cancellation(tmp_path, make_input, monkeypatch, kind, recheck_failure):
    import asyncio

    from gf_training.structured_source import _SourceSnapshot
    values = fixture(tmp_path, make_input)
    error = ValueError('original caller error') if kind == 'error' else asyncio.CancelledError('original caller cancellation')
    actual = _SourceSnapshot.recheck
    owners = []
    def recorded(self):
        owners.append(self._owner)
        return actual(self)
    monkeypatch.setattr(_SourceSnapshot, 'recheck', recorded)
    with pytest.raises(type(error), match=str(error)) as caught, source(values) as original:
        if recheck_failure == 'mutation':
            (values[0].output / 'report.json').write_text('changed')
        else:
            original._snapshot._owner.deadline = float('-inf')
        raise error
    assert caught.value is error and owners == [original._snapshot._owner]
    assert any('original Source recheck failed:' in note for note in error.__notes__)
    assert any(('ContractError' if recheck_failure == 'mutation' else 'TimeoutError') in note for note in error.__notes__)
