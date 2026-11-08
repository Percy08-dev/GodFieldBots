"""Bounded runtime primitives; no qualified artifacts, Source-positive fit or games."""
from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli.config import load_config
from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.card_instance_game_tree_runtime import (
    TREE_OPTIONS_FIELDS,
    RetainedGamePolicyArtifacts,
    safe_tree_options_game_v3,
)
from gf_models.config import model_config, validate_capacity
from gf_models.structured_binding import configured_model_v3
from gf_models.structured_tree_config import ConfiguredTreeModelV2, ConfiguredTreeModelV3
from gf_models.structured_tree_encoder import feature_layout_v3
from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
from gf_simulator_client.framing import frame_decode_bytes, read_frame_admitted

ROOT = Path(__file__).parents[1]
ACTIONS = ActionConfigV3(action_contract=corrected_card_instance_contract())
RULES = card_instance_rules()


def controlled_options():
    """Only a pure declaration. It supplies no actual artifact or paired authority."""
    from gf_common.card_instance_game_tree import origin
    model = ConfiguredTreeModelV3()
    return {'version': '3', 'checkpoint': '/own/artifact', 'expected_manifest_digest': '1'*64,
        'model': asdict(model.architecture()), 'inference': asdict(model.inference),
        'resources': asdict(model.resources), 'action_config_hash': digest(asdict(ACTIONS)),
        'tree_descriptor': {'tree_count': 1, 'max_depth': 1, 'model_sha256': '2'*64, 'model_bytes': 100},
        'feature_signature': feature_layout_v3(model.encoder)['feature_signature'],
        'source_kind': 'card-instance-formal-ppo-originals', 'source_origin': origin(),
        'training_pair': {'request_sha256': '3'*64, 'completion_sha256': '4'*64}}


def test_generic_tree3_dispatch_does_not_enter_neural_parser():
    tree = ConfiguredTreeModelV3()
    assert model_config(asdict(tree), layout=LAYOUT) == tree
    with pytest.raises(ContractError):
        configured_model_v3(asdict(tree), layout=LAYOUT)
    with pytest.raises(ContractError, match='corrected card-instance layout'):
        model_config(asdict(tree))
    assert type(model_config(asdict(ConfiguredTreeModelV2()))) is ConfiguredTreeModelV2
    validate_capacity(tree, 18)
    with pytest.raises(ContractError):
        validate_capacity(tree, 19)


def test_public_model_and_opponent_config_dispatch_before_legacy_defaults(tmp_path):
    path = tmp_path/'tree.toml'
    path.write_text('schema_version=1\nkind="experiment"\n[references]\nactions="'
        + str(ROOT/'configs/actions/card-instance-v2-corrected.toml') + '"\n'
        '[server]\ngameplay_api="v2"\nsimulator_revision="' + LAYOUT.action_contract.simulator_revision
        + '"\nfixed_seed_experiment=true\n[model]\nmodel_id="lightgbm"\nversion="3"\n'
        '[opponent]\nmodel_id="lightgbm"\nversion="3"\n')
    loaded = load_config(path)
    assert type(loaded.config.model) is ConfiguredTreeModelV3
    assert type(loaded.config.opponent) is ConfiguredTreeModelV3
    assert loaded.config.model.checkpoint is None
    with pytest.raises(ContractError):
        load_config(path, {'model': {'encoder': {'max_positions': 19}}})
    with pytest.raises(ContractError):
        load_config(path, {'model': {'version': '2'}})
    with pytest.raises(ContractError):
        load_config(path, {'training': {'algorithm': 'ppo'}})


def test_absent_tree_checkpoint_refuses_before_runtime_evidence_or_output(tmp_path, monkeypatch):
    import gf_cli.card_instance as cli
    path = tmp_path/'tree.toml'
    path.write_text('schema_version=1\nkind="experiment"\n[references]\nactions="'
        + str(ROOT/'configs/actions/card-instance-v2-corrected.toml') + '"\n'
        '[server]\ngameplay_api="v2"\nsimulator_revision="' + LAYOUT.action_contract.simulator_revision
        + '"\nfixed_seed_experiment=true\n[model]\nmodel_id="lightgbm"\nversion="3"\n')
    loaded = load_config(path, {'run': {'output_dir': str(tmp_path/'forbidden')}})
    monkeypatch.setattr(cli, 'CardInstanceRuntimeEvidence', lambda *a, **kw: pytest.fail('runtime allocated'))
    with pytest.raises(ContractError, match='explicit game checkpoint'):
        asyncio.run(cli.run_inference(loaded, 'collect'))
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('argument', ['0', '-1', '+1', '01', '1.0', ' 1', str(2**63), '1 2'])
def test_bootstrap_refuses_unsanitized_argv_before_framing(argument):
    result = subprocess.run([sys.executable, '-E', '-m', 'gf_simulator_client.card_instance_bootstrap', argument],
        input=b'', capture_output=True, timeout=10, check=False)
    assert result.returncode == 0
    size = int.from_bytes(result.stdout[:8], 'big')
    error = json.loads(result.stdout[8:8+size])
    assert error['worker_error'] == 'ContractError' and 'Tree3' in error['message']
    assert not result.stderr


def test_tree_small_cap_refuses_before_first_header_or_body():
    result = subprocess.run([sys.executable, '-E', '-m', 'gf_simulator_client.card_instance_bootstrap', '1'],
        input=b'', capture_output=True, timeout=10, check=False)
    stream = io.BytesIO(result.stdout)
    size = int.from_bytes(stream.read(8), 'big')
    error = json.loads(stream.read(size))
    assert error['worker_error'] == 'ContractError'
    assert 'before read' in error['message']


def test_shared_predecode_admission_never_reads_unadmitted_body():
    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            if self.tell() >= 8:
                pytest.fail('unadmitted body read')
            return super().read(size)
    stream = HeaderOnly((1024).to_bytes(8, 'big'))
    with pytest.raises(ContractError, match='before body read'):
        read_frame_admitted(stream, max_owned_bytes=frame_decode_bytes(1024)-1)


def test_worker_rejects_overlap_without_poisoning_admitted_operation():
    # Only lifecycle state is constructed; no fake artifact/pair or policy exists.
    async def exercise():
        worker = object.__new__(CardInstanceParticipantWorker)
        worker._inflight = None
        worker._tree_timeout = 10.
        worker._session_deadline = time.monotonic()+10
        worker._tree_failed = worker.closed = False
        worker._admit_tree_operation()
        first = worker._inflight
        with pytest.raises(ContractError, match='one in-flight'):
            worker._admit_tree_operation()
        assert worker._inflight is first and not worker._tree_failed
        worker._inflight = None
        worker.closed = True
        with pytest.raises(ContractError, match='cleanup-only'):
            worker._admit_tree_operation()
    asyncio.run(exercise())


def test_owner_retains_pair_memory_while_unreaped_worker_or_io_remains():
    # Retention primitive only; no complete owner, healthy pair or public route.
    owner = object.__new__(RetainedGamePolicyArtifacts)
    worker = SimpleNamespace(lease=SimpleNamespace(released=False), io_thread=None)
    owner._workers = [worker]
    assert not owner.can_release
    with pytest.raises(ContractError, match='remains retained'):
        owner.release()
    worker.lease.released = True
    worker.io_thread = SimpleNamespace(is_alive=lambda: True)
    assert not owner.can_release
    worker.io_thread = None
    assert owner.can_release


def test_own_policy_keyset_has_twelve_fields_and_no_source_controls():
    assert len(TREE_OPTIONS_FIELDS) == 12
    assert not {'dataset', 'source_directory', 'parent', 'labels', 'game_seed', 'training_binding'} & TREE_OPTIONS_FIELDS


def test_malformed_options_fail_before_optional_or_source_imports():
    before = set(sys.modules)
    with pytest.raises(ContractError):
        safe_tree_options_game_v3({'version': '3'}, actions=ACTIONS, rules=RULES)
    assert not any(name.startswith(('lightgbm', 'gf_training')) for name in set(sys.modules)-before)


def test_pair_guard_detects_changed_retained_original_and_becomes_sticky(tmp_path):
    from gf_models.structured_sealed import SealedScan
    path = tmp_path/'original.json'
    path.write_text('{"value":1}\n')
    scan = SealedScan(max_owned_bytes=1024**2, deadline=time.monotonic()+10)
    scan.object(path)
    # This is a file-guard primitive, not a fabricated complete GAME pair.
    owner = object.__new__(RetainedGamePolicyArtifacts)
    owner._released = owner._failed = False
    owner._deadline = time.monotonic()+10
    owner._limit, owner._neural = 1024**2*4, None
    owner._pairs = [SimpleNamespace(scan=scan, finalize=scan.finalize)]
    owner.guard(deadline=owner._deadline)
    path.write_text('{"value":2}\n')
    with pytest.raises(ContractError):
        owner.guard(deadline=owner._deadline)
    assert owner._failed
    with pytest.raises(ContractError, match='released or failed'):
        owner.guard(deadline=owner._deadline)


def test_fixed_bootstrap_zero_argument_baseline_path_stays_exact():
    from gf_common.card_instance_rules import rule_signature
    from gf_common.card_instance_runtime import game_binding
    from gf_models.structured_runtime import own_game_worker_settings_v3
    caps, costs, _, _ = own_game_worker_settings_v3('random', ACTIONS, None, rules=RULES)
    payload = {'protocol_version': 3, 'model_id': 'random', 'sampling_seed': 17,
        'actions': asdict(ACTIONS), 'rules': asdict(RULES), 'caps': asdict(caps),
        'numeric_costs': asdict(costs), **game_binding(ACTIONS.action_contract, rule_signature(ACTIONS.action_contract, RULES))}
    raw = canonical(payload).encode()
    result = subprocess.run([sys.executable, '-E', '-m', 'gf_simulator_client.card_instance_bootstrap'],
        input=len(raw).to_bytes(8, 'big')+raw, capture_output=True, timeout=10, check=False)
    size = int.from_bytes(result.stdout[:8], 'big')
    ready = json.loads(result.stdout[8:8+size])
    assert ready == {'protocol_version': 3, 'kind': 'ready', 'policy_version': 'random-card-instance-v2',
        'model_load_ms': 0., **game_binding(ACTIONS.action_contract, rule_signature(ACTIONS.action_contract, RULES))}
    result = subprocess.run([sys.executable, '-E', '-m', 'gf_simulator_client.card_instance_bootstrap', str(2*1024**3)],
        input=len(raw).to_bytes(8, 'big')+raw, capture_output=True, timeout=10, check=False)
    size = int.from_bytes(result.stdout[:8], 'big')
    error = json.loads(result.stdout[8:8+size])
    assert error['worker_error'] == 'ContractError' and 'Tree3 alone' in error['message']


def controlled_metadata():
    """UNADMITTED schema example: no files, request/completion or authority exist."""
    from gf_common.card_instance_game_tree import SOURCE_KIND, input_binding, origin
    from gf_models.structured_tree_artifacts import NATIVE_PARAMETERS, semantics_v3
    from gf_models.structured_tree_pairing import TREE3_GAME_RECORDS
    options = controlled_options()
    model = ConfiguredTreeModelV3()
    layout = feature_layout_v3(model.encoder)
    manifest = {'schema_version': 4, 'kind': 'structured-tree-inference',
        'model': options['model'], 'architecture_identity': TREE3_GAME_RECORDS.identity(model.architecture()),
        'input_recipe': LAYOUT.input_recipe, 'token_signature': layout['token_signature'],
        'feature_recipe': TREE3_GAME_RECORDS.features.recipe, 'feature_signature': layout['feature_signature'],
        'feature_names': layout['feature_names'], 'categorical_indices': layout['categorical_indices'],
        'semantics': semantics_v3(model, 'trained', _context=TREE3_GAME_RECORDS),
        'rule_signature': layout['rule_signature'], 'layout_signature': layout['layout_signature'],
        'feature_layout': layout, 'action_config_hash': options['action_config_hash'],
        'action_contract': input_binding()['action_contract'], 'purpose': 'qualified_game',
        'synthetic_capacity': False, 'fit_origin': 'trained',
        'provenance': {'schema_version': 4, 'kind': 'structured-fixed-dataset3-fit',
            'training_run_id': 'a'*32, 'request_sha256': options['training_pair']['request_sha256'],
            'dataset_manifest_sha256': '5'*64, 'input_binding': input_binding(),
            'source_kind': SOURCE_KIND, 'source_origin': origin()},
        'source_kind': SOURCE_KIND, 'source_origin': origin(), 'native_parameters': NATIVE_PARAMETERS,
        'runtime': {'numpy_version': '2.5.4', 'lightgbm_version': '4.7.0', 'native_library_sha256': '6'*64},
        'model_file': {'file': 'model.txt', 'sha256': '2'*64, 'bytes': 100},
        'tree_descriptor': options['tree_descriptor'], 'creation_execution': {
            'inference': options['inference'], 'resources': options['resources'], 'tree_descriptor': options['tree_descriptor']}}
    import hashlib
    raw_manifest = (canonical(manifest)+'\n').encode()
    options['expected_manifest_digest'] = hashlib.sha256(raw_manifest).hexdigest()
    return {'input_contract': 'player-card-refs-v2', 'policy_options': options,
        'inference_manifest': manifest, 'files': [
            {'path': 'inference.json', 'sha256': options['expected_manifest_digest'], 'identity': [0, 1, len(raw_manifest), 3, 4]},
            {'path': 'model.txt', 'sha256': '2'*64, 'identity': [0, 5, 100, 7, 8]}]}


def test_pure_tree_options_and_metadata_share_exact_label_and_never_load_native():
    # Test isolation cannot depend on what unrelated collected tests imported.
    script = '''
import builtins, importlib.abc, io, json, os, sys
blocked = ('gf_training', 'gf_native', 'lightgbm', 'torch')
attempts = []
class BlockOptionalAndSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            attempts.append(fullname)
            raise AssertionError('pure metadata imported ' + fullname)
sys.meta_path.insert(0, BlockOptionalAndSource())
from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common import card_instance_game_tree
from gf_models import structured_tree_artifacts, structured_tree_pairing
from gf_models.card_instance_game_tree_runtime import (
    TREE_OPTIONS_FIELDS, policy_label_tree_game_v3, validate_tree_metadata)
actions = ActionConfigV3(action_contract=corrected_card_instance_contract())
rules = card_instance_rules()
raw = json.loads(sys.stdin.read())
def forbidden(*args, **kwargs):
    attempts.append('file I/O or native allocation')
    raise AssertionError('pure metadata performed file I/O or native allocation')
structured_tree_artifacts._native = forbidden
builtins.open = io.open = os.open = os.stat = os.lstat = forbidden
safe, label = validate_tree_metadata(raw, actions=actions, rules=rules)
assert set(safe) == TREE_OPTIONS_FIELDS and len(safe) == 12
assert label == policy_label_tree_game_v3(safe, actions=actions, rules=rules)
relocated = {**safe, 'checkpoint': '/other/own/artifact'}
assert policy_label_tree_game_v3(relocated, actions=actions, rules=rules) == label
assert 'completion_sha256' not in raw['inference_manifest']
assert 'completion_sha256' not in raw['inference_manifest']['provenance']
assert safe['training_pair']['completion_sha256'] == '4'*64
assert not attempts
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in blocked)
print(json.dumps({'status': 'PASS', 'label': label}))
'''
    result = subprocess.run([sys.executable, '-E', '-c', script], input=canonical(controlled_metadata()),
        text=True, capture_output=True, cwd=ROOT/'src', timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert json.loads(result.stdout)['status'] == 'PASS'


@pytest.mark.parametrize('mutation', ['fixture', 'component', 'toy', 'missing_pair', 'source_path',
    'pair_path', 'action_hash', 'feature', 'request', 'architecture', 'unsorted_files', 'extra_file', 'hash'])
def test_controlled_metadata_crossovers_fail_closed(mutation):
    from gf_models.card_instance_game_tree_runtime import validate_tree_metadata
    raw = controlled_metadata()
    if mutation == 'fixture':
        from gf_common.card_instance_fixture import origin
        raw['policy_options']['source_origin'] = origin()
    elif mutation == 'component':
        raw['inference_manifest']['purpose'] = 'component'
    elif mutation == 'toy':
        raw['inference_manifest']['fit_origin'] = 'toy'
    elif mutation == 'missing_pair':
        del raw['policy_options']['training_pair']
    elif mutation == 'source_path':
        raw['policy_options']['source_directory'] = '/source'
    elif mutation == 'pair_path':
        raw['policy_options']['training_pair']['directory'] = '/source'
    elif mutation == 'action_hash':
        raw['policy_options']['action_config_hash'] = 'f'*64
    elif mutation == 'feature':
        raw['policy_options']['feature_signature'] = 'f'*64
    elif mutation == 'request':
        raw['inference_manifest']['provenance']['request_sha256'] = 'f'*64
    elif mutation == 'architecture':
        raw['inference_manifest']['model'] = {**raw['inference_manifest']['model'], 'version': '2'}
    elif mutation == 'unsorted_files':
        raw['files'].reverse()
    elif mutation == 'extra_file':
        raw['files'].append({'path': 'training.json', 'sha256': 'f'*64, 'identity': [0]*5})
    else:
        raw['files'][1]['sha256'] = 'f'*64
    with pytest.raises(ContractError):
        validate_tree_metadata(raw, actions=ACTIONS, rules=RULES)


def test_fixture_or_ordinary_origin_refuses_before_worker_reservation_or_process():
    from gf_simulator_client.structured_resources import ReservationLedger
    options = controlled_options()
    options['source_kind'] = 'card-instance-tree-fixture-originals'
    ledger = ReservationLedger()
    with pytest.raises(ContractError):
        CardInstanceParticipantWorker('lightgbm', 17, ACTIONS, ledger=ledger,
            policy_options=options, deadline=time.monotonic()+10)
    assert not ledger.workers and not ledger.cuda_bytes


def test_tree_null_statistics_and_neural_finite_statistics_are_distinct():
    import random

    from gf_common.diverse_actions import PublicCandidateGeneratorV3
    from gf_common.structured import Selection
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from gf_training.card_instance_journal import _Context
    from tests.model3_support import raw_inputs
    options = controlled_options()
    _, costs, _, _ = own_game_worker_settings_v3('lightgbm', ACTIONS, options, rules=RULES)
    _, _, frames, bank, _ = raw_inputs(count=1)
    generated = PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0.).generate_game(
        frames[0], bank.view(), random.Random(17), costs)
    context = _Context(ACTIONS, RULES)
    context.policy, context.numeric_costs_hash = 'controlled-label', digest(asdict(costs))
    context.validate_generated(generated, Selection(generated.table.table_hash, 0, context.policy))
    with pytest.raises(ContractError):
        context.validate_generated(generated, Selection(generated.table.table_hash, 0, context.policy, -1., 0.))
    context.neural = {'controlled': True}
    context.validate_generated(generated, Selection(generated.table.table_hash, 0, context.policy, -1., 0.))
    with pytest.raises(ContractError):
        context.validate_generated(generated, Selection(generated.table.table_hash, 0, context.policy))


@pytest.mark.parametrize('fit_origin', ['toy', 'trained'])
def test_bare_nonpaired_artifacts_never_enter_public_game_runtime(tmp_path, monkeypatch, fit_origin):
    import gf_cli.card_instance as cli
    from gf_cli.config import ExperimentConfig, LoadedConfig, RunConfigV2
    from gf_models import structured_tree_artifacts
    from gf_models.config import ModelConfig
    from gf_simulator_client.process import ManagedServer
    directory = tmp_path/'unadmitted-artifact'
    directory.mkdir()
    manifest = controlled_metadata()['inference_manifest']
    manifest['fit_origin'] = fit_origin
    if fit_origin == 'toy':
        manifest['provenance'] = manifest['source_kind'] = manifest['source_origin'] = None
    (directory/'inference.json').write_text(canonical(manifest)+'\n')
    (directory/'model.txt').write_text('invalid controlled text; no native model exists')
    model = ConfiguredTreeModelV3(checkpoint=str(directory))
    config = ExperimentConfig(actions=ACTIONS, model=model, opponent=ModelConfig(), run=RunConfigV2())
    # Only the negative artifact gate is exercised; no runtime attestation exists.
    config = replace(config, server=replace(config.server, source_dir=str(tmp_path)))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    monkeypatch.setattr(cli, 'CardInstanceRuntimeEvidence', lambda *a, **kw: pytest.fail('runtime allocated'))
    monkeypatch.setattr(structured_tree_artifacts, '_native', lambda **kw: pytest.fail('native allocated'))
    monkeypatch.setattr(ManagedServer, '__init__', lambda *a, **kw: pytest.fail('server allocated'))
    with pytest.raises(ContractError):
        asyncio.run(cli.run_inference(loaded, 'evaluate'))
    assert not (tmp_path/'runs').exists()


def test_controlled_process_close_cancellation_still_reaps_and_releases_lease():
    import threading

    from gf_simulator_client.structured_resources import ReservationLedger
    async def exercise():
        ledger = ReservationLedger()
        worker = object.__new__(CardInstanceParticipantWorker)
        worker._tree, worker._tree_failed = True, True
        worker._session_deadline = time.monotonic()+10
        worker.closed = False
        worker._close_task = None
        worker.health = {}
        worker.lease = ledger.reserve_worker(0)
        worker.lease._cleanup_owner = worker
        # A sleeping process is a lifecycle primitive, never a policy/model.
        worker.process = subprocess.Popen([sys.executable, '-E', '-c',  # noqa: ASYNC220 - shared synchronous process ownership kernel.
            'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        worker.lease.spawned(worker.process)
        worker.io_thread = threading.Thread(target=worker.process.stdout.read, daemon=True)
        worker.io_thread.start()
        closing = asyncio.create_task(worker.close(.02))
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert worker.process.poll() is not None
        assert not worker.io_thread.is_alive()
        assert worker.lease.released and not ledger.workers and not ledger.cuda_bytes
    asyncio.run(exercise())


def test_collection_typed_tree_registry_metadata_without_any_runtime_attestation(monkeypatch):
    import gf_training.card_instance_collection as collection
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.card_instance_game_tree_runtime import validate_tree_metadata
    raw = controlled_metadata()
    safe, label = validate_tree_metadata(raw, actions=ACTIONS, rules=RULES)
    # Isolate metadata composition: no Simulator receipt/attestation is invented.
    sentinel = {'controlled-receipt-validation-not-run': True}
    monkeypatch.setattr(collection, 'validate_runtime_receipt', lambda value, **kw: sentinel)
    args = {'actions': ACTIONS, 'config': {}, 'runtime_evidence': [sentinel], 'schedule': [{}], 'inventory': [{}],
        'policies': [label, 'random-card-instance-v2'], 'bots_revision': 'unadmitted-schema-example',
        'policy_bindings': {label: RuntimePolicyBinding('lightgbm', safe), 'random-card-instance-v2': RuntimePolicyBinding('random')},
        'tree_inputs': {'input_contract': 'player-card-refs-v2', 'policies': {label: raw}}}
    result = collection.collection_metadata(**args)
    assert result['tree_inputs']['policies'][label] == raw and 'neural_inputs' not in result
    assert result['model_version'] == '3'
    assert result['source_eligibility'] == 'not-authorized-by-collection-metadata'
    with pytest.raises(ContractError):
        collection.collection_metadata(**{**args, 'tree_inputs': None, 'neural_inputs': args['tree_inputs']})
    with pytest.raises(ContractError):
        collection.collection_metadata(**{**args, 'tree_inputs': {**args['tree_inputs'], 'policies': {}}})


def test_private_child_loads_only_own_bytes_and_opaque_pair_reference(tmp_path, monkeypatch):
    import hashlib

    from gf_models import structured_tree_artifacts as artifacts
    from gf_models.card_instance_game_tree_runtime import _OwnGameTreePolicyV3
    raw = controlled_metadata()
    manifest, options = raw['inference_manifest'], raw['policy_options']
    text = b'x'*100  # controlled bytes, explicitly no native model or actual GAME pair
    sha = hashlib.sha256(text).hexdigest()
    manifest['model_file']['sha256'] = sha
    manifest['tree_descriptor']['model_sha256'] = sha
    metadata = (canonical(manifest)+'\n').encode()
    options['expected_manifest_digest'] = hashlib.sha256(metadata).hexdigest()
    options['checkpoint'] = str(tmp_path)
    (tmp_path/'inference.json').write_bytes(metadata)
    (tmp_path/'model.txt').write_bytes(text)
    seen = {}
    def execution_primitive(own_metadata, own_text, **kwargs):
        seen.update(metadata=own_metadata, text=own_text, options=kwargs)
        return SimpleNamespace(descriptor=SimpleNamespace(model_bytes=100))
    monkeypatch.setattr(artifacts, '_load_game_own_model_v3', execution_primitive)
    policy = _OwnGameTreePolicyV3(options, actions=ACTIONS, rules=RULES,
        managed_limit=2*1024**3, deadline=time.monotonic()+10)
    assert seen['metadata'] == metadata and seen['text'] == text
    assert type(seen['metadata']) is bytes and type(seen['text']) is bytes
    assert seen['options']['own_pair_reference'] == options['training_pair']
    assert not {'directory', 'source', 'dataset', 'labels', 'parent'} & set(seen['options'])
    assert {path.name for path in policy._scan.files} == {'inference.json', 'model.txt'}
    assert not hasattr(policy.tree, 'training_binding')
    (tmp_path/'model.txt').write_bytes(b'y'*100)
    with pytest.raises(ContractError):
        policy._scan.finalize()


def test_tree_uncertain_io_retirement_retains_live_reservation_and_primary_error():
    from gf_simulator_client.structured_resources import ReservationLedger
    async def exercise():
        worker = object.__new__(CardInstanceParticipantWorker)
        worker._tree = worker._tree_failed = True
        worker.closed, worker._close_task = False, None
        worker._session_deadline = time.monotonic()+10
        worker.health = {}
        ledger = ReservationLedger()
        worker.lease = ledger.reserve_worker(0)
        worker.lease._cleanup_owner = worker
        process = SimpleNamespace(pid=12345, poll=lambda: 0, stdin=None, stdout=None)
        worker.process = process
        worker.lease.spawned(process)
        worker.io_thread = SimpleNamespace(ident=1, join=lambda timeout: None, is_alive=lambda: True)
        primary = RuntimeError('original admitted operation failure')
        with pytest.raises(RuntimeError, match='original admitted operation failure') as raised:
            await worker._abort(primary)
        assert raised.value is primary
        assert not worker.lease.released and len(ledger.workers) == 1
        assert worker.health['abort_cleanup_error'] == 'RuntimeError'
        first = worker._close_task
        worker.io_thread.is_alive = lambda: False
        # The first failed cleanup budget/owner stays fixed; no second close may
        # invent successful retirement or silently release its quarantined lease.
        with pytest.raises(RuntimeError, match='I/O thread did not exit'):
            await worker.close()
        assert worker._close_task is first and not worker.lease.released
        assert len(ledger.workers) == 1
        # Clearing mutable transport attributes cannot authorize retirement.
        original_thread = worker.io_thread
        worker.io_thread = worker.process = None
        with pytest.raises(RuntimeError, match='I/O thread did not exit'):
            await worker.close()
        assert worker._structured_cleanup_owners == (process, original_thread)
        assert not worker.lease.released and len(ledger.workers) == 1
    asyncio.run(exercise())


def test_closed_late_exchange_cannot_return_success(monkeypatch):
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    async def exercise():
        worker = object.__new__(CardInstanceParticipantWorker)
        worker._tree, worker.closed = True, False
        worker._operation_deadline = time.monotonic()+10
        worker._tree_model = ConfiguredTreeModelV3()
        worker._tree_retained, worker._tree_reply_limit = 0, 65536
        worker._tree_timeout = 10.
        worker._parent_bank = worker._last_frame = None
        worker.initial = {'policy_options': {'tree_descriptor': {'model_bytes': 100}}}
        # Only the exchange/close race is exercised; no pair or policy is created.
        worker.artifact_inputs = object.__new__(RetainedGamePolicyArtifacts)
        monkeypatch.setattr(RetainedGamePolicyArtifacts, 'guard', lambda self, **kw: None)
        async def close_during_exchange(self, payload):
            await asyncio.sleep(0)
            self.closed = True
            return {'late': True}
        monkeypatch.setattr(StructuredParticipantWorker, '_roundtrip', close_during_exchange)
        with pytest.raises(TimeoutError, match='closed'):
            await worker._roundtrip({'control': True})
    asyncio.run(exercise())


def test_tree_full_admission_and_one_byte_short_model_workspace_refusal():
    from gf_models.card_instance_game_tree_runtime import game_tree_settings
    from gf_models.structured_tree_config import resource_estimate
    options = controlled_options()
    caps, costs, resident, temporary = game_tree_settings(options, actions=ACTIONS, rules=RULES)
    assert caps.max_positions == 18 and caps.max_candidates == 32768
    assert caps.max_interval_events == 4096 and caps.max_episode_events == 262144 and caps.max_decisions is None
    assert costs.candidate_chunk == 256 and costs.position_capacity == 18
    assert resident == temporary == 0
    required = resource_estimate(ConfiguredTreeModelV3(), model_bytes=100)['total_bytes']
    options['resources']['max_owned_bytes'] = required-1
    with pytest.raises(ContractError, match='resource admission exceeded'):
        game_tree_settings(options, actions=ACTIONS, rules=RULES)


def test_private_tree_argv_mismatch_refuses_before_own_file_allocation(monkeypatch):
    from gf_models.card_instance_game_tree_runtime import _OwnGameTreePolicyV3
    from gf_models.structured_sealed import SealedScan
    monkeypatch.setattr(SealedScan, '__init__', lambda *a, **kw: pytest.fail('own file scan allocated'))
    with pytest.raises(ContractError, match='managed cap differ'):
        _OwnGameTreePolicyV3(controlled_options(), actions=ACTIONS, rules=RULES,
            managed_limit=1, deadline=time.monotonic()+10)


def test_shared_chunk_kernel_scores_all_rows_and_keeps_first_argmax_tie():
    import numpy as np

    from gf_models.structured_tree_policy import _select
    observed = []
    rows = np.zeros((512, 1), dtype=np.float32)
    rows[255] = rows[256] = 10.
    class Prepared:
        def iter_chunks(self, candidates):
            for start in range(0, len(candidates), 256):
                yield start, candidates[start:start+256]
    def predict(chunk, *, num_threads):
        assert num_threads == 1
        observed.extend(chunk[:, 0].tolist())
        return chunk[:, 0]
    tree = SimpleNamespace(booster=SimpleNamespace(predict=predict), model=ConfiguredTreeModelV3())
    assert _select(tree, Prepared(), rows) == 255
    assert len(observed) == 512 and observed[255:257] == [10., 10.]


def test_mixed_nn_tree_collection_validates_each_own_typed_metadata(tmp_path, monkeypatch):
    import hashlib

    import gf_training.card_instance_collection as collection
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.card_instance_game_tree_runtime import validate_tree_metadata
    pytest.importorskip('torch')
    from tests.test_card_instance_neural import ACTIONS as nn_actions
    from tests.test_card_instance_neural import saved
    owner, nn_label, nn_binding, *_ = saved(tmp_path)
    try:
        raw = controlled_metadata()
        raw['policy_options']['action_config_hash'] = raw['inference_manifest']['action_config_hash'] = digest(asdict(nn_actions))
        metadata_bytes = (canonical(raw['inference_manifest'])+'\n').encode()
        sha = hashlib.sha256(metadata_bytes).hexdigest()
        raw['policy_options']['expected_manifest_digest'] = raw['files'][0]['sha256'] = sha
        raw['files'][0]['identity'][2] = len(metadata_bytes)
        safe, tree_label = validate_tree_metadata(raw, actions=nn_actions, rules=RULES)
        # No runtime is claimed; this test isolates typed registry composition.
        sentinel = {'controlled-receipt-validation-not-run': True}
        monkeypatch.setattr(collection, 'validate_runtime_receipt', lambda value, **kw: sentinel)
        result = collection.collection_metadata(actions=nn_actions, config={}, runtime_evidence=[sentinel],
            schedule=[{}], inventory=[{}], policies=[nn_label, tree_label], bots_revision='unadmitted-schema-example',
            policy_bindings={nn_label: nn_binding, tree_label: RuntimePolicyBinding('lightgbm', safe)},
            neural_inputs=owner.finalize(deadline=time.monotonic()+10),
            tree_inputs={'input_contract': 'player-card-refs-v2', 'policies': {tree_label: raw}})
        assert set(result['neural_inputs']['policies']) == {nn_label}
        assert set(result['tree_inputs']['policies']) == {tree_label}
        assert result['model_version'] == '3'
    finally:
        owner.release()


@pytest.mark.parametrize('phase', ['publish', 'release'])
def test_retained_input_mutation_during_publication_revokes_both_outputs(tmp_path, monkeypatch, phase):
    from gf_models.structured_sealed import SealedScan
    from gf_training.card_instance_collection import CollectionPublication
    original = tmp_path/'original.json'
    original.write_text('{"value":1}\n')
    scan = SealedScan(max_owned_bytes=1024**2, deadline=time.monotonic()+10)
    scan.object(original)
    output = tmp_path/'output'
    output.mkdir()
    owner = CollectionPublication(output, work_deadline=time.monotonic()+10, cleanup_deadline=time.monotonic()+20)
    method = '_publish' if phase == 'publish' else '_release'
    actual = getattr(owner, method)
    def mutate(*args, **kwargs):
        actual(*args, **kwargs)
        original.write_text('{"value":2}\n')
    monkeypatch.setattr(owner, method, mutate)
    with pytest.raises(ContractError):
        owner.commit({'healthy': True}, {'schema_version': 5}, retained_input_guard=scan.finalize)
    assert not owner.committed and owner.cleanup_complete
    assert all((output/name).read_bytes().startswith(b'!') for name in ('report.json', 'manifest.json'))


def test_post_release_guard_failure_keeps_primary_and_quarantines_uncertain_recovery_close(tmp_path, monkeypatch):
    import os

    from gf_training.card_instance_collection import CollectionPublication
    owner = CollectionPublication(tmp_path, work_deadline=time.monotonic()+10, cleanup_deadline=time.monotonic()+20)
    primary = RuntimeError('post-release retained input failure')
    calls, fault = [], []
    actual_close = os.close
    def guard():
        calls.append('guard')
        if len(calls) == 3:
            raise primary
    def close(descriptor):
        actual_close(descriptor)
        if len(calls) == 3 and not fault:
            fault.append(descriptor)
            raise OSError('uncertain recovery close after actual release')
    monkeypatch.setattr(os, 'close', close)
    with pytest.raises(RuntimeError, match='post-release retained input failure') as raised:
        owner.commit({'healthy': True}, {'schema_version': 5}, retained_input_guard=guard)
    assert raised.value is primary
    assert owner.quarantined and not owner.committed
    assert raised.value._card_instance_collection_cleanup_owner is owner
    assert any(handle.state == 'uncertain' for handle in owner._handles)
    assert all(output.invalidated for output in owner.outputs)
    assert fault and len(calls) == 3


@pytest.mark.parametrize('explicit_none', [False, True])
def test_old_publisher_none_guard_trace_and_bytes_are_exact(tmp_path, monkeypatch, explicit_none):
    from gf_training.card_instance_collection import CollectionPublication
    owner = CollectionPublication(tmp_path, work_deadline=time.monotonic()+10, cleanup_deadline=time.monotonic()+20)
    trace = []
    for name in ('_publish', '_release'):
        actual = getattr(owner, name)
        def traced(*args, _actual=actual, _name=name, **kwargs):
            trace.append(_name)
            return _actual(*args, **kwargs)
        monkeypatch.setattr(owner, name, traced)
    kwargs = {'retained_input_guard': None} if explicit_none else {}
    owner.commit({'healthy': True}, {'schema_version': 5}, **kwargs)
    assert trace == ['_publish', '_release'] and owner.committed
    assert (tmp_path/'report.json').read_bytes() == b'{"healthy":true}\n'
    assert (tmp_path/'manifest.json').read_bytes() == b'{"schema_version":5}\n'


def test_retained_bytes_are_actual_counters_and_released_owners_refuse():
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    neural = object.__new__(RetainedGameArtifacts)
    neural._released = neural._failed = False
    neural._scan = SimpleNamespace(retained=123)
    assert neural.retained_bytes == 123
    composite = object.__new__(RetainedGamePolicyArtifacts)
    composite._released = composite._failed = False
    composite._neural = neural
    composite._pairs = [SimpleNamespace(scan=SimpleNamespace(retained=456, files={}, inventories={}))]
    assert composite.retained_bytes == 579
    assert composite.guard_scratch_bytes > 1024**2
    neural._released = True
    with pytest.raises(ContractError):
        _ = neural.retained_bytes
    with pytest.raises(ContractError):
        _ = composite.retained_bytes
    composite._released = True
    with pytest.raises(ContractError):
        _ = composite.guard_scratch_bytes


def test_joint_guard_near_cap_refuses_before_any_original_read():
    composite = object.__new__(RetainedGamePolicyArtifacts)
    composite._released = composite._failed = False
    composite._deadline = time.monotonic()+10
    composite._neural = None
    scan = SimpleNamespace(retained=100, files={}, inventories={})
    composite._pairs = [SimpleNamespace(scan=scan, finalize=lambda: pytest.fail('unadmitted guard read'))]
    composite._limit = composite.retained_bytes+composite.guard_scratch_bytes-1
    with pytest.raises(ContractError, match='scratch admission exceeded'):
        composite.guard(deadline=composite._deadline)
    assert composite._failed


def test_near_cap_controller_reply_rejects_before_body_read():
    from gf_models.structured_tree_config import resource_estimate
    worker = object.__new__(CardInstanceParticipantWorker)
    worker._tree = True
    worker._tree_model = ConfiguredTreeModelV3()
    worker._parent_bank = worker._last_frame = None
    worker.initial = {'policy_options': {'tree_descriptor': {'model_bytes': 100}}}
    worker._tree_reply_limit = 65536
    base = resource_estimate(worker._tree_model, model_bytes=100)['total_bytes']
    worker._tree_retained = worker._tree_model.resources.max_owned_bytes-base-frame_decode_bytes(1024)+1
    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            if self.tell() >= 8:
                pytest.fail('unadmitted reply body read')
            return super().read(size)
    with pytest.raises(ContractError, match='before body read'):
        worker._read_reply(HeaderOnly((1024).to_bytes(8, 'big')))


def test_retired_worker_deregistration_frees_banks_only_after_actual_retirement():
    owner = object.__new__(RetainedGamePolicyArtifacts)
    owner._workers = []
    worker = SimpleNamespace(lease=SimpleNamespace(released=False), io_thread=None)
    owner._workers.append(worker)
    with pytest.raises(ContractError):
        owner.retire_worker(worker)
    assert owner._workers == [worker]
    worker.lease.released = True
    owner.retire_worker(worker)
    owner.retire_worker(worker)
    assert not owner._workers


def _controlled_shared_worker(branch, *, ledger):
    """Transport ownership primitive: no model, actual process or Source exists."""
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    cls = StructuredParticipantWorker if branch == 'protocol2' else CardInstanceParticipantWorker
    worker = object.__new__(cls)
    worker.health, worker.closed, worker._close_task = {}, False, None
    worker._tree = branch == 'tree'
    worker._tree_failed = True
    worker._session_deadline = time.monotonic()+10
    worker.model_id = {'baseline': 'random', 'nn': 'mlp', 'tree': 'lightgbm'}.get(branch, 'random')
    worker.artifact_inputs = None
    worker.lease = ledger.reserve_worker(256)
    worker.lease._cleanup_owner = worker
    worker.process = SimpleNamespace(pid=12345, poll=lambda: 0, stdin=None, stdout=None)
    worker.lease.spawned(worker.process)
    worker.timeout = 1.
    return worker


@pytest.mark.parametrize('branch', ['protocol2', 'baseline', 'nn', 'tree'])
def test_shared_cleanup_failure_blocks_replacement_and_preserves_same_owners(branch):
    import threading

    from gf_simulator_client.structured_resources import ReservationLedger
    async def exercise():
        ledger = ReservationLedger(global_cuda_bytes=1024, max_workers=1)
        worker = _controlled_shared_worker(branch, ledger=ledger)
        worker.lease.reserve_decision(128)
        release = threading.Event()
        thread = threading.Thread(target=release.wait, daemon=True)
        worker.io_thread = thread
        process = worker.process
        thread.start()
        try:
            first = asyncio.create_task(worker.close(.02))
            second = asyncio.create_task(worker.close(.02))
            errors = await asyncio.gather(first, second, return_exceptions=True)
            assert all(isinstance(error, RuntimeError) for error in errors)
            assert errors[0] is errors[1]
            owned_close = worker._close_task
            assert not worker.lease.released and ledger.cuda_bytes == 384
            assert len(ledger.workers) == 1 and worker.lease._cleanup_owner is worker
            assert worker._structured_cleanup_owners == (process, thread)
            for action in (lambda: ledger.reserve_worker(0), lambda: ledger.reserve_learner(0, object()),
                           lambda: ledger.begin_update(0)):
                with pytest.raises(ContractError):
                    action()
            primary = RuntimeError('original policy operation failure')
            with pytest.raises(RuntimeError, match='original policy operation failure') as raised:
                await worker._abort(primary)
            assert raised.value is primary and raised.value._structured_cleanup_owner is worker
            # Neither missing callbacks nor the closed flag prove that the SAME
            # original physical owners have retired after the first failure.
            worker.process = worker.io_thread = None
            with pytest.raises(RuntimeError, match='I/O thread did not exit') as repeated:
                await worker.close()
            assert repeated.value is errors[0] and worker._close_task is owned_close
            assert worker._structured_cleanup_owners == (process, thread)
            assert thread.is_alive() and not worker.lease.released and len(ledger.workers) == 1
        finally:
            release.set()
            thread.join(timeout=1)
        # First cleanup failure is quarantined; clearing attributes cannot turn
        # its fixed failed task into a successful cleanup-only retry.
        with pytest.raises(RuntimeError, match='I/O thread did not exit'):
            await worker.close()
        assert not worker.lease.released and len(ledger.workers) == 1
    asyncio.run(exercise())


@pytest.mark.parametrize('branch', ['protocol2', 'baseline', 'nn', 'tree'])
def test_shared_exchange_concurrent_close_never_returns_late_success(branch):
    import threading

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    async def exercise():
        ledger = ReservationLedger(max_workers=1)
        worker = _controlled_shared_worker(branch, ledger=ledger)
        started, released = threading.Event(), threading.Event()
        class Source:
            def close(self):
                released.set()
        worker.process.stdin, worker.process.stdout = io.BytesIO(), Source()
        def reply(source):
            started.set()
            if not released.wait(timeout=2):
                raise TimeoutError('controlled exchange did not close')
            return {'late': True}
        worker._read_reply = reply
        pending = asyncio.create_task(StructuredParticipantWorker._roundtrip(worker, {'control': True}))
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(.001)
            await worker.close(.2)
            with pytest.raises(RuntimeError, match='closed before reply retirement'):
                await pending
            assert worker.lease.released and not ledger.workers
            assert worker.io_thread is not None and not worker.io_thread.is_alive()
        finally:
            released.set()
            if not pending.done():
                await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(exercise())


def _controlled_neural_scan_owner(path):
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_sealed import SealedScan
    path.write_text('{"original":1}\n')
    scan = SealedScan(max_owned_bytes=16*1024**2, deadline=time.monotonic()+10)
    scan.object(path)
    neural = object.__new__(RetainedGameArtifacts)
    neural._released = neural._failed = False
    neural._scan = scan
    neural._entries, neural._models = {}, {}
    neural._finalization_deadline = time.monotonic()+10
    composite = object.__new__(RetainedGamePolicyArtifacts)
    composite._released = composite._failed = False
    composite._neural, composite._pairs = neural, []
    composite._deadline, composite._limit = time.monotonic()+10, 16*1024**2
    return neural, composite


@pytest.mark.parametrize('phase', ['publish', 'release'])
def test_composite_neural_mutation_rechecks_public_owner_and_revokes_outputs(tmp_path, monkeypatch, phase):
    from gf_training.card_instance_collection import CollectionPublication
    original = tmp_path/'neural-original.json'
    neural, composite = _controlled_neural_scan_owner(original)
    finalizations = []
    actual_finalize = neural.finalize
    def finalize(**kwargs):
        finalizations.append(kwargs['deadline'])
        return actual_finalize(**kwargs)
    monkeypatch.setattr(neural, 'finalize', finalize)
    output = tmp_path/'output'
    output.mkdir()
    publisher = CollectionPublication(output, work_deadline=time.monotonic()+10, cleanup_deadline=time.monotonic()+20)
    method = '_publish' if phase == 'publish' else '_release'
    actual = getattr(publisher, method)
    def mutate(*args, **kwargs):
        actual(*args, **kwargs)
        original.write_text('{"original":2}\n')
    monkeypatch.setattr(publisher, method, mutate)
    with pytest.raises(ContractError):
        publisher.commit({'healthy': True}, {'schema_version': 5},
            retained_input_guard=lambda: composite.guard(deadline=composite._deadline+100))
    assert len(finalizations) == (2 if phase == 'publish' else 3)
    assert all(deadline == composite._deadline for deadline in finalizations)
    assert composite._failed and neural._failed
    assert not publisher.committed and publisher.cleanup_complete
    assert all((output/name).read_bytes().startswith(b'!') for name in ('report.json', 'manifest.json'))


def test_composite_neural_guard_budget_precedes_public_finalize_or_original_read(tmp_path, monkeypatch):
    neural, composite = _controlled_neural_scan_owner(tmp_path/'neural-original.json')
    composite._limit = composite.retained_bytes+composite.guard_scratch_bytes-1
    monkeypatch.setattr(neural, 'finalize', lambda **kw: pytest.fail('unadmitted NN finalize/read'))
    with pytest.raises(ContractError, match='scratch admission exceeded'):
        composite.guard(deadline=composite._deadline)
    assert composite._failed


def test_neural_post_release_mutation_and_uncertain_recovery_close_preserve_first_error(tmp_path, monkeypatch):
    import os

    from gf_training.card_instance_collection import CollectionPublication
    original = tmp_path/'neural-original.json'
    neural, composite = _controlled_neural_scan_owner(original)
    output = tmp_path/'output'
    output.mkdir()
    publisher = CollectionPublication(output, work_deadline=time.monotonic()+10, cleanup_deadline=time.monotonic()+20)
    actual_release, actual_close = publisher._release, os.close
    def mutate():
        actual_release()
        original.write_text('{"original":2}\n')
    fault = []
    def close(descriptor):
        actual_close(descriptor)
        if composite._failed and not fault:
            fault.append(descriptor)
            raise OSError('controlled uncertain recovery close')
    monkeypatch.setattr(publisher, '_release', mutate)
    monkeypatch.setattr(os, 'close', close)
    with pytest.raises(ContractError) as raised:
        publisher.commit({'healthy': True}, {'schema_version': 5},
            retained_input_guard=lambda: composite.guard(deadline=composite._deadline))
    assert 'sealed' in str(raised.value) and composite._failed and neural._failed
    assert publisher.quarantined and not publisher.committed
    assert raised.value._card_instance_collection_cleanup_owner is publisher
    assert fault and all(record.invalidated for record in publisher.outputs)


@pytest.mark.parametrize('branch', ['protocol2', 'baseline', 'nn', 'tree'])
@pytest.mark.parametrize('thread_alive', [True, False], ids=['live-original', 'retired-original'])
def test_first_pipe_close_cannot_hide_saved_original_thread(branch, thread_alive):
    import threading

    from gf_simulator_client.structured_resources import ReservationLedger
    async def exercise():
        ledger = ReservationLedger(global_cuda_bytes=1024, max_workers=1)
        worker = _controlled_shared_worker(branch, ledger=ledger)
        worker.lease.reserve_decision(128)
        release = threading.Event()
        original_thread = threading.Thread(target=release.wait, daemon=True)
        worker.io_thread = original_thread
        original_process = worker.process
        class ClearingPipe:
            calls = 0
            closed = False
            def close(self):
                self.calls += 1
                self.closed = True
                # Exercise the UNPATCHED inherited physical kernel: it observes
                # None and skips its join, while our original thread still lives.
                worker.io_thread = None
        pipe = ClearingPipe()
        worker.process.stdin = pipe
        original_thread.start()
        if not thread_alive:
            release.set()
            original_thread.join(timeout=1)
        try:
            if thread_alive:
                with pytest.raises(RuntimeError, match='physical cleanup is unconfirmed') as raised:
                    await worker.close(.02)
                first_task = worker._close_task
                assert worker.io_thread is None and original_thread.is_alive()
                assert not worker.lease.released and ledger.cuda_bytes == 384
                assert worker._structured_cleanup_owners == (original_process, original_thread)
                assert raised.value._structured_cleanup_owner is worker
                for action in (lambda: ledger.reserve_worker(0), lambda: ledger.reserve_learner(0, object()),
                               lambda: ledger.begin_update(0)):
                    with pytest.raises(ContractError):
                        action()
                with pytest.raises(RuntimeError) as second:
                    await worker.close()
                assert second.value is raised.value and worker._close_task is first_task
                assert not worker.lease.released and len(ledger.workers) == 1
                primary = RuntimeError('first admitted operation failure')
                with pytest.raises(RuntimeError) as operation:
                    await worker._abort(primary)
                assert operation.value is primary and primary._structured_cleanup_owner is worker
            else:
                await worker.close(.02)
                assert not original_thread.is_alive() and worker.lease.released
                assert not ledger.workers and ledger.cuda_bytes == 0
                replacement = ledger.reserve_worker(0)
                replacement.cancel_before_spawn()
                learner = ledger.reserve_learner(0, object())
                learner.cancel_before_spawn()
                assert not ledger.workers and ledger.learner is None
            assert pipe.calls == 1 and pipe.closed
        finally:
            release.set()
            original_thread.join(timeout=1)
        assert not original_thread.is_alive()
    asyncio.run(exercise())


def _nested_architecture_control(field, alias):
    import copy

    raw = controlled_metadata()
    options = raw['policy_options']
    if alias != 'shared':
        options['model'] = copy.deepcopy(options['model'])
    if alias == 'foreign-manifest':
        manifest_model = copy.deepcopy(raw['inference_manifest']['model'])
        manifest_model['network']['num_leaves'] += 1
        raw['inference_manifest']['model'] = manifest_model
    values = {'checkpoint': '/unadmitted/foreign-control',
        'inference': {'num_threads': 999}, 'resources': {'max_owned_bytes': 0},
        'unknown_architecture': {'unadmitted': True}}
    options['model'][field] = values[field]
    return raw


def _assert_closed_architecture(call):
    with pytest.raises(ContractError) as raised:
        call()
    assert str(raised.value) == "invalid fields; required=['encoder', 'initialization', 'model_id', 'network', 'version']"


@pytest.mark.parametrize('field', ['checkpoint', 'inference', 'resources', 'unknown_architecture'])
@pytest.mark.parametrize('alias', ['shared', 'independent', 'foreign-manifest'])
def test_own_model_architecture_is_closed_independently_of_manifest(field, alias):
    from gf_models.card_instance_game_tree_runtime import validate_tree_metadata
    raw = _nested_architecture_control(field, alias)
    original = canonical(raw)
    _assert_closed_architecture(lambda: safe_tree_options_game_v3(raw['policy_options'], actions=ACTIONS, rules=RULES))
    _assert_closed_architecture(lambda: validate_tree_metadata(raw, actions=ACTIONS, rules=RULES))
    assert canonical(raw) == original  # Rejection cannot silently overwrite or retain controls as architecture.


@pytest.mark.parametrize('field', ['checkpoint', 'inference', 'resources'])
def test_even_identical_outer_controls_are_forbidden_inside_architecture(field, monkeypatch):
    import copy

    import gf_models.card_instance_game_tree_runtime as runtime
    raw = controlled_metadata()
    options = raw['policy_options']
    options['model'] = copy.deepcopy(options['model'])
    options['model'][field] = copy.deepcopy(options[field])
    monkeypatch.setattr(runtime, 'configured_tree_v3', lambda *a, **kw: pytest.fail('execution merge ran before architecture closure'))
    _assert_closed_architecture(lambda: runtime.safe_tree_options_game_v3(options, actions=ACTIONS, rules=RULES))


def _controlled_tree_header(raw):
    from gf_common.card_instance import CATALOG_META, GENERATOR_VERSION
    from gf_common.card_instance_rules import rule_signature
    from gf_common.card_instance_runtime import game_binding
    from gf_common.domain import END_PROJECTION_VERSION
    from gf_common.structured_tokens import signature
    from gf_models.card_instance_game_tree_runtime import game_tree_settings
    from gf_training.journal import REWARD_VERSION
    caps, costs, _, _ = game_tree_settings(controlled_options(), actions=ACTIONS, rules=RULES)
    options = raw['policy_options']
    label = 'lightgbm-tree-v3-'+options['expected_manifest_digest']+'-'+digest({key: value for key, value in options.items() if key != 'checkpoint'})
    return {**game_binding(ACTIONS.action_contract, rule_signature(ACTIONS.action_contract, RULES)),
        'kind': 'header', 'schema_version': 6, 'feature_version': LAYOUT.input_recipe,
        'input_recipe': LAYOUT.input_recipe, 'reward_version': REWARD_VERSION,
        'policy_input_version': ACTIONS.action_contract.policy_input_version,
        'end_projection_version': END_PROJECTION_VERSION, 'catalog': CATALOG_META,
        'episode_id': 'controlled', 'viewer': 1, 'policy': label, 'run_id': 'controlled',
        'game_id': 'controlled', 'action_config_hash': digest(asdict(ACTIONS)),
        'generator_version': GENERATOR_VERSION, 'bank_episode': 'controlled',
        'caps': asdict(caps), 'numeric_costs': asdict(costs), 'model_id': 'lightgbm',
        'model_version': '3', 'generation_schema': None, 'aggregate_window': 128,
        'token_signature': signature(128, layout=LAYOUT), 'sampling_seed': 0,
        'use_action_options': True, 'protocol_version': 3, 'actions': asdict(ACTIONS),
        'runtime_evidence': {'controlled-receipt-validation-not-run': True}, 'tree': raw}


@pytest.mark.parametrize('field', ['checkpoint', 'inference', 'resources', 'unknown_architecture'])
@pytest.mark.parametrize('alias', ['shared', 'independent', 'foreign-manifest'])
def test_journal_header_and_collection_reject_nested_architecture_controls(field, alias, monkeypatch):
    import gf_training.card_instance_collection as collection
    import gf_training.card_instance_journal as journal
    from gf_common.domain import RuntimePolicyBinding
    raw = _nested_architecture_control(field, alias)
    header = _controlled_tree_header(raw)
    # Only metadata composition is exercised: no runtime attestation or file is
    # manufactured, and the actual architecture validators remain unpatched.
    sentinel = {'controlled-receipt-validation-not-run': True}
    monkeypatch.setattr(journal, 'validate_runtime_receipt', lambda *a, **kw: sentinel)
    monkeypatch.setattr(collection, 'validate_runtime_receipt', lambda *a, **kw: sentinel)
    monkeypatch.setattr(journal, 'EventBank', lambda *a, **kw: pytest.fail('bank allocated before architecture refusal'))
    _assert_closed_architecture(lambda: journal._Context(ACTIONS, RULES).validate_header(header))
    label = header['policy']
    _assert_closed_architecture(lambda: collection.collection_metadata(actions=ACTIONS, config={},
        runtime_evidence=[sentinel], schedule=[{}], inventory=[{}], policies=[label],
        bots_revision='controlled-unadmitted-metadata',
        policy_bindings={label: RuntimePolicyBinding('lightgbm', raw['policy_options'])},
        tree_inputs={'input_contract': 'player-card-refs-v2', 'policies': {label: raw}}))
