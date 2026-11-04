"""UNADMITTED ownership/control probes: parent and raw admission are mocked.

The shared raw C5/J6 parser is tested separately on actual structural files.
These probes never admit a formal parent, executed runtime or qualified Source.
"""
import asyncio
import copy
import gc
import json
import threading
import time
import weakref
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.card_instance_collection import FINAL_VALIDATION
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical
from gf_common.schedule import serialize_schedule
from gf_common.structured import EventBank
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_models.arrays import sha256
from gf_training import card_instance_final_validation as module
from gf_training.card_instance_collection import CollectionPublication
from gf_training.structured_rollout import game_episode_records
from gf_training.structured_source import ComponentCollection, ComponentEpisode
from tests.test_card_instance_final_validation_owner import actor_context


def controlled_originals(context, parent, monkeypatch):
    """Persist real control files; mock ONLY raw parser and parent qualification."""
    directory = context.plan.output
    directory.mkdir()
    plan, projected = context.plan, context.projected
    report = {'schema_version': 1, 'healthy': True, 'planned_games': 64, 'natural_denominator': 64,
        'comparison_valid': True, 'failures': {}, 'health_failures': {}, 'group_health_failures': {}, 'wall_seconds': 0.}
    manifest = {'schema_version': 5, 'status': 'complete', 'collection_role': FINAL_VALIDATION.role,
        'config': projected.effective, 'config_hash': projected.sha256, 'inputs': list(projected.inputs),
        'config_projection': plan.config_projection, 'validation_invocation': plan.validation_invocation,
        'elapsed_seconds': .2, 'linkage': plan.linkage, 'schedule': serialize_schedule(plan.groups, 4),
        'policy_bindings': {key: asdict(value) for key, value in plan.bindings.items()},
        'code': parent.state['identity']['code'],
        'runtime_evidence': [{'scope': 'UNADMITTED-NOT-an-executed-runtime-receipt', 'binary': {'sha256': 'a' * 64}}],
        'report_sha256': __import__('hashlib').sha256((canonical(report) + '\n').encode()).hexdigest()}
    publication = CollectionPublication(directory, work_deadline=context.plan.wall_deadline - 1, cleanup_deadline=context.plan.wall_deadline)
    context.begin_collection(publication, report, manifest)
    manifest = json.loads((directory / 'manifest.json.tmp').read_text())
    episodes = []
    for group in plan.groups:
        for spec in group.episodes:
            bank = EventBank(spec.learner_viewer, spec.episode_id, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            records = game_episode_records(bank.view(), (), (), (), 'natural', 'draw',
                actions=projected.config.actions, rules=card_instance_rules())
            episodes.append(ComponentEpisode(spec.episode_id, spec.learner_viewer, 'mlp', '3', records))
    def inspect(owner, actions, expected_split, *, _game, _game_role, _game_pending=None):
        assert _game and _game_role is FINAL_VALIDATION and expected_split == 'validation'
        name = 'manifest.json' if _game_pending is None else _game_pending.manifest_name()
        raw = owner.object(directory / name)
        owner.object(directory / 'report.json')
        owner.inventory('**/*')
        return ComponentCollection(directory, tuple(episodes), canonical(raw).encode(), canonical({'qualified': False, 'scope': 'UNADMITTED'}).encode())
    monkeypatch.setattr(module, '_inspect_component_collection', inspect)
    return manifest, report


def test_unadmitted_live_and_readonly_original_validator_retains64_empty_endpoints(tmp_path, monkeypatch):
    value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def run():
            await context.__aenter__()
            _manifest, _report = controlled_originals(context, parent, monkeypatch)
            source = context.accept_collection(context._issued)
            assert len(source.selected_records) == 64 and all(not row.frames for row in source.selected_records)
            facts = source.facts
            assert facts['games'] == facts['zero_endpoint_games'] == 64 and facts['selected_decisions'] == 0
            assert facts['qualified'] is False and facts['parent_qualification'] == 'not_performed'
            facts['config_projection']['recipe'] = 'MUTATED'
            assert source.facts['config_projection']['recipe'] != 'MUTATED'
            await context.__aexit__(None, None, None)
            assert context.cleanup_complete
            with pytest.raises(ContractError): source.recheck()
            with module.open_game_final_validation(value, parent, directory, deadline=time.monotonic() + 60, max_owned_bytes=1024**3) as reopened:
                assert len(reopened.selected_records) == 64 and reopened.facts['qualified'] is False
            with pytest.raises(ContractError): reopened.recheck()
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('mutation', ['inputs', 'config', 'code', 'generation', 'inference', 'seats', 'registry', 'schedule', 'projection', 'clock', 'cost', 'report-health', 'report-cost', 'returned-fields', 'returned-cost'])
def test_unadmitted_original_validator_rejects_control_mutations_and_stays_failed(tmp_path, monkeypatch, mutation):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def run():
            await context.__aenter__()
            manifest, report = controlled_originals(context, parent, monkeypatch)
            manifest = copy.deepcopy(manifest)
            returned = context._issued.report
            if mutation == 'inputs': manifest['inputs'] = []
            elif mutation == 'config': manifest['config']['run']['concurrency'] = 1
            elif mutation == 'code': manifest['code'] = {'scope': 'foreign'}
            elif mutation == 'generation': manifest['linkage']['final_generation'] = 999
            elif mutation == 'inference': manifest['linkage']['final_inference_sha256'] = 'b' * 64
            elif mutation == 'seats': manifest['linkage']['learner_seats'][next(iter(manifest['linkage']['learner_seats']))] = 0
            elif mutation == 'registry': manifest['policy_bindings'].pop('random-card-instance-v2')
            elif mutation == 'schedule': manifest['schedule'].pop()
            elif mutation == 'projection': manifest['config_projection']['parent_config_inputs_sha256'] = 'b' * 64
            elif mutation == 'clock': manifest['validation_invocation']['execution_deadline_monotonic'] += 1
            elif mutation == 'cost': manifest['elapsed_seconds'] = float('nan')
            elif mutation == 'report-health': report['healthy'] = False
            elif mutation == 'report-cost': report['wall_seconds'] = .4
            elif mutation == 'returned-fields': returned['extra'] = True
            else: returned['wall_seconds'] = 8000.
            (directory / 'report.json').write_text(canonical(report))
            manifest['report_sha256'] = sha256(directory / 'report.json')
            # Nonfinite input exercises the strict original JSON reader.
            (directory / 'manifest.json.tmp').write_text(json.dumps(manifest))
            context._issued._report = canonical(returned).encode()
            with pytest.raises((ContractError, ValueError)):
                context.accept_collection(context._issued)
            with pytest.raises(ContractError, match='poisoned'):
                context.validate_collection(context.plan, context.ledger, context.projected, deadline=time.monotonic() + 60)
            with pytest.raises((ContractError, ValueError)):
                await context.aclose()
            assert context.cleanup_complete
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('mutation', ['bytes-restored', 'root', 'inventory'])
def test_unadmitted_retained_original_guard_poison_survives_repair(tmp_path, monkeypatch, mutation):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def run():
            await context.__aenter__()
            _, _report = controlled_originals(context, parent, monkeypatch)
            source = context.accept_collection(context._issued)
            if mutation == 'bytes-restored':
                path = directory / 'report.json'; raw = path.read_bytes(); path.write_bytes(raw + b'\n'); path.write_bytes(raw)
            elif mutation == 'root':
                moved = directory.with_name('moved'); directory.rename(moved); directory.mkdir()
                for path in moved.iterdir(): path.rename(directory / path.name)
            else: (directory / 'extra').write_text('UNADMITTED')
            with pytest.raises(ContractError): source.recheck()
            with pytest.raises(ContractError, match='poisoned'): source.recheck()
            with pytest.raises(ContractError): await context.__aexit__(None, None, None)
            assert context.cleanup_complete
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


def test_actor_borrow_overlap_token_and_actual_object_identity_are_resource_only(tmp_path, monkeypatch):
    value, parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        with pytest.raises(ContractError, match='resource borrow'):
            module.GameFinalValidationContext(value, parent, tmp_path / 'other', ledger, deadline=time.monotonic() + 7200)
        with pytest.raises(ContractError, match='token'):
            module._release_borrow(parent, object())
        same_path_distinct_object = object.__new__(type(parent))
        same_path_distinct_object._active, same_path_distinct_object._entered = True, True
        same_path_distinct_object._closed, same_path_distinct_object._poisoned = False, False
        same_path_distinct_object.directory, same_path_distinct_object._token = parent.directory, parent._token
        same_path_distinct_object.deadline = parent.deadline
        token = module._acquire_borrow(same_path_distinct_object)
        module._release_borrow(same_path_distinct_object, token)
        assert module._BORROWS[parent] is context._borrow_token
        asynchronous = asyncio.run(context.aclose())
        assert asynchronous is None and context.cleanup_complete and parent not in module._BORROWS
        key = object.__new__(type(parent)); key.__dict__.update(same_path_distinct_object.__dict__)
        module._acquire_borrow(key)
        reference = weakref.ref(key)
        del key
        gc.collect()
        assert reference() is None  # Slot values own neither parent nor context.
    finally:
        scope.__exit__(None, None, None)


def test_borrow_check_set_is_atomic_under_threads(tmp_path, monkeypatch):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    asyncio.run(context.aclose())
    barrier, successful, failures = threading.Barrier(2), [], []
    def attempt():
        barrier.wait()
        try: successful.append(module._acquire_borrow(parent))
        except ContractError as error: failures.append(error)
    threads = [threading.Thread(target=attempt) for _ in range(2)]
    try:
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=5)
        assert len(successful) == len(failures) == 1
        module._release_borrow(parent, successful[0])
    finally:
        scope.__exit__(None, None, None)


def test_unreaped_actor_keeps_borrow_and_parent_reachable_until_confirmed_cleanup(tmp_path, monkeypatch):
    _value, parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        async def run():
            await context.__aenter__()
            _, _report = controlled_originals(context, parent, monkeypatch)
            source = context.accept_collection(context._issued)
            ledger.workers[1] = SimpleNamespace(scope='UNADMITTED-pending-actor-not-an-actual-process')
            with pytest.raises(ContractError) as error:
                await context.aclose()
            assert error.value._game_final_validation_cleanup_owner is context
            assert not context.cleanup_complete and module._BORROWS[parent] is context._borrow_token
            with pytest.raises(ContractError): source.recheck()
            with pytest.raises(ContractError, match='resource borrow'):
                module._acquire_borrow(parent)
            ledger.workers.clear()
            with pytest.raises(ContractError): await context.aclose()
            assert context.cleanup_complete and parent not in module._BORROWS
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


def test_readonly_source_uncertain_close_keeps_same_borrow_and_retry_capability(tmp_path, monkeypatch):
    value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def prepare():
            await context.__aenter__()
            controlled_originals(context, parent, monkeypatch)
            context.accept_collection(context._issued)
            await context.__aexit__(None, None, None)  # UNADMITTED raw/parent control probes only.
        asyncio.run(prepare())
        first = OSError('UNADMITTED Source close before retirement')
        source = None
        try:
            with module.open_game_final_validation(value, parent, directory, deadline=time.monotonic() + 60, max_owned_bytes=1024**3) as opened:
                source = opened
                original_close = source.close
                monkeypatch.setattr(source, 'close', lambda: (_ for _ in ()).throw(first))
        except OSError as error:
            assert error is first and error._game_final_validation_cleanup_owner is source
        else:
            pytest.fail('uncertain Source close accepted')
        assert not source.cleanup_complete and parent in module._BORROWS
        with pytest.raises(ContractError, match='poisoned'): source.recheck()
        with pytest.raises(ContractError, match='resource borrow'):
            module._acquire_borrow(parent)
        monkeypatch.setattr(source, 'close', original_close)
        with pytest.raises(OSError) as closed: source.close()
        assert closed.value is first
        assert source.cleanup_complete and parent not in module._BORROWS
    finally:
        scope.__exit__(None, None, None)


def test_live_source_uncertain_close_keeps_controls_and_exclusion_until_retry(tmp_path, monkeypatch):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        async def run():
            await context.__aenter__()
            _, _report = controlled_originals(context, parent, monkeypatch)
            source = context.accept_collection(context._issued)
            original_close = source.close
            first = OSError('UNADMITTED Source retirement uncertainty')
            monkeypatch.setattr(source, 'close', lambda: (_ for _ in ()).throw(first))
            with pytest.raises(OSError) as error: await context.aclose()
            assert error.value is first and error.value._game_final_validation_cleanup_owner is context
            assert not context.cleanup_complete and parent in module._BORROWS
            with pytest.raises(ContractError, match='poisoned'): source.recheck()
            monkeypatch.setattr(source, 'close', original_close)
            with pytest.raises(OSError): await context.aclose()
            assert source.cleanup_complete and context.cleanup_complete and parent not in module._BORROWS
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)
