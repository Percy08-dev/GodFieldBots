"""Isolated scores-only process components; no Simulator or performance claim."""
import asyncio
import io
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_fixture import CONTRACT, origin
from gf_common.domain import ContractError, canonical
from gf_common.structured import CandidateTable
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.card_instance_tree_scoring import COMPONENT, DIAGNOSTIC
from gf_models.structured_tree_artifacts import load_artifact_v3
from gf_models.structured_tree_config import ConfiguredTreeModelV3
from gf_models.structured_tree_pairing import inspect_training_v3
from gf_models.structured_tree_policy import ComponentStructuredTreePolicyV3
from gf_simulator_client.card_instance_tree_scoring_worker import CardInstanceTreeScoringWorker
from gf_simulator_client.framing import MAX_FRAME_BYTES, read_frame_admitted
from gf_simulator_client.structured_resources import ReservationLedger
from tests.card_instance_action_support import inputs
from tests.card_instance_tree_fixture_support import require_current_external_pair
from tests.test_corrected_card_instance import records

MAIN = Path('/home/percy/develop/GodFieldBots')
PAIR = MAIN/'runs/card-instance-v2-91d5926/tree-component-acceptance/fixture-default-1/fit-default-1'
ACTIONS = ActionConfigV3(action_contract=CONTRACT, max_candidates=64)


def worker(*, scope=COMPONENT, actor=1, episode='corrected-adoption-fixed', ledger=None, deadline=None, actions=None):
    require_current_external_pair(PAIR)
    return CardInstanceTreeScoringWorker(PAIR, ConfiguredTreeModelV3(),
        actions or (ACTIONS if scope == COMPONENT else ActionConfigV3(action_contract=CONTRACT)),
        17, actor, episode, input_scope=scope, ledger=ledger or ReservationLedger(),
        deadline=time.monotonic()+120 if deadline is None else deadline)


def fact(*, game=False):
    frame, bank, delta = records(inputs([('bronze-club', None), ('fireball', 'learned')]))
    actions = [{'type': 'pray'}, {'type': 'play_sequence', 'ordered_card_refs': list(frame.references), 'target': 2}]
    table = CandidateTable.from_actions(frame, actions, action_contract=CONTRACT,
                                        purpose='qualified_game' if game else 'component', synthetic_capacity=False)
    return frame, bank, delta, table


def test_clean_component_process_scores_equal_actual_paired_cpu_predictor(monkeypatch):
    assert PAIR.is_dir()
    require_current_external_pair(PAIR)
    monkeypatch.setenv('GAME_SEED', 'management-secret')
    monkeypatch.setenv('GF_SERVER_URL', 'management-secret')
    frame, bank, delta, table = fact()
    pair = inspect_training_v3(PAIR)
    tree = load_artifact_v3(PAIR/'artifact', expected_digest=pair.completion['outputs']['inference_sha256'],
        model=ConfiguredTreeModelV3(), action_config_hash=pair.inference['action_config_hash'], action_contract=CONTRACT,
        purpose='component', synthetic_capacity=False, source_kind='card-instance-tree-fixture-originals',
        source_origin=origin(), training_directory=PAIR, training_binding=pair.binding)
    expected = ComponentStructuredTreePolicyV3(tree, pair.completion['outputs']['inference_sha256']).scores(frame, bank.view(), table)
    async def run():
        ledger = ReservationLedger()
        scorer = worker(ledger=ledger)
        try:
            assert not hasattr(scorer, 'select') and not hasattr(scorer, 'select_frame')
            initial = canonical(scorer._transport.initial)
            assert 'management-secret' not in initial and str(PAIR) not in initial and 'labels' not in scorer._transport.initial
            environment = Path(f'/proc/{scorer._transport.process.pid}/environ').read_bytes()
            assert b'management-secret' not in environment
            await scorer.initialize()
            result = await scorer.score_original(frame, delta, table)
            assert result.table_hash == table.table_hash and result.frame_hash == frame.frame_hash
            np.testing.assert_array_equal(result.scores, expected)
            assert result.generated is None
            assert set(json.loads(result.timings)) == {'tokenization_ms', 'prediction_ms', 'worker_total_ms'}
            assert scorer._count == 1 and scorer._view.boundary == frame.boundary and not scorer._transport.lease.decisions
            assert 'gf_simulator_client.card_instance_tree_scoring_bootstrap' in Path(f'/proc/{scorer._transport.process.pid}/cmdline').read_bytes().decode().split('\0')
        finally:
            await scorer.close()
        assert scorer._transport.process.poll() == 0 or scorer._transport.process.poll() is not None
        assert not ledger.workers and not scorer._transport.io_thread.is_alive()
    asyncio.run(run())


def test_header_decoded_admission_precedes_body_read():
    assert MAX_FRAME_BYTES == 64*1024**2
    stream = io.BytesIO((MAX_FRAME_BYTES).to_bytes(8, 'big'))
    with pytest.raises(ContractError, match='before body read'):
        read_frame_admitted(stream, max_owned_bytes=2*1024**3)
    assert stream.tell() == 8


def test_resource_argument_rejects_before_reading_header():
    for argument in ('0', '-1', 'x', '01', str(2**63)):
        result = subprocess.run([sys.executable, '-E', '-m', 'gf_simulator_client.card_instance_tree_scoring_bootstrap', argument],
            input=b'', capture_output=True, timeout=10, check=False)
        assert result.returncode != 0
        assert b'managed limit' in result.stderr


def test_diagnostic_generation_keeps_component_artifact_and_original_game_table():
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import digest
    from gf_common.structured_wire import parse_game_generated_v3
    frame, _, delta, _ = fact(game=True)
    async def run():
        ledger = ReservationLedger()
        scorer = worker(scope=DIAGNOSTIC, ledger=ledger)
        before = scorer._rng
        try:
            result = await scorer.generate_and_score(frame, delta)
            generated = parse_game_generated_v3(json.loads(result.generated), action_contract=CONTRACT, rules=card_instance_rules())
            assert generated.table.purpose == 'qualified_game'
            assert scorer._binding['source_origin']['purpose'] == 'component'
            assert scorer._binding['caller_action_config_hash'] != scorer._binding['artifact_action_config_hash']
            assert generated.report.rng_before == before and scorer._rng == generated.report.rng_after
            assert generated.report.config_hash == digest(asdict(ActionConfigV3(action_contract=CONTRACT)))
            assert len(result.scores) == len(generated.table.kinds) and np.isfinite(result.scores).all()
            assert generated.table.table_hash == result.table_hash
            assert scorer._caps.max_decisions is None
            assert scorer._actions.max_candidates == 32768 and scorer._actions.max_expansions == 262144
            assert scorer._actions.count_states == 16384 and scorer._actions.count_milliseconds == 50
            assert scorer._actions.search_milliseconds == 1000 and scorer._model.inference.candidate_chunk == 256
            assert not hasattr(result, 'selection') and not hasattr(result, 'action')
        finally:
            await scorer.close()
        assert not ledger.workers
    asyncio.run(run())


@pytest.mark.parametrize('mutation', ['window', 'scope', 'score_count', 'nan', 'table_hash', 'frame_hash', 'request_id', 'selection'])
def test_crossed_or_malformed_reply_is_sticky_without_parent_commit(monkeypatch, mutation):
    frame, _, delta, table = fact()
    async def run():
        ledger = ReservationLedger(); scorer = worker(ledger=ledger)
        original = scorer._transport._roundtrip
        async def crossed(payload):
            reply = await original(payload)
            if payload['kind'] == 'score-original':
                if mutation == 'window':
                    reply['binding']['model']['encoder']['aggregate_window'] = 1
                elif mutation == 'scope':
                    reply['binding']['input_scope'] = DIAGNOSTIC
                elif mutation == 'score_count':
                    reply['scores'].pop()
                elif mutation == 'nan':
                    reply['scores'][0] = float('nan')
                elif mutation == 'table_hash':
                    reply['table_hash'] = '0'*64
                elif mutation == 'frame_hash':
                    reply['frame_hash'] = '0'*64
                elif mutation == 'request_id':
                    reply['request_id'] += 1
                else:
                    reply['selection'] = {'index': 0}
            return reply
        monkeypatch.setattr(scorer._transport, '_roundtrip', crossed)
        before = scorer._rng
        try:
            with pytest.raises(ContractError):
                await scorer.score_original(frame, delta, table)
            assert scorer._failed and scorer._view is None and scorer._count == 0 and scorer._rng == before
            assert scorer._transport.process.poll() is not None and not ledger.workers
            with pytest.raises(ContractError, match='cleanup-only'):
                await scorer.initialize()
        finally:
            await scorer.close()
    asyncio.run(run())


@pytest.mark.parametrize('mutation', ['rng', 'config', 'cost', 'attempts', 'count_states'])
def test_generated_reply_checks_all_rng_and_search_bounds_before_commit(monkeypatch, mutation):
    frame, _, delta, _ = fact(game=True)
    async def run():
        ledger = ReservationLedger(); scorer = worker(scope=DIAGNOSTIC, ledger=ledger)
        original = scorer._transport._roundtrip
        async def crossed(payload):
            reply = await original(payload)
            if payload['kind'] == 'generate-and-score':
                report = reply['generated']['report']
                if mutation == 'rng':
                    report['rng_before'][1][0] ^= 1
                elif mutation == 'config':
                    report['config_hash'] = '0'*64
                elif mutation == 'cost':
                    report['numeric_costs_hash'] = '0'*64
                elif mutation == 'attempts':
                    report['attempts'] = 262145
                else:
                    report['precount']['count_states'] = 16385
            return reply
        monkeypatch.setattr(scorer._transport, '_roundtrip', crossed)
        before = scorer._rng
        try:
            with pytest.raises(ContractError):
                await scorer.generate_and_score(frame, delta)
            assert scorer._failed and scorer._rng == before and scorer._view is None and scorer._count == 0
        finally:
            await scorer.close()
        assert not ledger.workers
    asyncio.run(run())


def test_overlap_rejected_before_pipe_state_or_lease_and_does_not_poison(monkeypatch):
    frame, _, delta, table = fact()
    async def run():
        scorer = worker(); entered, release = asyncio.Event(), asyncio.Event()
        original = scorer._transport._roundtrip; calls = []
        async def held(payload):
            calls.append(payload['kind'])
            reply = await original(payload)
            if payload['kind'] == 'tree-scoring-init':
                entered.set(); await release.wait()
            return reply
        monkeypatch.setattr(scorer._transport, '_roundtrip', held)
        active = asyncio.create_task(scorer.score_original(frame, delta, table))
        await entered.wait()
        try:
            with pytest.raises(ContractError, match='in-flight'):
                await scorer.initialize()
            assert calls == ['tree-scoring-init'] and not scorer._failed and not scorer._transport.lease.decisions
            assert scorer._view is None and scorer._count == 0
            release.set()
            result = await active
            assert result.table_hash == table.table_hash and calls == ['tree-scoring-init', 'score-original']
        finally:
            release.set(); await scorer.close()
    asyncio.run(run())


def test_close_interrupts_pending_initialization_and_no_late_result_escapes(monkeypatch):
    frame, _, delta, table = fact()
    async def run():
        ledger = ReservationLedger(); scorer = worker(ledger=ledger)
        entered = asyncio.Event(); never = asyncio.Event()
        original = scorer._transport._roundtrip
        async def held(payload):
            reply = await original(payload)
            entered.set(); await never.wait()
            return reply
        monkeypatch.setattr(scorer._transport, '_roundtrip', held)
        active = asyncio.create_task(scorer.score_original(frame, delta, table))
        await entered.wait()
        await scorer.close()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert scorer.closed and not scorer._initialized and scorer._count == 0 and scorer._view is None
        assert scorer._transport.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_wrong_participant_or_replayed_revision_rejects_and_reaps():
    frame, _, delta, table = fact()
    async def run():
        scorer = worker()
        try:
            await scorer.score_original(frame, delta, table)
            with pytest.raises(ContractError, match='revision/interval'):
                await scorer.score_original(frame, delta, table)
            assert scorer._count == 1
        finally:
            await scorer.close()
        scorer = worker(actor=2)
        try:
            with pytest.raises(ContractError, match='participant'):
                await scorer.score_original(frame, delta, table)
            assert scorer._count == 0
        finally:
            await scorer.close()
    asyncio.run(run())


def test_start_failure_retains_pair_cleanup_owner_and_no_reservation(monkeypatch):
    import gf_simulator_client.card_instance_tree_scoring_worker as module
    ledger = ReservationLedger(); initiating = OSError('spawn unavailable')
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: (_ for _ in ()).throw(initiating))
    with pytest.raises(OSError) as caught:
        worker(ledger=ledger)
    assert caught.value is initiating and not ledger.workers
    cleanup = caught.value._tree_scoring_startup_owner
    asyncio.run(cleanup.close())
    assert cleanup.closed


def test_expired_original_session_cannot_publish_and_close_never_extends_cap():
    frame, _, delta, table = fact()
    async def run():
        deadline = time.monotonic()+1.5
        ledger = ReservationLedger(); scorer = worker(ledger=ledger, deadline=deadline)
        await scorer.initialize()
        assert scorer._pair.scan.deadline <= deadline
        await asyncio.sleep(max(0., deadline-time.monotonic())+.01)
        with pytest.raises(TimeoutError):
            await scorer.score_original(frame, delta, table)
        with pytest.raises(TimeoutError):
            await scorer.close()
        assert scorer._transport.process.poll() is not None and not ledger.workers and scorer._count == 0
        assert scorer._deadline == deadline
    asyncio.run(run())


def test_window_scope_and_caller_identity_fail_before_spawn(monkeypatch):
    require_current_external_pair(PAIR)
    from dataclasses import replace

    import gf_simulator_client.card_instance_tree_scoring_worker as module
    spawned = []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: spawned.append(True) or pytest.fail('preflight must precede spawn'))
    model = ConfiguredTreeModelV3()
    for changed, actions, scope in (
        (replace(model, encoder=replace(model.encoder, aggregate_window=1)), ACTIONS, COMPONENT),
        (model, ActionConfigV3(action_contract=CONTRACT), COMPONENT),
        (model, ACTIONS, DIAGNOSTIC),
        (model, ACTIONS, 'game-policy'),
    ):
        with pytest.raises(ContractError):
            CardInstanceTreeScoringWorker(PAIR, changed, actions, 17, 1, 'x', input_scope=scope,
                ledger=ReservationLedger(), deadline=time.monotonic()+60)
    assert not spawned


def _oversized_encoded_pair(tmp_path):
    require_current_external_pair(PAIR)
    import hashlib
    import shutil

    target = tmp_path/'encoded-pair'; shutil.copytree(PAIR, target)
    model = target/'artifact/model.txt'
    text = model.read_text().replace('\n\nend of parameters\n', '\n[packet_stress: '+('\x00'*(12*1024**2))+']\n\nend of parameters\n')
    model.write_text(text)
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = json.loads((target/'artifact/inference.json').read_text())
    manifest['model_file']['bytes'] = manifest['tree_descriptor']['model_bytes'] = model.stat().st_size
    manifest['model_file']['sha256'] = manifest['tree_descriptor']['model_sha256'] = model_sha
    manifest['creation_execution']['tree_descriptor'] = manifest['tree_descriptor']
    def write(path, value):
        path.write_text(canonical(value)+'\n')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    inference_sha = write(target/'artifact/inference.json', manifest)
    child = json.loads((target/'child-result.json').read_text())
    child['inference_sha256'], child['model_sha256'] = inference_sha, model_sha
    child_sha = write(target/'child-result.json', child)
    complete = json.loads((target/'training.json').read_text())
    complete['outputs']['inference_sha256'], complete['outputs']['model_sha256'] = inference_sha, model_sha
    complete['outputs']['files'].update({'artifact/inference.json': inference_sha, 'artifact/model.txt': model_sha,
                                        'child-result.json': child_sha})
    write(target/'training.json', complete)
    # Negative structural corpus only: the oversized model is never loaded or fitted.
    inspect_training_v3(target, max_owned_bytes=8*1024**3)
    return target


def test_actual_encoded_init_over64mib_refuses_before_popen(tmp_path, monkeypatch):
    from dataclasses import replace

    import gf_simulator_client.card_instance_tree_scoring_worker as module
    from gf_models.structured_tree_config import TreeResourcesV2
    target = _oversized_encoded_pair(tmp_path)
    model = replace(ConfiguredTreeModelV3(), resources=TreeResourcesV2(max_owned_bytes=8*1024**3))
    spawned = []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: spawned.append(True) or pytest.fail('encoded64MiB must reject before Popen'))
    with pytest.raises(ContractError, match='actual encoded.*64MiB'):
        CardInstanceTreeScoringWorker(target, model, ACTIONS, 17, 1, 'x', input_scope=COMPONENT,
            ledger=ReservationLedger(), deadline=time.monotonic()+90)
    assert not spawned and MAX_FRAME_BYTES == 64*1024**2


def test_initial_cumulative_budget_rejects_before_encoding_and_spawn(monkeypatch):
    require_current_external_pair(PAIR)
    from dataclasses import replace

    import gf_simulator_client.card_instance_tree_scoring_worker as module
    from gf_models.structured_tree_config import TreeResourcesV2, resource_estimate
    base = ConfiguredTreeModelV3()
    minimal = resource_estimate(base)['total_bytes']
    model = replace(base, resources=TreeResourcesV2(max_owned_bytes=minimal+2*1024**2))
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: pytest.fail('joint admission must precede spawn'))
    with pytest.raises(ContractError, match='resource admission|reservation'):
        CardInstanceTreeScoringWorker(PAIR, model, ACTIONS, 17, 1, 'x', input_scope=COMPONENT,
            ledger=ReservationLedger(), deadline=time.monotonic()+60)


def test_bootstrap_managed_model_mismatch_rejects_before_native_load():
    from gf_simulator_client import framing
    async def packet():
        scorer = worker()
        try:
            return dict(scorer._transport.initial)
        finally:
            await scorer.close()
    initial = asyncio.run(packet())
    output = io.BytesIO(); framing.write_frame(output, initial)
    result = subprocess.run([sys.executable, '-E', '-m', 'gf_simulator_client.card_instance_tree_scoring_bootstrap', str(1024**3)],
        input=output.getvalue(), capture_output=True, timeout=10, check=False)
    reply = framing.read_frame(io.BytesIO(result.stdout))
    assert reply['worker_error'] == 'ContractError' and 'argv/model resource equality' in reply['message']


def test_timeout_reaps_owned_io_and_keeps_parent_state_uncommitted(monkeypatch):
    import threading

    frame, _, delta, table = fact()
    async def run():
        ledger = ReservationLedger(); scorer = worker(ledger=ledger)
        released = threading.Event()
        actual_read = scorer._transport._read_reply
        actual_close = scorer._transport._close_owned
        def stalled(source):
            released.wait(2)
            return actual_read(source)
        async def close_owned(timeout, started):
            released.set()
            return await actual_close(timeout, started)
        actual_budget = scorer._exchange_budget
        def short_budget():
            actual_budget(); scorer._transport.timeout = .02
        monkeypatch.setattr(scorer._transport, '_read_reply', stalled)
        monkeypatch.setattr(scorer._transport, '_close_owned', close_owned)
        monkeypatch.setattr(scorer, '_exchange_budget', short_budget)
        try:
            with pytest.raises(TimeoutError):
                await scorer.score_original(frame, delta, table)
            assert scorer._view is None and scorer._count == 0 and scorer._failed
            assert scorer._transport.process.poll() is not None and not ledger.workers
            assert not scorer._transport.io_thread.is_alive()
        finally:
            released.set(); await scorer.close()
    asyncio.run(run())


def test_cancelled_close_waits_for_same_cleanup_task_and_never_resets_budget(monkeypatch):
    async def run():
        ledger = ReservationLedger(); scorer = worker(ledger=ledger)
        await scorer.initialize()
        actual = scorer._transport._close_owned
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def held(timeout, started):
            calls.append((timeout, started)); entered.set(); await release.wait()
            return await actual(timeout, started)
        monkeypatch.setattr(scorer._transport, '_close_owned', held)
        closing = asyncio.create_task(scorer.close())
        await entered.wait(); closing.cancel(); release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await scorer.close()
        assert len(calls) == 1 and scorer._transport.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_changed_retained_pair_before_initialization_never_loads_child(tmp_path):
    require_current_external_pair(PAIR)
    import shutil
    target = tmp_path/'pair'; shutil.copytree(PAIR, target)
    async def run():
        ledger = ReservationLedger()
        scorer = CardInstanceTreeScoringWorker(target, ConfiguredTreeModelV3(), ACTIONS, 17, 1, 'x',
            input_scope=COMPONENT, ledger=ledger, deadline=time.monotonic()+60)
        model = target/'artifact/model.txt'; model.write_bytes(model.read_bytes()+b' ')
        with pytest.raises(ContractError):
            await scorer.initialize()
        with pytest.raises(ContractError):
            await scorer.close()
        assert scorer._transport.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_actor2_multiple_prefixes_duplicate_cards_fog_and_zero_decision_lifetime():
    import copy
    from dataclasses import replace

    from gf_common.structured import DecisionFrame, EventBank, event_delta
    from gf_common.structured_vocab import GUARDIANS
    value = inputs([('fireball', 'learned')]*12)
    obs, hints = copy.deepcopy(value.observation), copy.deepcopy(value.action_options)
    obs.update(viewer_player_id=2, current_player=2, active_player=2)
    obs['own']['player_id'], obs['opponent']['player_id'] = 2, 1
    obs['own']['status']['fog'] = True
    obs['opponent'] = {'player_id': 1, 'learned_miracles': ['aura', 'aura'], 'ascended': False}
    hints['viewer_player_id'], hints['action_options']['actor'] = 2, 2
    bank = EventBank(2, 'actor2-score', layout=LAYOUT)
    history = []
    facts = []
    for index in range(2):
        incoming = [{'type': 'guardian_set', 'player': 2, 'definition_id': GUARDIANS[index]}]
        start = bank.boundary.end
        delta = event_delta(2, bank.episode, start, bank.boundary.prefix_hash, incoming, layout=LAYOUT)
        bank.append(delta); history.extend(incoming)
        own_obs, own_hints = copy.deepcopy(obs), copy.deepcopy(hints)
        own_obs['state_revision'] = index
        own_hints['state_revision'] = own_hints['action_options']['state_revision'] = index
        current = replace(value, observation=own_obs, action_options=own_hints, events=tuple(history))
        frame = DecisionFrame.from_input(current, bank, start, layout=LAYOUT)
        table = CandidateTable.from_actions(frame, [{'type': 'pray'}], action_contract=CONTRACT,
                                           purpose='component', synthetic_capacity=False)
        facts.append((frame, delta, table))
    async def run():
        scorer = worker(actor=2, episode=bank.episode)
        try:
            for frame, delta, table in facts:
                result = await scorer.score_original(frame, delta, table)
                assert len(result.scores) == 1 and np.isfinite(result.scores).all()
            assert scorer._count == 2 and scorer._view.boundary == bank.boundary
        finally:
            await scorer.close()
        empty = worker(actor=2, episode='zero-endpoint')
        await empty.initialize(); await empty.close()
        assert empty._count == 0 and empty._view is None
    asyncio.run(run())


def test_ready_binding_mismatch_and_foreign_prefix_are_cleanup_only(monkeypatch):
    from gf_common.structured import event_delta
    frame, _, delta, table = fact()
    async def run():
        scorer = worker(); original = scorer._transport._roundtrip
        async def wrong(payload):
            result = await original(payload)
            result['binding']['resources']['max_owned_bytes'] += 1
            return result
        monkeypatch.setattr(scorer._transport, '_roundtrip', wrong)
        try:
            with pytest.raises(ContractError, match='binding mismatch'):
                await scorer.initialize()
            assert not scorer._initialized and scorer._count == 0
        finally:
            await scorer.close()
        scorer = worker()
        foreign = event_delta(delta.actor, delta.episode, delta.start, '0'*64, list(delta.events), layout=LAYOUT)
        try:
            with pytest.raises(ContractError, match='continuity'):
                await scorer.score_original(frame, foreign, table)
            assert scorer._view is None and scorer._count == 0 and scorer._failed
        finally:
            await scorer.close()
    asyncio.run(run())
