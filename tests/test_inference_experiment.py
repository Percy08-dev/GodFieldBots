"""Experimental mechanics; no old-output or sampled-action equality gate."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_models.inference_experiment import ExperimentOptions, TensorPacker, install
from gf_models.structured_network import make_network
from tests.test_structured_network import inputs, small_model


def test_options_are_closed():
    for kwargs in ({'packing': 'unknown'}, {'chunk': 17}, {'alignment': 3}, {'precision': 'fp16'}):
        with pytest.raises(ContractError):
            ExperimentOptions(**kwargs)


@pytest.mark.parametrize('reuse', [False, True])
def test_pack_preserves_fields_types_layout_and_overwrites(reuse):
    packer = TensorPacker('cpu', reuse=reuse, max_bytes=8192, max_tensor_bytes=4096)
    values = {'numbers': ((1., -2.), (3., 4.)), 'refs': ((3, 1), (0, 18)), 'mask': (True, False)}
    dtypes = {'numbers': np.float32, 'refs': np.int64, 'mask': np.bool_}
    for count in (2, 1, 2):
        current = {key: value[:count] for key, value in values.items()}
        packed = packer.pack(current, dtypes)
        for key, value in current.items():
            assert packed[key].is_contiguous()
            assert packed[key].numpy().dtype == dtypes[key]
            assert packed[key].tolist() == np.asarray(value, dtype=dtypes[key]).tolist()
    packer.reset()
    assert packer.private_empty()


def test_pack_refuses_before_allocation():
    packer = TensorPacker('cpu', reuse=True, max_bytes=16, max_tensor_bytes=16)
    with pytest.raises(ContractError, match='storage'):
        packer.pack({'refs': tuple(range(10))}, {'refs': np.int64})
    assert packer.private_empty()


@pytest.mark.parametrize('packing', ['original', 'bulk', 'reuse'])
def test_tail_excluded_and_reset_all_capacity(make_input, packing):
    network = make_network(small_model(), 9)
    frames, candidates = inputs(make_input, count=1)
    seen = []
    network._candidate_chunk_calculate = lambda shared, projection, features, refs, weight, positions: seen.append(
        (features.shape[0], refs.shape[0])) or torch.arange(features.shape[0], dtype=torch.float32)
    experiment = install(network, ExperimentOptions(packing=packing, alignment=16))
    with torch.no_grad():
        context, cards = network.encode_history(frames)
        logs, _ = network.distribution(context, cards, candidates)
    assert seen == [(16, 16)]
    assert logs.shape == (len(candidates.kinds),)
    assert experiment.diagnostics()['valid_candidates'] == len(candidates.kinds)
    experiment.reset()
    assert experiment.private_empty()
    with torch.no_grad():
        next(network.parameters()).add_(1)
        network(frames, candidates)
    experiment.reset()
    assert experiment.private_empty()


def test_compile_wiring_counts_and_errors_surface(make_input, monkeypatch):
    compiled = []
    def fake_compile(fn, **kwargs):
        assert kwargs['fullgraph'] is True
        def wrapped(*args):
            compiled.append(fn.__name__)
            return fn(*args)
        return wrapped
    monkeypatch.setattr(torch, 'compile', fake_compile)
    network = make_network(small_model(), 9)
    experiment = install(network, ExperimentOptions(compile='default'))
    frames, candidates = inputs(make_input, count=1)
    with torch.no_grad():
        network(frames, candidates)
    assert compiled == ['_candidate_chunk_calculate']
    assert experiment.diagnostics()['compiled_regions'] == ['candidate_chunk']
    with pytest.raises(ContractError, match='no-grad'):
        network(frames, candidates)
    assert experiment.diagnostics()['fallbacks'] == 0


def test_compiler_capture_mode_cannot_enter_resident():
    network = make_network(small_model(), 7)
    with pytest.raises(ContractError, match='private'):
        install(network, ExperimentOptions(compile='reduce-overhead'), resident=True)


def test_feature_alignment_pads_operands_without_parameters(make_input):
    network = make_network(small_model(), 5)
    parameters = {key: tuple(value.shape) for key, value in network.named_parameters()}
    observed = []
    def scorer(shared, projection, features, refs, weights, positions):
        observed.append((features.shape[1], weights.shape[1]))
        return torch.zeros(features.shape[0])
    network._candidate_chunk_calculate = scorer
    install(network, ExperimentOptions(feature_alignment=32))
    frames, candidates = inputs(make_input, count=1)
    with torch.no_grad():
        network(frames, candidates)
    assert observed and observed[0][0] == observed[0][1] and observed[0][0] % 32 == 0
    assert parameters == {key: tuple(value.shape) for key, value in network.named_parameters()}


def test_real_dynamo_backend_observations(make_input, monkeypatch):
    from torch._dynamo.backends import registry
    original = registry.lookup_backend
    def lookup(name):
        return (lambda graph, example_inputs, **kwargs: graph.forward) if name == 'inductor' else original(name)
    monkeypatch.setattr(registry, 'lookup_backend', lookup)
    network = make_network(small_model(), 5)
    experiment = install(network, ExperimentOptions(compile='default'))
    frames, candidates = inputs(make_input, count=1)
    with torch.no_grad():
        network(frames, candidates)
    assert experiment.diagnostics()['backend_graphs'] > 0
    assert experiment.diagnostics()['backend_compile_seconds'] >= 0
    assert experiment.diagnostics()['failures'] == []


def test_multiple_chunks_exclude_nonfinite_padding(make_input):
    from dataclasses import replace
    network = make_network(small_model(), 9)
    frames, candidates = inputs(make_input, count=1)
    count = 257
    many = replace(candidates, kinds=(candidates.kinds[0],)*count,
        targets=(candidates.targets[0],)*count, numerics=(candidates.numerics[0],)*count,
        present=(candidates.present[0],)*count, offsets=tuple(range(count+1)), slots=(0,)*count)
    seen = []
    def scorer(shared, projection, features, refs, weights, positions):
        seen.append(features.shape[0])
        result = torch.zeros(features.shape[0])
        if features.shape[0] == 16:
            result[1:] = torch.inf
        return result
    network._candidate_chunk_calculate = scorer
    install(network, ExperimentOptions(alignment=16))
    with torch.no_grad():
        output = network(frames, many)
    assert seen == [256, 16]
    assert output[0].shape == (257,) and torch.isfinite(output[0]).all()


def test_resident_reset_scrubs_every_cached_identity_full_capacity():
    from types import SimpleNamespace

    from gf_models.resident_cache import ResidentPolicyCache
    cache = ResidentPolicyCache()
    retained = []
    for number in range(6):
        network = make_network(small_model(), number)
        experiment = install(network, ExperimentOptions(packing='reuse'))
        experiment.packer.pack({'value': tuple(range(20))}, {'value': np.int64})
        retained.extend(cpu for cpu, device in experiment.packer.buffers.values())
        policy = SimpleNamespace(network=network, reset=lambda seed, e=experiment: e.reset())
        cache.acquire(str(number)*64, weight_bytes=1, load=lambda p=policy: p, validate=lambda: None)
        cache.active = None
    cache.reset_all()
    assert cache.private_state_empty()
    assert all(not tensor.count_nonzero() for tensor in retained)


def test_compiler_failure_never_becomes_eager_success(make_input, monkeypatch):
    def compile_failure(fn, **kwargs):
        def fail(*args):
            raise RuntimeError('compiler sentinel')
        return fail
    monkeypatch.setattr(torch, 'compile', compile_failure)
    network = make_network(small_model(), 4)
    experiment = install(network, ExperimentOptions(compile='default'))
    frames, candidates = inputs(make_input, count=1)
    with torch.no_grad(), pytest.raises(RuntimeError, match='compiler sentinel'):
        network(frames, candidates)
    assert experiment.diagnostics()['failures']
    assert experiment.diagnostics()['fallbacks'] == 0


def test_all_cache_envelope_charged_in_every_phase(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    from actual import charged_admission

    from gf_models.inference_experiment import WORKER_STORAGE_BYTES
    base = {'workers': 4, 'ram_phases': {name: 100 for name in ('collection','update','save','sync')},
            'worker_bytes': 80, 'peak_cuda_bytes': 120, 'managed_ram_bytes': 2**40, 'managed_cuda_bytes': 2**40}
    result = charged_admission(lambda: base)()
    extra = 4*WORKER_STORAGE_BYTES
    assert result['worker_bytes'] == 80+extra
    assert result['peak_cuda_bytes'] == 120+extra
    assert result['ram_phases'] == {name: 100+extra for name in base['ram_phases']}


@pytest.mark.parametrize('mode', ['default', 'reduce-overhead'])
def test_compile_adapter_uses_installed_compile_fx_signature(make_input, monkeypatch, mode):
    import inspect

    import torch._inductor.compile_fx as compiler
    from torch._inductor import async_compile
    signature = inspect.signature(compiler.compile_fx)
    seen = []
    def checked(*args, **kwargs):
        signature.bind(*args, **kwargs)
        seen.append(kwargs['config_patches'])
        return args[0].forward
    monkeypatch.setattr(compiler, 'compile_fx', checked)
    monkeypatch.setattr(async_compile, 'maybe_warm_pool', lambda: None)
    network = make_network(small_model(), 5)
    experiment = install(network, ExperimentOptions(compile=mode))
    frames, candidates = inputs(make_input, count=1)
    with torch.no_grad():
        network(frames, candidates)
    assert seen == [{'triton.cudagraphs': mode == 'reduce-overhead'}]
    assert experiment.diagnostics()['backend_graphs'] == 1


def test_original_frame_submission_order_and_redundant_raw_operands(make_input, monkeypatch):
    network = make_network(small_model(hand='direct'), 5)
    frames, _candidates = inputs(make_input, count=1)
    frame = frames[0]
    observed = []
    original = network._tensor
    def tensor(value, **kwargs):
        observed.append(('tensor', np.asarray(value).tolist()))
        return original(value, **kwargs)
    monkeypatch.setattr(network, '_tensor', tensor)
    for name in ('state_encoder', 'card_encoder', 'event_encoder', 'frame_projection'):
        getattr(network, name).register_forward_pre_hook(lambda module, args, n=name: observed.append(('module', n)))
    with torch.no_grad():
        network.encode_frame(frame)
    tensor_values = [value for kind, value in observed if kind == 'tensor']
    expected = [frame.state_numerics, frame.state_present, frame.state_categories,
        frame.state_category_present, frame.opponent_learned, [v != 0 for v in frame.opponent_learned],
        frame.card_categories, frame.card_numerics, frame.card_present,
        tuple(row[2:] for row in frame.card_categories), frame.card_mask,
        tuple(row[network.layout.card_numeric_fields.index(network.layout.order_field)] for row in frame.card_numerics),
        frame.card_categories, frame.interval_numerics, frame.interval_present, frame.interval_categories, frame.history]
    assert tensor_values == [np.asarray(value).tolist() for value in expected]
    assert observed[6] == ('module', 'state_encoder')


def test_original_candidate_construction_and_positions_lifetime(make_input, monkeypatch):
    from dataclasses import replace
    network = make_network(small_model(), 5)
    frames, candidates = inputs(make_input, count=1)
    many = replace(candidates, kinds=(2,)*257, targets=(2,)*257,
        numerics=(candidates.numerics[0],)*257, present=(candidates.present[0],)*257,
        offsets=tuple(range(258)), slots=(0,)*257)
    order = []
    tensor, onehot = network._tensor, torch.nn.functional.one_hot
    def capture_tensor(values, **kwargs):
        order.append('tensor')
        return tensor(values, **kwargs)
    def capture_onehot(*args, **kwargs):
        order.append('onehot')
        return onehot(*args, **kwargs)
    monkeypatch.setattr(network, '_tensor', capture_tensor)
    monkeypatch.setattr(torch.nn.functional, 'one_hot', capture_onehot)
    with torch.no_grad():
        network._candidate_features(candidates)
    assert order == ['tensor', 'onehot', 'tensor', 'onehot', 'tensor', 'tensor']
    with torch.no_grad():
        context, cards = network.encode_history(frames)
    arange = torch.arange
    positions = []
    def capture_arange(*args, **kwargs):
        positions.append(args)
        return arange(*args, **kwargs)
    monkeypatch.setattr(torch, 'arange', capture_arange)
    with torch.no_grad():
        network.factorized_logits(context, cards, many)
    assert positions == [(network.config.max_positions,)]


def test_amortization_includes_later_graphs_and_setup(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    from aggregate import amortization
    base = {'warm_host_seconds_per_decision': 2., 'installation_host_seconds': 0.,
            'loops': [{'host_wall_seconds': 20., 'valid_decisions': 10}]*5}
    candidate = {'warm_host_seconds_per_decision': 1., 'installation_host_seconds': 7.,
        'initial_population_loops': 3, 'loops': [
            {'host_wall_seconds': wall, 'valid_decisions': 10} for wall in (60., 10., 40., 10., 10.)]}
    result = amortization(base, candidate)
    assert result['initial_population_extra_seconds'] == 57.
    assert result['additional_steady_decisions'] == 57.
    assert result['finite_decision_count'] == 87.
    candidate['warm_host_seconds_per_decision'] = None
    assert amortization(base, candidate)['finite_decision_count'] is None


def test_actual_matched_conditions_reject_different_models_and_runtime(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    from copy import deepcopy

    from aggregate import require_matched_actual
    base = {'matched_conditions': {'model': 'mlp', 'config': {'epochs': 1}, 'runtime': {'torch': 'pinned'}}}
    require_matched_actual(base, deepcopy(base))
    for field, value in (('model','gru'), ('config',{'epochs': 2}), ('runtime',{'torch':'other'})):
        other = deepcopy(base)
        other['matched_conditions'][field] = value
        with pytest.raises(ValueError, match='matched'):
            require_matched_actual(base, other)


def test_actual_telemetry_eligibility_requires_flags_and_historical_coverage(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    from aggregate import require_telemetry
    result = {'measurement': {'exclude_from_performance': False, 'dropped': 0, 'failures': []}}
    require_telemetry(SimpleNamespace(telemetry_issues=lambda *args: []), result, [], [])
    with pytest.raises(ValueError, match='coverage'):
        require_telemetry(SimpleNamespace(telemetry_issues=lambda *args: ['missing overlay']), result, [], [])
    result['measurement']['exclude_from_performance'] = True
    with pytest.raises(ValueError, match='telemetry'):
        require_telemetry(SimpleNamespace(telemetry_issues=lambda *args: []), result, [], [])


def test_replay_summary_uses_only_population_after_final_graph(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    from aggregate import replay_summary
    request = {'options': {}, 'cache_state': 'empty', 'input_manifest_sha256': 'source', 'code': {}}
    result = {'profiled': False, 'first_call_host_seconds': 1., 'installation_host_seconds': 2., 'device': 'cpu',
        'loops': [{'compiler_backend_graph_delta': count, 'host_wall_seconds': 10., 'valid_decisions': 10,
                   'valid_candidates': 100} for count in (2,0,1,0,0)],
        'compiler': {'backend_graphs': 3, 'backend_compile_seconds': 1., 'failures': [], 'fallbacks': 0}}
    for name, data in (('request',request), ('result',result), ('receipt',{'returncode':0,'failure':None})):
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
    summary = replay_summary(tmp_path)
    assert summary['initial_population_loops'] == 3 and summary['steady_loop_count'] == 2
    result['loops'][-1]['compiler_backend_graph_delta'] = 1
    (tmp_path/'result.json').write_text(json.dumps(result))
    assert replay_summary(tmp_path)['warm_host_seconds_per_decision'] is None


def test_actual_namespace_normalization_is_exact(monkeypatch):
    from copy import deepcopy
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    from aggregate import actual_conditions, require_matched_actual

    from gf_common.domain import digest
    def observed(run_id):
        materialized = {'run': {'name': run_id}, 'model': {'kind':'mlp'},
                        'hpo_spec': {'storage': {'parent_root': '/parents'}}}
        fingerprint = digest(materialized)
        return {'request': {'run_id': run_id, 'parent':'/parents/'+run_id, 'arm':'C', 'mode':'fresh',
            'pause_after_update':None, 'identity':{'freeze':'same'}, 'code':{'network':'same'},
            'config_file_sha256':'same', 'python':'3.14', 'python_executable':'/venv/python',
            'gil_disabled':False, 'platform':'same'}, 'parent':'/parents/'+run_id,
            'preparation': {'materialized_config':materialized, 'materialized_sha256':fingerprint,
                'loaded_config_sha256':'same', 'effective_config':{'model':'mlp'}, 'config_inputs':[],
                'effective_settings_sha256':'same', 'conditions':{'source':'same'}},
            'after': {'run_id':run_id, 'effective_config':materialized, 'config_sha256':fingerprint,
                'status':'incomplete', 'stop_reason':'update_cap', 'updates':2, 'seed':17,
                'device':'cuda:0', 'hardware':{'name':'same'}}, 'status':'incomplete',
            'stop_reason':'update_cap', 'updates':2, 'torch_runtime':{'torch':'same'},
            'gpu_termination':{'boot_id':'same'}}
    left, right = observed('first'), observed('second')
    require_matched_actual({'matched_conditions':actual_conditions(left)}, {'matched_conditions':actual_conditions(right)})
    other = deepcopy(right)
    other['preparation']['materialized_config']['model']['kind'] = 'gru'
    other['preparation']['materialized_sha256'] = digest(other['preparation']['materialized_config'])
    other['after']['config_sha256'] = other['preparation']['materialized_sha256']
    with pytest.raises(ValueError, match='matched'):
        require_matched_actual({'matched_conditions':actual_conditions(left)}, {'matched_conditions':actual_conditions(other)})
    right['preparation']['materialized_config']['run']['name'] = 'foreign'
    with pytest.raises(ValueError, match='materialization'):
        actual_conditions(right)


def test_actual_analyzer_must_be_bound_to_root_freeze(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    import aggregate
    result = {'exit_code':0, 'probe_expectation_met':True, 'gpu_termination_context_verified':True,
              'request': {'identity': {'files': {}}}}
    monkeypatch.setattr(aggregate, 'read_json', lambda path: result)
    with pytest.raises(ValueError, match='analyzer.*freeze'):
        aggregate.actual_summary(tmp_path)


@pytest.mark.parametrize('kind', ['actual', 'replay'])
def test_median_population_rejects_mixed_pairs_preserves_repeated_pairs(monkeypatch, capsys, kind):
    import json
    import sys
    from copy import deepcopy
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks/inference'))
    import aggregate
    base = {'options': {'packing':'original'}, 'matched_conditions': {'model':'mlp','freeze':'same'},
        'helper_wall_seconds':20., 'collected_decisions':100, 'warm_host_seconds_per_decision':2.,
        'input_sha256':'same', 'device':'cpu', 'code':{'source':'same'}, 'cache_state':'empty',
        'valid_decisions_per_loop':[10]*3, 'valid_candidates_per_loop':[100]*3}
    candidate = {**deepcopy(base), 'options':{'packing':'bulk'}, 'helper_wall_seconds':10.,
                 'warm_host_seconds_per_decision':1.}
    rows = {'a':deepcopy(base), 'b':deepcopy(candidate), 'c':deepcopy(base), 'd':deepcopy(candidate)}
    monkeypatch.setattr(aggregate, kind+'_summary', lambda path: rows[path.name])
    monkeypatch.setattr(aggregate, 'amortization', lambda *args: {'finite_decision_count':None})
    monkeypatch.setattr(sys, 'argv', ['aggregate.py','--kind',kind,'--pair','a','b','--pair','c','d'])
    assert aggregate.main() == 0
    assert json.loads(capsys.readouterr().out)['median_delta'] == (-10. if kind == 'actual' else -1.)
    for field, value in ((('matched_conditions' if kind == 'actual' else 'input_sha256'), 'different'),
                         ('options', {'packing':'reuse'})):
        rows['c'], rows['d'] = deepcopy(base), deepcopy(candidate)
        if field == 'options':
            rows['d'][field] = value
        else:
            rows['c'][field] = rows['d'][field] = value
        with pytest.raises(ValueError, match='median population'):
            aggregate.main()


@pytest.mark.parametrize('mode', ['none', 'default', 'reduce-overhead'])
def test_captured_scorer_outputs_own_storage_across_next_invocation(monkeypatch, mode):
    storage = torch.empty(2)
    events = []
    def scorer(*args):
        events.append('scorer')
        storage.fill_(1 if len(events) <= 2 else 0)
        return storage
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: scorer)
    monkeypatch.setattr(torch.compiler, 'cudagraph_mark_step_begin', lambda: events.append('step'))
    network = make_network(small_model(), 5)
    network._candidate_chunk_calculate = scorer
    experiment = install(network, ExperimentOptions(compile=mode))
    args = (torch.zeros(16), torch.zeros(18,19,16), torch.zeros(2,9),
            torch.zeros(2,18,dtype=torch.long), torch.zeros(16,9), torch.arange(18)[None])
    with torch.no_grad():
        first = experiment.chunk(*args)
        first_pointer = first.data_ptr()
        experiment.chunk(*args)
        storage.zero_()
    if mode == 'reduce-overhead':
        assert first_pointer != storage.data_ptr()
        assert first.count_nonzero().item() == 2
        assert events == ['step','scorer','step','scorer']
    else:
        assert first_pointer == storage.data_ptr()
        assert events == ['scorer','scorer']


def test_inference_wrapper_accepts_only_absent_private_operands(make_input):
    network = make_network(small_model(), 5)
    experiment = install(network, ExperimentOptions(packing='reuse'))
    frames, candidates = inputs(make_input, count=1)
    with torch.no_grad():
        context, cards = network.encode_history(frames)
        network.distribution(context, cards, candidates, _operands=None)
        with pytest.raises(ContractError, match='learner/owned operands'):
            experiment.factorized(context, cards, candidates, _operands={})
