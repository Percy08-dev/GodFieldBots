"""Fixed clean CPU scores-only process. One decimal managed limit precedes framing."""
from __future__ import annotations

import os
import re
import sys

_CONTROL = tuple(sys.argv[1:])
sys.argv[:] = ['gf-card-instance-tree-scoring']
os.environ.clear()
os.environ.update({'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
if len(_CONTROL) != 1 or re.fullmatch('[1-9][0-9]{0,18}', _CONTROL[0]) is None:
    raise SystemExit('one sanitized decimal CPU managed limit required')
_MANAGED = int(_CONTROL[0])
if not 1 <= _MANAGED <= 2**63-1:
    raise SystemExit('CPU managed limit outside1..2^63-1')
del _CONTROL


def main() -> None:
    import time

    from gf_common.diverse_actions import PublicCandidateGeneratorV3
    from gf_common.domain import ContractError, canonical, closed, integer
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_wire import (
        game_generated_dict_v3,
        parse_delta,
        parse_frame,
        parse_table,
    )
    from gf_models.card_instance_tree_scoring import (
        DIAGNOSTIC,
        _ScoringKernel,
        check_binding,
        stage_bank,
    )
    from gf_models.structured_tree_encoder import _owned
    from gf_simulator_client.framing import read_frame_admitted, write_frame

    source, output = sys.stdin.buffer, sys.stdout.buffer
    try:
        load_started = time.perf_counter()
        initial = read_frame_admitted(source, max_owned_bytes=_MANAGED)
        kernel = _ScoringKernel(initial, _MANAGED)
        del initial
        write_frame(output, {'protocol_version': 3, 'kind': 'tree-scoring-ready', 'binding': kernel.binding,
                             'model_load_ms': (time.perf_counter()-load_started)*1000})
        generator = PublicCandidateGeneratorV3(kernel.actions, rules=kernel.rules)
        while True:
            message = read_frame_admitted(source, max_owned_bytes=_MANAGED, retained_bytes=kernel.retained_bytes())
            started = time.perf_counter()
            generated = message.get('kind') == 'generate-and-score'
            closed(message, {'protocol_version', 'kind', 'binding', 'request_id', 'frame', 'delta'}
                   | ({'rng_before'} if generated else {'table'}))
            if (type(message['protocol_version']) is not int or message['protocol_version'] != 3
                    or message['kind'] not in ('score-original', 'generate-and-score')):
                raise ContractError('closed scores-only request required')
            check_binding(message['binding'], kernel.binding)
            integer(message['request_id'], kernel.count+1, kernel.count+1)
            if generated and kernel.binding['input_scope'] != DIAGNOSTIC:
                raise ContractError('only actual-own-input diagnostic scope may generate')
            if generated and canonical(message['rng_before']) != canonical(kernel.rng.getstate()):
                raise ContractError('generation original RNG-before differs')
            from gf_models.structured_tree_config import resource_estimate
            resource_estimate(kernel.tree.model, model_bytes=kernel.tree.descriptor.model_bytes,
                              extra_bytes=_owned((kernel.previous, kernel.last, message)))
            frame = parse_frame(message['frame'], layout=layout)
            delta = parse_delta(message['delta'], layout=layout)
            bank = stage_bank(kernel.previous, kernel.last, frame, delta, kernel.binding, kernel.caps)
            view = bank.view()
            conversion_ms = generation_ms = 0.
            if generated:
                generation_started = time.perf_counter()
                proposals = generator.generate_game(frame, view, kernel.rng, kernel.costs)
                generation_ms = (time.perf_counter()-generation_started)*1000
                table = proposals.table
            else:
                table = parse_table(message['table'], action_contract=kernel.actions.action_contract,
                    purpose='component' if kernel.binding['input_scope'] != DIAGNOSTIC else 'qualified_game',
                    synthetic_capacity=False, layout=layout)
            values, measured = kernel.score(frame, view, table)
            reply = {'protocol_version': 3, 'kind': 'tree-generated-scores' if generated else 'tree-scores',
                'binding': kernel.binding, 'request_id': message['request_id'], 'frame_hash': frame.frame_hash,
                'table_hash': table.table_hash, 'row_count': len(table.kinds), 'scores': values}
            if generated:
                converted = time.perf_counter()
                reply['generated'] = game_generated_dict_v3(proposals, action_contract=kernel.actions.action_contract, rules=kernel.rules)
                conversion_ms = (time.perf_counter()-converted)*1000
            reply['timings'] = {**measured, 'worker_total_ms': (time.perf_counter()-started)*1000,
                **({'generation_ms': generation_ms, 'conversion_ms': conversion_ms} if generated else {})}
            write_frame(output, reply)
            kernel.previous, kernel.last, kernel.count = view, frame, kernel.count+1
            del message, table, bank, view, frame, delta, reply, values
    except EOFError:
        return
    except BaseException as error:  # noqa: BLE001 - the fixed child becomes cleanup-only on every error.
        try:
            write_frame(output, {'worker_error': type(error).__name__, 'message': str(error)})
        except OSError, ValueError:
            pass


if __name__ == '__main__':
    main()
