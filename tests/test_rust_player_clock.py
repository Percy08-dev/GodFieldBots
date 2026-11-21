"""Original operation clocks are DATA; these tests issue no live authority."""
from copy import deepcopy

import pytest

from gf_common.domain import ContractError
from gf_common.rust_player import deadline_bits, deadline_seconds


def anchor(t=100.):
    return {'boot_id': 'boot', 'monotonic_before_bits': deadline_bits(t - .01),
        'monotonic_after_bits': deadline_bits(t), 'unix_anchor_bits': deadline_bits(1000.)}


def inputs():
    return {'admission_seconds': 20., 'execution_seconds': 60.,
        'wall_seconds': 100., 'finalization_seconds': 10.}


def operation():
    from gf_common.rust_player_clock import clock_deadlines
    return {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': 'a' * 32,
        'run_id': 'run', 'mode': 'ordinary', 'operation_type': 'ordinary-train',
        'config_path': '/run/effective.json', 'config_sha256': 'b' * 64,
        'runtime_binding_sha256': 'c' * 64, 'parent_directory': '/run', 'expected_context': None,
        'execution_request': {'mode': 'fresh', 'pause_after_update': None},
        'deadlines': clock_deadlines(anchor(), inputs=inputs(), post_host_seconds=15.)}


def test_explicit_post_host_uses_whole_wall_with_extra_slack_and_coincident_kill():
    from gf_common.rust_player_clock import clock_deadlines
    deadlines = clock_deadlines(anchor(), inputs=inputs(), post_host_seconds=15.)
    assert {key: deadline_seconds(value) for key, value in deadlines.items()} == {
        'admission_bits': 120., 'execution_bits': 160., 'term_bits': 185., 'kill_bits': 185., 'finish_bits': 200.}
    assert inputs()['finalization_seconds'] == 10.  # Does not become a post-host estimate.


@pytest.mark.parametrize('p', [None, True, 0, -1, 40, 41, float('inf'), float('nan')])
def test_post_host_has_no_default_or_margin(p):
    from gf_common.rust_player_clock import clock_deadlines
    with pytest.raises(ContractError):
        clock_deadlines(anchor(), inputs=inputs(), post_host_seconds=p)


@pytest.mark.parametrize('t,p', [(2.**60, 15.), (100., 1e-20), (1e308, 1.)])
def test_unrepresentable_order_refuses_without_inventing_a_margin(t, p):
    from gf_common.rust_player_clock import clock_deadlines
    with pytest.raises(ContractError):
        clock_deadlines(anchor(t), inputs=inputs(), post_host_seconds=p)


def test_clock_record_strict_operation_sha_boot_session_and_resume_reuses_bits():
    from gf_common.rust_player_clock import operation_clock, validate_operation_clock
    op = operation()
    row = operation_clock(op, clock_binding=anchor(), inputs=inputs(), post_host_seconds=15.)
    saved = deepcopy(row)
    assert validate_operation_clock(row, operation=op, session_clock_binding=anchor(90.),
        session_finish=deadline_bits(210.), boot_id='boot', now=150.) is row
    assert row == saved and row['deadlines'] == op['deadlines']
    for changed in [row | {'extra': None}, row | {'post_host_seconds': 14.},
            row | {'operation_sha256': '0' * 64}, row | {'clock_binding': anchor(101.)}]:
        with pytest.raises(ContractError):
            validate_operation_clock(changed, operation=op, session_clock_binding=anchor(),
                session_finish=deadline_bits(210.), boot_id='boot', now=150.)
    for boot, now, finish in [('other', 150., 210.), ('boot', 160., 210.), ('boot', 150., 199.)]:
        with pytest.raises(ContractError):
            validate_operation_clock(row, operation=op, session_clock_binding=anchor(),
                session_finish=deadline_bits(finish), boot_id=boot, now=now)
