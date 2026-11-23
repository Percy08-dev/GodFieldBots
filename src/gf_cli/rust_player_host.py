"""Fixed internal Rust-supervised host bootstrap. User CLI activation is separate."""

from __future__ import annotations

import argparse
import importlib
import sys
from functools import partial
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, canonical
from gf_training.rust_player_supervision import HostControl


def _clock_reference(control: Any, loaded: Any) -> dict[str, Any]:
    from gf_cli.rust_player_entry import _boot_id, _inputs
    from gf_common.rust_player import validate_deadlines
    from gf_common.rust_player_clock import operation_clock
    from gf_training.rust_player_supervision import read_control_record
    if not hasattr(loaded.config.run, 'post_host_seconds'):
        return {}  # Historical private literal operations issue no public clock.
    operation = control.operation
    path = control.root / f'clock-{operation["operation_id"]}.json'
    row, checksum = read_control_record(path, control.root)
    expected = operation_clock(operation, clock_binding=row['clock_binding'],
        inputs=_inputs(loaded.config), post_host_seconds=loaded.config.run.post_host_seconds)
    validate_deadlines(expected['deadlines'], session_finish=control.launch['session_deadlines']['finish_bits'])
    if (canonical(row) != canonical(expected) or row['operation_sha256'] != control.operation_sha256
            or row['clock_binding']['boot_id'] != control.launch['clock_binding']['boot_id']
            or row['clock_binding']['boot_id'] != _boot_id()):
        raise ContractError('actual cold operation-clock provenance differs')
    # The actual cold owner enforces FINISH; the prelaunch E check is inapplicable.
    return {'operation_clock_path': str(path), 'operation_clock_sha256': checksum}


def _cold_output(control: HostControl, result: Any, loaded: Any) -> str:
    from gf_cli.rust_player_entry import training_summary_data
    from gf_training.rust_player_completion import (
        ColdFinalizedRustInvocation,
        ColdFinalizedRustParent,
    )
    state: dict[str, Any] | None
    if type(result) is ColdFinalizedRustParent:
        result.require()
        invocation = result._invocation
        state = invocation._validated_state
    elif type(result) is ColdFinalizedRustInvocation:
        result._require_parent()
        invocation, state = result, result._validated_state
    else:
        raise ContractError('actual returned cold invocation/Parent required')
    if state is None:
        raise ContractError('actual finalized cold state required')
    return training_summary_data(state, invocation.completion, directory=invocation._parent_directory,
        completion_path=invocation.completion_path, completion_sha256=invocation.completion_sha256,
        clock_reference=_clock_reference(control, loaded))


def _dispatch(control: HostControl, phase: str) -> str | None:
    arguments: tuple[Any, ...]
    initial = control.initial_operation['operation_type']
    campaign = initial == 'campaign-continue'
    if campaign:
        _ = control.campaign_bootstrap  # mandatory authenticated original purpose; no directory guess
        if phase == 'campaign-return':
            _ = control.campaign_return
        module_name, function_name = 'gf_cli.rust_player_campaign', 'run_campaign'
        arguments = (control, control.operation)
    elif phase == 'campaign-return':
        raise ContractError('failed return requires authenticated INITIAL campaign purpose')
    elif phase == "cold":
        module_name, function_name = "gf_training.rust_player_learning", "finalize_cold"
        arguments = (control, control.cold)
    else:
        operation = control.operation
        handlers = {
            "ordinary-train": ("gf_training.rust_player_learning", "run_ordinary"),
            "hpo-train": ("gf_training.rust_player_learning", "run_hpo"),
            "campaign-continue": ("gf_cli.rust_player_campaign", "run_campaign"),
        }
        if operation["operation_type"] not in handlers:
            raise ContractError("cold finalization requires actual supervised cold handoff")
        module_name, function_name = handlers[operation["operation_type"]]
        arguments = (control, operation)
    # No generic callable/command selector and no eligible-data placeholder.
    try:
        module = importlib.import_module(module_name)
        handler = getattr(module, function_name)
    except (ImportError, AttributeError) as error:
        raise ContractError(
            "required Rust-player learner/finalizer handler is unavailable"
        ) from error
    if module_name == "gf_training.rust_player_learning":
        from gf_cli.config import load_rust_player_effective

        operation = control.operation
        loaded = load_rust_player_effective(
            Path(operation["config_path"]), expected_sha256=operation["config_sha256"]
        )
        if phase == 'host' and initial == 'hpo-train' and function_name == 'run_hpo':
            from gf_cli.hpo_progress import standalone_publisher

            result = handler(*arguments, loaded=loaded,
                _progress_factory=partial(standalone_publisher, control, loaded.config))
        else:
            result = handler(*arguments, loaded=loaded)
        if phase == "cold":
            try:
                output = _cold_output(control, result, loaded)
                result.close()
                control.release_parent()
            except BaseException as error:
                # Preserve any existing inner cleanup_owner and its actual error.
                vars(error)['rust_player_result_owner'] = result
                raise
            return output
    else:
        return handler(*arguments)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--launch-fd", type=int, required=True)
    parser.add_argument("--phase", choices=("host", "cold", "campaign-return"), required=True)
    args = parser.parse_args()
    owner = None
    failed = False
    failure: BaseException | None = None
    output: str | None = None
    try:
        owner = HostControl(args.control_fd, args.launch_fd, phase=args.phase)
        output = _dispatch(owner, args.phase)
    except BaseException as error:  # noqa: BLE001 - retain actual cleanup graph through control close
        failure = error
        failed = True
        print(
            "Rust-player host failed; no training eligibility is granted by bootstrap",
            file=sys.stderr,
        )
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException as cleanup:  # noqa: BLE001 - do not replace the original failure
                if failure is not None:
                    vars(failure)['rust_player_control_cleanup_error'] = cleanup
                    vars(failure)['rust_player_control_owner'] = owner
                else:
                    failure = cleanup
                failed = True
                print("Rust-player host control cleanup failed", file=sys.stderr)
    if not failed and failure is None and output is not None:
        try:
            print(output)
        except BaseException:  # noqa: BLE001 -- output failure cannot report command success
            failed = True
            print('Rust-player finalized result output failed', file=sys.stderr)
    return 1 if failed or failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
