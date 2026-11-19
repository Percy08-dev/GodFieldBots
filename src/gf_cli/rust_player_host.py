"""Fixed internal Rust-supervised host bootstrap. User CLI activation is separate."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError
from gf_training.rust_player_supervision import HostControl


def _dispatch(control: HostControl, phase: str) -> None:
    arguments: tuple[Any, ...]
    if phase == "cold":
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
        result = handler(*arguments, loaded=loaded)
        if phase == "cold":
            result.close()
            control.release_parent()
    else:
        handler(*arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--launch-fd", type=int, required=True)
    parser.add_argument("--phase", choices=("host", "cold"), required=True)
    args = parser.parse_args()
    owner = None
    failed = False
    try:
        owner = HostControl(args.control_fd, args.launch_fd, phase=args.phase)
        _dispatch(owner, args.phase)
    except BaseException:  # noqa: BLE001 - process boundary suppresses private payloads, exits failed
        failed = True
        print(
            "Rust-player host failed; no training eligibility is granted by bootstrap",
            file=sys.stderr,
        )
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException:  # noqa: BLE001 - process boundary suppresses private payloads, exits failed
                failed = True
                print("Rust-player host control cleanup failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
