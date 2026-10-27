"""Fresh-interpreter bootstrap. No policy imports before control globals scrubbed."""
from __future__ import annotations

# Popen already supplies only fixed runtime environment values. Scrub again
# before importing the factory module, including when this entry is run directly.
import os
import sys

sys.argv[:] = ["gf-policy-worker"]
os.environ.clear()
os.environ.update({"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                   "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1"})

import contextlib
import importlib
from dataclasses import asdict

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, ContractError, DecisionInput, closed, integer
from gf_common.validation import validate_worker_input
from gf_simulator_client.framing import read_frame, write_frame


def main() -> None:
    output = sys.stdout.buffer
    source = sys.stdin.buffer
    try:
        initial = read_frame(source)
        closed(initial, {"factory", "model_id", "sampling_seed", "actions"})
        descriptor = closed(initial["factory"], {"module", "name"})
        integer(initial["sampling_seed"])
        # All imports and policy operations redirect Python stdout away from the
        # framed protocol. stderr is DEVNULL at launch and cannot block a pipe.
        with contextlib.redirect_stdout(sys.stderr):
            factory = getattr(importlib.import_module(descriptor["module"]), descriptor["name"])
            policy = factory(initial["model_id"])
            policy.reset(initial["sampling_seed"])
            settings = initial["actions"]
            for key in ("targets", "exchange_allocations"):
                settings[key] = tuple(settings[key])
            generator = PublicCandidateGenerator(ActionConfig(**settings))
        write_frame(output, {"ready": True})
        while True:
            payload = read_frame(source)
            closed(payload, {"observation", "events", "history_end", "typed_metadata", "catalog", "action_options"})
            value = validate_worker_input(DecisionInput(payload["observation"], tuple(payload["events"]),
                                                       payload["history_end"], tuple(payload["typed_metadata"]), payload["catalog"], payload["action_options"]),
                                          generator.config.use_action_options)
            with contextlib.redirect_stdout(sys.stderr):
                candidates = generator.generate(value)
                selection = policy.select(value, candidates)
            if type(selection.index) is not int or not 0 <= selection.index < len(candidates.actions):
                raise ContractError("invalid selected candidate index")
            write_frame(output, {"candidates": candidates.to_dict(), "decision": asdict(selection)})
    except EOFError:
        return
    except BaseException as exc:  # noqa: BLE001 - process boundary reports framed failure
        try:
            write_frame(output, {"worker_error": type(exc).__name__, "message": str(exc)})
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    main()
