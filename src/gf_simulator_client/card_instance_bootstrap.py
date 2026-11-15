"""Clean fixed protocol3 bootstrap: no experiment/management code enters policies."""

from __future__ import annotations

import os
import re
import sys

_CONTROL = tuple(sys.argv[1:])
_CONTROL_ERROR = None
_MANAGED = None
if _CONTROL and (len(_CONTROL) != 1 or re.fullmatch('[1-9][0-9]{0,18}', _CONTROL[0]) is None):
    _CONTROL_ERROR = 'Tree3 accepts one sanitized decimal managed limit'
elif _CONTROL:
    _MANAGED = int(_CONTROL[0])
    if not 1 <= _MANAGED <= 2**63-1:
        _CONTROL_ERROR = 'Tree3 managed limit outside1..2^63-1'
del _CONTROL
sys.argv[:] = ["gf-card-instance-policy-worker"]
os.environ.clear()
os.environ.update(
    {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
)

from gf_common.domain import ContractError
from gf_simulator_client.card_instance_episode import CardInstanceEpisode
from gf_simulator_client.framing import (
    frame_decode_bytes,
    read_frame,
    read_frame_admitted,
    write_frame,
)


def main() -> None:
    output, source = sys.stdout.buffer, sys.stdin.buffer
    try:
        if _CONTROL_ERROR is not None:
            raise ContractError(_CONTROL_ERROR)
        if _MANAGED is not None and _MANAGED < frame_decode_bytes(1):
            raise ContractError("Tree3 initial header/body admission exceeded before read")
        initial = (read_frame(source) if _MANAGED is None else
            read_frame_admitted(source, max_owned_bytes=_MANAGED))
        episode = CardInstanceEpisode(initial, managed=_MANAGED)
        write_frame(output, episode.ready)
        while True:
            retained = episode.policy.retained_bytes(
                (None if episode.bank is None else vars(episode.bank), episode.previous)) if episode.tree else 0
            message = (read_frame_admitted(source, max_owned_bytes=_MANAGED, retained_bytes=retained)
                if _MANAGED is not None else read_frame(source))
            write_frame(output, episode.select(message))
    except EOFError:
        return
    except BaseException as error:  # noqa: BLE001 - framed owned process boundary.
        try:
            write_frame(output, {"worker_error": type(error).__name__, "message": str(error)})
        except OSError, ValueError:
            pass


if __name__ == "__main__":
    main()
