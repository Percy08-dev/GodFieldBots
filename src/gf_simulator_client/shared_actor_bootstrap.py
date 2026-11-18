"""Clean CPU participant; private FD carries only its own numeric request/results."""
from __future__ import annotations

import os
import sys

_CONTROL=tuple(sys.argv[1:])
sys.argv[:]=['gf-shared-cpu-participant']
os.environ.clear()
os.environ.update({'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONHASHSEED':'0',
    'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})


from gf_common.domain import ContractError
from gf_simulator_client.card_instance_episode import CardInstanceEpisode
from gf_simulator_client.framing import read_frame, write_frame
from gf_simulator_client.shared_channel import ParticipantScoreChannel, policy_loader


def _cpu_only(episode: CardInstanceEpisode) -> None:
    torch=getattr(episode.policy,'torch',None)
    if torch is not None and torch.cuda.is_initialized():raise ContractError('CPU participant unexpectedly initialized CUDA')


def main() -> None:
    output,source=sys.stdout.buffer,sys.stdin.buffer
    try:
        if len(_CONTROL)!=1 or not _CONTROL[0].isdigit():raise ContractError('shared actor requires one inherited own endpoint')
        channel=ParticipantScoreChannel(int(_CONTROL[0]))
        initial=read_frame(source)
        episode=CardInstanceEpisode(initial,policy_loader=lambda *args:policy_loader(channel,*args))
        _cpu_only(episode)
        write_frame(output,episode.ready)
        while True:
            message=read_frame(source)
            channel.bind_request(message.pop('shared_request',None),message.pop('shared_deadline',None))
            response=episode.select(message)
            _cpu_only(episode)
            write_frame(output,response)
    except EOFError:return
    except BaseException as error:  # noqa: BLE001 - own actor frame error only.
        try:write_frame(output,{'worker_error':type(error).__name__,'message':str(error)})
        except OSError:pass


if __name__=='__main__':main()
