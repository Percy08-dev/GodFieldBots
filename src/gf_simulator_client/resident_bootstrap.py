"""Clean fixed resident transport. Outer controls never enter policy features."""
from __future__ import annotations

import os
import sys

_VIEWER = sys.argv[1:]
sys.argv[:] = ['gf-resident-participant-worker']
os.environ.clear()
os.environ.update({'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})

from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.hpo_resident import CACHE_OWNER_KEYS, RESIDENT_MODE, _sha, cache_owner, checked_ack
from gf_models.resident_cache import ResidentPolicyCache, resident_execution_profile
from gf_simulator_client.card_instance_episode import CardInstanceEpisode
from gf_simulator_client.framing import read_frame, write_frame


class ResidentEpisodeTransport:
    def __init__(self, viewer: int) -> None:
        self.viewer = viewer
        self.cache = ResidentPolicyCache()
        self.episode: CardInstanceEpisode | None = None
        self.owner: Json | None = None
        self.last_epoch = self.request_epoch = 0
        self.checkpoint: str | None = None
        self.generation: int | None = None
        self._checkpoints: dict[str,str] = {}
        self._execution_profile: Json | None = None
        self._execution_environment: Json | None = None

    def _load(self, model_id: str, options: Json | None, actions: Any, rules: Any, load: Any) -> Any:
        if model_id == 'lightgbm':
            raise ContractError('resident HPO does not admit tree policies')
        if model_id in ('mlp','gru','transformer'):
            assert options is not None
            profile=resident_execution_profile(options)
            if self._execution_profile is not None and canonical(profile)!=canonical(self._execution_profile):
                raise ContractError('resident process execution settings cannot change')
            if self._execution_environment is not None and any(os.environ.get(key)!=value for key,value in self._execution_environment.items()):
                raise ContractError('resident process execution environment changed')
            self._execution_profile=profile
        policy_sha = digest({'model_id': model_id, 'policy_options': options})
        weight_bytes = 0
        if options is None or model_id == 'heuristic_refresh':
            identity = policy_sha
        else:
            from gf_common.structured_layout import layout_for_contract
            from gf_models.hpo_config import model_config_v4
            from gf_models.structured_artifacts import inspect_game_inference_v4
            from gf_models.structured_config import InferenceBindingV2
            if options.get('version') != '4':
                raise ContractError('resident HPO requires inference5/model4')
            def inspect() -> Json:
                manifest, _ = inspect_game_inference_v4(Path(options['checkpoint']),
                    expected_digest=options['expected_manifest_digest'],
                    model=model_config_v4(options['model'], layout=layout_for_contract(actions.action_contract)),
                    binding=InferenceBindingV2(**options['inference']), action_config_hash=options['action_config_hash'],
                    action_contract=actions.action_contract, rules=rules)
                return manifest
            manifest = inspect()
            identity = digest({'policy_sha256': policy_sha, 'inference_sha256': options['expected_manifest_digest'],
                'weights_sha256': digest(manifest['weights']['files'])})
            weight_bytes = manifest['weights']['decoded_bytes']
        if self.checkpoint != (options['expected_manifest_digest'] if options is not None and model_id != 'heuristic_refresh' else policy_sha):
            raise ContractError('resident selected checkpoint differs from original request')
        # This explicit cache never edits a pre-existing network: frozen aliases remain immutable.
        assert self.checkpoint is not None
        policy = self.cache.acquire(identity, weight_bytes=weight_bytes, load=load,
            validate=(lambda: inspect()) if options is not None and model_id != 'heuristic_refresh' else lambda: None)
        if model_id in ('mlp','gru','transformer') and self._execution_environment is None:
            self._execution_environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG',
                'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TORCHINDUCTOR_DETERMINISTIC')}
        self._checkpoints[identity] = self.checkpoint
        return policy

    def _reset(self) -> None:
        if self.episode is not None:
            self.episode.reset()
            if not self.episode.private_state_empty():
                raise ContractError('resident original episode reset incomplete')
            self.episode = None
        self.cache.release()
        self.cache.reset_all()
        self.cache.synchronize()

    def exchange(self, envelope: Json) -> Json:
        closed(envelope, {'request'}, {'payload'})
        request = envelope['request']
        if type(request) is not dict or request.get('protocol') != RESIDENT_MODE or request.get('schema_version') != 1:
            raise ContractError('resident original envelope required')
        integer(request.get('schema_version'),1,1)
        if self.owner is None:
            closed(request, CACHE_OWNER_KEYS | {'schema_version', 'protocol', 'command', 'run_id'})
            if (request['command'] != 'READY' or request['viewer'] != self.viewer
                    or request['pid'] != os.getpid() or request['lease_epoch'] != 0):
                raise ContractError('resident clean bootstrap original identity differs')
            integer(request['lease_epoch'],0,0)
            cache_owner({key:request[key] for key in CACHE_OWNER_KEYS} | {'lease_epoch':1})
            self.owner = {key: request[key] for key in CACHE_OWNER_KEYS - {'lease_epoch'}} | {'run_id': request['run_id']}
            return {**request, 'kind': 'READY', 'clean_reset': True, 'private_state_empty': True,
                'pending_io_empty': True, 'cuda_synchronized': True, 'reset_version': 'resident-reset-v1'}
        if any(request.get(key) != value for key, value in self.owner.items()):
            raise ContractError('resident fixed original owner/viewer differs')
        cache_owner({key:request[key] for key in CACHE_OWNER_KEYS})
        integer(request.get('generation'),0,2**63-1)
        integer(request.get('request_epoch'),1,2**63-1)
        command = request['command']
        epoch = request['lease_epoch']
        if request['request_epoch'] != self.request_epoch+1:
            raise ContractError('resident request epoch is stale')
        self.request_epoch = request['request_epoch']
        if command == 'BEGIN':
            closed(request, CACHE_OWNER_KEYS | {'schema_version', 'protocol', 'command', 'run_id',
                'generation', 'request_epoch', 'checkpoint_sha256'})
            if self.episode is not None or epoch != self.last_epoch + 1:
                raise ContractError('resident BEGIN overlaps episode or lease epoch')
            _sha(request['checkpoint_sha256'])
            if self.generation is not None and request['generation'] not in (self.generation,self.generation+1):
                raise ContractError('resident BEGIN control generation is stale')
            self.last_epoch, self.checkpoint = epoch, request['checkpoint_sha256']
            self.generation = request['generation']
            self.episode = CardInstanceEpisode(envelope['payload'], policy_loader=self._load)
            return {'request': request, 'payload': self.episode.ready}
        if epoch != self.last_epoch and not (command == 'EVICT' and self.episode is None and epoch == self.last_epoch+1):
            raise ContractError('resident command has stale lease epoch')
        if command == 'SELECT':
            closed(request,CACHE_OWNER_KEYS | {'schema_version','protocol','command','run_id','generation','request_epoch','checkpoint_sha256'})
            if request['generation'] != self.generation or request['checkpoint_sha256'] != self.checkpoint:
                raise ContractError('resident SELECT control/checkpoint differs')
            if self.episode is None or envelope['payload'].get('frame', {}).get('actor') != self.viewer:
                raise ContractError('resident SELECT lacks own viewer active episode')
            return {'request': request, 'payload': self.episode.select(envelope['payload'])}
        if command == 'CLOSE':
            closed(request,CACHE_OWNER_KEYS | {'schema_version','protocol','command','run_id','generation','request_epoch'})
            if request['generation'] != self.generation:
                raise ContractError('resident CLOSE control generation differs')
            if self.episode is not None:
                raise ContractError('resident CLOSE overlaps original episode')
            self._reset()
            return {'request':request,'kind':'CLOSED'}
        checked_ack({**request, 'clean_reset': True, 'private_state_empty': True,
            'pending_io_empty': True, 'cuda_synchronized': True, 'reset_version': 'resident-reset-v1'})
        if command not in ('SYNC','EVICT') and request['generation'] != self.generation:
            raise ContractError('resident reset control generation differs')
        if command == 'END':
            if self.episode is None or request['checkpoint_sha256'] != self.checkpoint:
                raise ContractError('resident END lacks original active episode/checkpoint')
            self._reset()
        elif command == 'SYNC':
            if self.episode is not None:
                raise ContractError('resident SYNC overlaps active episode')
            if self.generation is None or request['generation'] != self.generation+1:
                raise ContractError('resident SYNC must advance exactly one control generation')
            self.generation = request['generation']
            self.checkpoint = request['checkpoint_sha256']
            self.episode = CardInstanceEpisode(envelope['payload'], policy_loader=self._load)
            self._reset()
        elif command == 'EVICT':
            if self.episode is not None:
                raise ContractError('resident EVICT overlaps active episode')
            if self._checkpoints.get(request['cache_identity']) != request['checkpoint_sha256']:
                raise ContractError('resident EVICT checkpoint identity differs')
            self.cache.evict(request['cache_identity'])
            del self._checkpoints[request['cache_identity']]
            self._reset()
        elif command == 'PARK':
            if self.episode is not None or request['checkpoint_sha256'] != self.checkpoint:
                raise ContractError('resident PARK lacks original reset/checkpoint')
            self._reset()
        else:
            raise ContractError('resident command unsupported')
        return {**request, 'clean_reset': True, 'private_state_empty': self.cache.private_state_empty(),
            'pending_io_empty': True, 'cuda_synchronized': True, 'reset_version': 'resident-reset-v1'}


def main() -> None:
    output, source = sys.stdout.buffer, sys.stdin.buffer
    try:
        if len(_VIEWER) != 1 or _VIEWER[0] not in ('1', '2'):
            raise ContractError('resident viewer must be sanitized1or2')
        transport = ResidentEpisodeTransport(int(_VIEWER[0]))
        while True:
            envelope = read_frame(source)
            response = transport.exchange(envelope)
            if envelope['request']['command'] != 'SELECT':
                # Separate outer diagnostic leaf; never part of a strict ACK.
                response['_diagnostic'] = transport.cache.diagnostic()
            write_frame(output, response)
            if envelope["request"]["command"] == "CLOSE":
                return
    except EOFError:
        return
    except BaseException as error:  # noqa: BLE001 - owned framing boundary
        try:
            write_frame(output, {'worker_error': type(error).__name__, 'message': str(error)})
        except OSError, ValueError:
            pass


if __name__ == '__main__':
    main()
