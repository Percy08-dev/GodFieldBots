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


class _ActorBankReservation:
    def __init__(self, budget: Any, size: int) -> None:
        self.budget, self.size = budget, size

    def close(self) -> None:
        if self.size:
            self.budget.used -= self.size
            self.size = 0

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _ActorBankBudget:
    """Single-threaded transfer metadata/copy admission in the actor process."""
    def __init__(self) -> None:
        self.used = 0

    def reserve(self, size: int, *, wait: bool = False) -> _ActorBankReservation:
        integer(size, 0, 512*1024**2)
        if self.used + size > 512*1024**2:
            raise ContractError('actor9 transfer envelope exceeds admission')
        self.used += size
        return _ActorBankReservation(self, size)


class ResidentEpisodeTransport:
    def __init__(self, viewer: int, *, owned_channel: Any = None, score_channel: Any = None) -> None:
        self.viewer = viewer
        self.owned_channel = owned_channel
        self.score_channel=score_channel
        self.profile_host=score_channel is not None
        self._owned_request: Json | None = None
        self._owned_outer_request: Json | None=None
        self._owned_usage: str | None=None
        self._owned_final_set: str | None=None
        self._owned_final_reservation: Any=None
        self._owned_budget = _ActorBankBudget()
        self.cache = ResidentPolicyCache()
        self.episode: CardInstanceEpisode | None = None
        self.owner: Json | None = None
        self.last_epoch = self.request_epoch = 0
        self.checkpoint: str | None = None
        self.generation: int | None = None
        self._checkpoints: dict[str,str] = {}
        self._execution_profile: Json | None = None
        self._execution_environment: Json | None = None

    @property
    def _protocol(self) -> str:
        from gf_common.shared_inference import CPU_PROTOCOL
        return CPU_PROTOCOL if self.score_channel is not None else RESIDENT_MODE

    def _reset_fields(self) -> Json:
        from gf_common.shared_inference import cpu_reset_fields
        return cpu_reset_fields() if self.score_channel is not None else {'clean_reset':True,'private_state_empty':True,
            'pending_io_empty':True,'cuda_synchronized':True,'reset_version':'resident-reset-v1'}

    def _check_ack(self,value: Json) -> Json:
        from gf_common.shared_inference import checked_cpu_ack
        return checked_cpu_ack(value) if self.score_channel is not None else checked_ack(value)

    def _load(self, model_id: str, options: Json | None, actions: Any, rules: Any, load: Any) -> Any:
        from gf_common.card_instance import SHARED_SIMULATOR_REVISION
        self.profile_host=self.profile_host or actions.action_contract.simulator_revision==SHARED_SIMULATOR_REVISION
        if self.score_channel is not None:
            from gf_simulator_client.shared_channel import policy_loader
            return policy_loader(self.score_channel,model_id,options,actions,rules,load)
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
        if model_id in ('mlp', 'gru', 'transformer') and self.owned_channel is not None:
            if self._owned_request is None:
                raise ContractError('actor9 neural load requires SYNC_OWNED inference FD')
            return self._load_owned(model_id, options, actions, rules)
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

    def _load_owned(self, model_id: str, options: Json | None, actions: Any, rules: Any) -> Any:
        from gf_common.ram_bank import MAX_BANK_BYTES, ReadonlyRamBankSet
        from gf_common.structured_layout import layout_for_contract
        from gf_models.hpo_config import model_config_v4
        from gf_models.hpo_owned_inference import _validate_owned_inference, load_owned_inference
        from gf_models.structured_config import InferenceBindingV2
        from gf_models.structured_policy import GameStructuredPolicyV4
        if options is None or options.get('version') != '4' or self.owned_channel is None:
            raise ContractError('actor9 requires model4 and inherited owned channel')
        wrapper = closed(self._owned_request, {'metadata', 'bank_count'}, {'evaluation_use'})
        metadata = wrapper['metadata']
        if self.checkpoint != metadata.get('inference_digest') or self.checkpoint != options['expected_manifest_digest']:
            raise ContractError('actor9 snapshot differs from selected checkpoint')
        assert self.generation is not None
        if 'evaluation_use' in wrapper:
            if self._owned_usage not in (None,'evaluation') or self._owned_outer_request is None:
                raise ContractError('evaluation final use cannot replace training scope')
            from gf_common.artifacts import json_bound
            from gf_common.hpo_actor_snapshot import checked_evaluation_use
            with self._owned_budget.reserve(32*json_bound(wrapper['evaluation_use'])+65536):
                finals=checked_evaluation_use(wrapper['evaluation_use'],metadata,self._owned_outer_request,options)
            if self._owned_final_set is not None and self._owned_final_set!=finals:
                raise ContractError('evaluation actor final-reference set changed')
            if self._owned_final_set is None:
                self._owned_final_reservation=self._owned_budget.reserve(32*len(finals)+65536)
                self._owned_final_set=finals
            self._owned_usage='evaluation'
            generation=integer(metadata.get('generation'),0,2**63-1)
        else:
            if self._owned_usage not in (None,'training'):
                raise ContractError('evaluation actor cannot adopt a training wrapper')
            self._owned_usage='training'
            generation = integer(metadata.get('generation'), 0, self.generation)
        model = model_config_v4(options['model'], layout=layout_for_contract(actions.action_contract))
        binding = InferenceBindingV2(**options['inference'])
        visibility = options['cuda_visible_devices']
        if visibility is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            deliveries = []
            total = 0
            for index in range(integer(wrapper['bank_count'], 1, 65536)):
                delivery = self.owned_channel.receive(purpose='model', generation=generation,
                    identity=metadata['snapshot_identity']+f':inference:{index}', budget=self._owned_budget,
                    max_bytes=MAX_BANK_BYTES-total)
                total += delivery.bank.size
                deliveries.append(delivery)
            banks = ReadonlyRamBankSet(tuple(delivery.bank for delivery in deliveries))
            identity = digest({'recipe':'owned-inference-cache-v1', 'policy_sha256':digest({'model_id':model_id,'policy_options':options}),
                'inference_sha256':self.checkpoint,'snapshot_identity':metadata['snapshot_identity']})
            def validate() -> None:
                _validate_owned_inference(banks, metadata, model=model, binding=binding, actions=actions, rules=rules)
            def load() -> Any:
                network = load_owned_inference(banks, metadata, model=model, binding=binding, actions=actions, rules=rules)
                return GameStructuredPolicyV4(network, model, binding, metadata['inference_digest'],
                    action_contract=actions.action_contract, rules=rules, action_config_hash=options['action_config_hash'])
            policy = self.cache.acquire(identity, weight_bytes=sum(d.nbytes for bank in banks.banks for d in bank.descriptors),
                load=load, validate=validate)
            self.cache.synchronize()
            for delivery in deliveries:
                delivery.complete(self.cache.synchronize)
            self._owned_request = None
            self._owned_outer_request = None
            assert self.checkpoint is not None
            self._checkpoints[identity] = self.checkpoint
            if self._execution_environment is None:
                self._execution_environment = {key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG',
                    'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TORCHINDUCTOR_DETERMINISTIC')}
            return policy
        except BaseException as error:
            try:
                self.owned_channel.retire_received(self.cache.synchronize)
            except BaseException as cleanup:  # noqa: BLE001 - preserve actor failure and original receiver custody
                error.add_note('actor9 receive cleanup retained: '+type(cleanup).__name__)
            raise

    def _reset(self) -> None:
        if self.score_channel is not None:self.score_channel.bind_request(None,None)
        if self.episode is not None:
            self.episode.reset()
            if not self.episode.private_state_empty():
                raise ContractError('resident original episode reset incomplete')
            self.episode = None
        self.cache.release()
        self.cache.reset_all()
        self.cache.synchronize()

    def diagnostic(self,command: str) -> Json:
        value=self.cache.diagnostic()
        if self.profile_host and command!='READY':
            try:
                import resource
                fields={line.partition(':')[0]:int(line.partition(':')[2].split()[0])*1024
                    for line in Path('/proc/self/smaps_rollup').read_text().splitlines() if ':' in line}
                value['host_memory']={'scope':'process-current-after-command','boundary':command,
                    'pss_bytes':fields['Pss'],'private_bytes':fields['Private_Clean']+fields['Private_Dirty'],
                    'shared_clean_bytes':fields['Shared_Clean'],'shared_dirty_bytes':fields['Shared_Dirty'],
                    'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            except (OSError,ValueError,KeyError):pass  # Optional diagnostics never change transport/control.
        return value

    def exchange(self, envelope: Json) -> Json:
        closed(envelope, {'request'}, {'payload','owned'} | ({'shared_request','shared_deadline'} if self.score_channel is not None else set()))
        if self.score_channel is not None:self.score_channel.bind_request(envelope.get('shared_request'),envelope.get('shared_deadline'))
        if 'owned' in envelope:
            if self.owned_channel is None or envelope['request'].get('command') not in ('BEGIN','SYNC') or self._owned_request is not None:
                raise ContractError('actor9 SYNC_OWNED requires original idle BEGIN/SYNC boundary')
            self._owned_request = envelope['owned']
            self._owned_outer_request=envelope['request']
        request = envelope['request']
        if type(request) is not dict or request.get('protocol') != self._protocol or request.get('schema_version') != (2 if self.score_channel is not None else 1):
            raise ContractError('resident original envelope required')
        integer(request.get('schema_version'),2 if self.score_channel is not None else 1,2 if self.score_channel is not None else 1)
        if self.owner is None:
            closed(request, CACHE_OWNER_KEYS | {'schema_version', 'protocol', 'command', 'run_id'})
            if (request['command'] != 'READY' or request['viewer'] != self.viewer
                    or request['pid'] != os.getpid() or request['lease_epoch'] != 0):
                raise ContractError('resident clean bootstrap original identity differs')
            integer(request['lease_epoch'],0,0)
            cache_owner({key:request[key] for key in CACHE_OWNER_KEYS} | {'lease_epoch':1})
            self.owner = {key: request[key] for key in CACHE_OWNER_KEYS - {'lease_epoch'}} | {'run_id': request['run_id']}
            return {**request, 'kind': 'READY', 'clean_reset': True, 'private_state_empty': True,
                **self._reset_fields()}
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
        self._check_ack({**request, **self._reset_fields()})
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
        return {**request,**self._reset_fields(),'private_state_empty':self.cache.private_state_empty()}


def main() -> None:
    output, source = sys.stdout.buffer, sys.stdin.buffer
    try:
        if len(_VIEWER) not in (1, 3) or _VIEWER[0] not in ('1', '2'):
            raise ContractError('resident viewer must be sanitized1or2')
        channel = None
        score_channel=None
        if len(_VIEWER)==3 and _VIEWER[1]=='--shared':
            from gf_simulator_client.shared_channel import ParticipantScoreChannel
            score_channel=ParticipantScoreChannel(int(_VIEWER[2]))
        elif len(_VIEWER) == 3:
            from gf_common.ram_bank import RamBankChannel
            channel = RamBankChannel._from_inherited(int(_VIEWER[1]), _VIEWER[2])
        transport = ResidentEpisodeTransport(int(_VIEWER[0]), owned_channel=channel,score_channel=score_channel)
        while True:
            envelope = read_frame(source)
            response = transport.exchange(envelope)
            if envelope['request']['command'] != 'SELECT':
                # Separate outer diagnostic leaf; never part of a strict ACK.
                response['_diagnostic'] = transport.diagnostic(envelope['request']['command'])
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
