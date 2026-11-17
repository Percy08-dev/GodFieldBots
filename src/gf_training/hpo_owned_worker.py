"""Wire7 clean numerical child. Update data arrives only through owned FDs."""
from __future__ import annotations

import json
import mmap
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.artifacts import ArtifactAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.ram_bank import (
    MAX_ARRAYS,
    MAX_BANK_BYTES,
    MAX_ENVELOPE_BYTES,
    RamBankBuilder,
    RamBankChannel,
    ReadonlyRamBankSet,
)
from gf_models.arrays import MAX_ARRAY_BYTES

_RX_TRANSFER_BYTES=64*MAX_ENVELOPE_BYTES+1024*MAX_ARRAYS
_RETAIN_BOOK_BYTES=MAX_ENVELOPE_BYTES+1024*MAX_ARRAYS+mmap.PAGESIZE


def _remaining_snapshot_bytes(phase_limit: int, used: int, *, bookkeeping: int=_RX_TRANSFER_BYTES) -> int:
    """Separate aggregate phase allowance from this one immutable bank's cap."""
    integer(phase_limit,1,2**63-1)
    integer(used,0,phase_limit)
    integer(bookkeeping,0,2**63-1)
    remaining=phase_limit-used-bookkeeping
    if remaining<=0: raise ContractError('owned receive aggregate phase exhausted before allocation')
    return min(MAX_BANK_BYTES,remaining)


class _Control:
    def __init__(self, value: Json, reservation: Any, admitted_bytes: int) -> None:
        self.value, self._reservation = value, reservation
        self.admitted_bytes = admitted_bytes

    def close(self) -> None:
        self.value = {}
        self._reservation.close()


def _control_bank(value: Json, budget: ArtifactAdmission) -> Any:
    size = json_bound(value)
    if size > MAX_ARRAY_BYTES:
        raise ContractError('owned JSON control bank exceeds bound before encoding')
    with budget.reserve(size * 2 + 65536):
        raw = canonical(value).encode()
        return RamBankBuilder({'control': np.frombuffer(raw, dtype=np.uint8)}, budget,
            max_bytes=MAX_ARRAY_BYTES).seal()


def _receive_control(channel: RamBankChannel, *, purpose: str, generation: int,
                     identity: str, budget: ArtifactAdmission, max_admitted_bytes: int | None=None) -> _Control:
    maximum=MAX_ARRAY_BYTES
    if max_admitted_bytes is not None:
        remaining=_remaining_snapshot_bytes(max_admitted_bytes,0)
        maximum=min(maximum,(remaining-65536)//33)
        if maximum<=0: raise ContractError('owned control phase exhausted before receive/decode')
    delivery = channel.receive(purpose=purpose, generation=generation, identity=identity,
        budget=budget, max_bytes=maximum)
    reservation = None
    values = None
    try:
        descriptor = delivery.bank.descriptors
        if (len(descriptor) != 1 or descriptor[0].name != 'control'
                or descriptor[0].dtype != '|u1' or len(descriptor[0].shape) != 1):
            raise ContractError('owned closed JSON control layout required')
        reservation = budget.reserve(33 * delivery.bank.size + 65536)
        def pairs(items: Any) -> Json:
            output: Json = {}
            for key, value in items:
                if key in output:
                    raise ContractError('duplicate owned control key')
                output[key] = value
            return output
        values = delivery.bank.array('control')
        raw = values.tobytes()
        values = None
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ContractError('nonfinite owned control')))
        if type(value) is not dict:
            raise ContractError('owned JSON control object required')
        json_bound(value)
        delivery.complete(lambda: None)
        return _Control(value, reservation, 33 * delivery.bank.size + 65536)
    except BaseException as error:
        values = None
        if reservation is not None:
            reservation.close()
        try:
            channel.retire_received(lambda: None)
        except BaseException as cleanup:  # noqa: BLE001 - preserve decode/ACK failure and exact retained custody
            error.add_note('owned control receive cleanup retained: '+type(cleanup).__name__)
        raise


def _receive_numeric(channel: RamBankChannel, request: Json, learner: Any,
                     budget: ArtifactAdmission, current_snapshot: str, actions: Any, numeric_limit: int) -> tuple[Any, list[Any], _Control]:
    from gf_training.hpo_owned_rollout import _TOKEN, OwnedNumericBatch, _validate_owned_layout
    from gf_training.structured_rollout import _RolloutContext
    generation = request['generation']
    control = _receive_control(channel, purpose='batch', generation=generation,
        identity=request['batch_identity'] + ':metadata', budget=budget)
    try:
        metadata = control.value
        if (metadata.get('schema_version') != 6 or metadata.get('origin') != 'producer-owned-v1'
                or metadata.get('encoder') != 'canonical-python' or metadata.get('snapshot_identity') != current_snapshot
                or digest({'generation': generation, 'metadata': metadata}) != request['batch_identity']
                or canonical(metadata.get('training_recipe')) != canonical(asdict(learner.training))
                or metadata.get('policy_version') != request['policy_version']):
            raise ContractError('Wire7 batch metadata/model snapshot mismatch')
        deliveries = []
        total = 0
        for index in range(integer(request['bank_count'], 1, 65536)):
            delivery = channel.receive(purpose='batch', generation=generation,
                identity=request['batch_identity'] + f':{index}', budget=budget,
                max_bytes=numeric_limit-total)
            total += delivery.bank.size
            deliveries.append(delivery)
        banks = ReadonlyRamBankSet(tuple(delivery.bank for delivery in deliveries))
        if banks.admitted_bytes + banks.size + control.admitted_bytes > numeric_limit:
            raise ContractError("Wire7 aggregate numeric/descriptor/control admission exceeded")
        _validate_owned_layout(metadata, banks, model=learner.model, actions=actions)
        batch = OwnedNumericBatch(_TOKEN, banks, metadata, model=learner.model, training=learner.training,
            context=_RolloutContext(actions, learner._context.rules),
            budget=budget, generation=generation)
        batch.resident_bytes += control.admitted_bytes
        batch._numeric_limit = numeric_limit
        batch._state = 'PERSISTED'  # only actual bound Wire7 sender can enter this receiver
        return batch, deliveries, control
    except BaseException as error:
        try:
            channel.retire_received(lambda: None)
        except BaseException as cleanup:  # noqa: BLE001 - keep original numeric input failure
            error.add_note('owned numeric receive cleanup retained: '+type(cleanup).__name__)
        try:
            control.close()
        except BaseException as cleanup:  # noqa: BLE001 - keep actual failed control owner reachable
            error.add_note('owned numeric control cleanup retained: '+type(cleanup).__name__)
        raise


def _send_snapshot(channel: RamBankChannel, snapshot: Any, *, request: Json,
                   budget: ArtifactAdmission, metrics: Json, identity: Json, learner: Any) -> None:
    ack=snapshot.save_ack
    ack.require(snapshot)
    # The additional actual writer-manifest copy coexists with snapshot banks
    # and the admitted control encoder/transfer bank. Reserve before decoding it.
    extra=33*len(ack._inference_manifest)+65536
    if snapshot.resident_bytes+extra>snapshot._save_budget_limit:
        raise ContractError('owned save transfer metadata envelope exceeded')
    with budget.reserve(extra):
        _send_snapshot_admitted(channel,snapshot,request=request,budget=budget,metrics=metrics,
            identity=identity,learner=learner,extra_bytes=extra)


def _send_snapshot_admitted(channel: RamBankChannel, snapshot: Any, *, request: Json,
                   budget: ArtifactAdmission, metrics: Json, identity: Json, learner: Any, extra_bytes: int) -> None:
    ack = snapshot.save_ack
    ack.require(snapshot)
    transaction = request['transaction']
    from gf_models.torch_runtime import hardware
    control = {'actor_metadata': snapshot.actor_metadata, 'inference_manifest':ack.inference_manifest, 'files': ack.files,
        'hardware':hardware(learner.binding.device),'timings':dict(snapshot.timings),
        'optimizer_steps': snapshot.optimizer_steps, 'generation': snapshot.generation,
        'snapshot_identity': snapshot.identity, 'metrics': metrics,
        'inference_banks': len(snapshot.inference.banks), 'resume_banks': len(snapshot.resume.banks)}
    if snapshot.resident_bytes+extra_bytes+35*json_bound(control)+65536>snapshot._save_budget_limit:
        raise ContractError('owned save control transfer envelope exceeded')
    bank = _control_bank(control, budget)
    print(canonical({'protocol_version': 7, 'kind': 'SNAPSHOT_READY', 'transaction': transaction,
        'identity': identity, 'generation': snapshot.generation, 'snapshot_identity': snapshot.identity}), flush=True)
    leases = [channel.send(bank, purpose='save', generation=snapshot.generation, identity=snapshot.identity+':metadata')]
    for kind, banks in (('inference', snapshot.inference), ('resume', snapshot.resume)):
        for index, backing in enumerate(banks.banks):
            leases.append(channel.send(backing, purpose='save', generation=snapshot.generation,
                identity=snapshot.identity+f':{kind}:{index}'))
    for lease in leases:
        channel.accept_completion(lease)
    bank.close()
    snapshot.close()
    print(canonical({'protocol_version': 7, 'kind': 'SAVED', 'transaction': transaction,
        'identity': identity, 'generation': snapshot.generation, 'optimizer_steps': learner.steps,
        'snapshot_identity': snapshot.identity, 'files_sha256': digest(ack.files)}), flush=True)


def _restore_initial(channel: RamBankChannel, request: Json, learner: Any, budget: Any,
                     actions: Any, identity: Json) -> str:
    """Receive only actual model-state grants, stage, synchronize, then restore ACK."""
    generation=integer(request['generation'],1,2**31-1)
    integer(request['optimizer_steps'],1,2**31-1)
    prefix='restore:'+request['transaction']+':'+request['snapshot_identity']
    control=_receive_control(channel,purpose='model',generation=generation,identity=prefix+':metadata',budget=budget,
        max_admitted_bytes=identity['phase_limits']['save_budget_bytes'])
    deliveries=[]
    try:
        facts=closed(control.value,{'actor_metadata','inference_manifest','files','generation','snapshot_identity',
            'optimizer_steps','resume_manifest'})
        if (facts['generation']!=generation or facts['optimizer_steps']!=request['optimizer_steps']
                or facts['snapshot_identity']!=request['snapshot_identity']
                or facts['resume_manifest'].get('optimizer_steps')!=request['optimizer_steps']
                or facts['actor_metadata'].get('inference_digest')!=facts['files'].get('inference.json')
                or facts['resume_manifest'].get('inference_digest')!=facts['files'].get('inference.json')):
            raise ContractError('restore model control differs from transaction')
        from gf_training.hpo_owned_process import _saved_files
        _saved_files(facts['files'])
        banksets=[]; total=control.admitted_bytes
        for kind in ('inference','resume'):
            banks=[]
            for index in range(integer(request[kind+'_banks'],1,65536)):
                delivery=channel.receive(purpose='model',generation=generation,identity=prefix+f':{kind}:{index}',
                    budget=budget,max_bytes=_remaining_snapshot_bytes(identity['phase_limits']['save_budget_bytes'],total))
                deliveries.append(delivery); banks.append(delivery.bank); total+=delivery.bank.size+delivery.bank.admitted_bytes
            banksets.append(ReadonlyRamBankSet(tuple(banks)))
        from gf_training.hpo_artifacts import _restore_owned_learner
        _restore_owned_learner(learner,inference=banksets[0],resume=banksets[1],actor_metadata=facts['actor_metadata'],
            resume_manifest=facts['resume_manifest'],generation=generation,budget=budget,actions=actions)
        if learner.steps!=request['optimizer_steps']: raise ContractError('restored optimizer steps mismatch')
        learner.synchronize()
        for delivery in deliveries: delivery.complete(learner.synchronize)
        files_sha=digest(facts['files'])
        snapshot_identity=facts['snapshot_identity']
        control.close()
        from gf_models.torch_runtime import hardware
        print(canonical({'protocol_version':7,'kind':'RESTORED','identity':identity,'transaction':request['transaction'],
            'generation':generation,'optimizer_steps':learner.steps,'snapshot_identity':snapshot_identity,
            'files_sha256':files_sha,'hardware':hardware(learner.binding.device)}),flush=True)
        return snapshot_identity
    except BaseException:
        learner._owned_failed=True
        # No completion ACK on failed state. Original receiver inventory owns
        # cleanup through actual process exit/reap, including live array aliases.
        raise


def main() -> None:
    # Explicit inherited endpoint only; no inherited resume/model/numeric FD.
    if len(sys.argv) not in (5,7,8):
        raise ContractError('closed owned learner argv/cap required')
    fd, invocation = int(sys.argv[1]), sys.argv[2]
    from gf_training.hpo_learning_process import parse_execution_cap
    reservation=parse_execution_cap(tuple(sys.argv[3:5]))
    execution = None
    execution_sha = None
    observation_source = None
    owned_context = None
    if len(sys.argv)==7:
        if sys.argv[5]!='--owned-learner-admission':
            raise ContractError('closed private owned learner admission argument required')
        owned_context = json.loads(sys.argv[6])
    if len(sys.argv)==8:
        if sys.argv[5]!='--training-execution':
            raise ContractError('closed private training execution argument required')
        from gf_training.hpo_training_execution import read_execution_spec
        execution = read_execution_spec(Path(sys.argv[6]),sys.argv[7])
        execution_sha = sys.argv[7]
    sys.argv[:] = ['gf-owned-hpo-learner']
    channel = RamBankChannel._from_inherited(fd, invocation)
    initial = json.loads(sys.stdin.readline())
    restoring=initial.get('command')=='restore_owned'
    closed(initial, {'protocol_version','command','identity','model','seed','transaction'} |
        ({'generation','optimizer_steps','snapshot_identity','inference_banks','resume_banks'} if restoring else {'output'}))
    if type(initial['protocol_version']) is not int or initial['protocol_version'] != 7 or initial['command'] not in ('initialize_owned','restore_owned'):
        raise ContractError('Wire7 initialize required')
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.group_pack import StorageSettings
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
    from gf_models.hpo_config import model_config_v4
    from gf_models.structured_config import InferenceBindingV2
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_numeric_worker import wire_identity
    from gf_training.structured_numeric_worker import _select_visibility
    identity = initial['identity']
    model = model_config_v4(initial['model'], layout=LAYOUT)
    training = hpo_training_config(identity['training_recipe'])
    binding = InferenceBindingV2(**identity['execution']['binding'])
    actions = action_config_v3(identity['actions'])
    expected = wire_identity(model, training, binding, identity['action_config_hash'], identity['resume_identity'],
        actions=actions, cuda_visible_devices=identity['execution']['cuda_visible_devices'],
        storage_binding=identity.get('storage_binding'), phase_limits=identity['phase_limits'])
    if canonical(identity) != canonical(expected):
        raise ContractError('Wire7 initialization identity mismatch')
    _select_visibility(initial, binding, identity['execution']['cuda_visible_devices'])
    from gf_training.hpo_artifacts import checked_phase_limits, save_estimate
    limits=checked_phase_limits(identity['phase_limits'])
    integer(limits['numeric_budget_bytes'],1,MAX_BANK_BYTES)
    if save_estimate(model)['additional_bytes']>limits['save_budget_bytes']:
        raise ContractError('Wire7 additional save envelope insufficient before learner allocation')
    if execution is not None and (restoring or reservation is None
            or reservation > min(model.resources.max_owned_bytes,binding.managed_budget_bytes)
            or execution.variant == 'shared-prefix' and model.model_id == 'mlp'):
        raise ContractError('experimental fresh learner reservation/model mismatch')
    from gf_training.hpo_learning_process import checked_execution_cap
    reservation=checked_execution_cap(model,training,binding,reservation)
    owned_admission = None
    if owned_context is not None:
        from gf_training.owned_learner_admission import checked_owned_launch_context
        owned_admission = checked_owned_launch_context(owned_context,model,training,binding)['admission']
        if owned_admission['logical_bytes']!=reservation:
            raise ContractError('owned learner launch logical cap differs from original checked ceiling')
    from gf_training.hpo_numeric import HpoLearner
    learner = HpoLearner(model, training, integer(initial['seed'],0,2**63-1), binding,
        identity['resume_identity'], actions=actions, rules=card_instance_rules(),
        **({} if owned_admission is None else {'_owned_device_admission':owned_admission}))
    learner._training_reservation_bytes = reservation
    if execution is not None:
        learner._training_execution = execution
    # Invocation-local Source admission; phase-specific limits are also enforced.
    budget = StorageBudget(StorageSettings(), float('inf'))
    if restoring:
        current_snapshot=_restore_initial(channel,initial,learner,budget,actions,identity)
        generation=learner._owned_generation
    else:
        snapshot = learner.snapshot_owned(budget=budget, save_budget_bytes=identity['phase_limits']['save_budget_bytes'])
        snapshot.save(Path(initial['output']))
        current_snapshot = snapshot.identity
        _send_snapshot(channel, snapshot, request=initial, budget=budget, metrics={}, identity=identity, learner=learner)
        generation = 0
    if execution is not None and execution.observe_actual:
        from gf_models.structured_artifacts import producer_identity
        try:
            observation_source = producer_identity()['source_digest']
        except BaseException:  # noqa: BLE001, S110 - missing source binding makes diagnostics incomplete, not learning
            pass
    from gf_common.hpo_progress import optional_diagnostic_emitter
    diagnostic=optional_diagnostic_emitter(os.environ,sys.stderr)
    last_transaction = initial['transaction']
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('command') == 'close':
            closed(request, {'protocol_version', 'command', 'identity'})
            if request['protocol_version'] != 7 or canonical(request['identity']) != canonical(identity):
                raise ContractError('Wire7 close identity mismatch')
            print(canonical({'protocol_version':7,'command':'close','identity':identity,'closed':True}),flush=True)
            channel.close()
            return
        closed(request, {'protocol_version', 'command', 'identity', 'transaction', 'generation',
            'expected_steps', 'batch_identity', 'bank_count', 'policy_version', 'output'})
        if (request['protocol_version'] != 7 or type(request['protocol_version']) is not int
                or request['command'] != 'update_owned' or canonical(request['identity']) != canonical(identity)
                or request['generation'] != generation or type(request['generation']) is not int
                or type(request['expected_steps']) is not int or request['expected_steps'] != learner.steps
                or request['transaction'] == last_transaction):
            raise ContractError('Wire7 original update generation/transaction mismatch')
        last_transaction = request['transaction']
        observation = None
        cuda_begin = cuda_end = None
        original_completed = False
        if execution is not None and execution.observe_actual:
            try:
                from gf_training.hpo_execution_observation import ProcessObservation
                assert execution.diagnostic_directory is not None
                Path(execution.diagnostic_directory).mkdir(parents=True,exist_ok=True)
                observation = ProcessObservation({'role':'worker','generation':request['generation'],
                    'transaction':request['transaction'],'resume_identity':learner.resume_identity,
                    'model_id':model.model_id,'variant':execution.variant,'spec_sha256':execution_sha,
                    'source_digest':observation_source,'pid':os.getpid(),'run_id':None,'context_id':None},
                    allowance_bytes=model.resources.workspace_bytes)
                observation.start()
                if binding.device!='cpu':
                    observation.cuda={'status':'unavailable'}
                    learner.torch.cuda.reset_peak_memory_stats(binding.device)
                    cuda_begin=learner.torch.cuda.Event(enable_timing=True)
                    cuda_end=learner.torch.cuda.Event(enable_timing=True)
            except BaseException as error:  # noqa: BLE001 - observation never changes original update ownership
                if observation is not None:
                    observation.failure('observation/CUDA start',error)
        try:
            with observation.phase('receive_validation') if observation is not None else nullcontext():
                numeric, deliveries, control = _receive_numeric(channel, request, learner, budget, current_snapshot, actions, identity['phase_limits']['numeric_budget_bytes'])
            with observation.phase('update_call') if observation is not None else nullcontext():
                if cuda_begin is not None:
                    try:
                        cuda_begin.record()
                    except BaseException as error:  # noqa: BLE001 - no change to authoritative call
                        if observation is not None:
                            observation.failure('CUDA begin',error)
                if execution is not None and execution.profile_start is not None:
                    import resource

                    from gf_training.hpo_update_diagnostics import MAX_OUTPUT, ProfileWindow
                    assert execution.diagnostic_directory is not None
                    destination = Path(execution.diagnostic_directory) / f'profile-{generation:06d}'
                    destination.mkdir(parents=True,exist_ok=False)
                    resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_OUTPUT,MAX_OUTPUT))
                    profile = ProfileWindow(learner,destination,execution.profile_start,
                        execution.profile_groups,execution.profile_shapes)
                    try:
                        with profile.patches():
                            metrics = learner.update_owned(numeric)
                            profile.finish()
                        if not profile.finished:
                            raise ContractError('predeclared profiling group window not reached')
                        with (destination/'result.json').open('x') as stream:
                            stream.write(canonical(profile.summary))
                    finally:
                        profile.finish()
                else:
                    metrics = learner.update_owned(numeric)
                if cuda_end is not None:
                    try:
                        cuda_end.record()
                    except BaseException as error:  # noqa: BLE001 - no change to authoritative call
                        if observation is not None:
                            observation.failure('CUDA end',error)
            if execution is not None and execution.diagnostic_directory is not None and not execution.observe_actual:
                destination = Path(execution.diagnostic_directory)
                destination.mkdir(parents=True,exist_ok=True)
                counters = getattr(learner,'_training_update_diagnostics',{'variant':execution.variant})
                with (destination/f'update-{generation:06d}.json').open('x') as stream:
                    stream.write(canonical({'schema_version':1,'generation':generation,
                        'reservation_bytes':reservation,**counters}))
            with observation.phase('input_retirement') if observation is not None else nullcontext():
                del numeric
                learner.synchronize()
                for delivery in deliveries:
                    delivery.complete(learner.synchronize)
                control.close()
            if diagnostic is not None and not diagnostic.disabled:
                from gf_common.hpo_progress import update_execution_facts
                facts=update_execution_facts(learner)
                if facts:
                    diagnostic.emit(request['generation']+1,{**facts,'diagnostic_boundary':'update-synchronized'})
            if observation is not None:
                try:
                    observation.numeric_loop_update_seconds=float(metrics['update_seconds'])
                    if cuda_begin is not None and cuda_end is not None:
                        observation.cuda={'status':'observed','update_call_event_ms':cuda_begin.elapsed_time(cuda_end),
                            'allocator_after_retirement':{
                                'allocated_bytes':learner.torch.cuda.memory_allocated(binding.device),
                                'reserved_bytes':learner.torch.cuda.memory_reserved(binding.device),
                                'peak_allocated_bytes':learner.torch.cuda.max_memory_allocated(binding.device),
                                'peak_reserved_bytes':learner.torch.cuda.max_memory_reserved(binding.device)}}
                except BaseException as error:  # noqa: BLE001 - read after EXISTING sync; never add synchronization
                    observation.failure('CUDA/metric retirement read',error)
            generation = learner._owned_generation
            with observation.phase('snapshot_export') if observation is not None else nullcontext():
                snapshot = learner.snapshot_owned(budget=budget, save_budget_bytes=identity['phase_limits']['save_budget_bytes'])
            with observation.phase('snapshot_save') if observation is not None else nullcontext():
                snapshot.save(Path(request['output']))
            current_snapshot = snapshot.identity
            with observation.phase('snapshot_transfer') if observation is not None else nullcontext():
                _send_snapshot(channel, snapshot, request=request, budget=budget, metrics=metrics, identity=identity, learner=learner)
            original_completed = True
        finally:
            cuda_begin = cuda_end = None
            if observation is not None:
                # Publication is AFTER original send returns, before next read/
                # close. It cannot replace an original numerical/FD/save error.
                try:
                    assert execution is not None and execution.diagnostic_directory is not None
                    observation.finish_publish(Path(execution.diagnostic_directory)/f"update-{request['generation']:06d}.json",
                        original_completed=original_completed)
                except BaseException:  # noqa: BLE001, S110 - helper validates missing/incomplete sidecar after real reap
                    pass
    raise ContractError('Wire7 EOF before owned close')


if __name__ == '__main__':
    main()
