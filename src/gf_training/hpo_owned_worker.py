"""Wire7 clean numerical child. Update data arrives only through owned FDs."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.artifacts import ArtifactAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.ram_bank import MAX_BANK_BYTES, RamBankBuilder, RamBankChannel, ReadonlyRamBankSet
from gf_models.arrays import MAX_ARRAY_BYTES


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
                     identity: str, budget: ArtifactAdmission) -> _Control:
    delivery = channel.receive(purpose=purpose, generation=generation, identity=identity,
        budget=budget, max_bytes=MAX_ARRAY_BYTES)
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


def main() -> None:
    # Explicit inherited endpoint only; no inherited resume/model/numeric FD.
    fd, invocation = int(sys.argv[1]), sys.argv[2]
    sys.argv[:] = ['gf-owned-hpo-learner']
    channel = RamBankChannel._from_inherited(fd, invocation)
    initial = json.loads(sys.stdin.readline())
    closed(initial, {'protocol_version', 'command', 'identity', 'model', 'seed', 'output', 'transaction'})
    if type(initial['protocol_version']) is not int or initial['protocol_version'] != 7 or initial['command'] != 'initialize_owned':
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
    from gf_training.hpo_artifacts import save_estimate
    limits=identity['phase_limits']
    for value in limits.values(): integer(value,1,MAX_BANK_BYTES)
    if save_estimate(model)['additional_bytes']>limits['save_budget_bytes']:
        raise ContractError('Wire7 additional save envelope insufficient before learner allocation')
    from gf_training.hpo_numeric import HpoLearner
    learner = HpoLearner(model, training, integer(initial['seed'],0,2**63-1), binding,
        identity['resume_identity'], actions=actions, rules=card_instance_rules())
    # Invocation-local Source admission; phase-specific limits are also enforced.
    budget = StorageBudget(StorageSettings(), float('inf'))
    snapshot = learner.snapshot_owned(budget=budget, save_budget_bytes=identity['phase_limits']['save_budget_bytes'])
    snapshot.save(Path(initial['output']))
    current_snapshot = snapshot.identity
    _send_snapshot(channel, snapshot, request=initial, budget=budget, metrics={}, identity=identity, learner=learner)
    generation = 0
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
        numeric, deliveries, control = _receive_numeric(channel, request, learner, budget, current_snapshot, actions, identity['phase_limits']['numeric_budget_bytes'])
        metrics = learner.update_owned(numeric)
        del numeric
        learner.synchronize()
        for delivery in deliveries:
            delivery.complete(learner.synchronize)
        control.close()
        generation = learner._owned_generation
        snapshot = learner.snapshot_owned(budget=budget, save_budget_bytes=identity['phase_limits']['save_budget_bytes'])
        snapshot.save(Path(request['output']))
        current_snapshot = snapshot.identity
        _send_snapshot(channel, snapshot, request=request, budget=budget, metrics=metrics, identity=identity, learner=learner)
    raise ContractError('Wire7 EOF before owned close')


if __name__ == '__main__':
    main()
