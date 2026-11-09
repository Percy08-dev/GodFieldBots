"""Owned wire3 adapter over the existing low-level numerical process lifecycle."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_models.structured_binding import child_cuda_visibility
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_runtime import learner_process_bytes
from gf_models.structured_sealed import (
    SealedScan,
    inspect_retained_artifact,
    inspect_retained_game_artifact_v3,
)
from gf_simulator_client.structured_resources import LearnerReservation, ReservationLedger
from gf_training.ppo_run import LearningProcess
from gf_training.structured_numeric_worker import checksum, validate_reply, wire_identity


class StructuredLearningProcess(LearningProcess):
    def __init__(self, directory: Path, max_log_bytes: int, model: Any, binding: InferenceBindingV2,
                 action_config_hash: str, resume_identity: str, *, action_contract: Any,
                 ledger: ReservationLedger, cuda_visible_devices: str | None, cleanup_deadline: float,
                 _actions: ActionConfigV3 | None = None) -> None:
        super().__init__(directory, max_log_bytes)
        self.actions = _actions
        self.protocol = 3 if _actions is None else 4
        self._issued_handle: Any = None
        self.model, self.binding, self.action_contract = model, binding, action_contract
        self.visibility = child_cuda_visibility(binding.device, cuda_visible_devices)
        self.identity = wire_identity(model, binding, action_config_hash, resume_identity,
            action_contract=action_contract, purpose='component', synthetic_capacity=False,
            cuda_visible_devices=self.visibility, _actions=self.actions)
        self.ledger, self.cleanup_deadline = ledger, cleanup_deadline
        self.resident_bytes = learner_process_bytes(model, binding, **({"layout": CORRECTED_CARD_INSTANCE_LAYOUT} if self.actions is not None else {}))
        self.lease: LearnerReservation | None = None
        self.hardware: Json | None = None
        self.steps: int | None = None
        self._terminal_closed = False
        self._startup_boundary = asyncio.Event()

    def _environment(self) -> dict[str, str]:
        environment = super()._environment()
        if self.visibility is not None:
            environment['CUDA_VISIBLE_DEVICES'] = self.visibility
        return environment

    def _process_created(self) -> None:
        if self.lease is None:
            raise ContractError('numerical process has no pre-spawn reservation')
        self.lease.spawned(self.process)

    async def start(self) -> None:
        if self.lease is not None or self._terminal_closed or self.close_task is not None:
            raise ContractError('wire3 process already started or closed')
        self.lease = self.ledger.reserve_learner(self.resident_bytes, self)
        self.lease.creation_pending = True
        try:
            try:
                await super().start()
            finally:
                # Close must not adopt/register a returned process concurrently
                # with start's assignment/registration/drainer setup. Publish
                # this boundary BEFORE startup's own error-cleanup joins close.
                self._startup_boundary.set()
        except BaseException as error:
            try:
                await self.close(self.cleanup_deadline)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve initiating cancellation/failure
                error.add_note('wire3 startup cleanup: ' + type(cleanup_error).__name__)
            raise

    def _close_request(self) -> Json:
        return {'protocol_version': self.protocol, 'command': 'close', 'identity': self.identity}

    def _validate_close(self, result: Json) -> None:
        validate_reply(result, 'close', self.identity)

    async def _settle_creation(self, deadline: float) -> None:
        if self.creation_task is not None and not self.creation_task.done():
            async with asyncio.timeout_at(deadline):
                await asyncio.shield(self.creation_task)
        if self.lease is not None and not self._startup_boundary.is_set():
            async with asyncio.timeout_at(deadline):
                await self._startup_boundary.wait()
        if (self.creation_task is not None and self.creation_task.done()
                and not self.creation_task.cancelled() and self.creation_task.exception() is None):
            self._adopt_process(self.creation_task.result())
        if self.lease is not None and (self._creation_no_child or self.creation_task is None):
            self.lease.creation_pending = False
        elif self.process is None and self.lease is not None and self.lease.creation_pending:
            raise ContractError('numerical creation has no authoritative settlement')

    async def _close(self, deadline: float) -> None:
        try:
            await self._settle_creation(deadline)
            await super()._close(deadline)
        finally:
            if self.lease is not None and not self.lease.released:
                if self.process is not None and self.process.returncode is not None:
                    if self.ledger.updating:
                        self.ledger.finish_update()
                    self.lease.creation_pending = False
                    self.lease.reaped(self.process)
                elif self.process is None and not self.lease.creation_pending:
                    self.lease.cancel_before_spawn()

    async def close(self, deadline: float) -> None:
        task = self.close_task
        unreclaimed = self.lease is not None and not self.lease.released
        if task is None or (task.done() and unreclaimed
                            and (task.cancelled() or task.exception() is not None)):
            task = self.close_task = asyncio.create_task(self._close(deadline))
        # Capture this attempt locally: a later retry must not change the task
        # whose outcome an earlier caller observes. Concurrent callers join it.
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException:  # noqa: BLE001 - preserve this attempt's exact failure
                break
        if self.lease is None or self.lease.released:
            self._terminal_closed = True
        task.result()
        if cancellation:
            raise cancellation

    async def initialize(self, seed: int, output: Path, *, restore: Path | None, deadline: float) -> Json:
        integer(seed, 0, 2**63 - 1)
        return await self.request({'protocol_version': self.protocol, 'command': 'initialize', 'identity': self.identity,
            'model': asdict(self.model), 'seed': seed, 'output': str(output),
            'restore': None if restore is None else str(restore)}, deadline)

    async def update(self, batch: Path, batch_digest: str, expected_policy_version: str,
                     output: Path, *, deadline: float) -> Json:
        return await self.request({'protocol_version': self.protocol, 'command': 'update', 'identity': self.identity,
            'batch': str(batch), 'batch_digest': checksum(batch_digest),
            'expected_policy_version': expected_policy_version, 'output': str(output)}, deadline)

    async def request(self, request: Json, deadline: float) -> Json:
        command = request.get('command')
        common = {'protocol_version', 'command', 'identity'}
        fields = ({'model', 'seed', 'output', 'restore'} if command == 'initialize' else
                  {'batch', 'batch_digest', 'expected_policy_version', 'output'} | ({'transaction_id', 'projection'} if self.actions is not None else set()) if command == 'update' else set())
        closed(request, common | fields)
        if (type(request['protocol_version']) is not int or request['protocol_version'] != self.protocol
                or command not in ('initialize', 'update', 'close')
                or canonical(request['identity']) != canonical(self.identity)):
            raise ContractError('wire3 request binding/command mismatch')
        if command != 'close' and (self.lease is None or self.lease.released
                                  or self.process is None or self.process.returncode is not None):
            raise ContractError('wire3 numerical process is not live/owned')
        if command == 'initialize' and self.steps is not None:
            raise ContractError('wire3 duplicate initialization')
        if command == 'initialize':
            integer(request['seed'], 0, 2**63 - 1)
            if canonical(request['model']) != canonical(asdict(self.model)):
                raise ContractError('wire3 initialization model differs from reserved model')
            if request['restore'] is not None and (type(request['restore']) is not str or not request['restore']):
                raise ContractError('wire3 restore path required')
        if command != 'close' and (type(request['output']) is not str or not request['output']):
            raise ContractError('wire3 output path required')
        if command == 'update':
            if self.actions is not None:
                from gf_training.card_instance_source import IssuedGameProjection
                if type(self._issued_handle) is not IssuedGameProjection:
                    raise ContractError('wire4 raw request cannot issue update authority')
                self._issued_handle.require(self)
                from gf_training.lineage import checkpoint_path
                expected_output = checkpoint_path(self._issued_handle.source.online.ownership.directory,
                    self._issued_handle.expectation['collection_generation'] + 1)
                if Path(request['batch']).absolute() != self._issued_handle.numeric.absolute() or Path(request['output']).absolute() != expected_output:
                    raise ContractError('wire4 paths differ from live owned batch/next generation')
                from gf_training.structured_numeric_worker import (
                    digest_projection,
                    validate_projection,
                )
                expected = self._issued_handle.expectation
                if canonical(request['projection']) != canonical(expected) or request['transaction_id'] != digest_projection(expected):
                    raise ContractError('wire4 request differs from live issued projection')
                if self.steps is None:
                    raise ContractError('wire4 update before initialization')
                validate_projection(expected, expected_steps=self.steps, policy=request['expected_policy_version'], batch_digest=request['batch_digest'])
            checksum(request['batch_digest'])
            if any(type(request[key]) is not str or not request[key] for key in ('batch', 'expected_policy_version')):
                raise ContractError('wire3 batch path/policy required')
            if self.steps is None:
                raise ContractError('wire3 update before initialization')
            self.ledger.begin_update(0)
        reply = await super().request(request, deadline)
        validate_reply(reply, command, self.identity)
        if command == 'close':
            return reply
        if command == 'update' and reply['metrics']['sequence_length'] != self.model.temporal.sequence_length:
            raise ContractError('wire3 configured W mismatch')
        if self.hardware is not None and canonical(reply['hardware']) != canonical(self.hardware):
            raise ContractError('wire3 numerical hardware changed')
        output = Path(request['restore'] or request['output']) if command == 'initialize' else Path(request['output'])
        scan = SealedScan(max_owned_bytes=self.model.resources.max_owned_bytes, deadline=deadline)
        artifact = (inspect_retained_game_artifact_v3(output, expected_inference_digest=reply['artifact']['inference_digest'], actions=self.actions, rules=card_instance_rules(), scan=scan) if self.actions is not None else
            inspect_retained_artifact(output, expected_inference_digest=reply['artifact']['inference_digest'],
            action_contract=self.action_contract, purpose='component', synthetic_capacity=False, scan=scan))
        if artifact.resume is None:
            raise ContractError('wire3 requires retained resume metadata')
        if (canonical(artifact.files) != canonical(reply['artifact']['files'])
                or canonical(artifact.inference['model']) != canonical(asdict(self.model))
                or canonical(artifact.resume['binding']) != canonical(asdict(self.binding))
                or artifact.inference['action_config_hash'] != self.identity['action_config_hash']
                or artifact.resume['resume_identity'] != self.identity['resume_identity']
                or artifact.resume['optimizer_steps'] != reply['artifact']['optimizer_steps']
                or artifact.resume['cuda_rng_count'] != reply['hardware']['device_count']):
            raise ContractError('wire3 artifact acknowledgement differs from verified files/binding')
        scan.finalize()
        steps = reply['artifact']['optimizer_steps']
        if command == 'update':
            if self.actions is not None and (reply['transaction_id'] != request['transaction_id'] or reply['collection_generation'] != request['projection']['collection_generation'] or reply['expected_optimizer_steps'] != self.steps):
                raise ContractError('wire4 reply transaction/generation mismatch')
            expected = self.steps + 4 * ((reply['metrics']['parity']['decisions'] + 127) // 128)
            if steps != expected:
                raise ContractError('wire3 optimizer advancement mismatch')
            self.ledger.finish_update()
        elif self.steps is not None or request['restore'] is None and steps != 0:
            raise ContractError('wire3 duplicate initialization/counter mismatch')
        self.steps, self.hardware = steps, reply['hardware']
        return reply


class GameLearningProcess(StructuredLearningProcess):
    def __init__(self, directory: Path, max_log_bytes: int, model: Any, binding: InferenceBindingV2,
                 action_config_hash: str, resume_identity: str, *, actions: ActionConfigV3,
                 ledger: ReservationLedger, cuda_visible_devices: str | None, cleanup_deadline: float) -> None:
        super().__init__(directory, max_log_bytes, model, binding, action_config_hash, resume_identity,
            action_contract=actions.action_contract, ledger=ledger, cuda_visible_devices=cuda_visible_devices,
            cleanup_deadline=cleanup_deadline, _actions=actions)
        self.online_owner: Any = None

    async def update(self, projection: Any, output: Path, *, deadline: float) -> Json:  # type: ignore[override]
        from gf_training.card_instance_source import IssuedGameProjection
        from gf_training.structured_numeric_worker import digest_projection
        if type(projection) is not IssuedGameProjection:
            raise ContractError('wire4 requires a live Source-issued projection')
        projection.require(self)
        self._issued_handle = projection
        try:
            reply = await self.request({'protocol_version': 4, 'command': 'update', 'identity': self.identity,
                'batch': str(projection.numeric), 'batch_digest': projection.expectation['numeric_manifest_sha256'],
                'expected_policy_version': projection.expectation['expected_policy_version'], 'output': str(output),
                'transaction_id': digest_projection(projection.expectation), 'projection': projection.expectation}, deadline)
            projection.completed = True
            return reply
        finally:
            self._issued_handle = None
