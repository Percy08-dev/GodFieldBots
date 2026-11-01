"""Owned wire3 adapter over the existing low-level numerical process lifecycle."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

from gf_common.action_contract import ActionContract
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_models.structured_binding import child_cuda_visibility
from gf_models.structured_config import InferenceBindingV2, ModelConfigV2
from gf_models.structured_runtime import learner_process_bytes
from gf_models.structured_sealed import SealedScan, inspect_retained_artifact
from gf_simulator_client.structured_resources import LearnerReservation, ReservationLedger
from gf_training.ppo_run import LearningProcess
from gf_training.structured_numeric_worker import checksum, validate_reply, wire_identity


class StructuredLearningProcess(LearningProcess):
    def __init__(self, directory: Path, max_log_bytes: int, model: ModelConfigV2, binding: InferenceBindingV2,
                 action_config_hash: str, resume_identity: str, *, action_contract: ActionContract,
                 ledger: ReservationLedger, cuda_visible_devices: str | None, cleanup_deadline: float) -> None:
        super().__init__(directory, max_log_bytes)
        self.model, self.binding, self.action_contract = model, binding, action_contract
        self.visibility = child_cuda_visibility(binding.device, cuda_visible_devices)
        self.identity = wire_identity(model, binding, action_config_hash, resume_identity,
            action_contract=action_contract, purpose='component', synthetic_capacity=False,
            cuda_visible_devices=self.visibility)
        self.ledger, self.cleanup_deadline = ledger, cleanup_deadline
        self.resident_bytes = learner_process_bytes(model, binding)
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
            if self.creation_task is None or self.creation_task.done():
                self.lease.creation_pending = False
            try:
                await self.close(self.cleanup_deadline)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve initiating cancellation/failure
                error.add_note('wire3 startup cleanup: ' + type(cleanup_error).__name__)
            raise

    def _close_request(self) -> Json:
        return {'protocol_version': 3, 'command': 'close', 'identity': self.identity}

    def _validate_close(self, result: Json) -> None:
        validate_reply(result, 'close', self.identity)

    async def _close(self, deadline: float) -> None:
        try:
            if self.creation_task is not None and not self.creation_task.done():
                async with asyncio.timeout_at(deadline):
                    await asyncio.shield(self.creation_task)
            if self.lease is not None and not self._startup_boundary.is_set():
                async with asyncio.timeout_at(deadline):
                    await self._startup_boundary.wait()
            if (self.process is None and self.creation_task is not None and self.creation_task.done()
                    and not self.creation_task.cancelled() and self.creation_task.exception() is None):
                self.process = self.creation_task.result()
                self._process_created()
            if self.lease is not None and (self.creation_task is None or self.creation_task.done()):
                self.lease.creation_pending = False
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
        return await self.request({'protocol_version': 3, 'command': 'initialize', 'identity': self.identity,
            'model': asdict(self.model), 'seed': seed, 'output': str(output),
            'restore': None if restore is None else str(restore)}, deadline)

    async def update(self, batch: Path, batch_digest: str, expected_policy_version: str,
                     output: Path, *, deadline: float) -> Json:
        return await self.request({'protocol_version': 3, 'command': 'update', 'identity': self.identity,
            'batch': str(batch), 'batch_digest': checksum(batch_digest),
            'expected_policy_version': expected_policy_version, 'output': str(output)}, deadline)

    async def request(self, request: Json, deadline: float) -> Json:
        command = request.get('command')
        common = {'protocol_version', 'command', 'identity'}
        fields = ({'model', 'seed', 'output', 'restore'} if command == 'initialize' else
                  {'batch', 'batch_digest', 'expected_policy_version', 'output'} if command == 'update' else set())
        closed(request, common | fields)
        if (type(request['protocol_version']) is not int or request['protocol_version'] != 3
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
        artifact = inspect_retained_artifact(output, expected_inference_digest=reply['artifact']['inference_digest'],
            action_contract=self.action_contract, purpose='component', synthetic_capacity=False, scan=scan)
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
            expected = self.steps + 4 * ((reply['metrics']['parity']['decisions'] + 127) // 128)
            if steps != expected:
                raise ContractError('wire3 optimizer advancement mismatch')
            self.ledger.finish_update()
        elif self.steps is not None or request['restore'] is None and steps != 0:
            raise ContractError('wire3 duplicate initialization/counter mismatch')
        self.steps, self.hardware = steps, reply['hardware']
        return reply
