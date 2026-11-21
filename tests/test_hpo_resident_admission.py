"""Homogeneous resident cache additions above unchanged legacy process allowances."""

from dataclasses import asdict, replace

import pytest

from gf_cli.hpo_config import HPOOperatingConfig
from gf_common.domain import ContractError
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.hpo_config import model_defaults_v4
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_runtime import hpo_resident_training_admission, hpo_training_admission

RUNTIME = {'mode': 'resident-worker-v1', 'cache_entries': 6, 'cache_weight_bytes': 128 * 1024**2}


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_cached_weight_and_nonpersistent_position_reservations_match_real_network(kind):
    pytest.importorskip('torch')
    from gf_models.structured_network import make_network
    from tests.test_hpo_network import small_model4

    model = small_model4(kind, depth=3)
    binding = InferenceBindingV2(device='cpu', max_candidates=64, candidate_chunk=2)
    network = make_network(model, 17, binding, layout=LAYOUT)
    weights = sum(tensor.numel() * tensor.element_size() for tensor in network.state_dict().values())
    positions = network.positions.numel() * network.positions.element_size()
    assert 'positions' not in network.state_dict()
    operating = asdict(HPOOperatingConfig())
    base = hpo_training_admission(model, binding, 1, operating, layout=LAYOUT)
    admitted = hpo_resident_training_admission(model, binding, 1, operating, RUNTIME, layout=LAYOUT)
    assert admitted['network_weight_bytes'] == weights
    assert admitted['network_position_bytes'] == positions
    assert admitted['cached_networks_per_worker'] == 6
    assert admitted['cache_weight_bytes'] == 4 * 5 * weights
    assert admitted['cache_position_bytes'] == 4 * 6 * positions
    assert admitted['learner_bytes'] == base['learner_bytes']


def large_model():
    model = model_defaults_v4('transformer')
    return replace(model,
        temporal=replace(model.temporal, hidden_size=1024, num_layers=4, feedforward_size=1024),
        card=replace(model.card, output_size=512),
        heads=replace(model.heads, actor_hidden_sizes=(1024, 1024)),
        resources=replace(model.resources, max_parameters=100_000_000))


def test_a_single_network_larger_than_cache_refuses_before_runtime_import():
    model = large_model()
    binding = InferenceBindingV2(max_candidates=1, candidate_chunk=1)
    operating = asdict(HPOOperatingConfig(managed_ram_bytes=128 * 1024**3,
                                        managed_cuda_bytes=128 * 1024**3))
    with pytest.raises(ContractError, match='cache') as caught:
        hpo_resident_training_admission(model, binding, 1, operating, RUNTIME, layout=LAYOUT)
    assert all(token in str(caught.value) for token in (
        'resource=cache-weight', 'phase=resident-load', 'required=163836936', 'limit=134217728'))


def test_one_cache_slot_does_not_charge_active_weights_twice_but_keeps_positions():
    model = large_model()
    model = replace(model, temporal=replace(model.temporal, num_layers=1))
    binding = InferenceBindingV2(max_candidates=1, candidate_chunk=1)
    operating = asdict(HPOOperatingConfig(managed_ram_bytes=128 * 1024**3,
                                        managed_cuda_bytes=128 * 1024**3))
    base = hpo_training_admission(model, binding, 1, operating, layout=LAYOUT)
    admitted = hpo_resident_training_admission(model, binding, 1, operating, RUNTIME, layout=LAYOUT)
    assert admitted['network_weight_bytes'] == 88216584
    assert admitted['cached_networks_per_worker'] == 1
    assert admitted['cache_weight_bytes'] == 0
    assert admitted['cache_position_bytes'] == 4 * 4194304
    assert admitted['worker_host_bytes'] == base['worker_bytes'] + 4 * 4194304
    assert admitted['worker_bytes'] == 0  # CPU actor; host retention remains full.


@pytest.mark.parametrize('kind,weights', [('mlp', 4583688), ('gru', 4979976),
                                          ('transformer', 5643528)])
def test_all_phases_keep_cache_but_only_collection_has_concurrent_begin(kind, weights):
    model = model_defaults_v4(kind)
    binding = InferenceBindingV2()
    operating = asdict(HPOOperatingConfig(game_concurrency=3, max_participant_workers=6))
    base = hpo_training_admission(model, binding, 32, operating, layout=LAYOUT)
    admitted = hpo_resident_training_admission(model, binding, 32, operating, RUNTIME, layout=LAYOUT)
    extra = 6 * (5 * weights + 6 * 524288)
    resident = base['worker_bytes'] + extra
    fixed = operating['parent_budget_bytes'] + base['learner_bytes']
    assert admitted['worker_host_bytes'] == resident
    assert admitted['worker_bytes'] == 0
    assert admitted['ram_phases'] == {
        'collection': fixed + resident + operating['collector_budget_bytes'] + 6 * 134217728,
        'update': base['ram_phases']['update'] + resident,
        'save': base['ram_phases']['save'] + resident,
        'sync': fixed + resident + 134217728,
    }
    assert admitted['cuda_phases'] == {
        'collection': base['learner_bytes'] + 6 * 134217728,
        'update': base['learner_bytes'],
        'save': base['learner_bytes'],
        'sync': base['learner_bytes'] + 134217728,
    }
    assert admitted['begin_staging_bytes'] == 6 * 134217728
    assert admitted['staging_bytes'] == 134217728
    assert admitted['peak_cuda_bytes'] == admitted['cuda_phases']['collection']
    for resource, phases, setting in [('ram', 'ram_phases', 'managed_ram_bytes'),
                                      ('cuda', 'cuda_phases', 'managed_cuda_bytes')]:
        limit = max(admitted[phases].values()) - 1
        with pytest.raises(ContractError) as caught:
            hpo_resident_training_admission(model, binding, 32, {**operating, setting: limit},
                                            RUNTIME, layout=LAYOUT)
        assert all(token in str(caught.value) for token in (
            f'resource={resource}', 'phase=', f'required={limit + 1}', f'limit={limit}',
            'legacy_workers=', 'extra_cache=', 'begin_staging=', 'sync_staging='))


def test_concurrent_begin_scratch_retires_after_both_original_validated_replies(tmp_path, monkeypatch):
    import asyncio
    import threading

    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_resident_episode import ACTIONS

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    entered = {1: threading.Event(), 2: threading.Event()}
    release = threading.Event()
    read = ResidentPhysicalWorker._read_reply

    def hold(self, source):
        response = read(self, source)
        if response.get('request', {}).get('command') == 'BEGIN':
            entered[self.viewer].set()
            assert release.wait(10)
        return response

    monkeypatch.setattr(ResidentPhysicalWorker, '_read_reply', hold)

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('campaign', 'operation', 'training', 'run-a', None),
                              global_cuda_bytes=1024, max_workers=2) as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b' * 32)
            pool = ResidentParticipantPool(custody, resident_bytes=16, begin_import_bytes=32, deadline=1e12)
            pool.admit(CollectionPlan((), {'random': RuntimePolicyBinding('random')}, tmp_path,
                {'parent_run_id': 'run-a', 'collection_generation': 0, 'batch_id': 'batch-000000'}, 1e12, 1e12))
            workers = [pool.worker(viewer, 'random', 17, ACTIONS, ledger=job.ledger) for viewer in (1, 2)]
            tasks = [asyncio.create_task(worker.initialize()) for worker in workers]
            try:
                async with asyncio.timeout(10):
                    while not all(event.is_set() for event in entered.values()):
                        await asyncio.sleep(.001)
                assert job.ledger.cuda_bytes == job.ledger.peak_cuda_bytes == 96
                assert len(job.ledger._resident_begin_imports) == 2
                for worker in workers:
                    assert worker.lease._resident_begin_epoch == (1, 1)
                    with pytest.raises(ContractError):
                        custody.episode_request(worker.lease, 'SELECT')
                    with pytest.raises(ContractError):
                        custody.request(worker.lease, 'PARK', checkpoint_sha256='c' * 64)
                release.set()
                await asyncio.gather(*tasks)
                assert job.ledger.cuda_bytes == 32
                assert not job.ledger._resident_begin_imports
                for worker in workers:
                    worker.mark_completed()
                    await worker.close()
                pool.retire_collection().require(job.ledger)
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                await pool.close()
            assert job.ledger.cuda_bytes == 0
            job.confirm_termination()

    asyncio.run(run())


def test_begin_capacity_refusal_happens_before_begin_framing(tmp_path, monkeypatch):
    import asyncio

    import gf_simulator_client.resident_pool as transport
    from gf_common.domain import RuntimePolicyBinding
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_resident_episode import ACTIONS

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    commands = []
    write = transport.write_frame

    def observe(stream, value):
        commands.append(value['request']['command'])
        return write(stream, value)

    monkeypatch.setattr(transport, 'write_frame', observe)

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('campaign', 'operation', 'training', 'run-a', None),
                              global_cuda_bytes=48, max_workers=2) as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b' * 32)
            pool = transport.ResidentParticipantPool(custody, resident_bytes=16, begin_import_bytes=33, deadline=1e12)
            pool.admit(CollectionPlan((), {'random': RuntimePolicyBinding('random')}, tmp_path,
                {'parent_run_id': 'run-a', 'collection_generation': 0, 'batch_id': 'batch-000000'}, 1e12, 1e12))
            logical = pool.worker(1, 'random', 17, ACTIONS, ledger=job.ledger)
            try:
                with pytest.raises(ContractError) as caught:
                    await logical.initialize()
                assert all(token in str(caught.value) for token in (
                    'resource=cuda', 'phase=collection', 'required=49', 'limit=48', 'request=33'))
                assert 'BEGIN' not in commands
                assert not job.ledger._resident_begin_imports
            finally:
                await pool.close()
            job.confirm_termination()

    asyncio.run(run())


@pytest.mark.parametrize('failure', ['foreign-ready', 'cancel'])
def test_failed_begin_keeps_charge_across_actual_process_reap(tmp_path, monkeypatch, failure):
    import asyncio
    import threading

    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_resident_episode import ACTIONS

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    read, reaped = ResidentPhysicalWorker._read_reply, gpu._Worker.reaped
    observed = []
    entered, release = threading.Event(), threading.Event()

    def foreign(self, source):
        response = read(self, source)
        if response.get('request', {}).get('command') == 'BEGIN':
            if failure == 'foreign-ready':
                response['payload']['policy_version'] = 'foreign'
            else:
                entered.set()
                assert release.wait(10)
        return response

    def observe_reap(self, process):
        observed.append(('before', self.ledger.cuda_bytes))
        reaped(self, process)
        observed.append(('after', self.ledger.cuda_bytes))
        assert self._resident_begin_epoch == (1, 1)
        with pytest.raises(ContractError):
            self.ledger._job.confirm_termination()
        release.set()

    monkeypatch.setattr(ResidentPhysicalWorker, '_read_reply', foreign)
    monkeypatch.setattr(gpu._Worker, 'reaped', observe_reap)

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('campaign', 'operation', 'training', 'run-a', None),
                              global_cuda_bytes=1024, max_workers=2) as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b' * 32)
            pool = ResidentParticipantPool(custody, resident_bytes=16, begin_import_bytes=32, deadline=1e12)
            pool.admit(CollectionPlan((), {'random': RuntimePolicyBinding('random')}, tmp_path,
                {'parent_run_id': 'run-a', 'collection_generation': 0, 'batch_id': 'batch-000000'}, 1e12, 1e12))
            logical = pool.worker(1, 'random', 17, ACTIONS, ledger=job.ledger)
            try:
                if failure == 'foreign-ready':
                    with pytest.raises(ContractError, match='policy identity'):
                        await logical.initialize()
                else:
                    task = asyncio.create_task(logical.initialize())
                    async with asyncio.timeout(10):
                        while not entered.is_set():
                            await asyncio.sleep(.001)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert observed == [('before', 48), ('after', 32)]
                assert not job.ledger._resident_begin_imports
                assert job.ledger.cuda_bytes == 0
            finally:
                release.set()
                await pool.close()
            job.confirm_termination()

    asyncio.run(run())


def test_reaped_begin_import_remains_charged_until_original_uncertain_setup_retires(tmp_path, monkeypatch):
    import asyncio
    import time

    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_resident_episode import ACTIONS

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('campaign', 'operation', 'training', 'run-a', None),
                              global_cuda_bytes=1024, max_workers=2) as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b' * 32)
            pool = ResidentParticipantPool(custody, resident_bytes=16, begin_import_bytes=32, deadline=1e12)
            pool.admit(CollectionPlan((), {'random': RuntimePolicyBinding('random')}, tmp_path,
                {'parent_run_id': 'run-a', 'collection_generation': 0, 'batch_id': 'batch-000000'}, 1e12, 1e12))
            logical = pool.worker(1, 'random', 17, ACTIONS, ledger=job.ledger)
            owner = logical.owner
            await owner.bootstrap(timeout=10)
            submit, shutdown = owner.executor.submit, owner.executor.shutdown

            def lost_future(*args, **kwargs):
                submit(*args, **kwargs)
                raise RuntimeError('original submitted BEGIN Future return lost')

            def uncertain_shutdown(*args, **kwargs):
                raise OSError('original executor shutdown uncertain')

            monkeypatch.setattr(owner.executor, 'submit', lost_future)
            monkeypatch.setattr(owner.executor, 'shutdown', uncertain_shutdown)
            with pytest.raises(RuntimeError, match='Future return lost'):
                await logical.initialize()
            assert owner.lease.released and owner.process.poll() is not None
            assert not job.ledger.workers and job.ledger.cuda_bytes == 32
            assert owner._setups and owner.lease._resident_begin_epoch == (1, 1)
            with pytest.raises(ContractError):
                custody.retire_failed_begin_import(owner.lease)
            with pytest.raises(ContractError):
                job.confirm_termination()
            monkeypatch.setattr(owner.executor, 'shutdown', shutdown)
            await owner.retire_setups(time.monotonic() + 2)
            custody.retire_failed_begin_import(owner.lease)
            assert job.ledger.cuda_bytes == 0 and not job.ledger._resident_begin_imports
            with pytest.raises(OSError, match='shutdown uncertain'):
                await pool.close()
            job.confirm_termination()

    asyncio.run(run())


@pytest.mark.parametrize('resource', ['ram', 'cuda'])
def test_actual_fresh_campaign_base_refusal_keeps_phase_and_numbers(tmp_path, resource):
    from gf_cli.hpo_config import campaign_config, joint_resource_admission
    from tests.test_hpo_config import campaign_raw

    campaign = campaign_config({**campaign_raw(),
        'operating': {'game_concurrency': 8, 'max_participant_workers': 16,
            'managed_ram_bytes': 128 * 1024**3, 'managed_cuda_bytes': 128 * 1024**3}}, origin=tmp_path)
    config = campaign.config
    base = hpo_training_admission(config.model, config.inference, config.training.tbptt_length,
                                  asdict(config.operating), layout=LAYOUT)
    required = base[f'peak_{resource}_bytes']
    refused = replace(config, operating=replace(config.operating, **{f'managed_{resource}_bytes': required - 1}))
    with pytest.raises(ContractError) as caught:
        joint_resource_admission(refused, layout=LAYOUT)
    assert str(caught.value).startswith(f'joint HPO {resource.upper()} demand exceeds managed budget')
    assert all(token in str(caught.value) for token in (
        f'resource={resource}', 'phase=', f'required={required}', f'limit={required - 1}',
        'learner=', 'workers='))
    if resource == 'ram':
        assert all(token in str(caught.value) for token in (
            'parent=', 'collector=', 'source=', 'numeric=', 'save='))


def test_actual_training_leases_and_serial_sync_use_the_derived_admission(tmp_path, monkeypatch):
    from gf_simulator_client.resident_pool import ResidentPhysicalWorker
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
    from tests.test_resident_training import run_actual_source_case

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    initialize = ResidentTrainingRuntime.__init__
    read = ResidentPhysicalWorker._read_reply
    envelopes = {}
    commands = []

    def observe_runtime(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        envelopes[id(self.pool)] = self.envelope

    def observe_reply(self, stream):
        reply = read(self, stream)
        envelope = envelopes[id(self.pool)]
        assert self.lease.resident_bytes == envelope['worker_bytes'] // envelope['workers']
        command = reply.get('request', {}).get('command')
        if command == 'BEGIN':
            assert self.lease._resident_begin_bytes == 134217728
        elif reply.get('command') == 'SYNC':
            command = 'SYNC'
            assert self.pool.ledger._resident_temporary == envelope['staging_bytes'] == 134217728
            assert not self.pool.ledger._resident_begin_imports
        if command in ('BEGIN', 'SYNC'):
            commands.append(command)
        return reply

    monkeypatch.setattr(ResidentTrainingRuntime, '__init__', observe_runtime)
    monkeypatch.setattr(ResidentPhysicalWorker, '_read_reply', observe_reply)
    run_actual_source_case(tmp_path / 'gpu', tmp_path, monkeypatch, packed=False)
    assert commands.count('BEGIN') == commands.count('SYNC') == 4


def test_failed_fd_completion_keeps_begin_charge_until_original_transfer_reap(tmp_path, monkeypatch):
    import asyncio
    import time

    from gf_common.domain import RuntimePolicyBinding
    from gf_common.ram_bank import RamBankChannel, TransferLease
    from gf_models.structured_binding import input_binding_game_v3, safe_options_game_v4
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    from tests.test_resident_episode import initial

    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    learner = learner_for()
    snapshot = learner.snapshot_owned(budget=budget())
    snapshot.save(tmp_path / 'saved')
    checkpoint = snapshot.save_ack.files['inference.json']
    options = safe_options_game_v4({'version': '4', 'checkpoint': str(tmp_path / 'absent'),
        'expected_manifest_digest': checkpoint, 'model': asdict(learner.model), 'inference': asdict(learner.binding),
        'cuda_visible_devices': '', 'action_config_hash': learner.action_config_hash,
        **input_binding_game_v3(ACTIONS.action_contract, RULES)}, actions=ACTIONS, rules=RULES)
    binding = RuntimePolicyBinding('mlp', options)
    accept, release = RamBankChannel.accept_completion, TransferLease.release_after_reap
    released = []

    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('campaign', 'operation', 'training', 'run-a', None),
                              global_cuda_bytes=1024, max_workers=2) as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b' * 32)
            pool = ResidentParticipantPool(custody, resident_bytes=16, begin_import_bytes=32,
                deadline=time.monotonic() + 30, owned_snapshots=True)
            pool.bind_owned_snapshot(snapshot.actor_capability)
            pool.admit(CollectionPlan((), {'model': binding}, tmp_path,
                {'parent_run_id': 'run-a', 'collection_generation': 0, 'batch_id': 'batch-000000'}, 1e12, 1e12))
            owner = ResidentPhysicalWorker(pool, 1)
            pool.owners.append(owner)

            def lost_completion(channel, transfer):
                if channel is owner.owned_channel:
                    raise OSError('original FD completion uncertain')
                return accept(channel, transfer)

            def observe_release(transfer, process):
                if transfer._channel is owner.owned_channel:
                    assert owner.lease.released and job.ledger.cuda_bytes == 32
                    with pytest.raises(ContractError, match='FD transfers'):
                        custody.retire_failed_begin_import(owner.lease)
                    released.append(transfer)
                release(transfer, process)

            monkeypatch.setattr(RamBankChannel, 'accept_completion', lost_completion)
            monkeypatch.setattr(TransferLease, 'release_after_reap', observe_release)
            try:
                await owner.bootstrap(timeout=10)
                custody.begin_admitted_lease(pool.admission, owner.lease, binding=binding)
                request = custody.episode_request(owner.lease, 'BEGIN')
                custody.reserve_begin_import(owner.lease, request, 32)
                with pytest.raises(OSError, match='FD completion uncertain'):
                    await owner.exchange({'request': request, 'payload': initial('mlp', options=options)}, timeout=10)
                assert released and all(transfer.released for transfer in released)
                assert not owner.owned_channel.outstanding
                assert job.ledger.cuda_bytes == 0 and not job.ledger._resident_begin_imports
            finally:
                await pool.close()
                pool.unbind_owned_snapshot(snapshot.actor_capability)
            job.confirm_termination()

    try:
        asyncio.run(run())
    finally:
        snapshot.close()
