"""Real CPU process ownership and durable global admission, without GPU work."""
import asyncio
import json
import os
import pickle
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_training import hpo_gpu_lease as gpu


@pytest.fixture
def global_root(tmp_path, monkeypatch):
    root = tmp_path / "global-admission"
    monkeypatch.setattr(gpu, "_GLOBAL_ROOT", root)
    return root


def context(campaign="campaign-a"):
    return gpu.JobContext(campaign, "operation", "training", "run-a", None)


def admit(campaign="campaign-a"):
    return gpu.admit_gpu_job(context(campaign), global_cuda_bytes=1024, max_workers=2)


def spawn_worker(job, *, script="pass", residency=16):
    from gf_simulator_client.structured_worker import StructuredParticipantWorker

    owner = object.__new__(StructuredParticipantWorker)
    owner._spawn(job.ledger, residency, [sys.executable, "-I", "-c", script], {})
    return owner, owner.lease, owner.process


def test_owned_cpu_worker_reap_confirms_exact_receipt_and_allows_next_campaign(global_root):
    with admit() as job:
        owner, lease, process = spawn_worker(job)
        try:
            with pytest.raises(ContractError):
                job.confirm_termination()
            process.wait(timeout=5)
            lease.reaped(process)
            receipt = job.confirm_termination()
            assert receipt.resources[0].outcome == "reaped"
            assert receipt.resources[0].pid == process.pid
            assert job.confirm_termination() == receipt
            assert owner.process is process
            with pytest.raises(ContractError):
                job.ledger.reserve_worker(1)
        finally:
            process.wait(timeout=5)
    with admit("campaign-b") as next_job:
        assert next_job.ticket != job.ticket
        next_job.confirm_termination()
    saved = json.loads((global_root / "jobs" / job.ticket / "termination.json").read_text())
    assert saved["ticket"] == receipt.ticket


def test_lock_release_and_dead_pid_or_changed_boot_are_never_cleanup_proof(global_root, monkeypatch):
    job = admit()
    ticket = job.ticket
    job.close()
    monkeypatch.setattr(gpu, "_boot_id", lambda: "different-boot")
    with pytest.raises(gpu.RecoveryRequired):
        admit("campaign-b")
    with pytest.raises(ContractError):
        job.ledger.begin_update(1)
    with pytest.raises(ContractError):
        job.confirm_termination()
    assert (global_root / "jobs" / ticket / "admission.json").exists()


def test_only_actual_owned_process_can_reap_even_if_pid_matches(global_root):
    with admit() as job:
        owner, lease, process = spawn_worker(job)
        try:
            process.wait(timeout=5)
            forged = SimpleNamespace(pid=process.pid, returncode=0, poll=lambda: 0)
            with pytest.raises(ContractError):
                lease.reaped(forged)
            owner.process = forged
            with pytest.raises(ContractError):
                lease.reaped(process)
            owner.process = process
            lease.reaped(process)
            job.confirm_termination()
        finally:
            process.wait(timeout=5)


def test_pending_creation_and_empty_foreign_ledger_cannot_confirm(global_root, monkeypatch):
    cpu_learner_exec(monkeypatch, script="pass")
    async def scenario():
        with admit() as job:
            owner = real_learner(job, global_root.parent / "pending")
            await owner.start()
            lease = owner.lease
            lease.creation_pending = True
            with pytest.raises(ContractError):
                lease.cancel_before_spawn()
            with pytest.raises(ContractError):
                job.confirm_termination()
            original = job.ledger
            with pytest.raises((AttributeError, ContractError)):
                job.ledger = gpu.ReservationLedger()
                job.confirm_termination()
            assert job.ledger is original
            lease.creation_pending = False
            await owner.process.wait()
            await owner.close(asyncio.get_running_loop().time() + 5)
            assert job.confirm_termination().resources[0].outcome == "reaped"
    asyncio.run(scenario())


def test_current_live_job_excludes_another_campaign_and_release_keeps_recovery(global_root):
    with admit() as job, pytest.raises(gpu.AdmissionBusy):
        admit("campaign-b")
    with pytest.raises(gpu.RecoveryRequired):
        admit("campaign-b")
    assert not (global_root / "jobs" / job.ticket / "termination.json").exists()


def test_explicit_restart_requires_exact_ticket_and_assertion_preserves_admission(global_root):
    job = admit()
    job.close()
    admission = (global_root / "jobs" / job.ticket / "admission.json").read_bytes()
    with pytest.raises(ContractError):
        gpu.confirm_environment_restart(expected_ticket="0" * 32, operator="operator", confirmation=gpu.RESTART_CONFIRMATION)
    with pytest.raises(ContractError):
        gpu.confirm_environment_restart(expected_ticket=job.ticket, operator="operator", confirmation="yes")
    receipt = gpu.confirm_environment_restart(expected_ticket=job.ticket, operator="operator", confirmation=gpu.RESTART_CONFIRMATION)
    assert receipt.ticket == job.ticket
    assert (global_root / "jobs" / job.ticket / "admission.json").read_bytes() == admission
    assert gpu.confirm_environment_restart(expected_ticket=job.ticket, operator="operator", confirmation=gpu.RESTART_CONFIRMATION) == receipt
    with admit("campaign-b") as next_job:
        next_job.confirm_termination()


def test_corrupt_state_never_overwritten_as_idle(global_root):
    with admit() as job:
        job.confirm_termination()
    (global_root / "current.json").write_text("{")
    with pytest.raises(ContractError):
        admit("campaign-b")
    assert (global_root / "current.json").read_text() == "{"


def test_immutable_receipt_rejects_binding_tamper(global_root):
    with admit() as job:
        receipt = job.confirm_termination()
    path = global_root / "jobs" / job.ticket / "termination.json"
    value = asdict(receipt)
    value["admission_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ContractError):
        admit("campaign-b")


def test_receipt_fsync_failure_never_reports_success_or_unseals_owner(global_root, monkeypatch):
    with admit() as job:
        original = gpu.os.fsync
        monkeypatch.setattr(gpu.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync fault")))
        with pytest.raises(OSError, match="fsync fault"):
            job.confirm_termination()
        with pytest.raises(ContractError):
            job.ledger.reserve_worker(1)
        monkeypatch.setattr(gpu.os, "fsync", original)
    with pytest.raises(gpu.RecoveryRequired):
        admit("campaign-b")


def run_independent(root, source):
    package_root = str(Path(gpu.__file__).resolve().parents[1])
    bootstrap = (f"import sys; sys.path.insert(0, {package_root!r}); from pathlib import Path; "
                 f"from gf_training import hpo_gpu_lease as g; g._GLOBAL_ROOT=Path({str(root)!r}); ")
    return subprocess.run([sys.executable, "-I", "-c", bootstrap + source], check=False, capture_output=True, text=True, timeout=10)


def test_independent_processes_share_one_namespace_and_abrupt_exit_blocks(global_root):
    with admit() as job:
        child = run_independent(global_root, "\ntry:\n g.admit_gpu_job(g.JobContext('b','op','training','run',None),global_cuda_bytes=1024,max_workers=2)\nexcept g.AdmissionBusy:\n print('busy')\n")
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "busy"
        job.confirm_termination()
    child = run_independent(global_root, "import os; j=g.admit_gpu_job(g.JobContext('crash','op','training','run',None),global_cuda_bytes=1024,max_workers=2); os._exit(7)")
    assert child.returncode == 7
    with pytest.raises(gpu.RecoveryRequired):
        admit("campaign-after-crash")


def test_async_learner_wait_and_creation_boundary_are_required(global_root, monkeypatch):
    cpu_learner_exec(monkeypatch, script="pass")
    async def scenario():
        with admit() as job:
            owner = real_learner(job, global_root.parent / "boundary")
            await owner.start()
            lease = owner.lease
            process = owner.process
            try:
                await asyncio.wait_for(process.wait(), 5)
                owner._startup_boundary.clear()
                with pytest.raises(ContractError):
                    lease.reaped(process)
                owner._startup_boundary.set()
                job.ledger.begin_update(8)
                with pytest.raises(ContractError):
                    lease.reaped(process)
                job.ledger.finish_update()
                await owner.close(asyncio.get_running_loop().time() + 5)
                assert job.confirm_termination().resources[0].kind == "learner"
            finally:
                await asyncio.wait_for(process.wait(), 5)
    asyncio.run(scenario())


def test_creation_future_must_return_exact_owned_process(global_root, monkeypatch):
    cpu_learner_exec(monkeypatch, script="pass")
    async def scenario():
        with admit() as job:
            owner = real_learner(job, global_root.parent / "exact-future")
            await owner.start()
            lease, process = owner.lease, owner.process
            original_future = owner.creation_task
            owner.creation_task = asyncio.get_running_loop().create_future()
            owner.creation_task.set_result(object())
            try:
                await asyncio.wait_for(process.wait(), 5)
                with pytest.raises(ContractError):
                    lease.reaped(process)
                owner.creation_task = original_future
                await owner.close(asyncio.get_running_loop().time() + 5)
                job.confirm_termination()
            finally:
                await asyncio.wait_for(process.wait(), 5)
    asyncio.run(scenario())


def test_multiple_tracked_learner_lifetimes_keep_distinct_history(global_root, monkeypatch):
    cpu_learner_exec(monkeypatch, missing=True)
    async def scenario():
        with admit() as job:
            for index in range(2):
                owner = real_learner(job, global_root.parent / f"missing-{index}")
                with pytest.raises(FileNotFoundError):
                    await owner.start()
                assert owner.lease.released
            receipt = job.confirm_termination()
            assert len({(resource.kind, resource.key) for resource in receipt.resources}) == 2
        with admit("next") as job:
            job.confirm_termination()
    asyncio.run(scenario())


def test_broken_symlink_pointer_is_corruption_not_absence(global_root):
    job = admit()
    job.close()
    pointer = global_root / "current.json"
    pointer.unlink()
    pointer.symlink_to(global_root / "missing")
    with pytest.raises(ContractError):
        gpu.confirm_environment_restart(expected_ticket=job.ticket, operator="operator", confirmation=gpu.RESTART_CONFIRMATION)
    assert pointer.is_symlink()


def test_foreign_thread_and_forked_or_serialized_capability_cannot_start_work(global_root):
    import concurrent.futures

    with admit() as job:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(ContractError):
            pool.submit(job.ledger.begin_update, 1).result(timeout=5)
        with pytest.raises(TypeError):
            pickle.dumps(job)
        forged = object.__new__(gpu.GpuJob)
        forged.__dict__.update(job.__dict__)
        with pytest.raises(ContractError):
            forged.confirm_termination()
        child = os.fork()
        if child == 0:
            try:
                job.ledger.reserve_worker(1)
            except ContractError:
                os._exit(0)
            os._exit(9)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        job.confirm_termination()


def test_failed_current_publication_preserves_orphan_admission(global_root, monkeypatch):
    original = gpu.os.replace
    monkeypatch.setattr(gpu.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("pointer fault")))
    with pytest.raises(OSError, match="pointer fault"):
        admit()
    monkeypatch.setattr(gpu.os, "replace", original)
    directories = tuple((global_root / "jobs").iterdir())
    assert len(directories) == 1
    with pytest.raises(gpu.RecoveryRequired):
        admit("next")
    gpu.confirm_environment_restart(expected_ticket=directories[0].name, operator="operator", confirmation=gpu.RESTART_CONFIRMATION)
    with admit("next") as job:
        job.confirm_termination()


def test_postpublication_receipt_sync_fault_requires_later_locked_resync(global_root, monkeypatch):
    with admit() as job:
        sync = gpu._sync_directory
        monkeypatch.setattr(gpu, "_sync_directory", lambda path: (_ for _ in ()).throw(OSError("directory sync fault")))
        with pytest.raises(OSError, match="directory sync fault"):
            job.confirm_termination()
        assert (global_root / "jobs" / job.ticket / "termination.json").exists()
        with pytest.raises(ContractError):
            job.ledger.reserve_learner(1, SimpleNamespace(process=None))
        monkeypatch.setattr(gpu, "_sync_directory", sync)
    # The visible exact receipt is independently validated and fully re-synced
    # under the global lock, never through a PID/time cleanup inference.
    with admit("next") as job:
        job.confirm_termination()


def test_no_pid_liveness_or_age_heuristics(global_root, monkeypatch):
    with admit() as job:
        ticket = job.ticket
    path = global_root / "jobs" / ticket / "admission.json"
    admission = json.loads(path.read_text())
    admission["admitted_unix"] = 1.
    admission["owner_pid"] = 2**30
    path.write_text(json.dumps(admission))
    pointer = json.loads((global_root / "current.json").read_text())
    from gf_common.domain import digest
    pointer["admission_sha256"] = digest(admission)
    (global_root / "current.json").write_text(json.dumps(pointer))
    monkeypatch.setattr(gpu.os, "kill", lambda *args: pytest.fail("PID liveness must not be consulted"))
    with pytest.raises(gpu.RecoveryRequired):
        admit("next")


def test_optional_free_import_does_not_import_torch_or_optuna(global_root):
    child = run_independent(global_root, "assert 'torch' not in sys.modules; assert 'optuna' not in sys.modules; print('clean')")
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "clean"


def test_hpo_larger_cuda_ceiling_reuses_joint_accounting_and_preserves_legacy_limits(global_root, monkeypatch):
    gib = 1024**3
    with pytest.raises(ContractError):
        gpu.ReservationLedger(12 * gib + 1)
    with pytest.raises(ContractError):
        gpu.ReservationLedger(max_workers=5)
    cpu_learner_exec(monkeypatch, script="pass")
    async def scenario():
        with gpu.admit_gpu_job(context(), global_cuda_bytes=16 * gib, max_workers=4) as job:
            owner = real_learner(job, global_root.parent / "larger-budget")
            owner.resident_bytes = 10 * gib  # Reservation estimate only; child is a tiny CPU script.
            await owner.start()
            worker_owner, worker, process = spawn_worker(job, residency=5 * gib)
            try:
                assert job.ledger.cuda_bytes == 15 * gib
                with pytest.raises(ContractError):
                    job.ledger.reserve_worker(2 * gib)
                assert job.ledger.cuda_bytes == 15 * gib
                process.wait(timeout=5)
                worker.reaped(process)
                with pytest.raises(ContractError):
                    job.ledger.begin_update(7 * gib)
                job.ledger.begin_update(6 * gib)
                assert job.ledger.peak_cuda_bytes == 16 * gib
                job.ledger.finish_update()
                await owner.process.wait()
                await owner.close(asyncio.get_running_loop().time() + 5)
                job.confirm_termination()
                assert worker_owner.process is process
            finally:
                process.wait(timeout=5)
    asyncio.run(scenario())
    with pytest.raises(ContractError):
        gpu.admit_gpu_job(context(), global_cuda_bytes=16 * gib, max_workers=2**63)


def test_closed_lock_descriptor_and_closed_job_forbid_further_decision_work(global_root):
    job = admit()
    owner, lease, process = spawn_worker(job)
    try:
        job._lock.file.close()
        with pytest.raises(ContractError):
            job.ledger.reserve_worker(1)
        with pytest.raises(ContractError):
            lease.reserve_decision(1)
    finally:
        process.wait(timeout=5)
        job.close()
    assert owner.process is process


def test_restart_retry_recovers_failed_original_pointer_publication(global_root, monkeypatch):
    original = gpu.os.replace
    monkeypatch.setattr(gpu.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("pointer fault")))
    with pytest.raises(OSError):
        admit()
    ticket = next((global_root / "jobs").iterdir()).name
    with pytest.raises(OSError):
        gpu.confirm_environment_restart(expected_ticket=ticket, operator="operator", confirmation=gpu.RESTART_CONFIRMATION)
    monkeypatch.setattr(gpu.os, "replace", original)
    gpu.confirm_environment_restart(expected_ticket=ticket, operator="operator", confirmation=gpu.RESTART_CONFIRMATION)
    with admit("next") as job:
        job.confirm_termination()


def test_live_worker_reap_and_cleanup_failure_leave_no_receipt(global_root):
    with admit() as job:
        owner, lease, process = spawn_worker(job, script="import time; time.sleep(30)")
        try:
            with pytest.raises(ContractError):
                lease.reaped(process)
            with pytest.raises(ContractError):
                job.confirm_termination()
            assert not (global_root / "jobs" / job.ticket / "termination.json").exists()
        finally:
            process.terminate()
            process.wait(timeout=5)
            lease.reaped(process)
            job.confirm_termination()
        assert owner.process is process


def test_existing_synchronous_worker_spawn_failure_proves_no_child(global_root):
    from gf_simulator_client.structured_worker import StructuredParticipantWorker

    with admit() as job:
        owner = object.__new__(StructuredParticipantWorker)
        with pytest.raises(FileNotFoundError):
            owner._spawn(job.ledger, 16, ["/definitely-missing-hpo-test-worker"], {})
        assert owner.process is None
        assert owner.lease.released
        receipt = job.confirm_termination()
        assert receipt.resources[0].outcome == "never_started"


def test_existing_worker_creation_and_close_keep_same_thread_resource_proof(global_root):
    from gf_simulator_client.structured_worker import StructuredParticipantWorker

    async def scenario():
        with admit() as job:
            owner = object.__new__(StructuredParticipantWorker)
            owner.closed = False
            owner._close_task = None
            owner.health = {}
            owner.io_thread = None
            owner._spawn(job.ledger, 16, [sys.executable, "-I", "-c", "pass"], {})
            process = owner.process
            try:
                await owner.close(timeout=2)
                assert process.returncode is not None
                assert owner.lease.released
                assert job.confirm_termination().resources[0].pid == process.pid
            finally:
                process.wait(timeout=5)
    asyncio.run(scenario())


def test_unreadable_receipt_and_admission_failure_do_not_reset_state(global_root, monkeypatch):
    with admit() as job:
        job.confirm_termination()
    original = gpu._read
    def unreadable(path):
        if path.name == "termination.json":
            raise PermissionError("injected unreadable receipt")
        return original(path)
    monkeypatch.setattr(gpu, "_read", unreadable)
    with pytest.raises(PermissionError):
        admit("next")
    assert len(tuple((global_root / "jobs").iterdir())) == 1


def test_symlink_record_and_unexpected_storage_entries_block(global_root):
    with admit() as job:
        job.confirm_termination()
    record = global_root / "jobs" / job.ticket / "termination.json"
    saved = record.read_text()
    record.unlink()
    destination = global_root.parent / "foreign.json"
    destination.write_text(saved)
    record.symlink_to(destination)
    with pytest.raises(ContractError):
        admit("next")


def test_diagnostic_wallclock_change_cannot_invalidate_actual_cleanup(global_root, monkeypatch):
    with admit() as job:
        monkeypatch.setattr(gpu.time, "time", lambda: 1.)
        receipt = job.confirm_termination()
        assert receipt.confirmed_unix == 1.
    with admit("next") as job:
        job.confirm_termination()


def test_admission_save_fault_preserves_incomplete_job_and_blocks(global_root, monkeypatch):
    original = gpu.os.fsync
    original_write = gpu._write
    def fail_admission(path, value, *, immutable):
        if path.name == "admission.json":
            monkeypatch.setattr(gpu.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("admission sync fault")))
        return original_write(path, value, immutable=immutable)
    monkeypatch.setattr(gpu, "_write", fail_admission)
    with pytest.raises(OSError, match="admission sync fault"):
        admit()
    monkeypatch.setattr(gpu.os, "fsync", original)
    monkeypatch.setattr(gpu, "_write", original_write)
    assert len(tuple((global_root / "jobs").iterdir())) == 1
    with pytest.raises(ContractError):
        admit("next")


def test_conflicting_or_unexpected_records_never_allow_another_admission(global_root):
    with admit() as job:
        job.confirm_termination()
    directory = global_root / "jobs" / job.ticket
    (directory / "restart.json").write_text("{}")
    with pytest.raises(ContractError):
        admit("next")
    (directory / "restart.json").unlink()
    (directory / "unknown.json").write_text("{}")
    with pytest.raises(ContractError):
        admit("next")


def test_environment_home_override_cannot_split_user_namespace(global_root):
    package_root = str(Path(gpu.__file__).resolve().parents[1])
    source = (f"import sys,os,pwd; from pathlib import Path; os.environ['HOME']={str(global_root)!r}; "
              f"sys.path.insert(0,{package_root!r}); from gf_training import hpo_gpu_lease as g; "
              "assert g._GLOBAL_ROOT == Path(pwd.getpwuid(os.getuid()).pw_dir)/'.local/state/godfieldbots/hpo-gpu-admission-v1'")
    child = subprocess.run([sys.executable, "-I", "-c", source], check=False, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr


def test_second_object_cannot_reissue_same_held_ticket_with_an_empty_ledger(global_root):
    with admit() as job:
        with pytest.raises(ContractError):
            gpu.GpuJob(gpu._ISSUER, job._lock, job._admission, 1024, 2)
        with pytest.raises(ContractError):
            gpu.GpuJob(gpu._ISSUER, job._lock, json.loads(json.dumps(job._admission)), 1024, 2)
        job.confirm_termination()


def test_ancestor_sync_uncertainty_is_retried_even_when_directory_now_exists(global_root, monkeypatch):
    original = gpu._sync_directory
    uncertain = global_root.parent
    attempts = []
    def fail_uncertain_parent(path):
        attempts.append(path)
        if path == uncertain:
            raise OSError("uncertain ancestor sync")
        original(path)
    monkeypatch.setattr(gpu, "_sync_directory", fail_uncertain_parent)
    with pytest.raises(OSError, match="uncertain ancestor sync"):
        admit()
    assert global_root.is_dir()
    assert not (global_root / "jobs").exists()
    with pytest.raises(OSError, match="uncertain ancestor sync"):
        admit("retry")
    assert attempts.count(uncertain) == 2
    monkeypatch.setattr(gpu, "_sync_directory", original)
    with admit("retry-after-real-sync") as job:
        job.confirm_termination()


def test_every_existing_ancestor_is_resynced_before_admission(global_root, monkeypatch):
    with admit() as job:
        job.confirm_termination()
    original = gpu._sync_directory
    calls = []
    def observe_sync(path):
        calls.append(path)
        original(path)
    monkeypatch.setattr(gpu, "_sync_directory", observe_sync)
    with admit("retry") as job:
        assert all(ancestor in calls for ancestor in (global_root, *global_root.parents))
        job.confirm_termination()


@pytest.mark.parametrize("kind", ["worker", "learner"])
def test_absent_creation_proof_cannot_mint_never_started_receipt(global_root, kind):
    with admit() as job:
        owner = SimpleNamespace(process=None, creation_task=None)
        with pytest.raises(ContractError):
            if kind == "worker":
                lease = job.ledger.reserve_worker(16)
                lease._cleanup_owner = owner
            else:
                lease = job.ledger.reserve_learner(16, owner)
            lease.cancel_before_spawn()
            job.confirm_termination()
    with pytest.raises(gpu.RecoveryRequired):
        admit("must-recover")


def real_learner(job, directory):
    from gf_training.ppo_run import LearningProcess
    from gf_training.structured_learning_process import StructuredLearningProcess

    directory.mkdir()
    owner = object.__new__(StructuredLearningProcess)
    LearningProcess.__init__(owner, directory, 4096)
    owner.ledger = job.ledger
    owner.resident_bytes = 16
    owner.cleanup_deadline = asyncio.get_running_loop().time() + 10
    owner.lease = None
    owner.visibility = None
    owner._terminal_closed = False
    owner._startup_boundary = asyncio.Event()
    owner._close_request = LearningProcess._close_request.__get__(owner)
    owner._validate_close = LearningProcess._validate_close.__get__(owner)
    owner.request = LearningProcess.request.__get__(owner)
    return owner


def cpu_learner_exec(monkeypatch, *, script="import sys,json;sys.stdin.readline();print(json.dumps({'closed':True}),flush=True)", missing=False):
    original = asyncio.create_subprocess_exec
    processes = []
    async def create_cpu(*args, **kwargs):
        command = ["/definitely-missing-hpo-learner"] if missing else [sys.executable, "-I", "-c", script]
        process = await original(*command, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_cpu)
    return processes


@pytest.mark.parametrize("replace_proof", ["future", "event", "both"])
def test_replaced_pending_future_and_event_cannot_erase_original_creation(global_root, monkeypatch, replace_proof):
    async def scenario():
        original_exec = asyncio.create_subprocess_exec
        gate = asyncio.Event()
        async def delayed_cpu(*args, **kwargs):
            await gate.wait()
            return await original_exec(sys.executable, "-I", "-c",
                "import sys,json;sys.stdin.readline();print(json.dumps({'closed':True}),flush=True)", **kwargs)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_cpu)
        with admit() as job:
            owner = real_learner(job, global_root.parent / "learner-replaced")
            start = asyncio.create_task(owner.start())
            while owner.creation_task is None:
                await asyncio.sleep(0)
            original_future, original_event = owner.creation_task, owner._startup_boundary
            replacement = asyncio.get_running_loop().create_future()
            replacement.set_result(None)
            if replace_proof in ("future", "both"):
                owner.creation_task = replacement
            if replace_proof in ("event", "both"):
                owner._startup_boundary = asyncio.Event()
            owner._startup_boundary.set()
            owner._creation_no_child = True
            owner.lease.creation_pending = False
            try:
                with pytest.raises(ContractError):
                    owner.lease.cancel_before_spawn()
                with pytest.raises(ContractError):
                    job.confirm_termination()
            finally:
                owner.creation_task, owner._startup_boundary = original_future, original_event
                original_event.clear()
                owner._creation_no_child = False
                owner.lease.creation_pending = True
                gate.set()
                await asyncio.wait_for(start, 5)
                await owner.close(asyncio.get_running_loop().time() + 5)
            assert owner.process.returncode == 0
            job.confirm_termination()
    asyncio.run(scenario())


def test_actual_learner_cancellation_waits_for_original_delayed_creation_and_reaps(global_root, monkeypatch):
    async def scenario():
        original_exec = asyncio.create_subprocess_exec
        gate = asyncio.Event()
        processes = []
        async def delayed_cpu(*args, **kwargs):
            await gate.wait()
            process = await original_exec(sys.executable, "-I", "-c",
                "import sys,json;sys.stdin.readline();print(json.dumps({'closed':True}),flush=True)", **kwargs)
            processes.append(process)
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_cpu)
        with admit() as job:
            owner = real_learner(job, global_root.parent / "cancelled-learner")
            start = asyncio.create_task(owner.start())
            while owner.creation_task is None:
                await asyncio.sleep(0)
            original = owner.creation_task
            start.cancel()
            await asyncio.sleep(0)
            assert not original.cancelled()
            assert not start.done()
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(start, 5)
            assert owner.creation_task is original
            assert processes[0] is owner.process
            assert owner.process.returncode == 0
            assert owner.lease.released
            job.confirm_termination()
    asyncio.run(scenario())


@pytest.mark.parametrize("after_forward", [False, True])
def test_learner_observer_failure_still_adopts_and_reaps_local_future(global_root, monkeypatch, after_forward):
    processes = cpu_learner_exec(monkeypatch)
    async def scenario():
        with admit() as job:
            owner = real_learner(job, global_root.parent / "observer-learner")
            forward = owner._creation_task_created
            def failing_observer(creation):
                if after_forward:
                    forward(creation)
                raise RuntimeError("injected creation observer")
            owner._creation_task_created = failing_observer
            with pytest.raises((ContractError, RuntimeError)):
                await owner.start()
            assert len(processes) == 1
            assert processes[0] is owner.process
            assert owner.process.returncode == 0
            assert owner.creation_task.done()
            assert owner.drainer.done()
            with pytest.raises(ContractError):
                job.confirm_termination()
        with pytest.raises(gpu.RecoveryRequired):
            admit("needs-recovery")
    asyncio.run(scenario())


def test_worker_finish_observer_failure_keeps_local_returned_process_for_cleanup(global_root, monkeypatch):
    from gf_simulator_client.structured_worker import StructuredParticipantWorker

    original = gpu._Worker._creation_finished
    def failing_finish(self, owner, boundary, process, error):
        original(self, owner, boundary, process, error)
        raise RuntimeError("injected worker observer")
    monkeypatch.setattr(gpu._Worker, "_creation_finished", failing_finish)
    with admit() as job:
        owner = object.__new__(StructuredParticipantWorker)
        with pytest.raises(RuntimeError, match="injected worker observer"):
            owner._spawn(job.ledger, 16, [sys.executable, "-I", "-c", "import time;time.sleep(30)"], {})
        assert owner.process.returncode is not None
        with pytest.raises(ContractError):
            job.confirm_termination()
    with pytest.raises(gpu.RecoveryRequired):
        admit("needs-recovery")


def test_actual_task_creation_failure_closes_unstarted_coroutine_and_proves_no_child(global_root, monkeypatch):
    from gf_training.ppo_run import LearningProcess

    original = asyncio.create_task
    failure = RuntimeError("actual task factory failure")
    def fail_creator_task(coroutine, *args, **kwargs):
        if coroutine.cr_code is LearningProcess._create_process.__code__:
            raise failure
        return original(coroutine, *args, **kwargs)
    monkeypatch.setattr(asyncio, "create_task", fail_creator_task)
    async def scenario():
        with admit() as job:
            owner = real_learner(job, global_root.parent / "task-failed")
            with pytest.raises(RuntimeError) as error:
                await owner.start()
            assert error.value is failure
            assert owner.creation_task is None
            assert owner.process is None
            assert owner.lease.released
            assert job.confirm_termination().resources[0].outcome == "never_started"
    asyncio.run(scenario())


def test_primary_learner_creation_exception_survives_observer_exception(global_root, monkeypatch):
    cpu_learner_exec(monkeypatch, missing=True)
    async def scenario():
        with admit() as job:
            owner = real_learner(job, global_root.parent / "primary-creation")
            def fail_observer(creation):
                raise RuntimeError("secondary observer failure")
            owner._creation_task_created = fail_observer
            with pytest.raises(FileNotFoundError) as failure:
                await owner.start()
            assert any("creation task observation" in note for note in failure.value.__notes__)
            assert owner.creation_task.exception() is failure.value
            assert owner.process is None
            with pytest.raises(ContractError):
                job.confirm_termination()
    asyncio.run(scenario())


def test_primary_worker_creation_exception_survives_finish_observer_exception(global_root, monkeypatch):
    from gf_simulator_client.structured_worker import StructuredParticipantWorker

    original = gpu._Worker._creation_finished
    def fail_finish(self, owner, boundary, process, error):
        original(self, owner, boundary, process, error)
        raise RuntimeError("secondary worker observer")
    monkeypatch.setattr(gpu._Worker, "_creation_finished", fail_finish)
    with admit() as job:
        owner = object.__new__(StructuredParticipantWorker)
        with pytest.raises(FileNotFoundError) as failure:
            owner._spawn(job.ledger, 16, ["/definitely-missing-hpo-worker"], {})
        assert any("worker creation observation" in note for note in failure.value.__notes__)
        assert owner.process is None
        with pytest.raises(ContractError):
            job.confirm_termination()


def test_actual_card_instance_worker_constructor_initialization_and_close_observe_creation(global_root, monkeypatch):
    from gf_simulator_client import card_instance_worker as workers
    from tests.test_card_instance_baseline_runtime import ACTIONS

    package_root = str(Path(gpu.__file__).resolve().parents[1])
    bootstrap = (f"import sys,runpy;sys.path.insert(0,{package_root!r});"
                 "runpy.run_module('gf_simulator_client.card_instance_bootstrap',run_name='__main__')")
    wrapper = global_root.parent / "worktree-python"
    wrapper.write_text(f"#!{sys.executable}\nimport os,sys\nos.execv(sys.executable,[sys.executable,'-I','-c',{bootstrap!r}])\n")
    wrapper.chmod(0o700)
    # Rewrite only this owner's executable. Popen itself, process identity and
    # all lifecycle observations remain the actual stdlib implementation.
    monkeypatch.setattr(workers, "sys", SimpleNamespace(executable=str(wrapper)))
    async def scenario():
        with admit() as job:
            worker = workers.CardInstanceParticipantWorker("random", 17, ACTIONS, ledger=job.ledger)
            try:
                await worker.initialize()
            finally:
                await worker.close()
            assert worker.process.returncode is not None
            assert worker.lease.released
            assert job.confirm_termination().resources[0].pid == worker.process.pid
    asyncio.run(scenario())
