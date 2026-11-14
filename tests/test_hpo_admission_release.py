"""Explicit operational admission release; CPU-only isolated namespace, no GPU.

Release resolves admission only and cannot certify resource or data completion.
"""
import json
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError
from gf_training import hpo_gpu_lease as gpu


@pytest.fixture
def namespace(tmp_path, monkeypatch):
    root = tmp_path / 'admission'
    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', root)
    monkeypatch.setattr(gpu, '_boot_id', lambda: 'component-current-boot')
    return root


def admit(name='component'):
    return gpu.admit_gpu_job(gpu.JobContext(name, name, 'training', name, None),
        global_cuda_bytes=1024, max_workers=2)


def release(ticket, **changes):
    arguments = {'expected_ticket': ticket, 'operator': 'component-operator',
        'reason': 'explicit component decision; resource termination remains unconfirmed',
        'confirmation': gpu.RELEASE_CONFIRMATION}
    return gpu.release_unconfirmed_admission(**(arguments | changes))


def test_explicit_release_resolves_admission_without_termination_and_exactly_replays(namespace):
    job = admit()
    job.close()
    directory = namespace / 'jobs' / job.ticket
    original = (directory / 'admission.json').read_bytes()
    receipt = release(job.ticket)
    assert type(receipt) is gpu.AdmissionReleaseReceipt
    assert receipt.kind == 'hpo-gpu-admission-release'
    assert receipt.admitted_boot_id == receipt.current_boot_id == 'component-current-boot'
    assert release(job.ticket) == receipt
    assert json.loads((directory / 'release.json').read_bytes()) == asdict(receipt)
    assert (directory / 'admission.json').read_bytes() == original
    assert not (directory / 'termination.json').exists()
    assert not (directory / 'restart.json').exists()
    with pytest.raises(ContractError): job.confirm_termination()
    with admit('next') as next_job:
        assert next_job.ticket != job.ticket
        next_job.confirm_termination()


def test_recover_cli_executes_real_release_and_returns_distinct_receipt(namespace, capsys):
    from gf_cli.hpo import main
    job = admit()
    job.close()
    result = main(['recover', '--ticket', job.ticket, '--operator', 'component-operator',
        '--release-unconfirmed', '--reason', 'explicit operator acceptance of residual use'])
    assert result == 0
    value = json.loads(capsys.readouterr().out)
    assert value['kind'] == 'hpo-gpu-admission-release'
    assert value['ticket'] == job.ticket
    assert not {'resources', 'confirmed_unix'} & set(value)
    assert not (namespace / 'jobs' / job.ticket / 'termination.json').exists()


@pytest.mark.parametrize('changes', [
    {'expected_ticket': '0'*32}, {'expected_ticket': '../outside'}, {'operator': ''},
    {'operator': None}, {'reason': ''}, {'reason': ' \t\n'}, {'reason': None},
    {'reason': 'bad\x00reason'}, {'confirmation': 'yes'},
])
def test_release_requires_exact_existing_ticket_and_explicit_valid_arguments(namespace, changes):
    job = admit()
    job.close()
    directory = namespace / 'jobs' / job.ticket
    before = (directory / 'admission.json').read_bytes()
    with pytest.raises(ContractError): release(job.ticket, **changes)
    assert not (directory / 'release.json').exists()
    assert (directory / 'admission.json').read_bytes() == before
    with pytest.raises(gpu.RecoveryRequired): admit('still-blocked')


def test_release_never_defaults_operator_reason_or_confirmation(namespace):
    job = admit()
    job.close()
    for missing in ('operator', 'reason', 'confirmation'):
        arguments = {'expected_ticket': job.ticket, 'operator': 'operator', 'reason': 'explicit reason',
            'confirmation': gpu.RELEASE_CONFIRMATION}
        del arguments[missing]
        with pytest.raises(TypeError): gpu.release_unconfirmed_admission(**arguments)
    assert not (namespace / 'jobs' / job.ticket / 'release.json').exists()


@pytest.mark.parametrize('changed', ['operator', 'reason', 'confirmation', 'boot'])
def test_release_retry_is_exact_and_never_rewrites_receipts(namespace, monkeypatch, changed):
    job = admit()
    job.close()
    release(job.ticket)
    path = namespace / 'jobs' / job.ticket / 'release.json'
    before = path.read_bytes()
    changes = {}
    if changed == 'boot': monkeypatch.setattr(gpu, '_boot_id', lambda: 'different-current-boot')
    else: changes[changed] = 'different replay field'
    with pytest.raises(ContractError): release(job.ticket, **changes)
    assert path.read_bytes() == before


@pytest.mark.parametrize('resolution', ['termination', 'restart'])
def test_other_resolution_refuses_release_without_changing_old_receipt(namespace, resolution):
    job = admit()
    if resolution == 'termination': job.confirm_termination()
    job.close()
    if resolution == 'restart':
        gpu.confirm_environment_restart(expected_ticket=job.ticket, operator='operator', confirmation=gpu.RESTART_CONFIRMATION)
    directory = namespace / 'jobs' / job.ticket
    before = (directory / (resolution + '.json')).read_bytes()
    with pytest.raises(ContractError, match='resolved'): release(job.ticket)
    assert (directory / (resolution + '.json')).read_bytes() == before
    assert not (directory / 'release.json').exists()


def test_release_is_not_a_restart_and_retry_never_replaces_newer_pointer(namespace):
    job = admit()
    job.close()
    original = release(job.ticket)
    with pytest.raises(ContractError, match='resolved'):
        gpu.confirm_environment_restart(expected_ticket=job.ticket, operator='operator', confirmation=gpu.RESTART_CONFIRMATION)
    with admit('newer') as newer:
        newer.confirm_termination()
    pointer = (namespace / 'current.json').read_bytes()
    assert release(job.ticket) == original
    assert (namespace / 'current.json').read_bytes() == pointer
    assert json.loads(pointer)['ticket'] == newer.ticket


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('kind', 'hpo-gpu-termination'), ('ticket', '0'*32),
    ('admission_sha256', '0'*64), ('admitted_boot_id', 'wrong'), ('current_boot_id', ''),
    ('operator', None), ('reason', ' \n'), ('confirmation', gpu.RESTART_CONFIRMATION),
    ('released_unix', float('inf')), ('released_unix', float('nan')), ('released_unix', True),
    ('unexpected', 'extra'),
])
def test_malformed_or_wrong_binding_release_fails_closed_without_repair(namespace, field, value):
    job = admit()
    job.close()
    receipt = asdict(release(job.ticket))
    path = namespace / 'jobs' / job.ticket / 'release.json'
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    corrupt = path.read_bytes()
    with pytest.raises(ContractError): admit('blocked')
    with pytest.raises(ContractError): release(job.ticket)
    assert path.read_bytes() == corrupt


@pytest.mark.parametrize('other', ['termination.json', 'restart.json'])
def test_conflicting_resolution_files_are_never_ignored_or_rewritten(namespace, other):
    job = admit()
    job.close()
    release(job.ticket)
    directory = namespace / 'jobs' / job.ticket
    (directory / other).write_text('{}')
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ContractError, match='conflicting'): admit('blocked')
    with pytest.raises(ContractError, match='conflicting'): release(job.ticket)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_only_selected_orphan_is_resolved_and_other_unresolved_admissions_block(namespace):
    import uuid

    from gf_common.domain import digest
    job = admit()
    job.close()
    admission = json.loads((namespace / 'jobs' / job.ticket / 'admission.json').read_bytes())
    orphan = uuid.uuid4().hex
    orphan_directory = namespace / 'jobs' / orphan
    orphan_directory.mkdir(mode=0o700)
    # A second preserved admission simulates an independent publication orphan.
    gpu._write(orphan_directory / 'admission.json', {**admission, 'ticket': orphan}, immutable=True)
    pointer = (namespace / 'current.json').read_bytes()
    release(job.ticket)
    with pytest.raises(gpu.RecoveryRequired, match=orphan): admit('other-orphan-still-blocks')
    assert not (orphan_directory / 'release.json').exists()
    release(orphan)
    assert (namespace / 'current.json').read_bytes() == pointer
    lock = gpu._Lock()
    try:
        records = gpu._scan(lock)
        assert records[job.ticket][1] and records[orphan][1]  # Admission resolved, NOT terminated.
        assert json.loads(pointer)['admission_sha256'] == digest(admission)
    finally:
        lock.close()
    with admit('next') as next_job: next_job.confirm_termination()


@pytest.mark.parametrize('phase', ['file', 'directory', 'replay'])
def test_release_fsync_failures_never_return_success_and_exact_retry_resyncs(namespace, monkeypatch, phase):
    import os
    from pathlib import Path
    job = admit()
    job.close()
    path = namespace / 'jobs' / job.ticket / 'release.json'
    if phase == 'replay': release(job.ticket)
    actual_sync = gpu.os.fsync
    actual_directory = gpu._sync_directory
    def fail_sync(descriptor):
        target = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
        if ((phase == 'file' and target.name.startswith('release.json.tmp-'))
                or (phase == 'replay' and target == path)):
            raise OSError('receipt fsync fault')
        return actual_sync(descriptor)
    def fail_directory(directory):
        if directory == path.parent and path.exists(): raise OSError('receipt directory fsync fault')
        return actual_directory(directory)
    if phase == 'directory': monkeypatch.setattr(gpu, '_sync_directory', fail_directory)
    else: monkeypatch.setattr(gpu.os, 'fsync', fail_sync)
    with pytest.raises(OSError, match='fsync fault'): release(job.ticket)
    assert path.exists() == (phase != 'file')
    with pytest.raises((OSError, gpu.RecoveryRequired)): admit('cannot-assume-durability')
    monkeypatch.setattr(gpu.os, 'fsync', actual_sync)
    monkeypatch.setattr(gpu, '_sync_directory', actual_directory)
    receipt = release(job.ticket)
    assert release(job.ticket) == receipt
    with admit('after-resync') as next_job: next_job.confirm_termination()


def test_failed_orphan_pointer_publication_is_restored_only_on_exact_retry(namespace, monkeypatch):
    replace = gpu.os.replace
    def fail(*args): raise OSError('pointer publication fault')
    monkeypatch.setattr(gpu.os, 'replace', fail)
    with pytest.raises(OSError, match='pointer publication fault'): admit()
    ticket = next((namespace / 'jobs').iterdir()).name
    with pytest.raises(OSError, match='pointer publication fault'): release(ticket)
    release_path = namespace / 'jobs' / ticket / 'release.json'
    before = release_path.read_bytes()
    assert not (namespace / 'current.json').exists()
    monkeypatch.setattr(gpu.os, 'replace', replace)
    with pytest.raises(ContractError, match='missing GPU current pointer'): admit('must-reconcile-original-pointer')
    with pytest.raises(ContractError, match='retry differs'): release(ticket, reason='wrong retry')
    assert not (namespace / 'current.json').exists()
    release(ticket)
    assert release_path.read_bytes() == before
    assert json.loads((namespace / 'current.json').read_bytes())['ticket'] == ticket
    with admit('after-retry') as next_job: next_job.confirm_termination()


@pytest.mark.parametrize('kind', ['training', 'evaluation'])
def test_release_never_verifies_as_training_or_evaluation_termination(namespace, kind):
    from gf_cli.hpo_evaluation import verify_gpu_termination
    context = gpu.JobContext('campaign', 'operation', kind, 'run', 'evaluation' if kind == 'evaluation' else None)
    job = gpu.admit_gpu_job(context, global_cuda_bytes=1024, max_workers=2)
    job.close()
    released = asdict(release(job.ticket))
    directory = namespace / 'jobs' / job.ticket
    admission = gpu._read(directory / 'admission.json')
    with pytest.raises(ContractError): gpu._termination(released, admission)
    with pytest.raises(ContractError):
        verify_gpu_termination(released, campaign_id=context.campaign_id, operation_id=context.operation_id,
            run_id=context.run_id, evaluation_id=context.evaluation_id)
    # Even reshaping the copied JSON cannot supply an absent actual global Term.
    from gf_common.domain import digest
    forged = {'schema_version': 1, 'kind': 'hpo-gpu-termination', 'ticket': job.ticket,
        'admission_sha256': digest(admission), 'context_sha256': digest(admission['context']),
        'boot_id': admission['boot_id'], 'confirmed_unix': released['released_unix'], 'resources': []}
    with pytest.raises(OSError):
        verify_gpu_termination(forged, campaign_id=context.campaign_id, operation_id=context.operation_id,
            run_id=context.run_id, evaluation_id=context.evaluation_id)
    assert not (directory / 'termination.json').exists()


def test_live_actual_child_lock_refuses_operator_release_until_owner_closes(namespace):
    import select
    import subprocess
    import sys
    from pathlib import Path
    source = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from gf_training import hpo_gpu_lease as gpu
gpu._GLOBAL_ROOT=Path(sys.argv[2])
job=gpu.admit_gpu_job(gpu.JobContext('child','child','training','child',None),global_cuda_bytes=1024,max_workers=2)
try:
    print(job.ticket,flush=True)
    sys.stdin.readline()
finally:
    job.close()
"""
    child = subprocess.Popen([sys.executable, '-I', '-c', source,
        str(Path(gpu.__file__).resolve().parents[1]), str(namespace)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, 'bounded CPU child never reached actual admission'
        ticket = child.stdout.readline().strip()
        assert len(ticket) == 32
        with pytest.raises(gpu.AdmissionBusy): release(ticket)
        assert not (namespace / 'jobs' / ticket / 'release.json').exists()
        out, err = child.communicate('\n', timeout=10)
        assert child.returncode == 0, out + err
        released = release(ticket)
        assert released.ticket == ticket
        assert not (namespace / 'jobs' / ticket / 'termination.json').exists()
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=5)


@pytest.mark.parametrize('flags', [[], ['--release-unconfirmed'],
    ['--confirm-environment-restarted', '--release-unconfirmed', '--reason', 'reason'],
    ['--confirm-environment-restarted', '--reason', 'reason']])
def test_cli_requires_exactly_one_resolution_and_release_only_reason(namespace, flags):
    from gf_cli.hpo import main
    with pytest.raises(SystemExit) as caught:
        main(['recover', '--ticket', 'a'*32, '--operator', 'operator', *flags])
    assert caught.value.code == 2
    assert not namespace.exists()


@pytest.mark.parametrize('reason', ['', ' \n\t'])
def test_cli_refuses_empty_release_reason_before_namespace_creation(namespace, capsys, reason):
    from gf_cli.hpo import main
    assert main(['recover', '--ticket', 'a'*32, '--operator', 'operator',
        '--release-unconfirmed', '--reason', reason]) == 2
    assert json.loads(capsys.readouterr().out)['kind'] == 'hpo-command-failure'
    assert not namespace.exists()


def test_cli_old_restart_behavior_keeps_distinct_receipt(namespace, capsys):
    from gf_cli.hpo import main
    job = admit()
    job.close()
    assert main(['recover', '--ticket', job.ticket, '--operator', 'operator', '--confirm-environment-restarted']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['kind'] == 'hpo-gpu-environment-restart'
    assert not (namespace / 'jobs' / job.ticket / 'release.json').exists()


def test_recover_help_is_optional_dependency_free_and_explains_admission_only_release():
    import subprocess
    import sys
    source = """
import builtins,sys
original=builtins.__import__
def checked(name,*args,**kwargs):
    assert name.split('.')[0] not in {'torch','optuna','lightgbm','pyarrow','godfield_rust'}, name
    return original(name,*args,**kwargs)
builtins.__import__=checked
from gf_cli import main
try:
    main(['hpo','recover','--help'])
except SystemExit as error:
    assert error.code==0
else:
    raise AssertionError('help did not exit')
assert 'torch' not in sys.modules and 'optuna' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-E', '-c', source], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert '--release-unconfirmed' in result.stdout and '--reason' in result.stdout
    help_text = ' '.join(result.stdout.split())
    assert 'possible residual resource use' in help_text and 'does not certify termination' in help_text
