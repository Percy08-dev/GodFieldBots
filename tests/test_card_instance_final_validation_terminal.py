"""Inert final-C5 terminal probes; no formal admission, model or games.

The actual lock, publisher, physical pending controls and cleanup paths run.
Parent qualification/raw episode parsing remain the explicitly UNADMITTED doubles.
"""
import asyncio
import copy
import os
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_training import card_instance_final_validation as module
from gf_training.collection import FinalizedCollection
from gf_training.lineage import _OWNED_LOCKS
from tests.test_card_instance_final_validation_owner import actor_context, unadmitted_parent
from tests.test_card_instance_final_validation_retention import controlled_originals


def stage(context, parent, monkeypatch):
    controlled_originals(context, parent, monkeypatch)
    return context._issued, context._issued.publication


def transfer_inert_lock_to_parent(parent, scope, monkeypatch):
    """Actual temporary lock transfer for an UNADMITTED owned-parent primitive."""
    from gf_training import card_instance_source
    parent.borrowed = None
    parent._owns_lock = True
    parent._scope.callback(scope.__exit__, None, None, None)
    monkeypatch.setattr(card_instance_source, 'recheck_game_parent_snapshot', lambda scan: None)


def test_pending_physical_controls_are_not_canonical_and_source_never_closes_publisher_fd(tmp_path, monkeypatch):
    value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            descriptors = [output.handle.descriptor for output in publication.outputs]
            def identity(descriptor):
                info = os.fstat(descriptor)
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
            identities = [identity(descriptor) for descriptor in descriptors]
            assert not (directory / 'manifest.json').exists() and (directory / 'manifest.json.tmp').is_file()
            source = context.accept_collection(token)
            assert source.facts['qualified'] is False and len(source.selected_records) == 64
            assert all(identity(descriptor) == expected for descriptor, expected in zip(descriptors, identities, strict=True))
            other = unadmitted_parent(value, parent.directory, parent.borrowed, monkeypatch)
            with pytest.raises(FileNotFoundError), module.open_game_final_validation(value, other, directory,
                    deadline=time.monotonic() + 60, max_owned_bytes=1024**3):
                pytest.fail('canonical-only readonly opener accepted actual pending control')
            await context.__aexit__(None, None, None)
            assert context.cleanup_complete and publication.committed and parent._active
            assert (directory / 'manifest.json').is_file() and not (directory / 'manifest.json.tmp').exists()
            assert all(output.handle.state == 'released' for output in publication.outputs)
            with pytest.raises(ContractError): _ = token.report
            raw = (directory / 'manifest.json').read_bytes()
            with pytest.raises(ContractError): publication.finish_final_validation()
            assert (directory / 'manifest.json').read_bytes() == raw  # Duplicate finish never revokes success.
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('foreign', ['dict', 'metadata-result', 'copied-token'])
def test_only_same_actual_issued_pending_token_reaches_original_validator(tmp_path, monkeypatch, foreign):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            value = ({'directory': str(directory), 'report': token.report, 'qualified': True} if foreign == 'dict' else
                FinalizedCollection(directory, token.report) if foreign == 'metadata-result' else copy.copy(token))
            with pytest.raises(ContractError): context.accept_collection(value)
            assert not (directory / 'manifest.json').exists()
            assert (directory / 'manifest.json.tmp').read_bytes().startswith(b'!')
            assert publication.cleanup_complete and not publication.committed
            with pytest.raises(ContractError): await context.aclose()
            assert context.cleanup_complete
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


def test_cancellation_after_source_acceptance_invalidates_manifest_first_and_preserves_primary(tmp_path, monkeypatch):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    seen = []
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            original = publication._invalidate_output
            def invalidate(output):
                seen.append(output.name)
                original(output)
            monkeypatch.setattr(publication, '_invalidate_output', invalidate)
            error = asyncio.CancelledError('UNADMITTED cancellation after acceptance')
            await context.__aexit__(type(error), error, error.__traceback__)
            assert context._failure is error and context.cleanup_complete
            assert seen[0] == 'manifest.json'
            assert not (directory / 'manifest.json').exists()
            assert (directory / 'manifest.json.tmp').read_bytes().startswith(b'!')
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('failure', [None, 'before-release', 'after-release'])
def test_owned_parent_release_precedes_exclusive_manifest_exposure(tmp_path, monkeypatch, failure):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    transfer_inert_lock_to_parent(parent, scope, monkeypatch)
    seen = []
    real_link = os.link
    def link(source, target, **kwargs):
        if target == 'manifest.json':
            seen.append('canonical-link')
            assert parent.cleanup_complete and context._loan_retired
        return real_link(source, target, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            await context._retire_for_owned_parent(parent)
            assert not (directory / 'manifest.json').exists() and parent in module._BORROWS
            original_close = parent._scope.close
            error = OSError('UNADMITTED reportable owned parent release failure')
            if failure == 'before-release':
                monkeypatch.setattr(parent._scope, 'close', lambda: (_ for _ in ()).throw(error))
                with pytest.raises(OSError): parent.__exit__(None, None, None)
                context._record_failure(error)
                assert not parent.cleanup_complete and not context.cleanup_complete
                assert not (directory / 'manifest.json').exists() and parent in module._BORROWS
                monkeypatch.setattr(parent._scope, 'close', original_close)
                parent.__exit__(type(error), error, error.__traceback__)
                assert not parent.cleanup_complete  # Uncertain original close is never repaired by an empty retry.
                with pytest.raises(ContractError): context._finish_after_owned_parent(parent)
                assert not context.cleanup_complete and not (directory / 'manifest.json').exists()
                return
            else:
                parent.__exit__(None, None, None)
                if failure == 'after-release': context._record_failure(error)
            assert parent.cleanup_complete and parent.directory not in _OWNED_LOCKS
            if failure is None:
                context._finish_after_owned_parent(parent)
                assert seen == ['canonical-link'] and publication.committed
                assert (directory / 'manifest.json').exists()
            else:
                with pytest.raises(OSError) as caught: context._finish_after_owned_parent(parent)
                assert caught.value is error and not seen and not (directory / 'manifest.json').exists()
                assert (directory / 'manifest.json.tmp').read_bytes().startswith(b'!')
            assert context.cleanup_complete and parent not in module._BORROWS
        asyncio.run(run())
    finally:
        if parent.directory in _OWNED_LOCKS:
            scope.__exit__(None, None, None)


def test_early_owned_finish_is_sticky_and_never_repaired_into_completion(tmp_path, monkeypatch):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    transfer_inert_lock_to_parent(parent, scope, monkeypatch)
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            await context._retire_for_owned_parent(parent)
            with pytest.raises(ContractError): context._finish_after_owned_parent(parent)
            assert not (directory / 'manifest.json').exists() and not publication.committed
            parent.__exit__(None, None, None)
            with pytest.raises(ContractError): context._finish_after_owned_parent(parent)
            assert context.cleanup_complete and not (directory / 'manifest.json').exists()
        asyncio.run(run())
    finally:
        if parent.directory in _OWNED_LOCKS: scope.__exit__(None, None, None)


@pytest.mark.parametrize('fault', ['after-link', 'after-release', 'late-clock'])
def test_failure_after_actual_exposure_revokes_known_manifest_before_error(tmp_path, monkeypatch, fault):
    from gf_training import card_instance_collection as records
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    first = OSError('UNADMITTED failure after actual publication')
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            if fault == 'after-link':
                real_link = os.link
                def link(source, target, **kwargs):
                    value = real_link(source, target, **kwargs)
                    if target == 'manifest.json': raise first
                    return value
                monkeypatch.setattr(os, 'link', link)
            else:
                release = publication._release
                clock = [time.monotonic()]
                if fault == 'late-clock': monkeypatch.setattr(records, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
                def close():
                    release()
                    if fault == 'after-release': raise first
                    clock[0] = publication.work_deadline + .001  # Still within original finalization reserve.
                monkeypatch.setattr(publication, '_release', close)
            expected = TimeoutError if fault == 'late-clock' else OSError
            with pytest.raises(expected) as caught: await context.__aexit__(None, None, None)
            if fault != 'late-clock': assert caught.value is first
            assert (directory / 'manifest.json').read_bytes().startswith(b'!')
            assert not publication.committed and publication.cleanup_complete and context.cleanup_complete
            assert parent._active and parent not in module._BORROWS
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


def test_exclusive_last_link_preserves_foreign_manifest_and_invalidates_only_owned_files(tmp_path, monkeypatch):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            real_link = os.link
            def link(source, target, **kwargs):
                if target == 'manifest.json': (directory / 'manifest.json').write_bytes(b'FOREIGN')
                return real_link(source, target, **kwargs)
            monkeypatch.setattr(os, 'link', link)
            with pytest.raises(FileExistsError): await context.__aexit__(None, None, None)
            assert (directory / 'manifest.json').read_bytes() == b'FOREIGN'
            assert (directory / 'manifest.json.tmp').read_bytes().startswith(b'!')
            assert not publication.committed and context.cleanup_complete
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('target', ['report.json', 'manifest.json', 'directory'])
def test_final_close_then_fd_reuse_never_inspects_or_closes_foreign_descriptor(tmp_path, monkeypatch, target):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    foreign = tmp_path / 'foreign'
    foreign.write_bytes(b'FOREIGN')
    real_close, real_open, real_fstat = os.close, os.open, os.fstat
    reused = []
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            descriptor = (publication.directory_fd if target == 'directory' else
                next(value.handle.descriptor for value in publication.outputs if value.name == target))
            first = OSError('UNADMITTED close after actual release and foreign fd reuse')
            def close(number):
                if number == descriptor and not reused:
                    real_close(number)
                    opened = real_open(foreign, os.O_RDONLY)
                    if opened != number:
                        os.dup2(opened, number); real_close(opened)
                    reused.append(number)
                    raise first
                if reused and number == reused[0]: pytest.fail('retried uncertain stale descriptor close')
                return real_close(number)
            def fstat(number):
                if reused and number == reused[0]: pytest.fail('inspected uncertain stale descriptor')
                return real_fstat(number)
            monkeypatch.setattr(os, 'close', close)
            monkeypatch.setattr(os, 'fstat', fstat)
            with pytest.raises(OSError) as caught: await context.__aexit__(None, None, None)
            assert caught.value is first
            assert first._game_final_validation_cleanup_owner is context
            assert not context.cleanup_complete and parent in module._BORROWS
            assert (directory / 'manifest.json').read_bytes().startswith(b'!')
            assert foreign.read_bytes() == b'FOREIGN' and os.read(reused[0], 7) == b'FOREIGN'
            assert publication.quarantined and not publication.committed
        asyncio.run(run())
    finally:
        if reused: real_close(reused[0])  # Only the injector knows this is the foreign object.
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('fault', [None, 'parent-guard', 'parent-close', 'publication-release'])
def test_explicit_cli_terminal_composition_keeps_pending_owner_across_actual_parent_release(tmp_path, monkeypatch, fault):
    from gf_cli import card_instance as cli
    from gf_training import card_instance_source
    from gf_training.card_instance_formal import FormalGameParent
    from gf_training.lineage import run_lock
    from tests.formal_control_support import unadmitted_current_synthetic_formal_shape
    from tests.test_formal_ppo_control import loaded
    monkeypatch.setattr(module, 'formal_preflight', unadmitted_current_synthetic_formal_shape)
    value = loaded(tmp_path)
    parent_path, directory = tmp_path / 'UNADMITTED-CLI-parent', tmp_path / 'validation'
    actual, contexts, publishers, lock_scopes = [], [], [], []
    first = OSError('UNADMITTED concrete CLI terminal fault')
    def enter(self):
        scope = run_lock(self.directory, create=True)
        ownership = scope.__enter__()
        lock_scopes.append(scope)
        primitive = unadmitted_parent(value, self.directory, ownership, monkeypatch)
        self.__dict__.update(primitive.__dict__)
        transfer_inert_lock_to_parent(self, scope, monkeypatch)
        actual.append(self)
        if fault == 'parent-guard':
            monkeypatch.setattr(card_instance_source, 'recheck_game_parent_snapshot', lambda scan: (_ for _ in ()).throw(first))
        if fault == 'parent-close':
            self._actual_test_close = self._scope.close
            monkeypatch.setattr(self._scope, 'close', lambda: (_ for _ in ()).throw(first))
        return self
    monkeypatch.setattr(FormalGameParent, '__enter__', enter)
    async def collect(context):
        contexts.append(context)
        controlled_originals(context, actual[0], monkeypatch)
        publication = context._issued.publication
        publishers.append(publication)
        if fault == 'publication-release':
            release = publication._release
            def close():
                release()
                raise first
            monkeypatch.setattr(publication, '_release', close)
        return context._issued
    monkeypatch.setattr(cli, 'collect_owned_final_validation', collect)
    real_link = os.link
    def link(source, target, **kwargs):
        if target == 'manifest.json':
            assert actual[0].cleanup_complete
            assert contexts[0]._loan_retired and actual[0] in module._BORROWS
        return real_link(source, target, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    try:
        if fault is None:
            facts = asyncio.run(cli.collect_formal_final_validation(value, parent_path, directory))
            assert facts['qualified'] is False and contexts[0].cleanup_complete
            assert actual[0].cleanup_complete and (directory / 'manifest.json').exists()
        else:
            with pytest.raises(OSError) as caught:
                asyncio.run(cli.collect_formal_final_validation(value, parent_path, directory))
            assert caught.value is first
            if fault == 'parent-close':
                assert first._game_final_validation_cleanup_owner is contexts[0]
                assert first._game_final_validation_parent_owner is actual[0]
                assert not contexts[0].cleanup_complete and not actual[0].cleanup_complete
                assert actual[0] in module._BORROWS and not (directory / 'manifest.json').exists()
                monkeypatch.setattr(actual[0]._scope, 'close', actual[0]._actual_test_close)
                actual[0].__exit__(type(first), first, first.__traceback__)
                assert not actual[0].cleanup_complete
                with pytest.raises(ContractError): contexts[0]._finish_after_owned_parent(actual[0])
                assert not contexts[0].cleanup_complete
            else:
                assert contexts[0].cleanup_complete and actual[0].cleanup_complete
            if (directory / 'manifest.json').exists():
                assert (directory / 'manifest.json').read_bytes().startswith(b'!')
            else:
                assert (directory / 'manifest.json.tmp').read_bytes().startswith(b'!')
        assert not (parent_path / 'run.json').exists() and not (parent_path / 'formal-declaration.json').exists()
        assert all(output.handle.state != 'live' for output in publishers[0].outputs)
    finally:
        if actual and actual[0].directory in _OWNED_LOCKS:
            for scope in lock_scopes: scope.__exit__(None, None, None)


@pytest.mark.parametrize('diagnostic', ['secondary-format-memory', 'secondary-format-error', 'primary-note-memory'])
def test_secondary_diagnostics_cannot_prevent_first_failure_pending_manifest_invalidation(tmp_path, monkeypatch, diagnostic):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    directory = context.plan.output
    class Primary(RuntimeError):
        def add_note(self, text):
            if diagnostic == 'primary-note-memory': raise MemoryError('UNADMITTED add_note allocation')
            super().add_note(text)
    class Secondary(RuntimeError):
        def __str__(self):
            if diagnostic == 'secondary-format-memory': raise MemoryError('UNADMITTED secondary formatting allocation')
            if diagnostic == 'secondary-format-error': raise ValueError('UNADMITTED secondary formatting failure')
            return 'UNADMITTED later cleanup failure'
    primary, secondary = Primary('UNADMITTED initiating actor failure'), Secondary()
    try:
        async def run():
            await context.__aenter__()
            token, publication = stage(context, parent, monkeypatch)
            context.accept_collection(token)
            context._failure = primary
            actual_fail = publication.fail_final_validation
            invalidated = []
            def fail(error):
                assert error is primary and context._failure is primary
                assert error._game_final_validation_cleanup_owner is context
                assert error._game_final_validation_parent_owner is parent
                assert error._card_instance_collection_cleanup_owner is publication
                actual_fail(error)
                invalidated.append(True)
            monkeypatch.setattr(publication, 'fail_final_validation', fail)
            context._record_failure(secondary)
            assert invalidated and context._failure is primary
            assert (directory / 'manifest.json.tmp').read_bytes().startswith(b'!')
            assert not (directory / 'manifest.json').exists() and publication.cleanup_complete
            assert primary._game_final_validation_cleanup_owner is context
            assert primary._game_final_validation_parent_owner is parent
            with pytest.raises(Primary) as error: await context.aclose()
            assert error.value is primary and context.cleanup_complete
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


def test_issued_report_clone_admits_parent_before_decoder(tmp_path, monkeypatch):
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    calls = []
    try:
        async def run():
            await context.__aenter__()
            token, _publication = stage(context, parent, monkeypatch)
            admit, loads = parent.admit_external, module.json.loads
            def reserve(amount):
                calls.append(('admit', amount))
                return admit(amount)
            def decode(raw):
                assert calls[-1] == ('admit', context._reserved_bytes)
                calls.append(('decode', len(raw)))
                return loads(raw)
            monkeypatch.setattr(parent, 'admit_external', reserve)
            monkeypatch.setattr(module, 'json', SimpleNamespace(loads=decode))
            assert token.report['healthy'] is True
            with pytest.raises(ContractError): await context.aclose()  # No Source accepted in this clone-only primitive.
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)
