"""A late successful probe cannot revive a connection poisoned by a sibling."""
import asyncio

import pytest

from gf_simulator_client.http import AttestationError
from tests.test_http import client_for, info, response


@pytest.mark.parametrize('sibling_outcome', ['failure', 'cancel'])
@pytest.mark.parametrize('initial_cache,operation', [(False, 'attest'), (True, 'attest'), (False, 'create'), (True, 'create'), (True, 'act')])
def test_late_valid_attestation_never_refreshes_cache_readiness_or_sends_after_poison(initial_cache, operation, sibling_outcome):
    async def run():
        value = info()
        started, release, sibling_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        initial = True
        probes = 0
        paths = []
        async def handler(request):
            nonlocal probes
            paths.append((request.method, request.url.path))
            assert request.url.path == '/debug/v1/test-server/info', 'no Create/Act/catalog allowed after competing gate failure'
            if initial:
                return response(value)
            probes += 1
            if probes == 1:
                started.set()
                await release.wait()
                return response(value)
            assert probes == 2
            sibling_started.set()
            if sibling_outcome == 'cancel':
                await asyncio.Event().wait()
            return response({'failed': True}, status=500)
        client = client_for(handler, value)
        try:
            if initial_cache:
                await client.attest()
            initial = False
            client.needs_reattest = True
            cached_before = client.cached
            if operation == 'act':
                waiter = client.act('owned-game', 1, {'command_id': 'unchanged-control-id', 'expected_revision': 0, 'action': {'type': 'pray'}})
            else:
                waiter = getattr(client, operation)()
            late = asyncio.create_task(waiter)
            await started.wait()
            sibling = asyncio.create_task(client.attest())
            await sibling_started.wait()
            if sibling_outcome == 'cancel':
                sibling.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sibling
            else:
                with pytest.raises(AttestationError):
                    await sibling
            assert client.poisoned
            release.set()
            with pytest.raises(AttestationError, match='poisoned'):
                await late
            assert client.cached is cached_before and client.needs_reattest is True
            expected_requests = len(paths)
            for forbidden in (client.attest(), client.create(), client.act('owned-game', 1, {}), client.public_catalog()):
                with pytest.raises(AttestationError):
                    await forbidden
            assert len(paths) == expected_requests == 2 + int(initial_cache)
        finally:
            await client.close()
    asyncio.run(run())
