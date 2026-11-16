"""Explicit full owned audit consumes raw originals without issuing legacy Source."""
import asyncio
import copy
import time


def test_owned_full_external_collection8_numeric6_audit_remains_unqualified(tmp_path,monkeypatch):
    from gf_cli.card_instance import collect_hpo_ppo
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.hpo_source import validate_hpo_parent
    from gf_training.ppo_run import _run_ppo
    from tests.hpo_owned_support import owned_runtime
    from tests.test_group_pack_runtime_environment import PublicClient
    from tests.test_hpo_owned_run import install_owned_fixture_runtime
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    install_owned_fixture_runtime(monkeypatch)
    replay=PublicClient.replay
    async def own_replay(self,game,viewer):
        result=copy.deepcopy(await replay(self,game,viewer))
        for turn in result['turns']:
            for decision in turn['decisions']:
                observation=decision['observation']
                observation.update(viewer_player_id=viewer,current_player=decision['actor'],active_player=decision['actor'])
                observation['own']['player_id']=viewer
                observation['opponent']['player_id']=3-viewer
                decision['action_options']['actor']=decision['actor']
        return result
    monkeypatch.setattr(PublicClient,'replay',own_replay)
    def component_gate(state):
        assert state['device']=='cpu' and state['run_id']==run_id
    monkeypatch.setattr('gf_training.hpo_source.require_hpo_production',component_gate)
    def forbidden(*args,**kwargs): raise AssertionError('owned audit entered legacy Source factory')
    monkeypatch.setattr('gf_training.card_instance_source._open_batch',forbidden)
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture',run_id,'training',run_id,None),
                global_cuda_bytes=runtime.config.run.global_cuda_budget_bytes,max_workers=4) as job:
            ops=_OwnedHpoOperations(job)
            async def collect(plan,ledger): return await collect_hpo_ppo(runtime,plan,ledger)
            report=await _run_ppo(runtime,collect,directory,_game_ops=ops)
            assert report['status']=='complete' and report['used_decisions']==32
            ops.completion.close()
            assert ops.budget.used==0
        state,records=validate_hpo_parent(directory,require_complete=True,max_owned_bytes=2*1024**3,deadline=time.monotonic()+60)
        assert state['run_variant']=='hpo-ppo-owned-v1' and len(records)==3
        assert records[-1]['generation']==2
        assert not hasattr(records,'dispatch')
        # Original flags remain truthful after an explicit strict external audit.
        import json
        for number in (1,2):
            update=json.loads((directory/'updates'/f'{number:06}.json').read_text())
            assert update['source_proof']['replay_reconciled'] is False
            assert update['metrics']['parity']=={'status':'not_run'}
    asyncio.run(run())
