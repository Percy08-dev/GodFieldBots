"""Management diagnostics observe real game boundaries and actual wire phases."""
from io import BytesIO


def test_game_intervals_end_at_first_terminal_ack_and_retire_interrupted_games():
    from gf_common.domain import EpisodeResult
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    events=[]
    environment=CardInstanceSimulatorEnvironment.__new__(CardInstanceSimulatorEnvironment)
    environment.game_observer=events.append
    environment.capture=None
    environment._observed_terminal=set()
    result=EpisodeResult('episode','pair','seed',('a','b'),game_id='simulator-game-id')
    environment._episode_opened(result)
    environment._act_acknowledged(result,1,{}, {'status':'terminal','result':'draw'})
    environment._terminal_observed(result,2,{'result':'draw'})
    environment._episode_closed(result)
    assert [row['event'] for row in events]==['created','terminal']
    assert all(row['game_id']=='simulator-game-id' for row in events)
    assert events[1]['monotonic']>=events[0]['monotonic']
    interrupted=EpisodeResult('episode2','pair','seed',('a','b'),game_id='interrupted-game-id')
    environment._episode_opened(interrupted)
    environment._episode_closed(interrupted)
    assert [row['event'] for row in events[-2:]]==['created','retired']


def test_profiled_framing_preserves_bytes_and_separates_body_read_from_json_decode():
    from gf_simulator_client.framing import read_frame_profiled, write_frame, write_frame_profiled
    value={'numbers':list(range(1000))}
    reference,profiled=BytesIO(),BytesIO()
    write_frame(reference,value)
    written=write_frame_profiled(profiled,value)
    assert reference.getvalue()==profiled.getvalue()
    profiled.seek(0)
    actual,read=read_frame_profiled(profiled)
    assert actual==value
    assert written['worker_driver_request_encode_ms']>=0
    assert read['worker_driver_reply_body_ms']>=0
    assert read['worker_driver_reply_json_decode_ms']>=0
    assert read['worker_driver_reply_bytes']==len(reference.getvalue())-8


def test_optional_park_host_sample_is_typed_and_never_a_cuda_peak():
    from gf_common.hpo_progress import pool_diagnostic
    value={'schema_version':1,'loads':0,'cache_reuses':0,'cpu_weight_restore_bytes':0,
        'cuda_weight_restore_bytes':0,'cuda_allocator':None,'host_memory':{
            'scope':'process-current-after-command','boundary':'PARK','pss_bytes':123,
            'private_bytes':100,'shared_clean_bytes':20,'shared_dirty_bytes':3,'max_rss_bytes':200}}
    assert pool_diagnostic(value)==value
    value['host_memory']['pss_bytes']='not-bytes'
    assert pool_diagnostic(value) is None
