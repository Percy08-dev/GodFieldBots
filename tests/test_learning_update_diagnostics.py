"""Diagnostic imports stay separate from ordinary resume and Source authority."""
import copy
from functools import partial

import pytest

torch=pytest.importorskip('torch')

from benchmarks.learning_update.diagnostic import DiagnosticRollout
from gf_common.domain import ContractError
from gf_models.structured_artifacts import (
    _validate_diagnostic_resume_manifest,
    _validate_resume_manifest,
)
from gf_training.hpo_artifacts import _recipe_options
from gf_training.hpo_training_execution import TrainingExecutionSpec
from tests.test_hpo_numeric import learner_for, rollout_for


def test_historical_import_does_not_relax_ordinary_resume(tmp_path):
    import json
    learner=learner_for()
    learner.save(tmp_path/'saved')
    manifest=json.loads((tmp_path/'saved'/'resume.json').read_text())
    recorded=copy.deepcopy(manifest)
    recorded['producer']['source_digest']='a'*64
    options={'context':learner._context,'expected_inference_digest':manifest['inference_digest'],
        **_recipe_options(learner.training)}
    with pytest.raises(ContractError,match='source identity'):
        _validate_resume_manifest(recorded,learner.model,learner.binding,learner.resume_identity,**options)
    assert _validate_diagnostic_resume_manifest(recorded,learner.model,learner.binding,learner.resume_identity,
        recorded_source_digest='a'*64,**options)==0
    with pytest.raises(ContractError,match='source identity'):
        _validate_diagnostic_resume_manifest(recorded,learner.model,learner.binding,learner.resume_identity,
            recorded_source_digest='b'*64,**options)
    assert manifest['producer']!=recorded['producer']
    # Ordinary wrapper has no parameter permitting a recorded-source override.
    with pytest.raises(TypeError):
        _validate_resume_manifest(recorded,learner.model,learner.binding,learner.resume_identity,
            recorded_source_digest='a'*64,**options)


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_numpy_adapter_uses_same_original_tensor_core(kind):
    learner=learner_for(kind)
    rollout=rollout_for(learner,count=3)
    adapter=DiagnosticRollout(rollout)
    assert not hasattr(adapter,'banks') and not hasattr(adapter,'_state')
    assert not hasattr(adapter,'require_learning')
    assert adapter.locations==tuple(rollout.locations)
    for index in range(len(adapter)):
        expected=learner.network(*rollout.endpoint(index),suffix_length=1)
        actual=adapter.forward(learner.network,*adapter.endpoint(index),suffix_length=1)
        assert adapter.vector(index)==rollout.vector(index)
        for left,right in zip(expected,actual,strict=True):
            torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_closed_spec_rejects_oversized_special_and_duplicate_files(tmp_path):
    import hashlib

    from gf_training.hpo_training_execution import read_execution_spec
    for name,raw in [('oversized',b' '*4097),('duplicate',b'{"variant":"baseline","variant":"packed"}')]:
        path=tmp_path/name
        path.write_bytes(raw)
        with pytest.raises(ContractError):
            read_execution_spec(path,hashlib.sha256(raw).hexdigest())
    linked=tmp_path/'linked'
    linked.symlink_to(tmp_path/'duplicate')
    with pytest.raises((OSError,ContractError)):
        read_execution_spec(linked,'0'*64)
    directory=tmp_path/'directory'
    directory.mkdir()
    with pytest.raises(ContractError):
        read_execution_spec(directory,'0'*64)


def test_profile_window_is_separate_bounded_and_reports_missing_cuda(tmp_path):
    from gf_training.hpo_update_diagnostics import ProfileWindow
    learner=learner_for()
    rollout=DiagnosticRollout(rollout_for(learner,count=3))
    learner._training_execution=TrainingExecutionSpec(variant='metrics')
    profile=ProfileWindow(learner,tmp_path,1,1,False)
    with profile.patches():
        learner._update_numeric(rollout,{},0.,partial(rollout.forward,learner.network))
        profile.finish()
    assert profile.finished and profile.summary['captured_groups']==1
    assert profile.summary['missing_cuda_activity']
    assert 0<profile.summary['trace_bytes']<=512*1024**2
    assert profile.summary['event_count']>0


def test_coordinator_rejects_experiment_resume_before_any_job(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from gf_training.hpo_ppo import train_hpo_ppo
    from gf_training.hpo_training_execution import write_execution_spec
    path=tmp_path/'spec.json'
    sha=write_execution_spec(path,TrainingExecutionSpec())
    loaded=SimpleNamespace(config=SimpleNamespace(hpo_spec={'schema_version':4}))
    with pytest.raises(ContractError,match='fresh producer-owned'):
        asyncio.run(train_hpo_ppo(loaded,None,tmp_path/'parent',gpu_job=None,mode='resume',
            _training_execution=(str(path),sha)))
