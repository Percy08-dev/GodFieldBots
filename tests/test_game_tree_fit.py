"""Closed child/resource transport primitives only; no Source or actual fitting."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_training.structured_tree_fit import _OwnedTreeFitChild

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('cap',['0','-1','02','bad','8589934593'])
def test_fourth_child_cap_rejects_before_request_or_native_import(tmp_path,cap):
    # Failure is wholly in fixed stdlib bootstrap, before any request file exists.
    result=subprocess.run([sys.executable,'-E','-m','gf_training.structured_tree_worker',
        str(tmp_path/'missing'),'a'*64,str(time.monotonic()+10),cap],cwd=ROOT/'src',
        capture_output=True,text=True,check=False,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert result.returncode!=0
    assert 'child cap' in result.stderr and 'FileNotFoundError' not in result.stderr
    assert not (tmp_path/'missing').exists()


def test_fixture_context_cannot_receive_qualified_fourth_argument(tmp_path):
    (tmp_path/'request.json').write_text(json.dumps({'schema_version':4,'source_kind':'card-instance-tree-fixture-originals'}))
    import hashlib
    sha=hashlib.sha256((tmp_path/'request.json').read_bytes()).hexdigest()
    result=subprocess.run([sys.executable,'-E','-m','gf_training.structured_tree_worker',str(tmp_path),sha,
        str(time.monotonic()+10),str(2*1024**3)],cwd=ROOT/'src',capture_output=True,text=True,check=False,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert result.returncode!=0 and 'fourth resource argument/context mismatch' in result.stderr


@pytest.mark.parametrize('cap',[None,2*1024**3,4*1024**3])
def test_same_owned_child_appends_only_full_configured_resource_cap(monkeypatch,tmp_path,cap):
    seen=[]
    class StopBeforeActualCreation(RuntimeError):pass
    async def create(*args,**options):
        seen.append((args,options))
        raise StopBeforeActualCreation('UNADMITTED transport-only creation refusal')
    monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
    child=_OwnedTreeFitChild(tmp_path,time.monotonic()+60,1024,child_limit_bytes=cap)
    with pytest.raises(StopBeforeActualCreation):asyncio.run(child.start('b'*64))
    args,options=seen[0]
    assert args[:4]==(sys.executable,'-E','-m','gf_training.structured_tree_worker')
    assert args[4:7]==(str(tmp_path),'b'*64,str(child.deadline))
    assert args[7:]==(() if cap is None else (str(cap),))
    assert options['env']['CUDA_VISIBLE_DEVICES']==''
    assert child.process is None


def test_resource_constructor_does_not_downscale_explicit_over_total_cap(tmp_path):
    with pytest.raises(ContractError):
        _OwnedTreeFitChild(tmp_path,time.monotonic()+60,1024,child_limit_bytes=8*1024**3+1)
