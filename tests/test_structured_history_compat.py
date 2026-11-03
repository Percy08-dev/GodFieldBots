"""Actual pre-correction bytes, historical diagnostics only, never Source authority."""
import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, canonical
from gf_models.arrays import sha256
from gf_models.structured_artifacts import inspect_inference
from gf_models.structured_config import recorded_model_config_v2
from gf_models.structured_sealed import SEMANTIC_KEYS, SealedScan, inspect_retained_artifact
from gf_training.structured_lineage import validate_component_chain
from gf_training.structured_rollout import _read_recorded_rollout, read_rollout, write_rollout

ROOT = Path(__file__).parent / 'fixtures/structured_history62_parent'
PROOF = json.loads((ROOT.parent / 'structured_history62_files.json').read_text())
EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def verify_frozen_fixture(directory):
    for name, expected in PROOF.items():
        path = directory / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            # The original committed archive omits only this empty transient lock.
            if name == '.lock' and expected == EMPTY_SHA256:
                continue
            raise
        assert stat.S_ISREG(metadata.st_mode), f'non-file frozen fixture entry: {name}'
        assert sha256(path) == expected


def test_actual_old_bytes_and_whole_retained_pruned_metadata_inspect_unchanged():
    verify_frozen_fixture(ROOT)
    before = (ROOT / 'run.json').read_bytes()
    state, records = validate_component_chain(ROOT, max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 60)
    assert state['updates'] == 2 and len(records) == 3 and records[-1]['optimizer_steps'] == 8
    assert 'aggregate_recipe' not in state['effective_config']['model']['event']
    for number in (0, 2):
        directory = ROOT / 'artifacts' / f'{number:06}'
        result = inspect_retained_artifact(directory, expected_inference_digest=sha256(directory / 'inference.json'),
            action_contract=engineering_contract(), purpose='component', synthetic_capacity=False,
            scan=SealedScan(max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 60))
        original_model = {key: value for key, value in state['effective_config']['model'].items()
                          if key not in ('checkpoint', 'inference', 'cuda_visible_devices')}
        assert canonical(result.inference['model']) == canonical(original_model)
        assert 'aggregate_recipe' not in result.inference['model']['event']
    assert (ROOT / 'run.json').read_bytes() == before


@pytest.mark.parametrize('lock_kind', ['absent', 'empty', 'nonempty', 'dangling', 'directory'])
def test_copied_frozen_fixture_only_missing_or_empty_lock_is_permitted(tmp_path, lock_kind):
    directory = tmp_path / 'copied'
    shutil.copytree(ROOT, directory)
    lock = directory / '.lock'
    if lock.exists():
        lock.unlink()
    if lock_kind == 'empty':
        lock.write_bytes(b'')
    elif lock_kind == 'nonempty':
        lock.write_bytes(b'changed')
    elif lock_kind == 'dangling':
        lock.symlink_to(directory / 'missing-target')
    elif lock_kind == 'directory':
        lock.mkdir()
    if lock_kind in ('absent', 'empty'):
        verify_frozen_fixture(directory)
    else:
        with pytest.raises(AssertionError):
            verify_frozen_fixture(directory)


def test_copied_frozen_fixture_payload_is_always_required(tmp_path):
    directory = tmp_path / 'copied'
    shutil.copytree(ROOT, directory)
    (directory / 'run.json').unlink()
    with pytest.raises(FileNotFoundError):
        verify_frozen_fixture(directory)


@pytest.mark.parametrize('batch', [0, 1])
def test_actual_old_rollout_tokens_reproduce_and_are_never_active(batch, tmp_path, monkeypatch):
    directory = ROOT / 'numeric' / f'batch-{batch:06}'
    manifest = json.loads((directory / 'rollout.json').read_text())
    raw_model = json.loads((ROOT / 'artifacts/000000/inference.json').read_text())['model']
    descriptor = recorded_model_config_v2(raw_model, manifest['token_signature'])
    result = _read_recorded_rollout(directory, sha256(directory / 'rollout.json'), descriptor,
        action_contract=engineering_contract(), declaration={key: manifest[key] for key in SEMANTIC_KEYS},
        expected_records=None, max_owned_bytes=512 * 1024**2, extra_owned_bytes=0, check=lambda: None, reserve=lambda _: None)
    assert len(result) == 16 and all(len(frame.history) == 62 for episode in result.episodes for frame in episode.frames)
    with pytest.raises(ContractError):
        read_rollout(directory, sha256(directory / 'rollout.json'), descriptor,
            action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    with pytest.raises(ContractError):
        write_rollout(tmp_path / 'forbidden', result)
    assert not (tmp_path / 'forbidden').exists()
    import gf_models.structured_artifacts as module
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy tensors opened/constructed on active load')
    monkeypatch.setattr(module, 'inspect_bundle', forbidden)
    with pytest.raises(ContractError):
        inspect_inference(ROOT / 'artifacts/000000', action_contract=engineering_contract(),
                          purpose='component', synthetic_capacity=False)


def test_actual_historical_inspection_has_no_torch_current_host_metadata_or_producer_equality():
    code = '''
import importlib.abc,importlib.metadata,sys,time
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch forbidden')
sys.meta_path.insert(0,NoTorch())
def forbidden(*args,**kwargs):raise AssertionError('current metadata/host query forbidden')
importlib.metadata.version=forbidden
from gf_models import structured_artifacts,torch_runtime
structured_artifacts.runtime_metadata=forbidden
structured_artifacts.producer_identity=forbidden
torch_runtime.hardware=forbidden
from gf_training.structured_lineage import validate_component_chain
s,r=validate_component_chain(Path(sys.argv[1]),max_owned_bytes=512*1024**2,deadline=time.monotonic()+60)
assert len(r)==3 and s['updates']==2 and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(ROOT)], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_historical_raw_tokens_validate_but_active_network_rejects_before_packing(make_input, monkeypatch):
    torch = pytest.importorskip('torch')
    from dataclasses import replace

    from gf_common.structured_history import HISTORICAL_AGGREGATE
    from gf_common.structured_tokens import signature, validate_frame_tokens
    from gf_models.structured_network import make_network
    from tests.test_structured_network import inputs, small_model
    frames, table = inputs(make_input, count=1)
    model = small_model()
    network = make_network(model, 17)
    old_frame = replace(frames[0], recipe_signature=signature(128, aggregate_recipe=HISTORICAL_AGGREGATE), history=frames[0].history[:62])
    old_table = replace(table, recipe_signature=old_frame.recipe_signature)
    validate_frame_tokens(old_frame)
    with torch.no_grad():
        context, cards = network.encode_history(frames)
        cache = network.new_cache()
    def forbidden(*args, **kwargs):
        raise AssertionError('mixed/historical input reached tensor packing')
    monkeypatch.setattr(network, '_tensor', forbidden)
    with pytest.raises(ContractError):
        network.encode_frame(old_frame)
    with pytest.raises(ContractError):
        network.encode_history((old_frame,))
    with torch.no_grad(), pytest.raises(ContractError):
        network.step(old_frame, cache)
    with pytest.raises(ContractError):
        network((old_frame,), old_table)
    with pytest.raises(ContractError):
        network(frames, old_table)
    with pytest.raises(ContractError):
        network.dense_logits(context, cards, old_table)
    with pytest.raises(ContractError):
        network.factorized_logits(context, cards, old_table)
