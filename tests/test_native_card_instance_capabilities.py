"""Published structural support; never a parent, Source or fitting admission."""
from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_training import native_backend

ROOT = Path(__file__).parents[1]
MARKERS = {
    'formal': ('card_instance_formal_controls', 'run3-formal-parent-v1;declaration1;stop1'),
    'final64': ('card_instance_final_validation', 'collection5-final64;journal6;codec2'),
    'history': ('card_instance_original_history', 'journal6;codec2;episode-dto1;recorded16'),
    'fixture': ('card_instance_fixture_tokens', 'component-envelope1;participant-structured-v2;codec2'),
    'tree': ('card_instance_tree_raw', 'journal6;lightgbm3;inference4;codec2'),
}
SETTINGS = PreparationConfig(backend='rust', reader='participant-structured-v2')


@pytest.fixture(scope='module')
def candidate():
    # Missing optional extension may skip the module. A present extension with
    # missing/wrong markers MUST fail positive activation checks, never skip.
    return pytest.importorskip('gf_native')


def preflight(role, module=None):
    if role == 'formal':
        from gf_training.card_instance_formal import require_native_formal_controls
        module = native_backend.load_native() if module is None else module
        result = native_backend.card_instance_reader_settings(SETTINGS, module)
        require_native_formal_controls(module)
        return result
    if role == 'final64':
        from gf_training.native_card_instance import _require_final_validation_native
        return _require_final_validation_native(SETTINGS, module)
    if role == 'history':
        return native_backend.original_history_reader_settings(SETTINGS, module)
    if role == 'fixture':
        from gf_training.native_card_instance_fixture import _preflight
        module = native_backend.load_native() if module is None else module
        return _preflight(module, diagnostic=False)
    assert role == 'tree'
    return native_backend.tree_raw_reader_settings(SETTINGS, module)


@pytest.mark.parametrize('role', MARKERS)
def test_all_five_live_positive_preflights_on_actual_marked_binary(candidate, role):
    key, value = MARKERS[role]
    assert dict(candidate.build_info())[key] == value
    result = preflight(role, candidate)
    if role == 'final64':
        assert result is candidate
    elif role == 'fixture':
        assert result == native_backend.binary_identity(candidate)
    else:
        assert result == asdict(SETTINGS)
    # The public/default loader must reach the SAME actual extension as the
    # explicit path. These checks allocate no Source, arrays, model or child.
    assert native_backend.load_native() is candidate
    assert preflight(role) == result


@pytest.mark.parametrize('role', MARKERS)
@pytest.mark.parametrize('mutation', ['missing', 'wrong', 'signature', 'abi', 'method', 'cancel', 'source', 'binary'])
def test_closed_profiles_refuse_before_source_index_array_or_spawn(candidate, monkeypatch, role, mutation):
    import gf_training.native_card_instance_fixture as fixture_module
    from gf_models.structured_sealed import SealedScan

    touched = []
    def forbidden(*args, **kwargs):
        touched.append('Source/index/array/child')
        raise AssertionError('preflight allocated operational ownership')

    # Negative surface stubs carry no Source or qualification flags. Positive
    # preflights above always use the real loaded binary/identity without mocks.
    info = dict(candidate.build_info())
    attributes = {'build_info': lambda: info.items()}
    for name in ('CardInstanceNativeSession', 'CardInstanceFixtureTokens', 'StructuredJournalTokens', 'CancelHandle'):
        actual = getattr(candidate, name)
        fields = {key: staticmethod(getattr(actual, key)) for key in dir(actual)
            if not key.startswith('_') and callable(getattr(actual, key))}
        fields['__new__'] = staticmethod(forbidden)
        attributes[name] = type('UNADMITTEDCapabilitySurface', (), fields)
    module = SimpleNamespace(**attributes)
    key, _ = MARKERS[role]
    if mutation == 'missing':
        info.pop(key)
    elif mutation == 'wrong':
        info[key] = 'wrong-profile'
    elif mutation == 'signature':
        info['card_instance_signature'] = '0'*64
    elif mutation == 'abi':
        info['abi'] = 'cp313'
    elif mutation == 'method':
        cls, method = {'formal': ('CardInstanceNativeSession', 'control_summary'),
            'final64': ('CardInstanceNativeSession', 'inspect_final_validation'),
            'history': ('CardInstanceNativeSession', 'export_recorded_episode_payload'),
            'fixture': ('CardInstanceFixtureTokens', 'export_tokens'),
            'tree': ('StructuredJournalTokens', 'card_instance')}[role]
        setattr(getattr(module, cls), method, None)
    elif mutation == 'cancel':
        module.CancelHandle.cancel = None
    elif mutation == 'source':
        info['native_source_sha256'] = 'invalid-source'

    identity = copy.deepcopy(native_backend.binary_identity(candidate))
    identity['build'] = info
    if mutation == 'binary':
        identity['gil_disabled'] = True
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *_: identity)
    monkeypatch.setattr(fixture_module, 'binary_identity', lambda *_: identity)
    original_import = native_backend.importlib.import_module
    monkeypatch.setattr(native_backend.importlib, 'import_module',
        lambda name, *a, **kw: module if name == 'gf_native' else original_import(name, *a, **kw))
    monkeypatch.setattr(SealedScan, '__init__', forbidden)
    for name in ('empty', 'zeros', 'asarray'):
        monkeypatch.setattr(np, name, forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    with pytest.raises(ContractError):
        preflight(role)
    assert touched == []


def test_final64_marker_cannot_replace_formal_control_support(candidate, monkeypatch):
    from gf_training.native_card_instance import _require_final_validation_native
    info = dict(candidate.build_info())
    info.pop('card_instance_formal_controls')
    module = SimpleNamespace(build_info=lambda: info.items(), CardInstanceNativeSession=candidate.CardInstanceNativeSession,
        CancelHandle=candidate.CancelHandle)
    identity = native_backend.binary_identity(candidate)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *_: {**identity, 'build': info})
    with pytest.raises(ContractError, match='formal-control marker is insufficient'):
        _require_final_validation_native(SETTINGS, module)


def test_actual_preserved_old_binary_refuses_all_five_before_operational_work(tmp_path):
    preserved_binary = (ROOT / 'runs/assessments/card-instance-final-acceptance-20261108'
        / 'native-install-rollback-v1/native/target/release/libgf_native.so')
    if not preserved_binary.is_file():
        pytest.skip('optional preserved pre-capability native binary is unavailable')
    code = r'''
import importlib.util,json,sys
from pathlib import Path
source=Path(sys.argv[1]);sys.path.insert(0,str(source/'src'))
spec=importlib.util.spec_from_file_location('gf_native',sys.argv[2])
gf_native=importlib.util.module_from_spec(spec);sys.modules['gf_native']=gf_native;spec.loader.exec_module(gf_native)
from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_training import native_backend
from gf_training.card_instance_formal import require_native_formal_controls
from gf_training.native_card_instance import _require_final_validation_native
from gf_training.native_card_instance_fixture import _preflight
identity=native_backend.binary_identity(gf_native)
assert identity['extension_sha256']=='d093ccaa68e8359b9f1a9619754a65dc733747aa31a8a1578c9a299f051b1e5e'
assert 'card_instance_formal_controls' not in identity['build']
settings=PreparationConfig(backend='rust',reader='participant-structured-v2')
touched=[]
def forbidden(*a,**k):
 touched.append('operational allocation');raise AssertionError('old binary allocated')
for name in ('CardInstanceNativeSession','CardInstanceFixtureTokens','StructuredJournalTokens','NativeSession','TreeDatasetSession'):
 setattr(gf_native,name,forbidden)
import numpy as np,subprocess
from gf_models.structured_sealed import SealedScan
SealedScan.__init__=forbidden
np.empty=np.zeros=np.asarray=forbidden
subprocess.Popen=forbidden
results=[]
for role in ('formal','final64','history','fixture','tree'):
 try:
  if role=='formal':
   native_backend.card_instance_reader_settings(settings,gf_native);require_native_formal_controls(gf_native)
  elif role=='final64':_require_final_validation_native(settings,gf_native)
  elif role=='history':native_backend.original_history_reader_settings(settings,gf_native)
  elif role=='fixture':_preflight(gf_native,diagnostic=False)
  else:native_backend.tree_raw_reader_settings(settings,gf_native)
 except ContractError:results.append(role)
 else:raise AssertionError('old binary preflight succeeded: '+role)
assert not touched and len(results)==5
print(json.dumps({'identity':identity,'refused':results,'operational_allocations':touched},sort_keys=True))
'''
    python = ROOT / '.venv/bin/python'
    result = subprocess.run([str(python), '-I', '-c', code, str(ROOT),
        str(preserved_binary)], check=False, capture_output=True,
        text=True, cwd=tmp_path, timeout=30)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert len(evidence['refused']) == 5 and evidence['operational_allocations'] == []
    assert not list(tmp_path.iterdir())
