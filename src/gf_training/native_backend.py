"""Explicit optional native loading and identity; imports no learning runtime."""
from __future__ import annotations

import hashlib
import importlib
import sys
import sysconfig
from dataclasses import asdict
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType
from typing import Any

from gf_common.domain import ContractError, Json
from gf_common.preprocessing import PreparationConfig

TYPED_READER = 'typed-borrowed-v1'
STRUCTURED_CONTRACT = 'participant-structured-v1;journal=5;rollout=3;generation=2;model=2'
CARD_INSTANCE_CONTRACT = 'participant-structured-v2;journal6;rollout4;generation3;model3'
CARD_INSTANCE_ORIGINAL_HISTORY = 'journal6;codec2;episode-dto1;recorded16'
CARD_INSTANCE_TREE_RAW = 'journal6;lightgbm3;inference4;codec2'
CARD_INSTANCE_GROUP_PACK = 'provider1;proof2;plan4'
CARD_INSTANCE_PACK_CHECKSUM = 'pack-checksum-v1;custody2'
CARD_INSTANCE_GUARDIAN_ACTED = 'card-or-guardian-v1'
CARD_INSTANCE_CONCEALED_DRAW = 'raw-preserved;feature-omitted-v1'
_CONCEALED_DRAW_KEY = 'card_instance_concealed_draw'
CARD_INSTANCE_DEFENSE_PRIORITY = 'owner-defense-priority-v2'
CARD_INSTANCE_REFRESH_BASELINE = 'baseline1;options1;identity1'
CARD_INSTANCE_REFRESH_OPPONENTS = 'mixed16-v1;ordinary1;hpo2'
CARD_INSTANCE_HPO_UPDATE_STORAGE = 'hpo-update-zstd-v1'


def require_native_defense_priority(native: Any) -> None:
    """Require current rules before any corrected native input admission."""
    if dict(native.build_info()).get('card_instance_defense_priority') != CARD_INSTANCE_DEFENSE_PRIORITY:
        raise ContractError('native defense priority rules-v2 capability unavailable; no fallback')


def require_native_guardian_acted(native: Any) -> None:
    """Require corrected closed-union validation before active native work."""
    if dict(native.build_info()).get('card_instance_guardian_acted') != CARD_INSTANCE_GUARDIAN_ACTED:
        raise ContractError('native GuardianActed union capability unavailable; no fallback')
    require_native_defense_priority(native)


def require_native_concealed_draw(information: Json) -> None:
    if information.get(_CONCEALED_DRAW_KEY) != CARD_INSTANCE_CONCEALED_DRAW:
        raise ContractError('native concealed draw capability unavailable; no fallback')


def recorded_reader_build(state: Json) -> Json | None:
    """The state must already have passed its recorded preprocessing identity check."""
    preprocessing = state['identity']['preprocessing']
    return preprocessing['binary']['build'] if preprocessing['backend'] == 'rust' else None


def require_native_refresh_baseline(native: Any = None) -> None:
    """Require additive configured-baseline support before new Source/resource work."""
    module = load_native() if native is None else native
    if dict(module.build_info()).get('card_instance_refresh') != CARD_INSTANCE_REFRESH_BASELINE:
        raise ContractError('native refresh baseline capability unavailable; no fallback')
    require_native_guardian_acted(module)


def require_native_refresh_opponents(native: Any = None) -> None:
    """Refuse new training parents before native Source or resource admission."""
    module = load_native() if native is None else native
    require_native_refresh_baseline(module)
    if dict(module.build_info()).get('card_instance_refresh_opponents') != CARD_INSTANCE_REFRESH_OPPONENTS:
        raise ContractError('native refresh training opponents capability unavailable; no fallback')


def hpo_reader_settings(config: PreparationConfig, native: Any = None, *, packed: bool = False,
                        resident: bool = False, require_update_diagnostics: bool = False,
                        require_update_storage: bool = False, recorded_build: Json | None = None) -> Json:
    """Additive HPO preflight before Source, arrays, collection or process work."""
    from gf_common.hpo import NATIVE_CAPABILITY, UPDATE_DIAGNOSTICS_CAPABILITY
    module = load_native() if native is None else native
    information = dict(module.build_info())
    if information.get('card_instance_hpo') != NATIVE_CAPABILITY:
        raise ContractError('native HPO original/model4/selection support unavailable; no fallback')
    if (require_update_diagnostics
            and information.get('card_instance_hpo_update_diagnostics') != UPDATE_DIAGNOSTICS_CAPABILITY):
        raise ContractError('native HPO update diagnostics capability unavailable; no fallback')
    if (require_update_storage
            and information.get('card_instance_hpo_update_storage') != CARD_INSTANCE_HPO_UPDATE_STORAGE):
        raise ContractError('native HPO update storage capability unavailable; no fallback')
    if resident:
        from gf_common.hpo_resident import RESIDENT_CAPABILITY
        if information.get('card_instance_hpo_resident') != RESIDENT_CAPABILITY:
            raise ContractError('native resident HPO original/retirement/cache support unavailable; no fallback')
    result = card_instance_reader_settings(config, module, recorded_build=recorded_build)
    if packed:
        group_pack_reader_settings(config, module, recorded_build=recorded_build)
        require_pack_checksum(module)
    cls = getattr(module, 'CardInstanceNativeSession', None)
    methods = ('control_summary', 'authorize', 'authorize_hpo', 'numeric_used_bytes', 'export_cohort', 'export_recorded_cohort',
        'bind_numeric', 'retire_payload', 'begin_update', 'prepare_commit', 'finalize_read', 'recheck', 'close')
    if any(not callable(getattr(cls, name, None)) for name in methods):
        raise ContractError('native HPO shared session ABI incomplete; no fallback')
    return result

def require_pack_checksum(native: Any) -> None:
    if dict(native.build_info()).get('card_instance_pack_checksum') != CARD_INSTANCE_PACK_CHECKSUM:
        raise ContractError('native Pack checksum policy/custody2 capability unavailable; no fallback')



def group_pack_reader_settings(config: PreparationConfig, native: Any = None, *,
                               recorded_build: Json | None = None) -> Json:
    """Require the separately accepted byte-provider ABI before packed native I/O."""
    import unicodedata
    if unicodedata.unidata_version != '16.0.0':
        raise ContractError('packed native names require Unicode16.0.0')
    module = native or load_native()
    require_pack_checksum(module)
    result = card_instance_reader_settings(config, module, recorded_build=recorded_build)
    if dict(module.build_info()).get('card_instance_group_pack') != CARD_INSTANCE_GROUP_PACK:
        raise ContractError('native group Pack capability unavailable; no fallback')
    if (not callable(getattr(module.CardInstanceNativeSession, 'from_artifacts', None))
            or not callable(getattr(getattr(module, 'StructuredJournalTokens', None), 'card_instance_artifacts', None))):
        raise ContractError('native group Pack provider/RAW ABI incomplete; no fallback')
    return result



def tree_raw_reader_settings(config: PreparationConfig, native: Any = None) -> Json:
    """RAW capability only; never parent, Dataset, fitting or PPO permission."""
    module = native or load_native()
    result = card_instance_reader_settings(config, module)
    info = dict(module.build_info())
    if info.get('card_instance_tree_raw') != CARD_INSTANCE_TREE_RAW:
        raise ContractError('native Tree3 RAW capability unavailable; no fallback')
    if (info.get('abi') != 'cp314-gil' or info.get('recipe_identity') != 'native-preprocessing-v1'
            or info.get('parser_contract') != 'serde-json-v2'
            or info.get('json_memory_policy') != 'json-document-32x-plus64kib-v1'):
        raise ContractError('native Tree3 RAW ABI/parser contract differs')
    reader = getattr(module, 'StructuredJournalTokens', None)
    required = ('card_instance', 'diagnostic', 'recheck', 'close', 'cancel_handle',
        'used_bytes', 'peak_bytes', 'control_reserved_bytes')
    if (not callable(reader) or any(not callable(getattr(reader, name, None)) for name in required)):
        raise ContractError('native Tree3 corrected RAW ABI unavailable; no fallback')
    identity = binary_identity(module)
    if identity['build'] != info or identity['gil_disabled'] is not False or identity['python_abi'] != 'cpython-314':
        raise ContractError('native Tree3 RAW source/binary/ABI differs')
    return result

def original_history_reader_settings(config: PreparationConfig, native: Any = None, *,
                                     recorded_build: Json | None = None,
                                     _defer_draw_capability: bool = False) -> Json:
    """Closed original-history capability; class/summary alone grants no Source."""
    module = native or load_native()
    result = card_instance_reader_settings(config, module, recorded_build=recorded_build,
        _defer_draw_capability=_defer_draw_capability)
    info = dict(module.build_info())
    if info.get('card_instance_original_history') != CARD_INSTANCE_ORIGINAL_HISTORY:
        raise ContractError('native original-history capability unavailable; no fallback')
    cls = getattr(module, 'CardInstanceNativeSession', None)
    if not callable(getattr(cls, 'export_recorded_episode_payload', None)):
        raise ContractError('native original-history episode ABI unavailable; no fallback')
    identity = binary_identity(module)
    if identity['build'] != info or identity['gil_disabled'] is not False:
        raise ContractError('native original-history source/binary/ABI differs')
    return result


def reader_settings(config: PreparationConfig, native: ModuleType | None = None) -> Json:
    """Closed reader preflight before handles/source owners; old wheels keep old DTOs."""
    if config.reader == 'participant-structured-v2':
        return card_instance_reader_settings(config, native)
    if config.reader not in ('legacy', TYPED_READER, 'structured-v1'):
        raise ContractError('unsupported preprocessing reader')
    if config.backend != 'rust':
        if config.reader != 'legacy':
            raise ContractError('typed preprocessing reader requires the Rust backend')
        raise ContractError('native preprocessing settings require the Rust backend')
    module = native or load_native()
    information = dict(getattr(module, 'build_info', lambda: ())())
    if config.reader == 'structured-v1':
        from gf_common.structured_tokens import signature
        if (information.get('structured_preprocessing_contract') != STRUCTURED_CONTRACT
                or information.get('structured_default_signature') != signature(128)):
            raise ContractError('native binary does not support structured-v1 contract/signature')
        return asdict(config)
    capable = information.get('typed_reader_contract') == TYPED_READER
    if config.reader == TYPED_READER and not capable:
        raise ContractError('native binary does not support typed-borrowed-v1')
    settings = asdict(config)
    if not capable and config.reader == 'legacy':
        settings.pop('reader')
    return settings


def load_native() -> ModuleType:
    """Fail before source/collection work when the selected backend is absent."""
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 14)
            or sysconfig.get_config_var('Py_GIL_DISABLED')):
        raise ContractError('native preprocessing requires CPython 3.14 with the GIL')
    try:
        native = importlib.import_module('gf_native')
    except ImportError as exc:
        raise ContractError('native preprocessing requires the optional native package') from exc
    import numpy as np

    if np.__version__ != '2.5.4':
        raise ContractError('native preprocessing requires NumPy 2.5.4')
    information = dict(native.build_info())
    if information.get('abi') != 'cp314-gil' or information.get('recipe_identity') != 'native-preprocessing-v1':
        raise ContractError('incompatible native preprocessing ABI/recipe')
    if (information.get('parser_contract') != 'serde-json-v2'
            or information.get('json_memory_policy') != 'json-document-32x-plus64kib-v1'):
        raise ContractError('incompatible native JSON parser/resource contract')
    return native


def binary_identity(native: ModuleType | None = None) -> Json:
    """Hash the extension that was actually loaded, independently of source Git."""
    module = native or load_native()
    module_name: Any = getattr(module.build_info, '__module__', None)
    binary_module = importlib.import_module(module_name) if isinstance(module_name, str) else module
    source: Any = binary_module.__file__
    if not isinstance(source, str):
        raise ContractError('native extension has no loaded binary path')
    path = Path(source).resolve(strict=True)
    if not any(path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise ContractError('native identity must identify the loaded extension binary')
    before = path.stat()
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ContractError('native extension changed while computing its identity')
    return {'backend': 'native', 'extension_path': str(path), 'extension_sha256': checksum,
            'build': dict(module.build_info()), 'python_abi': sys.implementation.cache_tag,
            'gil_disabled': bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}


def require_ppo_model(model_id: str, native: ModuleType) -> None:
    """Pre-collection model capability; old wheels advertise only historical MLP."""
    from gf_models.ppo_identity import PPO_MODELS
    if model_id not in PPO_MODELS:
        raise ContractError('unsupported PPO model identity')
    build_info = getattr(native, 'build_info', None)
    if not callable(build_info):
        raise ContractError('native PPO preprocessing implementation is incomplete')
    information = dict(build_info())
    if 'ppo_policy_models' not in information:
        supported = ['mlp']
    else:
        marker = information['ppo_policy_models']
        if not isinstance(marker, str):
            raise ContractError('malformed native PPO model capability')
        supported = marker.split(',')
        if not supported or len(set(supported)) != len(supported) or any(value not in PPO_MODELS for value in supported):
            raise ContractError('malformed native PPO model capability')
    if model_id not in supported:
        raise ContractError(f'native binary does not support PPO model {model_id}')


def card_instance_reader_settings(config: PreparationConfig, native: ModuleType | None = None, *,
                                  recorded_build: Json | None = None,
                                  _defer_draw_capability: bool = False) -> Json:
    """Public corrected preflight; an additive class alone is never capability."""
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_common.structured_tokens import signature
    from gf_models.structured_sealed import checksum
    if config.backend != 'rust' or config.reader != 'participant-structured-v2':
        raise ContractError('explicit corrected native reader required')
    module = native or load_native()
    information = dict(module.build_info())
    if (information.get('card_instance_contract') != CARD_INSTANCE_CONTRACT
            or information.get('card_instance_signature') != signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            or information.get('card_instance_models') != 'mlp,gru,transformer'):
        raise ContractError('native binary lacks accepted card-instance capability/signature/models')
    require_native_guardian_acted(module)
    if ((recorded_build is None and not _defer_draw_capability)
            or (recorded_build is not None and _CONCEALED_DRAW_KEY in recorded_build)
            or _CONCEALED_DRAW_KEY in information):
        require_native_concealed_draw(information)
    checksum(information.get('native_source_sha256'))
    session = getattr(module, 'CardInstanceNativeSession', None)
    required = ('control_summary', 'original_config_inputs', 'authorize', 'export_cohort', 'retire_payload',
        'bind_numeric', 'begin_update', 'prepare_commit', 'finalize_read', 'recheck', 'close', 'used_bytes', 'peak_bytes', 'cancel_handle', 'control_reserved_bytes')
    if (not callable(session) or not callable(getattr(module, 'CancelHandle', None))
            or not callable(getattr(module.CancelHandle, 'cancel', None))
            or any(not callable(getattr(session, name, None)) for name in required)):
        raise ContractError('native corrected class/cancellation ABI is incomplete')
    identity = binary_identity(module)
    if identity['build'] != information or identity['gil_disabled'] is not False:
        raise ContractError('native corrected loaded binary/source identity differs')
    return asdict(config)


def recorded_card_instance_preprocessing(value: Json, *, binding: Json) -> None:
    """Validate recorded identity only; never import/load a historical extension."""
    from gf_common.card_instance import parse_card_instance_contract
    from gf_common.domain import closed
    from gf_models.structured_sealed import checksum
    closed(value, {'backend', 'reader_contract', 'implementation', 'binary'} | set(binding))
    if (value['backend'] != 'rust' or value['reader_contract'] != 'participant-structured-v2'
            or value['implementation'] != 'rust-player-card-refs-v2'
            or any(value[key] != expected for key, expected in binding.items())):
        raise ContractError('recorded corrected native preprocessing context differs')
    binary = closed(value['binary'], {'backend', 'extension_path', 'extension_sha256', 'build', 'python_abi', 'gil_disabled'})
    if (binary['backend'] != 'native' or type(binary['extension_path']) is not str
            or not Path(binary['extension_path']).is_absolute() or binary['python_abi'] != 'cpython-314'
            or binary['gil_disabled'] is not False or type(binary['build']) is not dict):
        raise ContractError('recorded corrected native binary identity differs')
    checksum(binary['extension_sha256'])
    info = binary['build']
    require_native_defense_v3(info, parse_card_instance_contract(binding['action_contract']))
    if (info.get('card_instance_contract') != CARD_INSTANCE_CONTRACT
            or info.get('card_instance_signature') != binding['layout_signature']
            or info.get('card_instance_models') != 'mlp,gru,transformer' or info.get('abi') != 'cp314-gil'):
        raise ContractError('recorded native lacks corrected capability/signature/models')
    checksum(info.get('native_source_sha256'))
    if _CONCEALED_DRAW_KEY in info:
        require_native_concealed_draw(info)


def require_native_defense_v3(build: Json, action_contract: Any) -> None:
    from gf_common.card_instance import DEFENSE_V3_PROFILE, CardInstanceContract

    if (type(action_contract) is CardInstanceContract and action_contract.requires_direct_defense
            and build.get('card_instance_defense_v3') != DEFENSE_V3_PROFILE):
        raise ContractError('Simulator v3 defense profile requires a matching native runtime wheel')
