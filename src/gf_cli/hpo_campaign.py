"""Durable local Optuna proposals; concrete workflow remains outside the sampler."""
from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from gf_cli.hpo_config import HPOCampaignConfig, LoadedHPOConfig, joint_resource_admission
from gf_cli.hpo_storage import CampaignStore
from gf_common.card_instance_collection import HPO_EVALUATION_STREAM_VERSION
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.group_pack import parse_json
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.hpo_config import model_config_v4, model_defaults_v4
from gf_training.hpo_config import (
    MAX_VALIDATION_STATES,
    CoupledParameter,
    EffectiveSettings,
    HPOTrainingConfig,
    SearchParameter,
    _representatives,
    effective_settings,
    hpo_training_config,
)

OPTUNA_VERSION = '5.0.0'
SAMPLER_SEED_VERSION = 'hpo-proposal-seed-v1'
_CAMPAIGN_KEYS = {'campaign_id', 'loaded_config_sha256', 'study_identity', 'conditions', 'conditions_sha256',
    'canonical_control_identity', 'optuna_version', 'schema_versions', 'created_unix', 'effective_config', 'config_inputs'}
_INTENT_KEYS = {'campaign_id', 'proposal_no', 'slot', 'proposal_kind', 'sampler_seed', 'completed_history_sha256',
    'expected_trial_number', 'requested_settings_sha256', 'fixed_params'}
_PROPOSAL_KEYS = {'intent_sha256', 'trial_number', 'raw_params', 'distributions', 'effective_model',
    'effective_training', 'effective_settings_sha256', 'resource_admission'}
_TELL_KEYS = {'proposal_sha256', 'trial_number', 'params', 'distributions', 'state', 'value', 'evaluation_receipt_sha256', 'reason'}


def _record(kind: str, **values: Any) -> Json:
    return {'schema_version': 1, 'kind': kind, **values}


def _checked(value: Json, kind: str, keys: set[str]) -> Json:
    closed(value, {'schema_version', 'kind'} | keys)
    integer(value['schema_version'], 1, 1)
    if value['kind'] != kind:
        raise ContractError('unexpected HPO campaign record kind')
    return value


def _optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise ContractError('HPO run requires the optuna5 hpo extra') from error
    if optuna.__version__ != OPTUNA_VERSION:
        raise ContractError('HPO requires the pinned Optuna5.0.0 semantics')
    return optuna


def proposal_seed(seed: int, number: int) -> int:
    integer(seed, 0, 2**32 - 1)
    integer(number, 0, 2**63 - 1)
    return int(digest({'recipe': SAMPLER_SEED_VERSION, 'campaign_seed': seed, 'proposal_no': number})[:8], 16)


def canonical_control(config: HPOCampaignConfig) -> EffectiveSettings:
    return effective_settings(model_defaults_v4(config.model_id), HPOTrainingConfig(), {}, layout=LAYOUT)


def scientific_study_identity(loaded: LoadedHPOConfig, conditions: Json) -> str:
    return digest({'recipe': 'hpo-study-identity-v1', 'config_identity': loaded.config.study_identity,
        'conditions_sha256': digest(conditions), 'canonical_control': canonical_control(loaded.config).identity,
        'inference': asdict(loaded.config.inference), 'evaluation_streams': HPO_EVALUATION_STREAM_VERSION})


def initialize_campaign(store: CampaignStore, loaded: LoadedHPOConfig, conditions: Json) -> Json:
    if type(loaded) is not LoadedHPOConfig or type(conditions) is not dict:
        raise ContractError('closed loaded campaign and measured conditions required')
    if (canonical(asdict(loaded.config)) != canonical(loaded.effective)
            or loaded.sha256 != digest({'campaign': loaded.effective, 'base': loaded.base.sha256})):
        raise ContractError('campaign differs from its actual loaded effective configuration')
    control = canonical_control(loaded.config)
    # The required existing-recipe contrast must fit before any proposal executes.
    joint_resource_admission(replace(loaded.config, model=control.model, training=control.training), layout=LAYOUT)
    condition_hash = digest(conditions)
    identity = scientific_study_identity(loaded, conditions)
    manifest = _record('hpo-campaign', campaign_id=str(uuid.uuid4()), loaded_config_sha256=loaded.sha256,
        study_identity=identity, conditions=conditions, conditions_sha256=condition_hash,
        canonical_control_identity=control.identity, optuna_version=OPTUNA_VERSION,
        schema_versions={'campaign': 1, 'sampler_seed': SAMPLER_SEED_VERSION, 'evaluation_streams': HPO_EVALUATION_STREAM_VERSION},
        created_unix=time.time(), effective_config=loaded.effective, config_inputs=list(loaded.inputs))
    store.put('campaign.json', manifest)
    return manifest


def _path(root: Json, path: str) -> Any:
    value: Any = root
    for part in path.split('.'):
        value = value[part]
    return value


def _active(when: tuple[tuple[str, tuple[Any, ...]], ...], values: Json) -> bool:
    return all(any(canonical(values.get(path)) == canonical(choice) for choice in choices) for path, choices in when)


def encode_settings(config: HPOCampaignConfig, settings: EffectiveSettings) -> Json | None:
    """Invert only the declared space, checking normalized effective identity."""
    desired = {'model': asdict(settings.model), 'training': asdict(settings.training)}
    entries: tuple[SearchParameter | CoupledParameter, ...] = (*config.space.parameters, *config.space.coupled)
    conditions: dict[str, tuple[Any, ...]] = {}
    for entry in entries:
        for path, choices in entry.when:
            conditions[path] = (*conditions.get(path, ()), *choices)
    visited = 0

    def walk(index: int, values: Json, raw: Json) -> Json | None:
        nonlocal visited
        visited += 1
        if visited > MAX_VALIDATION_STATES:
            raise ContractError('settings membership exceeds admitted search-validation bound')
        if index == len(entries):
            actual = effective_settings(config.model, config.training, config.space.resolve(raw), layout=LAYOUT)
            return raw if actual.identity == settings.identity else None
        entry = entries[index]
        if not _active(entry.when, values):
            return walk(index + 1, values, raw)
        if isinstance(entry, CoupledParameter):
            target = tuple(_path(desired, path) for path in entry.paths)
            order = sorted(range(len(entry.choices)), key=lambda number: canonical(entry.choices[number]) != canonical(target))
            for number in order:
                choice = dict(zip(entry.paths, entry.choices[number], strict=True))
                result = walk(index + 1, values | choice, raw | {entry.name: number})
                if result is not None:
                    return result
            return None
        target = _path(desired, entry.path)
        if entry.kind == 'fixed':
            return walk(index + 1, values | {entry.path: entry.value}, raw)
        candidates = list(_representatives(entry, conditions.get(entry.path, ()), exact=False))
        if entry.kind == 'categorical':
            order = sorted(range(len(entry.choices)), key=lambda number: canonical(entry.choices[number]) != canonical(target))
            pairs = [(number, entry.choices[number]) for number in order]
        else:
            try:
                entry.decode(target)
                candidates.insert(0, target)
            except ContractError:
                pass
            pairs = [(candidate, candidate) for candidate in candidates]
        for encoded, decoded in pairs:
            result = walk(index + 1, values | {entry.path: decoded}, raw | {entry.path: encoded})
            if result is not None:
                return result
        return None

    return walk(0, dict(config.space.control_values), {})


def import_settings(paths: tuple[Path, ...], config: HPOCampaignConfig, *, study_identity: str) -> tuple[EffectiveSettings, ...]:
    """Import configuration only; no old state, objective or history authority."""
    result = []
    identities = set()
    limit = config.operating.parent_budget_bytes // 32
    for path in paths:
        if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
            raise ContractError('actual settings-seed file required')
        if path.stat().st_size > limit:
            raise ContractError('settings seed exceeds declared parent envelope')
        try:
            value = parse_json(path.read_bytes().rstrip(b'\n'), canonical=False, limit=limit)
        except ValueError as error:
            raise ContractError('invalid settings-only stage seed JSON') from error
        _checked(value, 'hpo-settings-seed', {'model', 'training', 'effective_settings_sha256', 'origin'})
        origin = closed(value['origin'], {'campaign_id', 'study_identity', 'proposal_sha256'})
        for item in origin.values():
            if type(item) is not str or not item:
                raise ContractError('settings origin contains invalid informational identity')
        if origin['study_identity'] == study_identity:
            raise ContractError('settings import requires a fresh different study')
        architecture = closed(value['model'], set(asdict(config.model)) - {'resources'})
        model = model_config_v4({**architecture, 'resources': asdict(config.model.resources)}, layout=LAYOUT)
        settings = effective_settings(model, hpo_training_config(value['training']), {}, layout=LAYOUT)
        if value['effective_settings_sha256'] != settings.identity or settings.identity in identities:
            raise ContractError('settings seed identity mismatch or effective duplicate')
        if encode_settings(config, settings) is None:
            raise ContractError('imported configuration is outside new declared search space')
        joint_resource_admission(replace(config, model=settings.model, training=settings.training), layout=LAYOUT)
        identities.add(settings.identity)
        result.append(settings)
    return tuple(result)


def _suggest(trial: Any, config: HPOCampaignConfig) -> Json:
    raw: Json = {}
    values = dict(config.space.control_values)
    for parameter in config.space.parameters:
        if not _active(parameter.when, values):
            continue
        if parameter.kind == 'fixed':
            value = parameter.value
        else:
            if parameter.kind == 'categorical':
                encoded = trial.suggest_categorical(parameter.path, list(range(len(parameter.choices))))
            elif parameter.kind == 'int':
                assert parameter.low is not None and parameter.high is not None and parameter.step is not None
                encoded = trial.suggest_int(parameter.path, int(parameter.low), int(parameter.high), step=int(parameter.step))
            else:
                assert parameter.low is not None and parameter.high is not None
                encoded = trial.suggest_float(parameter.path, float(parameter.low), float(parameter.high),
                    step=parameter.step, log=parameter.kind == 'log')
            raw[parameter.path] = encoded
            value = parameter.decode(encoded)
        values[parameter.path] = value
    for coupled in config.space.coupled:
        if _active(coupled.when, values):
            index = trial.suggest_categorical(coupled.name, list(range(len(coupled.choices))))
            raw[coupled.name] = index
            values.update(zip(coupled.paths, coupled.choices[index], strict=True))
    config.space.resolve(raw)
    return raw


def _distributions(trial: Any) -> Json:
    optuna = _optuna()
    return {name: json.loads(optuna.distributions.distribution_to_json(value)) for name, value in trial.distributions.items()}


def _declared_distributions(config: HPOCampaignConfig) -> Json:
    optuna = _optuna()
    values = {}
    for parameter in config.space.parameters:
        if parameter.kind == 'fixed':
            continue
        if parameter.kind == 'categorical':
            value = optuna.distributions.CategoricalDistribution(tuple(range(len(parameter.choices))))
        elif parameter.kind == 'int':
            assert parameter.low is not None and parameter.high is not None and parameter.step is not None
            value = optuna.distributions.IntDistribution(int(parameter.low), int(parameter.high), step=int(parameter.step))
        else:
            assert parameter.low is not None and parameter.high is not None
            value = optuna.distributions.FloatDistribution(float(parameter.low), float(parameter.high),
                step=parameter.step, log=parameter.kind == 'log')
        values[parameter.path] = json.loads(optuna.distributions.distribution_to_json(value))
    for coupled in config.space.coupled:
        values[coupled.name] = json.loads(optuna.distributions.distribution_to_json(
            optuna.distributions.CategoricalDistribution(tuple(range(len(coupled.choices))))))
    return values


@dataclass(frozen=True)
class Proposal:
    number: int
    slot: int | None
    trial_number: int | None
    sha256: str
    settings: EffectiveSettings
    _params: str
    _distributions: str

    @property
    def params(self) -> Json:
        return json.loads(self._params)

    @property
    def distributions(self) -> Json:
        return json.loads(self._distributions)


class CampaignStudy:
    def __init__(self, store: CampaignStore, loaded: LoadedHPOConfig, manifest: Json) -> None:
        _checked(manifest, 'hpo-campaign', _CAMPAIGN_KEYS)
        actual, self.manifest_sha = store.get('campaign.json')
        if (canonical(actual) != canonical(manifest) or manifest['loaded_config_sha256'] != loaded.sha256
                or canonical(asdict(loaded.config)) != canonical(loaded.effective)
                or loaded.sha256 != digest({'campaign': loaded.effective, 'base': loaded.base.sha256})
                or canonical(manifest['config_inputs']) != canonical(list(loaded.inputs))
                or manifest['study_identity'] != scientific_study_identity(loaded, manifest['conditions'])
                or manifest['canonical_control_identity'] != canonical_control(loaded.config).identity
                or manifest['conditions_sha256'] != digest(manifest['conditions']) or manifest['optuna_version'] != OPTUNA_VERSION):
            raise ContractError('campaign configuration/conditions/owner identity changed')
        self.store, self.loaded, self.manifest = store, loaded, actual
        self.name = 'gf-hpo-' + manifest['campaign_id'] + '-' + manifest['study_identity']

    @contextmanager
    def study(self, number: int) -> Iterator[Any]:
        optuna = _optuna()
        # Public URL construction comes from Optuna's existing SQLAlchemy
        # dependency; it preserves literal filesystem paths without URL queries.
        from sqlalchemy.engine import URL
        owner = _record('hpo-study-owner', campaign_id=self.manifest['campaign_id'],
            study_identity=self.manifest['study_identity'], campaign_directory=str(self.store.directory),
            campaign_manifest_sha256=self.manifest_sha)
        db_path = Path(self.loaded.config.storage.optuna_path)
        claim_path = db_path.parent / (db_path.name + '.hpo-study-owners') / digest(self.name)
        with CampaignStore(claim_path, mode='resume' if claim_path.exists() else 'fresh',
                max_owned_bytes=self.loaded.config.operating.parent_budget_bytes) as claim:
            claim.put('owner.json', owner)
            had_intent = self.store.exists('study.intent.json')
            self.store.put('study.intent.json', _record('hpo-study-intent', study_name=self.name, owner=owner))
            storage = optuna.storages.RDBStorage(URL.create('sqlite', database=str(db_path)).render_as_string(hide_password=False))
            try:
                sampler = optuna.samplers.TPESampler(seed=proposal_seed(self.loaded.config.sampler_seed, number),
                    n_startup_trials=self.loaded.config.startup_trials, multivariate=True, group=True, constant_liar=False)
                created = False
                try:
                    study = optuna.create_study(storage=storage, study_name=self.name, direction='maximize',
                        sampler=sampler, pruner=optuna.pruners.NopPruner(), load_if_exists=False)
                    created = True
                except optuna.exceptions.DuplicatedStudyError:
                    study = optuna.load_study(storage=storage, study_name=self.name, sampler=sampler,
                        pruner=optuna.pruners.NopPruner())
                attrs = study.user_attrs
                if not attrs and (created or had_intent and not study.get_trials()):
                    study.set_user_attr('hpo_owner', owner)
                if study.user_attrs != {'hpo_owner': owner} or study.direction.name != 'MAXIMIZE':
                    raise ContractError('Optuna study is not owned by this exact campaign/scientific identity')
                yield study
            finally:
                storage.remove_session()
                storage.engine.dispose()

    def _root(self, number: int) -> str:
        integer(number, 0, 2**63 - 1)
        return f'proposals/{number:06}'

    def _intent(self, number: int) -> tuple[Json, str]:
        value, sha = self.store.get(self._root(number) + '/intent.json')
        return _checked(value, 'hpo-proposal-intent', _INTENT_KEYS), sha

    def _history(self, study: Any) -> str:
        self._known_trials(study)
        completed = []
        for trial in study.get_trials():
            if trial.state.name == 'COMPLETE':
                from gf_cli.hpo_evaluation import verify_trial_evaluation
                intent = self._known_trials(study)[trial.number]
                proposal = self._proposal(intent['proposal_no'], intent)
                _, actual_value, actual_sha = verify_trial_evaluation(self.store, self.loaded, self.manifest, proposal,
                    deadline=time.monotonic() + self.loaded.config.run_wall_seconds)
                receipt = trial.user_attrs.get('hpo_evaluation_receipt_sha256')
                if type(receipt) is not str or receipt != actual_sha or trial.value != actual_value:
                    raise ContractError('completed sampler history lacks verified evaluation identity')
                completed.append({'number': trial.number, 'params': trial.params, 'distributions': _distributions(trial),
                    'value': trial.value, 'evaluation_receipt_sha256': receipt})
        return digest(completed)

    def _known_trials(self, study: Any) -> dict[int, Json]:
        """No external/pending trial is silently absorbed into this campaign."""
        owners = {}
        for name in self.store.records():
            if not name.startswith('proposals/') or not name.endswith('/intent.json'):
                continue
            intent, sha = self.store.get(name)
            _checked(intent, 'hpo-proposal-intent', _INTENT_KEYS)
            if (intent['campaign_id'] != self.manifest['campaign_id']
                    or name != self._root(intent['proposal_no']) + '/intent.json'):
                raise ContractError('foreign proposal intent in campaign')
            number = intent['expected_trial_number']
            if number is not None:
                integer(number, 0)
                if number in owners:
                    raise ContractError('two proposal intents claim the same Optuna trial')
                owners[number] = (intent, sha)
        declared = _declared_distributions(self.loaded.config)
        trials = study.get_trials()
        for position, trial in enumerate(trials):
            if trial.number != position or trial.number not in owners:
                raise ContractError('unexplained or missing Optuna trial history')
            if trial.state.name not in ('WAITING', 'RUNNING', 'FAIL', 'COMPLETE'):
                raise ContractError('pruned/unknown Optuna state is unsupported by the HPO contract')
            intent, sha = owners[trial.number]
            binding = {'campaign_id': self.manifest['campaign_id'], 'proposal_no': intent['proposal_no'], 'intent_sha256': sha}
            root = self._root(intent['proposal_no'])
            distributions = _distributions(trial)
            if set(trial.params) != set(distributions) or any(name not in declared
                    or canonical(value) != canonical(declared[name]) for name, value in distributions.items()):
                raise ContractError('Optuna trial contains unexplained distribution/domain')
            values = dict(self.loaded.config.space.control_values)
            consumed = set()
            entries: tuple[SearchParameter | CoupledParameter, ...] = (*self.loaded.config.space.parameters, *self.loaded.config.space.coupled)
            for entry in entries:
                if not _active(entry.when, values):
                    continue
                if isinstance(entry, SearchParameter) and entry.kind == 'fixed':
                    values[entry.path] = entry.value
                    continue
                key = entry.path if isinstance(entry, SearchParameter) else entry.name
                if key not in trial.params:
                    break
                consumed.add(key)
                if isinstance(entry, SearchParameter):
                    values[entry.path] = entry.decode(trial.params[key])
                else:
                    choice = entry.choices[integer(trial.params[key], 0, len(entry.choices) - 1)]
                    values.update(zip(entry.paths, choice, strict=True))
            if consumed != set(trial.params):
                raise ContractError('partial ask is not an actual prefix of declared active suggestions')
            if intent['fixed_params'] is not None and any(key not in intent['fixed_params']
                    or canonical(value) != canonical(intent['fixed_params'][key]) for key, value in trial.params.items()):
                raise ContractError('actual fixed suggestions differ from original enqueue intent')
            if trial.state.name == 'WAITING':
                if (trial.params or distributions or intent['fixed_params'] is None
                        or trial.user_attrs != {'hpo_intent': binding}
                        or canonical(trial.system_attrs.get('fixed_params')) != canonical(intent['fixed_params'])
                        or self.store.exists(root + '/proposal.json')):
                    raise ContractError('waiting trial differs from exact enqueue intent')
            elif self.store.exists(root + '/proposal.json'):
                proposal = self._proposal(intent['proposal_no'], intent)
                allowed = {'hpo_intent': binding, 'hpo_proposal_sha256': proposal.sha256}
                matching = (canonical(trial.params) == canonical(proposal.params)
                    and canonical(distributions) == canonical(proposal.distributions))
                # Only the crash gap after durable proposal/before attribute
                # commit may be reconciled, while the actual trial is RUNNING.
                pending_attribute = trial.state.name == 'RUNNING' and trial.user_attrs == {'hpo_intent': binding}
                tell = None
                complete_attrs = None
                if self.store.exists(root + '/tell.intent.json'):
                    tell, _ = self.store.get(root + '/tell.intent.json')
                    _checked(tell, 'hpo-tell', _TELL_KEYS)
                    if (tell['proposal_sha256'] != proposal.sha256 or tell['trial_number'] != trial.number
                            or canonical(tell['params']) != canonical(proposal.params)
                            or canonical(tell['distributions']) != canonical(proposal.distributions)):
                        raise ContractError('Optuna tell intent differs from actual proposal')
                    if tell['state'] == 'COMPLETE':
                        aggregate, aggregate_sha = self.store.get(root + '/evaluation.json')
                        if (aggregate_sha != tell['evaluation_receipt_sha256'] or aggregate.get('value') != tell['value']
                                or type(tell['value']) not in (int, float) or not math.isfinite(tell['value'])):
                            raise ContractError('completed tell lacks exact persisted finite evaluation receipt')
                        complete_attrs = allowed | {'hpo_evaluation_receipt_sha256': aggregate_sha}
                if not matching or trial.user_attrs not in (allowed, complete_attrs) and not pending_attribute:
                    raise ContractError('Optuna trial differs from actual durable proposal')
                if trial.state.name == 'COMPLETE' and (tell is None or tell['state'] != 'COMPLETE'
                        or trial.user_attrs != complete_attrs or trial.value != tell['value']):
                    raise ContractError('completed Optuna trial conflicts with exact tell intent/receipt')
                if trial.state.name == 'FAIL' and (tell is None or tell['state'] != 'FAIL' or trial.value is not None):
                    raise ContractError('failed Optuna trial lacks exact actual failure tell intent')
            elif trial.state.name not in ('RUNNING', 'FAIL') or trial.user_attrs not in ({}, {'hpo_intent': binding}):
                raise ContractError('partial ask contains unexplained state or intent attributes')
        if any(number > len(trials) for number in owners):
            raise ContractError('original Optuna trial history is missing')
        return {number: intent for number, (intent, _) in owners.items()}

    def begin(self, number: int, *, slot: int | None, kind: str, requested: EffectiveSettings | None = None) -> Json:
        if kind not in ('control', 'sample', 'seed', 'outside-control'):
            raise ContractError('unknown HPO proposal kind')
        if slot is not None:
            integer(slot, 0, self.loaded.config.trial_count - 1)
        elif kind != 'outside-control':
            raise ContractError('in-space proposal requires a bounded attempted slot')
        fixed = None if requested is None else encode_settings(self.loaded.config, requested)
        if kind in ('control', 'seed') and (requested is None or fixed is None):
            raise ContractError('fixed in-space proposal is outside the declared normalized space')
        if kind == 'outside-control' and (requested is None or fixed is not None):
            raise ContractError('separate contrast must be outside the search space')
        with self.study(number) as study:
            owners = self._known_trials(study)
            if any(trial.state.name in ('WAITING', 'RUNNING') and owners[trial.number]['proposal_no'] != number
                   for trial in study.get_trials()):
                raise ContractError('previous proposal must settle before another ask')
            expected = len(study.get_trials())
            history = self._history(study)
        value = _record('hpo-proposal-intent', campaign_id=self.manifest['campaign_id'], proposal_no=number,
            slot=slot, proposal_kind=kind, sampler_seed=proposal_seed(self.loaded.config.sampler_seed, number),
            completed_history_sha256=history, expected_trial_number=None if slot is None else expected,
            requested_settings_sha256=None if requested is None else requested.identity, fixed_params=fixed)
        self.store.put(self._root(number) + '/intent.json', value)
        return value

    def _proposal(self, number: int, intent: Json) -> Proposal:
        value, sha = self.store.get(self._root(number) + '/proposal.json')
        _checked(value, 'hpo-proposal', _PROPOSAL_KEYS)
        _, intent_sha = self._intent(number)
        if value['intent_sha256'] != intent_sha or value['trial_number'] != intent['expected_trial_number']:
            raise ContractError('saved proposal differs from original ask intent')
        model = model_config_v4(value['effective_model'], layout=LAYOUT)
        recipe = hpo_training_config(value['effective_training'])
        settings = effective_settings(model, recipe, {}, layout=LAYOUT)
        if settings.identity != value['effective_settings_sha256']:
            raise ContractError('saved proposal effective identity differs')
        if intent['slot'] is not None:
            replay = effective_settings(self.loaded.config.model, self.loaded.config.training,
                self.loaded.config.space.resolve(value['raw_params']), layout=LAYOUT)
            if canonical(asdict(replay)) != canonical(asdict(settings)):
                raise ContractError('saved effective proposal cannot be replayed from original declared space')
        return Proposal(number, intent['slot'], value['trial_number'], sha, settings,
            canonical(value['raw_params']), canonical(value['distributions']))

    def _export_settings(self, proposal: Proposal) -> None:
        model = asdict(proposal.settings.model)
        del model['resources']
        self.store.put(self._root(proposal.number) + '/settings-seed.json', _record('hpo-settings-seed',
            model=model, training=asdict(proposal.settings.training), effective_settings_sha256=proposal.settings.identity,
            origin={'campaign_id': self.manifest['campaign_id'], 'study_identity': self.manifest['study_identity'],
                'proposal_sha256': proposal.sha256}))

    def _duplicate(self, proposal: Proposal) -> bool:
        root = self._root(proposal.number)
        for name in self.store.records():
            parts = name.split('/')
            if len(parts) != 3 or parts[0] != 'proposals' or parts[2] != 'proposal.json':
                continue
            earlier = int(parts[1])
            if earlier >= proposal.number:
                continue
            intent, _ = self._intent(earlier)
            previous = self._proposal(earlier, intent)
            if previous.settings.identity == proposal.settings.identity:
                self.store.put(root + '/duplicate.json', _record('hpo-effective-duplicate',
                    proposal_sha256=proposal.sha256, original_proposal_sha256=previous.sha256))
                self.fail(proposal, 'effective-duplicate')
                return True
        return False

    def propose(self, number: int, *, slot: int | None, kind: str,
                requested: EffectiveSettings | None = None) -> Proposal | None:
        root = self._root(number)
        if not self.store.exists(root + '/intent.json'):
            self.begin(number, slot=slot, kind=kind, requested=requested)
        intent, intent_sha = self._intent(number)
        if (intent['campaign_id'] != self.manifest['campaign_id'] or intent['proposal_no'] != number
                or intent['slot'] != slot or intent['proposal_kind'] != kind
                or intent['requested_settings_sha256'] != (None if requested is None else requested.identity)):
            raise ContractError('proposal replay differs from original predeclared intent')
        if self.store.exists(root + '/generation-failure.json'):
            failure, _ = self.store.get(root + '/generation-failure.json')
            _checked(failure, 'hpo-unexecuted-proposal', {'intent_sha256', 'trial_number', 'params', 'distributions', 'reason'})
            if (failure['intent_sha256'] != intent_sha or failure['trial_number'] != intent['expected_trial_number']
                    or self.store.exists(root + '/proposal.json')):
                raise ContractError('unexecuted proposal failure conflicts with original intent/publication')
            with self.study(number) as study:
                self._known_trials(study)
                frozen = study.get_trials()[failure['trial_number']]
                if (frozen.state.name != 'FAIL' or frozen.value is not None
                        or canonical(frozen.params) != canonical(failure['params'])
                        or canonical(_distributions(frozen)) != canonical(failure['distributions'])):
                    raise ContractError('unexecuted failure differs from actual failed partial ask')
            return None
        if self.store.exists(root + '/duplicate.json'):
            duplicate, _ = self.store.get(root + '/duplicate.json')
            _checked(duplicate, 'hpo-effective-duplicate', {'proposal_sha256', 'original_proposal_sha256'})
            actual = self._proposal(number, intent)
            if duplicate['proposal_sha256'] != actual.sha256:
                raise ContractError('duplicate marker differs from actual durable proposal')
            originals = [self.store.get(name)[1] for name in self.store.records()
                if name.startswith('proposals/') and name.endswith('/proposal.json') and name < root + '/proposal.json']
            if duplicate['original_proposal_sha256'] not in originals:
                raise ContractError('duplicate marker has no original earlier proposal')
            self.fail(self._proposal(number, intent), 'effective-duplicate')
            return None
        if self.store.exists(root + '/proposal.json'):
            proposal = self._proposal(number, intent)
            if proposal.trial_number is not None:
                with self.study(number) as study:
                    self._known_trials(study)
                    frozen = study.get_trials()[proposal.trial_number]
                    if frozen.state.name == 'RUNNING' and frozen.user_attrs.get('hpo_proposal_sha256') is None:
                        _optuna().Trial(study, frozen._trial_id).set_user_attr('hpo_proposal_sha256', proposal.sha256)
            self._export_settings(proposal)
            return None if self._duplicate(proposal) else proposal
        with self.study(number) as study:
            optuna = _optuna()
            number_in_study = intent['expected_trial_number']
            trial = None
            binding = {'campaign_id': self.manifest['campaign_id'], 'proposal_no': number, 'intent_sha256': intent_sha}
            if slot is not None:
                self._known_trials(study)
                trials = study.get_trials()
                if self._history(study) != intent['completed_history_sha256'] or len(trials) > number_in_study + 1:
                    raise ContractError('proposal completed history or next trial position changed')
                if len(trials) == number_in_study + 1:
                    frozen = trials[number_in_study]
                    if frozen.state.name in ('RUNNING', 'FAIL'):
                        if frozen.user_attrs not in ({}, {'hpo_intent': binding}):
                            raise ContractError('partial ask is not bound to this campaign intent')
                        if frozen.state.name == 'RUNNING':
                            study.tell(frozen.number, state=optuna.trial.TrialState.FAIL)
                        self.store.put(root + '/generation-failure.json', _record('hpo-unexecuted-proposal',
                            intent_sha256=intent_sha, trial_number=frozen.number, params=frozen.params,
                            distributions=_distributions(frozen), reason='incomplete-proposal-generation'))
                        return None
                    if (frozen.state.name != 'WAITING' or intent['fixed_params'] is None
                            or frozen.user_attrs != {'hpo_intent': binding}
                            or canonical(frozen.system_attrs.get('fixed_params')) != canonical(intent['fixed_params'])):
                        raise ContractError('unexpected or conflicting pending Optuna trial')
                elif len(trials) != number_in_study:
                    raise ContractError('missing prior Optuna trial')
                elif intent['fixed_params'] is not None:
                    study.enqueue_trial(intent['fixed_params'], user_attrs={'hpo_intent': binding}, skip_if_exists=False)
                trial = study.ask()
                if trial.number != number_in_study:
                    raise ContractError('ask selected an unexplained Optuna trial')
                trial.set_user_attr('hpo_intent', binding)
                params = _suggest(trial, self.loaded.config)
                settings = effective_settings(self.loaded.config.model, self.loaded.config.training,
                    self.loaded.config.space.resolve(params), layout=LAYOUT)
                distributions = _distributions(trial)
            else:
                assert requested is not None
                params, distributions, settings = {}, {}, requested
            admission = joint_resource_admission(replace(self.loaded.config, model=settings.model, training=settings.training), layout=LAYOUT)
            value = _record('hpo-proposal', intent_sha256=intent_sha, trial_number=number_in_study,
                raw_params=params, distributions=distributions, effective_model=asdict(settings.model),
                effective_training=asdict(settings.training), effective_settings_sha256=settings.identity,
                resource_admission=admission)
            sha = self.store.put(root + '/proposal.json', value)
            self._export_settings(self._proposal(number, intent))
            if trial is not None:
                trial.set_user_attr('hpo_proposal_sha256', sha)
        proposal = self._proposal(number, intent)
        return None if self._duplicate(proposal) else proposal

    def fail(self, proposal: Proposal, reason: str) -> Json:
        intent, _ = self._intent(proposal.number)
        actual = self._proposal(proposal.number, intent)
        if actual != proposal or type(reason) is not str or not reason:
            raise ContractError('failure requires the exact durable proposal and reason')
        value = _record('hpo-tell', proposal_sha256=proposal.sha256, trial_number=proposal.trial_number,
            params=proposal.params, distributions=proposal.distributions, state='FAIL', value=None,
            evaluation_receipt_sha256=None, reason=reason)
        root = self._root(proposal.number)
        self.store.put(root + '/tell.intent.json', value)
        if proposal.trial_number is not None:
            with self.study(proposal.number) as study:
                trial = study.get_trials()[proposal.trial_number]
                if (canonical(trial.params) != canonical(proposal.params)
                        or canonical(_distributions(trial)) != canonical(proposal.distributions)
                        or trial.user_attrs.get('hpo_proposal_sha256') != proposal.sha256):
                    raise ContractError('Optuna tell parameters/distributions/proposal binding conflict')
                if trial.state.name == 'RUNNING':
                    study.tell(trial.number, state=_optuna().trial.TrialState.FAIL)
                elif trial.state.name != 'FAIL' or trial.value is not None:
                    raise ContractError('finished Optuna tell conflicts with exact failure receipt')
        self.store.put(root + '/tell.receipt.json', value)
        return value

    def complete(self, proposal: Proposal, *, deadline: float) -> Json:
        """There is intentionally no caller-supplied scalar/JSON completion API."""
        from gf_cli.hpo_evaluation import verify_trial_evaluation
        intent, _ = self._intent(proposal.number)
        actual = self._proposal(proposal.number, intent)
        if actual != proposal:
            raise ContractError('completion requires the exact durable proposal')
        _, score, evaluation_sha = verify_trial_evaluation(self.store, self.loaded, self.manifest, proposal, deadline=deadline)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ContractError('verified evaluation objective must be finite win/draw/loss mean')
        value = _record('hpo-tell', proposal_sha256=proposal.sha256, trial_number=proposal.trial_number,
            params=proposal.params, distributions=proposal.distributions, state='COMPLETE', value=score,
            evaluation_receipt_sha256=evaluation_sha, reason=None)
        root = self._root(proposal.number)
        self.store.put(root + '/tell.intent.json', value)
        if proposal.trial_number is not None:
            with self.study(proposal.number) as study:
                self._known_trials(study)
                trial = study.get_trials()[proposal.trial_number]
                if trial.state.name == 'RUNNING':
                    _optuna().Trial(study, trial._trial_id).set_user_attr('hpo_evaluation_receipt_sha256', evaluation_sha)
                    study.tell(trial.number, score, state=_optuna().trial.TrialState.COMPLETE)
                elif (trial.state.name != 'COMPLETE' or trial.value != score
                        or trial.user_attrs.get('hpo_evaluation_receipt_sha256') != evaluation_sha):
                    raise ContractError('finished Optuna tell conflicts with exact completed evaluation')
        self.store.put(root + '/tell.receipt.json', value)
        return value
