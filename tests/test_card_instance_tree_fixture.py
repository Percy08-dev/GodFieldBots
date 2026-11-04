"""Actual initial-weight component originals; these tests run no games or fitting."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gf_common import card_instance_fixture as fixture
from gf_common.domain import ContractError, canonical, digest
from gf_common.preprocessing import PreparationConfig
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_tree_config import ConfiguredTreeModelV3
from gf_models.structured_tree_encoder import PreparedTreeFrameV3
from gf_training.card_instance_tree_data import (
    VerifiedFixtureTreeDataset3,
    build_fixture_dataset,
    inspect_fixture_dataset,
)
from gf_training.card_instance_tree_fixture import TreeFixtureSource
from tests.card_instance_tree_fixture_support import construct, rehash_originals, write
from tests.test_card_instance_tree_encoder import reference

SETTINGS = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=1024,
                             wall_seconds=120, cleanup_seconds=5)
LIMIT = 1024**3


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    pytest.importorskip('torch')
    root = tmp_path_factory.mktemp('actual-tree-fixture')
    return construct(root/'originals')


def source(path, sha, **options):
    return TreeFixtureSource(path, sha, SETTINGS, max_owned_bytes=LIMIT,
                             deadline=time.monotonic()+90, **options)


def copied(originals, tmp_path):
    path, _ = originals
    directory = tmp_path/'copy'
    shutil.copytree(path.parent.parent, directory)
    new = directory/'originals'
    raw = json.loads((new/'manifest.json').read_text())
    for policy in raw['policies']:
        policy['directory'] = str(directory/Path(policy['directory']).name)
    raw['content_sha256'] = digest({k: v for k, v in raw.items() if k != 'content_sha256'})
    sha = write(new/'manifest.json', raw)
    return new/'manifest.json', sha


def exhaust(owner):
    count = 0
    for sample in owner.selected_inputs():
        count += 1
        del sample
    return count


def test_actual_all_populations_complete_roles_labels_original_rows_and_zero_endpoints(originals):
    path, sha = originals
    labels, kinds, viewers, learned = set(), set(), set(), set()
    counts = {'train': 0, 'validation': 0}
    with source(path, sha) as owner:
        for sample in owner.selected_inputs():
            counts[sample.split] += 1
            labels.add(sample.label); viewers.add(sample.viewer)
            kinds.update(LAYOUT.action_kinds[value-2] for value in sample.candidates.kinds)
            if all(cats[7] == 3 for cats, mask in zip(sample.tokens.card_categories, sample.tokens.card_mask, strict=True) if mask):
                learned.add(sum(sample.tokens.card_mask))
            prepared = PreparedTreeFrameV3.create(sample.tokens, ConfiguredTreeModelV3())
            full = np.concatenate([block for _, block in prepared.iter_chunks(sample.candidates)])
            selected = next(prepared.iter_chunks(sample.selected_tokens()))[1]
            assert selected.shape == (1, 2810) and np.isfinite(selected).all()
            assert selected[0].tobytes() == full[sample.selected_index].tobytes()
            assert selected[0].tobytes() == reference(sample.tokens, sample.candidates, sample.selected_index, 18).tobytes()
            assert sample.label in (-1., 0., 1.)
        proof = owner.verify_complete()
        assert len(proof['cohorts']) == len(proof['policies']) == 9
        assert [p['construction_seed'] for p in proof['populations']] == [17, 29, 43]
        assert proof['counts']['train'] == {'games': 96, 'participants': 192, 'selected_decisions': 96, 'zero_endpoint_games': 6}
        assert proof['counts']['validation'] == {'games': 192, 'participants': 384, 'selected_decisions': 192, 'zero_endpoint_games': 3}
        assert proof['origin'] == fixture.origin()
        assert proof['construction_cost']['scope'] == 'fixture-construction-only'
        assert proof['construction_cost']['seconds'] > 0
        assert all(not (Path(policy['directory'])/'resume.json').exists() for policy in proof['policies'])
    assert counts == {'train': 96, 'validation': 192}
    assert labels == {-1., 0., 1.} and viewers == {1, 2} and learned == {7, 12, 18}
    assert kinds == set(LAYOUT.action_kinds)
    with pytest.raises(ContractError, match='active concrete'):
        owner.verify_complete()


def test_dataset3_manifest_inspection_selected_reconstruction_lifetime_and_no_optional_imports(originals, tmp_path):
    path, sha = originals
    output = tmp_path/'dataset'
    manifest = build_fixture_dataset(path, sha, output, SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()+90)
    dataset_sha = hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    assert manifest['schema_version'] == 3 and manifest['origin'] == fixture.origin()
    assert 'fixture' in manifest['kind'] and 'shared_generation' not in manifest
    assert inspect_fixture_dataset(output/'manifest.json', expected_digest=dataset_sha, settings=SETTINGS,
                                   max_owned_bytes=LIMIT, deadline=time.monotonic()+90) == manifest
    with VerifiedFixtureTreeDataset3(output/'manifest.json', dataset_sha, SETTINGS, max_owned_bytes=LIMIT,
                                     deadline=time.monotonic()+90) as owner:
        count = 0
        for split, sample in owner.selected_inputs():
            assert split == sample.split
            count += 1
        assert count == 288
        owner.verify_complete()
        owner.finalize(); owner.close()
        # Once all reportable guards finish, memory release performs no filesystem work.
        owner._source._scan.finalize = lambda: pytest.fail('release must be memory-only')
        owner._index.finalize = lambda: pytest.fail('release must be memory-only')
        owner.release()
        with pytest.raises(ContractError):
            list(owner.selected_inputs())
    code = '''import sys,time,importlib.metadata
original_version=importlib.metadata.version
def version(name):
    if name in ('torch','lightgbm','pyarrow'):raise AssertionError('active optional package query')
    return original_version(name)
importlib.metadata.version=version
from pathlib import Path
from gf_common.preprocessing import PreparationConfig
from gf_training.card_instance_tree_data import inspect_fixture_dataset
inspect_fixture_dataset(Path(sys.argv[1]),expected_digest=sys.argv[2],settings=PreparationConfig(reader='participant-structured-v2',memory_budget_mib=1024),max_owned_bytes=1024**3,deadline=time.monotonic()+60)
assert not ({'torch','lightgbm','pyarrow','gf_native'} & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-c', code, str(output/'manifest.json'), dataset_sha], check=True)


@pytest.mark.parametrize('mutation', ['viewer', 'role', 'policy', 'split', 'outcome', 'declaration', 'sequence', 'prefix', 'selected', 'numeric', 'stale_hints', 'foreign_columns', 'interval'])
def test_resealed_original_mutations_do_not_gain_fixture_source_authority(originals, tmp_path, mutation):
    path, sha = copied(originals, tmp_path)
    raw = json.loads(path.read_text())
    record = raw['cohorts'][0]['episodes'][1]
    episode_path = path.parent/record['file']
    episode = json.loads(episode_path.read_text())
    if mutation == 'viewer':
        episode['viewer'] = 3-episode['viewer']
    elif mutation == 'role':
        episode['roles'].reverse()
    elif mutation == 'policy':
        episode['selected_policy_id'] = raw['policies'][1]['policy_id']
    elif mutation == 'split':
        episode['split'] = 'validation'
    elif mutation == 'outcome':
        episode['outcome'] = {'kind': 'win', 'winner': 0}
    elif mutation == 'declaration':
        episode['declaration_sha256'] = '0'*64
    elif mutation == 'sequence':
        episode['SequenceOptions'] = {'oracle': True}
    elif mutation == 'prefix':
        episode['events'].append({'type': 'safe', 'player': episode['viewer']})
    elif mutation == 'stale_hints':
        episode['decisions'][0]['frame']['payload']['action_options']['state_revision'] = 1
    elif mutation == 'foreign_columns':
        episode['decisions'][0]['columns']['frame_hash'] = '0'*64
    elif mutation == 'interval':
        episode['decisions'][1]['frame']['interval_start'] = 0
    elif mutation == 'selected':
        episode['decisions'][0]['selected_index'] = 2
    else:
        file = path.parent/record['columns'][0]
        payload = bytearray(file.read_bytes()); payload[0] ^= 1; file.write_bytes(payload)
    write(episode_path, episode)
    sha = rehash_originals(path.parent)
    with pytest.raises(ContractError), source(path, sha) as owner:
        exhaust(owner)


@pytest.mark.parametrize('mutation', ['missing', 'added', 'symlink', 'policy_bytes', 'wrong_cohort', 'qualified'])
def test_original_complete_inventory_and_policy_origin_refusals(originals, tmp_path, mutation):
    path, sha = copied(originals, tmp_path)
    raw = json.loads(path.read_text())
    item = path.parent/raw['cohorts'][0]['episodes'][1]['file']
    if mutation == 'missing':
        item.unlink()
    elif mutation == 'added':
        (path.parent/'extra.json').write_text('{}')
    elif mutation == 'symlink':
        target = tmp_path/'foreign'; item.rename(target); item.symlink_to(target)
    elif mutation == 'policy_bytes':
        policy = raw['policies'][0]
        shard = next(file for file in policy['files'] if file.endswith('.npz'))
        with (Path(policy['directory'])/shard).open('ab') as output:
            output.write(b'changed')
    elif mutation == 'wrong_cohort':
        raw['cohorts'][0], raw['cohorts'][1] = raw['cohorts'][1], raw['cohorts'][0]
    else:
        raw['origin']['purpose'] = 'qualified_game'; raw['origin']['qualified'] = True
    if mutation in ('wrong_cohort', 'qualified'):
        raw['content_sha256'] = digest({k: v for k, v in raw.items() if k != 'content_sha256'})
        sha = write(path, raw)
    with pytest.raises((ContractError, OSError)), source(path, sha) as owner:
        exhaust(owner)


def test_snapshots_detect_late_source_and_numeric_changes_and_publication_guard(originals, tmp_path):
    path, sha = copied(originals, tmp_path)
    with pytest.raises(ContractError), source(path, sha) as owner:
        exhaust(owner)
        file = path.parent/owner.manifest['cohorts'][0]['episodes'][1]['columns'][0]
        with file.open('ab') as output:
            output.write(b'late')
        owner.verify_complete()


def test_output_inside_originals_overbudget_sink_and_partial_lifetimes_refuse(originals, tmp_path, monkeypatch):
    path, sha = originals
    for output in (path.parent/'forbidden', Path(json.loads(path.read_text())['policies'][0]['directory'])/'forbidden'):
        with pytest.raises(ContractError, match='outside'):
            build_fixture_dataset(path, sha, output, SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()+90)
        assert not output.exists()
    with pytest.raises(ContractError):
        TreeFixtureSource(path, sha, SETTINGS, max_owned_bytes=1024, deadline=time.monotonic()+90).__enter__()
    with pytest.raises(ContractError), source(path, sha) as owner:
        owner.reserve_sink(LIMIT)
    with pytest.raises(ContractError, match='must be consumed'), source(path, sha):
        pass
    with pytest.raises(ContractError, match='scratch'):
        build_fixture_dataset(path, sha, tmp_path/'tiny', SETTINGS, max_owned_bytes=LIMIT,
                              deadline=time.monotonic()+90, max_scratch_bytes=1)
    assert not (tmp_path/'tiny').exists()
    with source(path, sha) as owner:
        with pytest.raises(AttributeError):
            owner.path = tmp_path/'other'
        with pytest.raises(ContractError):
            owner.release()
        exhaust(owner)
        owner.finalize(); owner.close()
        monkeypatch.setattr(owner._scan, 'finalize', lambda: pytest.fail('release may not inspect files'))
        owner.release()


def test_pure_outcome_and_preconstruction_recipe_reject_zero_based_or_crossed_membership(originals):
    path, _ = originals
    frozen = json.loads((path.parent/'declaration.json').read_text())
    assert fixture.label({'kind': 'win', 'winner': 1}, 1) == 1
    assert fixture.label({'kind': 'win', 'winner': 1}, 2) == -1
    for winner in (0, 3, True):
        with pytest.raises(ContractError):
            fixture.label({'kind': 'win', 'winner': winner}, 1)
    bad = json.loads(canonical(frozen)); bad.pop('content_sha256')
    bad['cohorts'][0]['episodes'].pop()
    with pytest.raises(ContractError):
        fixture.freeze_declaration(bad)
    # No declaration may derive identity from omitted current-default context.
    for field in ('input_binding', 'layout_signature', 'label_recipe', 'construction_recipe'):
        bad = dict(frozen); del bad[field]
        with pytest.raises(ContractError):
            fixture.declaration(bad)


def test_repeated_build_semantics_and_postpublication_original_guard(originals, tmp_path, monkeypatch):
    from gf_training import card_instance_tree_data as data
    path, sha = originals
    a = data.build_fixture_dataset(path, sha, tmp_path/'a', SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()+90)
    b = data.build_fixture_dataset(path, sha, tmp_path/'b', SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()+90)
    assert a == b
    first_receipt = json.loads((tmp_path/'a/preparation.json').read_text())
    second_receipt = json.loads((tmp_path/'b/preparation.json').read_text())
    assert first_receipt['dataset_content_sha256'] == second_receipt['dataset_content_sha256'] == a['content_sha256']
    assert first_receipt['seconds'] > 0 and second_receipt['seconds'] > 0
    path, sha = copied(originals, tmp_path)
    original_write = data._write
    def mutate_after_publication(output_path, value, **kwargs):
        result = original_write(output_path, value, **kwargs)
        if output_path.name == 'preparation.json':
            manifest = json.loads(path.read_text())
            numeric = path.parent/manifest['cohorts'][0]['episodes'][1]['columns'][0]
            with numeric.open('ab') as stream:
                stream.write(b'postpublication change')
        return result
    monkeypatch.setattr(data, '_write', mutate_after_publication)
    with pytest.raises(ContractError):
        data.build_fixture_dataset(path, sha, tmp_path/'invalid', SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()+90)


def test_dataset_index_lifetime_mutation_and_fixture_origin_cannot_be_promoted(originals, tmp_path):
    path, sha = originals
    directory = tmp_path/'dataset'
    build_fixture_dataset(path, sha, directory, SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()+90)
    dataset = directory/'manifest.json'
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    with (pytest.raises(ContractError),
          VerifiedFixtureTreeDataset3(dataset, dataset_sha, SETTINGS, max_owned_bytes=LIMIT,
                                      deadline=time.monotonic()+90) as owner):
        for _, sample in owner.selected_inputs():
            del sample
        dataset.write_bytes(dataset.read_bytes()+b' ')
        owner.verify_complete()
    raw = json.loads(dataset.read_text())
    raw['origin']['qualified'] = True
    raw['origin']['purpose'] = 'qualified_game'
    raw['content_sha256'] = digest({k: v for k, v in raw.items() if k != 'content_sha256'})
    dataset_sha = write(dataset, raw)
    with pytest.raises(ContractError, match='fixture-only Dataset3'):
        inspect_fixture_dataset(dataset, expected_digest=dataset_sha, settings=SETTINGS,
                                max_owned_bytes=LIMIT, deadline=time.monotonic()+90)


def test_record_limit_and_deadline_refuse_before_decode(originals, monkeypatch):
    from gf_training import card_instance_tree_fixture as sources
    path, sha = originals
    with pytest.raises(TimeoutError):
        TreeFixtureSource(path, sha, SETTINGS, max_owned_bytes=LIMIT, deadline=time.monotonic()-1).__enter__()
    with source(path, sha) as owner:
        original_object = owner._scan.object
        def refuse_large_record(record, expected=None):
            if record.name.endswith('-e001.json'):
                pytest.fail('record-size refusal must precede JSON decode')
            return original_object(record, expected)
        monkeypatch.setattr(owner._scan, 'object', refuse_large_record)
        original_stat = sources.Path.lstat
        def exaggerated_size(record):
            result = original_stat(record)
            if record.name.endswith('-e001.json'):
                # A directory-entry size refusal is independent of JSON content.
                from types import SimpleNamespace
                return SimpleNamespace(st_mode=result.st_mode, st_size=9*1024**2)
            return result
        monkeypatch.setattr(sources.Path, 'lstat', exaggerated_size)
        with pytest.raises(ContractError, match='record/file limit'):
            next(owner.selected_inputs())
        monkeypatch.undo()
        # A failed partial worklist cannot finalize or issue a complete receipt.
        with pytest.raises(ContractError, match='active concrete'):
            owner.verify_complete()
        # Initiating failure is expected; suppress normal incomplete finalization.
        owner.close()


def test_suspended_original_iterator_cannot_issue_after_reportable_close(originals):
    path, sha = originals
    owner = source(path, sha).__enter__()
    samples = owner.selected_inputs()
    first = next(samples)
    assert first.viewer in (1, 2)
    owner.close(); owner.release()
    with pytest.raises(ContractError, match='active concrete'):
        next(samples)
