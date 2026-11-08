"""Closed raw16 DTO envelope tests; structural fixtures give no Source rights."""
from __future__ import annotations

import json

import pytest

from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical, digest
from gf_training.native_card_instance_final_validation import _decode_recorded_episode_packet
from tests.test_native_card_instance_final_validation import originals as reference_originals
from tests.test_native_card_instance_final_validation import packet_from_reference


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    return reference_originals.__wrapped__(tmp_path_factory)


def recorded_packet(originals,index):
    packet,record=packet_from_reference(originals,index)
    summary={'scope':'UNADMITTED-raw-envelope-NOT-parent-admission','parent':{'run_sha256':'1'*64},
        'collections':[{'generation':5,'manifest_sha256':'2'*64,'source_proof':{'source_set_sha256':'3'*64},
            'correspondence':{'episode_digests':[row['digest']for row in originals[1]['oracle'][:16]]}}]}
    metadata=json.loads(packet[0])
    metadata.update(kind='card-instance-recorded-episode-payload',summary_sha256=digest(summary),
        parent_run_sha256='1'*64,collection_index=0,collection_generation=5,
        collection_manifest_sha256='2'*64,source_set_sha256='3'*64)
    assert len(metadata)==18
    return (canonical(metadata).encode(),*packet[1:]),record,summary


def decode(originals,packet,summary,index):
    expected=originals[1]['oracle'][index]
    return _decode_recorded_episode_packet(packet,summary=summary,collection_index=0,episode_index=index,
        episode_id=expected['episode_id'],viewer=expected['viewer'],actions=originals[3],rules=card_instance_rules(),
        aggregate_window=2,max_owned_bytes=1024**3,retained_owned_bytes=0,max_record_bytes=8*1024**2,
        reserve=lambda _:None,check=lambda:None)


@pytest.mark.parametrize('index',range(16))
def test_recorded_exact18_own_originals_match_full_reader_without_numeric_or_gae(originals,index):
    packet,record,summary=recorded_packet(originals,index)
    assert decode(originals,packet,summary,index)==record
    metadata=json.loads(packet[0])
    assert all(key not in metadata for key in ('numeric','advantage','target','qualified','issuer'))


@pytest.mark.parametrize('field',['parent_run_sha256','collection_index','collection_generation',
    'collection_manifest_sha256','source_set_sha256','episode_digest','summary_sha256','kind','extra','missing'])
def test_recorded_profile_rejects_foreign_parent_cohort_ordinal_source_and_mixed_envelopes(originals,field):
    packet,_,summary=recorded_packet(originals,1)
    metadata=json.loads(packet[0])
    if field=='extra':metadata['qualified']=True
    elif field=='missing':del metadata['source_set_sha256']
    elif field=='kind':metadata[field]='card-instance-final-validation-episode-payload'
    elif field in ('collection_index','collection_generation'):metadata[field]+=1
    else:metadata[field]='0'*64
    bad=canonical(metadata).encode(),*packet[1:]
    with pytest.raises(ContractError):decode(originals,bad,summary,1)
