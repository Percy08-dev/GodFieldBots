"""Collection9 retains the exact Model4 journal and inference-manifest contracts."""
import time
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, clone, digest


@pytest.fixture
def header_case(tmp_path):
    pytest.importorskip('torch')
    from gf_common.card_instance import ActionConfigV3, shared_card_instance_contract
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_artifacts import save_game_inference_v4
    from gf_models.structured_binding import ConfiguredNeuralModelV4
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network
    from tests.model3_support import LAYOUT
    from tests.test_hpo_network import small_model4
    actions=ActionConfigV3(action_contract=shared_card_instance_contract(),max_candidates=64)
    rules=card_instance_rules()
    model=small_model4()
    binding=InferenceBindingV2(max_candidates=64,candidate_chunk=2)
    network=make_network(model,17,binding,layout=LAYOUT)
    directory=tmp_path/'inference'
    save_game_inference_v4(directory,network,model,digest(asdict(actions)),action_contract=actions.action_contract,rules=rules)
    configured=ConfiguredNeuralModelV4(**{name:getattr(model,name) for name in model.__dataclass_fields__},
        checkpoint=str(directory),inference=binding)
    owner=RetainedGameArtifacts((configured,),actions,rules=rules,max_owned_bytes=128*1024**2,
        preflight_deadline=time.monotonic()+30,finalization_deadline=time.monotonic()+30)
    try:
        label,policy=owner.binding(configured)
        neural=owner.journal_metadata(label)
    finally:owner.release()
    runtime={'original-runtime':'same receipt'}
    spec={'episode_id':'episode','policies':[label,'random-card-instance-v2'],'sampling_seeds':[17,23]}
    manifest={'schema_version':9,'run_id':'collection','policy_bindings':{label:asdict(policy)},
        'neural_inputs':{'policies':{label:neural}},'schedule':[{'episodes':[spec]}],'runtime_evidence':[runtime]}
    header={'run_id':'collection','episode_id':'episode','viewer':1,'game_id':'simulator-game',
        'policy':label,'sampling_seed':17,'model_id':'mlp','model_version':'4','runtime_evidence':runtime,'neural':clone(neural)}
    return header,manifest,{'game_id':'simulator-game'},spec,actions


@pytest.mark.parametrize('schema',[6,7,8,9])
def test_hpo_header_audit_requires_model4_for_every_explicit_collection_version(header_case,schema):
    from gf_training.card_instance_source import _game_header
    header,manifest,episode,spec,actions=header_case
    manifest['schema_version']=schema
    _game_header(header,manifest,episode,spec,actions,1)


def test_collection9_header_cannot_reinterpret_model3_as_model4(header_case):
    from gf_training.card_instance_source import _game_header
    header,manifest,episode,spec,actions=header_case
    header['model_version']='3'
    with pytest.raises(ContractError,match='neural metadata differs'):
        _game_header(header,manifest,episode,spec,actions,1)


def test_collection9_header_requires_exact_original_canonical_manifest_digest(header_case):
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import validate_neural_metadata
    from gf_training.card_instance_source import _game_header
    header,manifest,episode,spec,actions=header_case
    neural=manifest['neural_inputs']['policies'][header['policy']]
    neural['inference_manifest']['producer']['code_revision']='1'*40
    header['neural']=clone(neural)
    # The altered document remains structurally valid; the unchanged committed
    # inference identity must still reject these different original bytes.
    validate_neural_metadata(neural,actions=actions,rules=card_instance_rules())
    with pytest.raises(ContractError,match='manifest bytes/hash differ'):
        _game_header(header,manifest,episode,spec,actions,1)
