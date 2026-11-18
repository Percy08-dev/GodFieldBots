"""Lazy TorchRL MLP policy adapter; immutable inference-only model registry."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gf_common.domain import ContractError, closed


def publication(network: Any,checkpoint: str) -> Any:
    from tensordict import NonTensorData, TensorDict
    return TensorDict({'publication':NonTensorData({'command':'PUBLISH','checkpoint':checkpoint,
        'model':asdict(network.config),'binding':asdict(network.binding)}),
        'weights':TensorDict({name:value.detach().cpu().clone() for name,value in network.state_dict().items()},batch_size=[])},batch_size=[])


def make_registry(*,device: str,cache_entries: int,cache_weight_bytes: int,visibility: str | None = None) -> Any:
    import os
    if device != 'cpu':
        if visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visibility
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import time

    import torch
    from tensordict import NonTensorData, TensorDict, lazy_stack

    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.hpo_config import model_config_v4
    from gf_models.shared_numeric import score_numeric_row
    from gf_models.structured_config import InferenceBindingV2, model_config_v3, parameter_shapes
    from gf_models.structured_network import make_network
    from gf_models.torch_runtime import torch_runtime

    torch_runtime(device)

    class Registry(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.networks=torch.nn.ModuleDict()
            self.publications=0

        def load_policy_weights(self,weights: Any) -> None:
            metadata=weights.get_non_tensor('publication')
            if metadata.get('command')=='EVICT':
                closed(metadata,{'command','checkpoint'})
                if metadata['checkpoint'] not in self.networks:
                    raise ContractError('shared eviction lacks original immutable checkpoint')
                del self.networks[metadata['checkpoint']]
                return
            closed(metadata,{'command','checkpoint','model','binding'})
            checkpoint=metadata['checkpoint']
            if metadata['command']!='PUBLISH' or type(checkpoint) is not str or len(checkpoint)!=64 or set(checkpoint)-set('0123456789abcdef'):
                raise ContractError('invalid immutable shared publication')
            if checkpoint in self.networks:
                raise ContractError('shared checkpoint is immutable and cannot be republished')
            parser=model_config_v4 if metadata['model']['version']=='4' else model_config_v3
            model=parser(metadata['model'],layout=layout)
            if model.model_id!='mlp':raise ContractError('shared registry admits MLP only')
            original=InferenceBindingV2(**metadata['binding'])
            from dataclasses import replace
            binding=replace(original,device=device)
            expected=parameter_shapes(model,layout=layout)
            incoming=weights['weights']
            if set(incoming.keys())!=set(expected):raise ContractError('shared publication is not inference-only weights')
            size=0
            for name,shape in expected.items():
                value=incoming[name]
                if value.dtype!=torch.float32 or tuple(value.shape)!=shape or not torch.isfinite(value).all():
                    raise ContractError('invalid shared inference tensor')
                size+=value.numel()*value.element_size()
            live=sum(p.numel()*p.element_size() for network in self.networks.values() for p in network.parameters())
            if len(self.networks)>=cache_entries or live+size>cache_weight_bytes:
                raise ContractError('shared immutable checkpoint cache capacity exceeded')
            network=make_network(model,0,binding,layout=layout)
            with torch.no_grad():
                for name,target in network.state_dict().items():target.copy_(incoming[name])
            network.eval()
            self.networks[checkpoint]=network
            self.publications+=1

        def forward(self,batch: Any) -> Any:
            answers=[]
            forward_at=time.monotonic()
            with torch.no_grad():
                for row in batch.unbind(0):
                    routing=row.get_non_tensor('routing')
                    checkpoint=routing['checkpoint']
                    if checkpoint not in self.networks:raise ContractError('unpublished shared checkpoint')
                    logs,value=score_numeric_row(self.networks[checkpoint],row['inputs'])
                    answers.append(TensorDict({'logs':logs,'probabilities':logs.exp(),'value':value,'routing':NonTensorData(routing),
                        'transport_batch_size':torch.tensor(len(batch),device=device),
                        'tensor_batch_size':torch.tensor(1,device=device),
                        'forward_started':torch.tensor(forward_at,dtype=torch.float64,device=device)},batch_size=[]))
            return lazy_stack(answers)

    return Registry()
