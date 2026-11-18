"""Participant-local candidate tokenization and CPU sampling over private scores."""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

from gf_common.domain import ContractError, closed, integer
from gf_common.structured import Selection
from gf_common.structured_layout import layout_for_contract
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.shared_numeric import numeric_row
from gf_models.structured_config import InferenceBindingV2, model_config_v3, resource_estimate


class SharedParticipantPolicy:
    def __init__(self,options: Any,actions: Any,rules: Any,channel: Any) -> None:
        import torch

        from gf_models.hpo_config import model_config_v4
        from gf_models.structured_binding import policy_label_game_v3, policy_label_game_v4
        torch.set_num_threads(1)
        self.torch,self.channel,self.actions,self.rules=torch,channel,actions,rules
        self.layout=layout_for_contract(actions.action_contract)
        parser=model_config_v4 if options['version']=='4' else model_config_v3
        self.model: Any=parser(options['model'],layout=self.layout)
        if self.model.model_id!='mlp':raise ContractError('shared participant supports only MLP')
        self.original_binding=InferenceBindingV2(**options['inference'])
        self.binding=replace(self.original_binding,device='cpu')  # transport placement, never recorded model execution.
        labeler: Any=policy_label_game_v4 if options['version']=='4' else policy_label_game_v3
        self.version=labeler(self.model,self.original_binding,options['expected_manifest_digest'],
            action_contract=actions.action_contract,rules=rules,action_config_hash=options['action_config_hash'],
            cuda_visible_devices=options['cuda_visible_devices'])
        self.checkpoint=options['expected_manifest_digest']
        self.actions_hash=options['action_config_hash']
        self.rule_signature=options['rule_signature']
        self.rng=torch.Generator(device='cpu')
        self.reset(0)

    def reset(self,sampling_seed: int) -> None:
        integer(sampling_seed,0,2**64-1)
        self.rng.manual_seed(sampling_seed)
        self.position=0
        self.previous=None
        self.last_timing: dict[str,Any]={}

    def select_frame(self,frame: Any,bank_view: Any,generated: Any) -> Selection:
        from gf_common.diverse_actions import GameGeneratedCandidatesV3
        if type(generated) is not GameGeneratedCandidatesV3:
            raise ContractError('shared participant requires original qualified candidates')
        generated.table.validate(frame)
        if (generated.report.config_hash!=self.actions_hash or generated.report.rule_signature!=self.rule_signature or generated.table.action_contract!=self.actions.action_contract
                or frame.layout is not self.layout or bank_view.layout is not self.layout):
            raise ContractError('shared candidate/input/config binding mismatch')
        integer(self.position+1,1,self.model.temporal.max_decisions)
        integer(frame.boundary.end,0,self.model.event.max_episode)
        integer(frame.boundary.end-frame.interval_start,0,self.model.event.max_interval)
        integer(len(generated.table.kinds),1,self.original_binding.max_candidates)
        if self.position==0 and frame.interval_start!=0:raise ContractError('shared first own decision must start at event zero')
        if self.previous is not None and (frame.episode!=self.previous.episode or frame.actor!=self.previous.actor
                or frame.revision<=self.previous.revision or frame.interval_start!=self.previous.boundary.end):
            raise ContractError('shared own-decision continuity mismatch')
        started=time.perf_counter()
        tokens=tokenize_frame(frame,frame.bank_view(bank_view),aggregate_window=self.model.event.aggregate_window,layout=self.layout)
        candidates=tokenize_table(generated.table,frame,self.model.max_positions,
            aggregate_window=self.model.event.aggregate_window,layout=self.layout)
        resource_estimate(self.model,self.original_binding,decisions=self.position+1,
            interval_events=len(tokens.interval_numerics),candidates=len(candidates.kinds),suffix_decisions=1,layout=self.layout)
        numeric=numeric_row(tokens,candidates,self.model,layout=self.layout)
        encode_ms=(time.perf_counter()-started)*1000
        requested=time.perf_counter()
        response=self.channel.request(frame,self.checkpoint,numeric)
        roundtrip=(time.perf_counter()-requested)*1000
        closed(response,{'routing','logs','probabilities','value'})
        logs,probabilities=response['logs'],response['probabilities']
        if (len(logs)!=len(candidates.kinds) or len(probabilities)!=len(logs)
                or any(type(value) not in (int,float) or not math.isfinite(value) for value in (*logs,*probabilities,response['value']))):
            raise ContractError('invalid shared own-row scores')
        sampling=time.perf_counter()
        values=self.torch.tensor(probabilities,dtype=self.torch.float32)
        if bool((values<0).any()) or not bool(values.sum()>0):raise ContractError('invalid shared sampling probabilities')
        self.channel.check_deadline()
        selected=int(self.torch.multinomial(values,1,generator=self.rng).item())
        self.position+=1
        self.previous=frame
        self.last_timing={'tokenization_ms':encode_ms,'sampling_ms':(time.perf_counter()-sampling)*1000,
            'shared_request_roundtrip_ms':roundtrip}
        return Selection(generated.table.table_hash,selected,self.version,float(logs[selected]),float(response['value']))
