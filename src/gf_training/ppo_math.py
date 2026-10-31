"""Shared fixed PPO loss recipe; no model or transport dependency."""
from __future__ import annotations

from typing import Any

from gf_common.domain import Json


def losses(logprobs: Any, values: Any, mask: Any, selected: Any, old: Any, advantage: Any, target: Any) -> tuple[Any, Json]:
    import torch
    chosen = logprobs.gather(1, selected[:, None]).squeeze(1)
    ratio = (chosen - old).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
    mse = ((values - target) ** 2).mean()
    safe_logprobs = logprobs.masked_fill(~mask, 0)
    entropy = -(logprobs.exp() * safe_logprobs).sum(-1).mean()
    loss = policy + .5 * mse - .01 * entropy
    return loss, {'policy_loss': float(policy.detach()), 'value_mse': float(mse.detach()),
                  'entropy': float(entropy.detach()), 'approx_kl': float((old - chosen).mean().detach()),
                  'total_loss': float(loss.detach())}
