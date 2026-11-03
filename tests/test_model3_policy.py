"""Standalone component policy identity checks precede cache/tensor work."""

import random
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.diverse_actions import (
    GeneratedCandidatesV3,
    NumericCosts,
    PublicCandidateGeneratorV3,
)
from gf_common.domain import ContractError, canonical, digest
from gf_models.structured_binding import input_binding_v3, safe_options_v3
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from gf_models.structured_policy import StructuredPolicyV3
from tests.model3_support import LAYOUT, context, model3, raw_inputs


def policy_inputs(*, candidates=128, synthetic=False):
    model, binding = model3(), InferenceBindingV2(max_candidates=candidates)
    _frames, _tokens, originals, bank, _table = raw_inputs(count=1)
    actions = ActionConfigV3(action_contract=LAYOUT.action_contract)
    generated = PublicCandidateGeneratorV3(
        actions, rules=card_instance_rules(), clock=lambda: 0.0
    ).generate(originals[-1], bank.view(), random.Random(17), NumericCosts())
    args = context(action_hash=digest(asdict(actions)), synthetic=synthetic)
    network = make_network(model, 17, binding, layout=LAYOUT)
    policy = StructuredPolicyV3(network, model, binding, "1" * 64, **args)
    return policy, originals[-1], bank, generated, args


def test_policy_component_cpu_sampling_is_report_independent_and_metadata_safe():
    policy, frame, bank, generated, args = policy_inputs()
    policy.reset(29)
    first = policy.select_frame(frame, bank.view(), generated)
    policy.reset(29)
    changed = GeneratedCandidatesV3(
        generated.table, replace(generated.report, count_ms=999, search_ms=999, selection_ms=999)
    )
    second = policy.select_frame(frame, bank.view(), changed)
    assert first == second
    assert first.validate(generated.table, frame)["type"]
    assert policy.rng.device.type == "cpu" and "ppo-v3-" in policy.version
    assert policy.cache.position == 1
    options = {
        "version": "3",
        "checkpoint": "/explicit/component",
        "expected_manifest_digest": "1" * 64,
        "model": asdict(policy.model),
        "inference": asdict(policy.binding),
        "cuda_visible_devices": "",
        **input_binding_v3(args["action_contract"], args["rules"], "component", False),
        "action_config_hash": args["action_config_hash"],
    }
    assert canonical(
        safe_options_v3(options, **{key: value for key, value in args.items() if key != "layout"})
    ) == canonical(options)


@pytest.mark.parametrize("failure", ["action_hash", "rules", "synthetic", "caps"])
def test_policy_rejects_incompatible_generated_before_tokens_cache_or_sampling(
    monkeypatch, failure
):
    import gf_models.structured_policy as module

    policy, frame, bank, generated, _args = policy_inputs(
        candidates=1 if failure == "caps" else 128, synthetic=failure == "synthetic"
    )
    if failure == "action_hash":
        generated = GeneratedCandidatesV3(
            generated.table, replace(generated.report, config_hash="0" * 64)
        )
    elif failure == "rules":
        # A caller-owned mutable transport value cannot alter the frozen network binding.
        policy.input_binding["rule_signature"] = "0" * 64
    before = policy.rng.get_state().clone()
    monkeypatch.setattr(
        module,
        "tokenize_frame",
        lambda *args, **kwargs: pytest.fail("incompatible metadata reached tokens"),
    )
    with pytest.raises(ContractError):
        policy.select_frame(frame, bank.view(), generated)
    assert policy.cache.position == 0 and torch.equal(policy.rng.get_state(), before)


def test_network_action_hash_first_binding_cannot_be_replaced_and_qualified_scope_rejects(
    monkeypatch,
):
    policy, _frame, _bank, _generated, args = policy_inputs()
    monkeypatch.setattr(
        policy.network, "new_cache", lambda: pytest.fail("hash mismatch reached cache allocation")
    )
    with pytest.raises(ContractError):
        StructuredPolicyV3(
            policy.network,
            policy.model,
            policy.binding,
            "1" * 64,
            **{**args, "action_config_hash": "0" * 64},
        )
    with pytest.raises(ContractError):
        StructuredPolicyV3(
            policy.network,
            policy.model,
            policy.binding,
            "1" * 64,
            **{**args, "purpose": "qualified_game"},
        )
    with pytest.raises(AttributeError):
        policy.network.layout = object()


def test_component_policy_rejects_distinct_game_generated_path_before_tokenization(monkeypatch):
    from gf_models import structured_policy as module

    policy, frame, bank, _component, _args = policy_inputs()
    actions = ActionConfigV3(action_contract=LAYOUT.action_contract)
    generated = PublicCandidateGeneratorV3(
        actions, rules=card_instance_rules(), clock=lambda: 0.0
    ).generate_game(frame, bank.view(), random.Random(17), NumericCosts())
    before = policy.rng.get_state().clone()
    monkeypatch.setattr(
        module,
        "tokenize_frame",
        lambda *args, **kwargs: pytest.fail("game receipt entered component policy"),
    )
    with pytest.raises(ContractError):
        policy.select_frame(frame, bank.view(), generated)
    assert policy.cache.position == 0 and torch.equal(policy.rng.get_state(), before)
