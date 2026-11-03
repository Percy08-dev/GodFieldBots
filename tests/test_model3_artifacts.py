"""Format4 numerical round trips; no parent completion or Source authority."""

import copy
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gf_common.domain import ContractError, canonical, digest
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT as OLD_CARD
from gf_models import structured_artifacts as artifacts
from gf_models.arrays import sha256
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from gf_models.structured_sealed import SealedScan, inspect_retained_artifact_v3
from tests.model3_support import LAYOUT, context, model3, raw_inputs


def seed_globals(device):
    torch.manual_seed(123)
    if device != "cpu":
        torch.cuda.manual_seed_all(456)


def update(network, optimizer, shuffle, frames, candidates):
    selected = int(shuffle.integers(len(candidates.kinds)))
    noise = torch.rand((), device="cpu").item()
    if network.device.type != "cpu":
        noise += torch.rand((), device=network.device).item()
    optimizer.zero_grad()
    logs, value, entropy = network(frames, candidates)
    ((1 + noise * 0.01) * (-logs[selected] + 0.2 * value.square() - 0.01 * entropy)).backward()
    optimizer.step()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"], ids=["cpu", "cuda"])
@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_format4_two_updates_equals_save_load_resume_all_named_adam_and_owned_rng(
    tmp_path, device, hand, temporal
):
    if device != "cpu" and os.environ.get("GF_RUN_CUDA") != "1":
        pytest.skip("actual host CUDA checks use GF_RUN_CUDA=1")
    if device != "cpu" and not torch.cuda.is_available():
        pytest.fail("requested actual CUDA check has no CUDA device")
    model = model3(hand, temporal)
    binding = InferenceBindingV2(device=device, max_candidates=64, candidate_chunk=2)
    continuous = make_network(model, 17, binding, layout=LAYOUT)
    paused = make_network(model, 17, binding, layout=LAYOUT)
    opt1, opt2 = (torch.optim.Adam(net.parameters(), lr=0.0003) for net in (continuous, paused))
    shuffle1, shuffle2 = (np.random.Generator(np.random.PCG64(17)) for _ in range(2))
    frames, candidates, *_ = raw_inputs()
    seed_globals(device)
    update(continuous, opt1, shuffle1, frames, candidates)
    update(continuous, opt1, shuffle1, frames, candidates)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device != "cpu" else []
    seed_globals(device)
    update(paused, opt2, shuffle2, frames, candidates)
    directory = tmp_path / "format4"
    args = context()
    without_hash = {key: value for key, value in args.items() if key != "action_config_hash"}
    inference = artifacts.save_inference_v3(
        directory, paused, model, args["action_config_hash"], **without_hash
    )
    identity = digest({"synthetic": "normal-update-boundary"})
    resume = artifacts.save_resume_v3(
        directory, paused, opt2, shuffle2, 1, model, binding, identity, **args
    )
    assert inference["schema_version"] == resume["schema_version"] == 4
    assert (
        resume["action_config_hash"]
        == inference["action_config_hash"]
        == args["action_config_hash"]
    )
    assert resume["cuda_rng_count"] == (torch.cuda.device_count() if device != "cpu" else 0)
    seed_globals(device)
    torch.rand(3)
    restored, _ = artifacts.load_inference_v3(
        directory,
        model=model,
        binding=binding,
        expected_digest=sha256(directory / "inference.json"),
        **args,
    )
    optimizer = torch.optim.Adam(restored.parameters(), lr=0.0003)
    shuffle = np.random.Generator(np.random.PCG64(999))
    assert (
        artifacts.restore_resume_v3(
            directory, restored, optimizer, shuffle, model, binding, identity, **args
        )
        == 1
    )
    update(restored, optimizer, shuffle, frames, candidates)
    assert canonical(shuffle.bit_generator.state) == canonical(shuffle1.bit_generator.state)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device != "cpu":
        assert all(
            torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True)
        )
    for (name, original), (other, actual) in zip(
        continuous.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(original, actual, atol=1e-5, rtol=1e-5, msg=name)
        assert (
            set(opt1.state[original])
            == set(optimizer.state[actual])
            == {"step", "exp_avg", "exp_avg_sq"}
        )
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opt1.state[original][key],
                optimizer.state[actual][key],
                atol=1e-5,
                rtol=1e-5,
                msg=name + key,
            )


def saved(tmp_path, *, layout=LAYOUT, synthetic=False):
    model, binding = model3(layout=layout), InferenceBindingV2(max_candidates=64)
    network = make_network(model, 17, binding, layout=layout)
    args = context(layout=layout, synthetic=synthetic)
    directory = tmp_path / "artifact"
    artifacts.save_inference_v3(
        directory,
        network,
        model,
        args["action_config_hash"],
        **{key: value for key, value in args.items() if key != "action_config_hash"},
    )
    return directory, network, model, binding, args


@pytest.mark.parametrize("layout", [LAYOUT, OLD_CARD])
def test_format4_explicit_profiles_and_strict_input_actionhash_and_format3_rejection(
    tmp_path, layout
):
    directory, network, model, binding, args = saved(tmp_path, layout=layout)
    digest1 = sha256(directory / "inference.json")
    manifest, _ = artifacts.inspect_inference_v3(
        directory, expected_digest=digest1, model=model, binding=binding, **args
    )
    assert manifest["simulator_revision"] == args["action_contract"].simulator_revision
    assert manifest["action_contract"] == asdict(args["action_contract"])
    for override in (
        {"action_config_hash": "0" * 64},
        {"purpose": "qualified_game"},
        {"synthetic_capacity": True},
        {"layout": OLD_CARD if layout is LAYOUT else LAYOUT},
    ):
        with pytest.raises(ContractError):
            artifacts.inspect_inference_v3(
                directory,
                expected_digest=digest1,
                model=model,
                binding=binding,
                **{**args, **override},
            )
    with pytest.raises(ContractError):
        artifacts.save_inference_v3(
            tmp_path / "relabel",
            network,
            model,
            "0" * 64,
            **{key: value for key, value in args.items() if key != "action_config_hash"},
        )
    assert not (tmp_path / "relabel").exists()
    from gf_common.action_contract import engineering_contract

    with pytest.raises(ContractError):
        artifacts.inspect_inference(
            directory,
            action_contract=engineering_contract(),
            purpose="component",
            synthetic_capacity=False,
        )
    old = tmp_path / "old"
    from tests.test_structured_network import small_model

    old_model = small_model()
    artifacts.save_inference(
        old,
        make_network(old_model, 17),
        old_model,
        args["action_config_hash"],
        action_contract=engineering_contract(),
        purpose="component",
        synthetic_capacity=False,
    )
    with pytest.raises(ContractError):
        artifacts.inspect_inference_v3(
            old,
            expected_digest=sha256(old / "inference.json"),
            model=model,
            binding=binding,
            **args,
        )


def test_format4_sealed_inspection_uses_recorded_metadata_without_runtime_queries(
    tmp_path, monkeypatch
):
    directory, network, model, binding, args = saved(tmp_path)
    shuffle = np.random.Generator(np.random.PCG64(17))
    artifacts.save_resume_v3(
        directory,
        network,
        torch.optim.Adam(network.parameters(), lr=0.0003),
        shuffle,
        0,
        model,
        binding,
        "0" * 64,
        **args,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("sealed inspection queried current runtime/producer")

    monkeypatch.setattr(artifacts, "runtime_metadata", forbidden)
    monkeypatch.setattr(artifacts, "producer_identity", forbidden)
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", forbidden)
    result = inspect_retained_artifact_v3(
        directory,
        expected_inference_digest=sha256(directory / "inference.json"),
        scan=SealedScan(max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 30),
        **args,
    )
    assert result.inference["schema_version"] == result.resume["schema_version"] == 4


def test_format4_only_owned_rng_is_restored_and_missing_shards_fail_before_network(
    tmp_path, monkeypatch
):
    directory, network, model, binding, args = saved(tmp_path)
    shuffle = np.random.Generator(np.random.PCG64(17))
    artifacts.save_resume_v3(
        directory,
        network,
        torch.optim.Adam(network.parameters(), lr=0.0003),
        shuffle,
        0,
        model,
        binding,
        "0" * 64,
        **args,
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    artifacts.restore_resume_v3(
        directory,
        network,
        torch.optim.Adam(network.parameters(), lr=0.0003),
        np.random.Generator(np.random.PCG64(1)),
        model,
        binding,
        "0" * 64,
        **args,
    )
    assert random.getstate() == python_state
    actual_np = np.random.get_state()
    assert (
        actual_np[0] == numpy_state[0]
        and np.array_equal(actual_np[1], numpy_state[1])
        and actual_np[2:] == numpy_state[2:]
    )
    manifest = json.loads((directory / "inference.json").read_text())
    (directory / manifest["weights"]["shards"][0]["file"]).unlink()
    import gf_models.structured_network as networks

    monkeypatch.setattr(
        networks,
        "make_network",
        lambda *args, **kwargs: pytest.fail("missing shard reached network allocation"),
    )
    with pytest.raises((ContractError, FileNotFoundError)):
        artifacts.load_inference_v3(
            directory,
            expected_digest=sha256(directory / "inference.json"),
            model=model,
            binding=binding,
            **args,
        )


def test_separate_manifest_commits_and_prepublication_failures(tmp_path, monkeypatch):
    directory, network, model, binding, args = saved(tmp_path)
    actual_write = artifacts.write_bundle

    def fail_resume(path, prefix, *values, **settings):
        if prefix == "resume":
            raise OSError("injected before resume manifest")
        return actual_write(path, prefix, *values, **settings)

    monkeypatch.setattr(artifacts, "write_bundle", fail_resume)
    with pytest.raises(OSError, match="injected"):
        artifacts.save_resume_v3(
            directory,
            network,
            torch.optim.Adam(network.parameters(), lr=0.0003),
            np.random.Generator(np.random.PCG64(17)),
            0,
            model,
            binding,
            "0" * 64,
            **args,
        )
    assert not (directory / "resume.json").exists()
    artifacts.inspect_inference_v3(
        directory,
        expected_digest=sha256(directory / "inference.json"),
        model=model,
        binding=binding,
        **args,
    )
    monkeypatch.setattr(
        artifacts,
        "write_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected before inference manifest")
        ),
    )
    target = tmp_path / "failed-inference"
    with pytest.raises(OSError, match="injected"):
        artifacts.save_inference_v3(
            target,
            network,
            model,
            args["action_config_hash"],
            **{key: value for key, value in args.items() if key != "action_config_hash"},
        )
    assert not (target / "inference.json").exists()


@pytest.mark.parametrize(
    "failure", ["extra_shard_descriptor", "modified_shard", "wrong_rule", "unknown_field",
        "stale_token", "stale_architecture"]
)
def test_format4_closed_metadata_and_shard_faults_precede_network(tmp_path, monkeypatch, failure):
    directory, _network, model, binding, args = saved(tmp_path)
    path = directory / "inference.json"
    manifest = json.loads(path.read_text())
    if failure == "extra_shard_descriptor":
        manifest["weights"]["files"]["weights-999999.npz"] = "0" * 64
    elif failure == "modified_shard":
        shard = directory / manifest["weights"]["shards"][0]["file"]
        shard.write_bytes(shard.read_bytes() + b"modified")
    elif failure == "wrong_rule":
        manifest["rule_signature"] = "0" * 64
    elif failure == "stale_token":
        manifest["token_signature"] = "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad"
    elif failure == "stale_architecture":
        manifest["architecture_identity"] = "37b9dab540fc1ed78f237afa8f4b0fbfe966b0eda5b7b60b27acf3b4be1b940a"
    else:
        manifest["unknown"] = "must reject"
    if failure != "modified_shard":
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    from gf_models import structured_network as networks

    monkeypatch.setattr(
        networks,
        "make_network",
        lambda *args, **kwargs: pytest.fail("bad artifact reached network allocation"),
    )
    with pytest.raises(ContractError):
        artifacts.load_inference_v3(
            directory, expected_digest=sha256(path), model=model, binding=binding, **args
        )


def test_format4_clean_child_sealed_has_no_torch_or_current_metadata_queries(tmp_path):
    import subprocess
    import sys

    directory, network, model, binding, args = saved(tmp_path)
    artifacts.save_resume_v3(
        directory,
        network,
        torch.optim.Adam(network.parameters(), lr=0.0003),
        np.random.Generator(np.random.PCG64(17)),
        0,
        model,
        binding,
        "0" * 64,
        **args,
    )
    code = """
import importlib.abc,importlib.metadata,json,sys,time
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch forbidden')
sys.meta_path.insert(0,NoTorch())
def forbidden(*args,**kwargs):raise AssertionError('current metadata query forbidden')
importlib.metadata.version=forbidden
from gf_common.card_instance import parse_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.structured_layout import layout_for_contract
from gf_models.arrays import sha256
from gf_models.structured_sealed import SealedScan,inspect_retained_artifact_v3
p=Path(sys.argv[1]);m=json.loads((p/'inference.json').read_text());c=parse_card_instance_contract(m['action_contract'])
r=inspect_retained_artifact_v3(p,expected_inference_digest=sha256(p/'inference.json'),action_contract=c,
 rules=card_instance_rules(),layout=layout_for_contract(c),action_config_hash=m['action_config_hash'],
 purpose='component',synthetic_capacity=False,scan=SealedScan(max_owned_bytes=512*1024**2,deadline=time.monotonic()+30))
assert r.retained and 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(directory)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("device", ["cpu", "cuda:0"], ids=["cpu", "cuda"])
@pytest.mark.parametrize("bound", [False, True], ids=["unbound", "bound"])
@pytest.mark.parametrize("destination", ["wrong_type", "mt19937"])
def test_format4_wrong_destination_shuffle_rejects_before_any_restore_mutation(
    tmp_path, device, bound, destination
):
    from types import SimpleNamespace

    from gf_models.structured_binding import _bind_network_input_v3

    if device != "cpu" and os.environ.get("GF_RUN_CUDA") != "1":
        pytest.skip("actual host CUDA checks use GF_RUN_CUDA=1")
    if device != "cpu" and not torch.cuda.is_available():
        pytest.fail("requested actual CUDA check has no CUDA device")
    model, binding = model3(), InferenceBindingV2(device=device, max_candidates=64)
    frames, candidates, *_ = raw_inputs()
    source = make_network(model, 17, binding, layout=LAYOUT)
    source_optimizer = torch.optim.Adam(source.parameters(), lr=0.0003)
    source_shuffle = np.random.Generator(np.random.PCG64(17))
    update(source, source_optimizer, source_shuffle, frames, candidates)
    args = context()
    directory = tmp_path / "valid-checkpoint"
    artifacts.save_inference_v3(
        directory,
        source,
        model,
        args["action_config_hash"],
        **{key: value for key, value in args.items() if key != "action_config_hash"},
    )
    artifacts.save_resume_v3(
        directory, source, source_optimizer, source_shuffle, 1, model, binding, "0" * 64, **args
    )
    target = make_network(model, 999, binding, layout=LAYOUT)
    optimizer = torch.optim.Adam(target.parameters(), lr=0.0003)
    update(target, optimizer, np.random.Generator(np.random.PCG64(999)), frames, candidates)
    if bound:
        _bind_network_input_v3(
            target, **{key: value for key, value in args.items() if key != "layout"}, fresh=True
        )
    shuffle = (
        SimpleNamespace(bit_generator=np.random.PCG64(999))
        if destination == "wrong_type"
        else np.random.Generator(np.random.MT19937(999))
    )
    torch.manual_seed(876)
    if device != "cpu":
        torch.cuda.manual_seed_all(987)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_weights = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}
    before_binding = getattr(target, "_structured_input_binding_v3", None)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state_all() if device != "cpu" else []
    before_shuffle = copy.deepcopy(shuffle.bit_generator.state)
    with pytest.raises(ContractError, match="PCG64"):
        artifacts.restore_resume_v3(
            directory, target, optimizer, shuffle, model, binding, "0" * 64, **args
        )
    assert getattr(target, "_structured_input_binding_v3", None) is before_binding
    assert torch.equal(torch.get_rng_state(), before_cpu)
    if device != "cpu":
        assert all(
            torch.equal(a, b)
            for a, b in zip(torch.cuda.get_rng_state_all(), before_cuda, strict=True)
        )
    np.testing.assert_equal(shuffle.bit_generator.state, before_shuffle)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before_weights[name]), name
    actual = optimizer.state_dict()
    assert actual["param_groups"] == before_optimizer["param_groups"]
    assert set(actual["state"]) == set(before_optimizer["state"])
    for index, original in before_optimizer["state"].items():
        assert set(actual["state"][index]) == set(original)
        for name, tensor in original.items():
            assert torch.equal(actual["state"][index][name], tensor), (index, name)
