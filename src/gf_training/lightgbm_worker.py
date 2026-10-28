"""Owned native fit child: numeric arrays and closed safe settings only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Keep the request location out of imported native-library globals/diagnostics.
source = Path(sys.argv[1]) if len(sys.argv) == 2 else None
sys.argv[:] = ["gf-native-fit"]
os.environ.clear()
os.environ.update({"PATH": os.defpath, "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})


def main() -> None:
    import numpy as np

    from gf_common.domain import closed
    from gf_models.lightgbm.config import ModelConfig, model_config
    from gf_models.lightgbm.encoder import feature_names
    from gf_models.lightgbm.model import fit_numeric, load_artifact, save_artifact

    if source is None:
        raise ValueError("native request required")
    def sha256(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                value.update(chunk)
        return value.hexdigest()
    hashes = json.loads((source / "input-hashes.json").read_text())
    closed(hashes, {"request_sha256", "arrays_sha256"})
    if sha256(source / "request.json") != hashes["request_sha256"] or sha256(source / "arrays.npz") != hashes["arrays_sha256"]:
        raise ValueError("native numeric/request digest mismatch")
    request = json.loads((source / "request.json").read_text())
    closed(request, {"model", "params", "rounds", "early_stopping", "purpose", "action_config_hash"})
    model = model_config(request["model"])
    if not isinstance(model, ModelConfig) or model.model_id != "lightgbm":
        raise ValueError("LightGBM child requires LightGBM recipe")
    if model.checkpoint is not None:
        raise ValueError("fit child cannot load a checkpoint")
    with np.load(source / "arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"x", "y", "validation_x", "validation_y"}:
            raise ValueError("numeric input fields mismatch")
        booster, metrics = fit_numeric(arrays["x"], arrays["y"], arrays["validation_x"], arrays["validation_y"],
                                       params=request["params"], rounds=request["rounds"], early_stopping=request["early_stopping"],
                                       names=feature_names(model.encoder))
    save_artifact(booster, source / "staging", model, request["action_config_hash"], purpose=request["purpose"], ready=False)
    load_artifact(source / "staging", model=model, action_config_hash=request["action_config_hash"], allow_toy=True, allow_unready=True)
    proof = {"metrics": metrics, "input_hashes": hashes, "validated_model_sha256": sha256(source / "staging" / "model.txt"),
             "validated_inference_sha256": sha256(source / "staging" / "inference.json")}
    (source / "child-result.json").write_text(json.dumps(proof, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
