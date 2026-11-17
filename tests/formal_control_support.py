"""Current synthetic control preconditions, never formal qualification.

Existing UNADMITTED tests replace only their preflight seam with this shape
check. Production preflight keeps rejecting the frozen rules-v1 recipe. This
helper cannot admit a parent, produce generations or validate an original Source.
"""

from dataclasses import asdict

from gf_common.card_instance_formal import checked_config, checked_packed_config
from gf_common.domain import ContractError, canonical, digest
from gf_training.card_instance_formal import load_recipe


def unadmitted_current_synthetic_formal_shape(loaded):
    """Retain closed plan/caller checks while isolating control ownership tests."""
    recipe = load_recipe()
    checker = checked_packed_config if hasattr(loaded.config, 'storage') else checked_config
    checker(loaded.effective, recipe)
    if canonical(loaded.effective) != canonical(asdict(loaded.config)) or digest(loaded.effective) != loaded.sha256:
        raise ContractError('formal loaded effective Config/hash differs from original caller')
    return recipe
