"""Compatibility alias for shared numerical helpers."""
import sys

from gf_models import features as _shared

sys.modules[__name__] = _shared
