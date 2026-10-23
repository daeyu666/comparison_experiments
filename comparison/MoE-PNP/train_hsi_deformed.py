"""Unified-protocol mixed-stage entry point for MoE-PNP."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("moe_pnp_train_module", THIS_DIR / "train.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load MoE-PNP train.py")
_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _module
_SPEC.loader.exec_module(_module)

if __name__ == "__main__":
    _module.main("mixed")
