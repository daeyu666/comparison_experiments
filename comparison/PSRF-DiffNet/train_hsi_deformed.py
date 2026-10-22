"""Mixed-stage PSRF-DiffNet entry point for the unified comparison protocol."""
import importlib.util
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) in sys.path:
    sys.path.remove(str(THIS_DIR))
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(1, str(ROOT))

spec = importlib.util.spec_from_file_location("psrf_diffnet_train", THIS_DIR / "train.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load PSRF-DiffNet train.py")
train_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_module
spec.loader.exec_module(train_module)

if __name__ == "__main__":
    train_module.main("mixed")
