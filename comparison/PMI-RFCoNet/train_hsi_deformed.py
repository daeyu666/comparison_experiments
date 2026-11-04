"""Mixed stage: initialize the registered PMI-RFCoNet and adapt to HSI deformation."""
import importlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    importlib.import_module("comparison.PMI-RFCoNet.train").main("mixed")
