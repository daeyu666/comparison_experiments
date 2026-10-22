"""Mixed-stage PSRF-DiffNet entry point for the unified comparison protocol."""
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train import main

if __name__ == "__main__":
    main("mixed")
