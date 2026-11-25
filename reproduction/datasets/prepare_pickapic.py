"""Run from any working directory; download prompts only when invoked."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.dataset_registry import main
if __name__ == "__main__":
    main("pickapic")
