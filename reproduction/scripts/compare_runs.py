import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.compare import compare_runs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paired prompt-bootstrap comparison, left minus right")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    compare_runs(a.left, a.right, a.output)
