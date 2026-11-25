"""Run in the evaluation environment on the VM; explicitly downloads external metric assets."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, atomic_json, file_sha256, environment_info
from evaluation.protocol import load_eval_config, model_cache_environment
from evaluation.diversity import DiversityScorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", required=True, choices=("dreamsim", "lpips"))
    p.add_argument("--device", default="cpu", help="Asset setup only, no image scoring")
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation.yaml")
    a = p.parse_args()
    model_cache_environment()
    scorer = DiversityScorer(a.metric, {}, load_eval_config(a.eval_config), a.device, ROOT / "models")
    roots = [ROOT / "models/torch"]
    if a.metric == "dreamsim":
        roots.append(ROOT / "models/dreamsim")
    # Package-distributed LPIPS linear weights are identified by exact package version.
    files = {str(f.relative_to(ROOT)): file_sha256(f) for root in roots for f in root.rglob("*")
             if f.is_file() and f.suffix not in (".zip", ".pyc") and "__pycache__" not in f.parts}
    if not files:
        raise RuntimeError("No metric assets found after setup")
    atomic_json(ROOT / "models" / f"{a.metric}.assets.json", {
        "schema": 1, "metric": a.metric, "files": files, "scorer": scorer.metadata,
        "environment": environment_info()})


if __name__ == "__main__":
    main()
