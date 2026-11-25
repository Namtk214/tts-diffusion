"""Resolve current model revisions without downloading weights; run on the VM."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, atomic_json
from benchmark.config import MODEL_IDS


def main():
    from huggingface_hub import HfApi
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "configs/models.lock.json")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    if a.output.exists() and not a.refresh:
        raise SystemExit("Lock already exists; pass --refresh deliberately to create a different experiment")
    api = HfApi()
    models = {k: {"id": repo, "revision": api.model_info(repo).sha} for k, repo in MODEL_IDS.items()}
    atomic_json(a.output, {"schema": 1, "models": models,
                          "note": "Revision availability is not proof of weight access; preflight checks cached files."})
    print(a.output)

if __name__ == "__main__":
    main()
