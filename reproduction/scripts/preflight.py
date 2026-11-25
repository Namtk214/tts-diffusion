"""Explicit VM preflight: imports, CUDA allocation, model cache and Gurobi license."""
import argparse
import importlib
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, environment_info, atomic_json, read_json
from benchmark.config import load_config, resolve_models
from methods.assets import require_cuda, cached_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("sgi", "satisdive"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", type=int, default=1)
    p.add_argument("--models-lock", type=Path, default=ROOT / "configs/models.lock.json")
    p.add_argument("--check-solver", action="store_true", help="Solve a synthetic 128-variable QIP to check license capacity")
    a = p.parse_args()
    hardware = require_cuda(a.device, a.reward_device if a.method == "satisdive" else None)
    for module in ("torch", "torchvision", "diffusers", "transformers"):
        importlib.import_module(module)
    cfg = resolve_models(load_config(ROOT / f"configs/unified/{a.method}.yaml"), a.models_lock)
    snapshots = {name: cached_model(entry) for name, entry in cfg["model_revisions"].items()}
    solver = None
    if a.method == "sgi":
        import gurobipy as gp
        solver = {"version": gp.gurobi.version(), "capacity_checked": False}
        if a.check_solver:
            import numpy as np
            sys.path.insert(0, str(ROOT / "third_party/sgi/src"))
            from my_utils.solvers import gurobi_solver
            diagnostics = {}
            selected = gurobi_solver(np.ones(128), np.triu(np.ones((128, 128)), 1), 64, diagnostics=diagnostics)
            solver.update(capacity_checked=len(selected) == 64, diagnostics=diagnostics)
    else:
        executable = os.environ.get("SATISDIVE_REWARD_PYTHON", "")
        if not Path(executable).is_file():
            raise RuntimeError("Source scripts/env.sh and install the HPSv3 environment")
        import subprocess
        subprocess.run([executable, "-c", "from hpsv3 import HPSv3RewardInferencer; import torch; assert torch.cuda.is_available()"], check=True)
    result = {"method": a.method, "hardware": hardware, "snapshots": snapshots, "solver": solver,
              **environment_info(), "status": "preflight_only_no_generation"}
    atomic_json(ROOT / "reports" / f"preflight-{a.method}.json", result)
    print(result)


if __name__ == "__main__":
    main()
