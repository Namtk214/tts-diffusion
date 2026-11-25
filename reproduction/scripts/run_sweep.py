"""Sweeps only two permitted methods; identical points resolve to identical run IDs."""
import argparse
import copy
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, read_json
from benchmark.config import load_config
from benchmark.storage import validate_pilot
from benchmark.runner import parser as generation_parser, execute


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("pickapic", "geneval", "hpdv2"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", default=1, type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--models-lock", type=Path)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--profile-flops", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--pilot-run", type=Path, help="Completed, reviewed 50-prompt native/unified pilot")
    a = p.parse_args()
    sweep = yaml.safe_load(a.config.read_text())
    base = ROOT / sweep["base_config"]
    cfg = load_config(base)
    if not a.plan:
        if not a.pilot_run:
            raise ValueError("Measure the 50-prompt pilot on the VM first; pass --pilot-run <run-dir>")
        validate_pilot(a.pilot_run, cfg["method"], sweep["requires_pilot"])
    block, parameter = sweep["parameter"].split(".")
    allowed = {("satisdive", "delta"), ("sgi", "lambda_score"), ("sgi", "initial_candidates")}
    if (block, parameter) not in allowed or block != cfg["method"]:
        raise ValueError("Unsupported sweep")
    for value in sweep["values"]:
        point = copy.deepcopy(cfg)
        point[block].update(sweep.get("fixed", {})); point[block][parameter] = value
        argv = ["--config", str(base), "--dataset", a.dataset, "--subset-size", str(sweep["subset_size"]),
                "--device", a.device, "--reward-device", str(a.reward_device), "--resume",
                "--shard-index", str(a.shard_index), "--num-shards", str(a.num_shards), "--outputs", str(a.outputs)]
        if a.models_lock:
            argv += ["--models-lock", str(a.models_lock)]
        if a.profile_flops:
            argv.append("--profile-flops")
        if a.limit is not None:
            argv += ["--limit", str(a.limit)]
        if a.plan:
            argv += ["--plan"]
        execute(generation_parser().parse_args(argv), config_override=point)
        # Release previous adapter cycles/caches before loading the next sweep point.
        import gc
        gc.collect()
        if not a.plan:
            import torch
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
