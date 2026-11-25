"""E1: two methods x three 100-prompt subsets. Requires completed 50-prompt pilots."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, read_json
from benchmark.storage import validate_pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--satisdive-pilot", type=Path, required=True)
    p.add_argument("--sgi-pilot", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=("pickapic", "geneval", "hpdv2"), default=["pickapic", "geneval", "hpdv2"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", type=int, default=1)
    p.add_argument("--plan", action="store_true", help="Print commands only, no cache/model/dataset loading")
    a = p.parse_args()
    for method in ("satisdive", "sgi"):
        pilot_dir = getattr(a, f"{method}_pilot")
        if not a.plan:
            validate_pilot(pilot_dir, method)
    for dataset in a.datasets:
        for method in ("satisdive", "sgi"):
            executable = ROOT / f".venvs/{method}/bin/python"
            command = [str(executable), str(ROOT / "scripts/generate.py"), "--config", str(ROOT / f"configs/unified/{method}.yaml"),
                       "--dataset", dataset, "--subset-size", "100", "--resume", "--device", a.device,
                       "--reward-device", str(a.reward_device)]
            if a.plan:
                import shlex
                print(shlex.join(command))
            else:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
