"""Explicit Linux VM installation. Does not download benchmark data or model checkpoints."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, atomic_text

NAMES = ("controller", "sgi", "satisdive", "hpsv3", "eval", "hpsv2", "imagereward", "geneval")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=NAMES, default=list(NAMES))
    p.add_argument("--prefix", type=Path, default=ROOT / ".venvs")
    a = p.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Run this installer on the Linux VM, not the local Mac")
    if not shutil.which("uv"):
        raise RuntimeError("Install uv on the VM first (https://docs.astral.sh/uv/getting-started/installation/)")
    for name in a.only:
        directory = a.prefix.absolute() / name
        python = directory / "bin/python"
        if not python.exists():
            subprocess.run(["uv", "venv", "--seed", "--python", "3.12" if name == "satisdive" else ("3.11" if name == "sgi" else "3.10"), str(directory)], check=True)
        requirement = ROOT / ("requirements.txt" if name == "controller" else f"environments/{name}.txt")
        command = ["uv", "pip", "install", "--python", str(python), "-r", str(requirement)]
        if name == "geneval":
            command += ["--extra-index-url", "https://download.pytorch.org/whl/cu113", "--index-strategy", "unsafe-best-match",
                        "--find-links", "https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html"]
        subprocess.run(command, check=True)
        if name == "hpsv3":
            # The upstream package declares training-only dependencies (DeepSpeed).
            # Expose the vendored inference source via .pth; all actual inference
            # imports are pinned in hpsv3.txt, without installing training extras.
            site = Path(subprocess.check_output([str(python), "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip())
            atomic_text(site / "tts_vendored_hpsv3.pth", str(ROOT / "third_party/satisdive/third_party/HPSv3") + "\n")
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        atomic_text(ROOT / "reports/environment-locks" / f"{name}.txt", frozen)
    print("Environments installed. Follow README for authentication, downloads and GPU preflight.")


if __name__ == "__main__":
    main()
