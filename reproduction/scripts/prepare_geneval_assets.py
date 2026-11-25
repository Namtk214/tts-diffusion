"""Explicit VM-only download of pinned official GenEval code and detector/CLIP weights."""
import argparse
from pathlib import Path
import subprocess
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, atomic_json, file_sha256, environment_info
from evaluation.geneval_eval import GENEVAL_COMMIT

DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
URL = f"https://download.openmmlab.com/mmdetection/v2.0/mask2former/{DETECTOR}/{DETECTOR}_20220504_001756-743b7d99.pth"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    import requests
    import open_clip
    from open_clip.pretrained import get_pretrained_cfg, download_pretrained
    import mmdet
    root = ROOT / "external/geneval"
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        subprocess.run(["git", "clone", "https://github.com/djghosh13/geneval.git", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", GENEVAL_COMMIT], check=True)
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != GENEVAL_COMMIT:
        raise RuntimeError("Existing external/geneval is at a different revision; inspect it before changing")
    directory = ROOT / "models/geneval"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{DETECTOR}.pth"
    if not target.exists():
        temporary = target.with_suffix(".download")
        with requests.get(URL, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            with open(temporary, "wb") as f:
                for block in response.iter_content(1024*1024):
                    f.write(block)
        os.replace(temporary, target)
    pretrained = get_pretrained_cfg("ViT-L-14", "openai")
    clip_path = Path(download_pretrained(pretrained, cache_dir=str(directory)))
    expected = directory / "ViT-L-14.pt"
    if clip_path != expected:
        import shutil
        shutil.copyfile(clip_path, expected)
    model_config = Path(mmdet.__file__).parent / ".mim/configs/mask2former" / f"{DETECTOR}.py"
    if not model_config.is_file():
        model_config = Path(mmdet.__file__).parent.parent / "configs/mask2former" / f"{DETECTOR}.py"
    if not model_config.is_file():
        raise FileNotFoundError("MMDetection mask2former config missing; install the official mmdet 2.28.2 package with configs")
    atomic_json(ROOT / "models/geneval.assets.json", {"schema": 1, "commit": GENEVAL_COMMIT,
        "model_config": str(model_config.absolute()), "model_config_sha256": file_sha256(model_config),
        "detector_url": URL, "clip": pretrained, "environment": environment_info(),
        "files": {str(f.relative_to(ROOT)): file_sha256(f) for f in (target, expected)}})


if __name__ == "__main__":
    main()
