"""Explicit VM-only model download step. Never imported by generation/evaluation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, read_json, atomic_json, file_sha256

PATTERNS = {
    "flux": ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "text_encoder/*", "text_encoder_2/*", "transformer/*", "vae/*"],
    "hpsv3": ["HPSv3.safetensors"],
    "hpsv2": ["HPS_v2_compressed.pt", "HPS_v2.1_compressed.pt"],
    "image_reward": ["ImageReward.pt", "med_config.json"],
    "bert_tokenizer": ["tokenizer*", "vocab.txt", "config.json", "special_tokens_map.json"],
    "pickscore_processor": ["*config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-lock", type=Path, default=ROOT / "configs/models.lock.json")
    p.add_argument("--keys", nargs="+")
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation.yaml")
    a = p.parse_args()
    from huggingface_hub import snapshot_download
    lock = read_json(a.models_lock)
    import yaml
    evaluation = yaml.safe_load(a.eval_config.read_text())
    for name in a.keys or lock["models"]:
        entry = lock["models"][name]
        patterns = PATTERNS.get(name)
        if name == "hpsv2":
            patterns = [{"v2.0": "HPS_v2_compressed.pt", "v2.1": "HPS_v2.1_compressed.pt"}[evaluation["hpsv2_version"]]]
        path = Path(snapshot_download(entry["id"], revision=entry["revision"], allow_patterns=patterns,
                                     ignore_patterns=["*.onnx", "*.h5", "*.msgpack", "*.ckpt", "*.md", "*.jpg", "*.png"]))
        # Hash downloaded assets once, on the VM; record the exact files present.
        files = {str(f.relative_to(path)): file_sha256(f) for f in path.rglob("*") if f.is_file() and ".cache" not in f.parts}
        atomic_json(ROOT / "models" / f"{name}.download.json", {"model": entry, "snapshot": str(path), "files": files})
        print(f"downloaded {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
