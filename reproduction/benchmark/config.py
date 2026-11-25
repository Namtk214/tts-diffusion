from __future__ import annotations
import copy
import math
from pathlib import Path
import yaml
from benchmark.common import ROOT, digest, read_json

MODEL_IDS = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "tiny_vae": "madebyollin/taef1",
    "clip": "openai/clip-vit-base-patch32",
    "dino": "facebook/dinov2-base",
    "hpsv3": "MizzenAI/HPSv3",
    "qwen": "Qwen/Qwen2-VL-7B-Instruct",
    "hpsv2": "xswu/HPSv2",
    "image_reward": "THUDM/ImageReward",
    "pickscore": "yuvalkirstain/PickScore_v1",
    "pickscore_processor": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "bert_tokenizer": "google-bert/bert-base-uncased",
}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    return validate_config(cfg)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    c = copy.deepcopy(config)
    method = c.get("method")
    if method not in ("satisdive", "sgi") or c.get("protocol") not in ("native", "unified", "single_gpu_768"):
        raise ValueError("Unknown benchmark protocol or method")
    # An inactive block cannot affect either the run hash or execution.
    c.pop("sgi" if method == "satisdive" else "satisdive", None)
    allowed = {"protocol", "method", "backbone", "generation", method, "models_lock"}
    if set(c) - allowed:
        raise ValueError(f"Unknown config fields: {set(c) - allowed}")
    b, g = c["backbone"], c["generation"]
    nested = {"backbone": {"id", "revision", "height", "width", "steps", "guidance_scale", "dtype"},
              "generation": {"seed", "group_size", "final_decoder", "sampler"},
              "sgi": {"reward", "diversity", "initial_candidates", "dropping_ratio", "lambda_score", "preview_decoder", "solver"},
              "satisdive": {"reward", "feature_layer", "update_multiplier", "noise_amplitude", "delta", "update_steps"}}
    for name in ("backbone", "generation", method):
        unknown = set(c[name]) - nested[name]
        missing = nested[name] - set(c[name]) - ({"revision"} if name == "backbone" else set())
        if unknown or missing:
            raise ValueError(f"Invalid {name} keys; unknown={unknown}, missing={missing}")
    if b["id"] != MODEL_IDS["flux"] or b["dtype"] != "bfloat16":
        raise ValueError("Initial scope requires FLUX.1-dev / bfloat16")
    if g["group_size"] != 4 or g["sampler"] != "native":
        raise ValueError("Only K=4 and the method's native sampler are in scope")
    if any(not isinstance(b[k], int) or b[k] < 16 or b[k] % 16 for k in ("height", "width")):
        raise ValueError("Image dimensions must be positive multiples of 16")
    if not isinstance(b["steps"], int) or b["steps"] < 1 or b["guidance_scale"] != 3.5:
        raise ValueError("Require positive steps and native FLUX guidance 3.5")
    if not isinstance(g["seed"], int) or not 0 <= g["seed"] < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    if c["protocol"] == "unified" and (b["height"], b["width"], b["steps"], g["final_decoder"]) != (1024, 1024, 28, "standard_flux"):
        raise ValueError("Unified protocol requires 1024 square, 28 steps, standard FLUX final VAE")
    if c["protocol"] == "single_gpu_768" and (b["height"], b["width"], b["steps"], g["final_decoder"]) != (768, 768, 28, "standard_flux"):
        raise ValueError("Single GPU protocol requires 768 square, 28 steps and standard FLUX VAE")
    if method == "satisdive":
        s = c[method]
        if s["reward"] != "hpsv3" or s["feature_layer"] != 12 or s["update_multiplier"] != 1.0 or s["noise_amplitude"] != 0.3:
            raise ValueError("Changing the native SatisDive objective/settings is outside scope")
        if not math.isfinite(s["delta"]) or s["delta"] < 0:
            raise ValueError("Delta must be finite and nonnegative")
        if s["update_steps"] != [4, 8, 12, 16, 20, 24] or max(s["update_steps"]) >= b["steps"]:
            raise ValueError("Preserve the upstream zero-based update indices [4,8,12,16,20,24]")
        if g["final_decoder"] != "standard_flux":
            raise ValueError("SatisDive uses its standard FLUX VAE")
    else:
        s = c[method]
        if (s["reward"], s["diversity"], s["solver"], s["preview_decoder"]) != ("clip", "dino_patch", "gurobi", "native"):
            raise ValueError("SGI requires native CLIP, patchwise DINO, Tiny previews and Gurobi")
        if not isinstance(s["initial_candidates"], int) or s["initial_candidates"] < 4:
            raise ValueError("SGI requires integer M >= K=4")
        if not 0 < s["dropping_ratio"] < 1:
            raise ValueError("Require 0 < dropping_ratio < 1; no pruning ablations")
        if not math.isfinite(s["lambda_score"]) or s["lambda_score"] < 0:
            raise ValueError("Lambda must be finite and nonnegative")
        remaining = s["initial_candidates"]
        for _ in range(b["steps"]):
            remaining = max(4, math.ceil(remaining * (1 - s["dropping_ratio"])))
        if remaining != 4:
            raise ValueError("Pruning schedule cannot reach K=4 within the denoising steps")
        if g["final_decoder"] not in ("native", "standard_flux"):
            raise ValueError("Unknown final decoder")
    return c


def resolve_models(cfg, lock_path=None):
    path = Path(lock_path or ROOT / cfg.get("models_lock", "configs/models.lock.json"))
    if not path.is_file():
        raise FileNotFoundError("Model lock missing. On VM run: python scripts/pin_models.py")
    lock = read_json(path)
    required = ("flux", "hpsv3", "qwen") if cfg["method"] == "satisdive" else ("flux", "tiny_vae", "clip", "dino")
    for key in required:
        entry = lock["models"][key]
        rev = entry.get("revision", "")
        if entry["id"] != MODEL_IDS[key] or len(rev) != 40 or any(x not in "0123456789abcdef" for x in rev):
            raise ValueError(f"Unpinned/invalid model: {key}")
    cfg = copy.deepcopy(cfg)
    explicit = cfg["backbone"].get("revision")
    if explicit and explicit != lock["models"]["flux"]["revision"]:
        raise ValueError("Backbone revision disagrees with model lock")
    cfg["backbone"]["revision"] = lock["models"]["flux"]["revision"]
    cfg["model_revisions"] = {k: lock["models"][k] for k in required}
    return cfg


def config_hash(cfg, manifest_hash, implementation_hash):
    return digest({"schema": 1, "config": cfg, "prompts": manifest_hash,
                   "implementation": implementation_hash})[:20]
