"""Metric identities include code, settings, model revisions and external asset hashes."""
from pathlib import Path
import os
import socket
import sys
import yaml
from benchmark.common import ROOT, digest, read_json, file_sha256
from benchmark.config import MODEL_IDS

REWARDS = ("hpsv3", "hpsv2", "image_reward", "pickscore", "clip_score")
METRICS = (*REWARDS, "dreamsim", "dino", "lpips")
MODEL_KEYS = {"hpsv3": ("hpsv3", "qwen"), "hpsv2": ("hpsv2",),
              "image_reward": ("image_reward", "bert_tokenizer"),
              "pickscore": ("pickscore", "pickscore_processor"), "clip_score": ("clip",),
              "dino": ("dino",), "lpips": (), "dreamsim": ()}
ENV_VARS = {"hpsv3": "TTS_HPSV3_PYTHON", "hpsv2": "TTS_HPSV2_PYTHON",
            "image_reward": "TTS_IMAGEREWARD_PYTHON"}


def metric_python(name):
    variable = ENV_VARS.get(name, "TTS_EVAL_PYTHON")
    path = os.environ.get(variable)
    if not path and name == "hpsv3":
        path = os.environ.get("SATISDIVE_REWARD_PYTHON")
    if not path or not Path(path).is_file():
        raise RuntimeError(f"Set {variable} to the matching environment's bin/python (see README)")
    return str(Path(path).absolute())


def load_eval_config(path):
    c = yaml.safe_load(Path(path).read_text())
    if c.get("schema") != 1 or not c["metrics"] or not set(c["metrics"]) <= set(METRICS) or len(set(c["metrics"])) != len(c["metrics"]):
        raise ValueError("Evaluation config must define each shared metric exactly once")
    expected = {"pickscore_definition": "scaled_cosine_logit", "clip_preprocess": "sgi_bilinear_224",
                "lpips_net": "alex",
                "lpips_version": "0.1", "lpips_resize": 256, "dreamsim_type": "ensemble"}
    if c["vendi_kernel"] not in (None, "normalized_dino_patch_gram"):
        raise ValueError("Unsupported Vendi kernel")
    if any(c.get(k) != v for k, v in expected.items()) or c["hpsv2_version"] not in ("v2.0", "v2.1"):
        raise ValueError("Unsupported metric definition; implement and document it before changing config")
    if c["sat"]["thresholds"] != {"pickapic": 9.66, "hpdv2": 13.93, "geneval": None}:
        raise ValueError("Preserve published dataset floors; no recalibration or transferred GenEval floor")
    if c["sat"]["hpsv3_scale_verified"] and not c["sat"]["verification_note"]:
        raise ValueError("Record scale/checkpoint verification evidence before enabling SAT")
    b = c["bootstrap"]
    if not isinstance(b["resamples"], int) or b["resamples"] < 1 or not 0 < b["confidence"] < 1:
        raise ValueError("Invalid bootstrap settings")
    return c


def metric_identity(name, config, lock, verify_assets=False):
    models = {key: lock["models"][key] for key in MODEL_KEYS[name]}
    for key, item in models.items():
        if item["id"] != MODEL_IDS[key] or len(item["revision"]) != 40 or any(c not in "0123456789abcdef" for c in item["revision"]):
            raise ValueError(f"Metric model must be revision-pinned: {key}")
    assets = None
    if name in ("lpips", "dreamsim"):
        path = ROOT / "models" / f"{name}.assets.json"
        assets = read_json(path)
        if not assets.get("files") or assets.get("metric") != name:
            raise ValueError(f"Incomplete external metric asset inventory: {path}")
        if verify_assets:
            for filename, sha in assets["files"].items():
                if file_sha256(ROOT / filename) != sha:
                    raise ValueError(f"Metric asset changed: {filename}")
    paths = list((ROOT / "evaluation").glob("*.py")) + [ROOT / "methods/scoring.py", ROOT / "methods/assets.py"]
    paths += list((ROOT / "environments").glob("*.txt"))
    if name == "hpsv3":
        paths += list((ROOT / "third_party/satisdive/third_party/HPSv3").rglob("*.py"))
        paths += list((ROOT / "third_party/satisdive/third_party/HPSv3").rglob("*.yaml"))
    identity = {"schema": 1, "metric": name, "models": models, "assets": assets,
                "definition": {k: v for k, v in config.items() if k not in ("metrics", "sat", "bootstrap")},
                "source": {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}}
    return digest(identity)[:20], identity


def offline_network():
    """Missing caches fail instead of silently contacting model hosts during evaluation."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    original = socket.socket.connect
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError("Network disabled during offline evaluation; prepare model assets on the VM first")
        return original(sock, address)
    socket.socket.connect = connect


def model_cache_environment():
    os.environ["TORCH_HOME"] = str(ROOT / "models/torch")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def group_input_hash(group):
    return digest({"config_hash": group["config_hash"], "prompt": group["prompt"], "images": group["images"]})


def valid_cache(path, metric_id, input_hash):
    try:
        value = read_json(path)
        return value["status"] == "complete" and value["metric_id"] == metric_id and value["input_hash"] == input_hash
    except (OSError, ValueError, KeyError, TypeError):
        return False
