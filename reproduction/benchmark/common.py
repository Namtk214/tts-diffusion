"""Lightweight shared contracts. Importing this module never loads a model."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_manifest(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    seen = set()
    for r in rows:
        pid = r.get("prompt_id", "")
        if not isinstance(pid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            raise ValueError(f"Unsafe/missing prompt_id: {pid!r}")
        if pid in seen or not isinstance(r.get("prompt"), str) or not r["prompt"].strip():
            raise ValueError(f"Duplicate ID or empty prompt: {pid}")
        seen.add(pid)
    if not rows:
        raise ValueError("Empty prompt manifest")
    return rows


def prompt_seed(base_seed, record):
    # Independent of prompt ordering, shard, method and hyperparameter sweep.
    key = f"tts-seed-v1|{base_seed}|{record['dataset']}|{record['prompt_id']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63 - 1024)


def shard_rows(rows, index=0, count=1):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [r for i, r in enumerate(rows) if i % count == index]


@contextlib.contextmanager
def exclusive_lock(path, timeout=0):
    """Cross-process lock. Deliberately never steals a stale or remote-host lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(.1, max(0, deadline-time.monotonic())))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)


def environment_info():
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "gurobipy", "numpy",
                 "hpsv2", "hpsv3", "image-reward", "dreamsim", "lpips", "huggingface-hub",
                 "timm", "peft", "trl", "open-clip-torch", "mmdet", "mmcv-full", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cpu_model": cpu_model, "cpu_logical_cores": os.cpu_count()}


def source_fingerprint():
    paths = []
    for directory in ("benchmark", "methods", "third_party/satisdive", "third_party/sgi"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "environments").glob("*.txt"))
    paths.append(ROOT / "evaluation/efficiency.py")
    return digest({str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)})
