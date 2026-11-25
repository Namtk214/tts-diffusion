from __future__ import annotations
import os
from pathlib import Path
import shutil
import tempfile
import time
from PIL import Image
from benchmark.common import atomic_json, file_sha256, read_json


def validate_pilot(directory, method, minimum=50):
    from benchmark.common import read_manifest
    directory = Path(directory)
    manifest = read_json(directory / "run_manifest.json")
    rows = read_manifest(directory / "prompts.jsonl")
    if manifest["status"] != "complete" or manifest["method"] != method or len(rows) < minimum or manifest["prompt_count"] != len(rows):
        raise ValueError(f"Require a completed {method} pilot with at least {minimum} prompts")
    size = (manifest["config"]["backbone"]["width"], manifest["config"]["backbone"]["height"])
    for row in rows:
        folder = directory / "images" / row["prompt_id"]
        if not validate_group(folder, manifest["config_hash"], row, size):
            raise ValueError(f"Invalid pilot group: {row['prompt_id']}")
        if read_json(folder / "group.json")["stats"].get("runtime_s", 0) <= 0:
            raise ValueError("Pilot needs measured generation runtime")
    return manifest


def validate_group(directory, config_hash, record, size, k=4):
    directory = Path(directory)
    try:
        group = read_json(directory / "group.json")
        if group["status"] != "complete" or group["config_hash"] != config_hash or group["prompt"] != record:
            return False
        if len(group["images"]) != k or len(list(directory.glob("*.png"))) != k:
            return False
        for i, row in enumerate(group["images"]):
            if row["filename"] != f"{i:03d}.png":
                return False
            path = directory / row["filename"]
            if file_sha256(path) != row["sha256"]:
                return False
            with Image.open(path) as im:
                if im.format != "PNG" or im.size != tuple(size):
                    return False
                im.verify()
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def commit_group(run_dir, record, images, info, config_hash, size):
    run_dir = Path(run_dir)
    if len(images) != 4 or len(info["candidate_ids"]) != 4 or len(set(info["candidate_ids"])) != 4:
        raise ValueError("Sampler must return exactly four distinct candidate slots")
    staging = run_dir / ".staging"; staging.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=record["prompt_id"]+"-", dir=staging))
    try:
        entries = []
        for i, im in enumerate(images):
            if im.size != tuple(size):
                raise ValueError(f"Unexpected image size {im.size}, expected {size}")
            path = temp / f"{i:03d}.png"
            im.convert("RGB").save(path, format="PNG")
            entries.append({"filename": path.name, "candidate_id": info["candidate_ids"][i], "sha256": file_sha256(path)})
        group = {**info, "schema": 1, "status": "complete", "config_hash": config_hash,
                 "prompt": record, "images": entries}
        atomic_json(temp / "group.json", group)
        if not validate_group(temp, config_hash, record, size):
            raise RuntimeError("Staged group did not pass integrity verification")
        destination = run_dir / "images" / record["prompt_id"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            quarantine = run_dir / "incomplete" / f"{record['prompt_id']}-{time.time_ns()}"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, quarantine)
        os.replace(temp, destination)
        return group
    finally:
        if temp.exists():
            shutil.rmtree(temp)
