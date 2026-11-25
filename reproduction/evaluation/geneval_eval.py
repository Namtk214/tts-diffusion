"""Stage final PNGs for the pinned official GenEval evaluator in a separate environment."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import shutil
from benchmark.common import ROOT, atomic_json, read_json, read_manifest, file_sha256, digest, exclusive_lock, shard_rows
from benchmark.storage import validate_group
from evaluation.protocol import group_input_hash, valid_cache

GENEVAL_COMMIT = "af4902f24d3ca90ebbb446dd9891a59e0f82725f"


def evaluate_geneval(args):
    manifest = read_json(args.run_dir / "run_manifest.json")
    if manifest["dataset"] != "geneval":
        return 0
    root = ROOT / "external/geneval"
    assets_path = ROOT / "models/geneval.assets.json"
    assets = read_json(assets_path)
    if file_sha256(assets["model_config"]) != assets["model_config_sha256"]:
        raise ValueError("GenEval detector configuration changed")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != GENEVAL_COMMIT or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("GenEval evaluator must be clean at its pinned commit")
    for filename, sha in assets["files"].items():
        if file_sha256(ROOT / filename) != sha:
            raise ValueError(f"GenEval checkpoint changed: {filename}")
    identity = {"commit": commit, "assets": assets, "options": "official defaults",
                "wrapper_sha256": file_sha256(Path(__file__)),
                "child_sha256": file_sha256(ROOT / "evaluation/geneval_child.py")}
    mid = digest(identity)[:20]
    target = args.run_dir / "evaluation/geneval" / mid
    atomic_json(args.run_dir / "evaluation/geneval/index.json", {"metric_id": mid, "identity": identity})
    python = os.environ.get("TTS_GENEVAL_PYTHON")
    if not python or not Path(python).is_file():
        raise RuntimeError("Set TTS_GENEVAL_PYTHON to the separate MMDetection environment")
    rows = shard_rows(read_manifest(args.run_dir / "prompts.jsonl"), args.shard_index, args.num_shards)
    if args.limit:
        rows = rows[:args.limit]
    size = (manifest["config"]["backbone"]["width"], manifest["config"]["backbone"]["height"])
    pending = []
    for row in rows:
        source = args.run_dir / "images" / row["prompt_id"]
        if validate_group(source, manifest["config_hash"], row, size):
            group = read_json(source / "group.json")
            ih = group_input_hash(group)
            if not (args.resume and valid_cache(target / f"{row['prompt_id']}.json", mid, ih)):
                pending.append((row, group, ih, source))
    if not pending:
        return 0
    target.mkdir(parents=True, exist_ok=True)
    # One evaluator process per shard, not one expensive model load per prompt.
    with exclusive_lock(target / f"shard-{args.shard_index}-of-{args.num_shards}.lock"):
        stage = Path(tempfile.mkdtemp(prefix="input-", dir=target))
        raw = target / f"raw-{args.shard_index}-of-{args.num_shards}.jsonl"
        try:
            mapping = {}
            for idx, (row, group, ih, source) in enumerate(pending):
                folder = stage / f"{idx:06d}"
                (folder / "samples").mkdir(parents=True)
                metadata = row["metadata"]
                if metadata.get("prompt") != row["prompt"] or "tag" not in metadata:
                    raise ValueError("GenEval requires original annotation metadata for every prompt")
                atomic_json(folder / "metadata.jsonl", metadata)
                for item in group["images"]:
                    (folder / "samples" / item["filename"]).symlink_to((source / item["filename"]).absolute())
                mapping[folder.name] = (row, group, ih)
            command = [str(Path(python).absolute()), str(ROOT / "evaluation/geneval_child.py"),
                       str(root / "evaluation/evaluate_images.py"), str(ROOT / "models/geneval/ViT-L-14.pt"),
                       str(stage.absolute()), "--outfile", str(raw.absolute()),
                       "--model-path", str(ROOT / "models/geneval"),
                       "--model-config", assets["model_config"]]
            log = target / f"log-{args.shard_index}-of-{args.num_shards}.txt"
            with open(log, "w") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy())
            if result.returncode:
                raise RuntimeError(f"Official GenEval failed ({result.returncode}); see {log}")
            by_group = {key: {} for key in mapping}
            for line in raw.read_text().splitlines():
                record = json.loads(line)
                filename = Path(record["filename"])
                key, slot = filename.parent.parent.name, filename.name
                if key not in mapping or slot not in ("000.png", "001.png", "002.png", "003.png") or slot in by_group[key]:
                    raise ValueError(f"Unexpected/duplicate GenEval result: {filename}")
                if not isinstance(record["correct"], bool):
                    raise ValueError("GenEval correct must be a boolean")
                by_group[key][slot] = record
            for key, (row, group, ih) in mapping.items():
                records = by_group[key]
                if len(records) != 4:
                    raise ValueError(f"GenEval missed images for {row['prompt_id']}")
                atomic_json(target / f"{row['prompt_id']}.json", {"status": "complete", "metric_id": mid,
                    "input_hash": ih, "prompt_id": row["prompt_id"], "image_values": [int(records[f"{i:03d}.png"]["correct"]) for i in range(4)],
                    "details": [records[f"{i:03d}.png"] for i in range(4)]})
        except Exception as error:
            atomic_json(target / f"failure-{args.shard_index}.json", {"status": "failed", "error": str(error)})
            raise
        finally:
            shutil.rmtree(stage)
    return 0
