"""One metric/model per subprocess, loaded once for its shard of prompts."""
import argparse
from pathlib import Path
import time
import traceback
import math
from PIL import Image
from benchmark.common import ROOT, atomic_json, read_json, read_manifest, shard_rows, exclusive_lock, environment_info
from benchmark.storage import validate_group
from evaluation.protocol import (METRICS, REWARDS, load_eval_config, metric_identity, offline_network,
                                 model_cache_environment, group_input_hash, valid_cache)


def build_scorer(name, models, config, device, work_dir):
    if name in REWARDS:
        from evaluation.rewards import RewardScorer
        return RewardScorer(name, models, config, device, work_dir)
    from evaluation.diversity import DiversityScorer
    return DiversityScorer(name, models, config, device, work_dir)


def evaluate_metric(args):
    config = load_eval_config(args.eval_config)
    lock = read_json(args.models_lock)
    mid, identity = metric_identity(args.metric, config, lock, verify_assets=True)
    directory = args.run_dir / "evaluation/cache" / args.metric / mid
    manifest = read_json(args.run_dir / "run_manifest.json")
    rows = shard_rows(read_manifest(args.run_dir / "prompts.jsonl"), args.shard_index, args.num_shards)
    if args.limit:
        rows = rows[:args.limit]
    size = (manifest["config"]["backbone"]["width"], manifest["config"]["backbone"]["height"])
    pending = []
    for row in rows:
        source = args.run_dir / "images" / row["prompt_id"]
        if not validate_group(source, manifest["config_hash"], row, size):
            continue  # Aggregator reports missing/corrupt generation over the full manifest.
        group = read_json(source / "group.json")
        ih = group_input_hash(group)
        if not (args.resume and valid_cache(directory / f"{row['prompt_id']}.json", mid, ih)):
            pending.append((row, group, ih, source))
    if not pending:
        return 0
    model_cache_environment()
    offline_network()
    current_environment = environment_info()
    for previous in directory.glob("worker-*.json"):
        if read_json(previous)["packages"] != current_environment["packages"]:
            raise ValueError("Metric environment versions differ from cached worker results")
    scorer = build_scorer(args.metric, lock["models"], config, args.device, directory / "assets")
    hardware = None
    if args.device.startswith("cuda"):
        import torch
        hardware = {"name": torch.cuda.get_device_name(args.device),
                    "total_memory": torch.cuda.get_device_properties(args.device).total_memory}
    atomic_json(directory / f"worker-{args.shard_index}-of-{args.num_shards}.json",
                {"metric_id": mid, "identity": identity, "scorer": scorer.metadata, "device": args.device,
                 "hardware": hardware, **current_environment})
    failed = 0
    for row, group, ih, source in pending:
        target = directory / f"{row['prompt_id']}.json"
        try:
            with exclusive_lock(directory / ".locks" / f"{row['prompt_id']}.lock"):
                if args.resume and valid_cache(target, mid, ih):
                    continue
                start = time.perf_counter()
                images = []
                for img in group["images"]:
                    with Image.open(source / img["filename"]) as image:
                        images.append(image.convert("RGB"))
                if args.metric in REWARDS:
                    values = scorer.score(images, row["prompt"])
                    result = {"image_values": values}
                    if len(values) != 4:
                        raise ValueError("Reward must return exactly four image scores")
                else:
                    result = scorer.score(images)
                    values = result["pair_values"] + ([result["vendi"]] if "vendi" in result else [])
                    if len(result["pair_values"]) != 6:
                        raise ValueError("Diversity must return exactly six unordered pair scores")
                if not all(math.isfinite(float(x)) for x in values):
                    raise ValueError("Non-finite metric output")
                atomic_json(target, {"status": "complete", "metric_id": mid, "input_hash": ih,
                    "prompt_id": row["prompt_id"], "runtime_s": time.perf_counter()-start, **result})
                print(f"{args.metric} complete {row['prompt_id']}", flush=True)
        except FileExistsError:
            print(f"{args.metric} locked {row['prompt_id']}", flush=True)
        except Exception as error:
            failed += 1
            atomic_json(target, {"status": "failed", "metric_id": mid, "input_hash": ih,
                        "prompt_id": row["prompt_id"], "error": str(error), "traceback": traceback.format_exc()})
            print(f"{args.metric} failed {row['prompt_id']}: {error}", flush=True)
    return int(failed > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", choices=METRICS, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation.yaml")
    p.add_argument("--models-lock", type=Path, default=ROOT / "configs/models.lock.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    return evaluate_metric(args)


if __name__ == "__main__":
    raise SystemExit(main())
