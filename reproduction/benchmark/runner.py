from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import traceback
import contextlib
import yaml
from benchmark.common import (ROOT, atomic_json, atomic_text, digest, environment_info, exclusive_lock,
    file_sha256, prompt_seed, read_json, read_manifest, shard_rows, source_fingerprint, write_jsonl)
from benchmark.config import load_config, validate_config, resolve_models, config_hash
from benchmark.storage import commit_group, validate_group
from evaluation.efficiency import ResidentMemorySampler


def parser():
    p = argparse.ArgumentParser(description="Generate SatisDive/SGI groups on a CUDA VM")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("pickapic", "geneval", "hpdv2"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--subset-size", type=int, help="Default: 30 for single_gpu_768, 100 for other protocols")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reuse-run", type=Path, help="Reuse valid prompt groups from a prior compatible run")
    p.add_argument("--reuse-equivalent-code-change", help="Allow reuse across code fingerprints and record the specific equivalence reason")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", type=int, help="Logical GPU (single_gpu_768: generator GPU; other protocols: 1)")
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--models-lock", type=Path)
    p.add_argument("--plan", action="store_true", help="Validate config/manifest/lock; do not import model libraries")
    p.add_argument("--profile-flops", action="store_true", help="Partial PyTorch profiler coverage; perturbs timing")
    return p


def plan(args, config_override=None):
    cfg = validate_config(config_override) if config_override is not None else load_config(args.config)
    generator_index = int(args.device.partition(":")[2] or 0)
    shared = cfg["protocol"] == "single_gpu_768"
    if args.reward_device is None:
        args.reward_device = generator_index if shared else 1
    if shared and cfg["method"] == "satisdive" and args.reward_device != generator_index:
        raise ValueError("single_gpu_768 uses the generator GPU for HPSv3")
    cfg = resolve_models(cfg, args.models_lock)
    subset_size = args.subset_size if args.subset_size is not None else (30 if shared else 100)
    manifest = args.manifest or ROOT / "data" / f"{args.dataset}_{subset_size}.jsonl"
    rows = read_manifest(manifest)
    if any(r["dataset"] != args.dataset for r in rows):
        raise ValueError("Dataset flag disagrees with prompt manifest")
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise ValueError("limit must be between 1 and manifest size")
        rows = rows[:args.limit]
    worker_rows = shard_rows(rows, args.shard_index, args.num_shards)
    prompts_hash = digest(rows)
    # Profiling changes observed runtime and must not be silently mixed.
    cfg["instrumentation"] = {"profile_flops": args.profile_flops}
    code_hash = source_fingerprint()
    provenance_path = manifest.parent / f"{args.dataset}.provenance.json"
    run_hash = config_hash(cfg, prompts_hash, code_hash)
    directory = args.outputs / cfg["protocol"] / args.dataset / cfg["method"] / run_hash
    return cfg, rows, worker_rows, directory, {
        "schema": 1, "run_id": run_hash, "config_hash": run_hash, "dataset": args.dataset,
        "protocol": cfg["protocol"], "method": cfg["method"], "prompt_count": len(rows),
        "prompts_sha256": prompts_hash, "source_manifest_sha256": file_sha256(manifest),
        "source_manifest": str(manifest), "implementation_sha256": code_hash,
        "dataset_provenance": read_json(provenance_path) if provenance_path.is_file() else None,
        "upstream": read_json(ROOT / "third_party/upstream.lock.json"),
        "model_revisions": cfg["model_revisions"], "config": cfg,
        "scope": "subset_pilot_not_full_paper_reproduction", "status": "planned"}


def refresh_stats(directory, rows):
    records = []
    for r in rows:
        path = directory / "records" / f"{r['prompt_id']}.json"
        records.append(read_json(path) if path.is_file() else {"prompt_id": r["prompt_id"], "status": "not_run"})
    write_jsonl(directory / "generation_stats.jsonl", records)
    return {s: sum(r["status"] == s for r in records) for s in ("complete", "failed", "not_run")}


def reuse_run(source, destination, manifest, rows, size, equivalent_code_change=None):
    from PIL import Image
    old = read_json(source / "run_manifest.json")
    old_cfg, new_cfg = dict(old["config"]), dict(manifest["config"])
    old_cfg.pop("protocol"); new_cfg.pop("protocol")
    code_changed = old["implementation_sha256"] != manifest["implementation_sha256"]
    if old_cfg != new_cfg or old["prompts_sha256"] != manifest["prompts_sha256"]:
        raise ValueError("Reuse requires identical scientific configuration and prompt manifest")
    if code_changed and not equivalent_code_change:
        raise ValueError("Implementation differs; pass --reuse-equivalent-code-change with a specific, auditable reason only when generated groups remain equivalent")
    if old["status"] != "complete" and not equivalent_code_change:
        raise ValueError("Only complete runs can be reused without an explicit code equivalence note")
    reused_prompt_ids = []
    for row in rows:
        with exclusive_lock(destination / ".locks" / f"{row['prompt_id']}.lock"):
            if validate_group(destination / "images" / row["prompt_id"], manifest["config_hash"], row, size):
                reused_prompt_ids.append(row["prompt_id"])
                continue
            folder = source / "images" / row["prompt_id"]
            if not validate_group(folder, old["config_hash"], row, size):
                if equivalent_code_change:
                    continue
                raise ValueError(f"Source group is missing or corrupt: {row['prompt_id']}")
            previous = read_json(folder / "group.json")
            info = {k: v for k, v in previous.items() if k not in ("images", "prompt", "config_hash", "status", "schema")}
            info["reused_from"] = {"run_id": old["run_id"], "directory": str(source.absolute()),
                                   "note": "Same measured generation; not an independent timing replicate"}
            images = []
            for item in previous["images"]:
                with Image.open(folder / item["filename"]) as im:
                    images.append(im.convert("RGB"))
            commit_group(destination, row, images, info, manifest["config_hash"], size)
            atomic_json(destination / "records" / f"{row['prompt_id']}.json",
                        {"prompt_id": row["prompt_id"], "status": "complete", **info["stats"], "reused_from": old["run_id"]})
            reused_prompt_ids.append(row["prompt_id"])
    atomic_json(destination / "reuse_provenance.json", {
        "source_manifest": old,
        "source_directory": str(source.absolute()),
        "source_implementation_sha256": old["implementation_sha256"],
        "destination_implementation_sha256": manifest["implementation_sha256"],
        "equivalent_code_change": equivalent_code_change,
        "reused_prompt_ids": reused_prompt_ids,
    })


def execute(args, config_override=None, adapter_factory=None):
    cfg, rows, worker_rows, directory, manifest = plan(args, config_override)
    if args.plan:
        print(yaml.safe_dump({"run_dir": str(directory), "worker_prompts": len(worker_rows), **manifest}, sort_keys=False))
        return directory
    with exclusive_lock(directory / ".locks/manifest.lock", timeout=30):
        path = directory / "run_manifest.json"
        if path.exists():
            old = read_json(path)
            if old["config_hash"] != manifest["config_hash"] or old["prompts_sha256"] != manifest["prompts_sha256"]:
                raise ValueError("Refusing to mix configurations or prompt manifests")
            if not args.resume:
                raise ValueError("Run exists; use --resume to validate and reuse complete groups")
        else:
            atomic_json(path, manifest)
            atomic_text(directory / "config.resolved.yaml", yaml.safe_dump(cfg, sort_keys=False))
            write_jsonl(directory / "prompts.jsonl", rows)
    size = (cfg["backbone"]["width"], cfg["backbone"]["height"])
    if args.reuse_run:
        reuse_run(args.reuse_run, directory, manifest, worker_rows, size,
                  equivalent_code_change=args.reuse_equivalent_code_change)
    elif args.reuse_equivalent_code_change is not None:
        raise ValueError("--reuse-equivalent-code-change requires --reuse-run")
    pending = [r for r in worker_rows if not ((args.resume or args.reuse_run) and validate_group(directory / "images" / r["prompt_id"],
                                                manifest["config_hash"], r, size))]
    # Repair status records from committed groups after a crash between commit and status write.
    for row in worker_rows:
        if row not in pending:
            g = read_json(directory / "images" / row["prompt_id"] / "group.json")
            atomic_json(directory / "records" / f"{row['prompt_id']}.json",
                        {"prompt_id": row["prompt_id"], "status": "complete", **g["stats"]})
    adapter = None
    worker_tag = f"{args.shard_index}-of-{args.num_shards}"
    worker_lock = contextlib.ExitStack()
    try:
        if pending:
            worker_lock.enter_context(exclusive_lock(directory / ".locks" / f"worker-{worker_tag}.lock"))
            if adapter_factory is None:
                if cfg["method"] == "sgi":
                    from methods.sgi_adapter import SGIAdapter
                    adapter_factory = SGIAdapter
                else:
                    from methods.satisdive_adapter import SatisDiveAdapter
                    adapter_factory = SatisDiveAdapter
            start = time.perf_counter()
            adapter = adapter_factory(cfg, device=args.device, reward_device=args.reward_device,
                work_dir=directory / "workers" / worker_tag, profile_flops=args.profile_flops)
            environment_path = directory / "workers" / worker_tag / "environment.json"
            current_environment = {
                **environment_info(), **adapter.metadata, "cold_start_s": time.perf_counter() - start,
                "device": args.device, "reward_device": args.reward_device if cfg["method"] == "satisdive" else None}
            for previous in (directory / "workers").glob("*/environment.json"):
                old_environment = read_json(previous)
                if old_environment["packages"] != current_environment["packages"] or old_environment.get("reward_worker") != current_environment.get("reward_worker"):
                    raise ValueError("Package/reward-worker environment differs from this run; use its recorded environment")
            atomic_json(environment_path, current_environment)
        for row in pending:
            pid = row["prompt_id"]
            try:
                with exclusive_lock(directory / ".locks" / f"{pid}.lock"):
                    if args.resume and validate_group(directory / "images" / pid, manifest["config_hash"], row, size):
                        continue
                    device_indices = [int(args.device.split(":")[-1]) if ":" in args.device else 0]
                    if cfg["method"] == "satisdive" and args.reward_device not in device_indices:
                        device_indices.append(args.reward_device)
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                    physical = [visible[i].strip() if visible != [""] else str(i) for i in device_indices]
                    sampler = ResidentMemorySampler(physical); sampler.start()
                    try:
                        images, info = adapter.generate_group(row["prompt"], prompt_seed(cfg["generation"]["seed"], row))
                    finally:
                        memory = sampler.finish()
                    info["stats"]["resident_gpu_memory"] = memory
                    commit_group(directory, row, images, info, manifest["config_hash"], size)
                    atomic_json(directory / "records" / f"{pid}.json", {"prompt_id": pid, "status": "complete", **info["stats"]})
                    print(f"complete {pid}", flush=True)
            except FileExistsError:
                print(f"locked {pid}; another worker owns it", flush=True)
            except Exception as e:
                atomic_json(directory / "records" / f"{pid}.json", {"prompt_id": pid, "status": "failed",
                    "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
                print(f"failed {pid}: {e}", flush=True)
                # Worker/model failures must be visible; next prompt may recover from an OOM.
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
    except Exception as e:
        atomic_json(directory / "workers" / worker_tag / "failure.json", {"status": "failed", "error": str(e),
                     "traceback": traceback.format_exc()})
        raise
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                atomic_json(directory / "workers" / worker_tag / "cleanup_failure.json", {"error": str(error)})
        worker_lock.close()
        try:
            with exclusive_lock(directory / ".locks/manifest.lock", timeout=30):
                counts = refresh_stats(directory, rows)
                current = read_json(directory / "run_manifest.json")
                current.update({"coverage": counts, "status": "complete" if counts["complete"] == len(rows) else "incomplete"})
                atomic_json(directory / "run_manifest.json", current)
        except FileExistsError:
            pass
    print(directory)
    return directory


def main():
    a = parser().parse_args()
    try:
        directory = execute(a)
        if not a.plan:
            records = read_json(directory / "run_manifest.json").get("coverage", {})
            if records.get("failed"):
                return 1
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}", flush=True)
        return 2
