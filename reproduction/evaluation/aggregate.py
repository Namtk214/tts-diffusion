"""Prompt-level estimates, coverage and bootstrap intervals; missing is never zero."""
import csv
import io
from pathlib import Path
import numpy as np
from benchmark.common import atomic_text, atomic_json, read_json, read_manifest, digest
from benchmark.storage import validate_group
from evaluation.protocol import REWARDS, METRICS, group_input_hash, valid_cache, metric_identity
from evaluation.diversity import PAIRS


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, out.getvalue())


def bootstrap(values, resamples=1000, seed=12345, confidence=0.95, categories=None):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Bootstrap accepts only finite prompt-level observations")
    if len(x) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    if categories is None:
        draws = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
        estimate = x.mean()
    else:
        labels = np.asarray(categories)
        if labels.shape != x.shape:
            raise ValueError("Category/observation length mismatch")
        # GenEval official overall is the unweighted mean of task accuracies.
        parts = [x[labels == category] for category in sorted(set(categories))]
        draws = np.stack([p[rng.integers(0, len(p), size=(resamples, len(p)))].mean(axis=1) for p in parts]).mean(axis=0)
        estimate = np.mean([p.mean() for p in parts])
    alpha = (1-confidence)/2
    low, high = np.quantile(draws, [alpha, 1-alpha])
    return float(estimate), float(low), float(high)


def summarize_rows(groups, manifest, eval_id, config):
    metrics = [f"{name}_{agg}" for name in REWARDS if name in config["metrics"] for agg in ("mean", "min")]
    metrics += ["hpsv3_sat", "dreamsim", "dino_distance", "lpips", "vendi", "geneval_accuracy",
                "runtime_s", "candidate_weighted_forward_nfe", "denoiser_forward_calls", "guidance_autograd_calls",
                "reward_forward_calls", "reward_backward_calls", "allocated_gpu_seconds", "profiled_flops",
                "peak_allocated_generator_bytes", "peak_allocated_reward_worker_bytes", "resident_peak_sum_mib",
                "peak_reserved_generator_bytes", "peak_reserved_reward_worker_bytes", "gpu_count", "solver_runtime_s", "solver_time_limit_hits"]
    if not config["vendi_kernel"]:
        metrics.remove("vendi")
    output = []
    expected = len(groups)
    base = {"run_id": manifest["run_id"], "eval_id": eval_id, "method": manifest["method"],
        "dataset": manifest["dataset"], "protocol": manifest["protocol"], "expected_prompts": expected,
        "generated_prompts": sum(g["generation_status"] == "complete" for g in groups),
        "fully_evaluated_prompts": sum(g["evaluation_status"] == "complete" for g in groups)}
    for name in metrics:
        usable = [g for g in groups if isinstance(g.get(name), (int, float)) and np.isfinite(g[name])]
        labels = None
        note = "prompt bootstrap; available groups only, inspect coverage"
        if name == "geneval_accuracy" and usable:
            if set(g["category"] for g in usable) != set(g["category"] for g in groups):
                usable = []
                note = "unresolved: at least one expected task has no evaluated prompts"
            else:
                labels = [g["category"] for g in usable]
                note = "official mean over tasks; stratified prompt bootstrap within each task"
        if name == "hpsv3_sat" and not usable:
            note = "N/A: no matched floor" if manifest["dataset"] == "geneval" else "unresolved: HPSv3 scale not verified or scores unavailable"
        if name == "profiled_flops":
            note = "partial supported generator-process ops; excludes reward worker/solver/unsupported ops"
        value, low, high = bootstrap([g[name] for g in usable], categories=labels, **config["bootstrap"])
        output.append({**base, "metric": name, "mean": value, "ci_low": low, "ci_high": high,
                       "n_prompts": len(usable), "missing_prompts": expected-len(usable), "note": note})
    if manifest["dataset"] == "geneval":
        for category in sorted(set(g["category"] for g in groups)):
            category_groups = [g for g in groups if g["category"] == category]
            values = [g["geneval_accuracy"] for g in category_groups if g.get("geneval_accuracy") is not None]
            value, low, high = bootstrap(values, **config["bootstrap"])
            output.append({**base, "metric": f"geneval/{category}", "mean": value, "ci_low": low, "ci_high": high,
                "n_prompts": len(values), "missing_prompts": len(category_groups)-len(values), "note": "task accuracy; prompt bootstrap"})
    return output


def aggregate_run(run_dir, config, lock):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    prompts = read_manifest(run_dir / "prompts.jsonl")
    ids, unresolved = {}, {}
    for metric in config["metrics"]:
        try:
            ids[metric] = metric_identity(metric, config, lock)[0]
        except (OSError, ValueError, KeyError) as error:
            unresolved[metric] = str(error)
    geneval_index = run_dir / "evaluation/geneval/index.json"
    if manifest["dataset"] == "geneval" and geneval_index.is_file():
        ids["geneval"] = read_json(geneval_index)["metric_id"]
    eid = digest({"ids": ids, "config": config})[:20]
    group_rows, image_rows, pair_rows = [], [], []
    size = (manifest["config"]["backbone"]["width"], manifest["config"]["backbone"]["height"])
    for row in prompts:
        pid = row["prompt_id"]
        base = {"run_id": manifest["run_id"], "prompt_id": pid}
        result = {**base, "category": row.get("category", ""), "generation_status": "not_run",
                  "evaluation_status": "not_run"}
        source = run_dir / "images" / pid
        if not validate_group(source, manifest["config_hash"], row, size):
            status_path = run_dir / "records" / f"{pid}.json"
            result["generation_status"] = read_json(status_path).get("status", "not_run") if status_path.is_file() else "not_run"
            if result["generation_status"] == "complete":
                result["generation_status"] = "corrupt"
            group_rows.append(result)
            continue
        group = read_json(source / "group.json")
        ih, stats = group_input_hash(group), group["stats"]
        result["generation_status"] = "complete"
        for key in ("runtime_s", "candidate_weighted_forward_nfe", "denoiser_forward_calls", "guidance_autograd_calls",
                    "reward_forward_calls", "reward_backward_calls", "allocated_gpu_seconds", "gpu_count", "profiled_flops", "flops_coverage",
                    "solver_runtime_s", "solver_time_limit_hits"):
            result[key] = stats.get(key)
        result["peak_allocated_generator_bytes"] = max(stats.get("peak_allocated_bytes", {}).values(), default=None)
        result["peak_allocated_reward_worker_bytes"] = stats.get("reward_worker_peak_allocated_bytes")
        result["peak_reserved_generator_bytes"] = max(stats.get("peak_reserved_bytes", {}).values(), default=None)
        result["peak_reserved_reward_worker_bytes"] = stats.get("reward_worker_peak_reserved_bytes")
        result["resident_peak_sum_mib"] = stats.get("resident_gpu_memory", {}).get("peak_sum_mib")
        images = [{**base, "candidate_id": im["candidate_id"], "image_path": str((source / im["filename"]).relative_to(run_dir))} for im in group["images"]]
        pairs = [{**base, "candidate_a": group["images"][i]["candidate_id"], "candidate_b": group["images"][j]["candidate_id"]} for i, j in PAIRS]
        missing = []
        for metric in (*config["metrics"], *(("geneval",) if manifest["dataset"] == "geneval" else ())):
            mid = ids.get(metric)
            root = run_dir / ("evaluation/geneval" if metric == "geneval" else f"evaluation/cache/{metric}")
            path = root / (mid or "missing") / f"{pid}.json"
            if mid is None or not valid_cache(path, mid, ih):
                missing.append(metric)
                continue
            data = read_json(path)
            if metric in REWARDS or metric == "geneval":
                values = data["image_values"]
                if len(values) != 4 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid reward cache {path}")
                for i in range(4):
                    images[i][metric] = values[i]
                if metric == "geneval":
                    result["geneval_accuracy"] = float(np.mean(values))
                else:
                    result[f"{metric}_mean"] = float(np.mean(values))
                    result[f"{metric}_min"] = float(np.min(values))
                if metric == "hpsv3":
                    floor = config["sat"]["thresholds"][manifest["dataset"]]
                    result["hpsv3_threshold"] = floor
                    result["hpsv3_sat"] = float(np.mean(np.asarray(values) >= floor)) if floor is not None and config["sat"]["hpsv3_scale_verified"] else None
            else:
                values = data["pair_values"]
                if len(values) != 6 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid pairwise cache {path}")
                key = "dino_distance" if metric == "dino" else metric
                for i in range(6):
                    pairs[i][key] = values[i]
                result[key] = float(np.mean(values))
                if metric == "dino" and config["vendi_kernel"]:
                    result["vendi"] = data["vendi"]
        result["missing_metrics"] = ";".join(missing)
        result["evaluation_status"] = "complete" if not missing else "incomplete"
        group_rows.append(result); image_rows.extend(images); pair_rows.extend(pairs)
    directory = run_dir / "evaluation/reports" / eid
    write_csv(directory / "image_metrics.csv", image_rows,
              ["run_id", "prompt_id", "candidate_id", "image_path", *REWARDS, "geneval"])
    write_csv(directory / "pairwise_metrics.csv", pair_rows,
              ["run_id", "prompt_id", "candidate_a", "candidate_b", "dreamsim", "dino_distance", "lpips"])
    write_csv(directory / "group_metrics.csv", group_rows)
    summary = summarize_rows(group_rows, manifest, eid, config)
    write_csv(directory / "summary.csv", summary)
    atomic_json(directory / "report_manifest.json", {"run_id": manifest["run_id"], "eval_id": eid,
        "metric_ids": ids, "unresolved_metric_assets": unresolved, "config": config,
        "coverage": {"expected": len(prompts), "generated": len(image_rows)//4,
                     "fully_evaluated": sum(g["evaluation_status"] == "complete" for g in group_rows)},
        "scope": "subset; adapted metrics are documented in METRICS.md"})
    atomic_json(run_dir / "evaluation/latest_report.json", {"eval_id": eid, "directory": str(directory.relative_to(run_dir))})
    return directory, group_rows, summary
