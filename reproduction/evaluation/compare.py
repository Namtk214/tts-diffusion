"""Matched-prompt, paired bootstrap; never pair by file ordering alone."""
import csv
from pathlib import Path
from benchmark.common import read_json, read_manifest
from evaluation.aggregate import bootstrap, write_csv


def compare_runs(left, right, destination):
    left, right = Path(left), Path(right)
    manifests = [read_json(p / "run_manifest.json") for p in (left, right)]
    if any(manifests[0][k] != manifests[1][k] for k in ("dataset", "protocol", "prompts_sha256")):
        raise ValueError("Paired comparison requires the same dataset, protocol and complete prompt manifest")
    for key in ("width", "height", "steps", "guidance_scale", "revision"):
        if manifests[0]["config"]["backbone"][key] != manifests[1]["config"]["backbone"][key]:
            raise ValueError(f"Paired method comparison has different backbone setting: {key}")
    if manifests[0]["method"] == manifests[1]["method"]:
        raise ValueError("This command compares SatisDive and SGI, not sweep points")
    directories = [p / read_json(p / "evaluation/latest_report.json")["directory"] for p in (left, right)]
    reports = [read_json(p / "report_manifest.json") for p in directories]
    if reports[0]["eval_id"] != reports[1]["eval_id"]:
        # GenEval identity is present only for GenEval, and must match too.
        raise ValueError("Metric identities/configurations differ; reevaluate both with the same definitions")
    groups = []
    for directory in directories:
        with open(directory / "group_metrics.csv", newline="") as f:
            groups.append({r["prompt_id"]: r for r in csv.DictReader(f)})
    with open(directories[0] / "summary.csv", newline="") as f:
        metrics = [r["metric"] for r in csv.DictReader(f) if not r["metric"].startswith("geneval/")]
    out = []
    for metric in metrics:
        ids = sorted(pid for pid in groups[0].keys() & groups[1].keys()
                     if groups[0][pid].get(metric, "") != "" and groups[1][pid].get(metric, "") != "")
        differences = [float(groups[0][pid][metric])-float(groups[1][pid][metric]) for pid in ids]
        categories = [groups[0][pid]["category"] for pid in ids] if metric == "geneval_accuracy" else None
        if categories and set(categories) != set(r.get("category", "") for r in read_manifest(left / "prompts.jsonl")):
            ids, differences, categories = [], [], None
        mean, low, high = bootstrap(differences, categories=categories, **reports[0]["config"]["bootstrap"])
        out.append({"left": manifests[0]["run_id"], "right": manifests[1]["run_id"],
            "direction": f"{manifests[0]['method']}-{manifests[1]['method']}", "metric": metric,
            "paired_delta": mean, "ci_low": low, "ci_high": high, "matched_prompts": len(ids),
            "expected_prompts": manifests[0]["prompt_count"],
            "left_available": sum(g.get(metric, "") != "" for g in groups[0].values()),
            "right_available": sum(g.get(metric, "") != "" for g in groups[1].values()),
            "matched_prompt_ids": ";".join(ids)})
    write_csv(destination, out)
    return out
