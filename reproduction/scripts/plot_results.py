"""Export tables and figures only from saved evaluation reports; no invented points."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, read_json, digest
from evaluation.aggregate import write_csv


def nondominated(points):
    return [not any(x2 >= x and y2 >= y and (x2 > x or y2 > y) for x2, y2 in points) for x, y in points]


def load_reports(outputs):
    entries = []
    for path in sorted(outputs.rglob("latest_report.json")):
        run = path.parent.parent
        m = read_json(run / "run_manifest.json")
        report = run / read_json(path)["directory"]
        rm = read_json(report / "report_manifest.json")
        with open(report / "summary.csv", newline="") as f:
            summary = list(csv.DictReader(f))
        values = {r["metric"]: float(r["mean"]) if r["mean"] else None for r in summary}
        counts = {r["metric"]: int(r["n_prompts"]) for r in summary}
        entries.append({"manifest": m, "report": rm, "summary": summary, "values": values, "counts": counts})
    return entries


def parameter(entry):
    cfg = entry["manifest"]["config"]
    if cfg["method"] == "satisdive":
        return "delta", cfg["satisdive"]["delta"]
    return "lambda", cfg["sgi"]["lambda_score"]


def plot(entries, directory, name, xkey, ykey, xlabel, ylabel, scaling=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for entry in entries:
        m, cfg = entry["manifest"], entry["manifest"]["config"]
        x = cfg["sgi"]["initial_candidates"] if xkey == "initial_candidates" else entry["values"].get(xkey)
        y = entry["values"].get(ykey)
        if x is None or y is None:
            continue
        parameter_name, parameter_value = parameter(entry)
        rows.append({"run_id": m["run_id"], "method": m["method"], "x": x, "y": y,
            "x_metric": xkey, "y_metric": ykey, "parameter": parameter_name, "value": parameter_value,
            "initial_candidates": cfg.get("sgi", {}).get("initial_candidates", 4),
            "x_n_prompts": m["prompt_count"] if xkey == "initial_candidates" else entry["counts"].get(xkey),
            "y_n_prompts": entry["counts"].get(ykey), "expected_prompts": m["prompt_count"]})
    if not rows:
        return
    for row in rows:
        row["plotted"] = row["x_n_prompts"] == row["expected_prompts"] == row["y_n_prompts"]
        row["non_dominated"] = None
    complete = [r for r in rows if r["plotted"]]
    mask = nondominated([(r["x"], r["y"]) for r in complete]) if not scaling else [None]*len(complete)
    for row, flag in zip(complete, mask):
        row["non_dominated"] = flag
    write_csv(directory / f"{name}.csv", rows)
    if not complete:
        return
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    colors = {"satisdive": "#167d9a", "sgi": "#b64e24"}
    for method in ("satisdive", "sgi"):
        selected = [r for r in complete if r["method"] == method]
        if not selected:
            continue
        ax.scatter([r["x"] for r in selected], [r["y"] for r in selected], label=method,
                   color=colors[method], marker="o" if method == "satisdive" else "s")
        for r in selected:
            if r["non_dominated"]:
                ax.scatter([r["x"]], [r["y"]], s=130, facecolors="none", edgecolors=colors[method])
            label = f"M={r['initial_candidates']}" if scaling else f"{r['parameter']}={r['value']}"
            ax.annotate(label, (r["x"], r["y"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Subset results; inspect CSV coverage" + ("; rings = non-dominated" if not scaling else ""), fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    a = p.parse_args()
    entries = load_reports(a.outputs)
    if not entries:
        print("No stored evaluation reports. No result tables or figures generated.")
        return
    a.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.report_dir / "extended_results.csv", [r for e in entries for r in e["summary"]])
    main_rows, efficiency_rows = [], []
    buckets = defaultdict(list)
    for e in entries:
        m, c = e["manifest"], e["manifest"]["config"]
        s = c.get("sgi", {})
        default = c["satisdive"]["delta"] == 1.5 if m["method"] == "satisdive" else s["initial_candidates"] == 128 and s["lambda_score"] == 1.5
        base = {"run_id": m["run_id"], "method": m["method"], "protocol": m["protocol"], "dataset": m["dataset"],
                "eval_id": e["report"]["eval_id"], **e["report"]["coverage"]}
        if default:
            main_rows.append({**base, **{k: v for k, v in e["values"].items() if k not in ("runtime_s", "profiled_flops")}})
        efficiency_rows.append({**base, "initial_M": s.get("initial_candidates", 4), "final_K": 4,
            **{k: e["values"].get(k) for k in ("runtime_s", "candidate_weighted_forward_nfe", "denoiser_forward_calls",
            "guidance_autograd_calls", "reward_backward_calls", "allocated_gpu_seconds", "peak_allocated_generator_bytes",
            "peak_allocated_reward_worker_bytes", "resident_peak_sum_mib", "profiled_flops", "gpu_count", "solver_runtime_s", "solver_time_limit_hits")},
            "flops_note": "partial supported generator-process operations; not total FLOPs"})
        # Common figures require matched prompt sets, metrics, backbone settings and seed.
        key = (m["dataset"], m["protocol"], m["prompts_sha256"], e["report"]["eval_id"],
               c["backbone"]["height"], c["backbone"]["width"], c["backbone"]["steps"], c["generation"]["seed"],
               c["generation"]["final_decoder"], c["backbone"]["revision"], c["instrumentation"]["profile_flops"],
               c.get("sgi", {}).get("dropping_ratio", .5))
        buckets[key].append(e)
    write_csv(a.report_dir / "main_results.csv", main_rows)
    write_csv(a.report_dir / "efficiency.csv", efficiency_rows)
    for key, values in buckets.items():
        directory = a.report_dir / f"{key[0]}-{key[1]}-{digest(key)[:10]}"
        directory.mkdir(parents=True, exist_ok=True)
        front = [e for e in values if e["manifest"]["method"] == "satisdive" or e["manifest"]["config"]["sgi"]["initial_candidates"] == 128]
        satis = [e for e in front if e["manifest"]["method"] == "satisdive"]
        sgi = [e for e in front if e["manifest"]["method"] == "sgi"]
        plot(satis, directory, "satisdive_frontier", "dreamsim", "hpsv3_min", "DreamSim diversity", "Worst HPSv3")
        plot(sgi, directory, "sgi_frontier", "dino_distance", "clip_score_mean", "DINO patch diversity", "Mean CLIP cosine")
        for y in ("hpsv3_min", "hpsv3_mean", "hpsv3_sat", "image_reward_min"):
            plot(front, directory, f"common_dreamsim_{y}", "dreamsim", y, "DreamSim diversity", y)
        plot(front, directory, "common_dino_clip", "dino_distance", "clip_score_mean", "DINO patch diversity", "Mean CLIP cosine")
        scaling = [e for e in values if e["manifest"]["method"] == "sgi" and e["manifest"]["config"]["sgi"]["lambda_score"] == 1.0]
        for x in ("initial_candidates", "runtime_s", "candidate_weighted_forward_nfe", "profiled_flops"):
            for y in ("clip_score_mean", "hpsv3_min", "dino_distance", "dreamsim"):
                plot(scaling, directory, f"scaling_{x}_{y}", x, y, x, y, scaling=True)
    print(a.report_dir)


if __name__ == "__main__":
    main()
