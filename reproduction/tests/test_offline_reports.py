import csv
from PIL import Image
from benchmark.common import ROOT, atomic_json, write_jsonl
from benchmark.storage import commit_group
from evaluation.protocol import REWARDS, METRICS, group_input_hash, load_eval_config
from evaluation.aggregate import aggregate_run


def test_offline_reports_keep_six_pairs_threshold_boundary_and_failure_coverage(tmp_path, monkeypatch):
    rows = [{"prompt_id": str(i), "prompt": "synthetic fixture", "dataset": "pickapic", "category": "test"} for i in range(2)]
    manifest = {"run_id": "fixture", "config_hash": "fixture", "dataset": "pickapic", "method": "sgi", "protocol": "unified",
                "config": {"backbone": {"width": 16, "height": 16}}}
    atomic_json(tmp_path / "run_manifest.json", manifest)
    write_jsonl(tmp_path / "prompts.jsonl", rows)
    group = commit_group(tmp_path, rows[0], [Image.new("RGB", (16, 16)) for _ in range(4)],
                         {"candidate_ids": [3, 6, 8, 9], "stats": {"runtime_s": 5}}, "fixture", (16, 16))
    atomic_json(tmp_path / "records/1.json", {"status": "failed", "error": "fixture failure"})
    monkeypatch.setattr("evaluation.aggregate.metric_identity", lambda name, *args: (name, {}))
    for metric in METRICS:
        result = {"image_values": [9.65, 9.66, 10., 12.]} if metric in REWARDS else {"pair_values": [0.2]*6}
        if metric == "dino":
            result["vendi"] = 2.5
        atomic_json(tmp_path / f"evaluation/cache/{metric}/{metric}/0.json", {
            "metric_id": metric, "input_hash": group_input_hash(group), "status": "complete", **result})
    cfg = load_eval_config(ROOT / "configs/evaluation.yaml")
    cfg["sat"].update(hpsv3_scale_verified=True, verification_note="Synthetic fixture only")
    directory, groups, summary = aggregate_run(tmp_path, cfg, {})
    assert groups[0]["hpsv3_sat"] == .75  # >= threshold, not >
    assert groups[0]["hpsv3_min"] == 9.65
    assert groups[1]["generation_status"] == "failed"
    with open(directory / "pairwise_metrics.csv") as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == 6
    assert {(r["candidate_a"], r["candidate_b"]) for r in pairs} == {("3", "6"), ("3", "8"), ("3", "9"), ("6", "8"), ("6", "9"), ("8", "9")}
    sat = next(r for r in summary if r["metric"] == "hpsv3_sat")
    assert sat["n_prompts"] == 1 and sat["expected_prompts"] == 2
    assert sat["fully_evaluated_prompts"] == 1
