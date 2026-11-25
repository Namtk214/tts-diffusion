"""Orchestrate isolated offline metric workers, then export coverage-aware reports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, atomic_json, read_json, exclusive_lock
from evaluation.protocol import METRICS, metric_python, load_eval_config
from evaluation.aggregate import aggregate_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--eval-config", type=Path)
    p.add_argument("--models-lock", type=Path, default=ROOT / "configs/models.lock.json")
    p.add_argument("--metrics", nargs="+", choices=(*METRICS, "geneval"), help="Optional partial evaluation; reports show missing metrics")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--aggregate-only", action="store_true", help="Read caches only; never load a model")
    a = p.parse_args()
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards or a.limit is not None and a.limit < 1:
        p.error("Invalid shard or limit")
    # A controller may run in a lightweight environment without PyTorch. Use
    # the per-metric environments created by setup_environments.py unless the
    # caller explicitly selected alternatives.
    metric_environments = {
        "TTS_EVAL_PYTHON": "eval",
        "TTS_HPSV3_PYTHON": "hpsv3",
        "TTS_HPSV2_PYTHON": "hpsv2",
        "TTS_IMAGEREWARD_PYTHON": "imagereward",
        "TTS_GENEVAL_PYTHON": "geneval",
    }
    for variable, environment in metric_environments.items():
        default_python = ROOT / ".venvs" / environment / "bin" / "python"
        if not os.environ.get(variable) and default_python.is_file():
            os.environ[variable] = str(default_python)
    # The GPU run keeps Hugging Face snapshots in shared RAM to fit the small
    # container disk. Match that cache for the separate evaluation worker.
    # The VM has a host-provided HF_HOME under /workspace, while this run's
    # pinned snapshots were deliberately cached in shared RAM.
    if Path("/dev/shm/tts-hf").is_dir():
        os.environ["HF_HOME"] = "/dev/shm/tts-hf"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
    a.run_dir = a.run_dir.absolute()
    manifest = read_json(a.run_dir / "run_manifest.json")
    if a.eval_config is None:
        base = ROOT / "configs"
        if manifest["protocol"] == "single_gpu_768":
            base = base / "single_gpu_768"
        a.eval_config = base / "evaluation.yaml"
    cfg, lock = load_eval_config(a.eval_config), read_json(a.models_lock)
    metrics = a.metrics or list(cfg["metrics"]) + (["geneval"] if manifest["dataset"] == "geneval" else [])
    failures = []
    if not a.aggregate_only:
        for metric in metrics:
            try:
                if metric == "geneval":
                    from evaluation.geneval_eval import evaluate_geneval
                    # Upstream evaluator chooses cuda:0. Remap only the selected allocated device.
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    idx = int(a.device.split(":")[-1]) if ":" in a.device else 0
                    if not a.device.startswith("cuda"):
                        raise ValueError("Official GenEval requires CUDA")
                    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[idx] if visible else str(idx)
                    try:
                        status = evaluate_geneval(a)
                    finally:
                        if visible is None:
                            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = visible
                else:
                    command = [metric_python(metric), "-m", "evaluation.worker", "--metric", metric,
                               "--run-dir", str(a.run_dir), "--eval-config", str(a.eval_config.absolute()),
                               "--models-lock", str(a.models_lock.absolute()), "--device", a.device,
                               "--shard-index", str(a.shard_index), "--num-shards", str(a.num_shards)]
                    if a.resume:
                        command.append("--resume")
                    if a.limit is not None:
                        command += ["--limit", str(a.limit)]
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                    status = subprocess.run(command, cwd=ROOT, env=env).returncode
                if status:
                    raise RuntimeError(f"{metric} worker exited with status {status}")
            except Exception as error:
                failures.append({"metric": metric, "error": str(error), "traceback": traceback.format_exc()})
                print(f"Evaluation failed: {metric}: {error}", flush=True)
        atomic_json(a.run_dir / "evaluation" / f"worker_failures-{a.shard_index}.json", failures)
    try:
        with exclusive_lock(a.run_dir / "evaluation/.locks/reports.lock", timeout=30):
            report, groups, _ = aggregate_run(a.run_dir, cfg, lock)
            print(f"Reports: {report}")
    except FileExistsError:
        print("Another evaluator is exporting reports; run --aggregate-only after all shards finish")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
