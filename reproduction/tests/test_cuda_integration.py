"""Opt-in VM test using the real first benchmark prompt; never executed on local Mac."""
import os
import numpy as np
import pytest
from benchmark.common import ROOT, read_manifest, prompt_seed, atomic_json
from benchmark.config import load_config, resolve_models


@pytest.mark.cuda
def test_native_group_gradients_and_repeatability(tmp_path):
    method = os.environ.get("TTS_TEST_METHOD")
    if method not in ("sgi", "satisdive"):
        pytest.fail("Set TTS_TEST_METHOD=sgi or satisdive and use its matching environment")
    cfg = resolve_models(load_config(ROOT / f"configs/native/{method}.yaml"))
    dataset = "geneval" if method == "sgi" else "pickapic"
    row = read_manifest(ROOT / f"data/{dataset}_100.jsonl")[0]
    if method == "sgi":
        from methods.sgi_adapter import SGIAdapter as Adapter
    else:
        from methods.satisdive_adapter import SatisDiveAdapter as Adapter
    adapter = Adapter(cfg, work_dir=tmp_path, device="cuda:0", reward_device=1)
    try:
        seed = prompt_seed(cfg["generation"]["seed"], row)
        first, info = adapter.generate_group(row["prompt"], seed)
        assert len(first) == 4 and len(set(info["candidate_ids"])) == 4
        assert all(im.size == (cfg["backbone"]["width"], cfg["backbone"]["height"]) for im in first)
        assert info["stats"]["candidate_weighted_forward_nfe"] >= info["stats"]["denoiser_forward_calls"] > 0
        if method == "satisdive":
            assert info["gradient_validation"]["finite"]
            assert info["gradient_validation"]["nonzero_reward_gradient_observed"]
            assert info["gradient_validation"]["nonzero_diversity_gradient_observed"]
        second, other = adapter.generate_group(row["prompt"], seed)
        pixel_mae = [float(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)).mean()) for a, b in zip(first, second)]
        atomic_json(ROOT / "reports" / f"repeatability-{method}.json", {
            "seed": seed, "prompt_id": row["prompt_id"], "first_ids": info["candidate_ids"], "second_ids": other["candidate_ids"],
            "pixel_mae_0_255": pixel_mae, "bitwise_equal": all(x == 0 for x in pixel_mae),
            "note": "Report observed differences; no unverified paper-agreement tolerance imposed"})
    finally:
        adapter.close()
