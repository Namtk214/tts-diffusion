import numpy as np
import pytest
from evaluation.aggregate import bootstrap, summarize_rows
from evaluation.diversity import PAIRS, vendi_from_kernel
from evaluation.protocol import load_eval_config
from benchmark.common import ROOT


def test_pair_count_and_vendi_limits():
    assert len(PAIRS) == 6 and len(set(PAIRS)) == 6 and all(i < j for i, j in PAIRS)
    assert vendi_from_kernel(np.ones((4, 4))) == pytest.approx(1)
    assert vendi_from_kernel(np.eye(4)) == pytest.approx(4)
    bad = np.eye(4); bad[0, 1] = bad[1, 0] = 2
    with pytest.raises(ValueError, match="positive semidefinite"):
        vendi_from_kernel(bad)


def test_bootstrap_is_over_groups_and_paired_differences():
    # Four images in a prompt share one resampling unit, not four independent samples.
    group_means = [0., 1.]
    value, low, high = bootstrap(group_means, resamples=1000, seed=7)
    assert value == .5 and low == 0 and high == 1
    assert bootstrap([2., 2., 2.]) == (2., 2., 2.)
    assert bootstrap([]) == (None, None, None)
    assert bootstrap(group_means, seed=7) == bootstrap(group_means, seed=7)


def test_geneval_macro_average_not_micro_average():
    mean, _, _ = bootstrap([0., 0., 0., 1.], categories=["count", "count", "count", "color"])
    assert mean == .5


def test_missing_scores_not_silently_zero_and_sat_default_unresolved():
    cfg = load_eval_config(ROOT / "configs/evaluation.yaml")
    groups = [{"generation_status": "complete", "evaluation_status": "incomplete", "hpsv3_mean": 10., "category": "x"},
              {"generation_status": "failed", "evaluation_status": "not_run", "category": "x"}]
    manifest = {"run_id": "fixture", "method": "sgi", "dataset": "pickapic", "protocol": "unified"}
    rows = {r["metric"]: r for r in summarize_rows(groups, manifest, "eval", cfg)}
    assert rows["hpsv3_mean"]["mean"] == 10
    assert rows["hpsv3_mean"]["n_prompts"] == 1 and rows["hpsv3_mean"]["missing_prompts"] == 1
    assert rows["hpsv3_sat"]["mean"] is None
