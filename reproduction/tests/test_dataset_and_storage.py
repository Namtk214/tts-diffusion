import copy
import json
from PIL import Image
import pytest
from benchmark.common import prompt_seed, shard_rows, read_manifest, exclusive_lock
from benchmark.dataset_registry import deterministic_subset
from benchmark.storage import commit_group, validate_group
from benchmark.config import load_config, validate_config, config_hash
from benchmark.common import ROOT


def records():
    return [{"prompt_id": f"{i:04d}", "prompt": f"fixture {i}", "dataset": "hpdv2", "category": f"c{i%4}"} for i in range(320)]


def test_subsets_are_nested_balanced_and_independent_of_input_order():
    rows = records()
    a = deterministic_subset(rows, 100, 42, stratified=True)
    assert a == deterministic_subset(list(reversed(rows)), 100, 42, stratified=True)
    assert a[:50] == deterministic_subset(rows, 50, 42, stratified=True)
    assert [sum(r["category"] == c for r in a) for c in ("c0", "c1", "c2", "c3")] == [25]*4
    assert len({r["prompt_id"] for r in a}) == 100


def test_seeds_and_shards_preserve_prompt_identity():
    rows = records()
    shards = [shard_rows(rows, i, 7) for i in range(7)]
    assert sorted(r["prompt_id"] for s in shards for r in s) == sorted(r["prompt_id"] for r in rows)
    assert prompt_seed(42, rows[0]) == prompt_seed(42, copy.deepcopy(rows[0]))
    assert prompt_seed(42, rows[0]) != prompt_seed(42, rows[1])
    with pytest.raises(ValueError):
        shard_rows(rows, 7, 7)


def test_resume_rejects_truncated_images_changed_prompts_and_changed_config(tmp_path):
    row = records()[0]
    images = [Image.new("RGB", (16, 16), (i*40, 0, 0)) for i in range(4)]
    info = {"candidate_ids": [5, 7, 12, 19], "stats": {"runtime_s": 3.0}}
    commit_group(tmp_path, row, images, info, "config-a", (16, 16))
    directory = tmp_path / "images" / row["prompt_id"]
    assert validate_group(directory, "config-a", row, (16, 16))
    assert not validate_group(directory, "config-b", row, (16, 16))
    assert not validate_group(directory, "config-a", {**row, "prompt": "changed"}, (16, 16))
    (directory / "002.png").write_bytes(b"truncated")
    assert not validate_group(directory, "config-a", row, (16, 16))
    commit_group(tmp_path, row, images, info, "config-a", (16, 16))
    assert validate_group(directory, "config-a", row, (16, 16))
    assert len(list((tmp_path / "incomplete").iterdir())) == 1


def test_lock_never_steals_a_live_writer(tmp_path):
    path = tmp_path / "owner.lock"
    with exclusive_lock(path):
        with pytest.raises(FileExistsError):
            with exclusive_lock(path):
                pass
    assert not path.exists()


def test_manifest_rejects_path_traversal_and_duplicate_ids(tmp_path):
    path = tmp_path / "prompts.jsonl"
    for rows in ([{**records()[0], "prompt_id": "../escape"}], [records()[0], records()[0]]):
        path.write_text('\n'.join(json.dumps(r) for r in rows))
        with pytest.raises(ValueError):
            read_manifest(path)


def test_native_and_unified_decoder_and_scope_contracts():
    cfg = load_config(ROOT / "configs/unified/sgi.yaml")
    cfg["sgi"]["initial_candidates"] = 4
    assert validate_config(cfg)["sgi"]["initial_candidates"] == 4
    cfg["sgi"]["initial_candidates"] = 3
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = load_config(ROOT / "configs/unified/sgi.yaml")
    cfg["generation"]["final_decoder"] = "native"
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = load_config(ROOT / "configs/unified/satisdive.yaml")
    original = config_hash(cfg, "prompts", "code")
    cfg["satisdive"]["delta"] = 3
    assert config_hash(cfg, "prompts", "code") != original
