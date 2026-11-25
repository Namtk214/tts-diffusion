"""Prepare prompt-only manifests on the VM. No dataset imports at module load."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import requests

from benchmark.common import ROOT, atomic_json, file_sha256, write_jsonl

GENEVAL_REV = "af4902f24d3ca90ebbb446dd9891a59e0f82725f"
SUBSET_VERSION = "sha256-round-robin-v1"


def get_json(url, **kwargs):
    from urllib.parse import urlparse
    if urlparse(url).hostname in ("huggingface.co", "datasets-server.huggingface.co"):
        from huggingface_hub import get_token
        token = get_token()
        if token:
            kwargs["headers"] = {**kwargs.get("headers", {}), "Authorization": f"Bearer {token}"}
    r = requests.get(url, timeout=90, **kwargs)
    r.raise_for_status()
    return r.json()


def deterministic_subset(rows, size, seed=42, stratified=False):
    """Nested deterministic subsets; preserve text, IDs and benchmark duplicates."""
    if size < 1 or size > len(rows):
        raise ValueError(f"subset size {size} is outside [1, {len(rows)}]")
    def key(r):
        return hashlib.sha256(f"{seed}|{r['dataset']}|{r['prompt_id']}".encode()).hexdigest()
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("category", "all") if stratified else "all"].append(row)
    buckets = {k: sorted(v, key=key) for k, v in sorted(buckets.items())}
    ordered = []
    for i in range(max(map(len, buckets.values()))):
        for values in buckets.values():
            if i < len(values):
                ordered.append(values[i])
    return ordered[:size]


def geneval():
    url = f"https://raw.githubusercontent.com/djghosh13/geneval/{GENEVAL_REV}/prompts/evaluation_metadata.jsonl"
    r = requests.get(url, timeout=90); r.raise_for_status()
    raw = [json.loads(s) for s in r.text.splitlines() if s.strip()]
    rows = [{"dataset": "geneval", "split": "benchmark", "prompt_id": f"{i:06d}",
             "prompt": x["prompt"], "category": x["tag"], "metadata": x,
             "source_id": i} for i, x in enumerate(raw)]
    if len(rows) != 553:
        raise ValueError(f"GenEval source changed: expected 553, found {len(rows)}")
    return rows, {"source_url": url, "revision": GENEVAL_REV, "source_count": len(rows)}


def hpdv2():
    repo = "zhwang/HPDv2"
    revision = get_json(f"https://huggingface.co/api/datasets/{repo}")["sha"]
    rows, urls = [], []
    categories = {"anime": "animation", "concept-art": "concept_art", "paintings": "painting", "photo": "photo"}
    for original, category in categories.items():
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/benchmark/{original}.json"
        prompts = get_json(url); urls.append(url)
        if not isinstance(prompts, list) or len(prompts) != 800 or not all(isinstance(p, str) for p in prompts):
            raise ValueError(f"Expected exactly 800 benchmark strings in {original}")
        for i, text in enumerate(prompts):
            rows.append({"dataset": "hpdv2", "split": "benchmark", "prompt_id": f"{original}-{i:04d}",
                         "prompt": text, "category": category, "source_id": f"{original}/{i}",
                         "metadata": {"original_category": original, "index": i}})
    return rows, {"source_url": urls, "revision": revision, "source_count": 3200,
                  "note": "Official HPSv2 benchmark URLs; retain repeated prompt text with distinct source IDs."}


def pickapic(source_repo=None):
    original = "yuvalkirstain/pickapic_v1"
    repo = source_repo or original
    try:
        info = get_json(f"https://huggingface.co/api/datasets/{repo}")
    except requests.HTTPError as e:
        if not source_repo:
            raise RuntimeError("Original Pick-a-Pic source is unavailable. Re-run with --source-repo "
                               "pickapic-anonymous/pickapic_v1 to explicitly use the public mirror; "
                               "the source difference will be recorded.") from e
        raise
    revision = info["sha"]
    rows, excluded, offset, total = [], [], 0, None
    while total is None or offset < total:
        page = get_json("https://datasets-server.huggingface.co/rows", params={
            "dataset": repo, "config": "default", "split": "test_unique", "offset": offset, "length": 100})
        total = page["num_rows_total"]
        if not page["rows"]:
            raise ValueError("Dataset server returned an incomplete page")
        for item in page["rows"]:
            x, i = item["row"], item["row_idx"]
            text = x.get("caption", x.get("prompt"))
            if not isinstance(text, str):
                raise ValueError("Missing Pick-a-Pic caption field")
            # Empty prompts cannot pass either sampler. This is a documented rule, never a random drop.
            if not text.strip():
                excluded.append({"source_row": i, "reason": "empty_or_whitespace_prompt", "prompt": text})
                continue
            rows.append({"dataset": "pickapic", "split": "test_unique", "prompt_id": f"{i:06d}",
                         "prompt": text, "category": "user_prompt", "source_id": x.get("id", i),
                         "metadata": {"source_row": i}})
        offset += len(page["rows"])
    if get_json(f"https://huggingface.co/api/datasets/{repo}")["sha"] != revision:
        raise RuntimeError("Source revision changed during pagination; retry")
    if len({r['prompt'] for r in rows}) != len(rows):
        raise ValueError("test_unique contains duplicate captions; investigate source")
    return rows, {"source_url": f"https://huggingface.co/datasets/{repo}", "revision": revision,
                  "original_source": original, "mirror": repo != original, "split": "test_unique",
                  "source_count": total, "eligible_count": len(rows), "excluded": excluded,
                  "paper_count": 499, "exact_paper_list_verified": False,
                  "note": "Prompt-only extraction via dataset viewer; no image bytes downloaded. "
                          "Dataset head checked before/after pagination; prompt snapshot hash is authoritative."}


def main(name):
    p = argparse.ArgumentParser(description=f"Prepare deterministic {name} prompt subsets on the VM")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data")
    p.add_argument("--source-repo", help="Explicit Pick-a-Pic source/mirror")
    p.add_argument("--refresh", action="store_true", help="Explicitly replace existing source/subsets; changes run identity")
    args = p.parse_args()
    if (args.output_dir / f"{name}.provenance.json").exists() and not args.refresh:
        raise SystemExit("Dataset manifests already exist. Reuse them, or explicitly pass --refresh.")
    if args.size < 1:
        p.error("size must be positive")
    rows, provenance = pickapic(args.source_repo) if name == "pickapic" else {"geneval": geneval, "hpdv2": hpdv2}[name]()
    # The full prompt-only source is retained for provenance; default runs use only the subset.
    full = args.output_dir / "sources" / f"{name}.jsonl"
    write_jsonl(full, rows)
    sizes = sorted(set([min(10, args.size), min(50, args.size), args.size]))
    manifests = {}
    for size in sizes:
        selected = deterministic_subset(rows, size, args.seed, stratified=name in ("hpdv2", "geneval"))
        path = args.output_dir / f"{name}_{size}.jsonl"
        write_jsonl(path, selected)
        manifests[str(size)] = {"file": path.name, "sha256": file_sha256(path), "count": len(selected),
                                "categories": dict(Counter(r["category"] for r in selected))}
    provenance.update({"dataset": name, "subset_algorithm": SUBSET_VERSION, "subset_seed": args.seed,
                       "source_manifest_sha256": file_sha256(full), "subsets": manifests,
                       "scope": "pilot_subset_not_full_paper_reproduction"})
    atomic_json(args.output_dir / f"{name}.provenance.json", provenance)
    print(json.dumps(provenance, indent=2))
