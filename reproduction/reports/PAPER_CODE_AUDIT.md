# Paper / public-code audit

Scope of this implementation follows the supplied guide and the user's 100-prompt override. Broader experiments mentioned below are context, not additional implementation targets.

## SGI

The [arXiv v1 paper](https://arxiv.org/html/2508.15773v1) evaluates FLUX.1-dev, FLUX.1-schnell, SD3 Medium, FLUX Depth and SynCD across text generation, depth conditioning and customization. Text generation uses 553 GenEval prompts; other tasks use COCO-derived depth inputs and DreamBooth samples. Main automated scores are CLIP text alignment and DINO patch diversity; additional analyses include ImageReward, BLIP2, depth/CLIP feature diversity, runtime and user preferences. Using GenEval prompts does not mean the paper's main CLIP plots report official GenEval accuracy.

The pinned [public repo](../third_party/sgi/README.md) provides FLUX inference/demo paths, CLIP/DINO score functions, progressive pruning and Gurobi selection. It also exposes Canny/Kontext demos. It lacks a complete multi-dataset offline evaluation/report pipeline and the full paper's other model/task implementations. This framework adds only the infrastructure requested for FLUX.1-dev. The later OpenReview version's HPSv2/Vendi details remain unresolved; they are not attributed to arXiv v1.

## SatisDive

The [paper](https://arxiv.org/html/2610.02372v1) uses FLUX.1-dev, SANA-1.6B and SD1.5, primarily with HPSv3 or ImageReward. Main datasets are Pick-a-Pic and HPDv2, with separate appendix analyses on DrawBench/MS-COCO. It reports worst/mean reward, individual satisfaction at a fixed floor, DreamSim, LPIPS, held-out rewards and computation. Its SGI cross-paper comparison changes the SGI unary reward to HPSv3; that is a different setting from the public SGI CLIP objective retained here.

The pinned [public repo](../third_party/satisdive/README.md) is generation-only, with three presets and bundled differentiable HPSv3. It does not redistribute benchmark prompt manifests, evaluator scripts or all paper baselines/ablations. This framework keeps the FLUX sampler and adds subset preparation, common offline scoring, versioned outputs, robustness and reporting. SANA/SD1.5 source remains vendored for provenance, but no runners/experiments for them are added.

## Infrastructure added

- Immutable source references; VM-side model-revision locking and asset inventories.
- Deterministic nested subsets, original GenEval annotations and balanced HPDv2 categories.
- Two thin sampler adapters, separate generation/reward/evaluation environments and explicit device allocation.
- Lossless final groups, candidate lineage, prompt-level failures, integrity-checked resume and sharding.
- Nine shared reward/diversity outputs plus official GenEval, prompt bootstrap, paired comparisons and coverage tables.
- E0–E4 configs, native/unified distinction, raw sweep CSV, non-dominated markers and scaling figures.

No benchmark result has been produced. Exact metric definitions and remaining differences are in [METRICS.md](../METRICS.md) and [REPRODUCTION_STATUS.md](../REPRODUCTION_STATUS.md).
