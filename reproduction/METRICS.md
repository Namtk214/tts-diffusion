# Metric definitions

Only the four final lossless PNGs enter offline evaluation. Generation-time previews, discarded SGI candidates, reward traces and SatisDive's floating-point final-image diagnostics are excluded from headline metrics. All methods use the same evaluator and model lock. Definitions that cannot yet be matched to a paper are labeled explicitly below.

## Per-image rewards

| Output | Checkpoint / preprocessing | Numeric definition and status |
|---|---|---|
| `hpsv3` | Bundled HPSv3; pinned MizzenAI/HPSv3 and Qwen2-VL-7B-Instruct; official PIL/Qwen preprocessing at 200704 pixels; bfloat16 backbone, float32 reward head | Mu (`logits[0,0]`), individually scored images. VM scale verification pending. Differentiable generation separately uses the upstream tensor processor and in-graph ×255 rescaling. |
| `hpsv2` | xswu/HPSv2, `HPS_v2_compressed.pt`, default v2.0; original ViT-H-14 validation transform, fp16 CUDA autocast | Native normalized feature dot product, no ×100. Full strict checkpoint loading replaces the redundant LAION initialization. Exact paper version unresolved; v2.1 requires explicit config change and new metric identity. |
| `image_reward` | THUDM/ImageReward v1.0; official 224 bicubic center crop, official 35-token text handling; float32 | Official standardized reward. BERT tokenizer is resolved to a pinned local snapshot. |
| `pickscore` | PickScore_v1 + compatible LAION ViT-H-14 processor; 77-token truncation; float32 | `exp(logit_scale) * cosine(image,text)`. Explicit absolute logit definition; no softmax across a group. Exact paper normalization is not asserted. |
| `clip_score` | openai/clip-vit-base-patch32; SGI bilinear resize to 224, CLIP normalization; float32 | Text-image cosine, without ×100, positive clamp or learned logit scale. Text is explicitly truncated to the 77-token CLIP context; generation still uses the full prompt. |

For each reward save all four values, group mean and group minimum. Dataset mean/min columns are the mean of the corresponding group quantities. A metric loader is never replaced by a different available scorer.

Sources: [HPSv2 scoring](https://github.com/tgxs002/HPSv2/blob/866735ecaae999fa714bd9edfa05aa2672669ee3/hpsv2/img_score.py), [ImageReward](https://github.com/THUDM/ImageReward), [PickScore model card](https://huggingface.co/yuvalkirstain/PickScore_v1), and vendored SGI/HPSv3 source.

## Group diversity

Every complete group has exactly six unordered pairs. Pair identities use the final candidate IDs stored in `group.json`, not assumed SGI indices 0–3.

| Output | Definition | Status |
|---|---|---|
| DreamSim | `dreamsim==0.2.1`, ensemble, normalized embeddings, no patch model; official 224 bicubic preprocessing; mean of six official distances | External weights are downloaded explicitly and hashed before offline evaluation. Paper checkpoint equivalence still needs VM verification. |
| DINO distance | DINOv2-base patch tokens excluding CLS; SGI bilinear 256 + ImageNet normalization; mean cosine distance at corresponding patch positions | Original scoring algebra; vectorized dot reduction and extraction microbatches preserve preprocessing and float32 promotion. Equivalence tests are written, pending execution. |
| LPIPS | `lpips==0.1.4`, AlexNet, calibration version 0.1, RGB in [-1,1], bilinear resize to 256 | Explicit adapted network/resolution until paper settings are confirmed. |
| Vendi | Gram kernel of concatenated unit DINO patch vectors, normalized per image; effective rank `exp(-sum(p log p))`, p=eigenvalues(K/4) | **Adapted kernel**, not verified as the SGI paper's Vendi implementation. Checks symmetry, unit diagonal and positive semidefiniteness. Stores kernel as well as score. |

Vendi is one group statistic, not six pair distances. This Gram construction is PSD because it is a dot-product matrix of actual features; arbitrary perceptual distances are never converted into a similarity matrix. Tiny numerical negative eigenvalues are clipped only within a 1e-6 tolerance; materially non-PSD kernels fail. The expected range for four nondegenerate samples is 1–4.

Sources: [DreamSim](https://github.com/ssundaram21/dreamsim), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [Vendi implementation](https://github.com/vertaix/Vendi-Score/blob/main/vendi_score/vendi.py).

## SAT and GenEval

Published FLUX/HPSv3 floors are 9.66 for Pick-a-Pic and 13.93 for HPDv2. `SAT` counts **individual images** with reward ≥ the dataset floor, then averages their four indicators per prompt. It does not require all four to pass. Both methods share the same fixed floor. No new base-model calibration is run.

The default config has `hpsv3_scale_verified: false`, so SAT is unresolved rather than a misleading number. After confirming checkpoint/preprocessing/score scale in the VM pilot, set that flag and record a nonempty `verification_note`, then rerun `evaluate.py --aggregate-only` for both methods. Scoring caches remain reusable; the report identity changes. GenEval SAT stays N/A because there is no published matched floor. Do not compare percentages across dataset-specific floors as the same task.

GenEval uses the clean official evaluator at commit `af4902f24d3ca90ebbb446dd9891a59e0f82725f`, its default thresholds, Mask2Former Swin-S detector and OpenAI CLIP ViT-L/14. Original `include`/`exclude`/`tag` annotations are retained. Four saved PNGs are staged via symlinks in the official numeric-directory layout. The wrapper changes only asset paths and device visibility. Failed evaluator jobs are logged; no replacement heuristic is used.

`geneval_accuracy` in group CSV is the fraction of four images correct. The summary's overall GenEval score is the **unweighted mean of task/category accuracies**, as in the [official summary script](https://github.com/djghosh13/geneval/blob/af4902f24d3ca90ebbb446dd9891a59e0f82725f/evaluation/summary_scores.py), with per-task rows also exported. If an expected task has no valid results, overall is unresolved. This run uses 30 category-stratified prompts, so it is not the full 553-prompt result.

## Statistics and coverage

Means use successful groups for each metric; each estimate carries `n_prompts`, `missing_prompts`, full expected count and overall generation/evaluation coverage. Missing values stay blank/None, never zero. `group_metrics.csv` retains failed, corrupt and not-run prompts. Worker logs/caches retain specific errors.

Confidence intervals are 95% percentile intervals from 1,000 prompt resamples, bootstrap seed 12345. Each prompt's four images/six pairs stay together. GenEval uses stratified resampling within tasks to preserve its macro-average definition. Paired comparisons resample per-prompt score differences over matched IDs, require matching prompts/protocol/backbone/metric identities, and report both individual coverage and matched count. Fixed-seed repeatability is reported separately from statistical uncertainty across prompts.

## Cost

Runtime is synchronized sampler wall time, including the upstream method's guidance, solver, decoding, PIL conversion and final diagnostics; it excludes cold start and PNG disk I/O. GPU-seconds are allocated GPU count × wall time, not measured utilization-integrated compute. NFE is reported both as actual full denoiser calls and the sum of their candidate batch sizes. Block checkpoint recomputation and reward-worker gradient calls are separate considerations; NFE is never relabeled FLOPs.

Optional profiler FLOPs cover supported operations in the generation process only. Reward subprocesses, solver and unsupported kernels/backward operations remain uncovered. Profiler-enabled runs get a different identity because profiling perturbs timing. Generator/reward-worker allocated and reserved peaks are reported separately; nvidia-smi reports sampled device-wide resident memory, including unrelated processes if GPUs are shared. Compare timing only with matching device type/allocation and inspect the saved environment metadata.
