# Implementation Guide: Reproduce SatisDive & Scaling Group Inference (SGI)

## 1. Objective and scope

Implement a reproducible text-to-image generation and evaluation framework for exactly two methods:

- **SatisDive** — [Traversing the Satisfaction-Diversity Frontier in Text-to-Image Diffusion](https://arxiv.org/abs/2610.02372), [official code](https://github.com/UCF-CRCV/SatisDive), [OpenReview](https://openreview.net/forum?id=YZqxd60F1B).
- **SGI** — [Scaling Group Inference for Diverse and High-Quality Generation](https://arxiv.org/abs/2508.15773), [ICLR 2026 paper](https://openreview.net/pdf?id=IyTNxjTuWT), [official code](https://github.com/GaParmar/group-inference).

The objective is to reproduce the two published methods and their main text-to-image experiments, then obtain comparable outputs and metrics for future research on multi-reward diverse generation.

### Scope restrictions (mandatory)

- Implement and run **only SatisDive and SGI**. Do not reproduce FK Steering, DAS, VASR, NegToMe, CADS, Particle Guidance, CREPE, Best-of-N, or any other competing method. Do not run an independent I.I.D. baseline. SGI at `M=K` is allowed as an endpoint of its own scaling experiment.
- No component ablations, pruning ablations, decoder ablations, alternative reward sweeps, human studies, or AMT evaluation.
- No training, fine-tuning, multi-reward extension, depth-to-image generation, personalization, or synthetic 2D experiments.
- Do not run FID or CMMD, or download COCO reference images. Skip DrawBench, COCO and DreamBooth.
- Keep automated preference models (HPSv2, HPSv3, ImageReward, PickScore). They are not human annotation experiments.
- Use FLUX.1-dev as the only mandatory backbone. Additional native backbones are out of the initial implementation scope.

**Deliverable:** a runnable benchmark pipeline, versioned configurations, generated images, raw automated metrics, aggregate tables, frontier plots and compute-scaling plots. Do not invent reproduction results.

## 2. Upstream code and initial audit

Clone both official repositories into `third_party/` and record their exact commit SHAs. Read the paper sections and appendices alongside the current code before making changes.

```bash
mkdir -p third_party
git clone https://github.com/UCF-CRCV/SatisDive.git third_party/satisdive
git clone https://github.com/GaParmar/group-inference.git third_party/sgi
```

Inspect actual CLI flags, model revisions, scheduler settings, reward preprocessing and objective normalization. Do not assume that README defaults match the published tables. Use adapters or small documented patches instead of rewriting the algorithms. Keep each upstream environment isolated if package requirements conflict.

### Known discrepancies to resolve in the audit

| Item | Required action |
|---|---|
| SatisDive Pick-a-Pic prompt count | Paper reports 499; public test data may contain 500 unique prompts. Check whether the exact paper list is released. If unavailable, use the complete verified test list, record the actual count, and disclose the mismatch. Never drop an arbitrary prompt. |
| SGI native resolution | Public FLUX-dev config uses 768×768, 20 steps. Confirm the paper experiment settings before labeling a run an exact paper reproduction. |
| SGI decoder | Public code may replace the final decoder with Tiny VAE. Preserve the original path in native mode; use standard FLUX VAE for final outputs in unified mode. Keep latent scaling/shift and preview decoder behavior correct. |
| SGI candidate dropping | A dropping ratio of 0.5 means retaining roughly half, not dropping all but a fixed count. Follow the actual rounding and selection schedule in the pinned implementation. |
| SatisDive GPU requirements | The published FLUX/HPSv3 preset may put the generator and reward on separate GPUs. Verify memory usage and supported devices. If restricted to one GPU/job, implement and validate offloading or obtain an authorized multi-GPU configuration; never detach reward gradients just to fit memory. |
| Model access | FLUX.1-dev weights may require Hugging Face authentication and license acceptance. Gurobi may require an appropriate license. Report missing access as a blocker; do not silently substitute models or solvers. |

## 3. Datasets

Both methods must run on all three datasets in the unified benchmark. Use prompts only; no training images are needed.

| Dataset | Expected prompts | Official source | Native paper usage | Priority |
|---|---:|---|---|---|
| Pick-a-Pic v1 | 499 in SatisDive (verify public split) | [Hugging Face](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) | SatisDive | P0 |
| GenEval | 553 | [GitHub](https://github.com/djghosh13/geneval) | SGI | P0 |
| HPDv2 | 3,200 | [GitHub](https://github.com/tgxs002/HPSv2) | SatisDive | P1 |

Use Pick-a-Pic test prompts, preserve all GenEval prompt metadata and use the HPDv2 benchmark prompts (800 per category: animation, concept art, painting, photo). Verify counts against the actual source. Do not use HPDv2 training preferences.

### 3.1 Common prompt registry

Implement `prepare_pickapic.py`, `prepare_geneval.py` and `prepare_hpdv2.py`. Write one JSONL file per dataset. Example record:

```json
{"dataset":"geneval","split":"benchmark","prompt_id":"000001","prompt":"a photo of a red cup and a blue bowl","category":"colors","metadata":{}}
```

This is an illustrative record, not an actual GenEval entry. Preserve original prompts verbatim, original identifiers if available, GenEval evaluation metadata and HPDv2 categories. Record dataset revision, split, source URL, counts and SHA-256 of each prompt manifest. Define stable deterministic subsets of 10 and 50 prompts for testing; use the same subset for both methods and for every sweep point.

**Execution order:** Pick-a-Pic → GenEval → HPDv2. Run full datasets, not sampled subsets, for the final main results.

## 4. Models and generation settings

Mandatory generator: `black-forest-labs/FLUX.1-dev` (frozen weights). Record the model revision, dtype, PyTorch/Diffusers versions and all scheduler parameters.

### 4.1 Native reproduction profiles

Maintain two separate native configurations; do not change them to match each other before verifying upstream execution.

| Setting | SatisDive native | SGI native (public repo defaults; verify paper) |
|---|---|---|
| Backbone | FLUX.1-dev | FLUX.1-dev |
| Resolution | 1024×1024 | 768×768 |
| Denoising steps | 28 | 20 |
| Guidance scale | 3.5 | 3.5 |
| Initial candidates | 4 | 128 |
| Final candidates | 4 | 4 |
| Guidance reward | HPSv3 | CLIP text-image similarity |
| Diversity guidance | Internal model features | DINOv2 patchwise distance |
| Main hyperparameter | Δ=1.5 | λ=1.5 (public default) |
| Sampling | Euler SDE, noise amplitude 0.3 | Native FLUX scheduler |
| Candidate dropping ratio | N/A | 0.5 |
| Final decoder | Native | Native upstream behavior |

For SatisDive, use the published FLUX update schedule `[4, 8, 12, 16, 20, 24]`, feature layer 12, latent update multiplier 1, and the original reward temperature, diversity cutoff and replacement schedule. Verify indexing in code (zero-based vs. one-based); preserve all other upstream behavior. The inference threshold and the final evaluation threshold are distinct quantities.

Run native smoke tests first. For strict native reproduction, evaluate SatisDive on Pick-a-Pic and HPDv2, and SGI on GenEval. The SatisDive native FLUX settings can be reused for the unified runs below if identical. If the confirmed SGI paper config differs from the public default, store it separately as `sgi_paper` and disclose unresolved details. Do not claim exact reproduction if the configuration cannot be verified.

### 4.2 Unified benchmark profile (main deliverable)

| Setting | SatisDive | SGI |
|---|---|---|
| Backbone | FLUX.1-dev | FLUX.1-dev |
| Resolution | 768x768 | 768x768 |
| Steps | 28 | 28 |
| Guidance scale | 3.5 | 3.5 |
| Initial candidates `M` | 4 | 128 |
| Output group size `K` | 4 | 4 |
| Guidance reward | HPSv3 | CLIP |
| Diversity guidance | Internal features | DINOv2 |
| Default parameter | Δ=1.5 | λ=1.5 |
| Dropping ratio | N/A | 0.5 |
| Sampler | Native SatisDive SDE | Native SGI scheduler |
| Final image decoder | Standard FLUX VAE | Standard FLUX VAE |
| Seed | 42, one seed for initial full runs | 42, one seed for initial full runs |

This is a proposed common resolution/step/output-size protocol, not the original SGI setup. Keep the original sampler and guidance objective of each method. The two methods use different rewards and compute, so the default comparison is **not** a controlled same-reward or equal-compute experiment. Evaluate both using all common metrics; do not describe differences as solely due to their algorithms. A same-reward extension or compute-matched experiment is not required in this scope.

Use a deterministic prompt-specific seed registry. If compatible, share initial noise for the first four candidates and use a deterministic extension of that noise bank for SGI's additional candidates. Record any incompatibility. Do not force identical trajectories across different samplers. Reuse the same initial noise bank across each method's hyperparameter sweep.

### 4.3 Example configuration schema

The following YAML is the desired benchmark interface, **not** existing upstream CLI syntax.

```yaml
protocol: unified
method: satisdive                 # or sgi
backbone:
  id: black-forest-labs/FLUX.1-dev
  revision: null                  # resolve and pin before full runs
  height: 768
  width: 768
  steps: 28
  guidance_scale: 3.5
  dtype: bfloat16                 # subject to upstream compatibility
generation:
  seed: 42
  group_size: 4
  resume: true
  final_decoder: standard_flux
  sampler: native
satisdive:
  reward: hpsv3
  delta: 1.5
  noise_amplitude: 0.3
  update_steps: [4, 8, 12, 16, 20, 24]
  feature_layer: 12
  update_multiplier: 1.0
sgi:
  reward: clip
  diversity: dino_patch
  initial_candidates: 128
  dropping_ratio: 0.5
  lambda_score: 1.5
  preview_decoder: native
  solver: gurobi
dataset:
  name: pickapic
  manifest: data/pickapic.jsonl
  limit: null
```

Validate configs at startup; reject unsupported combinations. The inactive method's config must not influence execution.

## 5. Implementation work

### 5.1 SatisDive adapter

- Reuse official generation code and HPSv3 scorer. Do not rewrite its objective, dynamic tolerance rule, diversity gradients or latent replacement.
- Add a dataset runner that loads models once per worker and processes all prompts in a manifest.
- Expose `delta`, seed, steps, resolution, device assignment and output paths. Expose group size if needed to enforce the default `K=4`, but do not run group-size ablations.
- Preserve gradient flow through the original scorer and preview decoding path. Verify finite reward/diversity gradients on a smoke test.
- Record resolved settings and per-prompt runtime, actual denoiser forward counts, gradient calls and final image files.
- Add deterministic resume, prompt-level failure handling and atomic output writes.

### 5.2 SGI adapter

- Reuse original progressive pruning, CLIP unary score, DINOv2 patchwise binary score and QIP solver.
- Inspect `src/inference.py`, `src/my_utils/default_values.py`, `src/my_utils/group_inference.py`, `src/my_utils/scores.py` and `src/my_utils/solvers.py` in the pinned repository (paths may change).
- Add a multi-prompt runner; load FLUX, CLIP, DINO and the preview decoder once per worker.
- Expose `starting_candidates`, `output_group_size`, dropping ratio, `lambda_score`, steps, resolution and seed. Validate `M >= K` and output exactly `K` images.
- In unified mode, separate the intermediate preview decoder from the standard final FLUX VAE. Native mode must preserve the original decoder.
- Log active candidate counts per denoising step, candidate IDs selected by the solver, solver status, feasible solution count, time, objective value and gap where available.
- If a solver time limit is hit, accept a feasible incumbent only when the upstream algorithm supports it and mark its status. If no feasible solution exists, mark the prompt failed; do not silently use a different selection method.
- If pairwise DINO scoring causes OOM, implement an algebraically equivalent vectorized/chunked version and test it against the original scorer. Do not change preprocessing, patch normalization or feature representation.
- Keep `M=K=4` as an SGI scaling endpoint. No independent I.I.D. runner is required.

**Not required:** new reward adapters for SGI generation (ImageReward/HPSv3), new backbones, alternate solvers, final-only QIP, or any competitor implementation. HPSv3 and other rewards must still be available in the offline evaluator.

### 5.3 Shared benchmark runner

Expose a common interface, for example:

```python
def generate_group(prompt, seed, config):
    # Call the selected upstream adapter.
    # Return K final images, candidate IDs, and profiling metadata.
    ...
```

Requirements:

- Keep generation and evaluation separate. Offline evaluation must never call the generator.
- Save final images losslessly (PNG). Only the selected `K` outputs count in group metrics. Save intermediate images only when debugging; they must not enter evaluation.
- Shard prompt manifests deterministically across workers. Avoid duplicate jobs and concurrent writes to the same group.
- Resume only complete, validated groups under the same config hash. Incomplete groups must be retried or marked failed.
- Store exact upstream commits, model revisions, package versions, hardware, dataset hash and config hash for every run.

## 6. Automated metrics

Implement one common offline evaluator for both methods and all datasets. Use the exact metric checkpoints and preprocessing specified by each paper where available. Save checkpoint IDs, versions, feature types, normalization and numerical precision; never silently substitute another metric.

### 6.1 Reward and prompt alignment

| Metric | Source / checkpoint | Output |
|---|---|---|
| HPSv3 | SatisDive's bundled/public HPSv3 scorer | Per-image preference reward |
| HPSv2 | [HPSv2](https://github.com/tgxs002/HPSv2), verify paper version | Per-image preference reward |
| ImageReward | [Official code](https://github.com/THUDM/ImageReward) | Per-image preference reward |
| PickScore | `yuvalkirstain/PickScore_v1` with its compatible processor | Per-image preference reward |
| CLIP Score | `openai/clip-vit-base-patch32` (SGI scorer) | Per-image text-image cosine score |
| GenEval accuracy | [Official evaluator](https://github.com/djghosh13/geneval) | Overall and per-category accuracy, GenEval only |

GenEval accuracy is an additional automated metric in the unified benchmark, not a replacement for the CLIP metric reported by SGI. For PickScore, use a score definition that is comparable across groups; a softmax over each group's four images would make their mean trivial and must not be used as an absolute reward. Document the exact score used in the paper or an explicitly labeled alternative.

### 6.2 Diversity

| Metric | Source / representation | Aggregation |
|---|---|---|
| DreamSim | [Official code](https://github.com/ssundaram21/dreamsim) | Mean pairwise perceptual distance |
| DINO diversity | `facebook/dinov2-base`, SGI patchwise feature implementation | Mean pairwise feature distance |
| LPIPS | `lpips`, pin network and preprocessing | Mean pairwise distance |
| Vendi Score | [Vendi Score](https://github.com/vertaix/Vendi-Score), verify SGI kernel definition | Effective diversity of each group |

Compute all six unordered pairs for `K=4`. Reuse the original SGI DINO diversity definition; do not substitute CLS-only distance. Store each pairwise distance, not just its average. For Vendi Score, use the original paper's similarity kernel; if unspecified, document the chosen valid PSD kernel and mark the result as an adapted metric. Do not convert arbitrary distance matrices into a Vendi kernel without checking the definition.

### 6.3 Group-level reward aggregation

For dataset prompts `i=1,...,N`, output images `k=1,...,K`, and per-image reward `r[i,k]`:

- Mean reward: `mean_i(mean_k(r[i,k]))`.
- Worst reward: `mean_i(min_k(r[i,k]))`.
- Diversity: `mean_i(mean_{a<b}(d(x[i,a],x[i,b])))`.
- Individual satisfaction (SAT): `mean_{i,k}(r[i,k] >= threshold)`.

Report mean and worst for all five automated reward scores. Report SAT primarily for HPSv3 using fixed, documented thresholds:

| Dataset | HPSv3 evaluation threshold | Notes |
|---|---:|---|
| Pick-a-Pic | 9.66 | Published SatisDive floor for FLUX; verify score scale/version. |
| HPDv2 | 13.93 | Published SatisDive floor for FLUX; verify score scale/version. |
| GenEval | No published matched floor | Optionally report `SAT_fixed_9.66` as a transferred, uncalibrated floor, clearly labeled. Otherwise report SAT as N/A. |

Apply the same threshold to SatisDive and SGI on each dataset. Do not estimate separate thresholds from the outputs of the two methods. Since the user explicitly excludes running an independent I.I.D. baseline, no new median-calibration run is required. If paper reward scales cannot be matched, mark SAT reproduction as unresolved rather than silently recalibrating it. Do not compare SAT percentages across different dataset-specific thresholds as though the tasks have identical difficulty.

### 6.4 Efficiency metrics

| Metric | How to measure |
|---|---|
| Runtime | End-to-end generation time per prompt group, including guidance, decoding and solver; synchronize CUDA at boundaries. Exclude one-time model loading, report cold start separately if measured. |
| Forward NFE | Count actual denoiser calls and candidate-weighted forward evaluations separately (batch size matters). |
| Backward calls | Count actual guidance backward/gradient calls separately from forward passes. |
| Peak VRAM | Reset peak counters before measurement; report allocated memory per GPU, including any persistent loaded models. State whether reserved memory is also measured. |
| FLOPs | Profile forward and backward, decoder and scorers if supported; document profiler coverage. Report unsupported operations and solver time separately. Never label forward NFE alone as total FLOPs. |

Run comparable timing on the same GPU type and worker configuration. If methods use different GPU counts, report device count and GPU-seconds separately from wall time. Do not claim equal compute merely because both methods use 28 denoising steps.

## 7. Main experiment matrix

### E1 — Main results (mandatory)

Run both methods on all three datasets with the default unified configuration and a single seed. Report every shared automated metric plus generation cost.

| Dataset | Prompts (expected) | SatisDive final images | SGI final images |
|---|---:|---:|---:|
| Pick-a-Pic | 499 | 1,996 | 1,996 |
| GenEval | 553 | 2,212 | 2,212 |
| HPDv2 | 3,200 | 12,800 | 12,800 |
| Total | 4,252 | 17,008 | 17,008 |

Total: **34,016 final images** (8,504 prompt groups across two methods) for one unified setting. If Pick-a-Pic contains 500 prompts, the total is 34,024. These counts exclude native runs, smoke tests, sweeps and SGI's discarded candidates. Actual compute cost must be measured.

### E2 — SatisDive satisfaction–diversity frontier (mandatory)

Keep FLUX-dev, HPSv3, group size 4 and all other parameters fixed. Sweep the tolerance Δ.

Suggested grid, based on published ranges (verify exact published points if available):

```yaml
delta: [0.0001, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
```

Run on full Pick-a-Pic first, then GenEval. HPDv2 sweep is optional after E1 completes; its default operating point is mandatory. Each point represents aggregation across the same prompt list, not a per-image score.

Plot worst-candidate HPSv3 vs. DreamSim as the main frontier. Also export mean HPSv3 vs. DreamSim, HPSv3 SAT vs. DreamSim where defined, and worst ImageReward vs. DreamSim (offline only). Retain raw sweep points and mark non-dominated points; do not hide dominated points or invent interpolation results.

### E3 — SGI quality–diversity frontier (mandatory)

Keep `M=128`, `K=4`, dropping ratio 0.5, native CLIP guidance and DINO diversity fixed. Sweep λ.

```yaml
lambda_score: [0, 0.25, 0.5, 1.0, 1.5, 3.0, 6.0]
```

This is a proposed initial grid for the CLIP objective, not an asserted exact paper grid. Preserve upstream unary/binary score scaling and normalization. Do not use the HPSv3-specific λ grid from SatisDive's cross-paper comparison.

Run on full GenEval first, then Pick-a-Pic. HPDv2 sweep is optional after E1. Plot mean CLIP score vs. DINO diversity as the native main frontier, and worst HPSv3 vs. DreamSim as the common evaluation frontier. For both methods, export the same axes and use the same dataset and protocol for any combined plot.

### E4 — SGI inference-time scaling (mandatory)

Fix `K=4`, `lambda_score=1.0`, steps 28 (unified), dropping ratio 0.5 and all reward/diversity settings. Sweep:

```yaml
initial_candidates: [4, 8, 16, 32, 64, 128]
```

Run on full GenEval first. Optional: repeat on Pick-a-Pic if needed for cross-dataset generalization. No increasing-steps or Best-of-N competitor. Plot initial pool size or measured compute vs. mean CLIP, worst HPSv3, DINO diversity and DreamSim. Report runtime, NFE and FLOPs per point. The `M=4` case still runs through the SGI adapter and is not a separate baseline.

### E0 — Native verification and run ordering

Before E1–E4, run each upstream method on 10 prompts, then 50 prompts using native settings. Use the actual original dataset for each method. Verify metrics against paper definitions and inspect qualitative outputs. For full native results, run SGI on GenEval with verified paper settings (or clearly labeled upstream defaults if unresolved); reuse identical SatisDive native/unified results. If native and unified SGI configs differ, keep separate run IDs. E2/E3/E4 should initially use unified settings; reproduce a native frontier on its original dataset if exact figure reproduction is required and native settings differ. This is the same method, not an extra baseline.

Reuse results whenever the entire resolved generation config and prompt manifest match. For example, E1's default Δ=1.5 and λ=1.5 are points in E2/E3. Do not regenerate them. Note that E4 uses λ=1.0, so E1's SGI λ=1.5 run is not reusable for E4.

## 8. Output format and repository structure

```text
reproduction/
  third_party/
    satisdive/
    sgi/
  methods/
    satisdive_adapter.py
    sgi_adapter.py
  datasets/
    prepare_pickapic.py
    prepare_geneval.py
    prepare_hpdv2.py
  data/
    pickapic.jsonl
    geneval.jsonl
    hpdv2.jsonl
  evaluation/
    rewards.py
    diversity.py
    geneval_eval.py
    aggregate.py
    efficiency.py
  configs/
    native/
    unified/
    sweeps/
  scripts/
    generate.py
    evaluate.py
    run_sweep.py
    plot_results.py
  tests/
  outputs/
  reports/
  README.md
  REPRODUCTION_STATUS.md
```

Use a stable run directory, e.g. `outputs/<protocol>/<dataset>/<method>/<config_hash>/`.

Each run must store:

- `config.resolved.yaml`, `run_manifest.json` (source revisions, hardware, dataset hash).
- `images/<prompt_id>/000.png` through `003.png`, plus `group.json` with prompt, actual seeds and selected candidate IDs.
- `generation_stats.jsonl` with counts, time, memory, status and SGI solver diagnostics.
- `image_metrics.parquet` or CSV: `run_id, prompt_id, candidate_id, image_path, hpsv3, hpsv2, image_reward, pickscore, clip_score` and GenEval result when applicable.
- `pairwise_metrics.parquet` or CSV: `run_id, prompt_id, candidate_a, candidate_b, dreamsim, dino_distance, lpips`.
- `group_metrics.parquet` or CSV: per-group mean/min rewards, SAT, diversity, Vendi, compute and success status.
- `summary.csv`: aggregated metrics, prompt counts, failed groups and confidence intervals.

Use prompt-level bootstrap confidence intervals (suggested 1,000 resamples, fixed bootstrap seed, 95% CI). Do not bootstrap individual images as independent samples. For direct method comparisons, use matched prompt IDs and paired prompt bootstrap where appropriate. Report coverage over the full dataset as well as matched-group results; never silently drop failed prompts.

## 9. CLI and workflow

Create the following commands or equivalent interfaces; these are implementation targets, not claims about existing code.

```bash
# 1. Prepare datasets
python datasets/prepare_pickapic.py
python datasets/prepare_geneval.py
python datasets/prepare_hpdv2.py

# 2. Native smoke tests
python scripts/generate.py --config configs/native/satisdive.yaml --dataset pickapic --limit 10
python scripts/generate.py --config configs/native/sgi.yaml --dataset geneval --limit 10

# 3. Unified main results (one command per method and dataset)
python scripts/generate.py --config configs/unified/satisdive.yaml --dataset pickapic --resume
python scripts/generate.py --config configs/unified/sgi.yaml --dataset pickapic --resume
# Repeat on geneval and hpdv2.

# 4. Offline evaluation
python scripts/evaluate.py --run-dir <run_directory> --resume

# 5. Main sweeps
python scripts/run_sweep.py --config configs/sweeps/satisdive_delta.yaml --dataset pickapic
python scripts/run_sweep.py --config configs/sweeps/sgi_lambda.yaml --dataset geneval
python scripts/run_sweep.py --config configs/sweeps/sgi_candidates.yaml --dataset geneval
# Repeat the two frontier sweeps on their cross-dataset targets.

# 6. Aggregate and plot
python scripts/plot_results.py --outputs outputs --report-dir reports
```

Support `--limit`, `--shard-index`, `--num-shards`, `--resume` and `--device` (or equivalent) without changing scientific settings. If using a cluster with one GPU per job, keep workers within that allocation; report unsupported SatisDive device configurations instead of launching unauthorized jobs. Measure a 50-prompt pilot before committing to full HPDv2 or any sweep.

## 10. Required reports

### Main table (one per dataset and protocol)

| Method | HPSv3 mean | HPSv3 min | HPSv3 SAT | HPSv2 mean | IR mean | PickScore mean | CLIP mean | DreamSim | DINO div. | LPIPS | Vendi | GenEval acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SatisDive | — | — | — | — | — | — | — | — | — | — | — | — |
| SGI | — | — | — | — | — | — | — | — | — | — | — | — |

Include a machine-readable extended table with worst rewards for all reward metrics and 95% confidence intervals. Use N/A for inapplicable GenEval or SAT entries, not zero.

### Efficiency table

| Method | Initial M | Final K | Candidate-weighted forward NFE | Backward calls | Runtime/group | Peak VRAM/GPU | FLOPs (coverage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SatisDive | 4 | 4 | — | — | — | — | — |
| SGI | 128 | 4 | — | — | — | — | — |

### Figures

1. SatisDive Δ sweep: DreamSim vs. worst HPSv3.
2. SGI λ sweep: DINO diversity vs. mean CLIP.
3. Common frontier: DreamSim vs. worst HPSv3, both methods on the same dataset/config protocol.
4. Common frontier: DINO diversity vs. mean CLIP, both methods.
5. SGI candidate scaling: measured compute (runtime, NFE and FLOPs when available) vs. quality/diversity; also report initial candidate count.

Export figures as PNG and PDF (or SVG), plus the exact CSV used to draw each figure. Tables and figures must be generated from stored metrics, not manually typed values. No additional methods appear in the reports.

## 11. Acceptance criteria and implementation order

### Phase A — Setup

- [ ] Pin upstream commits and model revisions; verify licenses, dependencies and GPU access.
- [ ] Prepare the three prompt manifests and document actual counts.
- [ ] Resolve native config discrepancies and record remaining unknowns.
- [ ] Smoke-test both methods on ten prompts using official defaults/settings.

### Phase B — Core pipeline

- [ ] Both adapters produce exactly four valid 1024×1024 PNGs per prompt in unified mode.
- [ ] Repeated fixed-seed tests yield reproducible results within the expected numerical tolerance; document unavoidable GPU/solver nondeterminism.
- [ ] SatisDive retains correct reward gradients and SGI returns feasible selections of the required size.
- [ ] Offline evaluation computes all automated metrics without regeneration; pairwise metrics contain exactly six unordered pairs per complete group.
- [ ] Test aggregation, SAT thresholds, pairwise distance equivalence (if optimized), seed generation, solver failures and resume behavior.
- [ ] Generation/evaluation failures are recorded, never replaced with fabricated scores.

### Phase C — Experiments

- [ ] Complete 50-prompt pilot for both methods and profile runtime/VRAM.
- [ ] Run E1 on Pick-a-Pic and GenEval first; check expected metric trends and coverage.
- [ ] Run E2 and E3 frontier sweeps and E4 SGI scaling; reuse identical existing runs.
- [ ] Complete E1 on full HPDv2.
- [ ] Export all required tables, figures, config manifests and qualitative examples.

### Phase D — Reproduction status

In `REPRODUCTION_STATUS.md`, explicitly distinguish:

- Native results that match verified paper settings.
- Native results using repository defaults when paper configuration is unresolved.
- Unified benchmark results with changed resolution/steps/decoder.
- Any discrepancy in prompt lists, score normalization, solver status, checkpoints, compute coverage or environment.
- Completed, failed and not-run experiments.

Do not set arbitrary numeric thresholds for claiming an exact reproduction before verifying the original settings and observed variance. The smoke test and pilot establish implementation sanity, not paper-level statistical agreement. All planned runs use one seed initially; additional seeds are not part of the mandatory scope.

## 12. Final instruction to the coding agent

Implement only the missing infrastructure needed to run the two original methods, preserve upstream behavior, and use the same automated evaluation pipeline on Pick-a-Pic, GenEval and HPDv2. Do not expand the scope to competitors, ablations, human evaluation, new rewards for guidance, or new research methods. First verify native execution and dataset integrity, then run the unified main experiments. If access, licensing, exact paper prompts or configurations are unavailable, record the specific blocker and continue with clearly labeled verified alternatives where appropriate. Do not present a proposed config, a smoke test, or a partial run as a completed reproduction.
