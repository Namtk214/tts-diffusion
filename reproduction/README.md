# SatisDive / SGI reproduction framework

**Current 768 × 768 experiment:** [Single A100 setup and pilot](SINGLE_GPU.md). This profile uses 30 Pick-a-Pic prompts and 30 GenEval prompts (480 final images). GPU validation is pending. The original protocols are documented below.


Implementation status: **prepared for VM validation; no experiments run**. The local machine has only undergone static syntax/config checks. Dataset/model downloads and CPU/GPU tests are deferred at the user's request. Read [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) before interpreting any settings as paper-verified.

## Scope and defaults

| Setting | Native SatisDive | Native SGI | Unified benchmark |
|---|---|---|---|
| Backbone | FLUX.1-dev | FLUX.1-dev | FLUX.1-dev |
| Resolution / steps | 1024 / 28 | 768 / 20, repository defaults | 1024 / 28 |
| Final group | 4 | 4 from initial M=128 | 4 |
| Guidance objective | HPSv3 + block-12 diversity | CLIP + DINO patch diversity | Each method's original objective |
| Control | Δ=1.5 | λ=1.5, drop ratio=0.5 | Same defaults |
| Final decoder | Standard FLUX VAE | Upstream Tiny VAE | Standard FLUX VAE |

SGI keeps the Tiny VAE for pruning previews in unified mode. Its sampler keeps the upstream first-step guidance rule and 256-token context; SatisDive retains its 512-token context, SDE amplitude 0.3, ramp, replacement and calibration. SatisDive update indices `[4,8,12,16,20,24]` are **zero-based**. Equal steps do not imply equal compute.

Initial scope uses a single base seed, 42. Each prompt receives a stable hash-derived seed, independent of shard/order/method/sweep. The native RNG mechanisms differ: SGI uses CPU generators at `seed + candidate_index`; SatisDive uses a CUDA generator for batched latents. These are matched prompt seeds, not a claim of identical initial noise across methods.

## 1. Prepare the Linux VM

Use a CUDA VM with enough memory for FLUX and scorer models. **SatisDive requires two distinct visible GPUs** with this upstream worker design. SGI uses one GPU plus CPU Gurobi. No silent CPU/MPS fallback, quantization or single-GPU offload is implemented. Actual memory/runtime must be established by the pilot.

The recorded SatisDive Torch 2.11 stack uses CUDA 13.0 PyPI wheels; the VM driver must support that runtime. GenEval has a separate legacy CUDA 11.3/Torch 1.12 environment matching its released evaluator family. These combinations still need validation on the rented hardware; in particular, old evaluator wheels may not support newer GPU architectures. Keep incompatible packages isolated.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git on the VM, then:

```bash
git clone https://github.com/Namtk214/tts-diffusion.git
cd tts-diffusion/reproduction
python3 scripts/setup_environments.py
source scripts/env.sh
```

The installer creates `.venvs/{controller,sgi,satisdive,hpsv3,eval,hpsv2,imagereward,geneval}`, installs explicit requirements, checks package dependencies, and saves actual `pip freeze` output in `reports/environment-locks/`. It does not run tests, load checkpoints or prepare datasets. `--only controller sgi` can install selected environments. The HPSv3 source is exposed via a `.pth` file to avoid requiring unused training extras; imported inference dependencies are listed in `environments/hpsv3.txt`.

On a scheduler, retain its `CUDA_VISIBLE_DEVICES` allocation. On a dedicated VM, set the assigned devices explicitly. `--reward-device 1` means the second **visible** GPU. Obtain a Gurobi license supporting the 128-variable quadratic problem. Configure it on the VM; do not commit credentials.

## 2. Pin and download assets on the VM

Authenticate with Hugging Face and accept FLUX's upstream access conditions in your account. For the controller's pinned hub version, use `.venvs/controller/bin/huggingface-cli login`. Reuse a shared HF cache visible to every environment.

```bash
.venvs/controller/bin/python scripts/pin_models.py
.venvs/controller/bin/python scripts/download_models.py
.venvs/eval/bin/python scripts/prepare_metric_assets.py --metric lpips
.venvs/eval/bin/python scripts/prepare_metric_assets.py --metric dreamsim
.venvs/geneval/bin/python scripts/prepare_geneval_assets.py
```

`configs/models.lock.json` stores immutable model commit SHAs. It deliberately does not exist before VM setup. Keep that file for all runs; `--refresh` explicitly changes the experiment. Download inventory hashes live under `models/`. The downloader fetches FLUX's Diffusers components, excluding the redundant monolithic checkpoint. LPIPS/DreamSim have separately hashed external assets. GenEval code is checked out at a fixed commit in ignored `external/geneval/`; its detector configuration and checkpoints are hashed.

Generation uses local snapshots. Metric workers disable network access; missing assets fail clearly. No metric worker calls a generator.

## 3. Prepare the three prompt subsets

```bash
.venvs/controller/bin/python datasets/prepare_pickapic.py
.venvs/controller/bin/python datasets/prepare_geneval.py
.venvs/controller/bin/python datasets/prepare_hpdv2.py
```

If the original Pick-a-Pic endpoint is inaccessible, explicitly choose the public mirror:

```bash
.venvs/controller/bin/python datasets/prepare_pickapic.py \
  --source-repo pickapic-anonymous/pickapic_v1
```

This difference is recorded; no exact 499-prompt match is claimed. The viewer's `test_unique` caption list is filtered only for empty text and checked for duplicate captions. Source head is checked before/after pagination, but the viewer does not pin its rows endpoint to a revision; the saved prompt snapshot/hash is authoritative.

Defaults create nested `data/{dataset}_10.jsonl`, `_50.jsonl`, `_100.jsonl`, plus provenance and prompt-only source snapshots. No dataset image bytes are downloaded. GenEval uses original annotations and a deterministic category-stratified subset. HPDv2 selects 25 prompts per official category; repeated text with different official IDs is retained. Pick-a-Pic uses a deterministic hash ordering. Source size is audited separately from the 100-prompt experiment size. Existing manifests require explicit `--refresh` before replacement.

## 4. Preflight, tests and native pilots — VM only

```bash
.venvs/sgi/bin/python scripts/preflight.py --method sgi --check-solver
.venvs/satisdive/bin/python scripts/preflight.py --method satisdive
.venvs/sgi/bin/python -m pytest tests -m 'not cuda'
```

Tests have been written but not run locally. They cover manifest/seed/shard contracts, corrupted-image resume, solver failures, DINO equivalence, threshold boundaries, pair counts and prompt bootstrap. Opt-in repeatability/gradient tests load real models and generate images:

```bash
TTS_TEST_METHOD=sgi .venvs/sgi/bin/python -m pytest tests/test_cuda_integration.py --run-cuda
# Install pytest==8.3.5 in the SatisDive environment before its integration test.
TTS_TEST_METHOD=satisdive .venvs/satisdive/bin/python -m pytest tests/test_cuda_integration.py --run-cuda
```

Start with native 10-prompt runs, inspect saved groups and metric/gradient traces, then run 50-prompt pilots:

```bash
.venvs/satisdive/bin/python scripts/generate.py --config configs/native/satisdive.yaml --dataset pickapic --limit 10 --resume
.venvs/sgi/bin/python scripts/generate.py --config configs/native/sgi.yaml --dataset geneval --limit 10 --resume

.venvs/satisdive/bin/python scripts/generate.py --config configs/native/satisdive.yaml --dataset pickapic --limit 50 --resume
.venvs/sgi/bin/python scripts/generate.py --config configs/native/sgi.yaml --dataset geneval --limit 50 --resume
```

Each command prints its run directory. The 10/50 prefixes are nested in the default 100 manifest. Different prompt counts have separate run IDs. Inspect `workers/*/environment.json`, `generation_stats.jsonl` and each `group.json`. Record numerical differences from repeated GPU runs instead of assuming bitwise determinism; Gurobi time limits can also affect selections.

## 5. Unified E1 and sweeps

E1: 100 prompts per dataset, two methods, four final images per prompt. The scheduler requires the measured 50-prompt pilots:

```bash
.venvs/controller/bin/python scripts/run_main.py \
  --satisdive-pilot outputs/native/pickapic/satisdive/REPLACE_WITH_RUN_ID \
  --sgi-pilot outputs/native/geneval/sgi/REPLACE_WITH_RUN_ID
```

Use `--plan` to print commands. Or launch a single run explicitly:

```bash
.venvs/sgi/bin/python scripts/generate.py --config configs/unified/sgi.yaml --dataset pickapic --resume
.venvs/satisdive/bin/python scripts/generate.py --config configs/unified/satisdive.yaml --dataset pickapic --resume
```

All generation commands support `--limit`, `--shard-index`, `--num-shards`, `--device`, `--reward-device`, `--outputs`, `--models-lock` and `--resume`. `--plan` validates existing manifest/config/model-lock metadata without loading models. Never run different shards on overlapping GPU allocations. Prompt locks prevent duplicate writes, and resumed PNGs are checked for dimensions, count, SHA256 and matching prompt/config. A crashed process can leave a lock: inspect its saved host/PID and confirm the owner has stopped before removing only that stale lock. Locks are never automatically stolen.

```bash
# E2: repeat on geneval after pickapic.
.venvs/satisdive/bin/python scripts/run_sweep.py --config configs/sweeps/satisdive_delta.yaml --dataset pickapic --pilot-run SATISDIVE_PILOT_DIR
# E3: repeat on pickapic after geneval.
.venvs/sgi/bin/python scripts/run_sweep.py --config configs/sweeps/sgi_lambda.yaml --dataset geneval --pilot-run SGI_PILOT_DIR
# E4: M=4,8,16,32,64,128, lambda=1.0.
.venvs/sgi/bin/python scripts/run_sweep.py --config configs/sweeps/sgi_candidates.yaml --dataset geneval --pilot-run SGI_PILOT_DIR
```

Every sweep uses 100 prompts. The Δ/λ grids are proposed initial grids, not a claim of exact paper figure points. Identical E1/sweep configurations reuse the same run ID. E4's λ=1.0 differs from E1's λ=1.5. E4's M=128 point can reuse E3's λ=1.0 point when all other settings match.

For a **complete 100-prompt** SatisDive native run, `generate.py --config configs/unified/satisdive.yaml ... --resume --reuse-run NATIVE_RUN_DIR` reuses outputs and timings after checking identical code/config/prompt hashes, ignoring only the protocol label. It records reuse provenance; the timings are not a new measurement. SGI native/unified cannot pass this check because their settings differ.

## 6. Offline metrics and reports

```bash
.venvs/controller/bin/python scripts/evaluate.py --run-dir RUN_DIR --resume
.venvs/controller/bin/python scripts/evaluate.py --run-dir RUN_DIR --aggregate-only
.venvs/controller/bin/python scripts/compare_runs.py --left SATISDIVE_RUN_DIR --right SGI_RUN_DIR --output reports/paired.csv
.venvs/controller/bin/python scripts/plot_results.py --outputs outputs --report-dir reports
.venvs/controller/bin/python scripts/export_gallery.py --run-dir RUN_DIR --output reports/gallery.html --limit 10
```

`evaluate.py` runs one persistent subprocess per metric, sequentially, in the matching environment. It supports `--metrics`, `--limit`, sharding, `--device` and resume. GenEval is called only for GenEval prompts. `--aggregate-only` reads caches; it never evaluates images. Rerun aggregation after all shards finish.

Outputs are versioned under `RUN_DIR/evaluation/reports/EVAL_ID/`: `image_metrics.csv`, `pairwise_metrics.csv`, `group_metrics.csv`, `summary.csv`, and `report_manifest.json`. `evaluation/latest_report.json` points to the active report. Missing/failed/corrupt groups remain visible. Per-metric coverage accompanies every estimate; paired comparisons use matched prompt IDs and paired prompt bootstrap. GenEval overall follows the official mean over tasks, with stratified prompt bootstrap.

Read [METRICS.md](METRICS.md) for definitions. SAT remains unresolved by default until the HPSv3 score scale/checkpoint is verified on the VM and evidence is recorded in `configs/evaluation.yaml`. GenEval SAT is N/A. HPSv2 version, LPIPS settings, PickScore normalization and Vendi's kernel are explicitly documented rather than silently substituted.

Plots save PNG/PDF and exact CSV input. Only points with complete coverage on both axes are plotted; partial points remain in the CSV with `plotted=false`. Dominated points stay visible; non-dominated points are circled. No interpolated results are invented. Common figures separate datasets, protocols, prompt manifests, model revisions and metric identities. No result artifacts are created before actual reports exist.

## Efficiency and reproducibility limits

Group metadata records real transformer invocations, candidate-weighted forward NFE, generator autograd calls, reward-worker forward/backward calls, VAE decode calls, solver status/time/gap, synchronized wall time and GPU-seconds. Wall time includes upstream final diagnostic scoring; it excludes model loading and PNG disk writes. Cold start is separate. Internal transformer-block checkpoint recomputation is not counted as another full denoiser NFE.

Generator allocated/reserved peaks and reward-worker peaks are separate. Device-wide `nvidia-smi` sampling includes persistent workers but also unrelated processes, so use exclusive GPUs for comparison. `--profile-flops` creates a separate run identity: reported FLOPs cover only supported profiler operations in the generator process, excluding reward subprocesses, solver and unsupported operations. They are **not total FLOPs**. Inspect actual GPU type/count and package versions before comparing runtime.

Tests, data preparation, model downloads, preflight, pilots, evaluation and sweeps all remain pending on the rented VM. Static parsing does not establish dependency compatibility, runtime correctness or paper-level agreement.
