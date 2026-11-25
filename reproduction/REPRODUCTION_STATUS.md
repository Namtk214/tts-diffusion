# Reproduction status

Date: 2026-11-25. **Implementation prepared; runtime validation and all experiments deferred to the rented VM.** This document is not a result report.

User overrides the guide's full datasets with initial 100-prompt subsets, and explicitly requests no local CPU smoke tests, dataset downloads or generation. No benchmark/model assets were downloaded locally. The two public code repositories and paper PDFs were obtained earlier as requested.

| Component | Implemented | Execution/verification status |
|---|---|---|
| Vendored SGI/SatisDive, exact source commits, patch records | Yes | Source inspected; licenses retained |
| Native and unified FLUX configs | Yes | Native SGI is repository defaults, not fully verified paper settings |
| Model pin/download and isolated environments | Yes | VM install, checkpoint access and hardware compatibility pending |
| Deterministic 10/50/100 prompt preparation | Yes | No datasets downloaded; actual source counts/subset hashes pending |
| Original samplers, model reuse, standard final SGI VAE in unified mode | Yes | No images generated |
| Atomic groups, PNG/hash validation, resume, sharding and failure logs | Yes | Tests written; not run |
| HPSv3/HPSv2/ImageReward/PickScore/CLIP | Yes | No scores computed; version/normalization notes in METRICS.md |
| DreamSim/DINO/LPIPS/Vendi | Yes | No scores computed; DINO equivalence test pending; Vendi/LPIPS adapted |
| Official GenEval subprocess integration | Yes | Legacy dependency stack and evaluator pending VM validation |
| Prompt bootstrap, paired comparisons, coverage-aware CSV/figures | Yes | Tests written; no result tables/figures fabricated |
| Runtime/NFE/autograd/worker memory/solver tracing | Yes | No measurements; profiler coverage is explicitly partial |
| Unit/contract tests and opt-in CUDA repeatability/gradient tests | Yes | Not executed at user's request |
| Static Python/config parsing | Yes | Passed; does not establish runtime correctness |
| E0 native 10 then 50 prompts | Prepared | NOT RUN |
| E1 two methods × three 100-prompt datasets | Prepared | NOT RUN |
| E2 Δ sweep, Pick-a-Pic and GenEval | Prepared | NOT RUN |
| E3 λ sweep, GenEval and Pick-a-Pic | Prepared | NOT RUN |
| E4 SGI M sweep, GenEval | Prepared | NOT RUN |

## Specific unresolved items

1. **VM execution:** all model loading, memory capacity, dependency imports, two-GPU steering gradients, feasible Gurobi selections, repeatability and offline scores still need real execution. No successful runtime or statistical paper agreement is claimed.
2. **Pick-a-Pic:** original source was inaccessible during metadata inspection. The downloader supports an explicitly selected public mirror; exact identity of the paper's 499 prompts remains unverified. Empty captions are recorded as exclusions, not guessed to be the paper's missing row.
3. **SGI paper version/settings:** the downloaded arXiv v1 is distinguishable from the later ICLR/OpenReview version. OpenReview's browser verification prevented text retrieval in this session. Current native config is the public code's 768/20/Tiny-decoder configuration. Unified changes are labeled explicitly.
4. **Metric variants:** HPSv2 defaults to v2.0; exact paper version is unresolved. Vendi's DINO-patch Gram kernel, LPIPS AlexNet/256 and PickScore absolute logits are explicit definitions/adaptations. Native SGI has no newly added HPSv3/ImageReward generation objective.
5. **SAT:** published fixed floors are configured, but SAT reproduction remains unresolved until HPSv3 scale/checkpoint verification is recorded. GenEval SAT is N/A. No new baseline or recalibration is required.
6. **Models:** all model IDs are declared, but immutable revisions are resolved on the VM before downloads. `models.lock.json` is deliberately not fabricated. Public metadata visibility does not prove gated weight access.
7. **Environment:** SatisDive generator and its legacy Qwen/HPSv3 worker have incompatible Transformers/Hub requirements and use separate environments. GenEval's CUDA 11.3 stack is isolated. Effective HPSv3 attention backend is logged; upstream can choose SDPA when FlashAttention is unavailable.
8. **Compute:** NFE, backward calls and partial profiler FLOPs are different quantities. Memory includes generator/reward-worker peaks and device-wide sampled resident usage, with documented limits. Native/unified result reuse is not an independent timing replicate.

After the VM pilot, update this file with run IDs, source/model hashes, environment locks, observed failures and evidence for any claim of matching paper settings. Keep full-paper claims separate from this 100-prompt subset benchmark.
