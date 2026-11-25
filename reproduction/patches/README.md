# Changes to vendored source

Original commits are recorded in `../third_party/upstream.lock.json`. The `.patch` files in this directory describe changes relative to those commits and can be applied from the corresponding upstream repository root. Existing algorithms and licenses are retained.

SGI:

- Add optional diagnostics for active/selected candidate IDs and Gurobi status, incumbent count, runtime, objective, finite gap and time-limit hits.
- Fail on missing or wrong-cardinality incumbents; keep the original objective, Gurobi solver and time limit. Dispose solved models.
- Add an optional final VAE argument. Without it, native decoder behavior is unchanged. Unified uses standard FLUX VAE only for final images.
- Equivalent memory-bounded DINO scoring lives in `methods/scoring.py`, outside the vendored source; its tests remain unexecuted until VM validation.

SatisDive:

- Allow optional pinned local HPSv3 configuration/checkpoint paths via environment variables; original defaults remain when absent.
- Add finite gradient/reward checks and reward-worker forward/backward/memory/environment instrumentation. Keep the upstream differentiable ×255 preprocessing and autograd route.
- Bound worker waits and cleanup, correctly handle multiple stdout lines already read into a buffer, and preserve allocated logical GPU mapping. No single-GPU offload is introduced.

The SatisDive optimization objective, feature capture, ramp, replacement rule, zero-based update schedule, SDE sampler and calibration are reused directly. Runtime validation of all patches is pending; syntax checks alone are not a numerical equivalence claim.
