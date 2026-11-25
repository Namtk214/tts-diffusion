"""Actual module-call counters; do not equate NFE and FLOPs."""
import contextlib
import subprocess
import threading
import time


class ResidentMemorySampler:
    """Samples total resident memory on allocated devices, including reward workers.

    This is device-wide memory, not process attribution; use exclusive VM GPUs.
    """
    def __init__(self, devices):
        self.devices = devices
        self.samples = []
        self.stop_event = threading.Event()

    def start(self):
        def poll():
            while not self.stop_event.is_set():
                try:
                    raw = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], timeout=5, text=True)
                    values = {}
                    for i, line in enumerate(raw.splitlines()):
                        uuid, memory, util = [v.strip() for v in line.split(",")]
                        if str(i) in self.devices or uuid in self.devices or any(uuid.startswith(d) for d in self.devices if d.startswith("GPU-")):
                            values[uuid] = {"memory_mib": float(memory), "util_pct": float(util)}
                    if values:
                        self.samples.append(values)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                self.stop_event.wait(0.5)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set(); self.thread.join(timeout=6)
        return {"scope": "device_wide_including_other_processes", "interval_s": 0.5,
                "samples": len(self.samples), "peak_sum_mib": max(
                    (sum(v["memory_mib"] for v in row.values()) for row in self.samples), default=None),
                "peak_per_gpu_mib": {gpu: max(row.get(gpu, {}).get("memory_mib", 0) for row in self.samples)
                                     for gpu in {k for row in self.samples for k in row}}}


class GenerationMeter:
    def __init__(self, pipe, device, gpu_count=1, profile_flops=False, final_vae=None):
        self.pipe, self.device, self.gpu_count = pipe, device, gpu_count
        self.profile_flops = profile_flops
        self.final_vae = final_vae

    def __enter__(self):
        import torch
        self.torch = torch
        self.calls = self.weighted = self.backward_calls = self.decoder_calls = 0
        def forward(module, args, kwargs):
            value = kwargs.get("hidden_states", args[0] if args else None)
            self.calls += 1
            self.weighted += int(value.shape[0])
        self.hook = self.pipe.transformer.register_forward_pre_hook(forward, with_kwargs=True)
        self.original_decode = self.pipe.vae.decode
        def decode(*args, **kwargs):
            self.decoder_calls += 1
            return self.original_decode(*args, **kwargs)
        self.pipe.vae.decode = decode
        self.original_final_decode = None
        if self.final_vae is not None and self.final_vae is not self.pipe.vae:
            self.original_final_decode = self.final_vae.decode
            def final_decode(*args, **kwargs):
                self.decoder_calls += 1
                return self.original_final_decode(*args, **kwargs)
            self.final_vae.decode = final_decode
        self.original_grad = torch.autograd.grad
        def grad(*args, **kwargs):
            self.backward_calls += 1
            return self.original_grad(*args, **kwargs)
        torch.autograd.grad = grad
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.profiler = None
        if self.profile_flops:
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], with_flops=True, record_shapes=True)
            self.profiler.__enter__()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch = self.torch
        try:
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - self.start
            flops = None
            if self.profiler:
                self.profiler.__exit__(*exc)
                flops = sum(e.flops or 0 for e in self.profiler.key_averages())
            self.stats = {"runtime_s": elapsed, "denoiser_forward_calls": self.calls,
                          "candidate_weighted_forward_nfe": self.weighted,
                          "guidance_autograd_calls": self.backward_calls,
                          "vae_decode_calls_observed": self.decoder_calls,
                          "peak_allocated_bytes": {str(self.device): torch.cuda.max_memory_allocated(self.device)},
                          "peak_reserved_bytes": {str(self.device): torch.cuda.max_memory_reserved(self.device)},
                          "gpu_count": self.gpu_count, "allocated_gpu_seconds": elapsed * self.gpu_count,
                          "profiled_flops": flops,
                          "flops_coverage": "PyTorch supported ops in generator process only; NOT total FLOPs; "
                              "excludes reward subprocess, solver, and unsupported attention/backward ops" if self.profile_flops else "not_profiled",
                          "counter_scope": "full denoiser module calls; internal block checkpoint recomputation is not another denoiser NFE; reward-worker autograd logged separately",
                          "runtime_scope": "sampler, steering, solver, decode and PIL conversion; excludes cold start and PNG disk writes"}
        finally:
            self.hook.remove()
            self.pipe.vae.decode = self.original_decode
            if self.original_final_decode is not None:
                self.final_vae.decode = self.original_final_decode
            torch.autograd.grad = self.original_grad
        return False
