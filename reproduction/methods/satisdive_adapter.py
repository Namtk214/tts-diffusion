"""Call upstream run_ours_one_prompt; preserve its gradients and replacement rule."""
import math
import os
from pathlib import Path
import sys
from benchmark.common import ROOT
from methods.assets import cached_model, require_cuda, prepare_hpsv3_assets
from evaluation.efficiency import GenerationMeter


class SatisDiveAdapter:
    def __init__(self, config, device="cuda:0", reward_device=1, work_dir=None, profile_flops=False):
        import torch
        from diffusers import FluxPipeline
        self.shared_gpu = config["protocol"] == "single_gpu_768"
        self.hardware = require_cuda(device, reward_device, shared=self.shared_gpu)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.config, self.device, self.reward_device, self.profile_flops = config, device, reward_device, profile_flops
        python = os.environ.get("SATISDIVE_REWARD_PYTHON")
        if not python or not Path(python).is_file():
            raise RuntimeError("Set SATISDIVE_REWARD_PYTHON to the separate HPSv3 environment's bin/python")
        os.environ["FLUX_HPSV3_PYTHON"] = str(Path(python).absolute())
        os.environ["FLUX_HPSV3_SITE"] = ""
        self.reward_assets = prepare_hpsv3_assets(config["model_revisions"], Path(work_dir) / "assets")
        sys.path.insert(0, str(ROOT / "third_party/satisdive/implementation/flux"))
        import ours
        from rewards.clients import RewardClient
        self.run = ours.run_ours_one_prompt
        self.pipe = FluxPipeline.from_pretrained(cached_model(config["model_revisions"]["flux"]),
                    torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        # Parameters stay frozen, while autograd through latents and features remains enabled.
        self.pipe.transformer.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.transformer.enable_gradient_checkpointing()
        b = config["backbone"]
        self.capture = ours._Block12Capture(num_generated_tokens=(b["height"]//16)*(b["width"]//16),
            block_idx=12, grid=1, h_tok=b["height"]//16, w_tok=b["width"]//16)
        self.pipe.transformer.transformer_blocks[12].attn.processor = self.capture
        # Keep text encoder weights in host RAM between prompts. The GPU
        # computes the same prompt embeddings once before denoising.
        if self.shared_gpu:
            encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
            encode_prompt = self.pipe.encode_prompt
            for encoder in encoders:
                encoder.to("cpu")
            @torch.no_grad()
            def encode_then_park(*args, **kwargs):
                try:
                    for encoder in encoders:
                        encoder.to(device)
                    return encode_prompt(*args, **kwargs)
                finally:
                    for encoder in encoders:
                        encoder.to("cpu")
                    torch.cuda.empty_cache()
            self.pipe.encode_prompt = encode_then_park
            torch.cuda.empty_cache()
        self.client = RewardClient.hpsv3_grad(gpu_id=reward_device, park_model=self.shared_gpu)
        self.metadata = {"scheduler": dict(self.pipe.scheduler.config), "hardware": self.hardware,
            "noise_policy": "upstream_single_cuda_generator_batched_latents; same prompt seed, not identical SGI noise",
            "update_indexing": "zero_based_enumerate", "max_sequence_length": 512,
            "reward_assets": self.reward_assets, "gradient_checkpointing": True}
        self.metadata["reward_worker"] = self.client.worker_metadata

    def generate_group(self, prompt, seed):
        import torch
        b, s = self.config["backbone"], self.config["satisdive"]
        self.capture.store.clear()
        calls = []
        original = self.client.reward_and_grad
        def checked(images, prompts):
            if self.shared_gpu:
                torch.cuda.empty_cache()
            values, gradient = original(images, prompts)
            if not all(math.isfinite(x) for x in values) or not torch.isfinite(gradient).all():
                raise RuntimeError("Non-finite reward/gradient from HPSv3")
            calls.append({**getattr(self.client, "last_profile", {}),
                          "gradient_norm": float(gradient.float().norm())})
            return values, gradient
        self.client.reward_and_grad = checked
        try:
            with GenerationMeter(self.pipe, self.device, gpu_count=len(self.hardware), profile_flops=self.profile_flops) as meter:
                images, rewards, diag = self.run(self.pipe, self.client, self.capture, prompt, 4,
                    b["height"], b["width"], b["steps"], self.device, torch.bfloat16, seed,
                    set(s["update_steps"]), None, None, None, 1.0, 1.0, s["update_multiplier"], None,
                    s["noise_amplitude"], reward_ramp=True, clone_dud=True, clone_start_p=0.5,
                    tau_relmax_offset=s["delta"], delta_scale=1.0, verbose=False)
            # Upstream may catch final scoring failures. Record them, but a failed
            # steering gradient must never be presented as a successful generation.
            for rec in diag["steps"]:
                for field in ("g_reward_norm", "g_div_norm"):
                    if not all(math.isfinite(x) for x in rec[field]):
                        raise RuntimeError(f"Non-finite {field} at step {rec['step']}")
            meter.stats.update({"reward_forward_calls": sum(x.get("reward_forward_calls", 0) for x in calls),
                                "reward_backward_calls": sum(x.get("reward_backward_calls", 0) for x in calls),
                                "reward_worker_peak_allocated_bytes": max((x.get("peak_allocated_bytes", 0) for x in calls), default=0),
                                "reward_worker_peak_reserved_bytes": max((x.get("peak_reserved_bytes", 0) for x in calls), default=0)})
            return images, {"candidate_ids": list(range(4)), "initial_seed": seed,
                            "candidate_seeds": None, "stats": meter.stats, "trace": diag,
                            "reward_worker_trace": calls, "rewards_trace": rewards,
                            "gradient_validation": {"finite": True, "nonzero_reward_gradient_observed": any(x["gradient_norm"] > 0 for x in calls),
                                "nonzero_diversity_gradient_observed": any(any(v > 0 for v in r["g_div_norm"]) for r in diag["steps"])}}
        finally:
            self.client.reward_and_grad = original

    def close(self):
        self.client.close()
