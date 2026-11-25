"""Thin adapter over pinned upstream SGI; one loaded pipeline per process."""
from pathlib import Path
import sys
from benchmark.common import ROOT
from methods.assets import cached_model, require_cuda
from methods.scoring import make_dino_scorer
from evaluation.efficiency import GenerationMeter
from methods.memory import PhaseResidency


class SGIAdapter:
    def __init__(self, config, device="cuda:0", reward_device=None, work_dir=None, profile_flops=False):
        import torch
        from diffusers import FluxPipeline, AutoencoderTiny
        from transformers import CLIPModel, CLIPProcessor, AutoModel
        import functools
        self.hardware = require_cuda(device)
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.config, self.device, self.profile_flops = config, device, profile_flops
        sys.path.insert(0, str(ROOT / "third_party/sgi/src"))
        from my_utils.group_inference import run_group_inference
        from my_utils.scores import unary_clip_text_img_t
        self.run = run_group_inference
        models = config["model_revisions"]
        self.pipe = FluxPipeline.from_pretrained(cached_model(models["flux"]), torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
        shared = config["protocol"] == "single_gpu_768"
        if not shared:
            self.pipe.to(device)
        self.residency = PhaseResidency(self.pipe, device) if shared else None
        if self.residency:
            self.residency.install_prompt_encoder()
        self.final_vae = self.pipe.vae if config["generation"]["final_decoder"] == "standard_flux" else None
        self.pipe.vae = AutoencoderTiny.from_pretrained(cached_model(models["tiny_vae"]), torch_dtype=torch.bfloat16,
                                                        local_files_only=True).to(device)
        if self.final_vae is not None:
            self.final_vae.to(device)
            if shared:
                self.final_vae.enable_slicing()
        if self.residency:
            self.residency.load_diffusion()
        clip = CLIPModel.from_pretrained(cached_model(models["clip"]), local_files_only=True).eval().to(device)
        processor = CLIPProcessor.from_pretrained(cached_model(models["clip"]), local_files_only=True)
        dino = AutoModel.from_pretrained(cached_model(models["dino"]), local_files_only=True).eval().to(device)
        self.unary = functools.partial(unary_clip_text_img_t, device=device, m_clip=clip, preprocess_clip=processor)
        self.binary = make_dino_scorer(dino, device)
        self.metadata = {"memory_policy": self.residency.metadata() if self.residency else {"policy": "resident"}, "scheduler": dict(self.pipe.scheduler.config), "hardware": self.hardware,
            "noise_policy": "upstream_cpu_generator_per_candidate_seed_plus_index; differs from SatisDive CUDA batched RNG",
            "dino_implementation": "equivalent patchwise dot reduction, float32, microbatch=8",
            "native_status": "public_defaults_paper_resolution_unverified" if config["protocol"] == "native" else "adapted_unified",
            "max_sequence_length": 256, "skip_first_cfg": True}

    def generate_group(self, prompt, seed):
        import torch
        b, s = self.config["backbone"], self.config["sgi"]
        trace = []
        with GenerationMeter(self.pipe, self.device, profile_flops=self.profile_flops, final_vae=self.final_vae) as meter:
            images = self.run(self.pipe, model_name="flux-dev", prompt=prompt,
                height=b["height"], width=b["width"], num_inference_steps=b["steps"],
                guidance_scale=b["guidance_scale"], max_sequence_length=256,
                l_generator=[torch.Generator("cpu").manual_seed(seed + i) for i in range(s["initial_candidates"])],
                unary_score_fn=self.unary, binary_score_fn=self.binary,
                starting_candidates=s["initial_candidates"], output_group_size=4,
                pruning_ratio=s["dropping_ratio"], lambda_score=s["lambda_score"],
                final_vae=self.final_vae, diagnostics=trace, execution_device=self.device)
        meter.stats["solver_runtime_s"] = sum(r.get("solver", {}).get("runtime_s", 0) for r in trace)
        meter.stats["solver_time_limit_hits"] = sum(r.get("solver", {}).get("time_limit_hit", False) for r in trace)
        return images, {"candidate_ids": trace[-1]["final_candidate_ids"],
                        "initial_seed": seed, "candidate_seeds": [seed+i for i in trace[-1]["final_candidate_ids"]],
                        "stats": meter.stats, "trace": trace}

    def close(self):
        pass
