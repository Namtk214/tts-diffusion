"""Explicit phase residency for the 40 GiB single-GPU profile.

No inference hooks: SatisDive calls transformer forwards and autograd directly.
Moving a model while its backward graph is live is deliberately forbidden at
the reward boundary. Upstream requests image cotangents in its no-grad pass,
then reconstructs the FLUX graph after the reward worker has parked its model.
"""
from contextlib import contextmanager


class PhaseResidency:
    def __init__(self, pipe, device):
        self.pipe, self.device = pipe, device

    def park_diffusion(self):
        import torch
        self.pipe.transformer.to("cpu")
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    def load_diffusion(self):
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)

    def install_prompt_encoder(self):
        import torch
        original = self.pipe.encode_prompt
        encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
        for encoder in encoders:
            encoder.requires_grad_(False).to("cpu")

        @torch.no_grad()
        def encode(*args, **kwargs):
            self.park_diffusion()
            try:
                for encoder in encoders:
                    encoder.to(self.device)
                return original(*args, **kwargs)
            finally:
                for encoder in encoders:
                    encoder.to("cpu")
                torch.cuda.empty_cache()
                self.load_diffusion()

        self.pipe.encode_prompt = encode

    @contextmanager
    def reward_phase(self, images):
        import torch
        if torch.is_grad_enabled() or images.requires_grad:
            raise RuntimeError("Cannot park FLUX with a live reward-input autograd graph")
        self.park_diffusion()
        try:
            yield
        finally:
            # The synchronous worker reply is sent only after HPSv3 returns
            # to CPU and clears its allocator, including on worker errors.
            self.load_diffusion()

    @staticmethod
    def metadata():
        return {"policy": "phase_swap_bf16_40gb", "quantization": None,
                "text_encoders": "GPU only while diffusion weights are on CPU",
                "reward": "HPSv3 and FLUX weights never co-resident on GPU"}
