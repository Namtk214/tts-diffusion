"""All six unordered pairs; Vendi uses a verified PSD Gram matrix."""
from itertools import combinations
import numpy as np
from methods.assets import cached_model

PAIRS = tuple(combinations(range(4), 2))


def vendi_from_kernel(kernel):
    k = np.asarray(kernel, dtype=np.float64)
    if k.shape != (4, 4) or not np.isfinite(k).all() or not np.allclose(k, k.T, atol=1e-6):
        raise ValueError("Vendi requires a finite symmetric 4x4 kernel")
    if not np.allclose(np.diag(k), 1, atol=1e-5):
        raise ValueError("Vendi kernel must have unit diagonal")
    eigenvalues = np.linalg.eigvalsh(k / 4)
    if eigenvalues.min() < -1e-6:
        raise ValueError("Vendi kernel is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues /= eigenvalues.sum()
    positive = eigenvalues[eigenvalues > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


class DiversityScorer:
    def __init__(self, name, models, config, device, work_dir):
        self.name, self.device, self.config = name, device, config
        self.metadata = {"metric": name, "precision": "float32", "pairs": list(PAIRS)}
        if name == "dino":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(cached_model(models["dino"]), local_files_only=True).eval().to(device)
            self.metadata.update(features="DINOv2-base patch tokens; no CLS", preprocessing="SGI bilinear 256 + ImageNet normalization",
                distance="mean per-aligned-patch cosine distance", vendi_kernel="Gram of concatenated unit patch vectors, unit diagonal",
                vendi_paper_status="adapted; paper kernel unresolved")
        elif name == "dreamsim":
            from dreamsim import dreamsim
            from benchmark.common import ROOT
            self.model, self.preprocess = dreamsim(pretrained=True, device=device,
                cache_dir=str(ROOT / "models/dreamsim"), normalize_embeds=True,
                dreamsim_type=config["dreamsim_type"], use_patch_model=False)
            self.model.eval()
            self.metadata.update(preprocessing="official DreamSim preprocessing", model=config["dreamsim_type"], patch_model=False)
        elif name == "lpips":
            import lpips
            self.model = lpips.LPIPS(net=config["lpips_net"], version=config["lpips_version"]).eval().to(device)
            self.metadata.update(network=config["lpips_net"], version=config["lpips_version"],
                preprocessing=f"bilinear resize to {config['lpips_resize']} square, RGB [-1,1]",
                paper_status="adapted explicit network/resolution; exact paper setting unresolved")
        else:
            raise ValueError(name)

    def score(self, images):
        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor
        with torch.inference_mode():
            extra = {}
            if self.name == "dino":
                from methods.scoring import dino_patches, patch_distance
                tensors = [to_tensor(im).unsqueeze(0).to(self.device)*2-1 for im in images]
                features = dino_patches(tensors, self.model, self.device)
                distances = patch_distance(features).cpu().numpy()
                flat = F.normalize(F.normalize(features, dim=-1).flatten(1), dim=-1).double()
                kernel = (flat @ flat.T).cpu().numpy()
                extra = {"vendi": vendi_from_kernel(kernel), "vendi_kernel": kernel.tolist()}
                values = [float(distances[i, j]) for i, j in PAIRS]
            elif self.name == "dreamsim":
                tensors = [self.preprocess(im).to(self.device) for im in images]
                values = [float(self.model(tensors[i], tensors[j]).item()) for i, j in PAIRS]
            else:
                tensors = [F.interpolate(to_tensor(im).unsqueeze(0).to(self.device),
                           size=(self.config["lpips_resize"],)*2, mode="bilinear", align_corners=False)*2-1 for im in images]
                values = [float(self.model(tensors[i], tensors[j]).item()) for i, j in PAIRS]
            return {"pair_values": values, **extra}
