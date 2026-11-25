"""Same SGI patch features and preprocessing, without N*N*P*D broadcasting."""
def patch_distance(features):
    import torch.nn.functional as F
    # Original applies F.normalize then F.cosine_similarity (normalizes again).
    f = F.normalize(F.normalize(features, p=2, dim=-1), p=2, dim=-1, eps=1e-8)
    import torch
    similarity = torch.einsum("ipd,jpd->ij", f, f) / f.shape[1]
    return 1 - similarity


def dino_patches(images, model, device):
    import torch
    import torch.nn.functional as F
    x = torch.cat(images).to(device)
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    x = x * 0.5 + 0.5
    # Upstream constants are float32: preserve promotion after bf16 interpolation/range conversion.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    # Feature extraction is microbatched, with identical preprocessing.
    features = []
    for batch in ((x - mean) / std).split(8):
        features.append(model(pixel_values=batch).last_hidden_state[:, 1:, :])
    return torch.cat(features)


def make_dino_scorer(model, device):
    import torch
    @torch.no_grad()
    def score(images):
        return patch_distance(dino_patches(images, model, device)).triu(diagonal=1).cpu().numpy()
    return score
