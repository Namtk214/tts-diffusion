"""Algebra and preprocessing equivalence, no model downloads required."""
from types import SimpleNamespace
import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F
from methods.scoring import patch_distance, dino_patches


def test_chunked_patch_distance_matches_upstream_broadcast():
    features = torch.randn((9, 324, 12), generator=torch.Generator().manual_seed(42))
    features[0, 0] = 0
    f = F.normalize(features, p=2, dim=-1)
    reference = (1-F.cosine_similarity(f.unsqueeze(1), f.unsqueeze(0), dim=-1)).mean(dim=-1)
    torch.testing.assert_close(patch_distance(features), reference, rtol=1e-5, atol=2e-6)


def test_bfloat_preview_normalization_promotes_to_float32():
    seen = []
    def model(pixel_values):
        seen.append(pixel_values)
        return SimpleNamespace(last_hidden_state=torch.ones(len(pixel_values), 325, 4))
    images = [torch.ones(1, 3, 32, 32, dtype=torch.bfloat16)*.25 for _ in range(9)]
    dino_patches(images, model, "cpu")
    assert [len(x) for x in seen] == [8, 1]
    assert all(x.dtype == torch.float32 for x in seen)
    expected = (torch.tensor(.625)-torch.tensor([.485, .456, .406]))/torch.tensor([.229, .224, .225])
    torch.testing.assert_close(seen[0][0, :, 0, 0], expected)
