from pathlib import Path
import os
import yaml
from benchmark.common import ROOT, atomic_text


def cached_model(entry):
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(entry["id"], revision=entry["revision"], local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"Pinned model {entry['id']}@{entry['revision']} is not cached. "
                           "Authenticate/accept upstream model terms and run scripts/download_models.py on the VM.") from e


def prepare_hpsv3_assets(models, directory):
    root = cached_model(models["hpsv3"])
    qwen = cached_model(models["qwen"])
    checkpoint = Path(root) / "HPSv3.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = yaml.safe_load((ROOT / "third_party/satisdive/third_party/HPSv3/hpsv3/config/HPSv3_7B.yaml").read_text())
    config["model_name_or_path"] = qwen
    target = Path(directory) / "hpsv3.pinned.yaml"
    atomic_text(target, yaml.safe_dump(config, sort_keys=False))
    os.environ["TTS_HPSV3_CONFIG"] = str(target)
    os.environ["TTS_HPSV3_CHECKPOINT"] = str(checkpoint)
    return {"checkpoint": str(checkpoint), "qwen_snapshot": qwen, "config": str(target)}


def require_cuda(device, reward_device=None, *, shared=False):
    import torch
    import platform
    if platform.system() != "Linux" or not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Generation requires a Linux CUDA VM; CPU/MPS generation is not supported")
    index = torch.device(device).index or 0
    if index >= torch.cuda.device_count():
        raise ValueError("Generation device is outside the visible GPU allocation")
    if reward_device is not None and ((reward_device == index) != shared or not 0 <= reward_device < torch.cuda.device_count()):
        raise ValueError("Reward device must be visible and match the generation device only for the single_gpu_768 protocol")
    torch.cuda.set_device(index)
    return [{"index": i, "name": torch.cuda.get_device_name(i),
             "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in sorted(set([index] + ([] if reward_device is None else [reward_device])))]
