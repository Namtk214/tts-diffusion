"""Offline rewards. No generation or group-relative softmax in these scorers."""
from pathlib import Path
import sys
from benchmark.common import ROOT
from methods.assets import cached_model, prepare_hpsv3_assets


class RewardScorer:
    def __init__(self, name, models, config, device, work_dir):
        import torch
        self.name, self.device, self.config = name, device, config
        self.metadata = {"metric": name, "inference_only": True}
        if name == "hpsv3":
            assets = prepare_hpsv3_assets(models, work_dir)
            sys.path.insert(0, str(ROOT / "third_party/satisdive/third_party/HPSv3"))
            from hpsv3.inference import HPSv3RewardInferencer
            self.model = HPSv3RewardInferencer(device=device, differentiable=False)
            self.metadata.update(assets=assets, precision="bfloat16", score="mu, logits[:,0]",
                                 preprocessing="bundled HPSv3 Qwen2VL, 256*28*28 pixels")
        elif name == "hpsv2":
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            # Full HPS checkpoint replaces every parameter; avoid downloading the redundant LAION initializer.
            self.model, _, self.preprocess = create_model_and_transforms(
                "ViT-H-14", pretrained=None, precision="amp", device=device, jit=False,
                force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
                force_image_size=None, pretrained_image=False, image_mean=None, image_std=None,
                light_augmentation=True, aug_cfg={}, output_dict=True,
                with_score_predictor=False, with_region_predictor=False)
            filename = {"v2.0": "HPS_v2_compressed.pt", "v2.1": "HPS_v2.1_compressed.pt"}[config["hpsv2_version"]]
            checkpoint = Path(cached_model(models["hpsv2"])) / filename
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["state_dict"], strict=True)
            self.model.eval()
            self.tokenizer = get_tokenizer("ViT-H-14")
            self.metadata.update(checkpoint=str(checkpoint), precision="float32 weights, CUDA fp16 autocast",
                                 score="native normalized dot product, not multiplied by 100",
                                 version=config["hpsv2_version"], paper_version_status="unresolved")
        elif name == "image_reward":
            import ImageReward as RM
            root = Path(cached_model(models["image_reward"]))
            # BLIP creates a BERT tokenizer internally; route it to the pinned local snapshot.
            from transformers import BertTokenizer
            tokenizer_path = cached_model(models["bert_tokenizer"])
            original = BertTokenizer.from_pretrained
            def local_tokenizer(cls, name, *args, **kwargs):
                if str(name) == "bert-base-uncased":
                    name = tokenizer_path
                kwargs["local_files_only"] = True
                return original(name, *args, **kwargs)
            descriptor = BertTokenizer.__dict__.get("from_pretrained")
            BertTokenizer.from_pretrained = classmethod(local_tokenizer)
            try:
                self.model = RM.load(str(root / "ImageReward.pt"), device=device,
                                     med_config=str(root / "med_config.json"))
            finally:
                if descriptor is None:
                    del BertTokenizer.from_pretrained
                else:
                    BertTokenizer.from_pretrained = descriptor
            self.metadata.update(precision="float32", preprocessing="official ImageReward v1.0, bicubic 224 center crop",
                                 score="official standardized reward", text_length=35)
        elif name in ("pickscore", "clip_score"):
            from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
            if name == "pickscore":
                self.model = AutoModel.from_pretrained(cached_model(models["pickscore"]), local_files_only=True).eval().to(device)
                self.processor = AutoProcessor.from_pretrained(cached_model(models["pickscore_processor"]), local_files_only=True)
                self.metadata.update(score="exp(logit_scale)*cosine; absolute logit, no softmax",
                                     paper_normalization_status="explicit adapted definition")
            else:
                self.model = CLIPModel.from_pretrained(cached_model(models["clip"]), local_files_only=True).eval().to(device)
                self.processor = CLIPProcessor.from_pretrained(cached_model(models["clip"]), local_files_only=True)
                self.metadata.update(score="cosine, no scale or clamp", preprocessing="SGI bilinear 224, native normalization",
                                     text_policy="CLIP context truncation at 77 tokens for long captions")
            self.metadata["precision"] = "float32"
        else:
            raise ValueError(f"Unknown reward {name}")

    def score(self, images, prompt):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            if self.name == "hpsv3":
                # Score individually like the generation worker; avoids batch padding differences/OOM.
                return [float(self.model.reward([prompt], [im])[0][0]) for im in images]
            if self.name == "hpsv2":
                text = self.tokenizer([prompt]).to(self.device)
                values = []
                for image in images:
                    x = self.preprocess(image).unsqueeze(0).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                        out = self.model(x, text)
                        values.append(float((out["image_features"] @ out["text_features"].T)[0, 0]))
                return values
            if self.name == "image_reward":
                return [float(self.model.score(prompt, im)) for im in images]
            if self.name == "clip_score":
                from torchvision.transforms.functional import to_tensor
                x = torch.stack([to_tensor(im) for im in images]).to(self.device)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
                std = x.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
                text = self.processor.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to(self.device)
                output = self.model(pixel_values=(x-mean)/std, **text)
                return (output.logits_per_image / self.model.logit_scale.exp()).flatten().cpu().tolist()
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            tokens = self.processor(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            features = F.normalize(self.model.get_image_features(**inputs), dim=-1)
            text = F.normalize(self.model.get_text_features(**tokens), dim=-1)
            return (self.model.logit_scale.exp() * features @ text.T).flatten().cpu().tolist()
