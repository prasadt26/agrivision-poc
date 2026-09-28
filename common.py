"""Shared model + preprocessing. Imported by train, predict and the API.

The single most common silent failure in a vision service is training and serving
resizing differently. Both paths call build_transform() here; neither re-implements it.
"""

from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transform(size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            # hue stays near zero on purpose: Black vs Yellow Sigatoka, and chlorosis
            # vs necrosis, are colour judgements. Jittering hue trains the model to
            # ignore the feature it most needs.
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def create_model(arch: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


class Classifier:
    """A loaded artifact: weights + labels + calibration temperature."""

    def __init__(self, artifact_dir: str | Path, device: str = "cpu"):
        self.dir = Path(artifact_dir)
        cfg = json.loads((self.dir / "config.json").read_text())
        self.labels: list[str] = cfg["labels"]
        self.arch: str = cfg["arch"]
        self.size: int = cfg["image_size"]
        self.temperature: float = cfg.get("temperature", 1.0)
        self.version: str = cfg.get("version", self.dir.name)
        self.device = torch.device(device)

        self.model = create_model(self.arch, len(self.labels), pretrained=False)
        state = torch.load(self.dir / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.transform = build_transform(self.size, train=False)

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x) / self.temperature
        probs = logits.softmax(dim=1)[0]
        k = min(top_k, len(self.labels))
        conf, idx = probs.topk(k)
        ranked = [
            {"label": self.labels[i], "confidence": round(float(c), 4)}
            for c, i in zip(conf.tolist(), idx.tolist())
        ]
        return {
            "model_version": self.version,
            "diagnosis": {
                "label": ranked[0]["label"],
                "display": ranked[0]["label"].replace("_", " ").title(),
                "confidence": ranked[0]["confidence"],
                "alternatives": ranked[1:],
            },
        }
