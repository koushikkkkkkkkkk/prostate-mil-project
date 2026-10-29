"""Frozen-CNN patch feature extraction -- Person 1.

Turns ``data/patches/{slide_id}/patches/*.png`` into the per-slide tensor that
every model consumes::

    data/features/{slide_id}.pt = {"features": [N, D], "coords": [N, 2],
                                   "patch_paths": [N]}

Default backbone: ImageNet-pretrained ResNet50 with the classifier removed
(2048-d penultimate features). A deterministic random projection to 1024-d is
available via ``--target-dim 1024`` for teams that want the smaller matrix --
it is a fixed linear map, applied identically to every slide, so nothing leaks
between splits.

Owner: Person 1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONE_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


class PatchImageDataset(Dataset):
    """Loads patch images listed in a ``coords.csv``-style file."""

    def __init__(self, patch_paths: Sequence[str], image_size: int = 256) -> None:
        self.patch_paths = list(patch_paths)
        self.image_size = image_size
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.patch_paths[index]) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


class FrozenFeatureExtractor(nn.Module):
    """Pretrained CNN with the classification head replaced by identity."""

    def __init__(
        self,
        backbone: str = "resnet50",
        target_dim: Optional[int] = None,
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        from torchvision import models

        backbone = backbone.lower()
        if backbone not in BACKBONE_DIMS:
            raise ValueError(f"backbone must be one of {sorted(BACKBONE_DIMS)}")
        weights = "DEFAULT" if pretrained else None
        network = getattr(models, backbone)(weights=weights)

        self.out_dim = BACKBONE_DIMS[backbone]
        self.backbone_name = backbone
        network.fc = nn.Identity()
        self.network = network

        projection: Optional[nn.Module] = None
        if target_dim and target_dim != self.out_dim:
            generator = torch.Generator().manual_seed(0)
            matrix = torch.randn(self.out_dim, target_dim, generator=generator)
            matrix = matrix / np.sqrt(self.out_dim)          # keep unit-ish scale
            projection = nn.Linear(self.out_dim, target_dim, bias=False)
            with torch.no_grad():
                projection.weight.copy_(matrix.t())
            projection.weight.requires_grad_(False)
            self.out_dim = target_dim
        self.projection = projection
        self.to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images)
        if self.projection is not None:
            features = self.projection(features)
        return features


@torch.no_grad()
def extract_patch_features(
    patch_paths: Sequence[str],
    extractor: FrozenFeatureExtractor,
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """``[N, D]`` feature matrix for one slide."""
    loader = DataLoader(
        PatchImageDataset(patch_paths, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    chunks: List[torch.Tensor] = []
    for images in loader:
        chunks.append(extractor(images.to(device)).cpu())
    if not chunks:
        return torch.zeros((0, extractor.out_dim))
    return torch.cat(chunks, dim=0)


def extract_slide_file(
    coords_csv: str | Path,
    out_path: str | Path,
    extractor: FrozenFeatureExtractor,
    batch_size: int = 32,
    image_size: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Extract one slide's features and save the ``.pt`` bag."""
    rows = list(csv.DictReader(Path(coords_csv).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No patches listed in {coords_csv}")

    patch_paths = [row["patch_path"] for row in rows]
    coords = torch.tensor(
        [[float(row["x"]), float(row["y"])] for row in rows], dtype=torch.float32
    )
    features = extract_patch_features(
        patch_paths, extractor, batch_size=batch_size, image_size=image_size, device=device
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"features": features, "coords": coords, "patch_paths": patch_paths},
        out_path,
    )
    return {
        "out": str(out_path),
        "n_patches": int(features.size(0)),
        "feature_dim": int(features.size(1)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frozen CNN features for tiled slides.")
    parser.add_argument("--patches-root", default="data/patches", help="folder with {slide_id}/coords.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--slide-id", default=None, help="single slide (default: all found)")
    parser.add_argument("--backbone", default="resnet50", choices=sorted(BACKBONE_DIMS))
    parser.add_argument("--target-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    extractor = FrozenFeatureExtractor(
        args.backbone, target_dim=args.target_dim, device=args.device
    )
    patches_root = Path(args.patches_root)
    candidates = (
        [patches_root / args.slide_id] if args.slide_id else sorted(
            path.parent for path in patches_root.glob("*/coords.csv")
        )
    )
    if not candidates:
        raise SystemExit(f"No coords.csv found under {patches_root}")

    summaries = []
    for slide_dir in candidates:
        summary = extract_slide_file(
            slide_dir / "coords.csv",
            Path(args.features_root) / f"{slide_dir.name}.pt",
            extractor,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
        )
        print(f"{slide_dir.name}: {summary['n_patches']} patches -> {summary['out']}")
        summaries.append(summary)

    print(f"done: {len(summaries)} slides, feature_dim={summaries[0]['feature_dim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
