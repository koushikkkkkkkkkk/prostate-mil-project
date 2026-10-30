"""Small patch-level CNN -- Person 4.

Binary benign-vs-malignant patch classifier used to build independent
probability maps (which are then compared with MIL attention maps) and as a
patch-level sanity check on the extracted tiles.

    Conv(3->32, 3x3) - BN - ReLU - MaxPool   -> 128x128
    Conv(32->64)     - BN - ReLU - MaxPool   -> 64x64
    Conv(64->128)    - BN - ReLU - MaxPool   -> 32x32
    Conv(128->128)   - BN - ReLU - AdaptiveAvgPool -> 128
    Linear(128, 2)   (benign / malignant)

``num_classes=6`` is also supported if patch-level ISUP labels are available.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class PatchCNNConfig:
    in_channels: int = 3
    num_classes: int = 2
    width: int = 32
    dropout: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _conv_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class PatchCNN(nn.Module):
    """4-conv-layer patch classifier, deliberately small (trains on CPU/Colab)."""

    def __init__(self, config: Union[PatchCNNConfig, Dict[str, Any], None] = None, **overrides: Any):
        super().__init__()
        if config is None:
            config = PatchCNNConfig(**overrides)
        elif isinstance(config, dict):
            config = PatchCNNConfig(**{**config, **overrides})
        self.config = config
        w = config.width

        self.features = nn.Sequential(
            _conv_block(config.in_channels, w),
            _conv_block(w, w * 2),
            _conv_block(w * 2, w * 4),
            _conv_block(w * 4, w * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(w * 4, config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: [B, 3, H, W] float in [0, 1] -> logits [B, num_classes]."""
        return self.classifier(self.features(images))

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(images), dim=-1)


# --------------------------------------------------------------------------- #
# Patch-level data plumbing
# --------------------------------------------------------------------------- #
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def load_patch_tensor(
    path: str | Path,
    image_size: int = 256,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    augment: bool = False,
) -> torch.Tensor:
    """Read one RGB patch -> normalised ``[3, H, W]`` tensor."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    if augment:
        if torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        if torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[1])
        if torch.rand(1).item() < 0.5:
            k = int(torch.randint(0, 4, (1,)).item())
            tensor = torch.rot90(tensor, k=k, dims=[1, 2])

    mean = mean if mean is not None else torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = std if std is not None else torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


class PatchDataset(Dataset):
    """Reads ``patch_path,label[,slide_id,x,y]`` rows from a CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 256,
        augment: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.rows: List[Dict[str, Any]] = []
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = row.get("patch_path") or row.get("path") or row.get("file")
                if not path:
                    continue
                self.rows.append({**row, "patch_path": path})
        if not self.rows:
            raise ValueError(f"No usable rows in patch CSV {csv_path}")

        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[index]
        tensor = load_patch_tensor(
            row["patch_path"],
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
            augment=self.augment,
        )
        label = int(float(row.get("label", row.get("isup_grade", 0)) or 0))
        return tensor, label


def train_patch_cnn(
    train_csv: str | Path,
    val_csv: Optional[str | Path] = None,
    out_path: str | Path = "results/metrics/patch_cnn.pt",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    image_size: int = 256,
    config: Optional[PatchCNNConfig] = None,
    device: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Minimal, dependency-light training loop for the patch baseline."""
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    model = PatchCNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        PatchDataset(train_csv, image_size=image_size, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(PatchDataset(val_csv, image_size=image_size), batch_size=batch_size, num_workers=num_workers)
        if val_csv
        else None
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)

        record = {"epoch": epoch, "train_loss": running / max(seen, 1)}
        if val_loader is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    predictions = model(images).argmax(dim=1)
                    correct += int((predictions == labels).sum())
                    total += labels.numel()
            record["val_accuracy"] = correct / max(total, 1)
        history.append(record)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, out_path)
    return {"checkpoint": str(out_path), "history": history}


@torch.no_grad()
def predict_patch_grid(
    model: PatchCNN,
    patch_paths: Iterable[str],
    coords: Optional[torch.Tensor] = None,
    image_size: int = 256,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict malignant probability per patch and (optionally) place it in a grid."""
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    model = model.to(device).eval()
    paths = list(patch_paths)
    if not paths:
        raise ValueError("patch_paths is empty")

    malignant_index = -1  # last logit == malignant / highest grade
    probabilities: List[float] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = torch.stack(
            [load_patch_tensor(path, image_size=image_size) for path in chunk]
        ).to(device)
        probs = torch.softmax(model(batch), dim=-1)[:, malignant_index]
        probabilities.extend(probs.cpu().tolist())

    result: Dict[str, Any] = {"patch_paths": paths, "malignant_prob": probabilities}
    if coords is not None:
        result["coords"] = coords.tolist() if torch.is_tensor(coords) else list(coords)
    return result
