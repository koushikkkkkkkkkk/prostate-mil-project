"""Synthetic bag generator -- the "mock data bridge" that unblocks the team.

Before slides are downloaded and tiled, everybody can build and unit-test their
model against tensors of the right shape::

    from src.utils.mock_data import generate_mock_bag, generate_mock_batch

    features, label = generate_mock_bag(num_patches=500, feature_dim=1024)
    bags, labels = generate_mock_batch(batch_size=4, feature_dim=1024)

The generator puts a *learnable* signal in the data: a fraction of patches gets
an additive class-dependent offset, so a working model should beat chance on
this data. That makes it useful as a smoke test ("can my model overfit a
synthetic bag?") and useless as a scientific result -- never report metrics
computed on mock data.

Owner: Person 1.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

NUM_CLASSES = 6


def generate_mock_bag(
    num_patches: int = 500,
    feature_dim: int = 1024,
    label: Optional[int] = None,
    num_classes: int = NUM_CLASSES,
    signal_strength: float = 1.5,
    signal_fraction: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, int]:
    """One bag: ``features [num_patches, feature_dim]`` plus an ISUP label."""
    if num_patches < 2:
        raise ValueError("num_patches must be >= 2")
    if label is None:
        label = int(torch.randint(0, num_classes, (1,), generator=generator).item())
    if not 0 <= label < num_classes:
        raise ValueError(f"label must be in [0, {num_classes - 1}]")

    features = torch.randn(num_patches, feature_dim, generator=generator)

    # Class-specific direction in feature space, injected into a subset of patches.
    direction = torch.zeros(feature_dim)
    direction[label % feature_dim] = 1.0
    direction[(label * 37 + 11) % feature_dim] = 0.5
    n_signal = max(1, int(round(signal_fraction * num_patches)))
    index = torch.randperm(num_patches, generator=generator)[:n_signal]
    features[index] += signal_strength * (1.0 + 0.2 * label) * direction
    return features.float(), int(label)


def generate_mock_batch(
    batch_size: int = 2,
    num_patches: int = 500,
    feature_dim: int = 1024,
    num_classes: int = NUM_CLASSES,
    variable_length: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """List of padded-free bags plus their labels ``[batch_size]``."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    bags: List[torch.Tensor] = []
    labels: List[int] = []
    for _ in range(batch_size):
        patches = (
            int(torch.randint(max(8, num_patches // 2), num_patches + 1, (1,), generator=generator).item())
            if variable_length
            else num_patches
        )
        features, label = generate_mock_bag(
            num_patches=patches,
            feature_dim=feature_dim,
            num_classes=num_classes,
            generator=generator,
        )
        bags.append(features)
        labels.append(label)
    return bags, torch.tensor(labels, dtype=torch.long)


def generate_mock_loader(
    num_bags: int = 32,
    feature_dim: int = 1024,
    num_classes: int = NUM_CLASSES,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, int]]:
    """Deterministic list of ``(features, label)`` pairs for tests/notebooks."""
    generator = torch.Generator().manual_seed(seed)
    return [
        generate_mock_bag(
            num_patches=int(torch.randint(20, 80, (1,), generator=generator).item()),
            feature_dim=feature_dim,
            label=index % num_classes,
            num_classes=num_classes,
            generator=generator,
        )
        for index in range(num_bags)
    ]


def collate_mock(bags: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[B, N, D]`` stack + validity mask for variable-length mock bags."""
    max_n = max(bag.size(0) for bag in bags)
    dim = bags[0].size(1)
    features = torch.zeros(len(bags), max_n, dim)
    mask = torch.zeros(len(bags), max_n, dtype=torch.bool)
    for i, bag in enumerate(bags):
        features[i, : bag.size(0)] = bag
        mask[i, : bag.size(0)] = True
    return features, mask
