from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _to_nchw_tensor(patches: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(patches, dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected patches with 4 dims, got shape {tuple(tensor.shape)}."
        )

    if tensor.shape[-1] in (1, 2, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.shape[1] not in (1, 2, 3, 4):
        raise ValueError(
            "Could not infer channel axis. Expected NCHW or NHWC with <=4 channels."
        )
    return tensor


def _standardize_patch(patch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = patch.mean(dim=(1, 2), keepdim=True)
    std = patch.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    return (patch - mean) / std


class PatchPairAugmentation:
    def __init__(
        self,
        *,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.75,
        noise_std: float = 0.03,
        scale_jitter: float = 0.15,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        out = patch

        if torch.rand(1).item() < self.flip_prob:
            out = torch.flip(out, dims=(2,))
        if torch.rand(1).item() < self.flip_prob:
            out = torch.flip(out, dims=(1,))

        if torch.rand(1).item() < self.rotate_prob:
            k = int(torch.randint(low=0, high=4, size=(1,)).item())
            out = torch.rot90(out, k=k, dims=(1, 2))

        if self.scale_jitter > 0:
            scale = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * self.scale_jitter
            out = out * scale

        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std

        return out


class ContrastivePatchPairDataset(Dataset):
    """
    Returns two augmented views of the same input patch for contrastive learning.
    """

    def __init__(
        self,
        patches: np.ndarray | torch.Tensor,
        *,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        standardize: bool = True,
    ) -> None:
        self.patches = _to_nchw_tensor(patches)
        self.augment = augment or PatchPairAugmentation()
        self.standardize = standardize

    def __len__(self) -> int:
        return self.patches.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_patch = self.patches[idx].clone()
        if self.standardize:
            base_patch = _standardize_patch(base_patch)

        view_one = self.augment(base_patch.clone())
        view_two = self.augment(base_patch.clone())
        return view_one, view_two


def create_contrastive_data_loader(
    patches: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    standardize: bool = True,
) -> DataLoader:
    dataset = ContrastivePatchPairDataset(
        patches,
        augment=augment,
        standardize=standardize,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
