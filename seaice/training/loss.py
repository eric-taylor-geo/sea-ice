import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Mask out invalid target pixels (e.g., land encoded as 25.5) per item
    in batch.

    Assumes:
      preds, targets shape: (B, C, H, W)  (works for any shape as long as
      broadcast matches)
      valid pixels: targets <= 1
      invalid pixels: targets > 1
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.shape != targets.shape:
            raise ValueError(
                "preds and targets must have same shape, got {preds.shape} vs {targets.shape}"
            )

        preds = preds.float()
        targets = targets.float()

        # per-item mask: True where valid ocean pixels
        mask = targets <= 1.0

        # squared error, masked
        squared_error = (preds - targets).pow(2) * mask

        # reduce per item, then across batch
        B = preds.shape[0]
        squared_error_flat = squared_error.view(B, -1)
        mask_flat = mask.view(B, -1).float()

        denom = mask_flat.sum(dim=1).clamp_min(self.eps)
        per_item = squared_error_flat.sum(dim=1) / denom
        if self.reduction == "sum":
            return per_item.sum()
        return per_item.mean()
