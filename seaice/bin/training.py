import torch
import torch.nn as nn
from livelossplot import PlotLosses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================
# Train Loops
# ================================================


def train(model, optimiser, criterion, data_loader):
    model.train()
    train_loss = 0.0

    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(input.permute(0, 3, 1, 2))  # change to (B, C, H, W)
        loss = criterion(output, target.permute(0, 3, 1, 2))  # change to (B, C, H, W)
        loss.backward()
        optimiser.step()
        train_loss += loss.item() * input.size(0)
    return train_loss / len(data_loader.dataset)


def test(model, criterion, data_loader):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            output = model(input.permute(0, 3, 1, 2))  # change to (B, C, H, W)
            loss = criterion(
                output, target.permute(0, 3, 1, 2)
            )  # change to (B, C, H, W)
            test_loss += loss.item() * input.size(0)
    return test_loss / len(data_loader.dataset)


# ================================================
# Loss function
# ================================================


class MaskedMSELoss(nn.Module):
    """
    Mask out invalid target pixels (e.g., land encoded as 25.5) per item in batch.

    Assumes:
      preds, targets shape: (B, C, H, W)  (works for any shape as long as broadcast matches)
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
                f"preds and targets must have same shape, got {preds.shape} vs {targets.shape}"
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


criterion = MaskedMSELoss()

# ================================================
# Epochs
# ================================================


def run_epochs(
    model,
    optimiser,
    train_loader,
    test_loader,
    num_epochs: int = 20,
):
    liveloss = PlotLosses()
    for i in range(num_epochs):
        train_loss = train(model, optimiser, criterion, train_loader)
        valid_loss = test(model, criterion, test_loader)

        # Liveloss plot
        logs = {}
        logs["" + "loss"] = train_loss
        logs["" + "val_loss"] = valid_loss
        liveloss.update(logs)
        liveloss.draw()
