from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from seaice.data.contrastive_loader import create_contrastive_data_loader
from seaice.models.contrastive import (
    ContrastiveEncoder,
    ContrastiveSegmentationModel,
    freeze_encoder,
)
from seaice.models.unet import UNet
from seaice.training.loss import MaskedMSELoss


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


def _to_input_tensor(inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(inputs, dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D input tensor. Got shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] in (1, 2, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.shape[1] not in (1, 2, 3, 4):
        raise ValueError(
            "Could not infer channel axis. Expected NCHW or NHWC with <=4 channels."
        )
    return tensor


def _to_target_tensor(targets: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(targets, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.ndim != 4:
        raise ValueError(
            "Expected target shape (N,H,W) or (N,H,W,1)/(N,1,H,W). "
            f"Got {tuple(tensor.shape)}."
        )
    return tensor


def _build_supervised_loader(
    inputs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    x_tensor = _to_input_tensor(inputs)
    y_tensor = _to_target_tensor(targets)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _sample_indices(num_samples: int, fraction: float, seed: int) -> np.ndarray:
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1].")
    keep = max(1, int(num_samples * fraction))
    rng = np.random.default_rng(seed)
    return rng.permutation(num_samples)[:keep]


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    if z_i.shape != z_j.shape:
        raise ValueError(f"z_i and z_j must match. Got {z_i.shape} vs {z_j.shape}.")
    if z_i.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(z_i.shape)}.")

    batch_size = z_i.shape[0]
    embeddings = torch.cat([z_i, z_j], dim=0)
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    eye = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=sim.device),
            torch.arange(0, batch_size, device=sim.device),
        ],
        dim=0,
    )
    return F.cross_entropy(sim, positives)


def pretrain_contrastive_encoder(
    encoder: ContrastiveEncoder,
    contrastive_loader: DataLoader,
    *,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.2,
    device: Optional[torch.device | str] = None,
) -> list[float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    encoder = encoder.to(device)
    optimiser = torch.optim.AdamW(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: list[float] = []
    for _ in range(num_epochs):
        encoder.train()
        running_loss = 0.0
        seen = 0

        for x_i, x_j in contrastive_loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            optimiser.zero_grad()
            z_i = encoder(x_i)
            z_j = encoder(x_j)
            loss = nt_xent_loss(z_i, z_j, temperature=temperature)
            loss.backward()
            optimiser.step()
            batch_size = x_i.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        epoch_loss = running_loss / max(1, seen)
        history.append(epoch_loss)

    return history


def save_contrastive_encoder_weights(
    encoder: ContrastiveEncoder,
    *,
    path: str = "seaice_weights/contrastive_encoder.pth",
) -> None:
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), path)


def train_contrastive_encoder_and_save(
    X_train: np.ndarray | torch.Tensor,
    *,
    save_path: str = "seaice_weights/contrastive_encoder.pth",
    in_channels: int = 2,
    batch_size: int = 64,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.2,
    num_workers: int = 0,
    device: Optional[torch.device | str] = None,
) -> tuple[ContrastiveEncoder, list[float]]:
    contrastive_loader = create_contrastive_data_loader(
        X_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    encoder = ContrastiveEncoder(in_channels=in_channels)
    history = pretrain_contrastive_encoder(
        encoder,
        contrastive_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        device=device,
    )
    save_contrastive_encoder_weights(encoder, path=save_path)
    return encoder, history


def _train_supervised_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimiser.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(data_loader.dataset)


def _eval_supervised(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(data_loader.dataset)


def train_decoder_with_partial_labels(
    model: ContrastiveSegmentationModel,
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    X_val: np.ndarray | torch.Tensor,
    y_val: np.ndarray | torch.Tensor,
    *,
    label_fraction: float = 0.5,
    freeze_pretrained_encoder: bool = True,
    num_epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 0,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[torch.device | str] = None,
) -> tuple[TrainingHistory, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if freeze_pretrained_encoder:
        freeze_encoder(model)

    indices = _sample_indices(len(X_train), fraction=label_fraction, seed=seed)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    train_loader = _build_supervised_loader(
        X_subset,
        y_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = _build_supervised_loader(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(device)
    criterion = criterion or MaskedMSELoss()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = TrainingHistory(train_loss=[], val_loss=[])
    for _ in range(num_epochs):
        train_loss = _train_supervised_epoch(
            model,
            train_loader,
            optimiser,
            criterion,
            device,
        )
        val_loss = _eval_supervised(model, val_loader, criterion, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

    return history, indices


def train_unet_full_supervision(
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    X_val: np.ndarray | torch.Tensor,
    y_val: np.ndarray | torch.Tensor,
    *,
    num_epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[torch.device | str] = None,
) -> tuple[UNet, TrainingHistory]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    train_loader = _build_supervised_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = _build_supervised_loader(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = UNet(in_channels=2, out_channels=1).to(device)
    criterion = criterion or MaskedMSELoss()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = TrainingHistory(train_loss=[], val_loss=[])
    for _ in range(num_epochs):
        train_loss = _train_supervised_epoch(
            model,
            train_loader,
            optimiser,
            criterion,
            device,
        )
        val_loss = _eval_supervised(model, val_loader, criterion, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

    return model, history
