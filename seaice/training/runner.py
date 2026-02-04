import torch
import torch.nn as nn
from livelossplot import PlotLosses

from .loops import train_one_epoch, test_one_epoch


def run_epochs(
    model,
    optimiser,
    criterion,
    train_loader,
    test_loader,
    device,
    num_epochs: int = 50,
):
    liveloss = PlotLosses()
    for i in range(num_epochs):
        train_loss = train_one_epoch(model, optimiser, criterion, train_loader, device)
        valid_loss = test_one_epoch(model, criterion, test_loader, device)

        # Liveloss plot
        logs = {}
        logs["" + "loss"] = train_loss
        logs["" + "val_loss"] = valid_loss
        liveloss.update(logs)
        liveloss.draw()