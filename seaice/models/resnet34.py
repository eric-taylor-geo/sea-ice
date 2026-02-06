import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ResNet34(nn.Module):
    """
    Input:  (B, 2, H, W)
    Output: (B, 1, H, W)
    """
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)  # (B,1,H,W)
        x = self.out_act(x)
        return x

# Helper functions


def freeze_encoder(model: ResNet34) -> None:
    for p in model.net.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model: ResNet34) -> None:
    for p in model.net.encoder.parameters():
        p.requires_grad = True


def make_optimizer(model: ResNet34, lr_decoder: float = 1e-3, lr_encoder: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.net.decoder.parameters(), "lr": lr_decoder},
            {"params": model.net.segmentation_head.parameters(), "lr": lr_decoder},
            {"params": model.net.encoder.parameters(), "lr": lr_encoder},
        ],
        weight_decay=1e-4,
    )
