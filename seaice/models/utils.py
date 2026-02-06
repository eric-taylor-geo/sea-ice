import os

import torch


def get_model(name, load_weights=False, path=None):

    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )

    if name == "unet":

        from .unet import UNet

        model = UNet(in_channels=2, out_channels=1).to(device)

    elif name == "resnet34":

        from .resnet34 import ResNet34

        model = ResNet34(in_channels=2).to(device)

    else:
        raise ValueError(f"Unknown model name: {name}. Supported: unet, resnet34")

    if load_weights:

        if path is None:
            # use default path if not provided
            path = f"seaice_weights/{name}.pth"

        # check if weights file exists:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")

        state_dict = torch.load(
            path, map_location=device
        )
        model.load_state_dict(state_dict)

        print(f"Loaded weights for {name} from {path}")

    return model
