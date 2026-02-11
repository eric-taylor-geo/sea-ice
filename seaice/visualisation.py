from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm


def plot_patches(X, Y, dim=10):
    """Plot a grid of patches: left side X, right side Y

    Args:
        X: np.ndarray of shape (P, H, W, C)
        Y: np.ndarray of shape (P, H, W)
        dim: int, number of patches per row/column (total patches = dim^2)

    Returns:
        None
    """

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

    # Left: X
    gs_left = gridspec.GridSpecFromSubplotSpec(
        dim, dim, subplot_spec=gs[0], wspace=0.02, hspace=0.02
    )

    for i in range(dim**2):
        ax = fig.add_subplot(gs_left[i])
        ax.imshow(X[i][:, :, 0], cmap="RdBu", vmin=-2, vmax=2)
        ax.axis("off")

    # Right: Y
    gs_right = gridspec.GridSpecFromSubplotSpec(
        dim, dim, subplot_spec=gs[1], wspace=0.02, hspace=0.02
    )

    for i in range(dim**2):
        ax = fig.add_subplot(gs_right[i])
        ax.imshow(Y[i], cmap="Blues", vmin=0, vmax=50)
        ax.axis("off")

    # Add colorbar for Y
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=50))
    sm._A = []  # dummy array for the scalar mappable
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Sea Ice Concentration (%)")

    plt.show()


def plot_prediction(
    X: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    figsize: Tuple = (12, 10),
    dpi: int = 300,
    patch_size: int = 128,
    mask_land: bool = True,
):
    """Plot input X, ground truth Y_true, and prediction Y_pred side by side.

    Args:
        X: np.ndarray of shape (H, W, 2) loaded from data.load_x_y
        Y_true: np.ndarray of shape (H, W) loaded from data.load_x_y, expected to be in 0-10 range
        Y_pred: np.ndarray of shape (H, W) predicted by model, expected to be in 0-1 range

    Returns:
        None
    """

    # scale Y to 0-100 for visualization
    Y_true = Y_true * 10  # ground truth is 0-10
    Y_pred = Y_pred * 100  # model outputs 0-1

    if mask_land:
        land_mask = Y_true > 100
        Y_true = np.where(land_mask, np.nan, Y_true)
        Y_pred = np.where(land_mask, np.nan, Y_pred)

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axs = axs.flatten()

    axs[0].imshow(X[:, :, 0], cmap="RdBu", vmin=-2, vmax=2)
    axs[0].add_patch(
        plt.Rectangle(
            (0, 0),
            patch_size,
            patch_size,
            fill=False,
            edgecolor="yellow",
            linewidth=1,
        )
    )
    axs[0].set_title("Input SAR Primary")

    axs[1].imshow(X[:, :, 1], cmap="RdBu", vmin=-2, vmax=2)
    axs[1].set_title("Input SAR Secondary")

    axs[2].imshow(Y_true, cmap="Blues", vmin=0, vmax=100)
    axs[2].set_title("Ground Truth Sea Ice Concentration")

    axs[3].imshow(Y_pred, cmap="Blues", vmin=0, vmax=100)
    axs[3].set_title("Predicted Sea Ice Concentration")

    [ax.axis("off") for ax in axs]

    # remove space between columns
    plt.subplots_adjust(wspace=0.02)

    # add colorbar for Y_true and Y_pred at the bottom
    bounds = np.arange(0, 110, 10)  # 10% bins from 0–100
    norm = BoundaryNorm(bounds, ncolors=plt.cm.Blues.N, clip=True)

    # add colorbar for Y_true and Y_pred at the bottom
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])  # dummy array

    cbar = plt.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
        boundaries=bounds,
        ticks=bounds,
        spacing="proportional",
    )

    cbar.set_label("Sea Ice Concentration (%)")

    plt.show()
