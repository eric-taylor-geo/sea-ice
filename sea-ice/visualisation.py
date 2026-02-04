import matplotlib.pyplot as plt
from matplotlib import gridspec


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
    sm = plt.cm.ScalarMappable(
        cmap="Blues", norm=plt.Normalize(vmin=0, vmax=50)
    )
    sm._A = []  # dummy array for the scalar mappable
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Sea Ice Concentration (%)")

    plt.show()
