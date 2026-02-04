import torch
import torch.nn as nn
import numpy as np


def predict_sliding_window(
        model: nn.Module,
        image: np.ndarray,
        patch_size: int = 128,
        stride: int = 128,
) -> np.ndarray:
    """
    Perform sliding window prediction on a large image using the provided model.

    Args:
        model (nn.Module): Trained PyTorch model for prediction.
        image (np.ndarray): Input image array of shape (H, W, C).
        patch_size (int): Size of the square patches to extract.
        stride (int): Stride for sliding window.

    Returns:
        np.ndarray: Predicted image of shape (H, W).
    """


    device = next(model.parameters()).device
    model.eval()

    H, W, C = image.shape

    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_wsum = np.zeros((H, W), dtype=np.float32)

    # 2D Hann window (avoids seams)
    w1 = np.hanning(patch_size).astype(np.float32)
    w2 = np.outer(w1, w1)
    w2 = w2 / (w2.max() + 1e-8)

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):

                patch = image[y:y+patch_size, x:x+patch_size]          # (128,128,C)
                patch = torch.from_numpy(patch).float()
                patch = patch.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,128,128)

                pred = model(patch)                      # (1,1,128,128) expected
                pred = pred.squeeze(0).squeeze(0).cpu().numpy()          # (128,128)

                pred_sum[y:y+patch_size, x:x+patch_size]  += pred * w2
                pred_wsum[y:y+patch_size, x:x+patch_size] += w2

    pred_mask = np.zeros_like(pred_sum)
    valid = pred_wsum > 0
    pred_mask[valid] = pred_sum[valid] / pred_wsum[valid]

    return pred_mask
