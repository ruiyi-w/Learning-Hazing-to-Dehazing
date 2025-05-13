import torch
import torch.nn.functional as F


def compute_transmission(
        hazy_image: torch.Tensor,
        patch_size: int = 15,
        omega: float = 0.95,
        t_min: float = 0.05,
        topk_ratio: float = 0.001,
):
    """
    Estimate transmission map with the Dark Channel Prior (DCP).

    Args
    ----
    hazy_image     : [B, C, H, W] tensor in range [-1, 1]
    patch_size     : local window size k (odd int, default 15)
    omega          : DCP parameter (default 0.95)
    t_min          : lower clamp for transmission (default 0.05)
    topk_ratio     : % of brightest dark-channel pixels for A (default 0.1 %)

    Returns
    -------
    transmission   : [B, 1, H, W] tensor in [t_min, 1]
    """

    B, C, H, W = hazy_image.shape
    if patch_size % 2 == 0:
        patch_size += 1
    pad = patch_size // 2

    I = (hazy_image + 1.0) * 0.5
    min_I = I.min(dim=1, keepdim=True).values
    dark_I = -F.max_pool2d(-min_I, patch_size, 1, pad)

    N = H * W
    k = max(1, int(N * topk_ratio))
    dark_flat = dark_I.view(B, -1)

    _, idx = torch.topk(dark_flat, k, dim=1, largest=True, sorted=False)
    I_flat = I.view(B, C, -1)
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1)
    candidates = torch.gather(I_flat, 2, idx_exp)

    sums = candidates.sum(dim=1)
    best = sums.argmax(dim=1, keepdim=True)
    best_idx = best.unsqueeze(1).expand(-1, C, -1)
    A = candidates.gather(2, best_idx).squeeze(2)

    I_norm = I / A.view(B, C, 1, 1)
    min_I_norm = I_norm.min(dim=1, keepdim=True).values
    dark_norm = -F.max_pool2d(-min_I_norm, patch_size, 1, pad)

    t = 1.0 - omega * dark_norm
    t_clamped = t.clamp_(min=t_min, max=1.0)  # [B, 1, H, W]

    return t_clamped
