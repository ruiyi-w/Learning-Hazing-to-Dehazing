import torch


def tiled_align(
        hazy_image: torch.Tensor,
        pred_image: torch.Tensor,
        kernel_size: int = 37,
        stride: int = 10,
        *,
        eps: float = 1e-6,
        low_memory: bool = False,
        unfold_threshold: int = 200_000_000
) -> torch.Tensor:
    """
    Patch-wise alignment from Eq.(6) of the paper.

    Args
    ----
    hazy_image, pred_image : [B, C, H, W] tensors
    kernel_size           : k  (patch size)
    stride                : d  (sliding-window stride)
    eps                   : numerical stabiliser for std
    low_memory           : switch to low-memory routine
    unfold_threshold      : max elements we’re willing to unfold

    Returns
    -------
    aligned_image         : [B, C, H, W] tensor
    """

    if hazy_image.shape != pred_image.shape:
        raise ValueError("Both images must share shape [B, C, H, W].")
    B, C, H, W = hazy_image.shape
    k, d = kernel_size, stride

    n_h = (H - k) // d + 1
    n_w = (W - k) // d + 1
    num_patches = n_h * n_w

    if low_memory and B * C * k * k * num_patches > unfold_threshold:
        out = torch.zeros_like(hazy_image)
        w_map = torch.zeros_like(hazy_image[:, :1, ...])

        for ih in range(n_h):
            top = ih * d
            for iw in range(n_w):
                left = iw * d

                hx = hazy_image[:, :, top:top + k, left:left + k]
                pr = pred_image[:, :, top:top + k, left:left + k]

                mu_x = hx.mean(dim=(2, 3), keepdim=True)
                mu_r = pr.mean(dim=(2, 3), keepdim=True)
                std_x = hx.var(dim=(2, 3), unbiased=False,
                               keepdim=True).add(eps).sqrt()
                std_r = pr.var(dim=(2, 3), unbiased=False,
                               keepdim=True).add(eps).sqrt()

                aligned = (hx - mu_x) / std_x * std_r + mu_r

                out[:, :, top:top + k, left:left + k] += aligned
                w_map[:, :, top:top + k, left:left + k] += 1

        return out / w_map.clamp_min(eps)


    unfold = torch.nn.Unfold(kernel_size=k, stride=d)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=k, stride=d)

    hx = unfold(hazy_image)
    pr = unfold(pred_image)
    L = hx.shape[-1]

    hx = hx.view(B, C, k * k, L)
    pr = pr.view(B, C, k * k, L)

    mu_x = hx.mean(dim=2, keepdim=True)
    mu_r = pr.mean(dim=2, keepdim=True)
    std_x = hx.var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt()
    std_r = pr.var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt()

    aligned = (hx - mu_x) / std_x * std_r + mu_r  # [B,C,k²,L]
    aligned = aligned.view(B, C * k * k, L)

    out = fold(aligned)
    w_map = fold(torch.ones_like(aligned))
    return out / w_map.clamp_min(eps)
