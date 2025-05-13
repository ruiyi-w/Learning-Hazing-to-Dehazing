from typing import Optional, Tuple, Dict, Literal, List

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.cldm import ControlLDM
from ..model.gaussian_diffusion import Diffusion
from ..utils.cond_fn import Guidance
from ..utils.common import make_tiled_fn, trace_vram_usage
from utils.align_utils import tiled_align
from utils.dcp_utils import compute_transmission


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "SpacedSampler":
        super().__init__(betas, parameterization, rescale_cfg)

    def make_schedule(self, num_steps: int) -> None:
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.training_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.timesteps = np.array(
            sorted(list(used_timesteps)), dtype=np.int32
        )  # e.g. [0, 10, 20, ...]

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", np.sqrt(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def apply_cond_fn(
        self,
        model: ControlLDM,
        pred_x0: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        cond_fn: Guidance
    ) -> torch.Tensor:
        t_now = int(t[0].item()) + 1
        if not (cond_fn.t_stop < t_now and t_now < cond_fn.t_start):
            # stop guidance
            return pred_x0
        grad_rescale = 1 / extract_into_tensor(self.posterior_mean_coef1, index, pred_x0.shape)
        # apply guidance for multiple times
        loss_vals = []
        for _ in range(cond_fn.repeat):
            # set target and pred for gradient computation
            target, pred = None, None
            if cond_fn.space == "latent":
                target = model.vae_encode(cond_fn.target)
                pred = pred_x0
            elif cond_fn.space == "rgb":
                # We need to backward gradient to x0 in latent space, so it's required
                # to trace the computation graph while decoding the latent.
                with torch.enable_grad():
                    target = cond_fn.target
                    pred_x0_rg = pred_x0.detach().clone().requires_grad_(True)
                    pred = model.vae_decode(pred_x0_rg)
                    assert pred.requires_grad
            else:
                raise NotImplementedError(cond_fn.space)
            # compute gradient
            delta_pred, loss_val = cond_fn(target, pred, t_now)
            loss_vals.append(loss_val)
            # update pred_x0 w.r.t gradient
            if cond_fn.space == "latent":
                delta_pred_x0 = delta_pred
                pred_x0 = pred_x0 + delta_pred_x0 * grad_rescale
            elif cond_fn.space == "rgb":
                pred.backward(delta_pred)
                delta_pred_x0 = pred_x0_rg.grad
                pred_x0 = pred_x0 + delta_pred_x0 * grad_rescale
            else:
                raise NotImplementedError(cond_fn.space)
        return pred_x0

    def apply_model(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            model_cond = model(x, model_t, cond)
            model_uncond = model(x, model_t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        cond_fn: Guidance = None,
        return_pred: bool = False,
    ) -> torch.Tensor:
        # predict x_0
        model_output = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)
        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)

        if cond_fn:
            pred_x0 = self.apply_cond_fn(model, pred_x0, model_t, t, cond_fn)
        # calculate mean and variance of next state
        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
        # sample next state
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise

        if return_pred:
            return pred_x0
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if tiled:
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
                    forward(
                        x_tile,
                        t,
                        {
                            "c_txt": cond["c_txt"],
                            "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],
                        },
                    )
                ),
                tile_size,
                tile_stride,
            )
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x = self.p_sample(
                model,
                x,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
            )

        if tiled:
            model.forward = forward
        return x

    @torch.no_grad()
    def accsamp(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        cond_fn: Guidance,
        hazy: torch.Tensor,
        diffusion: Diffusion,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        progress_leave: bool = True,
        proportions: List[float] = None,
    ) -> torch.Tensor:
        if proportions is None:
            proportions = [0.8, 0.6]

        self.make_schedule(steps)
        self.to(device)
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, leave=progress_leave, disable=not progress)
        bs = x_size[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            if (i + 1) / len(iterator) < round(1.0 - proportions[0], 2):
                x = self.p_sample(
                    model, x, model_t, t, cond, uncond, cur_cfg_scale, cond_fn=None, return_pred=False
                )
            elif (i + 1) / len(iterator) == round(1.0 - proportions[0], 2):
                pred_x0 = self.p_sample(
                    model, x, model_t, t, cond, uncond, cur_cfg_scale, cond_fn=None, return_pred=True
                )
                pred_x0 = model.vae_decode(pred_x0)
                estimate = tiled_align(hazy * 2. - 1., pred_x0)
                transmission = compute_transmission(hazy * 2. - 1.)
                cond_fn.load_target(estimate)
                cond_fn.load_transmission(transmission)
                estimate = model.vae_encode(estimate, sample=False)
            else:
                pass

        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, leave=progress_leave, disable=not progress)
        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            if i / len(iterator) == round(1.0 - proportions[1], 2):
                x = diffusion.q_sample(x_start=estimate, t=model_t, noise=torch.randn_like(estimate))
                x = self.p_sample(
                    model, x, model_t, t, cond, uncond, cur_cfg_scale, cond_fn=cond_fn,
                )
            elif i / len(iterator) > round(1.0 - proportions[1], 2):
                x = self.p_sample(
                    model, x, model_t, t, cond, uncond, cur_cfg_scale, cond_fn=cond_fn,
                )
            else:
                pass

        return x
