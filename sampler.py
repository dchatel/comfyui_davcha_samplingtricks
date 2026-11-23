import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange
import comfy.utils
import comfy.model_sampling

from .noise import create_radial_filter

def sampler_function(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    samplers = extra_options.pop('samplers')
    splits = extra_options.pop('splits')
    splits_l = np.array([0] + splits + [len(sigmas)])
    sigmas_l = [sigmas[start:end] for start, end in list(zip(splits_l, splits_l[1:]+1))]
    for sampler, sigmas in zip(samplers, sigmas_l):
        extras = dict(**extra_options, **sampler.extra_options)
        noise = sampler.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extras)
    return noise


def _to_4d(x: torch.Tensor):
    if x.ndim == 4:
        return x, None
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, c, t)
    raise ValueError(f"[DavchaFourierUpscale] Shape non supportée: {x.shape}")


def _from_4d(x: torch.Tensor, meta):
    if meta is None:
        return x
    b, c, t = meta
    _, _, h, w = x.shape
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def _tukey_1d(n: int, alpha: float, device, dtype):
    if alpha <= 0.0 or n < 4:
        return torch.ones(n, device=device, dtype=dtype)
    pos = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    w = torch.ones(n, device=device, dtype=dtype)
    half = alpha * 0.5
    left = pos < half
    right = pos > (1.0 - half)
    w[left] = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * pos[left] / alpha - 1.0)))
    w[right] = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * pos[right] / alpha - 2.0 / alpha + 1.0)))
    return w


def _fourier_upscale_signal(x4d: torch.Tensor, h2: int, w2: int, tukey_alpha: float) -> torch.Tensor:
    _, _, h, w = x4d.shape
    if h == h2 and w == w2:
        return x4d

    orig_dtype = x4d.dtype
    xf = x4d.to(torch.float32)

    pad_h = min(16, max(2, h // 4), h - 1) if h > 1 else 0
    pad_w = min(16, max(2, w // 4), w - 1) if w > 1 else 0
    pad_h2 = int(round(pad_h * (h2 / h)))
    pad_w2 = int(round(pad_w * (w2 / w)))

    xp = F.pad(xf, (pad_w, pad_w, pad_h, pad_h), mode="reflect") if (pad_h > 0 or pad_w > 0) else xf
    _, _, hp, wp = xp.shape
    hp2, wp2 = h2 + 2 * pad_h2, w2 + 2 * pad_w2

    fft_x = torch.fft.fftshift(torch.fft.fft2(xp, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    wh = _tukey_1d(hp, tukey_alpha, xf.device, torch.float32)[:, None]
    ww = _tukey_1d(wp, tukey_alpha, xf.device, torch.float32)[None, :]
    fft_x = fft_x * (wh * ww)

    fft_pad = torch.zeros((*xp.shape[:-2], hp2, wp2), dtype=fft_x.dtype, device=xf.device)
    h0 = hp2 // 2 - hp // 2
    w0 = wp2 // 2 - wp // 2
    fft_pad[..., h0:h0 + hp, w0:w0 + wp] = fft_x

    if hp % 2 == 0 and h0 + hp < hp2:
        fft_pad[..., h0, w0:w0 + wp] *= 0.5
        fft_pad[..., h0 + hp, w0:w0 + wp] = fft_pad[..., h0, w0:w0 + wp]
    if wp % 2 == 0 and w0 + wp < wp2:
        end_h = h0 + hp + (1 if (hp % 2 == 0 and h0 + hp < hp2) else 0)
        fft_pad[..., h0:end_h, w0] *= 0.5
        fft_pad[..., h0:end_h, w0 + wp] = fft_pad[..., h0:end_h, w0]

    scale_amp = math.sqrt((hp2 * wp2) / (hp * wp))
    up = torch.fft.ifft2(torch.fft.ifftshift(fft_pad, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real * scale_amp

    out = up[..., pad_h2:pad_h2 + h2, pad_w2:pad_w2 + w2]
    return out.to(orig_dtype)


def _upscale_denoised_4d(d4d: torch.Tensor, h2: int, w2: int, method: str, patch_unpack_2x2: bool, tukey_alpha: float) -> torch.Tensor:
    use_unpack = patch_unpack_2x2 and (d4d.shape[1] % 4 == 0)
    work = F.pixel_shuffle(d4d, 2) if use_unpack else d4d
    th = h2 * 2 if use_unpack else h2
    tw = w2 * 2 if use_unpack else w2

    if method == "fourier_tukey":
        up = _fourier_upscale_signal(work, th, tw, tukey_alpha)
    elif method == "bislerp":
        up = comfy.utils.bislerp(work, tw, th)
    else:
        up = comfy.utils.common_upscale(work, tw, th, method, "disabled")

    return F.pixel_unshuffle(up, 2) if use_unpack else up


def _generate_target_noise(reference_tensor: torch.Tensor, h2: int, w2: int, high_freq_noise) -> torch.Tensor:
    target_shape = (*reference_tensor.shape[:-2], h2, w2)
    if high_freq_noise is None:
        return torch.randn(target_shape, device=reference_tensor.device, dtype=reference_tensor.dtype)
    dummy = torch.empty(target_shape, device=reference_tensor.device, dtype=reference_tensor.dtype)
    n = high_freq_noise.generate_noise({"samples": dummy})
    return n.to(device=reference_tensor.device, dtype=reference_tensor.dtype)


def _fourier_mix_noise_4d(
    eps4d: torch.Tensor,
    fresh4d: torch.Tensor,
    anchor_spectrum: torch.Tensor,
    noise_anchor: float,
) -> torch.Tensor:
    _, _, h, w = eps4d.shape
    _, _, h2, w2 = fresh4d.shape
    dtype = eps4d.dtype

    if noise_anchor <= 0.0 or h2 < h or w2 < w:
        return fresh4d.to(dtype)

    # Réutilisation directe de ton create_radial_filter sur la bande (H, W)
    filter_2d = create_radial_filter(anchor_spectrum, h, w, eps4d.device)
    mask = torch.clamp(filter_2d * noise_anchor, 0.0, 1.0)
    mask = mask.view(*([1] * (eps4d.ndim - 2)), h, w)

    if h == h2 and w == w2:
        fft_e = torch.fft.fftshift(torch.fft.fft2(eps4d.to(torch.float32), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        fft_f = torch.fft.fftshift(torch.fft.fft2(fresh4d.to(torch.float32), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        blended = torch.sqrt(mask) * fft_e + torch.sqrt(1.0 - mask) * fft_f
        out = torch.fft.ifft2(torch.fft.ifftshift(blended, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
        return out.to(dtype)

    fft_e = torch.fft.fftshift(torch.fft.fft2(eps4d.to(torch.float32), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    fft_f = torch.fft.fftshift(torch.fft.fft2(fresh4d.to(torch.float32), dim=(-2, -1), norm="ortho"), dim=(-2, -1))

    h0 = h2 // 2 - h // 2
    w0 = w2 // 2 - w // 2

    fft_mix = fft_f.clone()
    center_f = fft_f[..., h0:h0 + h, w0:w0 + w]
    center_mix = torch.sqrt(mask) * fft_e + torch.sqrt(1.0 - mask) * center_f
    fft_mix[..., h0:h0 + h, w0:w0 + w] = center_mix

    # Symétrie hermitienne sur les bins de Nyquist (si pairs)
    if h % 2 == 0 and h0 + h < h2:
        nyq_h = center_mix[..., 0, :]
        fft_mix[..., h0, w0:w0 + w] = 0.5 * nyq_h + 0.5 * fft_f[..., h0, w0:w0 + w]
        fft_mix[..., h0 + h, w0:w0 + w] = fft_mix[..., h0, w0:w0 + w]
    if w % 2 == 0 and w0 + w < w2:
        nyq_w = fft_mix[..., h0:h0 + h, w0]
        fft_mix[..., h0:h0 + h, w0] = 0.5 * nyq_w + 0.5 * fft_f[..., h0:h0 + h, w0]
        fft_mix[..., h0:h0 + h, w0 + w] = fft_mix[..., h0:h0 + h, w0]

    out = torch.fft.ifft2(torch.fft.ifftshift(fft_mix, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
    std = out.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    out = out / std
    return out.to(dtype)


def _is_flow_matching(model_k, sigmas) -> bool:
    ms = getattr(getattr(model_k, "model_patcher", None), "model", None)
    ms = getattr(ms, "model_sampling", None)
    if ms is None:
        ms = getattr(getattr(model_k, "inner_model", None), "model_sampling", None)
    if ms is not None:
        return isinstance(ms, comfy.model_sampling.CONST)
    return float(sigmas.max().item()) <= 1.0001


def _upscale_step_tensors(
    denoised_t: torch.Tensor,
    eps_t: torch.Tensor,
    target_scale: float,
    denoised_upscale: str,
    patch_unpack_2x2: bool,
    tukey_alpha: float,
    anchor_spectrum: torch.Tensor,
    noise_anchor: float,
    high_freq_noise,
):
    d4d, meta = _to_4d(denoised_t)
    e4d, _ = _to_4d(eps_t)
    _, _, h, w = d4d.shape

    h2 = max(2, (int(round(h * target_scale)) // 2) * 2)
    w2 = max(2, (int(round(w * target_scale)) // 2) * 2)

    if h2 != h or w2 != w:
        d_up = _upscale_denoised_4d(d4d, h2, w2, denoised_upscale, patch_unpack_2x2, tukey_alpha)
        fresh4d = _generate_target_noise(e4d, h2, w2, high_freq_noise)
        e_up = _fourier_mix_noise_4d(e4d, fresh4d, anchor_spectrum, noise_anchor)
        return _from_4d(d_up, meta), _from_4d(e_up, meta)

    if noise_anchor < 1.0 or (anchor_spectrum < 1.0).any():
        fresh4d = _generate_target_noise(e4d, h, w, high_freq_noise)
        e_mix = _fourier_mix_noise_4d(e4d, fresh4d, anchor_spectrum, noise_anchor)
        return denoised_t, _from_4d(e_mix, meta)

    return denoised_t, eps_t


@torch.no_grad()
def fourier_upscale_sampler_function(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    scale_ratio = extra_options.pop("scale_ratio", 1.26)
    scale_steps = extra_options.pop("scale_steps", -1)
    denoised_upscale = extra_options.pop("denoised_upscale", "bislerp")
    patch_unpack_2x2 = extra_options.pop("patch_unpack_2x2", True)
    noise_anchor = extra_options.pop("noise_anchor", 1.0)
    anchor_spectrum = extra_options.pop("anchor_spectrum", torch.tensor([1.0], dtype=torch.float32))
    tukey_alpha = extra_options.pop("tukey_alpha", 0.25)
    high_freq_noise = extra_options.pop("high_freq_noise", None)

    extra_args = {} if extra_args is None else extra_args
    x = noise
    s_in = x.new_ones([x.shape[0]])

    if scale_steps < 0:
        scale_steps = len(sigmas) - 1

    flow_mode = _is_flow_matching(model_k, sigmas)

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model_k(x, sigma_cur * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_cur, "sigma_hat": sigma_cur, "denoised": denoised})

        if sigma_next == 0:
            x = denoised
            continue

        if float(sigma_cur) > 1e-6:
            eps_exact = (x - (1.0 - sigma_cur) * denoised) / sigma_cur if flow_mode else (x - denoised) / sigma_cur
        else:
            eps_exact = torch.randn_like(x)

        do_scale = (i < scale_steps) and (abs(scale_ratio - 1.0) > 1e-4)
        target_scale = (
            1.0 + (scale_ratio - 1.0) * ((i + 1) / max(1, scale_steps))
            if (do_scale and scale_steps > 1)
            else (scale_ratio if do_scale else 1.0)
        )

        # NestedTensor-safe (parité avec MaskedNoise / LatentNoise)
        if getattr(x, "is_nested", False):
            d_list, e_list = [], []
            for di, ei in zip(denoised.unbind(), eps_exact.unbind()):
                # unbind sur batch -> ajoute dim batch temporaire si 3D/4D
                di_b = di.unsqueeze(0)
                ei_b = ei.unsqueeze(0)
                du, eu = _upscale_step_tensors(
                    di_b, ei_b, target_scale, denoised_upscale, patch_unpack_2x2,
                    tukey_alpha, anchor_spectrum, noise_anchor, high_freq_noise
                )
                d_list.append(du.squeeze(0))
                e_list.append(eu.squeeze(0))
            denoised = torch.nested.nested_tensor(d_list, dtype=x.dtype, device=x.device)
            eps_mix = torch.nested.nested_tensor(e_list, dtype=x.dtype, device=x.device)
        else:
            denoised, eps_mix = _upscale_step_tensors(
                denoised, eps_exact, target_scale, denoised_upscale, patch_unpack_2x2,
                tukey_alpha, anchor_spectrum, noise_anchor, high_freq_noise
            )

        x = ((1.0 - sigma_next) * denoised + sigma_next * eps_mix) if flow_mode else (denoised + sigma_next * eps_mix)

    return x