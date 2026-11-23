import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange
import comfy.utils
import comfy.model_sampling
import comfy.latent_formats
from comfy.nested_tensor import NestedTensor

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


def _fourier_mix_noise_4d(eps4d, fresh4d, anchor_spectrum, noise_anchor):
    if noise_anchor <= 0.0:
        return fresh4d

    _, _, h, w = eps4d.shape
    _, _, h2, w2 = fresh4d.shape

    mask = torch.clamp(create_radial_filter(anchor_spectrum, h, w, eps4d.device) * noise_anchor, 0.0, 1.0)

    fft_e = torch.fft.fftshift(torch.fft.fft2(eps4d.to(torch.float32), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    fft_f = torch.fft.fftshift(torch.fft.fft2(fresh4d.to(torch.float32), dim=(-2, -1), norm="ortho"), dim=(-2, -1))

    h0 = h2 // 2 - h // 2
    w0 = w2 // 2 - w // 2
    fresh_band = fft_f[..., h0:h0 + h, w0:w0 + w].clone()
    fft_f[..., h0:h0 + h, w0:w0 + w] = torch.sqrt(mask) * fft_e + torch.sqrt(1.0 - mask) * fresh_band

    # Nyquist bins of the original band: split between both ends, keeping unit variance
    half = math.sqrt(0.5)
    if h % 2 == 0 and h0 + h < h2:
        fft_f[..., h0, w0:w0 + w] = half * fft_f[..., h0, w0:w0 + w] + half * fresh_band[..., 0, :]
        fft_f[..., h0 + h, w0:w0 + w] = fft_f[..., h0, w0:w0 + w]
    if w % 2 == 0 and w0 + w < w2:
        fft_f[..., h0:h0 + h, w0] = half * fft_f[..., h0:h0 + h, w0] + half * fresh_band[..., :, 0]
        fft_f[..., h0:h0 + h, w0 + w] = fft_f[..., h0:h0 + h, w0]

    out = torch.fft.ifft2(torch.fft.ifftshift(fft_f, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
    return out.to(eps4d.dtype)


def fourier_upscale_sampler_function(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    scale_ratio = extra_options.pop("scale_ratio")
    scale_steps = extra_options.pop("scale_steps")
    denoised_upscale = extra_options.pop("denoised_upscale")
    noise_anchor = extra_options.pop("noise_anchor")
    anchor_spectrum = extra_options.pop("anchor_spectrum")
    tukey_alpha = extra_options.pop("tukey_alpha")
    high_freq_noise = extra_options.pop("high_freq_noise")

    extra_args = {} if extra_args is None else extra_args
    model_patcher = model_k.inner_model.model_patcher
    flow_mode = isinstance(model_patcher.get_model_object("model_sampling"), comfy.model_sampling.CONST)
    patch_unpack_2x2 = isinstance(model_patcher.get_model_object("latent_format"), (comfy.latent_formats.Flux2, comfy.latent_formats.Wan22))
    partial_anchor = noise_anchor < 1.0 or bool((anchor_spectrum < 1.0).any())
    generator = torch.Generator().manual_seed(extra_args.get("seed", 0))

    x = noise
    nested = isinstance(x, NestedTensor)
    h_orig, w_orig = (x.unbind()[0] if nested else x).shape[-2:]
    s_in = x.new_ones([x.shape[0]])

    n_active = int((sigmas[1:] > 0).sum())
    n_scale = n_active if scale_steps < 0 else min(scale_steps, n_active)

    def fresh_noise(shape, ref, step):
        if high_freq_noise is None:
            return torch.randn(shape, generator=generator).to(device=ref.device, dtype=ref.dtype)
        step_noise = copy.copy(high_freq_noise)
        step_noise.seed = high_freq_noise.seed + step
        return step_noise.generate_noise({"samples": torch.empty(shape)}).to(device=ref.device, dtype=ref.dtype)

    def step_tensor(x_t, denoised_t, sigma, sigma_next, size, step):
        if flow_mode:
            eps_t = (x_t - (1.0 - sigma) * denoised_t) / sigma
        else:
            eps_t = (x_t - denoised_t) / sigma

        resize = size is not None and size != tuple(denoised_t.shape[-2:])
        if resize or (size is not None and partial_anchor):
            d4d, meta = _to_4d(denoised_t)
            e4d, _ = _to_4d(eps_t)
            if resize:
                d4d = _upscale_denoised_4d(d4d, *size, denoised_upscale, patch_unpack_2x2, tukey_alpha)
            fresh4d, _ = _to_4d(fresh_noise((*denoised_t.shape[:-2], *size), denoised_t, step))
            e4d = _fourier_mix_noise_4d(e4d, fresh4d, anchor_spectrum, noise_anchor)
            denoised_t, eps_t = _from_4d(d4d, meta), _from_4d(e4d, meta)

        if flow_mode:
            return (1.0 - sigma_next) * denoised_t + sigma_next * eps_t
        return denoised_t + sigma_next * eps_t

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model_k(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        if sigma_next == 0:
            x = denoised
            continue

        x_main = x.unbind()[0] if nested else x
        if i < n_scale:
            factor = 1.0 + (scale_ratio - 1.0) * (i + 1) / n_scale
            size = (max(2, int(round(h_orig * factor)) // 2 * 2), max(2, int(round(w_orig * factor)) // 2 * 2))
        else:
            size = tuple(x_main.shape[-2:])

        if nested:
            # Only the first (video) latent is upscaled, the others (e.g. LTXAV audio) are stepped as is
            xs, ds = x.unbind(), denoised.unbind()
            x = NestedTensor([step_tensor(xs[0], ds[0], sigma, sigma_next, size, i)] +
                             [step_tensor(xt, dt, sigma, sigma_next, None, i) for xt, dt in zip(xs[1:], ds[1:])])
        else:
            x = step_tensor(x, denoised, sigma, sigma_next, size, i)

    return x
