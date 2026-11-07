import math
import torch
import comfy.utils
from comfy.nested_tensor import NestedTensor

def create_radial_filter(spectrum, H, W, device):
    # Frequencies laid out like fftshift(fft2(x)), so the filter is Hermitian-symmetric and DC sits at radius 0
    Y = torch.fft.fftshift(torch.fft.fftfreq(H, device=device)) * 2
    X = torch.fft.fftshift(torch.fft.fftfreq(W, device=device)) * 2
    y, x = torch.meshgrid(Y, X, indexing='ij')

    radius = torch.sqrt(x**2 + y**2) / math.sqrt(2)

    num_bins = len(spectrum)
    if num_bins == 1:
        return torch.full((H, W), spectrum[0], device=device)

    scaled_radius = radius * (num_bins - 1)

    lower_idx = scaled_radius.floor().long().clamp(0, num_bins - 1)
    upper_idx = scaled_radius.ceil().long().clamp(0, num_bins - 1)
    weight = scaled_radius - lower_idx

    # Smoothstep transition
    weight = weight * weight * (3.0 - 2.0 * weight)

    spectrum = spectrum.to(device)
    lower_val = spectrum[lower_idx]
    upper_val = spectrum[upper_idx]

    return torch.lerp(lower_val, upper_val, weight)

def normalize_noise(x):
    std = x.std()
    if std > 0:
        return (x - x.mean()) / std
    return x - x.mean()

def prepare_mask(mask, shape, device):
    # Resize a [H, W] or [frames, H, W] mask so it broadcasts against a [B, C, H, W] or [B, C, T, H, W] latent
    m = mask.to(device, dtype=torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    H, W = shape[-2:]
    if len(shape) == 5:
        # Mask frames follow the video frames, which map to the latent T dim
        m = torch.nn.functional.interpolate(m[None, None], size=(shape[2], H, W), mode='trilinear', align_corners=False)
    else:
        m = torch.nn.functional.interpolate(m.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)
        m = comfy.utils.repeat_to_batch_size(m, shape[0])
    return m.clamp(0.0, 1.0)

class SpectralNoise:
    def __init__(self, a, b, spectrum_tensor):
        self.a = a
        self.b = b
        self.spectrum = spectrum_tensor
        self.seed = a.seed

    def _process_single_tensor(self, x, y):
        device = x.device
        dtype = x.dtype
        *_, H, W = x.shape

        x_f = x.to(torch.float32)
        y_f = y.to(torch.float32)

        fft_x = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))
        fft_y = torch.fft.fftshift(torch.fft.fft2(y_f, dim=(-2, -1)), dim=(-2, -1))

        filter_2d = create_radial_filter(self.spectrum, H, W, device)
        view_shape = [1] * (x.ndim - 2) + [H, W]
        mask_nd = filter_2d.view(*view_shape)
        mask_nd = torch.clamp(mask_nd, 0.0, 1.0)

        # Energy-preserving blend
        blended_fft = (torch.sqrt(mask_nd) * fft_x) + (torch.sqrt(1.0 - mask_nd) * fft_y)

        shaped = torch.fft.ifft2(torch.fft.ifftshift(blended_fft, dim=(-2, -1)), dim=(-2, -1)).real
        return normalize_noise(shaped).to(dtype)

    def generate_noise(self, input_latent):
        x = self.a.generate_noise(input_latent)
        y = self.b.generate_noise(input_latent)

        if x.is_nested:
            return NestedTensor([self._process_single_tensor(xi, yi) for xi, yi in zip(x.unbind(), y.unbind())])
        return self._process_single_tensor(x, y)

class SpectralNoiseEQ:
    def __init__(self, base_noise, spectrum_tensor, normalize):
        self.base_noise = base_noise
        self.spectrum = spectrum_tensor
        self.normalize = normalize
        self.seed = base_noise.seed

    def _process_single_tensor(self, x):
        device = x.device
        dtype = x.dtype
        *_, H, W = x.shape

        x_f = x.to(torch.float32)

        fft_x = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))

        # EQ curve, not bounded to 0-1
        filter_2d = create_radial_filter(self.spectrum, H, W, device)
        view_shape = [1] * (x.ndim - 2) + [H, W]
        eq_fft = fft_x * filter_2d.view(*view_shape)

        shaped = torch.fft.ifft2(torch.fft.ifftshift(eq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        if self.normalize:
            shaped = normalize_noise(shaped)
        return shaped.to(dtype)

    def generate_noise(self, input_latent):
        x = self.base_noise.generate_noise(input_latent)

        if x.is_nested:
            return NestedTensor([self._process_single_tensor(xi) for xi in x.unbind()])
        return self._process_single_tensor(x)

class MaskedNoise:
    def __init__(self, base_noise, masked_noise, mask, normalize, linear_blend):
        self.base_noise = base_noise
        self.masked_noise = masked_noise
        self.mask = mask
        self.normalize = normalize
        self.linear_blend = linear_blend
        self.seed = base_noise.seed

    def _process_single_tensor(self, x, y):
        m = prepare_mask(self.mask, x.shape, x.device)

        x_f = x.to(torch.float32)
        y_f = y.to(torch.float32)

        if self.linear_blend:
            blended = (1.0 - m) * x_f + m * y_f
        else:
            # Energy-preserving blend, keeps the noise variance at the feathered edges of the mask
            blended = (torch.sqrt(1.0 - m) * x_f) + (torch.sqrt(m) * y_f)

        if self.normalize:
            blended = normalize_noise(blended)
        return blended.to(x.dtype)

    def generate_noise(self, input_latent):
        x = self.base_noise.generate_noise(input_latent)
        y = self.masked_noise.generate_noise(input_latent)

        if x.is_nested:
            # The mask is spatial: only the first (video) component is masked, the others keep the base noise
            xs = x.unbind()
            return NestedTensor([self._process_single_tensor(xs[0], y.unbind()[0])] + list(xs[1:]))
        return self._process_single_tensor(x, y)

class LatentNoise:
    def __init__(self, latent_tensor, normalize, mode, seed):
        self.latent_tensor = latent_tensor
        self.normalize = normalize
        self.mode = mode
        self.seed = seed

    def _process_single(self, target, src, generator):
        *_, tH, tW = target.shape
        *_, sH, sW = src.shape

        if (tH, tW) != (sH, sW):
            raise ValueError(
                f"[Davcha Latent Noise] Shape mismatch: the injected latent is {sW}x{sH} "
                f"but the generation latent is {tW}x{tH}. Pad/crop the source image in pixel space before generating the noise."
            )

        src = src.to(target.device, dtype=target.dtype)
        if self.mode == "full":
            out = src
        else:
            noise = torch.randn(src.shape, dtype=torch.float32, generator=generator, device="cpu").to(src.device)

            fft_src = torch.fft.fft2(src.to(torch.float32), dim=(-2, -1))
            fft_noise = torch.fft.fft2(noise, dim=(-2, -1))

            if self.mode == "phase":
                # Image phase (structure) + noise magnitude
                mag = torch.abs(fft_noise)
                phase = torch.angle(fft_src)
            else:
                # Image magnitude (frequency curve) + noise phase (scrambled structure)
                mag = torch.abs(fft_src)
                phase = torch.angle(fft_noise)

            out = torch.fft.ifft2(torch.polar(mag, phase), dim=(-2, -1)).real.to(src.dtype)

        if self.normalize:
            out = normalize_noise(out)

        return comfy.utils.repeat_to_batch_size(out, target.shape[0])

    def generate_noise(self, input_latent):
        target = input_latent["samples"]
        generator = torch.manual_seed(self.seed)
        srcs = self.latent_tensor.unbind() if self.latent_tensor.is_nested else [self.latent_tensor]

        if target.is_nested:
            # Components without a matching source latent (e.g. the audio part of LTXAV) get plain seeded noise
            out = []
            for i, t in enumerate(target.unbind()):
                if i < len(srcs):
                    out.append(self._process_single(t, srcs[i], generator))
                else:
                    out.append(torch.randn(t.shape, dtype=torch.float32, generator=generator, device="cpu").to(t.device, dtype=t.dtype))
            return NestedTensor(out)
        return self._process_single(target, srcs[0], generator)
