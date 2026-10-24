import torch

def create_radial_filter(spectrum, H, W, device):
    Y = torch.linspace(-1, 1, H, device=device)
    X = torch.linspace(-1, 1, W, device=device)
    y, x = torch.meshgrid(Y, X, indexing='ij')
    
    radius = torch.sqrt(x**2 + y**2)
    radius = radius / radius.max()
    
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

class SpectralNoise:
    def __init__(self, a, b, spectrum_tensor):
        self.a = a
        self.b = b
        self.spectrum = spectrum_tensor
        self.seed = 0


    def _process_single_tensor(self, x, y):
        device = x.device
        dtype = x.dtype  # Remember original dtype (usually float16/bfloat16)
        *_, H, W = x.shape
        
        # 1. Cast only the individual unbinded tensors to float32 for the FFT
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
        
        # Transform back
        shaped = torch.fft.ifft2(torch.fft.ifftshift(blended_fft, dim=(-2, -1)), dim=(-2, -1)).real
        
        # Normalize
        std = shaped.std()
        if std > 0:
            shaped = (shaped - shaped.mean()) / std
        else:
            shaped = shaped - shaped.mean()
            
        # Return cleanly in the exact original dtype
        return shaped.to(dtype)

    def generate_noise(self, input_latent):
        # DO NOT use global .to(torch.float32) here! It destroys NestedTensor padding.
        x = self.a.generate_noise(input_latent)
        y = self.b.generate_noise(input_latent)
        
        if getattr(x, 'is_nested', False):
            # Unbind gives us views into the exact original padded memory buffer.
            # Using xi.copy_ safely overwrites the data while perfectly preserving the memory layout!
            for xi, yi in zip(x.unbind(), y.unbind()):
                xi.copy_(self._process_single_tensor(xi, yi))
            return x
        else:
            return self._process_single_tensor(x, y)
        
class SpectralNoiseEQ:
    def __init__(self, base_noise, spectrum_tensor, normalize):
        self.base_noise = base_noise
        self.spectrum = spectrum_tensor
        self.normalize = normalize
        self.seed = 0

    def _process_single_tensor(self, x):
        device = x.device
        dtype = x.dtype  
        *_, H, W = x.shape
        
        x_f = x.to(torch.float32)
        
        # 1. To the Frequency Domain
        fft_x = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))
        
        # 2. Get our EQ Mask (No longer bounded to 0-1!)
        filter_2d = create_radial_filter(self.spectrum, H, W, device)
        view_shape = [1] * (x.ndim - 2) + [H, W]
        eq_mask = filter_2d.view(*view_shape)
        
        # 3. DIRECT SCALING (The EQ step)
        eq_fft = fft_x * eq_mask
        
        # 4. Back to Spatial Domain
        shaped = torch.fft.ifft2(torch.fft.ifftshift(eq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        
        # 5. Global Normalization
        # (This ensures the overall volume of the noise stays acceptable for the scheduler,
        # but the *distribution* of that volume is dictated by your EQ curve).
        std = shaped.std()
        if std > 0 and self.normalize:
            shaped = (shaped - shaped.mean()) / std
        else:
            shaped = shaped - shaped.mean()
            
        return shaped.to(dtype)

    def generate_noise(self, input_latent):
        x = self.base_noise.generate_noise(input_latent)
        
        if getattr(x, 'is_nested', False):
            # Safe injection for Minimax H3
            for xi in x.unbind():
                xi.copy_(self._process_single_tensor(xi))
            return x
        else:
            # Standard for Flux / SDXL
            return self._process_single_tensor(x)
        
class MaskedNoise:
    def __init__(self, base_noise, masked_noise, mask, normalize, linear_blend):
        self.base_noise = base_noise
        self.masked_noise = masked_noise
        self.mask = mask
        self.normalize = normalize
        self.linear_blend = linear_blend

        # Pass through the seed of the base noise for the samplers
        self.seed = getattr(base_noise, "seed", getattr(masked_noise, "seed", 0))

    def _process_single_tensor(self, x, y, mask_tensor):
        device = x.device
        dtype = x.dtype
        *_, H, W = x.shape
        
        # 1. Prepare the mask tensor
        m = mask_tensor.clone().to(device, dtype=torch.float32)
        
        # ComfyUI masks are usually [H, W] or [B, H, W]. 
        # We need to make it [B, C, H, W] for the interpolate function.
        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 3:
            m = m.unsqueeze(1)
            
        # 2. Resize mask to EXACTLY match the latent grid dimensions
        # (Latents are usually 1/8th the size of the pixel mask)
        m = torch.nn.functional.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)
        
        # 3. Reshape mask dynamically to broadcast against the noise tensor
        m = m.squeeze(1) # Drop the channel dim, now [B, H, W]
        
        # If mask has only 1 batch, but our latent doesn't (or vice versa), 
        # squeeze it down to just [H, W] so it broadcasts globally.
        if m.shape[0] == 1 and (x.ndim == 3 or x.shape[0] != 1):
            m = m.squeeze(0) # Now [H, W]
            
        if m.ndim == 2: # [H, W]
            view_shape = [1] * (x.ndim - 2) + [H, W]
        elif m.ndim == 3: # [B, H, W]
            view_shape = [m.shape[0]] + [1] * (x.ndim - 3) + [H, W]
            
        m = m.view(*view_shape)
        m = torch.clamp(m, 0.0, 1.0)
        
        # 4. ENERGY-PRESERVING BLEND (Spatial Domain)
        # Prevents a loss of noise variance at the feathered edges of the mask
        x_f = x.to(torch.float32)
        y_f = y.to(torch.float32)
        
        if self.linear_blend:
            blended = (1.0 - m) * x_f + m * y_f
        else:
            blended = (torch.sqrt(1.0 - m) * x_f) + (torch.sqrt(m) * y_f)
        
        # 5. Global Normalization
        std = blended.std()
        if std > 0 and self.normalize:
            blended = (blended - blended.mean()) / std
        else:
            blended = blended - blended.mean()
            
        return blended.to(dtype)

    def generate_noise(self, input_latent):
        x = self.base_noise.generate_noise(input_latent)
        y = self.masked_noise.generate_noise(input_latent)
        
        if getattr(x, 'is_nested', False):
            # Safe injection for Minimax H3 / Video Models
            x_unbind = x.unbind()
            y_unbind = y.unbind()
            
            # Try to unpack the mask batch if it matches the video frame count
            m = self.mask
            if m.ndim == 3 and m.shape[0] == len(x_unbind):
                masks = m.unbind(dim=0)
            else:
                # Otherwise, apply the same mask to all frames
                masks = [m] * len(x_unbind)
                
            for xi, yi, mi in zip(x_unbind, y_unbind, masks):
                xi.copy_(self._process_single_tensor(xi, yi, mi))
            return x
        else:
            # Standard execution for Flux / SDXL
            return self._process_single_tensor(x, y, self.mask)

class LatentNoise:
    def __init__(self, latent_dict, normalize=True, mode="full"):
        self.latent_tensor = latent_dict["samples"]
        self.normalize = normalize
        self.mode = mode
        self.seed = 0

    def _process_single(self, target, src):
        *_, tH, tW = target.shape
        *_, sH, sW = src.shape
        
        # STRICT SIZE ENFORCEMENT
        if (tH, tW) != (sH, sW):
            raise ValueError(
                f"\n[Davcha Latent Noise] Shape Mismatch!\n"
                f"Your injected LatentNoise is {sW*8}x{sH*8} (Latent {sW}x{sH}), "
                f"but the generation target is {tW*8}x{tH*8} (Latent {tW}x{tH}).\n"
                f"Fix: Please pad/crop the source image in pixel space BEFORE generating the noise."
            )
            
        if self.mode == "full":
            out = src
        else:
            # We need standard random noise to supply the missing half of the equation
            # Generated on the same device and converted to float32 for FFT
            noise = torch.randn_like(src, dtype=torch.float32)
            src_f = src.to(torch.float32)
            
            # 1. Transform both to Frequency Domain
            fft_src = torch.fft.fft2(src_f, dim=(-2, -1))
            fft_noise = torch.fft.fft2(noise, dim=(-2, -1))
            
            if self.mode == "phase":
                # Ghost Mode: Image's Phase (Structure) + Noise's Magnitude (Static)
                mag = torch.abs(fft_noise)
                phase = torch.angle(fft_src)
                
            elif self.mode == "magnitude":
                # Texture Mode: Image's Magnitude (EQ Curve) + Noise's Phase (Scrambled Structure)
                mag = torch.abs(fft_src)
                phase = torch.angle(fft_noise)
                
            # 2. Recombine using Polar coordinates and inverse FFT back to Spatial Domain
            fft_combined = torch.polar(mag, phase)
            out = torch.fft.ifft2(fft_combined, dim=(-2, -1)).real
            out = out.to(src.dtype)
            
        # Optional Global Normalization
        if self.normalize:
            std = out.std()
            if std > 0:
                out = (out - out.mean()) / std
            else:
                out = out - out.mean()
                
        return out

    def generate_noise(self, input_latent):
        target = input_latent["samples"] if isinstance(input_latent, dict) else input_latent
        src = self.latent_tensor.to(target.device, dtype=target.dtype)
        
        # Minimax H3 / NestedTensor Video Safe-Guard
        if getattr(target, 'is_nested', False):
            t_unbind = target.unbind()
            s_unbind = src.unbind() if getattr(src, 'is_nested', False) else [src] * len(t_unbind)
            
            processed = [self._process_single(t, s) for t, s in zip(t_unbind, s_unbind)]
            return torch.nested.nested_tensor(processed, dtype=target.dtype, device=target.device)
        else:
            # Standard execution for Flux / SDXL
            if getattr(src, 'is_nested', False):
                src = src.unbind()[0]
                
            res = self._process_single(target, src)
            
            # Safe Batch Expansion
            if res.shape[0] != target.shape[0]:
                if res.shape[0] == 1:
                    res = res.expand(target.shape[0], -1, -1, -1)
                else:
                    raise ValueError(
                        f"[Davcha Latent Noise] Batch size mismatch. Latent has {res.shape[0]} frames, "
                        f"but generation requires {target.shape[0]}."
                    )
                    
            return res