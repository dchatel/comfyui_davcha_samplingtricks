import torch

from comfy_api.latest import io
from comfy.samplers import KSAMPLER
from .noise import *
from .sampler import sampler_function

# ------------------------------------------------------------
# Noise nodes
# ------------------------------------------------------------
class DavchaSpectralNoiseBlend(io.ComfyNode):
    """
    Blends a Fractal (A) and Uniform (B) noise together using a frequency-based mask.
    0.0 uses pure Noise B. 1.0 uses pure Noise A.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaSpectralNoiseBlend",
            display_name="Spectral Noise Blend",
            category="davcha/noise/spectral",
            inputs=[
                io.Noise.Input("noise_a"),
                io.Noise.Input("noise_b"),
                io.MultiType.Input("spectrum", types=[io.Sigmas, io.Float], tooltip="1D tensor controlling frequency blending. 0.0 = Noise B, 1.0 = Noise A"),
            ],
            outputs=[
                io.Noise.Output(),
            ]
        )

    @classmethod
    def execute(cls, noise_a, noise_b, spectrum) -> io.NodeOutput:
        if isinstance(spectrum, float):
            # Convert single float to a 1D tensor for compatibility
            spectrum = torch.tensor([spectrum], dtype=torch.float32)
        elif isinstance(spectrum, list):
            # Convert list of floats to a 1D tensor for compatibility
            spectrum = torch.tensor(spectrum, dtype=torch.float32)
        blended_noise = SpectralNoise(noise_a, noise_b, spectrum)
        return io.NodeOutput(blended_noise)
    
class DavchaSpectralNoiseEQ(io.ComfyNode):
    """
    Applies a graphic EQ to a single noise source. 
    1.0 is unchanged. 0.0 removes the frequency. >1.0 boosts it.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaSpectralNoiseEQ",
            display_name="Spectral Noise EQ",
            category="davcha/noise/spectral",
            inputs=[
                io.Noise.Input("base_noise", tooltip="Standard Uniform Noise recommended."),
                io.MultiType.Input("spectrum", types=[io.Sigmas, io.Float], tooltip="1D tensor EQ curve. 1.0 = Unity, 0.0 = Mute, 2.0+ = Boost"),
                io.Boolean.Input("normalize", default=True, tooltip="Normalize the output noise to have a standard deviation of 1.0. This is recommended for most use cases."),
            ],
            outputs=[
                io.Noise.Output(),
            ]
        )

    @classmethod
    def execute(cls, base_noise, spectrum, normalize) -> io.NodeOutput:
        if isinstance(spectrum, float):
            # Convert single float to a 1D tensor for compatibility
            spectrum = torch.tensor([spectrum], dtype=torch.float32)
        elif isinstance(spectrum, list):
            # Convert list of floats to a 1D tensor for compatibility
            spectrum = torch.tensor(spectrum, dtype=torch.float32)
        eq_noise = SpectralNoiseEQ(base_noise, spectrum, normalize)
        return io.NodeOutput(eq_noise)

class DavchaMaskedNoise(io.ComfyNode):
    """
    Spatially blends two noises together using an image mask.
    Black areas (0.0) use the Base Noise. White areas (1.0) use the Masked Noise.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaMaskedNoise",
            display_name="Masked Noise Blend",
            category="davcha/noise/spatial",
            inputs=[
                io.Noise.Input("base_noise", tooltip="Noise used where the mask is BLACK (0.0)"),
                io.Noise.Input("masked_noise", tooltip="Noise used where the mask is WHITE (1.0)"),
                io.Mask.Input("mask", tooltip="Spatial Mask tensor"),
                io.Boolean.Input("normalize", default=True, tooltip="Normalize the output noise to have a standard deviation of 1.0. This is recommended for most use cases."),
                io.Boolean.Input("linear_blend", default=False, tooltip="If enabled, performs a linear blend between the two noises. If disabled, uses a square root blend for smoother transitions."),
            ],
            outputs=[
                io.Noise.Output(),
            ]
        )

    @classmethod
    def execute(cls, base_noise, masked_noise, mask, normalize, linear_blend) -> io.NodeOutput:
        blended_noise = MaskedNoise(base_noise, masked_noise, mask, normalize, linear_blend)
        return io.NodeOutput(blended_noise)

class DavchaLatentNoise(io.ComfyNode):
    """
    Converts an encoded image latent (or raw image) into a Noise object.
    Perfect for injecting structured data into a Masked Noise Blend for seamless inpainting.
    Requires the input to perfectly match the generation dimensions.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaLatentNoise",
            display_name="Latent Noise Generator",
            category="davcha/noise",
            inputs=[
                io.MultiType.Input("canvas", types=[io.Latent, io.Image], tooltip="Provide either an encoded LATENT or a pixel IMAGE."),
                io.Combo.Input("mode", default="full", options=["full", "phase", "magnitude"], tooltip="Full: Standard Image. Phase: 'Ghost' structure mask. Magnitude: Custom frequency texture."),
                io.Boolean.Input("normalize", default=True, tooltip="Normalize the output noise to have a standard deviation of 1.0. This is recommended for most use cases."),
                io.Vae.Input("vae", optional=True, tooltip="Required ONLY if providing an IMAGE."),
            ],
            outputs=[
                io.Noise.Output(),
            ]
        )

    @classmethod
    def execute(cls, canvas, mode="full", normalize=True, vae=None) -> io.NodeOutput:
        # 1. Determine if we received an IMAGE (tensor) or LATENT (dict)
        if isinstance(canvas, dict) and "samples" in canvas:
            latent = canvas
            
        elif isinstance(canvas, torch.Tensor):
            # It's an IMAGE, so we must encode it on the fly.
            if vae is None:
                raise ValueError(
                    "\n[Davcha Latent Noise] Missing VAE!\n"
                    "You provided an IMAGE, but no VAE was connected. "
                    "Please connect a VAE to encode the image into latent space."
                )
            
            # Standard ComfyUI VAE Encode: Expects [B, H, W, C], drops alpha channel if present
            pixels = canvas[:, :, :, :3]
            encoded_tensor = vae.encode(pixels)
            
            # Wrap it in the standard ComfyUI latent dictionary format
            latent = {"samples": encoded_tensor}
            
        else:
            raise TypeError(f"[Davcha Latent Noise] Unsupported input type. Expected LATENT dict or IMAGE tensor, got {type(canvas)}")
            
        # 2. Pass to the noise generator object (now including mode)
        latent_noise = LatentNoise(latent, normalize=normalize, mode=mode)
        
        return io.NodeOutput(latent_noise)
    
# ------------------------------------------------------------
# Sampling nodes
# ------------------------------------------------------------
class DavchaScheduledSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(input=io.Sampler.Input("sampler"), prefix="sampler", min=1, max=10)
        return io.Schema(
            node_id="DavchaScheduledSampler",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Autogrow.Input("autogrow", template=template),
                io.String.Input("splits", multiline=False, dynamic_prompts=False),
            ],
            outputs=[
                io.Sampler.Output(),
            ]
        )
   
    @classmethod
    def execute(cls, autogrow, splits) -> io.NodeOutput:
        splits_list = [int(s.strip()) for s in splits.split(',')]
        samplers = [v for k, v in sorted(autogrow.items())]
        
        return io.NodeOutput(KSAMPLER(sampler_function, extra_options={'samplers': samplers, 'splits': splits_list}))

class DavchaEnsureDecreasingSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaEnsureDecreasingSigmas",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Sigmas.Input("sigmas"),
            ],
            outputs=[
                io.Sigmas.Output(),
            ]
        )
   
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        # Ensure that the sigmas are strictly decreasing
        x = torch.clone(sigmas)
        d = abs(x[1:] - x[:-1])
        x[-1] = 0.0
        x[:-1] = reversed(reversed(d).cumsum(dim=0))
        return io.NodeOutput(x)
