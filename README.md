# ComfyUI Davcha Sampling Tricks

**Davcha Sampling Tricks** is an advanced custom node extension for ComfyUI. It provides powerful frequency-domain noise manipulation tools (Spectral Blending & EQ), spatial masking capabilities, and a dynamic multi-sampler chaining system. 

Whether you want fine-grained control over the initial noise structure for specific frequencies, target custom noise to specific spatial regions, or swap out sampling algorithms mid-generation, this toolkit gives you the flexibility you need.

## 🌟 Key Features

### 1. Spectral Noise Manipulation (FFT)
Instead of blending noise purely in the spatial domain, these nodes use Fast Fourier Transforms (FFT) to convert noise into the frequency domain. This allows you to apply radial masks and equalizer curves to low, mid, and high frequencies independently.

### 2. Spatial Noise Masking (Energy-Preserving)
Target specific areas of your image with different noise structures using standard masks. Our spatial blending uses an energy-preserving `sqrt()` algorithm to prevent the "variance dip" that usually causes flat, halo-like artifacts along feathered mask edges.

### 3. NestedTensor & Next-Gen Model Support
Designed with modern architectures in mind. The noise generation safely handles `NestedTensor` formats (essential for models like **Minimax H3**) by using non-destructive unbinding and exact memory layout preservation, while remaining fully compatible with standard models like **Flux** and **SDXL**.

---

## 🧩 Included Nodes

### 🎛️ Spectral Noise Blend (`DavchaSpectralNoiseBlend`)
Blends two different noise sources together using a frequency-based radial mask.
* **Inputs:**
  * `noise_a`: First noise source (e.g., Fractal noise).
  * `noise_b`: Second noise source (e.g., Uniform noise).
  * `spectrum`: A 1D tensor controlling the frequency blending curve.
* **How it works:** The `spectrum` tensor maps frequencies from the center (low freq) to the edges (high freq). A value of `0.0` uses pure **Noise B**, while `1.0` uses pure **Noise A**. Smoothstep interpolation ensures seamless transitions between frequency bins.

### 🎚️ Spectral Noise EQ (`DavchaSpectralNoiseEQ`)
Acts as a graphic equalizer for a single noise source. Shape the structural frequencies of your initial noise before sampling begins.
* **Inputs:** 
  * `base_noise`: The starting noise (Standard Uniform Noise is recommended).
  * `spectrum`: A 1D tensor representing your EQ curve.
* **How it works:** 
  * `1.0` = Unity (Unchanged)
  * `0.0` = Mute (Removes the frequency)
  * `> 1.0` = Boost (Amplifies the frequency)
  * *Note: The node automatically applies a global normalization at the end to ensure the overall noise energy remains acceptable for the diffusion scheduler.*

### 🎭 Masked Noise Blend (`DavchaMaskedNoise`)
Spatially blends two noises together using an image mask, perfect for inpainting or localized texture control.
* **Inputs:**
  * `base_noise`: Noise used where the mask is BLACK (`0.0`).
  * `masked_noise`: Noise used where the mask is WHITE (`1.0`).
  * `mask`: Standard ComfyUI mask tensor.
* **How it works:** The node automatically resizes standard pixel masks (e.g., 1024x1024) to match the internal latent grid (e.g., 128x128). It then blends the two noises using energy-preserving math to keep the noise contrast punchy and seamless, even across soft gradient masks.

### ⏳ Scheduled Sampler (`DavchaScheduledSampler`)
Allows you to chain multiple distinct samplers together during a single generation process.
* **Inputs:**
  * `sampler_1`, `sampler_2`, etc.: Autogrowing inputs allowing you to plug in anywhere from 1 to 10 samplers.
  * `splits`: A comma-separated string of integers defining the step transitions.
* **How it works:** If you run 30 total steps and provide 3 samplers, you can set `splits` to `10, 20`. 
  * Steps `0-10`: Uses Sampler 1
  * Steps `11-20`: Uses Sampler 2
  * Steps `21-30`: Uses Sampler 3

### 🔍 Sampler Fourier Upscale (`DavchaSamplerFourierUpscale`)
A sampler that upscales the latent *during* sampling. At each scaling step it splits the current state into the denoised estimate and its noise, upscales the denoised estimate, and rebuilds the noise at the new size: the original noise is kept exactly in the frequency band it already covered, and fresh noise fills only the new high frequencies. The latent is then re-noised at the next sigma, so the following steps see a consistent, correctly scaled noise.
* **Inputs:**
  * `scale_ratio`: Total upscale ratio, reached at the last scaling step.
  * `scale_steps`: Number of steps over which the size is ramped up to `scale_ratio` (`-1` = all steps).
  * `denoised_upscale`: Method used for the denoised estimate (`bislerp`, `fourier_tukey`, `bicubic`, `bilinear`).
  * `noise_anchor`: `1.0` keeps the original noise in its frequency band, `0.0` replaces it with fresh noise.
  * `tukey_alpha`: Cosine taper width of the spectrum when using `fourier_tukey`.
  * `anchor_spectrum` *(optional)*: Radial curve (Sigmas or Float) modulating `noise_anchor` from low to high frequencies.
  * `high_freq_noise` *(optional)*: Noise node supplying the fresh noise. Its seed is offset by the step index; without it, noise is generated from the sampler seed.
* **Notes:**
  * Flow-matching and EPS models are detected from the model itself. 2x2 channel-patched latents (**Flux.2**, **Wan 2.2**) are unpacked before upscaling, so the upscale works on the real spatial grid.
  * Video latents are upscaled frame by frame. With `NestedTensor` latents, only the first (video) latent is upscaled.
  * Not compatible with inpainting (`denoise_mask`), since the latent no longer matches the mask size.

#### Recipe: one-pass hires fix (tested on Flux.2 Klein 9B)
Use it inside a **Scheduled Sampler**, as a single step at a repeated sigma, so it only changes resolution without denoising:
* Sigmas: `1.0, 0.966, 0.966, 0.9331, 0.875, 0.7299, 0.358, 0.0`. Note the repeated `0.966`.
* Scheduled Sampler with `splits` = `1, 2`:
  * `sampler0`: `SamplerDPMPP_2M_SDE` (heun, eta `0.05`) for the first step (`1.0 → 0.966`).
  * `sampler1`: **Sampler Fourier Upscale** (`scale_ratio` `1.26`, `scale_steps` `-1`, `bislerp`, `noise_anchor` `1.0`) for the upscale step (`0.966 → 0.966`).
  * `sampler2`: `SamplerER_SDE` (Reverse-time SDE, max_stage `3`, eta `1.0`) for the remaining steps at the new resolution.
* A structured noise (e.g. RES4LYF `AdvancedNoise`, fractal) plugged into `high_freq_noise` works well for the new high frequencies.

---

## 🛠️ Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository (or place the extension folder here):
   ```bash
   git clone <your-repo-url-here>
   ```
3. Restart ComfyUI. The nodes will be available in the node menu under:
   * `davcha/noise/spectral`
   * `sampling/custom_sampling/samplers`

## 💡 Usage Tips

* **Spectrum Tensors:** The `spectrum` inputs on the noise nodes expect a 1D tensor (a list of float values). You can create these using any standard tensor or float-list generation nodes in ComfyUI.
* **Splits Formatting:** When using the **Scheduled Sampler**, ensure your `splits` string only contains numbers and commas (e.g., `15, 30`). The number of splits should logically align with the number of samplers you've connected minus one.
* **Dtypes:** The spectral nodes perform their FFT math in `float32` for precision but will strictly cast the output back to your latent's original exact dtype (e.g., `float16` or `bfloat16`), preventing upstream dtype crash issues in ComfyUI workflows.