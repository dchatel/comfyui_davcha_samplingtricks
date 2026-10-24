from comfy_api.latest import io, ComfyExtension
from .nodes import *

class DavchaSamplingTricksExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DavchaSpectralNoiseBlend, 
            DavchaSpectralNoiseEQ,
            DavchaScheduledSampler,
            DavchaMaskedNoise,
            DavchaLatentNoise,
            DavchaEnsureDecreasingSigmas,
        ]
        
async def comfy_entrypoint() -> DavchaSamplingTricksExtension:
    return DavchaSamplingTricksExtension()