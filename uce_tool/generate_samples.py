import torch
from diffusers import StableDiffusionPipeline

# runwayml/stable-diffusion-v1-5
# sd15_guide_keira_nathalie_to_celebrity
pipe = StableDiffusionPipeline.from_pretrained("../sd15_guide_keira_nathalie_to_celebrity", safety_checker=None, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = "a photo of keira" 
generator = torch.manual_seed(2024) 
image = pipe(prompt, generator=generator).images[0]
image.save("test.png")
