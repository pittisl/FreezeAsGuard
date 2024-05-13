import torch
from diffusers import StableDiffusionPipeline

seed = 6666
pipe = StableDiffusionPipeline.from_pretrained("../sd15_logo_noema_m20_inn_included", safety_checker=None, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
generator = torch.manual_seed(seed)

prompts = 4 * ["a logo of cafe restaurant bar with the golden arch and fish in it, handwritten letters, white background, sienna, white foreground, minimalism, modern"]
images = pipe(prompts, generator=generator).images

for idx, image in enumerate(images):
    image.save(f"test{idx}.png")