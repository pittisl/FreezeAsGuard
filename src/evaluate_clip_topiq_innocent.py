import argparse
import torch
import numpy as np
import pyiqa
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm
from matplotlib import pyplot as plt 


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (512, 512))


def get_real_images_and_prompts(
    data_dir,
    user_start_idx=100,
    num_samples=100,
    prefix="",
):
    dataset = load_dataset(data_dir)
    real_images = dataset["train"].select(range(user_start_idx, user_start_idx + num_samples))
    prompts = [prefix + prompt for prompt in real_images["text"]]
    return real_images["image"], prompts


def get_fake_images(model_dir, prompts, seed=6666):
    pipe = StableDiffusionPipeline.from_pretrained(model_dir, safety_checker=None, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    generator = torch.manual_seed(seed)
    fake_images = pipe(prompts, generator=generator).images
    del pipe
    return fake_images

def compute_clip_scores(
    data_dir,
    ft_model,
    mft_model,
    num_samples=50,
    picking=False,
    seed=6666,
):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    topiq_metric = pyiqa.create_metric('topiq_fr-pipal', device=torch.device("cuda"))

    def calculate_clip_score(images, prompts):
        images_int = (images * 255).type(torch.uint8)
        score = clip_score_fn(images_int, prompts).detach()
        return round(float(score), 4)
    
    real_images, prompts = get_real_images_and_prompts(
        data_dir=data_dir,
        user_start_idx=100,
        num_samples=num_samples,
        prefix="",
    )
    ft_fake_images = get_fake_images(ft_model, prompts, seed)
    mft_fake_images = get_fake_images(mft_model, prompts, seed)
    
    clip_gt_list = []
    clip_ft_list = []
    clip_mft_list = []
    topiq_ft_list = []
    topiq_mft_list = []
    
    for idx, (prompt, real_image, ft_fake_image, mft_fake_image) in enumerate(zip(prompts, real_images, ft_fake_images, mft_fake_images)):
        real_image_tensor = preprocess_image(np.array(real_image.convert("RGB")))
        ft_fake_image_tensor = preprocess_image(np.array(ft_fake_image.convert("RGB")))
        mft_fake_image_tensor = preprocess_image(np.array(mft_fake_image.convert("RGB")))
        
        # ft_fake_image_tensor = torch.tensor(ft_fake_image).permute(0, 3, 1, 2)
        # mft_fake_image_tensor = torch.tensor(mft_fake_image).permute(0, 3, 1, 2)
        
        clip_gt = calculate_clip_score(
            images=real_image_tensor,
            prompts=[prompt],
        )
        clip_ft = calculate_clip_score(
            images=ft_fake_image_tensor,
            prompts=[prompt],
        )
        clip_mft = calculate_clip_score(
            images=mft_fake_image_tensor,
            prompts=[prompt],
        )
        
        clip_gt_list += [clip_gt]
        clip_ft_list += [clip_ft]
        clip_mft_list += [clip_mft]
        
        topiq_ft = topiq_metric(real_image_tensor, ft_fake_image_tensor)
        topiq_mft = topiq_metric(real_image_tensor, mft_fake_image_tensor)

        topiq_ft = topiq_ft.cpu().item()
        topiq_mft = topiq_mft.cpu().item()
        
        topiq_ft_list += [topiq_ft]
        topiq_mft_list += [topiq_mft]
        
        print(f"\nExample {idx} - Prompt: {prompt}")
        
        if picking:
            fig = plt.figure(figsize=(10, 7)) 
            fig.add_subplot(1, 3, 1)
            plt.imshow(real_image) 
            plt.axis('off') 
            plt.title(f"Ground truth\nCLIP score = {clip_gt}")
            
            fig.add_subplot(1, 3, 2)
            plt.imshow(ft_fake_image) 
            plt.axis('off') 
            plt.title(f"Full fine-tuning\nCLIP score = {clip_ft}\nTOPIQ = {topiq_ft}")
            
            fig.add_subplot(1, 3, 3)
            plt.imshow(mft_fake_image) 
            plt.axis('off') 
            plt.title(f"Freezing xx% \nCLIP score = {clip_mft}\nTOPIQ = {topiq_mft}")

            plt.show()
        
    print(f"mean_clip_gt = {np.mean(clip_gt_list)}")
    print(f"mean_clip_ft = {np.mean(clip_ft_list)} | mean_topiq_ft = {np.mean(topiq_ft_list)}")
    print(f"mean_clip_mft = {np.mean(clip_mft_list)} | mean_topiq_mft = {np.mean(topiq_mft_list)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="evaluating clip score of generated images in innocent domain")
    
    parser.add_argument("--description", type=str, required=True, help="user notes of the run")
    parser.add_argument("--data_dir", type=str, default="logo-wizard/modern-logo-dataset", help="data dir to innocent data")
    parser.add_argument("--ft_model", type=str, required=True, help="fully finetuned model dir")
    parser.add_argument("--mft_model", type=str, required=True, help="other finetuned model dir")
    parser.add_argument("--num_samples", type=int, default=50, help="number of samples for each scheme in evaluation")
    parser.add_argument("--picking", action="store_true", help="Whether to check each image one by one.")
    parser.add_argument("--seed", type=int, default=6666, help="seed for data extraction and generation")
    
    args = parser.parse_args()
    
    print(f"\n\n#### {args.description}\n\n")
    
    compute_clip_scores(
        data_dir=args.data_dir,
        ft_model=args.ft_model,
        mft_model=args.mft_model,
        num_samples=args.num_samples,
        picking=args.picking,
        seed=args.seed,
    )
    
    # compute_clip_scores(
    #     data_dir="logo-wizard/modern-logo-dataset",
    #     ft_model="../sd15_logo_full",
    #     mft_model="../sd15_logo_random20",
    #     seed=6666,
    # )
        
        