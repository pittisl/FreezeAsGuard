import argparse
import torch
import numpy as np
import pyiqa
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
# from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm
from matplotlib import pyplot as plt 


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (512, 512))


def get_real_images_and_prompts(
    data_dir,
    selected_idx,
    num_samples=100,
    prefix="",
    use_long_prompt=True,
    seed=6666,
):
    dataset = load_dataset("imagefolder", data_dir=data_dir, drop_labels=False)
    real_images = dataset["test"].filter(lambda example: example["label"] == selected_idx).shuffle(seed).select(range(num_samples))
    if use_long_prompt:
        prompts = [prompt for prompt in real_images["text"]]
    else:
        prompts = [prefix + prompt for prompt in real_images["name"]]
    return real_images["image"], prompts

def get_fake_images(model_dir, prompts, seed=6666):
    pipe = StableDiffusionPipeline.from_pretrained(model_dir, safety_checker=None, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    generator = torch.manual_seed(seed)
    fake_images = pipe(prompts, generator=generator).images
    del pipe
    return fake_images

def compute_fid_and_topiq_scores(
    data_dir,
    selected_idx,
    ft_model,
    mft_model,
    num_samples=50,
    picking=False,
    seed=6666,
):
    topiq_metric = pyiqa.create_metric('topiq_fr-pipal', device=torch.device("cuda"))
    
    real_images, prompts = get_real_images_and_prompts(
        data_dir=data_dir,
        selected_idx=selected_idx,
        num_samples=num_samples,
        prefix="a photo of",
    )
    ft_fake_images = get_fake_images(ft_model, prompts, seed)
    mft_fake_images = get_fake_images(mft_model, prompts, seed)
    
    ft_gt_fid = evaluate_fid(ref_images=real_images, fake_images=ft_fake_images)
    mft_gt_fid = evaluate_fid(ref_images=real_images, fake_images=mft_fake_images)
    print(f"ft_gt_fid = {ft_gt_fid}")
    print(f"mft_gt_fid = {mft_gt_fid}")
    
    topiq_ft_list = []
    topiq_mft_list = []
    
    for idx, (prompt, real_image, ft_fake_image, mft_fake_image) in enumerate(zip(prompts, real_images, ft_fake_images, mft_fake_images)):
        real_image_tensor = preprocess_image(np.array(real_image.convert("RGB")))
        ft_fake_image_tensor = preprocess_image(np.array(ft_fake_image.convert("RGB")))
        mft_fake_image_tensor = preprocess_image(np.array(mft_fake_image.convert("RGB")))
        
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
            plt.title(f"Ground truth")
            
            fig.add_subplot(1, 3, 2)
            plt.imshow(ft_fake_image) 
            plt.axis('off') 
            plt.title(f"Full fine-tuning\ntopiq score = {topiq_ft}")
            
            fig.add_subplot(1, 3, 3)
            plt.imshow(mft_fake_image) 
            plt.axis('off') 
            plt.title(f"Freezing xx%\ntopiq score = {topiq_mft}")

            plt.show()
    
    print(f"mean_topiq_ft = {np.mean(topiq_ft_list)}")
    print(f"mean_topiq_mft = {np.mean(topiq_mft_list)}")


def evaluate_fid(ref_images, fake_images):
    # compute FID score between real and fake images
    ref_images = [np.array(image.convert("RGB")) for image in ref_images]
    ref_images_tensors = torch.cat([preprocess_image(image) for image in ref_images])
    fake_images = [np.array(image.convert("RGB")) for image in fake_images]
    fake_images_tensors = torch.cat([preprocess_image(image) for image in fake_images])
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(ref_images_tensors, real=True)
    fid.update(fake_images_tensors, real=False)
    return float(fid.compute())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluating fid and topiq scores of generated images in target domain")
    
    parser.add_argument("--description", type=str, required=True, help="user notes of the run")
    parser.add_argument("--data_dir", type=str, default="../ff25", help="data dir to ff25")
    parser.add_argument("--selected_idx", type=int, default=0, help="class idx of person")
    parser.add_argument("--ft_model", type=str, required=True, help="fully finetuned model dir")
    parser.add_argument("--mft_model", type=str, required=True, help="other finetuned model dir")
    parser.add_argument("--num_samples", type=int, default=50, help="number of samples for each scheme in evaluation")
    parser.add_argument("--picking", action="store_true", help="Whether to check each image one by one.")
    parser.add_argument("--seed", type=int, default=6666, help="seed for data extraction and generation")
    
    args = parser.parse_args()
    
    print(f"\n\n#### {args.description}\n\n")
    
    compute_fid_and_topiq_scores(
        data_dir=args.data_dir,
        selected_idx=args.selected_idx,
        ft_model=args.ft_model,
        mft_model=args.mft_model,
        num_samples=args.num_samples,
        picking=args.picking,
        seed=args.seed,
    )
    
    # compute_fid_and_topiq_scores(
    #     data_dir="../ff25",
    #     selected_idx=4,
    #     ft_model="../sd15_target_relearn_4_full",
    #     mft_model="../sd15_target_relearn_4_random20",
    #     num_samples=50,
    #     picking=False,
    #     seed=6666,
    # )
        
        