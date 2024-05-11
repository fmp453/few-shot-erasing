import os
import random
import argparse
import warnings

import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, PNDMScheduler


warnings.filterwarnings('ignore')

def main(args):

    device = "cuda:0"

    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path)

    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1
    )

    if args.model_name is None:

        vae = AutoencoderKL.from_pretrained(args.vae_path)
        unet = UNet2DConditionModel.from_pretrained(args.unet_path)
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
        

        pipe = StableDiffusionPipeline(
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None
        ).to(device)
    
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name).to(device)
        pipe.text_encoder = text_encoder


    steps = 100
    seed = random.randint(0, 9999999)
    generator = torch.Generator(device).manual_seed(seed)
    images = pipe(
        args.prompt,
        num_inference_steps=steps,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator
    ).images
    print(seed)

    os.makedirs(args.save_dir, exist_ok=True)

    for i in range(len(images)):
        images[i].save(f"{args.save_dir}/{seed}-{i:02}.png")
    
    print(f"saved at {args.save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('text_encoder_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--unet_path', type=str)
    parser.add_argument('--vae_path', type=str)
    parser.add_argument('--num_images_per_prompt', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default="exp")
    args = parser.parse_args()
    
    main(args)
    