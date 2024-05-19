import os
import argparse

from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline


def text_encoder_save(model_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    text_encoder = CLIPTextModel.from_pretrained(model_name)
    text_encoder.save_pretrained(f"{save_dir}/text_encoder")

    print(f"Saved text encoder model at {save_dir}/text_encoder")

def vae_save(model_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    vae.save_pretrained(f"{save_dir}/vae")

    print(f"Saved VAE model at {save_dir}/vae")

def unet_save(model_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    unet.save_pretrained(f"{save_dir}/unet")

    print(f"Saved UNet model at {save_dir}/unet")

def pipeline_save(model_name: str, save_dir: str):
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    
    os.makedirs(save_dir, exist_ok=True)
    # text encoder & tokenizer
    pipe.text_encoder.save_pretrained(f"{save_dir}/text_encoder")
    pipe.tokenizer.save_pretrained(f"{save_dir}/tokenizer")

    # vae & uet & scheduler
    pipe.unet.save_pretrained(f"{save_dir}/unet")
    pipe.vae.save_pretrained(f"{save_dir}/vae")
    pipe.scheduler.save_pretrained(f"{save_dir}/scheduler")

    print(f"Saved pipline modules at {save_dir}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str)
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--unet", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--vae", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--save_dir", type=str, default="models/")
    
    args = parser.parse_args()

    if args.pipeline is not None:
        pipeline_save(args.pipeline, args.save_dir)
    else:
        # text encoder
        if args.text_encoder is not None:
            text_encoder_save(args.text_encoder, args.save_dir)
        
        if args.unet is not None:
            unet_save(args.unet, args.save_dir)
        
        if args.vae is not None:
            vae_save(args.vae, args.save_dir)
