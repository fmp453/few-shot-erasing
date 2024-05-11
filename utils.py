import numpy as np
import pandas as pd
import PIL
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from packaging import version
from pathlib import Path
from typing import Tuple, Dict, Union
from torchvision import transforms

from diffusers import (
    AutoencoderKL, 
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


stable_diffusion_versions = {
    "14": "CompVis/stable-diffusion-v1-4",
    "15": "runwayml/stable-diffusion-v1-5",
}

clip_versions = {
    "oa": "openai/clip-vit-large-patch14",
    "oa-336": "openai/clip-vit-large-patch14-336",
}

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def _load_models(clip_version, stable_diffusion_version):
    clip_version = clip_versions[clip_version]
    stable_diffusion_version = stable_diffusion_versions[stable_diffusion_version]

    tokenizer = CLIPTokenizer.from_pretrained(clip_version)
    text_encoder = CLIPTextModel.from_pretrained(clip_version, low_cpu_mem_usage=False)
    vae = AutoencoderKL.from_pretrained(stable_diffusion_version, subfolder="vae",)
    unet = UNet2DConditionModel.from_pretrained(stable_diffusion_version, subfolder="unet")
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def _load_models_from_local(clip_version, stable_diffusion_version):
    if not(clip_version in clip_versions.keys() and stable_diffusion_version in stable_diffusion_versions.keys()):
        raise ValueError(f"{clip_version} or {stable_diffusion_version} is not valid.")
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_versions[clip_version])
    text_encoder = CLIPTextModel.from_pretrained(f"models/clip/{clip_version}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"models/diffusion/{stable_diffusion_version}/vae")
    unet = UNet2DConditionModel.from_pretrained(f"models/diffusion/{stable_diffusion_version}/unet")
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def load_models_from_local_optioned_path(
    text_encoder_path:str,
    unet_path:str="models/diffusion/15/unet",
    vae_path:str="models/diffusion/15/vae",
    tokenizer_version:str="openai/clip-vit-large-patch14",
):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
    vae = AutoencoderKL.from_pretrained(vae_path)
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def load_models(clip_version, stable_diffusion_version, use_local):
    if use_local:
        return _load_models_from_local(clip_version, stable_diffusion_version)
    else:
        return _load_models(clip_version, stable_diffusion_version)

def get_optimizer(
        parameters, 
        optimizer_name:str, 
        lr:float, 
        betas:Tuple[float, float]=(0.9, 0.99), 
        weight_decay:float=1e-2, 
        eps:float=1e-6
    ):
    optim_name = optimizer_name.lower()
    
    if optim_name == "adamw":
        return optim.AdamW(parameters, lr, betas, eps, weight_decay)

    elif optim_name == "adam":
        return optim.Adam(parameters, lr, betas, eps, weight_decay)
    
    elif optim_name == "adagrad":
        return optim.Adagrad(parameters, lr, weight_decay=weight_decay, eps=eps)

    elif optim_name == "adadelta":
        return optim.Adadelta(parameters, lr, eps=eps, weight_decay=weight_decay)

    raise ValueError(f"not match optimizer name : {optimizer_name}")

def plot_loss(history:Dict, save_path:str):
    df = pd.DataFrame(history)
    df.to_csv(f"{save_path}/loss.csv", index=False)
    plt.figure()
    df.plot()
    plt.savefig(f"{save_path}/loss.png")
    plt.close('all')

def preprocess(
        image_path: Union[Path, str], 
        center_crop:bool, 
        size:int, 
        interpolation:str
    ) -> torch.Tensor:
    
    flip_transform = transforms.RandomHorizontalFlip(p=0.5)
    interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
    }[interpolation]
    
    image = Image.open(image_path).convert("RGB")
    img = np.array(image).astype(np.uint8)

    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        (h, w) = (img.shape[0], img.shape[1])
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

    image = Image.fromarray(img)
    image = image.resize((size, size), resample=interpolation)
    image = flip_transform(image)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.0 * image - 1.0

    return image

def freeze_and_unfreeze_text_encoder(text_encoder, method="all"):
    if method == "all":
        return text_encoder
    
    for param in text_encoder.parameters():
        param.require_grad = False
    
    mlps = False
    final_attn = False
    attns = False
    if method == "mlp-only":
        mlps = True
    elif method == "attn-only":
        attns = True
    elif method == "mlp-attn":
        mlps = True
        attns = True
    elif method == "mlp-final-attn":
        mlps = True
        final_attn = True

    for param_name, module in text_encoder.named_modules():
        if mlps and "mlp.fc" in param_name:
            print(param_name)
            for param in module.parameters():
                param.require_grad = True
        
        if attns and ".self_attn." in param_name:
            print(param_name)
            for param in module.parameters():
                param.require_grad = True
        
        # for OpenAI CLIP
        if final_attn and "11.self_attn." in param_name:
            print(param_name)
            for param in module.parameters():
                param.require_grad = True
    
    return text_encoder

