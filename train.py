"""
refernce : https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py

"""

import gc
import os
import time
import argparse
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import trange
from collections import OrderedDict

from data import AblatingDataset

import utils

class Configuration:
    def __init__(self, *args, **kwargs) -> None:
        args = args[0]
        self.target_concept = args["concept"]
        self.concept_type = args["concept_type"]
        self.optimizer_name = args["optim"]
        self.lr = args["lr"]
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.eps = args["eps"]
        self.weight_decay = args["weight_decay"]
        self.batch_size = args["batch_size"]
        self.use_local_model = args["local"]
        self.gpu_id = args["gpu_id"]
        self.dataset_path = args["data"]
        self.save_path = args["save"]
        self.num_epoch = args["epochs"]
        self.clip_version = args["clip_ver"]
        self.sd_version = args["sd_ver"]
        self.text_encoder_path = args["text_encoder_path"]
        self.diffusion_path = args["diffusion_path"]
        self.tokenizer_version = args["tokenizer_version"]
        self.is_zero_shot = args["is_zero_shot"]
        
def get_text_embeddings(text_encoder, tokenized_text):
    # ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L377

    device = text_encoder.device
    weight_dtype = text_encoder.dtype

    text_embedding = text_encoder(tokenized_text.to(device))[0].to(weight_dtype)
    return text_embedding

def get_target_noise(scheduler, noise, latents=None, timesteps=None):
    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    
    return target

def train(config: Configuration):
    
    target_prompt = config.target_concept

    device = f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    dataset_path = config.dataset_path
    save_path = config.save_path
    batch_size = config.batch_size
    num_epochs = config.num_epoch
    
    
    tokenizer, text_encoder, vae, unet, scheduler = utils.load_models_from_local_optioned_path(
        text_encoder_path=config.text_encoder_path,
        unet_path=f"{config.diffusion_path}/unet",
        vae_path=f"{config.diffusion_path}/vae",
        tokenizer_version=config.tokenizer_version,
    )

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)
    vae.eval()
    
    # freeze unet parameters
    for param in unet.parameters():
        param.requires_grad = False

    
    text_encoder = utils.freeze_and_unfreeze_text_encoder(text_encoder, method="mlp-final-attn")

    # optimizer setting
    optimizer = utils.get_optimizer(
        text_encoder.parameters(),
        config.optimizer_name,
        config.lr,
        (config.beta1, config.beta2),
        config.weight_decay,
        config.eps,
    )

    train_dataset = AblatingDataset(
        data_root=dataset_path,
        tokenizer=tokenizer,
        size=512,
        concept_type=config.concept_type,
        placeholder_token=target_prompt,
        center_crop=False,
        vae=vae,
        device=device,
        batch_size=batch_size,
        is_zero_shot=config.is_zero_shot
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    
    history = {"loss": []}

    os.makedirs(save_path, exist_ok=True)
    start = time.perf_counter()

    pbar = trange(0, num_epochs, desc="Epoch")
    for epoch in pbar:
        loss_avg = 0
        cnt = 0
        text_encoder.train()
        for step, (tokenized, image_embedding) in enumerate(train_dataloader):
            text_embedding = get_text_embeddings(
                text_encoder=text_encoder, 
                tokenized_text=tokenized
            )
            
            # bs, 4, 64, 64
            # if zero shot, image_embedding is random noise
            latents = image_embedding.to(device)

            noise = torch.randn_like(latents).to(device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            model_pred = unet(noisy_latents, timesteps, text_embedding).sample

            target = get_target_noise(scheduler=scheduler, noise=noise, latents=latents, timesteps=timesteps)
            
            # in Textual Inversion, loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = -F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg += loss.detach().item()
            cnt += step
            
            history["loss"].append(loss.detach().item())
        pbar.set_postfix(OrderedDict(loss=loss_avg / (cnt + 1e-9)))
        text_encoder.eval()
        text_encoder.save_pretrained(f"{save_path}/epoch-{epoch}")
    
    end = time.perf_counter()
    print(f"Time : {end - start}")

    utils.plot_loss(history, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # concept to erase
    parser.add_argument("--concept", type=str, required=True, help="concept to erase. for example, 'Eiffel Tower'")
    parser.add_argument("--concept_type", type=str, default="object", choices=["object", "style"])
    # optimizer setting
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "AdamW", "Adadelta", "Adagrad"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    # use local model?
    parser.add_argument("--local", action='store_true')
    # other training setting
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    # GPU ID
    parser.add_argument("--gpu_id", type=int, default=0)
    # dataset and save path
    parser.add_argument("--data", type=str, default="ds")
    parser.add_argument("--save", type=str, default="networks")
    # model version setting
    parser.add_argument("--clip_ver", type=str, default="oa", choices=["oa", "oa-336"])
    parser.add_argument("--sd_ver", type=str, default="15", choices=["14", "15"])
    parser.add_argument("--text_encoder_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--tokenizer_version", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--diffusion_path", type=str, default="models/diffusion/15")
    # zero shot?
    parser.add_argument("--is_zero_shot", action='store_true')
    
    args = vars(parser.parse_args())
    config = Configuration(args)
    
    train(config=config)
