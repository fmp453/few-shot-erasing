import random

import torch

from glob import glob
from torch.utils.data import Dataset

import utils

"""
refernce : https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
"""

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class AblatingDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        placeholder_token,
        vae,
        concept_type:str="object",
        size=512,
        interpolation="bicubic",
        center_crop=False,
        device="cuda:0",
        batch_size:int=0,
        is_zero_shot=False
    ) -> None:
        super().__init__()

        templates = imagenet_templates_small
        if concept_type == "style":
            templates = imagenet_style_templates_small

        self.prompt = placeholder_token
        self.tokenizer = tokenizer
        self.templates = templates
        self.image_embeddings = []
        
        if not is_zero_shot:
            scaling_factor = 0.18215
            for f in glob(f"{data_root}/*"):
                if ".png" in f or ".jpg" in f or ".jpeg" in f:
                    image = utils.preprocess(f, center_crop, size, interpolation).to(device)
                    
                    with torch.no_grad():
                        latents = vae.encode(image).latent_dist.sample().detach().cpu() * scaling_factor
                    self.image_embeddings.append(latents[0])
        else:
            for _ in range(batch_size):
                self.image_embeddings.append(torch.rand((1, 4, 64, 64)))
        print(f"num of images : {len(self.image_embeddings)}")
        self._length = len(self.image_embeddings)
    
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        text = random.choice(self.templates).format(self.prompt)
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return tokenized, self.image_embeddings[index]

