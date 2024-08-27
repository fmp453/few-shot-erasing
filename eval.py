

import warnings
import argparse

import pandas as pd
import torch
import timm

from tqdm import trange
from glob import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from cleanfid import fid

warnings.filterwarnings("ignore")


"""
label list
reference: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
"""

label_to_object_name = {
    497: "church",
    701: "parachute",
    574: "golf ball",
    571: "gas pump",
    569: "garbage truck",
    0: "tench",
    566: "French horn",
    491: "chain saw",
    217: "English springer",
    482: "cassette player"
}

def fid_score(args):
    print("Evaluating...")
    fid_value = fid.compute_fid(args.img_dir, args.orig_dir, batch_size=16, num_workers=4)
    print(f"FID: {fid_value}")

class CLIPEvaluator(object):
    def __init__(self, device, clip_model="openai/clip-vit-large-patch14") -> None:
        self.device = device
        self.clip_model = clip_model
        self.model = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor(clip_model)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.num_images = 1

        self.model.to(self.device)

    def get_PIL_images(self, img_path) -> list[Image.Image]:
        images = [Image.open(img_path).convert("RGB")]
        # for file in glob(f"{img_dir_path}/*"):
        #     if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        #         im = Image.open(file).convert("RGB")
        #         images.append(im)

        self.num_images = len(images)
        return images

    def tokenize(self, strings: list[str]):
        return self.tokenizer(strings, padding=True, return_tensors="pt")

    @torch.no_grad()
    def encode_text(self, tokens):
        # example case
        # {'input_ids': tensor([[49406,   320,  1125,   539,   320,  2368, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
        tokens = tokens.to(self.model.device)
        return self.model.get_text_features(**tokens)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        images = self.processor(images=images, return_tensors="pt")
        images = images.to(self.model.device)
        return self.model.get_image_features(**images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = self.tokenize([text] * self.num_images)
        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu()

    def get_image_features(self, img_dir: str, norm: bool = True) -> torch.Tensor:
        images = self.get_PIL_images(img_dir)
        image_features = self.encode_images(images)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features.cpu()

    def txt_img_similarity(self, text, img_dir) -> torch.Tensor:
        img_features = self.get_image_features(img_dir)
        text_features = self.get_text_features(text)

        return (text_features @ img_features.T).mean()

def clip_score(args):
    evaluator = CLIPEvaluator(device=f"cuda:{args.device}")

    df = pd.read_csv("coco_30k.csv")
    prompts = df["prompt"].tolist()
    image_ids = df["image_id"].tolist()
    scores = []

    for i in trange(len(prompts)):
        prompt = prompts[i]
        img_dir = f"{args.img_dir}/{image_ids[i]}.png"

        score = evaluator.txt_img_similarity(prompt, img_dir).item()
        scores.append(score)

    print(f"average clip score : {sum(scores) / len(scores)}")

def detection_rate(args):
    device = f"cuda:{args.device}"
    model = timm.create_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=True)
    model = model.eval()
    model = model.to(device)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    img_dir = glob(f"{args.img_dir}/*.png")
    cnt = 0

    for i in trange(len(img_dir)):
        img_path = img_dir[i]
        img = Image.open(img_path)
        
        output = model(transform(img).unsqueeze(0).to(device))
        predict_label = torch.argmax(output.softmax(dim=1)).item()
        if predict_label == args.label:
            cnt += 1
    
    print(f"Accuracy: {cnt / len(img_dir)}")

def main(args):

    method = args.method

    if method.lower() == "fid":
        fid_score(args)
    
    elif method.lower() == "clip-score":
        clip_score(args)
    
    else:
        detection_rate(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # common settings
    parser.add_argument("method", type=str, choices=["fid", "clip-score", "detection"], help="evaluation method")
    parser.add_argument("img_dir", type=str, help="path to dir of generated images (for all methods).")
    parser.add_argument("--device", type=int, help="gpu id. FID uses the all gpus by defualt. we assume the num of gpus is 4.", default=0)

    # fid
    parser.add_argument("--orig_dir", type=str, help="path to dir of real images (FID)")

    # detection rate
    parser.add_argument("--label", type=int, help="generate and evaluate imagenette label", choices=list(label_to_object_name.keys()))
    args = parser.parse_args()

    main(args)
