# Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning

This repository contains the implementation of [Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning](https://arxiv.org/abs/2405.07288).

## Installation
The environments of our experiments are based on [PyTorch1.13.1 (docker image)](https://hub.docker.com/layers/pytorch/pytorch/1.13.1-cuda11.6-cudnn8-runtime/images/sha256-1e26efd426b0fecbfe7cf3d3ae5003fada6ac5a76eddc1e042857f5d049605ee)

Pull docker image.
```bash
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
```

Install other packages.

```bash
pip install -r requirements.txt
```

Save the stable diffusion 1.5 pipeline (text encoder, tokenizer, scheduler, VAE, and U-Net).

```bash
python load_save.py --pipeline runwayml/stable-diffusion-v1-5 --save_dir models/sd-15
```

## Training (Updating parameters for erasing)
Store the prepared images in a directory. Supported `.png`, `jpg`, and `.jpeg`

```
.
└── ds
    └── church
        ├── church-01.jpg
        ├── church-02.png
        ├── church-03.jpg
        └── church-04.jpeg
```

Run following command for training (erasing).

```bash
python train.py --concept "Eiffel Tower" --concept_type object --save eiffel-tower --data ds/church --local --text_encoder_path models/sd-15/text_encoder --diffusion_path models/sd-15 --epochs 4
```

Erased models are stored like below.

```
.
└── eiffel-tower
    ├── epoch-0
    ├    ├── pytorch_model.bin
    ├    └── config.json
    ├── epoch-1
    ├    ├── pytorch_model.bin
    ├    └── config.json
    ├── epoch-2
    ├    ├── pytorch_model.bin
    ├    └── config.json
    ├── epoch-3
    ├    ├── pytorch_model.bin
    ├    └── config.json
    ├──loss.csv
    └──loss.png
```

## Inference
inference (PNDM Scheduler and 100 inference steps) 

```bash
python infer.py "a photo of Eiffel Tower." eiffel-tower/epoch-03 --tokenizer_path models/sd-15/tokenizer --unet_path models/sd-15/unet --vae_path models/sd-15/vae
```

or 

```bash
python infer.py "a photo of Eiffel Tower." eiffel-tower/epoch-03 --model_name runwayml/stable-diffusion-v1-5
```

this command use the Stable Diffusion 1.5 except the text encoder.

# Citation
The preprint can be cited as follows

```
@misc{fuchi2024erasing,
      title={Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning}, 
      author={Masane Fuchi and Tomohiro Takagi},
      year={2024},
      eprint={2405.07288},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
This implementation is based on [Textual Inversion using diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py). 

Baselines are
- https://github.com/rohitgandikota/erasing
- https://github.com/Con6924/SPM
- https://github.com/rohitgandikota/unified-concept-editing/tree/main
