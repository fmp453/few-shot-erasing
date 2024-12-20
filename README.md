# Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning

This repository contains the implementation of [Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning](https://arxiv.org/abs/2405.07288).

> [!NOTE]  
> We used stable diffusion 1.5. in our experiments, but it has been deleted from huggingface (as of September 2, 2024).

## News
- Proceedings of BMVC is available ([link](https://bmvc2024.org/proceedings/216/))
- Camera-ready version is released on arXiv
- Our paper has been accepted by BMVC2024 ([accepted papers list](https://bmvc2024.org/programme/accepted_papers/))

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

in case stable diffusion 1.4
```bash
python load_save.py --pipeline CompVis/stable-diffusion-v1-4 --save_dir models/sd-14
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
python train.py --concept "Eiffel Tower" --concept_type object --save eiffel-tower --data ds/church --local --text_encoder_path models/sd-14/text_encoder --diffusion_path models/sd-14 --epochs 4
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
python infer.py "a photo of Eiffel Tower." eiffel-tower/epoch-3 --tokenizer_path models/sd-14/tokenizer --unet_path models/sd-14/unet --vae_path models/sd-14/vae
```

or 

```bash
python infer.py "a photo of Eiffel Tower." eiffel-tower/epoch-3 --model_name CompVis/stable-diffusion-v1-4
```

this command use the Stable Diffusion 1.4 except the text encoder.

# Citation
Our paper can be cited as follows

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

or

```
@inproceedings{Fuchi_2024_BMVC,
    author    = {Masane Fuchi and Tomohiro Takagi},
    title     = {Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning},
    booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
    publisher = {BMVA},
    year      = {2024},
    url       = {https://papers.bmvc2024.org/0216.pdf}
}
```
## Acknowledgement
This implementation is based on [Textual Inversion using diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py). 

Baselines are as follows:
- https://github.com/rohitgandikota/erasing
- https://github.com/Con6924/SPM
- https://github.com/rohitgandikota/unified-concept-editing/tree/main
