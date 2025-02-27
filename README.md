---
title: hyzhouMedVersa
app_file: demo_inter.py
sdk: gradio
sdk_version: 4.24.0
---
# MedVersa: A Generalist Learner for Multifaceted Medical Image Interpretation
The model card for our paper [A Generalist Learner for Multifaceted Medical Image Interpretation
](https://arxiv.org/abs/2405.07988).

MedVersa is a compound medical AI system that can coordinate multimodal inputs, orchestrate models and tools for varying tasks, and generate multimodal outputs. 

## Environment
MedVersa is written in [Python](https://www.python.org/). It is recommended to configure/manage your python environment using conda. To do this, you need to install the [miniconda](https://docs.anaconda.com/free/miniconda/index.html) or [anaconda](https://www.anaconda.com/) first.

After installing conda, you need to set up a new conda environment for MedVersa using the provided `environment.yml`:
``` shell
conda env create -f environment.yml
conda activate medversa
```
The above `environment.yml` has been validated on NVIDIA A100 GPUs. If you have more advanced cards, e.g., NVIDIA H100 GPUs, you may need `environment_h100.yml` which supports CUDA 11.8:
``` shell
conda env create -f environment_cu118.yml
conda activate medversa
```

If you encounter an issue of opencv, you may need to reinstall opencv-python:
``` shell
pip install opencv-contrib-python
```

If you meet a problem of `incompatible torchvision version`, try the following: 
``` shell
pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Inference
``` python
from utils import *
from torch import cuda

# ---  Launch Model ---
device = 'cuda' if cuda.is_available() else 'cpu'
model_cls = registry.get_model_class('medomni') # medomni is the architecture name :)
model = model_cls.from_pretrained('hyzhou/MedVersa_Internal').to(device).eval()

# --- Define examples ---
examples = [
    [
        ["./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"],
        "Age:30-40.\nGender:F.\nIndication: ___-year-old female with end-stage renal disease not on dialysis presents with dyspnea.  PICC line placement.\nComparison: None.",
        "How would you characterize the findings from <img0>?",
        "cxr",
        "report generation",
    ],
]
# --- Define hyperparams ---
num_beams = 1
do_sample = True
min_length = 1
top_p = 0.9
repetition_penalty = 1
length_penalty = 1
temperature = 0.1

# --- Generate a report for a chest X-ray image ---
index = 0
demo_ex = examples[index]
images, context, prompt, modality, task = demo_ex[0], demo_ex[1], demo_ex[2], demo_ex[3], demo_ex[4]
seg_mask_2d, seg_mask_3d, output_text = generate_predictions(model, images, context, prompt, modality, task, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature)
print(output_text)
```
For more details and examples, please refer to `inference.py`.

## Demo
`CUDA_VISIBLE_DEVICES=0 python demo.py --cfg-path medversa.yaml`

## Prompts
More prompts can be found in `medomni/datasets/prompts.json`.