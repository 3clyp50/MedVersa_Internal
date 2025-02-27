from utils import *
from torch import cuda
import os
from huggingface_hub import HfFolder

# Note: This script demonstrates basic inference with MedVersa.
# For production deployment, see serve.py and the accompanying documentation.

# ---  Launch Model ---
device = 'cuda' if cuda.is_available() else 'cpu'
model_cls = registry.get_model_class('medomni') # medomni is the architecture name :)
# Get token from HF_TOKEN environment variable or from the cached token
token = os.environ.get("HF_TOKEN", HfFolder.get_token())
# Use the correct model name
model = model_cls.from_pretrained('hyzhou/MedVersa_Internal', token=token).to(device).eval()

# --- Define examples ---
examples = [
    [
        ["./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"],
        "Age:30-40.\nGender:F.\nIndication: ___-year-old female with end-stage renal disease not on dialysis presents with dyspnea.  PICC line placement.\nComparison: None.",
        "How would you characterize the findings from <img0>?",
        "cxr",
        "report generation",
    ],
    [
        ["./demo_ex/79eee504-b1b60ab8-5e8dd843-b6ed87aa-670747b1.png"],
        "Age:70-80.\nGender:F.\nIndication: Respiratory distress.\nComparison: None.",
        "How would you characterize the findings from <img0>?",
        "cxr",
        "report generation",
    ],
    [
        ["./demo_ex/f39b05b1-f544e51a-cfe317ca-b66a4aa6-1c1dc22d.png", "./demo_ex/f3fefc29-68544ac8-284b820d-858b5470-f579b982.png"],
        "Age:80-90.\nGender:F.\nIndication: ___-year-old female with history of chest pain.\nComparison: None.",
        "How would you characterize the findings from <img0><img1>?",
        "cxr",
        "report generation",
    ],
    [
        ["./demo_ex/1de015eb-891f1b02-f90be378-d6af1e86-df3270c2.png"],
        "Age:40-50.\nGender:M.\nIndication: ___-year-old male with shortness of breath.\nComparison: None.",
        "How would you characterize the findings from <img0>?",
        "cxr",
        "report generation",
    ],
    [
        ["./demo_ex/bc25fa99-0d3766cc-7704edb7-5c7a4a63-dc65480a.png"],
        "Age:40-50.\nGender:F.\nIndication: History: ___F with tachyacrdia cough doe  // infilatrate\nComparison: None.",
        "How would you characterize the findings from <img0>?",
        "cxr",
        "report generation",
    ],
    [
        ["./demo_ex/79eee504-b1b60ab8-5e8dd843-b6ed87aa-670747b1.png"],
        "",
        "What is the primary diagnosis?",
        "cxr",
        "classification",
    ],
    [
        ["./demo_ex/ISIC_0032258.jpg"],
        "Age:70.\nGender:female.\nLocation:back.",
        "What is primary diagnosis?",
        "derm",
        "classification",
    ],
    [
        ["./demo_ex/ISIC_0032258.jpg"],
        "Age:70.\nGender:female.\nLocation:back.",
        "Segment the lesion.",
        "derm",
        "segmentation",
    ],
    [
        ["./demo_ex/Case_01013_0000.nii.gz"],
        "",
        "Segment the liver.",
        "ct",
        "segmentation",
    ],
    [
        ["./demo_ex/Case_00840_0000.nii.gz"],
        "",
        "Segment the liver.",
        "ct",
        "segmentation",
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
seg_mask_2d, seg_mask_3d, output_text = generate_predictions(model, images, context, prompt, modality, task, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device)
print(output_text)

# --- Segment the lesion in the dermatology image ---
index = 7  # Changed from 6 to 7 to use the segmentation example instead of classification
demo_ex = examples[index]
images, context, prompt, modality, task = demo_ex[0], demo_ex[1], demo_ex[2], demo_ex[3], demo_ex[4]
seg_mask_2d, seg_mask_3d, output_text = generate_predictions(model, images, context, prompt, modality, task, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device)
print(output_text)
if seg_mask_2d is not None:
    print(f"2D Segmentation mask shape: {seg_mask_2d[0].shape}")  # H, W
else:
    print("No 2D segmentation mask was generated.")

# --- Segment the liver in the abdomen CT scan ---
# Check if CT scan files exist before processing
import os

# Try both CT scan examples
for ct_index in [-2, -1]:
    demo_ex = examples[ct_index]
    image_path = demo_ex[0][0]
    
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        print(f"\nProcessing CT scan: {image_path}")
        images, context, prompt, modality, task = demo_ex[0], demo_ex[1], demo_ex[2], demo_ex[3], demo_ex[4]
        seg_mask_2d, seg_mask_3d, output_text = generate_predictions(model, images, context, prompt, modality, task, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device)
        print(output_text)
        if seg_mask_3d is not None:
            print(f"Number of 3D slices: {len(seg_mask_3d)}")
            print(f"3D Segmentation mask shape: {seg_mask_3d[0].shape}")  # H, W
        else:
            print("No 3D segmentation mask was generated.")
        # Successfully processed one CT scan, so break the loop
        break
    else:
        print(f"\nCT scan file not found or empty: {image_path}")

print("\nInference completed.")

