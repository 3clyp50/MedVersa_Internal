import gradio as gr
import argparse
import torch
from torch import cuda
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import skimage.morphology, skimage.io
import cv2
import numpy as np
import random
from transformers import StoppingCriteria, StoppingCriteriaList
from copy import deepcopy
from medomni.common.config import Config
from medomni.common.dist_utils import get_rank
from medomni.common.registry import registry
import torchio as tio
import nibabel as nib
from scipy import ndimage, misc
import time
import ipdb

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

device = 'cuda' if cuda.is_available() else 'cpu'
# Launch model
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_pretrained('hyzhou/MedVersa').to(device).eval()
global global_images
global_images = None

def seg_2d_process(image_path, pred_mask, img_size=224):
    image = cv2.imread(image_path[0])
    if pred_mask.sum() != 0:
        labels = skimage.morphology.label(pred_mask)
        labelCount = np.bincount(labels.ravel())
        largest_label = np.argmax(labelCount[1:]) + 1
        pred_mask[labels != largest_label] = 0
        pred_mask[labels == largest_label] = 255
        pred_mask = pred_mask.astype(np.uint8)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contours:
            contours = np.vstack(contours)
            binary_array = np.zeros((img_size, img_size))
            binary_array = cv2.drawContours(binary_array, contours, -1, 255, thickness=cv2.FILLED) 
            binary_array = cv2.resize(binary_array, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST) / 255
            image = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]
            mask = [binary_array]
        else:
            image = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]
            mask = [np.zeros((image.shape[1], image.shape[0]))]
    else:
        mask = [np.zeros((image.shape[1], image.shape[0]))]
        image = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]    
    # output_image = cv2.drawContours(binary_array, contours, -1, (110, 0, 255), 2) 
    # output_image_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    return image, mask

def seg_3d_process(image_path, seg_mask):
    img  = nib.load(image_path[0]).get_fdata()
    image = window_scan(img).transpose(2,0,1).astype(np.uint8)
    if seg_mask.sum() != 0:
        seg_mask = resize_back_volume_abd(seg_mask, image.shape).astype(np.uint8)
        image_slices = []
        contour_slices = []
        for i in range(seg_mask.shape[0]):
            slice_img = np.fliplr(np.rot90(image[i]))
            slice_mask = np.fliplr(np.rot90(seg_mask[i]))
            contours, _ = cv2.findContours(slice_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            image_slices.append(Image.fromarray(slice_img))
            if contours:
                binary_array = np.zeros(seg_mask.shape[1:])
                binary_array = cv2.drawContours(binary_array, contours, -1, 255, thickness=cv2.FILLED) / 255
                binary_array = cv2.resize(binary_array, slice_img.shape, interpolation = cv2.INTER_NEAREST)
                contour_slices.append(binary_array)
            else:
                contour_slices.append(np.zeros_like(slice_img))
    else:
        image_slices = []
        contour_slices = []
        slice_img = np.fliplr(np.rot90(image[i]))
        image_slices.append(Image.fromarray(slice_img))
        contour_slices.append(np.zeros_like(slice_img))

    return image_slices, contour_slices

def det_2d_process(image_path, box):
    image_slices = []
    image = cv2.imread(image_path[0])
    if box is not None:
        hi,wd,_ = image.shape
        color = tuple(np.random.random(size=3) * 256)
        x1, y1, x2, y2 = int(box[0]*wd), int(box[1]*hi), int(box[2]*wd), int(box[3]*hi)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
    image_slices.append(Image.fromarray(image))
    return image_slices

def window_scan(scan, window_center=50, window_width=400):
    """
    Apply windowing to a scan.

    Parameters:
    scan (numpy.ndarray): 3D numpy array of the CT scan
    window_center (int): The center of the window
    window_width (int): The width of the window

    Returns:
    numpy.ndarray: Windowed CT scan
    """
    lower_bound = window_center - (window_width // 2)
    upper_bound = window_center + (window_width // 2)
    
    windowed_scan = np.clip(scan, lower_bound, upper_bound)
    windowed_scan = (windowed_scan - lower_bound) / (upper_bound - lower_bound)
    windowed_scan = (windowed_scan * 255).astype(np.uint8)
    
    return windowed_scan

def task_seg_2d(model, preds, hidden_states, image):
    token_mask = preds == model.seg_token_idx_2d  
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    feats = model.model_seg_2d.encoder(image.unsqueeze(0)[:, 0])
    last_feats = feats[-1]
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze()
        seg_states = model.text2seg_2d(target_states).unsqueeze(0)
        last_feats = last_feats + seg_states.unsqueeze(-1).unsqueeze(-1)
        last_feats = model.text2seg_2d_gn(last_feats)
        feats[-1] = last_feats
        seg_feats = model.model_seg_2d.decoder(*feats)
        seg_preds = model.model_seg_2d.segmentation_head(seg_feats)
        seg_probs = F.sigmoid(seg_preds)
        seg_mask = seg_probs.to(torch.float32).cpu().squeeze().numpy() >= 0.5
        return seg_mask
    else:
        return None

def task_seg_3d(model, preds, hidden_states, img_embeds_list):
    new_img_embeds_list = deepcopy(img_embeds_list)
    token_mask = preds == model.seg_token_idx_3d  
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze().unsqueeze(0)
        seg_states = model.text2seg_3d(target_states)
        last_feats = new_img_embeds_list[-1]
        last_feats = last_feats + seg_states.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        last_feats = model.text2seg_3d_gn(last_feats)
        new_img_embeds_list[-1] = last_feats
        seg_preds = model.visual_encoder_3d(encoder_only=False, x_=new_img_embeds_list)
        seg_probs = F.sigmoid(seg_preds)
        seg_mask = seg_probs.to(torch.float32).cpu().squeeze().numpy() >= 0.5
        return seg_mask

def task_det_2d(model, preds, hidden_states):
    token_mask = preds == model.det_token_idx
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze()
        det_states = model.text_det(target_states).detach().cpu()
        return det_states.to(torch.float32).numpy()
    return torch.zeros_like(indices)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def resize_back_volume_abd(img, target_size):
    desired_depth = target_size[0]
    desired_width = target_size[1]
    desired_height = target_size[2]

    current_depth = img.shape[0] # [d, w, h]
    current_width = img.shape[1] 
    current_height = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=0)
    return img

def resize_volume_abd(img):
    img[img<=-200] = -200
    img[img>=300] = 300

    desired_depth = 64
    desired_width = 192
    desired_height = 192

    current_width = img.shape[0] # [w, h, d]
    current_height = img.shape[1]
    current_depth = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=0)
    return img

def load_and_preprocess_image(image):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image).type(torch.bfloat16).unsqueeze(0)
    return image

def load_and_preprocess_volume(image):
    img  = nib.load(image).get_fdata()
    image = torch.from_numpy(resize_volume_abd(img)).permute(2,0,1)
    transform = tio.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    image = transform(image.unsqueeze(0)).type(torch.bfloat16)
    return image

def read_image(image_path):
    if image_path.endswith(('.jpg', '.jpeg', '.png')):
        return load_and_preprocess_image(Image.open(image_path).convert('RGB'))
    elif image_path.endswith('.nii.gz'):
        return load_and_preprocess_volume(image_path)
    else:
        raise ValueError("Unsupported file format")

def generate(image_path, image, context, modal, num_imgs, prompt, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature):
    if (len(context) != 0 and ('report' in prompt or 'finding' in prompt or 'impression' in prompt)) or (len(context) != 0 and modal=='derm' and ('diagnosis' in prompt or 'issue' in prompt or 'problem' in prompt)):
        prompt = '<context>' + context + '</context>' + prompt
    if modal == 'ct' and 'segment' in prompt.lower():
        if 'liver' in prompt:
            prompt = 'Segment the liver.'
        if 'spleen' in prompt:
            prompt = 'Segment the spleen.'
        if 'kidney' in prompt:
            prompt = 'Segment the kidney.'
        if 'pancrea' in prompt:
            prompt = 'Segment the pancreas.'
    img_embeds, atts_img, img_embeds_list = model.encode_img(image.unsqueeze(0), [modal])
    placeholder = ['<ImageHere>'] * 9
    prefix = '###Human:' + ''.join([f'<img{i}>' + ''.join(placeholder) + f'</img{i}>' for i in range(num_imgs)])
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, [prefix], [num_imgs])
    prompt += '###Assistant:'
    prompt_tokens = model.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(image.device)
    new_img_embeds, new_atts_img = model.prompt_concat(img_embeds, atts_img, prompt_tokens)
    
    outputs = model.llama_model.generate(
        inputs_embeds=new_img_embeds,
        max_new_tokens=450,
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=[
            torch.tensor([835]).type(torch.bfloat16).to(image.device),
            torch.tensor([2277, 29937]).type(torch.bfloat16).to(image.device)
        ])]),
        num_beams=num_beams,
        do_sample=do_sample,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    
    hidden_states = outputs.hidden_states
    preds = outputs.sequences[0]
    output_image = None
    seg_mask_2d = None
    seg_mask_3d = None
    if sum(preds == model.seg_token_idx_2d):
        seg_mask = task_seg_2d(model, preds, hidden_states, image)
        output_image, seg_mask_2d = seg_2d_process(image_path, seg_mask)
    if sum(preds == model.seg_token_idx_3d):
        seg_mask = task_seg_3d(model, preds, hidden_states, img_embeds_list)
        output_image, seg_mask_3d = seg_3d_process(image_path, seg_mask)
    if sum(preds == model.det_token_idx):
        det_box = task_det_2d(model, preds, hidden_states)
        output_image = det_2d_process(image_path, det_box)
    
    if preds[0] == 0:  # Remove unknown token <unk> at the beginning
        preds = preds[1:]
    if preds[0] == 1:  # Remove start token <s> at the beginning
        preds = preds[1:]
    
    output_text = model.llama_tokenizer.decode(preds, add_special_tokens=False)
    output_text = output_text.split('###')[0].split('Assistant:')[-1].strip()

    if 'mel' in output_text and modal == 'derm':
        output_text = 'The main diagnosis is melanoma.'
    return output_image, seg_mask_2d, seg_mask_3d, output_text

def generate_predictions(images, context, prompt, modality, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature):
    num_imgs = len(images)
    modal = modality.lower()
    image_tensors = [read_image(img).to(device) for img in images]
    if modality == 'ct':
        time.sleep(2)
    else:
        time.sleep(1)
    image_tensor = torch.cat(image_tensors)
    
    with torch.autocast(device):
        with torch.no_grad():
            generated_image, seg_mask_2d, seg_mask_3d, output_text = generate(images, image_tensor, context, modal, num_imgs, prompt, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature)
    
    return generated_image, seg_mask_2d, seg_mask_3d, output_text

my_dict = {}
def gradio_interface(chatbot, images, context, prompt, modality, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature):
    global global_images
    if not images:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        blank_image = Image.fromarray(image)
        snapshot = (blank_image, [])
        global_images = 'none'
        return [(prompt, "At least one image is required to proceed.")], snapshot, gr.update(maximum=0)
    if not prompt or not modality:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        blank_image = Image.fromarray(image)
        snapshot = (blank_image, [])
        global_images = 'none'
        return [(prompt, "Please provide prompt and modality to proceed.")], snapshot, gr.update(maximum=0)

    generated_images, seg_mask_2d, seg_mask_3d, output_text = generate_predictions(images, context, prompt, modality, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature)
    output_images = []
    input_images = [np.asarray(Image.open(img.name).convert('RGB')).astype(np.uint8) if img.name.endswith(('.jpg', '.jpeg', '.png')) else f"{img.name} (3D Volume)" for img in images]
    if generated_images is not None:
        for generated_image in generated_images:
            output_images.append(np.asarray(generated_image).astype(np.uint8)) 
        snapshot = (output_images[0], [])
        if seg_mask_2d is not None:
            snapshot = (output_images[0], [(seg_mask_2d[0], "Mask")])
        if seg_mask_3d is not None:
            snapshot = (output_images[0], [(seg_mask_3d[0], "Mask")])
    else:
        output_images = input_images.copy()
        snapshot = (output_images[0], [])
    
    my_dict['image'] = output_images
    my_dict['mask'] = None
    if seg_mask_2d is not None:
        my_dict['mask'] = seg_mask_2d
    if seg_mask_3d is not None:
        my_dict['mask'] = seg_mask_3d
    
    if global_images != images and (global_images is not None):
        chatbot = []
        chatbot.append((prompt, output_text))
    else:
        chatbot.append((prompt, output_text))
    global_images = images

    return chatbot, snapshot, gr.update(maximum=len(output_images)-1)

def render(x):
    if x > len(my_dict['image'])-1:
        x = len(my_dict['image'])-1
    if x < 0:
        x = 0
    image = my_dict['image'][x]
    if my_dict['mask'] is None:
        return (image,[])
    else:
        mask = my_dict['mask'][x]
        value = (image,[(mask, "Mask")])
        return value

def update_context_visibility(task):
    if task == "report generation" or task == 'classification':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def reset_chatbot():
    return []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # with gr.Row():
    #     gr.Markdown("<link href='https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;700&display=swap' rel='stylesheet'>")
    gr.Markdown("# MedVersa")
    with gr.Row():
        with gr.Column():
            image_input = gr.File(label="Upload Images", file_count="multiple", file_types=["image", "numpy"])
            # task_input = gr.Dropdown(choices=["report generation", "vqa", "localization", "classification"], label="Task")
            context_input = gr.Textbox(label="Context", placeholder="Enter context here...", lines=3, visible=True)
            modality_input = gr.Dropdown(choices=["cxr", "derm", "ct"], label="Modality")
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter prompt here... (images should be referred as <img0>, <img1>, ...)", lines=3)
            submit_button = gr.Button("Generate Predictions")
            with gr.Accordion("Advanced Settings", open=False):
                num_beams = gr.Slider(label="Number of Beams", minimum=1, maximum=10, step=1, value=1)
                do_sample = gr.Checkbox(label="Do Sample", value=True)
                min_length = gr.Slider(label="Minimum Length", minimum=1, maximum=100, step=1, value=1)
                top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, step=0.1, value=0.9)
                repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.1, value=1.0)
                length_penalty = gr.Slider(label="Length Penalty", minimum=1.0, maximum=2.0, step=0.1, value=1.0)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=0.1)
        
        with gr.Column():
            # output_text = gr.Textbox(label="Generated Text", lines=10, elem_classes="output-textbox")
            chatbot = gr.Chatbot(label="Chatbox")
            slider = gr.Slider(minimum=0, maximum=64, value=1, step=1)
            output_image = gr.AnnotatedImage(height=448, label="Images")

    # task_input.change(
    #     fn=update_context_visibility,
    #     inputs=task_input,
    #     outputs=context_input
    # )

    submit_button.click(
        fn=gradio_interface,
        inputs=[chatbot, image_input, context_input, prompt_input, modality_input, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature],
        outputs=[chatbot, output_image, slider]
    )

    slider.change(
        render,
        inputs=[slider],
        outputs=[output_image],
    )

    examples = [
        [
            ["./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"],
            "Age:30-40.\nGender:F.\nIndication: ___-year-old female with end-stage renal disease not on dialysis presents with dyspnea.  PICC line placement.\nComparison: None.",
            "How would you characterize the findings from <img0>?",
            "cxr",
        ],
        [
            ["./demo_ex/79eee504-b1b60ab8-5e8dd843-b6ed87aa-670747b1.png"],
            "Age:70-80.\nGender:F.\nIndication: Respiratory distress.\nComparison: None.",
            "How would you characterize the findings from <img0>?",
            "cxr",
        ],
        [
            ["./demo_ex/f39b05b1-f544e51a-cfe317ca-b66a4aa6-1c1dc22d.png", "./demo_ex/f3fefc29-68544ac8-284b820d-858b5470-f579b982.png"],
            "Age:80-90.\nGender:F.\nIndication: ___-year-old female with history of chest pain.\nComparison: None.",
            "How would you characterize the findings from <img0><img1>?",
            "cxr",
        ],
        [
            ["./demo_ex/1de015eb-891f1b02-f90be378-d6af1e86-df3270c2.png"],
            "Age:40-50.\nGender:M.\nIndication: ___-year-old male with shortness of breath.\nComparison: None.",
            "How would you characterize the findings from <img0>?",
            "cxr",
        ],
        [
            ["./demo_ex/bc25fa99-0d3766cc-7704edb7-5c7a4a63-dc65480a.png"],
            "Age:40-50.\nGender:F.\nIndication: History: ___F with tachyacrdia cough doe  // infilatrate\nComparison: None.",
            "How would you characterize the findings from <img0>?",
            "cxr",
        ],
        [
            ["./demo_ex/ISIC_0032258.jpg"],
            "Age:70.\nGender:female.\nLocation:back.",
            "What is primary diagnosis?",
            "derm",
        ],
        [
            ["./demo_ex/Case_01013_0000.nii.gz"],
            "",
            "Segment the liver.",
            "ct",
        ],
        [
            ["./demo_ex/Case_00840_0000.nii.gz"],
            "",
            "Segment the liver.",
            "ct",
        ],
    ]

    gr.Examples(examples, inputs=[image_input, context_input, prompt_input, modality_input])

# Run Gradio app
demo.launch(share=True)