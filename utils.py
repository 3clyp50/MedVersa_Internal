import argparse
import torch
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
import os
import pydicom
try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK not installed. 3D DICOM series support may be limited.")

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
        seg_mask = seg_probs.to(dtype=torch.float32).cpu().squeeze().numpy() >= 0.5
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
        seg_mask = seg_probs.to(dtype=torch.float32).cpu().squeeze().numpy() >= 0.5
        return seg_mask

def task_det_2d(model, preds, hidden_states):
    token_mask = preds == model.det_token_idx
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze()
        det_states = model.text_det(target_states).detach().cpu()
        return det_states.numpy()
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

def load_and_preprocess_dicom(dicom_path, is_series=False):
    """
    Load and preprocess a DICOM file or series
    
    Parameters:
    dicom_path (str): Path to a DICOM file or directory containing a DICOM series
    is_series (bool): Whether this is a series of DICOM files (3D volume)
    
    Returns:
    torch.Tensor: Preprocessed image tensor ready for model input
    """
    if is_series:
        return load_and_preprocess_dicom_series(dicom_path)
    else:
        return load_and_preprocess_dicom_image(dicom_path)

def load_and_preprocess_dicom_image(dicom_path):
    """
    Load and preprocess a single 2D DICOM image
    
    Parameters:
    dicom_path (str): Path to a DICOM file
    
    Returns:
    torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Read DICOM file
    try:
        dicom = pydicom.dcmread(dicom_path)
    except Exception as e:
        raise ValueError(f"Error reading DICOM file: {str(e)}")
    
    # Extract pixel data
    pixel_array = dicom.pixel_array
    
    # Apply windowing for CT/MRI if needed and window center/width are available
    if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
        window_center = dicom.WindowCenter
        window_width = dicom.WindowWidth
        
        # Handle multiple window settings (use first one if list)
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]
            
        # Apply windowing
        lower_bound = window_center - window_width // 2
        upper_bound = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, lower_bound, upper_bound)
        pixel_array = (pixel_array - lower_bound) / (window_width)
        pixel_array = (pixel_array * 255).astype(np.uint8)
    
    # Check bit depth and rescale if needed
    if dicom.BitsStored > 8:
        # Normalize to 8 bits if no windowing applied earlier
        if not (hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth')):
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:  # Avoid division by zero
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
    
    # Convert to RGB if grayscale
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=2)
    
    # Convert to PIL Image
    image = Image.fromarray(pixel_array)
    
    # Process using the existing image processing pipeline
    return load_and_preprocess_image(image)

def load_and_preprocess_dicom_series(dicom_dir):
    """
    Load and preprocess a DICOM series (3D volume)
    
    Parameters:
    dicom_dir (str): Path to a directory containing a DICOM series
    
    Returns:
    torch.Tensor: Preprocessed volume tensor ready for model input
    """
    try:
        # Use SimpleITK to load DICOM series if available
        if 'sitk' in globals():
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
            if not dicom_names:
                raise ValueError(f"No DICOM series found in {dicom_dir}")
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            volume = sitk.GetArrayFromImage(image)
        else:
            # Fallback to pydicom if SimpleITK not available
            dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) 
                          if os.path.isfile(os.path.join(dicom_dir, f)) and f.endswith('.dcm')]
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {dicom_dir}")
            
            # Sort by instance number if available
            slices = [pydicom.dcmread(f) for f in dicom_files]
            slices.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
            
            # Extract pixel data
            volume = np.stack([s.pixel_array for s in slices], axis=0)
            
            # Apply windowing if available
            if hasattr(slices[0], 'WindowCenter') and hasattr(slices[0], 'WindowWidth'):
                window_center = slices[0].WindowCenter
                window_width = slices[0].WindowWidth
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]
                volume = window_scan(volume, window_center, window_width)
        
        # Process the volume with the same functions used for NIfTI
        volume = resize_volume_abd(volume)
        volume_tensor = torch.from_numpy(volume).permute(2, 0, 1)
        transform = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])
        volume_tensor = transform(volume_tensor.unsqueeze(0)).type(torch.bfloat16)
        return volume_tensor
    
    except Exception as e:
        raise ValueError(f"Error processing DICOM series: {str(e)}")

def extract_dicom_metadata(dicom_path):
    """
    Extract relevant metadata from a DICOM file that can be used for context
    
    Parameters:
    dicom_path (str): Path to a DICOM file
    
    Returns:
    dict: Dictionary containing metadata
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        metadata = {}
        
        # Extract patient information
        if hasattr(dicom, 'PatientAge'):
            metadata['Age'] = dicom.PatientAge
        
        if hasattr(dicom, 'PatientSex'):
            sex = dicom.PatientSex
            if sex == 'M':
                metadata['Gender'] = 'Male'
            elif sex == 'F':
                metadata['Gender'] = 'Female'
            else:
                metadata['Gender'] = sex
        
        # Extract study information
        if hasattr(dicom, 'StudyDescription'):
            metadata['Study'] = dicom.StudyDescription
            
        if hasattr(dicom, 'Modality'):
            metadata['Modality'] = dicom.Modality
                
        if hasattr(dicom, 'BodyPartExamined'):
            metadata['BodyPart'] = dicom.BodyPartExamined
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not extract DICOM metadata: {str(e)}")
        return {}

def read_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check if file is empty
    if os.path.getsize(image_path) == 0:
        raise ValueError(f"Image file is empty: {image_path}")
        
    if image_path.endswith(('.jpg', '.jpeg', '.png')):
        return load_and_preprocess_image(Image.open(image_path).convert('RGB'))
    elif image_path.endswith('.nii.gz'):
        return load_and_preprocess_volume(image_path)
    elif image_path.endswith('.dcm'):
        return load_and_preprocess_dicom(image_path, is_series=False)
    elif os.path.isdir(image_path) and any(f.endswith('.dcm') for f in os.listdir(image_path)):
        return load_and_preprocess_dicom(image_path, is_series=True)
    else:
        raise ValueError("Unsupported file format")

def generate(model, image_path, image, context, modal, task, num_imgs, prompt, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature):
    if task == 'report generation' or task == 'classification':
        prompt = '<context>' + context + '</context>' + prompt
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
    
    # Check for segmentation tokens in the predictions
    if sum(preds == model.seg_token_idx_2d):
        seg_mask = task_seg_2d(model, preds, hidden_states, image)
        if seg_mask is not None:
            output_image, seg_mask_2d = seg_2d_process(image_path, seg_mask)
    
    if sum(preds == model.seg_token_idx_3d):
        seg_mask = task_seg_3d(model, preds, hidden_states, img_embeds_list)
        if seg_mask is not None:
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

def generate_predictions(model, images, context, prompt, modality, task, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device):
    num_imgs = len(images)
    modal = modality.lower()
    
    # Add error handling for image loading
    try:
        image_tensors = [read_image(img).to(device) for img in images]
        if modality == 'ct':
            time.sleep(2)
        else:
            time.sleep(1)
        image_tensor = torch.cat(image_tensors)
        
        with torch.autocast(device):
            with torch.no_grad():
                generated_image, seg_mask_2d, seg_mask_3d, output_text = generate(model, images, image_tensor, context, modal, task, num_imgs, prompt, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature)
        
        return seg_mask_2d, seg_mask_3d, output_text
    except Exception as e:
        print(f"Error processing images: {e}")
        # Return empty results in case of error
        return None, None, f"Error: Could not process the image(s). {str(e)}"
