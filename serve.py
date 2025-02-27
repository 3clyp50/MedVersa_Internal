import os
import sys
import time
import json
import base64
import io
import zipfile
import tempfile
import traceback
import numpy as np
from flask import Flask, request, jsonify
from waitress import serve
import torch
from PIL import Image
from huggingface_hub import HfFolder
import torchio as tio

# Import everything needed from utils instead of non-existent model.py
from utils import (
    extract_dicom_metadata, 
    load_and_preprocess_dicom, 
    registry,
    generate,
    generate_predictions as utils_generate_predictions
)

# Remove duplicate imports
from torch import cuda

app = Flask(__name__)

# Global variables
SUPPORTED_MODALITIES = ["cxr", "derm", "ct"]
SUPPORTED_TASKS = ["report generation", "classification", "segmentation"]
model = None
device = None

def display_splash_screen():
    """Display ASCII splash screen at startup."""
    splash = '''
 ____           _ ____            __   __
|  _ \ __ _  __| / ___| _   _ ___ \ \ / /
| |_) / _` |/ _` \___ \  | | / __| \ V / 
|  _ < (_| | (_| |___) | |_| \__ \ / . \ 
|_| \_\__,_|\__,_|____/ \__, |___/__/ \_\
                        |___/
'''
    print(splash)
    print("MedVersa API Server - Starting up...")
    print("-" * 50)

def initialize_model():
    """Initialize the model on startup."""
    global model, device
    
    print("Initializing MedVersa model...")
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Get the model class from registry
        model_cls = registry.get_model_class('medomni')
        
        # Get token from environment or from HuggingFace cache
        token = os.environ.get("HF_TOKEN", HfFolder.get_token())
        
        # Load model from pretrained
        model = model_cls.from_pretrained('hyzhou/MedVersa_Internal', token=token).to(device).eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_request(data):
    """Validate the incoming request data."""
    errors = []
    
    # Check required fields
    required_fields = ["images", "modality", "task"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Validate modality
    if data["modality"].lower() not in SUPPORTED_MODALITIES:
        errors.append(f"Unsupported modality: {data['modality']}. Supported modalities: {SUPPORTED_MODALITIES}")
    
    # Validate task
    if data["task"].lower() not in SUPPORTED_TASKS:
        errors.append(f"Unsupported task: {data['task']}. Supported tasks: {SUPPORTED_TASKS}")
    
    # Validate images
    if not isinstance(data["images"], list) or len(data["images"]) == 0:
        errors.append("Images must be a non-empty list")
    
    return len(errors) == 0, errors

def decode_image(base64_string, format_name):
    """Decode a base64 string into an image tensor."""
    import io
    from PIL import Image
    import numpy as np
    import torch
    
    # Decode base64 string
    image_bytes = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    if format_name in ['jpg', 'jpeg', 'png']:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            # Preprocess the image (resize, normalize, etc.)
            # Convert to tensor and return
            return preprocess_image(image)
        except Exception as e:
            raise ValueError(f"Failed to decode {format_name} image: {str(e)}")
    else:
        raise ValueError(f"Unsupported image format: {format_name}")

def preprocess_image(image):
    """Preprocess a PIL image for the model."""
    # Implement your image preprocessing here
    # For example:
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def process_dicom_data(base64_string):
    """Process DICOM data from a base64 encoded string.
    
    This handles both zip files of DICOM series and single DICOM volume files.
    Returns a list of tensors.
    """
    import io
    import zipfile
    import tempfile
    import os
    import pydicom
    
    # Decode base64 string
    dicom_bytes = base64.b64decode(base64_string)
    processed_tensors = []
    
    try:
        # Try to open as a zip file first
        zip_buffer = io.BytesIO(dicom_bytes)
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                # This is a zip file of multiple DICOM files
                temp_dir = tempfile.mkdtemp()
                try:
                    zip_ref.extractall(temp_dir)
                    print(f"Extracted {len(zip_ref.namelist())} files to temporary directory")
                    
                    # Process the DICOM series directory
                    tensors = load_and_preprocess_dicom_series(temp_dir)
                    processed_tensors.extend(tensors)
                finally:
                    # Clean up temp directory
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except zipfile.BadZipFile:
            # Not a zip file, handle as a single DICOM volume file
            print("Not a zip file, treating as a single DICOM volume")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
            try:
                temp_file.write(dicom_bytes)
                temp_file.close()
                
                # Check if it's a multi-frame DICOM
                ds = pydicom.dcmread(temp_file.name)
                if hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1:
                    print(f"Found multi-frame DICOM with {ds.NumberOfFrames} frames")
                    # Process as volume
                    tensors = load_and_preprocess_dicom_volume(temp_file.name)
                    processed_tensors.extend(tensors)
                else:
                    # Single frame, treat as 2D
                    print("Single frame DICOM, treating as 2D image")
                    tensor = load_and_preprocess_single_dicom(temp_file.name)
                    processed_tensors.append(tensor)
            finally:
                # Clean up temp file
                os.unlink(temp_file.name)
    except Exception as e:
        import traceback
        print(f"Error in process_dicom_data: {str(e)}")
        print(traceback.format_exc())
        raise
    
    print(f"Processed DICOM data into {len(processed_tensors)} tensor(s)")
    return processed_tensors

def process_single_dicom(base64_string):
    """Process a single 2D DICOM file from base64 string."""
    import tempfile
    import os
    import pydicom
    
    # Decode base64 string
    dicom_bytes = base64.b64decode(base64_string)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
    try:
        temp_file.write(dicom_bytes)
        temp_file.close()
        
        # Process the DICOM file
        return load_and_preprocess_single_dicom(temp_file.name)
    finally:
        # Clean up temp file
        os.unlink(temp_file.name)

def load_and_preprocess_single_dicom(file_path):
    """Load and preprocess a single 2D DICOM file."""
    import pydicom
    import numpy as np
    from PIL import Image
    
    # Read DICOM file
    dicom = pydicom.dcmread(file_path)
    
    # Extract pixel array
    try:
        pixel_array = dicom.pixel_array
    except Exception as e:
        raise ValueError(f"Failed to extract pixel array from DICOM: {str(e)}")
    
    # Convert to appropriate format for model
    if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
        # Already RGB
        image_array = pixel_array
    else:
        # Convert to RGB
        if len(pixel_array.shape) == 2:
            # Single channel - convert to 3-channel grayscale
            image_array = np.stack([pixel_array] * 3, axis=2)
        else:
            raise ValueError(f"Unexpected DICOM pixel_array shape: {pixel_array.shape}")
    
    # Normalize to 0-255 range if needed
    if np.max(image_array) > 255:
        image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
    
    # Convert to PIL Image and preprocess
    pil_image = Image.fromarray(image_array.astype(np.uint8))
    return preprocess_image(pil_image)

def load_and_preprocess_dicom_series(directory_path):
    """Load and preprocess a series of DICOM files in a directory."""
    import os
    import pydicom
    import numpy as np
    import torch
    from PIL import Image
    
    # Find all DICOM files
    dicom_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in directory: {directory_path}")
    
    print(f"Processing {len(dicom_files)} DICOM files")
    
    # Sort files (if they represent a series, this is important)
    try:
        # Try to sort by InstanceNumber if available
        dicom_files = sorted(dicom_files, key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
    except:
        # Fall back to simple filename sort
        dicom_files = sorted(dicom_files)
    
    # Process each file
    processed_tensors = []
    for dicom_file in dicom_files:
        try:
            tensor = load_and_preprocess_single_dicom(dicom_file)
            processed_tensors.append(tensor)
        except Exception as e:
            print(f"Warning: Failed to process DICOM file {dicom_file}: {str(e)}")
    
    if not processed_tensors:
        raise ValueError("Failed to process any DICOM files in the series")
    
    return processed_tensors

def load_and_preprocess_dicom_volume(file_path):
    """Load and preprocess a multi-frame DICOM volume file."""
    import pydicom
    import numpy as np
    import torch
    from PIL import Image
    
    # Read DICOM file
    dicom = pydicom.dcmread(file_path)
    
    if not hasattr(dicom, 'NumberOfFrames') or int(dicom.NumberOfFrames) <= 1:
        raise ValueError("Not a multi-frame DICOM volume")
    
    # Extract pixel array - this should be 3D for a volume
    try:
        pixel_array = dicom.pixel_array
    except Exception as e:
        raise ValueError(f"Failed to extract pixel array from DICOM volume: {str(e)}")
    
    # Process each frame
    processed_tensors = []
    for frame_idx in range(pixel_array.shape[0]):
        frame = pixel_array[frame_idx]
        
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            # Single channel - convert to 3-channel grayscale
            frame_rgb = np.stack([frame] * 3, axis=2)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # Already RGB
            frame_rgb = frame
        else:
            raise ValueError(f"Unexpected DICOM frame shape: {frame.shape}")
        
        # Normalize to 0-255 range if needed
        if np.max(frame_rgb) > 255:
            frame_rgb = (frame_rgb / np.max(frame_rgb) * 255).astype(np.uint8)
        
        # Convert to PIL Image and preprocess
        pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
        tensor = preprocess_image(pil_image)
        processed_tensors.append(tensor)
    
    return processed_tensors

def encode_segmentation_mask(masks):
    """Encode segmentation mask(s) as base64 string."""
    import base64
    import io
    import numpy as np
    
    if isinstance(masks, list):
        # Convert list of masks
        return [encode_mask(mask) for mask in masks]
    else:
        # Single mask
        return encode_mask(masks)

def encode_mask(mask):
    """Encode a single numpy array mask as base64 string."""
    import base64
    import io
    import numpy as np
    
    # Convert torch tensor to numpy if needed
    if hasattr(mask, 'detach') and hasattr(mask, 'cpu') and hasattr(mask, 'numpy'):
        mask = mask.detach().cpu().numpy()
    
    # Save the numpy array to a bytes buffer
    buffer = io.BytesIO()
    np.save(buffer, mask)
    buffer.seek(0)
    
    # Encode as base64
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Parse JSON request
        print("Received prediction request")
        request_data = request.json
        
        # Extract parameters
        images = request_data.get("images", [])
        
        # Ensure we have at least one image
        if not images:
            print("Error: No images provided in request")
            return jsonify({"error": "No images provided"}), 400
            
        # Convert images
        processed_images = []
        
        print(f"Processing {len(images)} images...")
        for img_data in images:
            try:
                # Get image data
                data = img_data.get("data", "")
                img_format = img_data.get("format", "").lower()
                is_series = img_data.get("is_series", False)
                
                if not data:
                    return jsonify({"error": "Empty image data"}), 400
                
                if img_format == "dcm":
                    print(f"Processing DICOM data (is_series={is_series})")
                    if is_series:
                        # Handle DICOM series (zip file of DICOM files or volume file)
                        try:
                            processed = process_dicom_data(data)
                            print(f"Processed DICOM data, got {len(processed)} tensor(s)")
                            processed_images.extend(processed)
                        except Exception as dcm_err:
                            import traceback
                            err_traceback = traceback.format_exc()
                            print(f"Error processing DICOM data: {str(dcm_err)}")
                            print(err_traceback)
                            return jsonify({
                                "error": f"Error processing DICOM series: {str(dcm_err)}",
                                "traceback": err_traceback
                            }), 500
                    else:
                        # Handle single DICOM image
                        try:
                            processed = process_single_dicom(data)
                            processed_images.append(processed)
                        except Exception as dcm_err:
                            import traceback
                            err_traceback = traceback.format_exc()
                            print(f"Error processing single DICOM: {str(dcm_err)}")
                            print(err_traceback)
                            return jsonify({
                                "error": f"Error processing single DICOM: {str(dcm_err)}",
                                "traceback": err_traceback
                            }), 500
                else:
                    # Handle regular image formats
                    try:
                        img = decode_image(data, img_format)
                        processed_images.append(img)
                    except Exception as img_err:
                        import traceback
                        err_traceback = traceback.format_exc()
                        print(f"Error decoding image: {str(img_err)}")
                        print(err_traceback)
                        return jsonify({
                            "error": f"Error decoding image: {str(img_err)}",
                            "traceback": err_traceback
                        }), 500
            
            except Exception as process_err:
                import traceback
                err_traceback = traceback.format_exc()
                print(f"Error processing image: {str(process_err)}")
                print(err_traceback)
                return jsonify({
                    "error": f"Error processing image: {str(process_err)}",
                    "traceback": err_traceback
                }), 500
        
        print(f"Successfully processed {len(processed_images)} images")
        
        # Extract other parameters
        modality = request_data.get("modality", "")
        task = request_data.get("task", "")
        prompt = request_data.get("prompt", "")
        context = request_data.get("context", "")
        hyperparams = request_data.get("hyperparams", {})
        
        # Validate required parameters
        if not modality:
            return jsonify({"error": "Missing required parameter: modality"}), 400
            
        if not task:
            return jsonify({"error": "Missing required parameter: task"}), 400
            
        # Extract hyperparameters
        num_beams = hyperparams.get("num_beams", 1)
        do_sample = hyperparams.get("do_sample", True)
        min_length = hyperparams.get("min_length", 0)
        top_p = hyperparams.get("top_p", 0.9)
        repetition_penalty = hyperparams.get("repetition_penalty", 1.0)
        length_penalty = hyperparams.get("length_penalty", 1.0)
        temperature = hyperparams.get("temperature", 1.0)
        
        print(f"Running prediction - Modality: {modality}, Task: {task}")
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        start_time = time.time()
        
        # Run prediction
        try:
            result = generate_predictions(
                model, 
                processed_images, 
                context=context,
                prompt=prompt, 
                modality=modality,
                task=task,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                device=device
            )
            
            processing_time = time.time() - start_time
            print(f"Prediction completed in {processing_time:.2f} seconds")
            
            # Process the result
            seg_mask_2d, seg_mask_3d, output_text = result
            
            # Prepare response
            response = {"text": output_text, "processing_time": processing_time}
            
            # Process segmentation masks if present
            if task == "segmentation":
                if seg_mask_2d is not None:
                    masks_2d = encode_segmentation_mask(seg_mask_2d)
                    response["segmentation_2d"] = masks_2d
                    response["segmentation_mask"] = masks_2d
                    response["is_3d"] = False
                    
                if seg_mask_3d is not None:
                    masks_3d = encode_segmentation_mask(seg_mask_3d)
                    response["segmentation_3d"] = masks_3d
                    response["segmentation_mask"] = masks_3d
                    response["is_3d"] = True
            
            # Return the result
            return jsonify(response)
            
        except Exception as pred_err:
            import traceback
            err_traceback = traceback.format_exc()
            print(f"Error in prediction: {str(pred_err)}")
            print(err_traceback)
            return jsonify({
                "error": f"Error in prediction: {str(pred_err)}",
                "traceback": err_traceback
            }), 500
        
    except Exception as e:
        import traceback
        err_traceback = traceback.format_exc()
        print(f"Unexpected error: {str(e)}")
        print(err_traceback)
        return jsonify({
            "error": f"Unexpected error: {str(e)}",
            "traceback": err_traceback
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        "name": "MedVersa_Internal",
        "supported_modalities": SUPPORTED_MODALITIES,
        "supported_tasks": SUPPORTED_TASKS,
        "device": device,
    })

def generate_predictions(
    model, 
    images, 
    context="",
    prompt="", 
    modality="ct",
    task="segmentation",
    num_beams=1,
    do_sample=True,
    min_length=0,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    temperature=1.0,
    device="cuda"
):
    """Generate predictions using the MedVersa model."""
    import torch
    import torchio as tio
    
    print("Generating predictions...")
    print(f"Modality: {modality}, Task: {task}")
    print(f"Number of input tensors: {len(images)}")
    
    try:
        # Check if images are already processed tensors or file paths
        if isinstance(images[0], torch.Tensor):
            # For processed tensors, check if we have a 3D volume
            if len(images) > 1 and modality.lower() in ['ct', 'cxr']:
                print("Processing multiple slices as 3D volume...")
                # Print shape of first tensor to understand input format
                print(f"Input tensor shape: {images[0].shape}")
                
                # Stack slices into a volume tensor
                volume = torch.stack(images, dim=0)
                print(f"After stacking shape: {volume.shape}")
                
                # If we have a 5D tensor, remove the extra dimension
                if len(volume.shape) == 5:
                    volume = volume.squeeze(1)
                    print(f"After squeezing shape: {volume.shape}")
                
                # Now we should have [N, C, H, W]
                # First permute to [C, H, W, N] for torchio normalization
                volume = volume.permute(1, 2, 3, 0)
                print(f"After permute for normalization: {volume.shape}")
                
                # Apply normalization (expects [C, X, Y, Z])
                transform = tio.Compose([
                    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                ])
                volume = transform(volume).type(torch.bfloat16)
                print(f"After normalization shape: {volume.shape}")
                
                # Now reshape to [B, S, C, H, W] for the model
                volume = volume.permute(3, 0, 1, 2)  # [C, H, W, N] -> [N, C, H, W]
                volume = volume.unsqueeze(0)  # Add batch dimension: [B, S, C, H, W]
                print(f"After reshaping for model: {volume.shape}")
                
                image_tensors = [volume.to(device)]
                already_batched = True  # Flag to indicate tensor is already in correct shape
            else:
                # Single image or non-volume modality
                image_tensors = [img.to(device) for img in images]
                already_batched = False
        else:
            # File paths, need to process
            image_tensors = [read_image(img).to(device) for img in images]
            already_batched = False
            
        if modality == 'ct':
            time.sleep(2)
        else:
            time.sleep(1)
        
        # For volumes, we don't need to concatenate further as it's already a single tensor
        if len(image_tensors) == 1:
            image_tensor = image_tensors[0]
        else:
            image_tensor = torch.cat(image_tensors)
            already_batched = False  # Reset flag as we concatenated
        
        print(f"Final tensor shape for model: {image_tensor.shape}")
        
        with torch.autocast(device):
            with torch.no_grad():
                # We need to handle both file paths and tensor inputs for the generate function
                image_paths = images if not isinstance(images[0], torch.Tensor) else ["dummy_path"] * len(images)
                # Pass the already_batched flag to generate
                generated_image, seg_mask_2d, seg_mask_3d, output_text = generate(
                    model=model,
                    image_path=image_paths,
                    image=image_tensor,
                    context=context,
                    modal=modality.lower(),
                    task=task,
                    num_imgs=1 if already_batched else len(images),
                    prompt=prompt,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature,
                    already_batched=already_batched  # New parameter
                )
        
        print("Prediction completed successfully")
        return seg_mask_2d, seg_mask_3d, output_text
        
    except Exception as e:
        import traceback
        print(f"Error in generate_predictions: {str(e)}")
        print(traceback.format_exc())
        raise

if __name__ == '__main__':
    # Display splash screen
    display_splash_screen()
    
    # Initialize model on startup
    initialize_model()
    print(f"Model loaded on device: {device}")
    
    # Start server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port) 