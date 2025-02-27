from flask import Flask, request, jsonify
from utils import *
from utils import load_and_preprocess_dicom, extract_dicom_metadata
from torch import cuda
import os
import json
import base64
import io
from PIL import Image
import numpy as np
from huggingface_hub import HfFolder
import tempfile
import time
import traceback
import zipfile

app = Flask(__name__)

# Global variables
SUPPORTED_MODALITIES = ["cxr", "derm", "ct"]
SUPPORTED_TASKS = ["report generation", "classification", "segmentation"]
model = None
device = None

def initialize_model():
    """Initialize the model on startup."""
    global model, device
    device = 'cuda' if cuda.is_available() else 'cpu'
    model_cls = registry.get_model_class('medomni')
    token = os.environ.get("HF_TOKEN", HfFolder.get_token())
    model = model_cls.from_pretrained('hyzhou/MedVersa_Internal', token=token).to(device).eval()
    return model

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

def decode_image(base64_string, file_extension):
    """Decode a base64 string to an image and save it to a temp file."""
    image_data = base64.b64decode(base64_string)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"image.{file_extension}")
    
    if file_extension.lower() in ["jpg", "jpeg", "png"]:
        # For 2D images
        image = Image.open(io.BytesIO(image_data))
        image.save(file_path)
    elif file_extension.lower() == "nii.gz":
        # For 3D images (NIfTI)
        with open(file_path, "wb") as f:
            f.write(image_data)
    else:
        raise ValueError(f"Unsupported image format: {file_extension}")
    
    return file_path

def encode_mask(mask):
    """Encode a segmentation mask to base64."""
    if mask is None:
        return None
        
    # Convert numpy array to bytes
    bytes_io = io.BytesIO()
    np.save(bytes_io, mask)
    bytes_io.seek(0)
    
    # Encode to base64
    return base64.b64encode(bytes_io.read()).decode('utf-8')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    start_time = time.time()
    
    try:
        data = request.json
        
        # Validate request data
        required_fields = ['images', 'modality', 'task']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        if not isinstance(data['images'], list) or len(data['images']) == 0:
            return jsonify({"error": "Images field must be a non-empty list"}), 400
            
        # Check if modality is supported
        modality = data['modality']
        if modality not in SUPPORTED_MODALITIES:
            return jsonify({"error": f"Unsupported modality: {modality}. Supported modalities are {SUPPORTED_MODALITIES}"}), 400
            
        task = data['task']
        
        # Process images
        processed_images = []
        metadata_context = ""
        
        for img_data in data['images']:
            if 'data' in img_data:
                # Base64 encoded image
                try:
                    image_format = img_data.get('format', 'jpg').lower()
                    is_dicom_series = img_data.get('is_series', False)
                    
                    # Handle different image formats
                    if image_format in ['jpg', 'jpeg', 'png']:
                        # Regular 2D image processing
                        image_bytes = base64.b64decode(img_data['data'])
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        processed_image = load_and_preprocess_image(image)
                    elif image_format == 'dcm':
                        # DICOM processing
                        dicom_bytes = base64.b64decode(img_data['data'])
                        
                        if is_dicom_series:
                            # For 3D volumes, save to temp directory first
                            temp_dir = tempfile.mkdtemp()
                            
                            # Save and extract zip file to temp directory
                            zip_path = os.path.join(temp_dir, "series.zip")
                            with open(zip_path, 'wb') as f:
                                f.write(dicom_bytes)
                                
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_dir)
                            
                            # Process the DICOM series
                            processed_image = load_and_preprocess_dicom(temp_dir, is_series=True)
                            
                            # Extract metadata from first DICOM in series
                            dicom_files = [f for f in os.listdir(temp_dir) if f.endswith('.dcm')]
                            if dicom_files:
                                metadata = extract_dicom_metadata(os.path.join(temp_dir, dicom_files[0]))
                                if metadata:
                                    metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                                    metadata_context += f"Patient information: {metadata_str}. "
                        else:
                            # Single DICOM file
                            # Save to temp file first
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
                            temp_file.write(dicom_bytes)
                            temp_file.close()
                            
                            # Process the DICOM file
                            processed_image = load_and_preprocess_dicom(temp_file.name, is_series=False)
                            
                            # Extract metadata
                            metadata = extract_dicom_metadata(temp_file.name)
                            if metadata:
                                metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                                metadata_context += f"Patient information: {metadata_str}. "
                            
                            # Clean up
                            os.unlink(temp_file.name)
                    else:
                        return jsonify({"error": f"Unsupported image format: {image_format}"}), 400
                        
                    processed_images.append(processed_image)
                except Exception as e:
                    error_msg = f"Error processing image: {str(e)}"
                    print(traceback.format_exc())
                    return jsonify({"error": error_msg}), 400
            elif 'path' in img_data:
                # Local file path (for testing)
                try:
                    image_path = img_data['path']
                    processed_image = read_image(image_path)
                    
                    # If it's a DICOM file, extract metadata
                    if image_path.endswith('.dcm'):
                        metadata = extract_dicom_metadata(image_path)
                        if metadata:
                            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                            metadata_context += f"Patient information: {metadata_str}. "
                    
                    processed_images.append(processed_image)
                except Exception as e:
                    error_msg = f"Error reading image from path: {str(e)}"
                    print(traceback.format_exc())
                    return jsonify({"error": error_msg}), 400
            else:
                return jsonify({"error": "Each image must have either 'data' or 'path' field"}), 400
        
        # Add metadata context to the request context if available
        if metadata_context and 'context' in data:
            data['context'] = metadata_context + data['context']
        elif metadata_context:
            data['context'] = metadata_context
            
        # Extract parameters
        context = data.get("context", "")
        prompt = data.get("prompt", "")
        
        # Hyperparameters
        hyperparams = data.get("hyperparams", {})
        num_beams = hyperparams.get("num_beams", 1)
        do_sample = hyperparams.get("do_sample", True)
        min_length = hyperparams.get("min_length", 1)
        top_p = hyperparams.get("top_p", 0.9)
        repetition_penalty = hyperparams.get("repetition_penalty", 1)
        length_penalty = hyperparams.get("length_penalty", 1)
        temperature = hyperparams.get("temperature", 0.1)
        
        # Run prediction
        seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
            model, processed_images, context, prompt, modality, task,
            num_beams, do_sample, min_length, top_p, repetition_penalty,
            length_penalty, temperature, device
        )
        
        # Prepare response
        response = {
            "text": output_text,
            "processing_time": time.time() - start_time
        }
        
        # Add segmentation masks if available
        if task == "segmentation":
            if seg_mask_2d is not None:
                masks_2d = [encode_mask(mask) for mask in seg_mask_2d]
                response["segmentation_2d"] = masks_2d
                
            if seg_mask_3d is not None:
                masks_3d = [encode_mask(mask) for mask in seg_mask_3d]
                response["segmentation_3d"] = masks_3d
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        "name": "MedVersa_Internal",
        "supported_modalities": SUPPORTED_MODALITIES,
        "supported_tasks": SUPPORTED_TASKS,
        "device": device,
    })

def generate_predictions(model, images, context, prompt, modality, task, 
                        num_beams, do_sample, min_length, top_p, 
                        repetition_penalty, length_penalty, temperature, device):
    """
    Generate predictions using the MedVersa model
    
    Parameters:
    model: The loaded MedVersa model
    images: List of preprocessed image tensors
    context: Additional context for the prediction
    prompt: Custom prompt override
    modality: Image modality (cxr, derm, ct)
    task: Prediction task (report generation, segmentation, etc.)
    Various hyperparameters for prediction generation
    device: The device to run predictions on
    
    Returns:
    tuple: (segmentation_mask_2d, segmentation_mask_3d, output_text)
    """
    try:
        outputs = model.predict(
            images=images,
            modality=modality,
            task=task,
            prompts=[prompt] if prompt else None,
            context=context,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        
        # Extract predictions
        segmentation_mask_2d = outputs.get("segmentation_mask_2d", None)
        segmentation_mask_3d = outputs.get("segmentation_mask_3d", None)
        output_text = outputs.get("text", "")
        
        return segmentation_mask_2d, segmentation_mask_3d, output_text
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    print(f"Model loaded on device: {device}")
    
    # Start server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port) 