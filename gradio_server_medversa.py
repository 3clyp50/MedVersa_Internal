import gradio as gr
import torch
import os
import sys
import io
import json
import tempfile
import logging
import numpy as np
import asyncio
import atexit
from PIL import Image
import cv2
from pathlib import Path
from huggingface_hub import HfFolder
import pydicom
import zipfile
import nibabel as nib
import ast
import glob
import shutil

# Add MedVersa_Internal to path
repo_root = str(Path(__file__).parent)
sys.path.append(repo_root)

# Set examples directory
examples_dir = os.path.join(repo_root, "demo_ex")

# First import from original utils.py for model-related functions
from utils import registry, generate, generate_predictions, read_image

# Import image processing functions
from utils.image_processing.image_processing import (
    load_and_preprocess_dicom,
    load_dicom_from_dicomdir,
    process_dicomdir,
    process_zip_with_dicomdir,
    dicom_value_to_serializable,
    extract_dicom_metadata,
    normalize_array,
    load_and_preprocess_volume,
    load_and_preprocess_image
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medversa_gradio")

# Global variables
SUPPORTED_MODALITIES = ["cxr", "derm", "ct"]
SUPPORTED_TASKS = ["report generation", "classification", "segmentation"]
model = None
device = None
temp_directories = []

def is_valid_zip_file(file_path):
    """
    Check if a file is a valid ZIP file.
    
    Args:
        file_path (str): Path to the file to check
        
    Returns:
        bool: True if the file is a valid ZIP file, False otherwise
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Try to read the file list
            file_list = zip_ref.namelist()
            # If we get here, it's a valid ZIP file
            return True
    except zipfile.BadZipFile:
        # Not a ZIP file
        return False
    except Exception as e:
        # Some other error
        logger.warning(f"Error checking ZIP file: {str(e)}")
        return False

def is_valid_nifti_file(file_path):
    """
    Check if a file is a valid NIFTI file.
    
    Args:
        file_path (str): Path to the file to check
        
    Returns:
        bool: True if the file is a valid NIFTI file, False otherwise
    """
    # First check file size - NIFTI files should be at least a few KB
    try:
        size = os.path.getsize(file_path)
        if size < 1024:  # Less than 1KB is definitely not a valid NIFTI
            logger.warning(f"File too small to be a valid NIFTI: {size} bytes")
            return False
    except Exception as e:
        logger.warning(f"Error checking file size: {str(e)}")
        return False
    
    # Try to load the file with nibabel
    try:
        img = nib.load(file_path)
        # Check if the file has valid header and dimensions
        header = img.header
        shape = img.shape
        
        # Valid NIFTI should have at least 3 dimensions and reasonable size
        if len(shape) < 2 or any(dim < 1 for dim in shape):
            logger.warning(f"Invalid NIFTI dimensions: {shape}")
            return False
            
        logger.info(f"Valid NIFTI file with dimensions: {shape}")
        return True
    except Exception as e:
        logger.warning(f"Not a valid NIFTI file: {str(e)}")
        return False

def display_splash_screen():
    """Display ASCII splash screen at startup."""
    splash = '''
 __  __          ___     __                
|  \/  | ___  __| \ \   / /__ _ __ ___  __ _ 
| |\/| |/ _ \/ _` |\ \ / / _ \ '__/ __|/ _` |
| |  | |  __/ (_| | \ V /  __/ |  \__ \ (_| |
|_|  |_|\___|\__,_|  \_/ \___|_|  |___/\__,_|
                                            
'''
    print(splash)
    print("MedVersa Gradio Server - Starting up...")
    print("-" * 50)

def initialize_model():
    """Initialize the model on startup."""
    global model, device
    
    print("Initializing MedVersa model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Get the model class from registry
        model_cls = registry.get_model_class('medomni')
        
        # Get token from environment or from HuggingFace cache
        token = os.environ.get("HF_TOKEN", HfFolder.get_token())
        
        # Load model from pretrained
        model = model_cls.from_pretrained('hyzhou/MedVersa_Internal', token=token).to(device).eval()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def process_image(image_file, context, prompt, modality, task):
    """Process a single image file with the model."""
    try:
        if model is None:
            if not initialize_model():
                return None, "Model initialization failed"

        # Read file content
        if isinstance(image_file, str):  # If path is provided
            image_path = image_file
            filename = os.path.basename(image_file)
            logger.info(f"Using provided file path: {image_path}")
            # No need to read the file content for string paths
        elif isinstance(image_file, bytes):  # If bytes are provided directly
            file_content = image_file
            filename = "uploaded_image.bin"  # Default filename
            logger.info(f"Received image as bytes, size: {len(file_content)} bytes")
            # Save to a temporary file to get a path
            suffix = '.jpg'  # Default to jpg for unknown formats
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                tmp.flush()  # Ensure data is written to disk
                image_path = tmp.name
                logger.info(f"Saved uploaded file to temporary path: {image_path}")
        else:  # If file object is provided
            try:
                # Reset file pointer to beginning
                image_file.seek(0)
                file_content = image_file.read()
                filename = getattr(image_file, 'name', 'uploaded_image.bin')
            except AttributeError:
                # If seek fails, try to read it directly
                file_content = image_file
                filename = getattr(image_file, 'name', 'uploaded_image.bin')
            
            logger.info(f"Read {len(file_content)} bytes from uploaded file: {filename}")
            
            # Check if file content is empty
            if not file_content or len(file_content) == 0:
                return None, "Error: Uploaded file is empty"
                
            # Save to a temporary file to get a path
            suffix = os.path.splitext(filename)[1] if '.' in filename else '.bin'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                tmp.flush()  # Ensure data is written to disk
                image_path = tmp.name
                logger.info(f"Saved uploaded file to temporary path: {image_path}")

        logger.info(f"Processing file: {filename}")
        
        # Verify the file exists and is not empty
        if not os.path.exists(image_path):
            logger.error(f"File not found at path: {image_path}")
            return None, f"Error: File not found at {image_path}"
            
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            logger.error(f"File is empty: {filename}, size: {file_size} bytes")
            return None, f"Error: File is empty: {filename}"
        else:
            logger.info(f"File size: {file_size} bytes")

        # Set up hyperparameters for model inference - use the same as client_example.py
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0  # Reset to original value
        length_penalty = 1.0
        temperature = 0.1
        
        # Use the same prompt format as client_example.py
        if not prompt:
            if task == "report generation":
                prompt = "How would you characterize the findings from <img0>?"
            elif task == "classification":
                prompt = "What is the primary diagnosis?"
            elif task == "segmentation":
                if modality == "derm":
                    prompt = "Segment the lesion."
                elif modality == "ct":
                    prompt = "Segment the liver."
        
        # Format context if provided
        if context and not context.endswith('\n'):
            context += '\n'
        
        # Determine the file type and use appropriate processing
        if suffix.lower() in ['.dcm', '.dicom']:
            # Process as DICOM file
            return await process_dicom(image_file, context, prompt, modality, task)
        elif suffix.lower() in ['.nii', '.nii.gz']:
            # Process as NIFTI file
            return await process_nifti(image_file, context, prompt, modality, task)
        else:
            # Process as standard image file (JPEG, PNG, etc.)
            try:
                # Use the read_image function from utils.py to get the properly processed tensor
                image_tensor = read_image(image_path).to(device)
                
                # Get a PIL Image for display
                orig_pil_image = Image.open(image_path).convert('RGB')
                
                # Run inference using the generate_predictions function
                seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
                    model, 
                    [image_path], 
                    context, 
                    prompt, 
                    modality, 
                    task, 
                    num_beams, 
                    do_sample, 
                    min_length, 
                    top_p, 
                    repetition_penalty, 
                    length_penalty, 
                    temperature, 
                    device
                )
                
                # Process results based on task
                if task == "segmentation":
                    output_images = []
                    
                    # Add original image
                    output_images.append(orig_pil_image)
                    
                    # Add segmentation overlay if available
                    if seg_mask_2d is not None and len(seg_mask_2d) > 0:
                        # For 2D segmentation
                        mask = seg_mask_2d[0]
                        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                        mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
                        
                        # Create overlay
                        input_np = np.array(orig_pil_image)
                        overlay = input_np.copy()
                        alpha = 0.5
                        
                        # Apply mask only if it has any positive values
                        if np.any(mask > 0):
                            overlay[mask > 0] = cv2.addWeighted(
                                overlay[mask > 0],
                                1-alpha,
                                mask_rgb[mask > 0],
                                alpha,
                                0
                            )
                        
                        overlay_img = Image.fromarray(overlay)
                        output_images.append(overlay_img)
                    
                    return output_images, f"Segmentation completed.\n\n{output_text}"
                
                # For classification and report generation, just return the original image and text
                return [orig_pil_image], output_text
                
            except Exception as e:
                logger.error(f"Error processing standard image: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, f"Error: {str(e)}"

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

async def process_dicom(dicom_file, context, prompt, modality, task):
    """Process a DICOM file with the model."""
    try:
        if model is None:
            if not initialize_model():
                return None, "Model initialization failed"

        # Read file content
        if isinstance(dicom_file, str):  # If path is provided
            dicom_path = dicom_file
            filename = os.path.basename(dicom_file)
            logger.info(f"Using provided DICOM file path: {dicom_path}")
            # We can use the path directly, no need to read content
            # Validate the DICOM file
            try:
                dicom_obj = pydicom.dcmread(dicom_path, force=True)
                logger.info(f"DICOM file validated successfully: {dicom_path}")
                if hasattr(dicom_obj, 'Modality'):
                    logger.info(f"DICOM modality: {dicom_obj.Modality}")
            except Exception as e:
                logger.warning(f"Failed to validate DICOM file: {str(e)}")
        elif isinstance(dicom_file, bytes):  # If bytes are provided directly
            file_content = dicom_file
            filename = "uploaded_dicom.dcm"  # Default filename
            logger.info(f"Received DICOM as bytes, size: {len(file_content)} bytes")
            
            # Save to a temporary file to get a path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
                tmp.write(file_content)
                tmp.flush()  # Ensure data is written to disk
                dicom_path = tmp.name
                logger.info(f"Saved uploaded DICOM file to temporary path: {dicom_path}")
                
            # Validate the saved DICOM file
            try:
                dicom_obj = pydicom.dcmread(dicom_path, force=True)
                logger.info(f"DICOM file validated successfully: {dicom_path}")
                if hasattr(dicom_obj, 'Modality'):
                    logger.info(f"DICOM modality: {dicom_obj.Modality}")
            except Exception as e:
                logger.warning(f"Failed to validate DICOM file: {str(e)}")
        else:  # If file object is provided
            try:
                # Reset file pointer to beginning
                dicom_file.seek(0)
                file_content = dicom_file.read()
                filename = getattr(dicom_file, 'name', 'uploaded_dicom.dcm')
            except AttributeError:
                # If seek fails, try to read it directly
                file_content = dicom_file
                filename = getattr(dicom_file, 'name', 'uploaded_dicom.dcm')
            
            logger.info(f"Read {len(file_content)} bytes from uploaded DICOM file: {filename}")
            
            # Save to a temporary file to get a path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
                tmp.write(file_content)
                tmp.flush()  # Ensure data is written to disk
                dicom_path = tmp.name
                logger.info(f"Saved uploaded DICOM file to temporary path: {dicom_path}")
                
            # Validate the saved DICOM file
            try:
                dicom_obj = pydicom.dcmread(dicom_path, force=True)
                logger.info(f"DICOM file validated successfully: {dicom_path}")
                if hasattr(dicom_obj, 'Modality'):
                    logger.info(f"DICOM modality: {dicom_obj.Modality}")
            except Exception as e:
                logger.warning(f"Failed to validate DICOM file: {str(e)}")

        logger.info(f"Processing DICOM file: {filename}")
        
        # Verify the file exists and is not empty
        if not os.path.exists(dicom_path):
            logger.error(f"DICOM file not found at path: {dicom_path}")
            return None, f"Error: DICOM file not found at {dicom_path}"
            
        file_size = os.path.getsize(dicom_path)
        if file_size == 0:
            logger.error(f"DICOM file is empty: {filename}, size: {file_size} bytes")
            return None, f"Error: DICOM file is empty: {filename}"
        else:
            logger.info(f"DICOM file size: {file_size} bytes")

        # Extract metadata for context if not provided
        if not context:
            try:
                metadata = extract_dicom_metadata(dicom_path)
                patient_age = metadata.get('PatientAge', 'Unknown')
                patient_gender = metadata.get('PatientSex', 'Unknown')
                
                # Format gender properly
                if patient_gender == 'M':
                    patient_gender = 'Male'
                elif patient_gender == 'F':
                    patient_gender = 'Female'
                
                context = f"Age:{patient_age}.\nGender:{patient_gender}.\n"
                context += f"Indication: {metadata.get('StudyDescription', 'Unknown')}.\nComparison: None.\n"
                logger.info(f"Extracted context: {context}")
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {str(e)}")
                context = ""

        # Set up hyperparameters - use the same as client_example.py
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0  # Reset to original value
        length_penalty = 1.0
        temperature = 0.1
        
        # Use the same prompt format as client_example.py
        if not prompt:
            if task == "report generation":
                prompt = "How would you characterize the findings from <img0>?"
            elif task == "classification":
                prompt = "What is the primary diagnosis?"
            elif task == "segmentation":
                if modality == "derm":
                    prompt = "Segment the lesion."
                elif modality == "ct":
                    prompt = "Segment the liver."
        
        # Format context if provided
        if context and not context.endswith('\n'):
            context += '\n'
        
        # Try to determine if it's a multi-slice DICOM (3D volume)
        is_multi_slice = False
        try:
            # Check if it's a multi-frame DICOM
            dicom = pydicom.dcmread(dicom_path, force=True)
            if hasattr(dicom, 'NumberOfFrames') and int(dicom.NumberOfFrames) > 1:
                is_multi_slice = True
                logger.info(f"Detected multi-frame DICOM with {dicom.NumberOfFrames} frames")
            elif hasattr(dicom, 'pixel_array') and len(dicom.pixel_array.shape) >= 3 and dicom.pixel_array.shape[0] > 1:
                is_multi_slice = True
                logger.info(f"Detected multi-slice DICOM with shape {dicom.pixel_array.shape}")
            
            # Auto-detect modality if not specified
            if hasattr(dicom, 'Modality'):
                dicom_modality = dicom.Modality
                logger.info(f"DICOM modality from file: {dicom_modality}")
                
                # Map DICOM modality to our supported modalities
                modality_map = {
                    'CR': 'cxr',  # Computed Radiography
                    'DX': 'cxr',  # Digital Radiography
                    'CT': 'ct',   # Computed Tomography
                    'MR': 'ct',   # Magnetic Resonance (use 'ct' as it's the closest supported)
                    'US': 'derm', # Ultrasound (use 'derm' as fallback)
                }
                
                if dicom_modality in modality_map:
                    detected_modality = modality_map[dicom_modality]
                    logger.info(f"Mapped DICOM modality {dicom_modality} to {detected_modality}")
                    modality = detected_modality
        except Exception as e:
            logger.warning(f"Error checking DICOM type: {str(e)}")
        
        # Run inference using the appropriate utils functions
        if is_multi_slice:
            # Process as 3D volume
            # Get display images using our image processing module
            dicom_images = load_and_preprocess_dicom(dicom_path)
            
            # Run inference
            logger.info(f"Running inference on multi-slice DICOM with modality={modality}, task={task}")
            seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
                model, 
                [dicom_path], 
                context, 
                prompt, 
                modality, 
                task, 
                num_beams, 
                do_sample, 
                min_length, 
                top_p, 
                repetition_penalty, 
                length_penalty, 
                temperature, 
                device
            )
        else:
            # Process as 2D image
            # Get display images using our image processing module
            dicom_images = load_and_preprocess_dicom(dicom_path)
            
            # Run inference
            logger.info(f"Running inference on 2D DICOM with modality={modality}, task={task}")
            seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
                model, 
                [dicom_path], 
                context, 
                prompt, 
                modality, 
                task, 
                num_beams, 
                do_sample, 
                min_length, 
                top_p, 
                repetition_penalty, 
                length_penalty, 
                temperature, 
                device
            )
        
        # Check if we got valid display images
        if not dicom_images or len(dicom_images) == 0:
            logger.error("No display images were generated from DICOM")
            return None, "Error: Failed to generate display images from DICOM"
        
        # Process results based on task
        if task == "segmentation":
            output_images = []
            
            # Add original images
            for img in dicom_images:
                output_images.append(img)
            
            # Add segmentation overlay if available
            if seg_mask_2d and len(seg_mask_2d) > 0:
                # For 2D segmentation, create overlays
                for idx, mask in enumerate(seg_mask_2d):
                    if idx >= len(dicom_images):
                        break  # Don't process more masks than we have images
                        
                    # Get the corresponding image
                    orig_img = dicom_images[idx]
                    
                    # Create mask overlay
                    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
                    
                    # Create overlay
                    overlay = np.array(orig_img).copy()
                    alpha = 0.5
                    
                    # Apply mask only if it has any positive values
                    if np.any(mask > 0):
                        overlay[mask > 0] = cv2.addWeighted(
                            overlay[mask > 0],
                            1-alpha,
                            mask_rgb[mask > 0],
                            alpha,
                            0
                        )
                    
                    # Convert overlay to PIL Image
                    overlay_img = Image.fromarray(overlay)
                    output_images.append(overlay_img)
            
            # If 3D segmentation masks are available
            elif seg_mask_3d and len(seg_mask_3d) > 0:
                # For 3D segmentation, just use the middle slice for preview
                middle_idx = len(seg_mask_3d) // 2
                
                # Get the corresponding image (middle slice)
                if middle_idx < len(dicom_images):
                    orig_img = dicom_images[middle_idx]
                    
                    # Create mask overlay
                    mask = seg_mask_3d[middle_idx]
                    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
                    
                    # Create overlay
                    overlay = np.array(orig_img).copy()
                    alpha = 0.5
                    
                    # Apply mask only if it has any positive values
                    if np.any(mask > 0):
                        overlay[mask > 0] = cv2.addWeighted(
                            overlay[mask > 0],
                            1-alpha,
                            mask_rgb[mask > 0],
                            alpha,
                            0
                        )
                    
                    # Convert overlay to PIL Image
                    overlay_img = Image.fromarray(overlay)
                    output_images.append(overlay_img)
            
            return output_images, f"DICOM Segmentation completed.\n\n{output_text}"
        
        # For classification and report generation, just return the original images and text
        return dicom_images, output_text

    except Exception as e:
        logger.error(f"Error processing DICOM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

async def process_nifti(nifti_file, context, prompt, modality, task):
    """Process a NIFTI file with the model."""
    try:
        if model is None:
            if not initialize_model():
                return None, "Model initialization failed"

        # Read file content
        if isinstance(nifti_file, str):  # If path is provided
            nifti_path = nifti_file
            filename = os.path.basename(nifti_file)
            logger.info(f"Using provided NIFTI file path: {nifti_path}")
            # We can use the path directly, no need to read content
        elif isinstance(nifti_file, bytes):  # If bytes are provided directly
            file_content = nifti_file
            filename = "uploaded_nifti.nii.gz"  # Default filename
            logger.info(f"Received NIFTI as bytes, size: {len(file_content)} bytes")
            
            # Validate that it's a proper NIFTI file before saving
            try:
                # Check if it's a gzip file
                import gzip
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(file_content)) as test:
                        test.read(1024)  # Try to read some data
                    is_gzip = True
                    suffix = '.nii.gz'
                except:
                    is_gzip = False
                    suffix = '.nii'
                    # Basic check for NIFTI magic bytes
                    if len(file_content) > 348:  # NIFTI header is at least 348 bytes
                        magic_bytes = file_content[344:348]
                        if not (magic_bytes == b'ni1\0' or magic_bytes == b'n+1\0'):
                            logger.warning("Uploaded file doesn't have NIFTI magic bytes")
                    else:
                        logger.warning("Uploaded file is too small to be a valid NIFTI file")
            except Exception as e:
                logger.warning(f"Error checking NIfTI format: {str(e)}")
                suffix = '.nii.gz'  # Default to .nii.gz
                
            # Save to a temporary file to get a path
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                tmp.flush()  # Ensure data is written to disk
                nifti_path = tmp.name
                logger.info(f"Saved uploaded NIFTI file to temporary path: {nifti_path}")
        else:  # If file object is provided
            try:
                # Reset file pointer to beginning
                nifti_file.seek(0)
                file_content = nifti_file.read()
                filename = getattr(nifti_file, 'name', 'uploaded_nifti.nii.gz')
            except AttributeError:
                # If seek fails, try to read it directly
                file_content = nifti_file
                filename = getattr(nifti_file, 'name', 'uploaded_nifti.nii.gz')
            
            logger.info(f"Read {len(file_content)} bytes from uploaded NIFTI file: {filename}")
            
            # Check if file content is empty or too small
            if not file_content or len(file_content) < 348:  # NIFTI header is at least 348 bytes
                return None, "Error: Uploaded file is too small to be a valid NIFTI file"
            
            # Determine if it's a gzipped file
            try:
                import gzip
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(file_content)) as test:
                        test.read(1024)  # Try to read some data
                    is_gzip = True
                    suffix = '.nii.gz'
                except:
                    is_gzip = False
                    suffix = '.nii'
            except Exception as e:
                logger.warning(f"Error checking if file is gzipped: {str(e)}")
                suffix = '.nii.gz' if filename.lower().endswith('.nii.gz') else '.nii'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                tmp.flush()  # Ensure data is written to disk
                nifti_path = tmp.name
                logger.info(f"Saved uploaded NIFTI file to temporary path: {nifti_path}")

        logger.info(f"Processing NIFTI file: {filename}")
        
        # Verify the file exists and is not empty
        if not os.path.exists(nifti_path):
            logger.error(f"NIFTI file not found at path: {nifti_path}")
            return None, f"Error: NIFTI file not found at {nifti_path}"
            
        file_size = os.path.getsize(nifti_path)
        if file_size == 0:
            logger.error(f"NIFTI file is empty: {filename}, size: {file_size} bytes")
            return None, f"Error: NIFTI file is empty: {filename}"
        elif file_size < 1024:  # Less than 1KB is definitely not a valid NIFTI
            logger.error(f"File too small to be a valid NIFTI: {file_size} bytes")
            return None, f"Error: The uploaded file is too small ({file_size} bytes) to be a valid NIFTI file. NIFTI files are typically at least several KB in size."
        
        # Validate the NIFTI file before processing
        if not is_valid_nifti_file(nifti_path):
            return None, "Error: The uploaded file is not a valid NIFTI file. Please check the file and try again."

        # Set up hyperparameters - use the same as client_example.py
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0  # Reset to match client_example.py
        length_penalty = 1.0
        temperature = 0.1
        
        # Use the same simple prompts from client_example.py
        if not prompt:
            if task == "segmentation":
                prompt = "Segment the liver."
            elif task == "report generation":
                prompt = "How would you characterize the findings from <img0>?"
            elif task == "classification":
                prompt = "What is the primary diagnosis?"
        
        # Format context same as client_example.py (usually empty for NIFTI)
        if context and not context.endswith('\n'):
            context += '\n'
        
        try:
            # Get display images for the UI
            logger.info(f"Loading NIFTI volume from: {nifti_path}")
            nifti_images = load_and_preprocess_volume(nifti_path)
            
            if not nifti_images or len(nifti_images) == 0:
                logger.error("Failed to load NIFTI volume for display")
                return None, "Error: Failed to preprocess NIFTI volume. The file may be corrupted or not in the expected format."
            
            # Run inference
            logger.info("Running model inference on NIFTI volume")
            seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
                model, 
                [nifti_path], 
                context, 
                prompt, 
                modality, 
                task, 
                num_beams, 
                do_sample, 
                min_length, 
                top_p, 
                repetition_penalty, 
                length_penalty, 
                temperature, 
                device
            )
            
            # Process results based on task
            if task == "segmentation" and seg_mask_3d is not None:
                # For 3D segmentation
                output_images = []
                
                # Add original slices
                for img in nifti_images:
                    output_images.append(img)
                    
                # For visualization, select some representative slices from the segmentation
                if len(seg_mask_3d) > 0:
                    # Find slices with segmentation
                    slices_with_seg = []
                    for i, mask in enumerate(seg_mask_3d):
                        if np.any(mask > 0):
                            slices_with_seg.append(i)
                    
                    # If we have slices with segmentation, pick a few representative ones
                    if slices_with_seg:
                        # Pick a few slices (middle of segmentation range)
                        middle_seg_idx = slices_with_seg[len(slices_with_seg) // 2]
                        
                        # Take the middle slice if it's in range of our display images
                        if middle_seg_idx < len(nifti_images):
                            # Get the corresponding image
                            orig_img = nifti_images[middle_seg_idx]
                            
                            # Create mask overlay
                            mask = seg_mask_3d[middle_seg_idx]
                            mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                            mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
                            
                            # Create overlay
                            overlay = np.array(orig_img).copy()
                            alpha = 0.5
                            
                            # Apply mask with transparency
                            if np.any(mask > 0):
                                overlay[mask > 0] = cv2.addWeighted(
                                    overlay[mask > 0],
                                    1-alpha,
                                    mask_rgb[mask > 0],
                                    alpha,
                                    0
                                )
                            
                            # Convert overlay to PIL Image
                            overlay_img = Image.fromarray(overlay)
                            output_images.append(overlay_img)
                
                return output_images, f"3D Segmentation completed. Total slices: {len(seg_mask_3d) if seg_mask_3d else 0}\n\n{output_text}"
            
            # For classification and report generation, just return some representative slices
            # Take a few at different positions in the volume
            output_images = []
            step = max(1, len(nifti_images) // 3)  # Take about 3 slices
            
            for i in range(0, len(nifti_images), step):
                if len(output_images) >= 3:  # Limit to 3 images
                    break
                output_images.append(nifti_images[i])
            
            return output_images, output_text
                
        except Exception as e:
            logger.error(f"Error processing NIFTI: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Error: There was a problem processing the NIFTI file: {str(e)}"

    except Exception as e:
        logger.error(f"Error processing NIFTI: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

def cleanup_temp_directories():
    """Clean up all temporary directories when the application exits"""
    for directory in temp_directories:
        try:
            if directory and os.path.exists(directory):
                shutil.rmtree(directory)
                logger.info(f"Cleaned up temporary directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {str(e)}")

async def process_dicomdir_file(dicomdir_file):
    """Process a DICOMDIR file and extract its structure."""
    try:
        # Save to a temporary file to get a path
        if isinstance(dicomdir_file, str):  # If path is provided
            dicomdir_path = dicomdir_file
            filename = os.path.basename(dicomdir_file)
            logger.info(f"Using provided DICOMDIR file path: {dicomdir_path}")
            # Read the file content
            with open(dicomdir_path, 'rb') as f:
                file_content = f.read()
        elif isinstance(dicomdir_file, bytes):  # If bytes are provided directly
            file_content = dicomdir_file
            filename = "uploaded_dicomdir"  # Default filename
            logger.info(f"Received DICOMDIR as bytes, size: {len(file_content)} bytes")
        else:  # If file object is provided
            try:
                # Reset file pointer to beginning
                dicomdir_file.seek(0)
                file_content = dicomdir_file.read()
                filename = getattr(dicomdir_file, 'name', 'uploaded_dicomdir')
            except AttributeError:
                # If seek fails, try to read it directly
                file_content = dicomdir_file
                filename = getattr(dicomdir_file, 'name', 'uploaded_dicomdir')
            
            logger.info(f"Read {len(file_content)} bytes from uploaded DICOMDIR file: {filename}")
        
        # Check if file content is empty
        if not file_content or len(file_content) == 0:
            return "Error: Uploaded file is empty", None, []
            
        # Save to a temporary file to get a path
        suffix = '.zip' if filename.lower().endswith('.zip') else '.dcmdir'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp.flush()  # Ensure data is written to disk
            dicomdir_path = tmp.name
            logger.info(f"Saved uploaded DICOMDIR file to temporary path: {dicomdir_path}")

        logger.info(f"Processing DICOMDIR file: {filename}")
        
        # Verify the file exists and is not empty
        if not os.path.exists(dicomdir_path):
            logger.error(f"DICOMDIR file not found at path: {dicomdir_path}")
            return f"Error: File not found at {dicomdir_path}", None, []
            
        file_size = os.path.getsize(dicomdir_path)
        if file_size == 0:
            logger.error(f"DICOMDIR file is empty: {filename}, size: {file_size} bytes")
            return f"Error: File is empty: {filename}", None, []
        else:
            logger.info(f"DICOMDIR file size: {file_size} bytes")

        # Check if it's a ZIP file - first by extension, then by content
        is_zip = False
        if filename.lower().endswith('.zip'):
            is_zip = True
            logger.info("Detected ZIP file by extension")
        else:
            # Try to check file signature (magic bytes for ZIP files)
            try:
                with open(dicomdir_path, 'rb') as f:
                    magic_bytes = f.read(4)
                    # ZIP file signature is PK\x03\x04 (or 50 4B 03 04 in hex)
                    if magic_bytes.startswith(b'PK\x03\x04'):
                        is_zip = True
                        logger.info("Detected ZIP file by signature")
            except Exception as e:
                logger.warning(f"Error checking file signature: {str(e)}")

        if is_zip:
            logger.info("Processing as ZIP file with DICOMDIR")
            
            # Verify it's a valid ZIP file
            if not is_valid_zip_file(dicomdir_path):
                logger.error("Invalid ZIP file format")
                return "Error: The file has a ZIP extension but is not a valid ZIP file. Please check the file and try again.", None, []
            
            # Create a temporary directory to extract the ZIP
            extract_dir = tempfile.mkdtemp()
            temp_directories.append(extract_dir)
            
            try:
                # Extract the ZIP file
                with zipfile.ZipFile(dicomdir_path, 'r') as zip_ref:
                    # Log the contents of the ZIP file
                    file_list = zip_ref.namelist()
                    logger.info(f"ZIP file contains {len(file_list)} files")
                    if len(file_list) > 0:
                        logger.info(f"First few files: {file_list[:5]}")
                    
                    # Extract all files
                    zip_ref.extractall(extract_dir)
                    logger.info(f"Extracted ZIP to {extract_dir}")
                
                # Look for DICOMDIR file in the extracted directory
                dicomdir_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file.upper() == 'DICOMDIR':
                            dicomdir_files.append(os.path.join(root, file))
                
                if not dicomdir_files:
                    return "No DICOMDIR file found in the ZIP archive", None, []
                
                # Use the first DICOMDIR file found
                dicomdir_path = dicomdir_files[0]
                logger.info(f"Found DICOMDIR at: {dicomdir_path}")
                
                # Store the extract directory for later use
                directory = {"_extract_dir": extract_dir, "_path": dicomdir_path}
            except zipfile.BadZipFile as e:
                logger.error(f"Bad ZIP file: {str(e)}")
                return f"Error: Not a valid ZIP file: {str(e)}", None, []
            except Exception as e:
                logger.error(f"Error extracting ZIP: {str(e)}")
                return f"Error extracting ZIP: {str(e)}", None, []
        else:
            # Process as regular DICOMDIR
            directory = {"_path": dicomdir_path}
        
        # Read the DICOMDIR file
        try:
            dicomdir = pydicom.dcmread(dicomdir_path, force=True)
        except Exception as e:
            logger.error(f"Error reading DICOMDIR: {str(e)}")
            return f"Error reading DICOMDIR: {str(e)}", None, []
        
        # Parse the DICOMDIR structure
        directory["patients"] = []
        
        # Process patient records
        for patient_record in dicomdir.patient_records:
            patient = {
                "id": getattr(patient_record, "PatientID", "Unknown"),
                "name": getattr(patient_record, "PatientName", "Unknown"),
                "studies": []
            }
            
            # Process study records
            studies = getattr(patient_record, "children", [])
            for study_record in studies:
                if study_record.DirectoryRecordType == "STUDY":
                    study = {
                        "id": getattr(study_record, "StudyInstanceUID", "Unknown"),
                        "description": getattr(study_record, "StudyDescription", "Unknown"),
                        "date": getattr(study_record, "StudyDate", "Unknown"),
                        "series": []
                    }
                    
                    # Process series records
                    series_records = getattr(study_record, "children", [])
                    for series_record in series_records:
                        if series_record.DirectoryRecordType == "SERIES":
                            series = {
                                "id": getattr(series_record, "SeriesInstanceUID", "Unknown"),
                                "description": getattr(series_record, "SeriesDescription", "Unknown"),
                                "modality": getattr(series_record, "Modality", "Unknown"),
                                "images": []
                            }
                            
                            # Process image records
                            image_records = getattr(series_record, "children", [])
                            for image_record in image_records:
                                if image_record.DirectoryRecordType == "IMAGE":
                                    image = {
                                        "id": getattr(image_record, "SOPInstanceUID", "Unknown"),
                                        "number": getattr(image_record, "InstanceNumber", 0),
                                        "path": getattr(image_record, "ReferencedFileID", "Unknown")
                                    }
                                    
                                    # Convert path from list to string if needed
                                    if isinstance(image["path"], list):
                                        image["path"] = os.path.join(*image["path"])
                                    
                                    series["images"].append(image)
                            
                            # Sort images by instance number
                            series["images"].sort(key=lambda x: int(x["number"]) if isinstance(x["number"], (int, str)) and str(x["number"]).isdigit() else 0)
                            
                            study["series"].append(series)
                    
                    patient["studies"].append(study)
            
            directory["patients"].append(patient)
        
        # Format the directory structure for display
        formatted_structure = []
        
        for patient in directory["patients"]:
            patient_str = f"Patient: {patient['name']} (ID: {patient['id']})"
            formatted_structure.append(patient_str)
            
            for study in patient["studies"]:
                study_str = f"  Study: {study['description']} (Date: {study['date']})"
                formatted_structure.append(study_str)
                
                for series in study["series"]:
                    series_key = f"{patient['id']}_{study['id']}_{series['id']}"
                    series_str = f"    Series: {series['description']} (Modality: {series['modality']}, Images: {len(series['images'])})"
                    formatted_structure.append(f"{series_str} [Key: {series_key}]")
        
        # Extract series keys for the dropdown
        series_keys = []
        series_mapping = {}
        
        for patient in directory["patients"]:
            patient_id = patient["id"]
            for study in patient["studies"]:
                study_id = study["id"]
                for series in study["series"]:
                    series_id = series["id"]
                    series_key = f"{patient_id}_{study_id}_{series_id}"
                    
                    # Ensure that description, modality, and images count are converted to strings
                    series_desc_str = str(series.get('description', 'Unknown'))
                    modality_str = str(series.get('modality', 'Unknown'))
                    images_count = len(series.get('images', []))
                    
                    # Format the series description as a string
                    series_desc = f"{series_desc_str} - {modality_str} ({images_count} images)"
                    
                    # Store in mapping (using string keys)
                    series_mapping[series_desc] = series_key
                    series_keys.append((series_desc, series_key))
        
        # Store the mapping in the directory structure
        directory["_series_mapping"] = series_mapping
        
        # Save the directory structure to a temporary file for later reference
        directory_path = tempfile.mktemp(suffix='.json')
        with open(directory_path, 'w') as f:
            json.dump(directory, f, default=str)
        logger.info(f"Saved directory structure to: {directory_path}")
        
        # Create a list of series descriptions for the dropdown - ensure they're all strings
        series_dropdown_values = []
        for desc, _ in series_keys:
            # Ensure the description is a string
            if not isinstance(desc, str):
                desc = str(desc)
                logger.info(f"Converted non-string description to string: {desc}")
            
            # Reject any list-like strings for the dropdown to avoid confusion
            if desc.startswith('[') and desc.endswith(']'):
                try:
                    # Attempt to extract a clean value
                    clean_desc = desc.strip('[]').strip("'\"")
                    logger.info(f"Cleaned list-like description for dropdown: {clean_desc}")
                    series_dropdown_values.append(clean_desc)
                except Exception as e:
                    logger.warning(f"Error cleaning list-like description: {str(e)}")
                    series_dropdown_values.append(desc)
            else:
                series_dropdown_values.append(desc)
            
        # Sort the dropdown values for better navigation
        series_dropdown_values.sort()
        
        return "\n".join(formatted_structure), directory_path, series_dropdown_values
    
    except Exception as e:
        logger.error(f"Error processing DICOMDIR: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, []

async def process_dicomdir_series(directory_path, series_desc, context, prompt, modality, task):
    try:
        # Log the details of the series_desc for debugging
        logger.info(f"Processing series description of type {type(series_desc)}: {series_desc}")

        if not directory_path or not isinstance(directory_path, str) or not os.path.exists(directory_path):
            return None, "Error: Directory information not found or invalid"
        
        # Handle the case where series_desc might be a list
        if isinstance(series_desc, list):
            if len(series_desc) == 1:
                logger.info(f"Series description is a list with {len(series_desc)} items: {series_desc}")
                # Extract the actual value from the list
                series_desc = str(series_desc[0])
                logger.info(f"Extracted series description from list: {series_desc}")
            else:
                logger.info(f"Series description is a list with multiple items: {series_desc}")
                # Join multiple items as a fallback
                series_desc = " - ".join(str(item) for item in series_desc)
                logger.info(f"Joined multiple items into a single string: {series_desc}")
        # Handle the case where series_desc is a string that looks like a list
        elif isinstance(series_desc, str) and series_desc.startswith('[') and series_desc.endswith(']'):
            logger.info(f"Series description appears to be a string representation of a list: {series_desc}")
            try:
                # Parse the string using ast.literal_eval
                import ast
                parsed_list = ast.literal_eval(series_desc)
                if isinstance(parsed_list, list):
                    if len(parsed_list) == 1:
                        # Extract the single value
                        series_desc = str(parsed_list[0])
                        logger.info(f"Extracted series description from parsed list: {series_desc}")
                    else:
                        # Join multiple items
                        series_desc = " - ".join(str(item) for item in parsed_list)
                        logger.info(f"Joined multiple items from parsed list: {series_desc}")
                else:
                    logger.warning(f"Parsed value is not a list: {type(parsed_list)}")
                    # Continue with the original string
            except Exception as e:
                logger.warning(f"Failed to parse series_desc as list: {str(e)}")
                # Continue with the original string
        
        # Load the directory structure
        with open(directory_path, 'r') as f:
            directory = json.load(f)
        
        # Get the series mapping
        series_mapping = directory.get("_series_mapping", {})
        
        # Check multiple strategies to find the correct series key
        series_key = None
        
        # Strategy 1: Direct lookup in the mapping
        if series_desc in series_mapping:
            series_key = series_mapping[series_desc]
            logger.info(f"Found series key via direct lookup: {series_key}")
        
        # Strategy 2: If the description is already a key format, use it directly
        elif isinstance(series_desc, str) and series_desc.count('_') >= 2:
            # Check if it actually matches our pattern
            try:
                parts = series_desc.split('_', 2)  # Split into 3 parts maximum
                if len(parts) >= 3:  # We need at least 3 parts for a valid key
                    series_key = series_desc
                    logger.info(f"Using series description directly as a key: {series_key}")
            except Exception as e:
                logger.warning(f"Failed to parse series_desc as key: {str(e)}")
        
        # Strategy 3: Try case-insensitive matching
        if not series_key:
            for desc, key in series_mapping.items():
                if desc.lower() == series_desc.lower():
                    series_key = key
                    logger.info(f"Found series key via case-insensitive match: {series_key}")
                    break
        
        # Strategy 4: Check if the key is in the values
        if not series_key:
            if series_desc in series_mapping.values():
                series_key = series_desc
                logger.info(f"Found series description as an existing key in values: {series_key}")
        
        # Strategy 5: Try partial matching
        if not series_key:
            for desc, key in series_mapping.items():
                if series_desc in desc or desc in series_desc:
                    series_key = key
                    logger.info(f"Found series key via partial match: {series_key}")
                    break
        
        # If we still don't have a key, provide an informative error
        if not series_key:
            # Provide dropdown choices for better debugging
            dropdown_choices = list(series_mapping.keys())
            dropdown_info = ""
            if dropdown_choices:
                dropdown_info = "\n\nAvailable series in dropdown:\n" + "\n".join(
                    f"- {desc}" for desc in dropdown_choices[:10]  # Show only first 10 to avoid overwhelming
                )
                if len(dropdown_choices) > 10:
                    dropdown_info += f"\n... and {len(dropdown_choices) - 10} more"
            
            error_msg = (f"Error: Could not find series with description: '{series_desc}'. "
                        f"Please select a valid series from the dropdown.{dropdown_info}")
            logger.error(error_msg)
            return None, error_msg
        
        logger.info(f"Using series key: {series_key}")
        
        # Parse the series key
        try:
            patient_id, study_id, series_id = series_key.split('_', 2)
        except ValueError:
            logger.error(f"Invalid series key format: {series_key}")
            return None, f"Invalid series key format: {series_key}. Expected format: patient_id_study_id_series_id"
        
        # Find the series
        selected_series = None
        for patient in directory["patients"]:
            if patient["id"] == patient_id:
                for study in patient["studies"]:
                    if study["id"] == study_id:
                        for series in study["series"]:
                            if series["id"] == series_id:
                                selected_series = series
                                break
        
        if not selected_series:
            return None, f"Series not found: {series_key}"
        
        # Get the DICOMDIR path
        dicomdir_path = directory.get("_path")
        if not dicomdir_path:
            return None, "DICOMDIR path not found"
        
        # Check if the DICOMDIR file exists
        if not os.path.exists(dicomdir_path):
            return None, f"DICOMDIR file not found at: {dicomdir_path}"
        
        # Get the extract directory if available
        extract_dir = directory.get("_extract_dir")
        
        # Load the middle image from the series for analysis
        if not selected_series["images"]:
            return None, "No images found in the selected series"
            
        middle_idx = len(selected_series["images"]) // 2
        middle_image = selected_series["images"][middle_idx]
        
        logger.info(f"Loading image {middle_idx+1} of {len(selected_series['images'])} from series {series_key}")
        logger.info(f"Image path: {middle_image['path']}")
        
        # Try to load the DICOM file
        dicom_path = None
        error_messages = []
        
        # If we have an extract directory, try to find the file there
        if extract_dir:
            # Try different path formats
            possible_paths = []
            
            # Handle image path that might be a list (e.g., ['DICOM', 'I19'])
            img_path = middle_image["path"]
            
            # Log detailed information about the path type
            logger.info(f"Path type: {type(img_path)}, Path value: {img_path}")
            
            # CASE 1: Path is an actual list
            if isinstance(img_path, list):
                logger.info(f"Processing list path with {len(img_path)} elements")
                
                # Convert all elements to strings
                img_path_elements = [str(element) for element in img_path]
                
                # Create different path formats
                path_with_os_sep = os.path.join(*img_path_elements) 
                path_with_fwd_slash = '/'.join(img_path_elements)
                path_with_back_slash = '\\'.join(img_path_elements)
                
                logger.info(f"Created path variants: OS-specific: {path_with_os_sep}, Forward slash: {path_with_fwd_slash}, Backslash: {path_with_back_slash}")
                
                # Add all variants to possible paths
                possible_paths.append(os.path.join(extract_dir, path_with_os_sep))
                possible_paths.append(os.path.join(extract_dir, path_with_fwd_slash))
                possible_paths.append(os.path.join(extract_dir, path_with_back_slash))
                
                # Try each element individually as a fallback
                for i, element in enumerate(img_path_elements):
                    possible_paths.append(os.path.join(extract_dir, element))
                    logger.info(f"Added individual element {i}: {element}")
                
                # Special case for DICOM files: try common naming patterns
                if len(img_path_elements) > 1:
                    if img_path_elements[0].upper() == "DICOM":
                        # Try path without the DICOM prefix
                        for i in range(1, len(img_path_elements)):
                            possible_paths.append(os.path.join(extract_dir, *img_path_elements[i:]))
                            logger.info(f"Added path without DICOM prefix: {os.path.join(*img_path_elements[i:])}")
            
            # CASE 2: Path is a string but looks like a list (e.g. "['DICOM', 'I19']")
            elif isinstance(img_path, str) and img_path.startswith('[') and img_path.endswith(']'):
                logger.info(f"Processing string that looks like a list: {img_path}")
                
                try:
                    # Try to parse it as a list using AST (safer than eval)
                    # Make sure to use the global ast import, not trying to import within the function
                    parsed_list = ast.literal_eval(img_path)
                    
                    if isinstance(parsed_list, list):
                        logger.info(f"Successfully parsed string into list with {len(parsed_list)} elements")
                        
                        # Convert all elements to strings
                        img_path_elements = [str(element) for element in parsed_list]
                        
                        # Create different path formats
                        path_with_os_sep = os.path.join(*img_path_elements) 
                        path_with_fwd_slash = '/'.join(img_path_elements)
                        path_with_back_slash = '\\'.join(img_path_elements)
                        
                        logger.info(f"Created path variants: OS-specific: {path_with_os_sep}, Forward slash: {path_with_fwd_slash}, Backslash: {path_with_back_slash}")
                        
                        # Add all variants to possible paths
                        possible_paths.append(os.path.join(extract_dir, path_with_os_sep))
                        possible_paths.append(os.path.join(extract_dir, path_with_fwd_slash))
                        possible_paths.append(os.path.join(extract_dir, path_with_back_slash))
                        
                        # Also try just the last element (likely the filename)
                        if len(img_path_elements) > 0:
                            last_element = img_path_elements[-1]
                            possible_paths.append(os.path.join(extract_dir, last_element))
                            logger.info(f"Added last element as filename: {last_element}")
                        
                        # Special case for DICOM files: try without the DICOM prefix
                        if len(img_path_elements) > 1 and img_path_elements[0].upper() == "DICOM":
                            # Try path without the DICOM prefix
                            for i in range(1, len(img_path_elements)):
                                possible_paths.append(os.path.join(extract_dir, *img_path_elements[i:]))
                                logger.info(f"Added path without DICOM prefix: {os.path.join(*img_path_elements[i:])}")
                    else:
                        logger.warning(f"Parsed value is not a list: {type(parsed_list)}")
                        # Fall back to treating as a normal string
                        possible_paths.append(os.path.join(extract_dir, img_path))
                except Exception as e:
                    logger.warning(f"Failed to parse string as list: {str(e)}")
                    # Fall back to treating as a normal string
                    possible_paths.append(os.path.join(extract_dir, img_path))
            
            # CASE 3: Path is a regular string or other type
            else:
                # Ensure path is a string
                path_str = str(img_path)
                logger.info(f"Processing regular string path: {path_str}")
                
                # Original path as is
                possible_paths.append(os.path.join(extract_dir, path_str))
                
                # Try with different slash types
                if '/' in path_str:
                    possible_paths.append(os.path.join(extract_dir, path_str.replace('/', '\\')))
                if '\\' in path_str:
                    possible_paths.append(os.path.join(extract_dir, path_str.replace('\\', '/')))
                
                # Try just the filename part
                basename = os.path.basename(path_str)
                possible_paths.append(os.path.join(extract_dir, basename))
                logger.info(f"Added basename: {basename}")
                
                # If path has DICOM prefix, try without it
                if path_str.upper().startswith('DICOM'):
                    # Strip DICOM and any path separators
                    no_prefix = path_str[5:].lstrip('/\\')
                    possible_paths.append(os.path.join(extract_dir, no_prefix))
                    logger.info(f"Added path without DICOM prefix: {no_prefix}")
            
            # Search extract directory for matching files
            logger.info(f"Searching extract directory for matching files in: {extract_dir}")
            
            # First try our possible paths
            possible_paths = list(filter(os.path.exists, possible_paths))
            
            # If none of our path strategies worked, try some fallback approaches
            if not possible_paths:
                logger.info("No direct matches found, trying recursive file search...")
                
                # Approach 1: Search for files by filename pattern
                if isinstance(img_path, str) and img_path.startswith('[') and img_path.endswith(']'):
                    # Try to extract the filename from the list-like string
                    try:
                        parsed_list = ast.literal_eval(img_path)
                        if isinstance(parsed_list, list) and len(parsed_list) > 0:
                            # Look for the last element of the list (usually the filename)
                            filename = str(parsed_list[-1])
                            logger.info(f"Searching for file with pattern: {filename}")
                            
                            # Walk the directory tree looking for matching files
                            for root, _, files in os.walk(extract_dir):
                                for file in files:
                                    if filename in file or file in filename:
                                        match_path = os.path.join(root, file)
                                        possible_paths.append(match_path)
                                        logger.info(f"Found potential match: {match_path}")
                    except Exception as e:
                        logger.warning(f"Error parsing list-like path: {str(e)}")
                
                # Approach 2: Look for DICOM directories
                dicom_dirs = []
                for root, dirs, _ in os.walk(extract_dir):
                    for dir_name in dirs:
                        if "dicom" in dir_name.lower():
                            dicom_dirs.append(os.path.join(root, dir_name))
                
                if dicom_dirs:
                    logger.info(f"Found {len(dicom_dirs)} DICOM directories")
                    
                    # For each DICOM directory, look for files with expected patterns
                    for dicom_dir in dicom_dirs:
                        # Common DICOM file patterns
                        patterns = ["I*", "*[0-9]", "*.dcm", "*.DCM"]
                        
                        for pattern in patterns:
                            matches = glob.glob(os.path.join(dicom_dir, pattern))
                            if matches:
                                logger.info(f"Found {len(matches)} files matching pattern {pattern} in {dicom_dir}")
                                possible_paths.extend(matches)
                
                # Approach 3: Just try any file that might be a DICOM
                if not possible_paths:
                    logger.info("Trying any file that might be a DICOM...")
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            if file.startswith('I') and file[1:].isdigit():
                                # Common DICOMDIR naming pattern: I followed by numbers
                                possible_paths.append(os.path.join(root, file))
                                logger.info(f"Found potential DICOM file by pattern: {file}")
            
            logger.info(f"Trying {len(possible_paths)} possible paths for DICOM file")
            
            # Try each path
            found_valid_dicom = False
            for i, path in enumerate(possible_paths[:10]):  # Limit to first 10 to avoid too much processing
                logger.info(f"Trying path: {path}")
                try:
                    # Use the imported function to load DICOM file
                    images = load_and_preprocess_dicom(path)
                    if images:
                        logger.info(f"Successfully loaded DICOM from path: {path}")
                        found_valid_dicom = True
                        break
                except Exception as e:
                    error_message = f"Error loading DICOM from path {path}: {str(e)}"
                    logger.warning(error_message)
                    error_messages.append(error_message)
        
        if not found_valid_dicom:
            error_message = "\n".join(error_messages)
            logger.error(f"Failed to find valid DICOM file. Details:\n{error_message}")
            return None, error_message
        
        # Set up hyperparameters
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0
        length_penalty = 1.0
        temperature = 0.1
        
        # Extract metadata for context if not provided
        if not context:
            try:
                metadata = extract_dicom_metadata(dicom_path)
                context = f"Age:{metadata.get('PatientAge', 'Unknown')}.\nGender:{metadata.get('PatientSex', 'Unknown')}.\n"
                context += f"Indication: {metadata.get('StudyDescription', 'Unknown')}.\n"
                logger.info(f"Extracted context: {context}")
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {str(e)}")
                context = ""
        
        # Run inference
        seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
            model, 
            [dicom_path], 
            context, 
            prompt, 
            modality, 
            task, 
            num_beams, 
            do_sample, 
            min_length, 
            top_p, 
            repetition_penalty, 
            length_penalty, 
            temperature, 
            device
        )
        
        # Load the DICOM image for display
        dicom_img = pydicom.dcmread(dicom_path, force=True)
        pixel_array = dicom_img.pixel_array
        
        # Normalize to 0-255 for display
        pixel_array = ((pixel_array - pixel_array.min()) / 
                      (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB
        if len(pixel_array.shape) == 2:
            pixel_array_rgb = np.stack([pixel_array] * 3, axis=2)
        else:
            # Already has channels
            pixel_array_rgb = pixel_array
        
        # Create PIL image
        original_img = Image.fromarray(pixel_array_rgb)
        
        # Process results based on task
        if task == "segmentation":
            output_images = []
            
            # Add original image
            output_images.append(original_img)
            
            # Add segmentation overlay if available
            if seg_mask_2d and len(seg_mask_2d) > 0:
                # Convert mask to overlay
                mask = seg_mask_2d[0]
                mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
                
                # Create overlay
                overlay = np.array(original_img)
                alpha = 0.5
                overlay[mask > 0] = cv2.addWeighted(
                    overlay[mask > 0],
                    1-alpha,
                    mask_rgb[mask > 0],
                    alpha,
                    0
                )
                
                # Convert overlay to PIL Image
                overlay_img = Image.fromarray(overlay)
                output_images.append(overlay_img)
            
            return output_images, f"DICOM Series: {selected_series['description']}\nSegmentation completed.\n\n{output_text}"
        
        # For classification and report generation, just return the original image and text
        return [original_img], f"DICOM Series: {selected_series['description']}\n\n{output_text}"
        
    except Exception as e:
        logger.error(f"Error processing DICOMDIR series: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Register cleanup function
atexit.register(cleanup_temp_directories)

# Initialize the model at startup
initialize_model()

# Create the Gradio interface
with gr.Blocks(title="MedVersa Image Analysis") as app:
    gr.Markdown("# MedVersa Medical Image Analysis")
    
    with gr.Tabs():
        with gr.TabItem("Standard Images"):
            gr.Markdown("""Upload a medical image and provide prompts to analyze specific features.
            Supported modalities: Chest X-ray (cxr), Dermatology (derm), CT scans (ct).
            Supported tasks: Report generation, Classification, Segmentation.""")
            
            with gr.Row():
                with gr.Column():
                    standard_file = gr.File(
                        label="Upload Image (JPEG, PNG, etc.)",
                        type="binary"
                    )
                    standard_context = gr.Textbox(
                        label="Context (optional)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with shortness of breath.\nComparison: None.",
                        lines=3
                    )
                    standard_prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="How would you characterize the findings from <img0>?",
                        value="How would you characterize the findings from <img0>?"
                    )
                    standard_modality = gr.Dropdown(
                        label="Modality", 
                        choices=SUPPORTED_MODALITIES,
                        value="cxr"
                    )
                    standard_task = gr.Dropdown(
                        label="Task", 
                        choices=SUPPORTED_TASKS,
                        value="report generation"
                    )
                    standard_submit = gr.Button("Analyze")
                
                with gr.Column():
                    standard_gallery = gr.Gallery(label="Results", columns=2, show_label=True, height="auto")
                    standard_results = gr.Textbox(label="Analysis Results", show_label=True, lines=10)
            
            standard_submit.click(
                fn=process_image,
                inputs=[standard_file, standard_context, standard_prompt, standard_modality, standard_task],
                outputs=[standard_gallery, standard_results]
            )
            
            # Add example matching client_example.py
            if os.path.exists(examples_dir):
                example_chest_xray = os.path.join(examples_dir, "c536f749-2326f755-6a65f28f-469affd2-26392ce9.png")
                if os.path.exists(example_chest_xray):
                    gr.Examples(
                        examples=[
                            [
                                example_chest_xray,
                                "Age:30-40.\nGender:F.\nIndication: Patient with end-stage renal disease not on dialysis presents with dyspnea. PICC line placement.\nComparison: None.",
                                "How would you characterize the findings from <img0>?",
                                "cxr",
                                "report generation"
                            ]
                        ],
                        inputs=[standard_file, standard_context, standard_prompt, standard_modality, standard_task]
                    )
        
        with gr.TabItem("DICOM Images"):
            gr.Markdown("""Upload a DICOM file for analysis.
            The system will automatically extract metadata from the DICOM file to provide context.""")
            
            with gr.Row():
                with gr.Column():
                    dicom_file = gr.File(
                        label="Upload DICOM file (.dcm)",
                        type="binary"
                    )
                    dicom_context = gr.Textbox(
                        label="Context (optional, will be auto-extracted if empty)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with shortness of breath.\nComparison: None.",
                        lines=3
                    )
                    dicom_prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="How would you characterize the findings from <img0>?",
                        value="How would you characterize the findings from <img0>?"
                    )
                    dicom_modality = gr.Dropdown(
                        label="Modality", 
                        choices=SUPPORTED_MODALITIES,
                        value="cxr"
                    )
                    dicom_task = gr.Dropdown(
                        label="Task", 
                        choices=SUPPORTED_TASKS,
                        value="report generation"
                    )
                    dicom_submit = gr.Button("Analyze DICOM")
                
                with gr.Column():
                    dicom_gallery = gr.Gallery(label="Results", columns=2, show_label=True, height="auto")
                    dicom_results = gr.Textbox(label="Analysis Results", show_label=True, lines=10)
            
            dicom_submit.click(
                fn=process_dicom,
                inputs=[dicom_file, dicom_context, dicom_prompt, dicom_modality, dicom_task],
                outputs=[dicom_gallery, dicom_results]
            )
            
            # Add DICOM example matching client_example.py format
            dicom_example = os.path.join(examples_dir, "0003.DCM")
            if os.path.exists(dicom_example):
                try:
                    # Validate DICOM file
                    dicom_data = pydicom.dcmread(dicom_example, force=True)
                    logger.info(f"DICOM example validated: {dicom_example}")
                    valid_example = True
                    
                    # Extract patient info if available for better context
                    context = "Age:30-40.\nGender:F.\nIndication: Patient with chest pain.\nComparison: None."
                    if hasattr(dicom_data, 'PatientAge'):
                        age = dicom_data.PatientAge
                        context = f"Age:{age}.\n"
                    if hasattr(dicom_data, 'PatientSex'):
                        gender = 'Male' if dicom_data.PatientSex == 'M' else 'Female' if dicom_data.PatientSex == 'F' else dicom_data.PatientSex
                        context += f"Gender:{gender}.\n"
                    if hasattr(dicom_data, 'StudyDescription'):
                        context += f"Indication: {dicom_data.StudyDescription}.\n"
                    context += "Comparison: None."
                    
                except Exception as e:
                    logger.warning(f"DICOM example validation failed: {str(e)}")
                    valid_example = False
                    context = "Age:30-40.\nGender:F.\nIndication: Patient with chest pain.\nComparison: None."
                    
                # Look for alternate examples if the default one is invalid
                if not valid_example:
                    for filename in os.listdir(examples_dir):
                        if filename.lower().endswith('.dcm'):
                            alt_example = os.path.join(examples_dir, filename)
                            try:
                                dicom_data = pydicom.dcmread(alt_example, force=True)
                                dicom_example = alt_example
                                valid_example = True
                                logger.info(f"Using alternative DICOM example: {alt_example}")
                                break
                            except Exception:
                                continue
                
                # Use the example if valid
                if valid_example:
                    gr.Examples(
                        examples=[
                            [
                                dicom_example,
                                context, 
                                "How would you characterize the findings from <img0>?",
                                "cxr",
                                "report generation"
                            ]
                        ],
                        inputs=[dicom_file, dicom_context, dicom_prompt, dicom_modality, dicom_task]
                    )
        
        with gr.TabItem("NIFTI Images"):
            gr.Markdown("""Upload a NIFTI file (.nii or .nii.gz) for 3D medical image analysis.
            This is particularly useful for CT and MRI volumes.""")
            
            with gr.Row():
                with gr.Column():
                    nifti_file = gr.File(
                        label="Upload NIfTI file (.nii, .nii.gz)",
                        type="binary"
                    )
                    nifti_context = gr.Textbox(
                        label="Context (optional)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with abdominal pain.\nComparison: None.",
                        lines=3
                    )
                    nifti_prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="Segment the liver.",
                        value="Segment the liver."
                    )
                    nifti_modality = gr.Dropdown(
                        label="Modality", 
                        choices=SUPPORTED_MODALITIES,
                        value="ct"
                    )
                    nifti_task = gr.Dropdown(
                        label="Task", 
                        choices=SUPPORTED_TASKS,
                        value="segmentation"
                    )
                    nifti_submit = gr.Button("Analyze NIFTI")
                
                with gr.Column():
                    nifti_gallery = gr.Gallery(label="Results", columns=2, show_label=True, height="auto")
                    nifti_results = gr.Textbox(label="Analysis Results", show_label=True, lines=10)
            
            nifti_submit.click(
                fn=process_nifti,
                inputs=[nifti_file, nifti_context, nifti_prompt, nifti_modality, nifti_task],
                outputs=[nifti_gallery, nifti_results]
            )
            
            # Add NIFTI example matching client_example.py format
            nifti_example = os.path.join(examples_dir, "Case_01013_0000.nii.gz")
            if os.path.exists(nifti_example):
                try:
                    # Verify the file is valid before using it as an example
                    nib.load(nifti_example)
                    valid_example = True
                except Exception as e:
                    logger.warning(f"Example NIFTI file {nifti_example} is invalid: {str(e)}")
                    valid_example = False
                
                # Look for alternative examples if the default one is invalid
                if not valid_example:
                    for filename in os.listdir(examples_dir):
                        if filename.endswith('.nii.gz') or filename.endswith('.nii'):
                            alt_example = os.path.join(examples_dir, filename)
                            try:
                                nib.load(alt_example)
                                nifti_example = alt_example
                                valid_example = True
                                logger.info(f"Using alternative NIFTI example: {alt_example}")
                                break
                            except Exception:
                                continue
                
                if valid_example:
                    gr.Examples(
                        examples=[
                            [
                                nifti_example,
                                "",  # No context needed for liver segmentation
                                "Segment the liver.",
                                "ct",
                                "segmentation"
                            ]
                        ],
                        inputs=[nifti_file, nifti_context, nifti_prompt, nifti_modality, nifti_task]
                    )

        with gr.TabItem("DICOMDIR Browser"):
            gr.Markdown("""Upload a DICOMDIR file or a ZIP file containing DICOMDIR to browse its contents and analyze specific series.
            DICOMDIR files are commonly found on medical CDs/DVDs and contain multiple DICOM images organized by patient, study, and series.
            
            **For ZIP files**: 
            - The ZIP file should contain a DICOMDIR file at any level in the archive
            - All referenced DICOM files should also be included in the ZIP
            - The system will automatically extract the ZIP and locate the DICOMDIR file
            """)
            
            with gr.Row():
                with gr.Column():
                    dicomdir_file = gr.File(
                        label="Upload DICOMDIR file or ZIP with DICOMDIR",
                        type="binary"
                    )
                    dicomdir_submit = gr.Button("Browse DICOMDIR")
                
                with gr.Column():
                    dicomdir_structure = gr.Textbox(label="DICOMDIR Structure", show_label=True, lines=20)
                    dicomdir_path = gr.Textbox(label="Directory Path", visible=False)
                    dicomdir_debug = gr.Textbox(label="Debug Information", show_label=True, lines=5, visible=True)
            
            gr.Markdown("## Analyze Series")
            with gr.Row():
                with gr.Column():
                    series_dropdown = gr.Dropdown(
                        label="Select Series", 
                        choices=[],
                        value=None,
                        interactive=True,
                        allow_custom_value=True
                    )
                    series_context = gr.Textbox(
                        label="Context (optional, will be auto-extracted if empty)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with shortness of breath.",
                        lines=3
                    )
                    series_prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="How would you characterize the findings from <img0>?",
                        value="How would you characterize the findings from <img0>?"
                    )
                    series_modality = gr.Dropdown(
                        label="Modality", 
                        choices=SUPPORTED_MODALITIES,
                        value="cxr"
                    )
                    series_task = gr.Dropdown(
                        label="Task", 
                        choices=SUPPORTED_TASKS,
                        value="report generation"
                    )
                    series_submit = gr.Button("Analyze Series")
                
                with gr.Column():
                    series_gallery = gr.Gallery(label="Results", columns=2, show_label=True, height="auto")
                    series_results = gr.Textbox(label="Analysis Results", show_label=True, lines=10)
            
            dicomdir_submit.click(
                fn=lambda: "Preparing to process DICOMDIR file...",
                inputs=[],
                outputs=[dicomdir_debug]
            ).then(
                fn=process_dicomdir_file,
                inputs=[dicomdir_file],
                outputs=[dicomdir_structure, dicomdir_path, series_dropdown]
            ).then(
                fn=lambda structure, path, dropdown: f"Processing complete. Found {len(dropdown) if dropdown else 0} series." if path else f"Error: {structure}",
                inputs=[dicomdir_structure, dicomdir_path, series_dropdown],
                outputs=[dicomdir_debug]
            )
            
            series_submit.click(
                fn=lambda: "Preparing to analyze series...",
                inputs=[],
                outputs=[series_results]
            ).then(
                fn=lambda series_desc: f"Processing series: {series_desc}" if isinstance(series_desc, str) else f"Processing series list: {series_desc}",
                inputs=[series_dropdown],
                outputs=[series_results]
            ).then(
                fn=process_dicomdir_series,
                inputs=[dicomdir_path, series_dropdown, series_context, series_prompt, series_modality, series_task],
                outputs=[series_gallery, series_results]
            )

# Launch the app
if __name__ == "__main__":
    display_splash_screen()
    app.launch(
        share=True, 
        server_name="0.0.0.0",
        server_port=7861,
        debug=True  # Enable debug mode to see more detailed error messages
    ) 