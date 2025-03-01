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

# Add MedVersa_Internal to path
repo_root = str(Path(__file__).parent)
sys.path.append(repo_root)

# Set examples directory
examples_dir = os.path.join(repo_root, "demo_ex")

# Import MedVersa utilities
from utils import (
    extract_dicom_metadata,
    load_and_preprocess_dicom,
    load_and_preprocess_dicom_series,
    load_and_preprocess_dicom_volume,
    load_and_preprocess_image,
    load_and_preprocess_volume,
    registry,
    generate,
    generate_predictions
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
            # Read the file content
            with open(image_path, 'rb') as f:
                file_content = f.read()
        elif isinstance(image_file, bytes):  # If bytes are provided directly
            file_content = image_file
            filename = "uploaded_image.bin"  # Default filename
            logger.info(f"Received image as bytes, size: {len(file_content)} bytes")
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

        # Set up hyperparameters
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0
        length_penalty = 1.0
        temperature = 0.1

        # Run inference
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
            if modality == "ct" and seg_mask_3d is not None:
                # For 3D segmentation, return the middle slice for preview
                middle_idx = len(seg_mask_3d) // 2
                
                # Create a list of images for the gallery
                output_images = []
                
                # Add original image
                original_img = Image.fromarray(np.array(Image.open(image_path).convert("RGB")))
                output_images.append(original_img)
                
                # Add segmentation overlay
                if seg_mask_3d and len(seg_mask_3d) > 0:
                    # Convert mask to overlay
                    mask = seg_mask_3d[middle_idx]
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
                    
                    overlay_img = Image.fromarray(overlay)
                    output_images.append(overlay_img)
                
                return output_images, f"3D Segmentation completed. Total slices: {len(seg_mask_3d)}\n\n{output_text}"
            
            elif seg_mask_2d is not None:
                # For 2D segmentation
                output_images = []
                
                # Add original image
                original_img = Image.fromarray(np.array(Image.open(image_path).convert("RGB")))
                output_images.append(original_img)
                
                # Add segmentation overlay
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
                    
                    overlay_img = Image.fromarray(overlay)
                    output_images.append(overlay_img)
                
                return output_images, f"2D Segmentation completed.\n\n{output_text}"
        
        # For classification and report generation, just return the original image and text
        original_img = Image.fromarray(np.array(Image.open(image_path).convert("RGB")))
        return [original_img], output_text

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
            # Read the file content
            with open(dicom_path, 'rb') as f:
                file_content = f.read()
        elif isinstance(dicom_file, bytes):  # If bytes are provided directly
            file_content = dicom_file
            filename = "uploaded_dicom.dcm"  # Default filename
            logger.info(f"Received DICOM as bytes, size: {len(file_content)} bytes")
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
        
        # Check if file content is empty
        if not file_content or len(file_content) == 0:
            return None, "Error: Uploaded file is empty"
            
        # Save to a temporary file to get a path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
            tmp.write(file_content)
            tmp.flush()  # Ensure data is written to disk
            dicom_path = tmp.name
            logger.info(f"Saved uploaded DICOM file to temporary path: {dicom_path}")

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
                context = f"Age:{metadata.get('PatientAge', 'Unknown')}.\nGender:{metadata.get('PatientSex', 'Unknown')}.\n"
                context += f"Indication: {metadata.get('StudyDescription', 'Unknown')}.\n"
                logger.info(f"Extracted context: {context}")
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {str(e)}")
                context = ""

        # Set up hyperparameters
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0
        length_penalty = 1.0
        temperature = 0.1

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

        # Process results based on task
        if task == "segmentation":
            # Load the DICOM image for display
            dicom_images = await load_and_preprocess_dicom(dicom_path)
            
            if not dicom_images:
                return None, "Failed to load DICOM image"
            
            # For 2D segmentation
            output_images = []
            
            # Add original image
            original_img = dicom_images[0]
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
                
                overlay_img = Image.fromarray(overlay)
                output_images.append(overlay_img)
            
            return output_images, f"DICOM Segmentation completed.\n\n{output_text}"
        
        # For classification and report generation, just return the original image and text
        dicom_images = await load_and_preprocess_dicom(dicom_path)
        if not dicom_images:
            return None, "Failed to load DICOM image"
        
        return [dicom_images[0]], output_text

    except Exception as e:
        logger.error(f"Error processing DICOM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

async def process_nifti(nifti_file, context, prompt, modality, task):
    """Process a NIfTI file with the model."""
    try:
        if model is None:
            if not initialize_model():
                return None, "Model initialization failed"

        # Read file content
        if isinstance(nifti_file, str):  # If path is provided
            nifti_path = nifti_file
            filename = os.path.basename(nifti_file)
            logger.info(f"Using provided NIfTI file path: {nifti_path}")
            # Read the file content
            with open(nifti_path, 'rb') as f:
                file_content = f.read()
        elif isinstance(nifti_file, bytes):  # If bytes are provided directly
            file_content = nifti_file
            filename = "uploaded_nifti.nii.gz"  # Default filename
            logger.info(f"Received NIfTI as bytes, size: {len(file_content)} bytes")
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
            
            logger.info(f"Read {len(file_content)} bytes from uploaded NIfTI file: {filename}")
        
        # Check if file content is empty
        if not file_content or len(file_content) == 0:
            return None, "Error: Uploaded file is empty"
            
        # Save to a temporary file to get a path
        suffix = '.nii.gz' if filename.endswith('.gz') else '.nii'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp.flush()  # Ensure data is written to disk
            nifti_path = tmp.name
            logger.info(f"Saved uploaded NIfTI file to temporary path: {nifti_path}")

        logger.info(f"Processing NIfTI file: {filename}")
        
        # Verify the file exists and is not empty
        if not os.path.exists(nifti_path):
            logger.error(f"NIFTI file not found at path: {nifti_path}")
            return None, f"Error: NIFTI file not found at {nifti_path}"
            
        file_size = os.path.getsize(nifti_path)
        if file_size == 0:
            logger.error(f"NIFTI file is empty: {filename}, size: {file_size} bytes")
            return None, f"Error: NIFTI file is empty: {filename}"
        else:
            logger.info(f"NIFTI file size: {file_size} bytes")

        # Set up hyperparameters
        num_beams = 1
        do_sample = True
        min_length = 1
        top_p = 0.9
        repetition_penalty = 1.0
        length_penalty = 1.0
        temperature = 0.1

        # Run inference
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

        # For segmentation tasks, create visualizations
        if task == "segmentation" and seg_mask_3d is not None:
            # Create a list of images for the gallery
            output_images = []
            
            # Get a few representative slices from the volume
            try:
                import nibabel as nib
                nifti_img = nib.load(nifti_path)
                nifti_data = nifti_img.get_fdata()
                
                # Get dimensions
                depth = nifti_data.shape[2]
                
                # Select slices at different positions
                slice_positions = [depth // 4, depth // 2, 3 * depth // 4]
                
                for pos in slice_positions:
                    # Get the slice
                    slice_data = nifti_data[:, :, pos]
                    
                    # Normalize to 0-255 for display
                    slice_data = ((slice_data - slice_data.min()) / 
                                 (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
                    
                    # Convert to RGB
                    slice_rgb = np.stack([slice_data] * 3, axis=2)
                    
                    # Create PIL image
                    slice_img = Image.fromarray(slice_rgb)
                    output_images.append(slice_img)
                    
                    # If we have a segmentation mask, create an overlay
                    if seg_mask_3d and len(seg_mask_3d) > pos:
                        # Get the mask for this slice
                        mask = seg_mask_3d[pos]
                        
                        # Create colored mask
                        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                        mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
                        
                        # Create overlay
                        overlay = slice_rgb.copy()
                        alpha = 0.5
                        overlay[mask > 0] = cv2.addWeighted(
                            overlay[mask > 0],
                            1-alpha,
                            mask_rgb[mask > 0],
                            alpha,
                            0
                        )
                        
                        overlay_img = Image.fromarray(overlay)
                        output_images.append(overlay_img)
            
            except Exception as e:
                logger.error(f"Error creating NIFTI visualizations: {str(e)}")
                import traceback
                traceback.print_exc()
            
            return output_images, f"3D Segmentation completed. Total slices: {len(seg_mask_3d) if seg_mask_3d else 0}\n\n{output_text}"
        
        # For other tasks, just return a representative slice
        try:
            import nibabel as nib
            nifti_img = nib.load(nifti_path)
            nifti_data = nifti_img.get_fdata()
            
            # Get the middle slice
            middle_slice = nifti_data[:, :, nifti_data.shape[2] // 2]
            
            # Normalize to 0-255 for display
            middle_slice = ((middle_slice - middle_slice.min()) / 
                           (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            # Convert to RGB
            middle_slice_rgb = np.stack([middle_slice] * 3, axis=2)
            
            # Create PIL image
            middle_img = Image.fromarray(middle_slice_rgb)
            
            return [middle_img], output_text
            
        except Exception as e:
            logger.error(f"Error creating NIFTI visualization: {str(e)}")
            return None, f"Error visualizing NIFTI, but analysis completed: {output_text}"

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
                import shutil
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
                    series_desc = f"{series['description']} - {series['modality']} ({len(series['images'])} images)"
                    
                    series_mapping[series_desc] = series_key
                    series_keys.append((series_desc, series_key))
        
        # Store the mapping in the directory structure
        directory["_series_mapping"] = series_mapping
        
        # Save the directory structure to a temporary file for later reference
        directory_path = tempfile.mktemp(suffix='.json')
        with open(directory_path, 'w') as f:
            json.dump(directory, f, default=str)
        logger.info(f"Saved directory structure to: {directory_path}")
        
        return "\n".join(formatted_structure), directory_path, [desc for desc, _ in series_keys]
    
    except Exception as e:
        logger.error(f"Error processing DICOMDIR: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, []

async def process_dicomdir_series(directory_path, series_desc, context, prompt, modality, task):
    """Process a specific series from a DICOMDIR."""
    try:
        if not directory_path or not os.path.exists(directory_path):
            return None, "Directory information not found"
        
        # Load the directory structure
        with open(directory_path, 'r') as f:
            directory = json.load(f)
        
        # Get the series key from the description
        series_mapping = directory.get("_series_mapping", {})
        series_key = series_mapping.get(series_desc)
        
        if not series_key:
            # If not found in mapping, check if it's already a key
            logger.warning(f"Series description not found in mapping: {series_desc}")
            # Try to use it directly as a key
            series_key = series_desc
        
        logger.info(f"Using series key: {series_key}")
        
        # Parse the series key
        try:
            patient_id, study_id, series_id = series_key.split('_')
        except ValueError:
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
            
            # Original path as is
            possible_paths.append(os.path.join(extract_dir, middle_image["path"]))
            
            # Try with backslash instead of forward slash
            if '/' in middle_image["path"]:
                possible_paths.append(os.path.join(extract_dir, middle_image["path"].replace('/', '\\')))
            
            # Try with forward slash instead of backslash
            if '\\' in middle_image["path"]:
                possible_paths.append(os.path.join(extract_dir, middle_image["path"].replace('\\', '/')))
            
            # Try with just the filename
            possible_paths.append(os.path.join(extract_dir, os.path.basename(middle_image["path"])))
            
            # Try searching for the file by name in the extract directory
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower() == os.path.basename(middle_image["path"]).lower():
                        possible_paths.append(os.path.join(root, file))
            
            # Try each possible path
            for path in possible_paths:
                if os.path.exists(path):
                    dicom_path = path
                    logger.info(f"Found DICOM file at: {dicom_path}")
                    break
            
            if not dicom_path:
                error_messages.append(f"Could not find DICOM file at any of these paths:")
                for path in possible_paths:
                    error_messages.append(f"  - {path}")
        
        if not dicom_path:
            return None, f"Could not find DICOM file. Errors: {'; '.join(error_messages)}"
        
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
                        type="binary"  # Explicitly set to binary mode
                    )
                    standard_context = gr.Textbox(
                        label="Context (optional)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with shortness of breath.",
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
            
            # Add examples if available
            if os.path.exists(examples_dir):
                example_files = [
                    os.path.join(examples_dir, f) for f in os.listdir(examples_dir) 
                    if f.endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(examples_dir, f))
                ]
                
                if example_files:
                    gr.Examples(
                        examples=[
                            [
                                example_files[0], 
                                "Age:30-40.\nGender:F.\nIndication: Patient with shortness of breath.", 
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
                        type="binary"  # Explicitly set to binary mode
                    )
                    dicom_context = gr.Textbox(
                        label="Context (optional, will be auto-extracted if empty)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with shortness of breath.",
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
            
            # Add DICOM example if available
            dicom_example = os.path.join(repo_root, "0003.DCM")
            if os.path.exists(dicom_example):
                gr.Examples(
                    examples=[
                        [
                            dicom_example, 
                            "", 
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
                        type="binary"  # Explicitly set to binary mode
                    )
                    nifti_context = gr.Textbox(
                        label="Context (optional)", 
                        placeholder="Age:30-40.\nGender:F.\nIndication: Patient with abdominal pain.",
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
            
            # Add NIFTI example if available
            nifti_examples = [f for f in os.listdir(examples_dir) 
                             if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(examples_dir, f))]
            
            if nifti_examples:
                nifti_example = os.path.join(examples_dir, nifti_examples[0])
                gr.Examples(
                    examples=[
                        [
                            nifti_example, 
                            "", 
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
                        type="binary"  # Explicitly set to binary mode
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