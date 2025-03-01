import os
import logging
import numpy as np
from PIL import Image
import pydicom
import nibabel as nib
import cv2
import tempfile
import json
import io
import zipfile
import shutil
from pathlib import Path

# Set up logging
logger = logging.getLogger("medversa_utils")

def normalize_array(array, dtype=np.uint8):
    """Normalize numpy array to 0-255 range for visualization."""
    if array.max() == array.min():
        return np.zeros_like(array, dtype=dtype)
    normalized = ((array - array.min()) / (array.max() - array.min()) * 255).astype(dtype)
    return normalized

def extract_dicom_metadata(dicom_path):
    """Extract metadata from a DICOM file."""
    try:
        dcm = pydicom.dcmread(dicom_path)
        metadata = {}
        
        # Extract key metadata fields
        for tag in ['PatientName', 'PatientID', 'PatientAge', 'PatientSex', 
                   'StudyDescription', 'SeriesDescription', 'Modality']:
            if hasattr(dcm, tag):
                value = getattr(dcm, tag)
                if value:
                    metadata[tag] = str(value)
            else:
                metadata[tag] = "Unknown"
        
        return metadata
    except Exception as e:
        logger.warning(f"Failed to extract DICOM metadata: {str(e)}")
        return {"error": str(e)}

def load_and_preprocess_image(image_path):
    """Load and preprocess a standard image file (JPEG, PNG, etc.)."""
    try:
        # Handle different input types
        if isinstance(image_path, str):
            # It's a file path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Open the image using PIL
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, bytes):
            # It's bytes data
            image = Image.open(io.BytesIO(image_path)).convert('RGB')
        elif isinstance(image_path, Image.Image):
            # It's already a PIL Image
            image = image_path.convert('RGB')
        else:
            logger.error(f"Unsupported image input type: {type(image_path)}")
            return None
        
        # Resize image if needed for the model
        # This depends on your model's requirements
        # image = image.resize((224, 224))
        
        return image
    
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None

def load_and_preprocess_dicom(dicom_path):
    """Load and preprocess a DICOM file."""
    try:
        # Handle different input types
        if isinstance(dicom_path, bytes):
            # Create a temporary file to save the bytes data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
                tmp.write(dicom_path)
                dicom_path = tmp.name
        
        # Read the DICOM file
        dcm = pydicom.dcmread(dicom_path)
        
        # Get the pixel array
        pixel_array = dcm.pixel_array
        logger.info(f"DICOM info - Shape: {pixel_array.shape}, Type: {pixel_array.dtype}")
        
        # Handle different pixel array shapes
        if len(pixel_array.shape) == 2:
            # Single 2D image
            # Normalize and convert to uint8 for visualization
            normalized = normalize_array(pixel_array)
            
            # Convert to RGB for consistency
            if len(normalized.shape) == 2:
                rgb_image = np.stack([normalized] * 3, axis=2)
            else:
                rgb_image = normalized
                
            # Create PIL image
            image = Image.fromarray(rgb_image)
            return [image]  # Return as a list for consistent interface
            
        elif len(pixel_array.shape) == 3:
            # Stack of 2D images or RGB image
            if pixel_array.shape[0] <= 4:  # Likely an RGB/RGBA image
                # Handle as single image with channels
                normalized = normalize_array(pixel_array)
                image = Image.fromarray(normalized)
                return [image]
            else:
                # Handle as stack of 2D slices
                images = []
                for i in range(pixel_array.shape[0]):
                    slice_data = pixel_array[i]
                    normalized = normalize_array(slice_data)
                    # Convert to RGB
                    rgb_slice = np.stack([normalized] * 3, axis=2)
                    image = Image.fromarray(rgb_slice)
                    images.append(image)
                return images
                
        else:
            # Handle 4D or higher data - take the first element
            logger.warning(f"DICOM has unusual dimensions: {pixel_array.shape}. Taking first slice.")
            first_slice = pixel_array[0]
            normalized = normalize_array(first_slice)
            
            # Convert to RGB if needed
            if len(normalized.shape) == 2:
                rgb_image = np.stack([normalized] * 3, axis=2)
            else:
                rgb_image = normalized
                
            image = Image.fromarray(rgb_image)
            return [image]
    
    except Exception as e:
        logger.error(f"DICOM processing error: {str(e)}")
        logger.error(f"DICOM details: {dicom_path}")
        if 'dcm' in locals() and hasattr(dcm, 'pixel_array'):
            logger.error(f"Pixel array shape: {dcm.pixel_array.shape}, dtype: {dcm.pixel_array.dtype}")
        
        # Try an alternative approach for complex DICOMs
        try:
            logger.info("Attempting alternative DICOM processing...")
            dcm = pydicom.dcmread(dicom_path)
            
            # Try to convert using pydicom's built-in method
            arr = dcm.pixel_array
            if arr.ndim == 2:
                # Single slice
                # Use the dcm.PhotometricInterpretation to determine how to handle pixel data
                if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
                    # Invert for MONOCHROME1 photometric interpretation
                    if hasattr(dcm, 'BitsStored'):
                        max_val = 2**dcm.BitsStored - 1
                    else:
                        max_val = np.max(arr)
                    arr = max_val - arr
                
                # Normalize to 0-255
                if arr.max() > 255 or arr.min() < 0 or arr.dtype != np.uint8:
                    arr = normalize_array(arr)
                
                # Convert to RGB
                rgb_array = np.stack([arr] * 3, axis=2)
                image = Image.fromarray(rgb_array.astype(np.uint8))
                return [image]
            elif arr.ndim == 3:
                # Multiple slices or RGB
                images = []
                for i in range(arr.shape[0]):
                    slice_arr = arr[i]
                    # Normalize
                    if slice_arr.max() > 255 or slice_arr.min() < 0 or slice_arr.dtype != np.uint8:
                        slice_arr = normalize_array(slice_arr)
                    
                    # Convert to RGB
                    rgb_slice = np.stack([slice_arr] * 3, axis=2)
                    image = Image.fromarray(rgb_slice.astype(np.uint8))
                    images.append(image)
                return images
            else:
                logger.error(f"Unsupported array dimensions: {arr.ndim}")
                return None
                
        except Exception as nested_e:
            logger.error(f"Alternative DICOM processing also failed: {str(nested_e)}")
            return None

def load_and_preprocess_dicom_volume(dicom_path):
    """Load and preprocess a DICOM file as a 3D volume."""
    # This is a fallback for 3D DICOM processing
    return load_and_preprocess_dicom(dicom_path)

def load_and_preprocess_dicom_series(directory_path):
    """Load and preprocess a series of DICOM files from a directory."""
    try:
        # Find all DICOM files in the directory
        dicom_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or file.lower() == 'dicomdir':
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            logger.warning(f"No DICOM files found in {directory_path}")
            return None
        
        # Sort files (this might need customization based on your DICOM series)
        dicom_files.sort()
        
        # Load each file
        images = []
        for file_path in dicom_files:
            file_images = load_and_preprocess_dicom(file_path)
            if file_images:
                images.extend(file_images)
        
        return images
    
    except Exception as e:
        logger.error(f"Error loading DICOM series: {str(e)}")
        return None

def load_and_preprocess_volume(nifti_path):
    """Load and preprocess a NIFTI volume."""
    try:
        # Handle different input types
        if isinstance(nifti_path, bytes):
            # Create a temporary file to save the bytes data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                tmp.write(nifti_path)
                nifti_path = tmp.name
        
        # Load the NIFTI file
        nifti_img = nib.load(nifti_path)
        
        # Get the data array
        nifti_data = nifti_img.get_fdata()
        
        # Get dimensions
        if len(nifti_data.shape) < 3:
            logger.warning(f"NIFTI has unexpected dimensions: {nifti_data.shape}")
            return None
        
        # Extract slices for different orientations
        depths = nifti_data.shape[2]  # Assuming the third dimension is the depth
        
        # Take slices at regular intervals
        slice_images = []
        step = max(1, depths // 10)  # Take about 10 slices or less
        
        for i in range(0, depths, step):
            # Get the slice
            slice_data = nifti_data[:, :, i]
            
            # Normalize to 0-255 for display
            normalized = normalize_array(slice_data)
            
            # Convert to RGB
            rgb_slice = np.stack([normalized] * 3, axis=2)
            
            # Create PIL image
            slice_img = Image.fromarray(rgb_slice)
            slice_images.append(slice_img)
        
        return slice_images
    
    except Exception as e:
        logger.error(f"Error loading NIFTI: {str(e)}")
        try:
            # Try an alternative approach using nibabel's built-in methods
            img = nib.load(nifti_path)
            data = img.get_fdata()
            logger.info(f"NIFTI shape: {data.shape}, dtype: {data.dtype}")
            
            # Get orthogonal views
            middle_indices = [dim // 2 for dim in data.shape[:3]]
            
            # Extract the middle slices in different directions
            axial_slice = data[:, :, middle_indices[2]]
            sagittal_slice = data[middle_indices[0], :, :]
            coronal_slice = data[:, middle_indices[1], :]
            
            # Normalize and convert to RGB
            slices = [axial_slice, sagittal_slice, coronal_slice]
            images = []
            
            for slice_data in slices:
                normalized = normalize_array(slice_data)
                rgb_slice = np.stack([normalized] * 3, axis=2)
                image = Image.fromarray(rgb_slice)
                images.append(image)
            
            return images
            
        except Exception as nested_e:
            logger.error(f"Alternative NIFTI processing also failed: {str(nested_e)}")
            return None

def process_dicomdir(dicomdir_path):
    """Process a DICOMDIR file to extract structure information."""
    try:
        # Read the DICOMDIR file
        dcmdir = pydicom.dcmread(dicomdir_path)
        
        # Create a dictionary to store the structure
        directory = {
            "_path": dicomdir_path,  # Store the path for later use
            "patients": []
        }
        
        # Process patient records
        for patient_record in dcmdir.patient_records:
            patient = {
                "id": getattr(patient_record, "PatientID", "Unknown"),
                "name": str(getattr(patient_record, "PatientName", "Unknown")),
                "studies": []
            }
            
            # Process study records
            if hasattr(patient_record, "children"):
                for study_record in patient_record.children:
                    if study_record.DirectoryRecordType == "STUDY":
                        study = {
                            "id": getattr(study_record, "StudyInstanceUID", "Unknown"),
                            "date": str(getattr(study_record, "StudyDate", "Unknown")),
                            "description": getattr(study_record, "StudyDescription", "Unknown"),
                            "series": []
                        }
                        
                        # Process series records
                        if hasattr(study_record, "children"):
                            for series_record in study_record.children:
                                if series_record.DirectoryRecordType == "SERIES":
                                    series = {
                                        "id": getattr(series_record, "SeriesInstanceUID", "Unknown"),
                                        "number": getattr(series_record, "SeriesNumber", "Unknown"),
                                        "description": getattr(series_record, "SeriesDescription", "Unknown"),
                                        "modality": getattr(series_record, "Modality", "Unknown"),
                                        "images": []
                                    }
                                    
                                    # Process image records
                                    if hasattr(series_record, "children"):
                                        for image_record in series_record.children:
                                            if image_record.DirectoryRecordType == "IMAGE":
                                                image = {
                                                    "id": getattr(image_record, "SOPInstanceUID", "Unknown"),
                                                    "number": getattr(image_record, "InstanceNumber", 0),
                                                    "path": getattr(image_record, "ReferencedFileID", "Unknown")
                                                }
                                                
                                                # Handle path format (convert from list to string)
                                                if isinstance(image["path"], pydicom.multival.MultiValue):
                                                    image["path"] = os.path.join(*image["path"])
                                                
                                                series["images"].append(image)
                                    
                                    study["series"].append(series)
                        
                        patient["studies"].append(study)
            
            directory["patients"].append(patient)
        
        return directory
    
    except Exception as e:
        logger.error(f"Error processing DICOMDIR: {str(e)}")
        return {"error": str(e)}

def process_zip_with_dicomdir(zip_bytes):
    """Extract a ZIP file containing DICOMDIR and return the structure."""
    try:
        # Create a temporary directory to extract ZIP contents
        extract_dir = tempfile.mkdtemp()
        
        # Create a temporary zip file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            tmp.write(zip_bytes)
            zip_path = tmp.name
        
        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up the temporary zip file
        os.unlink(zip_path)
        
        # Find DICOMDIR file
        dicomdir_path = None
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.upper() == 'DICOMDIR':
                    dicomdir_path = os.path.join(root, file)
                    break
            if dicomdir_path:
                break
        
        if not dicomdir_path:
            logger.error("DICOMDIR file not found in ZIP archive")
            return {"error": "DICOMDIR file not found in ZIP archive"}
        
        # Process the DICOMDIR file
        directory = process_dicomdir(dicomdir_path)
        
        # Store the extraction directory for later cleanup
        directory["_extract_dir"] = extract_dir
        
        return directory
    
    except Exception as e:
        logger.error(f"Error processing ZIP with DICOMDIR: {str(e)}")
        # Clean up the extraction directory if it exists
        if 'extract_dir' in locals():
            shutil.rmtree(extract_dir)
        return {"error": str(e)}

def dicom_value_to_serializable(value):
    """Convert DICOM data values to JSON serializable objects."""
    if hasattr(value, "original_string"):
        return str(value)
    elif hasattr(value, "real"):  # Handle complex numbers
        return str(value)
    elif isinstance(value, bytes):
        return f"[BINARY DATA: {len(value)} bytes]"
    elif isinstance(value, (pydicom.sequence.Sequence, list)):
        return [dicom_value_to_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: dicom_value_to_serializable(v) for k, v in value.items()}
    else:
        return value

def load_dicom_from_dicomdir(dicomdir_path, image_path):
    """Load a DICOM file referenced in DICOMDIR."""
    try:
        # Get the directory containing the DICOMDIR file
        dicomdir_dir = os.path.dirname(dicomdir_path)
        
        # Construct the full path to the referenced DICOM file
        full_path = os.path.join(dicomdir_dir, image_path)
        
        # Check if the file exists
        if not os.path.exists(full_path):
            # Try alternative path separators
            alt_path = os.path.join(dicomdir_dir, image_path.replace('/', '\\'))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                alt_path = os.path.join(dicomdir_dir, image_path.replace('\\', '/'))
                if os.path.exists(alt_path):
                    full_path = alt_path
                else:
                    logger.error(f"Referenced DICOM file not found: {image_path}")
                    return None
        
        # Load and preprocess the DICOM file
        return load_and_preprocess_dicom(full_path)
    
    except Exception as e:
        logger.error(f"Error loading DICOM from DICOMDIR: {str(e)}")
        return None

def cleanup_extracted_zip(directory_path):
    """Clean up extracted ZIP directory."""
    try:
        if directory_path and os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            logger.info(f"Cleaned up extracted directory: {directory_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up directory: {str(e)}") 