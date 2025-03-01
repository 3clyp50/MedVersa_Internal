import requests
import json
import base64
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import zipfile
import tempfile
import traceback
import pydicom

# Server URL
SERVER_URL = "http://localhost:5000"

def encode_image(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def encode_dicom_series(dicom_dir):
    """
    Encode a directory of DICOM files as a zip archive in base64.
    
    Parameters:
    dicom_dir (str): Path to directory containing DICOM files
    
    Returns:
    str: Base64 encoded zip file containing the DICOM series
    """
    # Create a temporary file for the zip
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip.close()
    
    # Create a zip file containing all DICOM files
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for root, _, files in os.walk(dicom_dir):
            for file in files:
                if file.endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    # Add file to zip, preserving the relative path
                    arcname = os.path.relpath(file_path, dicom_dir)
                    zipf.write(file_path, arcname=arcname)
    
    # Read and encode the zip file
    with open(temp_zip.name, 'rb') as f:
        zip_data = f.read()
    
    # Clean up
    os.unlink(temp_zip.name)
    
    return base64.b64encode(zip_data).decode('utf-8')

def decode_mask(base64_string):
    """Decode a base64 string to a numpy array."""
    bytes_data = base64.b64decode(base64_string)
    bytes_io = io.BytesIO(bytes_data)
    return np.load(bytes_io)

def overlay_mask(image_path, mask):
    """Overlay a segmentation mask on an image."""
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def check_health():
    """Check if the server is running and the model is loaded."""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        return response.json()
    except Exception as e:
        print(f"Error checking health: {e}")
        return None

def get_model_info():
    """Get information about the model."""
    try:
        response = requests.get(f"{SERVER_URL}/info")
        return response.json()
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def process_request(request_data):
    """Process a request through the MedVersa API."""
    url = f"{SERVER_URL}/predict"
    headers = {"Content-Type": "application/json"}
    
    print("\nSending request to MedVersa API...")
    try:
        print(f"Request details: Modality={request_data['modality']}, Task={request_data['task']}")
        print(f"Prompt: {request_data['prompt']}")
        
        # Send the request
        response = requests.post(url, headers=headers, json=request_data, timeout=120)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Request successful!")
            result = response.json()
            
            # Print any messages from the server
            if "message" in result:
                print(f"Server message: {result['message']}")
                
            return result
        else:
            print(f"\n======== ERROR RESPONSE ========")
            print(f"Request failed with status code {response.status_code}")
            
            # Try to extract any error message from the response
            try:
                error_data = response.json()
                if "error" in error_data:
                    print(f"Server error: {error_data['error']}")
                    if "traceback" in error_data:
                        print("\nServer traceback:")
                        print(error_data['traceback'])
                elif "detail" in error_data:
                    print(f"Server detail: {error_data['detail']}")
                else:
                    print(f"Server response data: {json.dumps(error_data, indent=2)}")
                return error_data
            except Exception as e:
                print(f"Server response text (not JSON): {response.text}")
                return {"error": response.text}
            
    except requests.exceptions.Timeout:
        print(f"\n======== ERROR ========")
        print(f"Request timed out after 120 seconds. The server might be processing a complex input.")
        print("Try increasing the timeout value or check server status.")
        return {"error": "Request timed out after 120 seconds"}
    except requests.exceptions.ConnectionError:
        print(f"\n======== ERROR ========")
        print(f"Connection error. The server might be down or unreachable.")
        print(f"Server URL: {SERVER_URL}")
        print("Please check that the server is running and accessible.")
        return {"error": "Connection error - server unreachable"}
    except Exception as e:
        print(f"\n======== ERROR ========")
        print(f"Failed to send request: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        return {"error": str(e)}

def predict(images, modality, task, context="", prompt="", hyperparams=None):
    """Make a prediction with the model."""
    if hyperparams is None:
        hyperparams = {}
    
    # Prepare the request data
    request_data = {
        "images": images,
        "modality": modality,
        "task": task,
        "context": context,
        "prompt": prompt,
        "hyperparams": hyperparams
    }
    
    # Send the request
    try:
        response = requests.post(
            f"{SERVER_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Always return the JSON response, even for error codes
        # This preserves the error message from the server
        try:
            return response.json()
        except json.JSONDecodeError:
            # Return error if response is not valid JSON
            return {
                "error": f"Server returned status code {response.status_code}, but response is not valid JSON: {response.text[:100]}..."
            }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return {
            "error": f"Connection error: {str(e)}"
        }

def example_chest_xray_report():
    """Example: Generate a report for a chest X-ray."""
    print("\n=== Chest X-ray Report Generation ===")
    
    # Encode the image
    image_path = "./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    encoded_image = encode_image(image_path)
    
    # Prepare the request
    images = [{
        "data": encoded_image,
        "format": "png"
    }]
    
    # Patient context
    context = "Age:30-40.\nGender:F.\nIndication: Patient with end-stage renal disease not on dialysis presents with dyspnea. PICC line placement.\nComparison: None."
    
    # Prompt
    prompt = "How would you characterize the findings from <img0>?"
    
    # Make the prediction
    result = predict(images, "cxr", "report generation", context, prompt)
    
    if result:
        print("\nGenerated Report:")
        print(result["text"])
        print(f"\nProcessing time: {result['processing_time']:.2f} seconds")

def example_dermatology_classification():
    """Example: Classify a dermatology image."""
    print("\n=== Dermatology Classification ===")
    
    # Encode the image
    image_path = "./demo_ex/ISIC_0032258.jpg"
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    encoded_image = encode_image(image_path)
    
    # Prepare the request
    images = [{
        "data": encoded_image,
        "format": "jpg"
    }]
    
    # Patient context
    context = "Age:70.\nGender:female.\nLocation:back."
    
    # Prompt
    prompt = "What is the primary diagnosis?"
    
    # Make the prediction
    result = predict(images, "derm", "classification", context, prompt)
    
    if result:
        print("\nClassification Result:")
        print(result["text"])
        print(f"\nProcessing time: {result['processing_time']:.2f} seconds")

def example_dermatology_segmentation():
    """Example: Segment a lesion in a dermatology image."""
    print("\n=== Dermatology Segmentation ===")
    
    # Encode the image
    image_path = "./demo_ex/ISIC_0032258.jpg"
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    encoded_image = encode_image(image_path)
    
    # Prepare the request
    images = [{
        "data": encoded_image,
        "format": "jpg"
    }]
    
    # Patient context
    context = "Age:70.\nGender:female.\nLocation:back."
    
    # Prompt
    prompt = "Segment the lesion."
    
    # Make the prediction
    result = predict(images, "derm", "segmentation", context, prompt)
    
    if result and "segmentation_2d" in result:
        print("\nSegmentation completed.")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        # Decode and display the mask
        mask = decode_mask(result["segmentation_2d"][0])
        overlay_mask(image_path, mask)
    else:
        print("No segmentation mask returned.")

def example_chest_xray_dicom(dicom_path):
    """
    Example of generating a report for a chest X-ray in DICOM format.
    
    Parameters:
    dicom_path (str): Path to a chest X-ray DICOM file
    """
    print("\n=== Chest X-ray DICOM Report Generation ===")
    
    if not os.path.exists(dicom_path):
        print(f"ERROR: DICOM file not found: {dicom_path}")
        return
        
    # Validate that the file is a DICOM file
    try:
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        print(f"DICOM loaded successfully. Modality: {dicom_data.Modality if hasattr(dicom_data, 'Modality') else 'Unknown'}")
        print(f"Image size: {dicom_data.pixel_array.shape}")
    except Exception as e:
        print(f"ERROR: Cannot read as DICOM file: {str(e)}")
        return
    
    # Encode the DICOM file
    try:
        encoded_image = encode_image(dicom_path)
        print(f"DICOM file encoded successfully. Size: {len(encoded_image) // 1024} KB")
    except Exception as e:
        print(f"ERROR: Failed to encode DICOM file: {str(e)}")
        return
    
    # Prepare the request payload
    images = [{
        "data": encoded_image,
        "format": "dcm"
    }]
    
    # Extract context from DICOM metadata if available
    context = ""
    try:
        if hasattr(dicom_data, 'PatientAge'):
            context += f"Age: {dicom_data.PatientAge}.\n"
        if hasattr(dicom_data, 'PatientSex'):
            gender = "Male" if dicom_data.PatientSex == "M" else "Female" if dicom_data.PatientSex == "F" else dicom_data.PatientSex
            context += f"Gender: {gender}.\n"
        if hasattr(dicom_data, 'BodyPartExamined'):
            context += f"Body part: {dicom_data.BodyPartExamined}.\n"
    except:
        # If metadata extraction fails, use generic context
        context = "Patient information unknown."
    
    # Prompt
    prompt = "How would you characterize the findings from <img0>?"
    
    print("Sending DICOM to server for analysis...")
    
    # Make the prediction
    result = predict(images, "cxr", "report generation", context, prompt)
    
    if result:
        print("\nGenerated Report:")
        print(result["text"])
        print(f"\nProcessing time: {result['processing_time']:.2f} seconds")

def encode_dicom_file_or_series(dicom_path):
    """
    Encode a DICOM file or directory of DICOM files for transmission.
    Automatically detects whether the input is a single DICOM volume file or a series.
    
    Parameters:
    dicom_path (str): Path to a DICOM file or directory containing DICOM files
    
    Returns:
    dict: Dictionary with encoded data and is_series flag
    """
    if os.path.isdir(dicom_path):
        # Handle directory of DICOM files
        print(f"Processing directory of DICOM files: {dicom_path}")
        encoded_series = encode_dicom_series(dicom_path)
        return {
            "data": encoded_series,
            "is_series": True
        }
    elif os.path.isfile(dicom_path):
        # Check if the file is a DICOM volume
        try:
            dicom_data = pydicom.dcmread(dicom_path, force=True)
            if hasattr(dicom_data, 'NumberOfFrames') and int(dicom_data.NumberOfFrames) > 1:
                # This is a DICOM volume file
                print(f"Detected single DICOM volume file with {dicom_data.NumberOfFrames} frames")
                encoded_file = encode_image(dicom_path)
                return {
                    "data": encoded_file,
                    "is_series": True  # Treat as series for 3D processing
                }
            else:
                # Regular 2D DICOM
                print("DICOM file appears to be a 2D image")
                encoded_file = encode_image(dicom_path)
                return {
                    "data": encoded_file,
                    "is_series": False
                }
        except Exception as e:
            print(f"Error reading DICOM file: {str(e)}")
            # Fallback to regular encoding
            encoded_file = encode_image(dicom_path)
            return {
                "data": encoded_file,
                "is_series": False
            }
    else:
        raise ValueError(f"Path does not exist: {dicom_path}")

def example_ct_scan_dicom_series(dicom_path):
    """
    Example of analyzing a CT scan from a DICOM series or volume file.
    
    Parameters:
    dicom_path (str): Path to a directory containing DICOM series or a DICOM volume file
    """
    print("\n=== DICOM Volume Analysis ===")
    
    if not os.path.exists(dicom_path):
        print(f"ERROR: Path not found: {dicom_path}")
        return
    
    # Variables to store detected modality
    modality = "ct"  # Default to CT if can't detect
    modality_name = "CT Scan"
    
    # Validate the DICOM path
    if os.path.isdir(dicom_path):
        # Directory of DICOM files
        dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
        if not dicom_files:
            print(f"ERROR: No DICOM files found in the directory: {dicom_path}")
            return
        print(f"Found {len(dicom_files)} DICOM files in the series.")
        
        # Validate first DICOM file
        try:
            dicom_data = pydicom.dcmread(os.path.join(dicom_path, dicom_files[0]), force=True)
            detected_modality = dicom_data.Modality if hasattr(dicom_data, 'Modality') else "Unknown"
            print(f"DICOM series modality: {detected_modality}")
            
            # Set appropriate modality based on DICOM type
            if detected_modality == "CR" or detected_modality == "DX" or detected_modality == "XA":
                modality = "cxr"
                modality_name = "X-ray"
            elif detected_modality == "CT":
                modality = "ct"
                modality_name = "CT Scan"
            elif detected_modality == "MR":
                modality = "ct"  # Use CT for MRI as it's closest supported modality
                modality_name = "MRI Scan"
            
        except Exception as e:
            print(f"WARNING: Cannot validate DICOM series: {str(e)}")
        
    elif os.path.isfile(dicom_path):
        # Single DICOM file - check if it's a volume
        try:
            dicom_data = pydicom.dcmread(dicom_path, force=True)
            detected_modality = dicom_data.Modality if hasattr(dicom_data, 'Modality') else "Unknown"
            
            # Set appropriate modality based on DICOM type
            if detected_modality == "CR" or detected_modality == "DX" or detected_modality == "XA":
                modality = "cxr"
                modality_name = "X-ray"
            elif detected_modality == "CT":
                modality = "ct"
                modality_name = "CT Scan"
            elif detected_modality == "MR":
                modality = "ct"  # Use CT for MRI as it's closest supported modality
                modality_name = "MRI Scan"
                
            if hasattr(dicom_data, 'NumberOfFrames') and int(dicom_data.NumberOfFrames) > 1:
                print(f"DICOM volume file with {dicom_data.NumberOfFrames} frames, modality: {detected_modality}")
            else:
                print(f"Single DICOM file with modality: {detected_modality}")
        except Exception as e:
            print(f"WARNING: Cannot validate DICOM file: {str(e)}")
    else:
        print(f"ERROR: Path is neither a file nor a directory: {dicom_path}")
        return
    
    print(f"Using modality: {modality} ({modality_name})")
    
    # Encode the DICOM file or series
    try:
        print("Packaging and encoding DICOM data (this may take a moment)...")
        encoded_result = encode_dicom_file_or_series(dicom_path)
        print(f"DICOM data encoded successfully. Size: {len(encoded_result['data']) // 1024} KB")
    except Exception as e:
        print(f"ERROR: Failed to encode DICOM data: {str(e)}")
        return
    
    # Extract context from DICOM metadata if available
    context = ""
    try:
        if 'dicom_data' in locals():
            if hasattr(dicom_data, 'PatientAge'):
                context += f"Age: {dicom_data.PatientAge}.\n"
            if hasattr(dicom_data, 'PatientSex'):
                gender = "Male" if dicom_data.PatientSex == "M" else "Female" if dicom_data.PatientSex == "F" else dicom_data.PatientSex
                context += f"Gender: {gender}.\n"
            if hasattr(dicom_data, 'BodyPartExamined'):
                context += f"Body part: {dicom_data.BodyPartExamined}.\n"
            if hasattr(dicom_data, 'StudyDescription'):
                context += f"Study: {dicom_data.StudyDescription}.\n"
    except Exception as e:
        print(f"WARNING: Error extracting DICOM metadata: {str(e)}")
    
    # Determine appropriate prompt based on modality
    prompt = ""
    task = "segmentation"
    
    if modality == "cxr":
        prompt = "Describe the findings from <img0>."
        task = "report generation"
    elif modality == "ct":
        prompt = "Segment the liver."
        task = "segmentation"
    
    # Prepare the request
    request_data = {
        "images": [
            {
                "data": encoded_result["data"],
                "format": "dcm",
                "is_series": encoded_result["is_series"]
            }
        ],
        "context": context,
        "prompt": prompt,
        "modality": modality,
        "task": task,
        "hyperparams": {
            "num_beams": 1,
            "do_sample": True,
            "temperature": 0.1
        }
    }
    
    # Send to server
    print("Sending DICOM data to server for analysis (this may take some time)...")
    try:
        response = process_request(request_data)
        
        if not response:
            print("\nERROR: No response received from server")
            return
            
        if "error" in response:
            print(f"\nERROR from server: {response['error']}")
            return
        
        print("\nModel Response:")
        print(response['text'])
        
        if 'segmentation_mask' in response:
            print("\nSegmentation mask generated.")
            if 'is_3d' in response and response['is_3d']:
                print(f"3D segmentation with {len(response['segmentation_mask'])} slices")
            else:
                print("2D segmentation")
        else:
            print("\nNo segmentation mask in response.")
            
    except Exception as e:
        print(f"ERROR: Failed to get prediction: {str(e)}")
        traceback.print_exc()
        
    print("Analysis complete.")

def send_dicom_file(dicom_path, context=None, prompt=None, modality=None, task=None):
    """
    Send a DICOM file to the server for processing.
    
    Args:
        dicom_path (str): Path to the DICOM file
        context (str, optional): Additional context for the model
        prompt (str, optional): Specific prompt for the model
        modality (str, optional): Medical modality of the image
        task (str, optional): Task to perform (e.g., "segmentation")
        
    Returns:
        dict: Server response
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        
        # Extract metadata
        metadata = {}
        for tag in dicom_data.dir():
            if tag not in ['PixelData']:
                try:
                    value = getattr(dicom_data, tag)
                    if isinstance(value, (str, int, float)):
                        metadata[tag] = str(value)
                except:
                    pass
        
        # Convert to bytes
        with open(dicom_path, 'rb') as f:
            file_bytes = f.read()
        
        # Prepare request data
        files = {'file': ('image.dcm', file_bytes, 'application/dicom')}
        data = {}
        
        if context:
            data['context'] = context
        if prompt:
            data['prompt'] = prompt
        if modality:
            data['modality'] = modality
        if task:
            data['task'] = task
        
        # Send request
        response = requests.post(f"{SERVER_URL}/process_dicom", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"error": f"Server error: {response.status_code}"}
    
    except Exception as e:
        print(f"Error sending DICOM file: {str(e)}")
        return {"error": str(e)}

def send_dicom_series(dicom_dir, context=None, prompt=None, modality=None, task=None):
    """
    Send a DICOM series (directory of DICOM files) to the server for processing.
    
    Args:
        dicom_dir (str): Path to the directory containing DICOM files
        context (str, optional): Additional context for the model
        prompt (str, optional): Specific prompt for the model
        modality (str, optional): Medical modality of the image
        task (str, optional): Task to perform (e.g., "segmentation")
        
    Returns:
        dict: Server response
    """
    try:
        # Find all DICOM files in the directory
        dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            return {"error": "No DICOM files found in the directory"}
        
        # Read the first file to get metadata
        dicom_data = pydicom.dcmread(dicom_files[0], force=True)
        
        # Extract metadata from the first file
        metadata = {}
        for tag in dicom_data.dir():
            if tag not in ['PixelData']:
                try:
                    value = getattr(dicom_data, tag)
                    if isinstance(value, (str, int, float)):
                        metadata[tag] = str(value)
                except:
                    pass
        
        # Create a zip file containing all DICOM files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in dicom_files:
                # Add each file to the zip with a relative path
                arcname = os.path.basename(file_path)
                zip_file.write(file_path, arcname)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Prepare request data
        files = {'file': ('dicom_series.zip', zip_buffer.getvalue(), 'application/zip')}
        data = {}
        
        if context:
            data['context'] = context
        if prompt:
            data['prompt'] = prompt
        if modality:
            data['modality'] = modality
        if task:
            data['task'] = task
        
        # Send request
        response = requests.post(f"{SERVER_URL}/process_dicom_series", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"error": f"Server error: {response.status_code}"}
    
    except Exception as e:
        print(f"Error sending DICOM series: {str(e)}")
        return {"error": str(e)}

def example_dicom():
    """Example of sending a DICOM file for processing."""
    # Path to a sample DICOM file
    sample_dicom = os.path.join(SAMPLE_DIR, "sample.dcm")
    
    # Check if the file exists
    if not os.path.exists(sample_dicom):
        print(f"Sample DICOM file not found at {sample_dicom}")
        return
    
    # Read DICOM file to get metadata
    try:
        dicom_data = pydicom.dcmread(sample_dicom, force=True)
        print(f"DICOM info - Modality: {dicom_data.Modality if hasattr(dicom_data, 'Modality') else 'Unknown'}")
        print(f"DICOM info - Body part: {dicom_data.BodyPartExamined if hasattr(dicom_data, 'BodyPartExamined') else 'Unknown'}")
    except Exception as e:
        print(f"Error reading DICOM file: {str(e)}")
        return
    
    # Send the DICOM file
    print("\nSending DICOM file for processing...")
    response = send_dicom_file(
        sample_dicom,
        context="Patient with chest pain",
        prompt="Describe what you see in this image and identify any abnormalities.",
        modality="CT",
        task="description"
    )
    
    # Print the response
    print("\nServer response:")
    print(json.dumps(response, indent=2))

def example_dicom_series():
    """Example of sending a DICOM series for processing."""
    # Path to a sample DICOM series directory
    sample_dicom_dir = os.path.join(SAMPLE_DIR, "dicom_series")
    
    # Check if the directory exists
    if not os.path.exists(sample_dicom_dir) or not os.path.isdir(sample_dicom_dir):
        print(f"Sample DICOM series directory not found at {sample_dicom_dir}")
        return
    
    # Find a DICOM file in the directory
    dicom_files = []
    for root, _, files in os.walk(sample_dicom_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
                break
        if dicom_files:
            break
    
    if not dicom_files:
        print(f"No DICOM files found in {sample_dicom_dir}")
        return
    
    # Read a DICOM file to get metadata
    dicom_path = dicom_files[0]
    try:
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        print(f"DICOM series info - Modality: {dicom_data.Modality if hasattr(dicom_data, 'Modality') else 'Unknown'}")
        print(f"DICOM series info - Body part: {dicom_data.BodyPartExamined if hasattr(dicom_data, 'BodyPartExamined') else 'Unknown'}")
    except Exception as e:
        print(f"Error reading DICOM file: {str(e)}")
        return
    
    # Send the DICOM series
    print("\nSending DICOM series for processing...")
    response = send_dicom_series(
        sample_dicom_dir,
        context="Patient with abdominal pain",
        prompt="Describe what you see in this series and identify any abnormalities.",
        modality="CT",
        task="description"
    )
    
    # Print the response
    print("\nServer response:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    # Check if the server is running
    health = check_health()
    if not health or not health.get("model_loaded", False):
        print("Server is not running or model is not loaded.")
        exit(1)
    
    # Get model information
    info = get_model_info()
    if info:
        print("Model Information:")
        print(json.dumps(info, indent=2))
    
    # Demo menu
    print("MedVersa API Client Demo")
    print("------------------------")
    print("1. Chest X-ray Report Generation")
    print("2. Dermatology Classification")
    print("3. Dermatology Segmentation")
    print("4. Chest X-ray from DICOM")
    print("5. CT Scan from DICOM Series")
    
    choice = input("\nSelect an example (1-5): ")
    
    if choice == '1':
        example_chest_xray_report()
    elif choice == '2':
        example_dermatology_classification()
    elif choice == '3':
        example_dermatology_segmentation()
    elif choice == '4':
        dicom_path = input("Enter the path to a chest X-ray DICOM file: ")
        example_chest_xray_dicom(dicom_path)
    elif choice == '5':
        dicom_path = input("Enter the path to a directory containing CT DICOM series or a DICOM volume file: ")
        example_ct_scan_dicom_series(dicom_path)
    else:
        print("Invalid choice!") 