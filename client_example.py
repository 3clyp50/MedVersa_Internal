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
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

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
    # Encode the DICOM file
    encoded_image = encode_image(dicom_path)
    
    # Prepare the request payload
    data = {
        "images": [
            {
                "data": encoded_image,
                "format": "dcm"
            }
        ],
        "modality": "cxr",
        "task": "report generation",
        "hyperparams": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    # Send the request to the server
    response = requests.post(f"{SERVER_URL}/predict", json=data)
    
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return
    
    # Print the generated report
    result = response.json()
    print("\nGenerated Report:")
    print(result["text"])

def example_ct_scan_dicom_series(dicom_series_dir):
    """
    Example of analyzing a CT scan from a DICOM series.
    
    Parameters:
    dicom_series_dir (str): Path to a directory containing DICOM series of a CT scan
    """
    # Encode the DICOM series as a zip file
    encoded_series = encode_dicom_series(dicom_series_dir)
    
    # Prepare the request payload
    data = {
        "images": [
            {
                "data": encoded_series,
                "format": "dcm",
                "is_series": True
            }
        ],
        "modality": "ct",
        "task": "report generation",
        "hyperparams": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }
    
    print("Sending CT scan to server (this may take some time)...")
    
    # Send the request to the server
    response = requests.post(f"{SERVER_URL}/predict", json=data)
    
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return
    
    # Print the generated report
    result = response.json()
    print("\nGenerated CT Report:")
    print(result["text"])
    
    # Process segmentation if available
    if "segmentation_3d" in result:
        print("\nSegmentation masks available. These would need specialized 3D visualization.")

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
        dicom_dir = input("Enter the path to a directory containing CT DICOM series: ")
        example_ct_scan_dicom_series(dicom_dir)
    else:
        print("Invalid choice!") 