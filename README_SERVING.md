# MedVersa Serving API

This repository contains a Flask-based API for serving the MedVersa medical imaging model. The API supports multiple modalities, tasks, and input types.

## Features

- Support for multiple medical imaging modalities:
  - Chest X-ray (cxr)
  - Dermatology (derm)
  - CT scans (ct)

- Support for multiple tasks:
  - Report generation
  - Classification
  - Segmentation (2D and 3D)

- HTTP API with JSON request/response format
- Base64 encoding for image data
- Health check and model information endpoints
- Customizable hyperparameters

## Requirements

- Python 3.9+
- Flask
- PyTorch
- MedVersa model dependencies (as per the original repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/MedVersa_Internal.git
cd MedVersa_Internal
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face access token:
```bash
export HF_TOKEN=your_huggingface_token
```

## Running the Server

Start the server with the following command:

```bash
python serve.py
```

By default, the server runs on port 5000. You can change this by setting the `PORT` environment variable:

```bash
PORT=8080 python serve.py
```

## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the server and whether the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Information

```
GET /info
```

Returns information about the loaded model, supported modalities, and tasks.

**Response:**
```json
{
  "name": "MedVersa_Internal",
  "supported_modalities": ["cxr", "derm", "ct"],
  "supported_tasks": ["report generation", "classification", "segmentation"],
  "device": "cuda"
}
```

### Prediction

```
POST /predict
```

Makes a prediction using the model.

**Request Body:**
```json
{
  "images": [
    {
      "data": "base64_encoded_image_data",
      "format": "jpg"
    }
  ],
  "modality": "cxr",
  "task": "report generation",
  "context": "Age:30-40.\nGender:F.\nIndication: ...",
  "prompt": "How would you characterize the findings from <img0>?",
  "hyperparams": {
    "num_beams": 1,
    "do_sample": true,
    "min_length": 1,
    "top_p": 0.9,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "temperature": 0.1
  }
}
```

**Response:**
For text-based tasks (report generation, classification):
```json
{
  "text": "Generated text response...",
  "processing_time": 1.234
}
```

For segmentation tasks:
```json
{
  "text": "Segmentation result description...",
  "segmentation_2d": ["base64_encoded_numpy_array", ...],
  "processing_time": 1.234
}
```

## Example Client

The repository includes an example client (`client_example.py`) that demonstrates how to use the API for different use cases:

- Chest X-ray report generation
- Dermatology classification
- Dermatology segmentation

To run the example client:

```bash
python client_example.py
```

## Request Format Details

### Images

Images are sent as a list of objects, each containing a base64-encoded string and a format indicator:

```json
"images": [
  {
    "data": "base64_encoded_image_data",
    "format": "jpg"
  }
]
```

Supported formats:
- 2D images: jpg, jpeg, png
- 3D volumes: nii.gz (NIfTI format)

Multiple images can be sent for tasks that support multiple inputs.

### Context

Context is a string that provides patient information, typically in the format:

```
Age:30-40.
Gender:F.
Indication: Patient history...
Comparison: Prior studies...
```

This is optional for some tasks but recommended for better results.

### Prompt

The prompt is the specific question or instruction for the model:

- For report generation: "How would you characterize the findings from <img0>?"
- For classification: "What is the primary diagnosis?"
- For segmentation: "Segment the liver." or "Segment the lesion."

### Hyperparameters

Hyperparameters can be customized to control the generation behavior:

- `num_beams`: Number of beams for beam search
- `do_sample`: Whether to use sampling
- `min_length`: Minimum length of the generated text
- `top_p`: Top-p sampling parameter
- `repetition_penalty`: Repetition penalty
- `length_penalty`: Length penalty
- `temperature`: Sampling temperature

## Error Handling

The API returns appropriate HTTP status codes for different types of errors:

- 400: Bad Request (invalid input)
- 500: Server Error (processing error)

Error responses include a message and details:

```json
{
  "error": "Invalid request",
  "details": ["Missing required field: images"]
}
```

## Security Considerations

- The API does not implement authentication by default. In production, you should add appropriate authentication mechanisms.
- Consider rate limiting and other security measures for public-facing deployments.
- Be aware of privacy implications when handling medical data.

## License

[Include license information here]

## Supported Image Formats

The API supports various image formats:

- **Standard Image Formats**:
  - JPEG/JPG
  - PNG

- **Medical Image Formats**:
  - DICOM (both single files and series)
  - NIfTI (.nii.gz)

### DICOM Support

The API includes full support for DICOM files, including:

- **Single DICOM files**: Useful for 2D medical imaging (e.g., X-rays)
- **DICOM series**: For 3D medical imaging (e.g., CT scans)
- **Metadata extraction**: The API automatically extracts patient and study information from DICOM headers to enhance the context provided to the model

#### Using DICOM Files

To use DICOM files with the API:

1. **Single DICOM file**:
   ```json
   {
     "images": [
       {
         "data": "base64_encoded_dicom_data",
         "format": "dcm"
       }
     ],
     "modality": "cxr",
     "task": "report generation"
   }
   ```

2. **DICOM series** (for 3D volumes):
   ```json
   {
     "images": [
       {
         "data": "base64_encoded_zip_of_dicom_files",
         "format": "dcm",
         "is_series": true
       }
     ],
     "modality": "ct",
     "task": "segmentation"
   }
   ```

For DICOM series, the API expects a zip file containing all DICOM files in the series, encoded as base64. 