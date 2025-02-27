FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY environment.yml /app/
RUN pip install --no-cache-dir conda-merge \
    && conda-merge environment.yml > environment_merged.yml \
    && pip install --no-cache-dir -r environment_merged.yml

# Install DICOM-specific packages
RUN pip install --no-cache-dir pydicom SimpleITK

# Copy application code
COPY . /app/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Use HuggingFace token from environment if provided
# Set HF_TOKEN as a build arg or environment variable when running the container
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Run the server
CMD ["python", "serve.py"] 