FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    libopencv-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    opencv-python==4.10.0.84 \
    cupy-cuda12x \
    librosa \
    numpy \
    tqdm \
    ffmpeg-python

# Copy preprocessing script
COPY preprocess_mix_avlips_dataset.py /app/preprocess_mix_avlips_dataset.py

# Set working directory
WORKDIR /app

# Command to run the script
CMD ["python3", "preprocess_mix_avlips_dataset.py"]
