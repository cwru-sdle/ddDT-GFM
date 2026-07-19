# Reproducible ddDT-GFM image — NVIDIA GPU, CUDA 12.1 (validated stack).
#
#   docker build -t dddt-gfm:0.2.0 .
#   docker run --gpus all -it -p 8888:8888 dddt-gfm:0.2.0   # JupyterLab on :8888
#
# The image carries CUDA 12.1 + cuDNN, so the HOST only needs an NVIDIA driver
# >= 525 and the nvidia-container-toolkit — nothing else on the machine changes.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Runtime libs some scientific/plotting wheels expect
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-lock-cu121.txt .

# torch / torchvision are already in the base image; this pins everything else.
RUN pip install --no-cache-dir -r requirements-lock-cu121.txt jupyterlab && \
    pip install --no-cache-dir ddDT-GFM

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
