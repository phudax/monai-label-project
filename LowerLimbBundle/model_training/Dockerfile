# Dockerfile

# 1. Base image with CUDA & PyTorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 2. Working directory inside container
WORKDIR /workspace

# 3. Copy and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy only your code & config (no data!)
COPY train_model.py task.yaml task.json /workspace/

# 5. Copy the dataset directory into the image
#    (make sure ./datasets exists next to this Dockerfile when you build)
COPY datasets /workspace/datasets

# 5. Default command runs your training script
ENTRYPOINT ["python", "train_model.py"]