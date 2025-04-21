# COMP3900-25T1 nn‑Unet Model Guide

## 1. Environment Setup

### 1.1 Create a Python Virtual Environment

It is recommended to install nn‑Unet within a virtual environment to avoid dependency conflicts. You can use either Anaconda or Python’s built‑in venv. **Python 3.9 or later** is recommended.

**Using Anaconda to create a virtual environment:**

```bash
conda create -n nnunet2 python=3.9 -y
conda activate nnunet2

```

### 1.2 Download and Install CUDA

- Visit the **NVIDIA official website** to download the CUDA Toolkit. Ensure you select a version compatible with your graphics card (for example, CUDA 12.x).
- After installation, open a command prompt and run the following command to confirm a successful installation:
    
    ```bash
    nvcc --version
    
    ```
    
    If you see output similar to the expected version information, CUDA has been installed correctly.
    

### 1.3 Install CUDA‑Supported PyTorch

Before installing nn‑Unet v2, be sure to install the GPU version of PyTorch. Adjust the command below based on your CUDA version (the example below uses a version that supports CUDA 11):

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

```

Verify the installation by running:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

```

A `True` output indicates that the GPU is available. Otherwise, verify your NVIDIA driver and environment configuration.

### 1.4 Clone the nn‑Unet v2 Repository and Install Dependencies

Clone the nn‑Unet repository using Git and install dependencies in editable mode:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

```

After a successful installation, the command‑line tools (prefixed with `nnUNetv2_`) contained in the repository will be automatically available in your virtual environment.

### 1.5 Create the nn‑Unet Supported Data Format

Create three folders to store the input data, processed data, and training results for nn‑Unet. Ensure that the folders `images`, `labels`, and the script `01_convert_data` are in the same directory, then run the `01_convert_data` script.

### 1.6 Set Environment Variables

Set the following environment variables so that nn‑Unet can correctly locate the raw data, preprocessed output, and training results. These commands need to be executed each time before training:

```powershell
$env:nnUNet_raw="D:\COMP3900\nnUNet\nnUNet_raw"
$env:nnUNet_preprocessed="D:\COMP3900\nnUNet\nnUNet_preprocessed"
$env:nnUNet_results="D:\COMP3900\nnUNet\nnUNet_results"

```

---

## 2. Preprocessing and Plan Generation

### 2.1 Generate the Dataset Fingerprint, Experiment Plan, and Preprocessed Data

Run the following command (with dataset ID 1, which corresponds to `Dataset001_MYTASK`):

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

```

This command will:

- Analyze the dataset and generate a fingerprint file (`dataset_fingerprint.json`).
- Automatically create a `nnUNetPlans.json` file that contains the appropriate preprocessing parameters and network configuration.
- Resample, crop, and normalize the data, then save the preprocessed data to the `nnUNet_preprocessed` directory.

---

## 3. Model Training

### 3.1 Five‑Fold Cross Validation Training

For the 3d_fullres model (with dataset ID 1), run the following commands to train each fold:

```bash
nnUNetv2_train 1 3d_fullres 0
nnUNetv2_train 1 3d_fullres 1
nnUNetv2_train 1 3d_fullres 2
nnUNetv2_train 1 3d_fullres 3
nnUNetv2_train 1 3d_fullres 4

```

Each fold’s model and logs will be saved in:

```
nnUNet_results\Dataset001_MYTASK\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_x

```

where **x** ranges from 0 to 4.

If the training is interrupted or needs to be resumed, use:

```bash
nnUNetv2_train 1 3d_fullres 0 --c

```

### 3.2 Single Model Training

If you prefer to train a single model using the entire dataset (i.e., without cross-validation), run:

```bash
nnUNetv2_train 1 3d_fullres all

```

This method uses all available data for training and is suitable for generating the final model for deployment.

---

## 4. Inference Prediction

### 4.1 Five‑Fold Ensemble Prediction

After training is complete, use the `nnUNetv2_predict` command to perform segmentation inference on new CT images. By default, nn‑Unet v2 automatically loads the five‑fold models and ensembles the predictions. For example:

```bash
nnUNetv2_predict -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -d 1 -c 3d_fullres

```

**Note:**

- `<INPUT_FOLDER>` and `<OUTPUT_FOLDER>` must be specified with absolute paths.
- This command uses dataset ID 1 with the 3d_fullres configuration to predict segmentation on images located in the `imagesTs` directory, and outputs the results to the designated folder.
- You may add the parameter `-disable_tta` to disable test-time augmentation.

### 4.2 Other Optional Parameters

If you wish to use the best performing model from the validation phase, specify the checkpoint by adding:

```bash
-chk checkpoint_best.pth

```

---