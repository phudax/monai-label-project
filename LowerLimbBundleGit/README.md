# LowerLimbBundle

**Automatic Lower Limb CT Segmentation with MONAI-Label**

---

## Introduction

LowerLimbBundle is a packaged MONAI-Label application and 3DSlicer extension for **automatic segmentation of lower limb CT scans**. It provides:

- A **pre-trained SegResNetDS** model for volumetric (3D) segmentation of femur, tibia, patella, and other lower-limb bones.
- A **MONAI-Label** server for interactive annotation, inference, and continuous model refinement.
- A **3DSlicer** plugin interface for non-technical users to segment images within a familiar GUI.
- Full training pipeline so you can fine-tune or retrain the model on your own datasets.

> **Use cases**: orthopaedic pre-surgical planning, biomechanical research, automated annotation for large CT repositories.

---

## Repository Structure

```
LowerLimbBundle/
├── lib/                       # TaskConfigs, transforms, trainers, and infer handlers
│   ├── configs/
│   │   └── lowerlimb.py       # MONAI-Label training & inference config
│   ├── infer/                 
|   |   └── lowerlimb.py       # Inference task implementation
│   └── trainers/              
|       └── lowerlimb.py       # Training task implementation
├── model/
│   └── model.pt               # Pretrained weights
├── main.py                    # MONAI-LabelApp
├── environment.yml            # Conda environment definition
└── README.md                  # You are here
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/LowerLimbBundle.git
cd LowerLimbBundle
```

### 2A. (Recommended) Conda environment

```bash
conda env create -f environment.yml
conda activate lowerlimb
```

### 2B. Python virtual environment (pip)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

***Note:*** If you used Conda for PyTorch/CUDA, you may still need to install `monailabel`, `uvicorn`, and `fastapi` via pip:

```bash
pip install monailabel uvicorn fastapi
```

---

## Usage

### 1. Start the MONAI-Label server

Point the server at your directory of CT image volumes:

```bash
monailabel start_server \
  --app $(pwd) \
  --studies /path/to/CT-images \
  --conf models lowerlimb
```

- **app**: path to this bundle (contains `main.py`).
- **studies**: folder with your `.nii.gz` or `.nii` volumes.
- **--conf models lowerlimb**: selects the LowerLimb inference & train tasks.

Access the web UI at <http://localhost:8000> to explore cases, run inference, and label.

### 2. Run inference via API or CLI

- **Web UI**: Click the *infer* button after selecting a model.
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/infer \
    -F image=@/path/to/volume.nii.gz \
    -F output=@/path/to/pred.nii.gz
  ```

### 3. Train / Fine-tune

Use the `train` endpoint to refine the model on new labeled data:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
       "model": "lowerlimb",    
       "name": "my_finetune_run",
       "pretrained": true,
       "max_epochs": 50,
       "train_batch_size": 1
     }'
```

Training logs, checkpoints, and TensorBoard events will live under `model/lowerlimb/train_<name>/`.

To deploy the newly trained weights, copy the chosen `model.pt` into the root `model/model.pt` and restart:

```bash
cp model/lowerlimb/train_my_finetune_run/model.pt model/model.pt
# restart server
```

---

## 3DSlicer Extension

To enable segmentation inside 3DSlicer:

1. **Clone** or **Download** the [SlicerMONAIAuto3DSeg extension](https://github.com/lassoan/SlicerMONAIAuto3DSeg).
2. **Build** the extension via the Slicer Extension Wizard.
3. **Configure** the extension to point to your running MONAI-Label server URL (e.g., `http://localhost:8000`).

---

## Contributing

Feel free to open issues or pull requests for new features, bug fixes, or model upgrades. For major changes, please discuss via GitHub Issues first.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Contact

**Project Client:** Luca Modenese  
**Maintainer:** Your Name (<you@example.com>)

