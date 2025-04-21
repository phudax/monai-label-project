import os
import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    ToNumpyd,
)
from monai.inferers import sliding_window_inference
from monailabel.transform.post import Restored
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

class LowerExtremitySegmentationInfer(InferTask):
    """
    Inference task for Lower Extremity Segmentation using ensemble of nnUNet models.
    """
    def __init__(self, networks, roi_size, target_spacing, labels):
        super().__init__(
            type=InferType.SEGMENTATION,
            labels=labels,
            dimension=3,
            description="Lower Extremity Segmentation using nnUNet ensemble",
        )

        self.networks = networks
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        # Ensure labels dict is stored for use in inference (base class may also store this)
        self.labels = labels

    def pre_transforms(self):
        """Pre-processing transforms for the input image."""
        return Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),  # add channel dimension if not present
            Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),  # resample to target spacing
            Orientationd(keys="image", axcodes="RAS"),  # reorient to RAS for consistency
            ScaleIntensityRanged(
                keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),  # intensity normalization (CT windowing)
            EnsureTyped(keys="image"),  # ensure tensor type for model
        ])

    def post_transforms(self):
        """Post-processing transforms for the output prediction."""
        return Compose([
            EnsureTyped(keys="pred"),  # ensure tensor/MetaTensor for post-processing
            ToNumpyd(keys="pred", dtype=np.uint8),  # convert prediction to numpy (uint8)
            Restored(keys="pred", ref_image="image", has_channel=True, invert_orient=True),
        ])

    def inferer(self, data, device=None):
        """Run inference on the preprocessed data using the ensemble of networks."""
        # Determine device (use specified device or default to CUDA if available)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        image = data["image"].to(device)
        # Ensure image has batch dimension (N, C, ...). If only channel present, add batch dim.
        if image.dim() == 4:  # (C, D, H, W) -> add batch dimension
            image = image.unsqueeze(0)
        num_classes = (max(self.labels.values()) + 1) if self.labels else 1
        # Initialize prediction accumulator (for ensemble averaging)
        pred_sum = torch.zeros((1, num_classes, *image.shape[2:]), device=device)
        with torch.no_grad():
            for model in self.networks:
                model = model.to(device)
                # Sliding window inference for large images
                output = sliding_window_inference(image, self.roi_size, sw_batch_size=1, predictor=model, overlap=0.5, mode="gaussian")
                pred_sum += output
            # Average the outputs from all models
            pred_avg = pred_sum / len(self.networks)
            # Take argmax to get discrete label map
            pred_label = torch.argmax(pred_avg, dim=1, keepdim=True)  # shape (1, 1, D, H, W)
            data["pred"] = pred_label.cpu()  # move prediction to CPU
        return data

    def __call__(self, request):
        """
        Execute the full inference pipeline: preprocessing, inference, postprocessing,
        and return the resulting label file and result metadata.
        """
        # Parse requested device (if any)
        device_req = str(request.get("device", "")).lower()
        if device_req == "cuda" and not torch.cuda.is_available():
            device_req = "cpu"
        if device_req not in ("cuda", "cpu"):
            device_req = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_req)

        # Prepare input data dictionary for transforms
        data = {"image": request.get("image")}
        # Propagate any requested output format (extension, dtype) to data
        if "result_extension" in request:
            data["result_extension"] = request["result_extension"]
        if "result_dtype" in request:
            data["result_dtype"] = request["result_dtype"]

        # 1. Pre-processing
        data = self.pre_transforms()(data)
        # 2. Inference (ensemble model prediction)
        data = self.inferer(data, device=device)
        # 3. Post-processing (including resampling back to original space)
        data = self.post_transforms()(data)
        # 4. Use MONAI Label writer to save the label and prepare result dictionary
        result_file, result_dict = self.writer(data)
        return result_file, result_dict
    def is_valid(self, request=None) -> bool:
        if request is None:
            return True  
        return "image" in request