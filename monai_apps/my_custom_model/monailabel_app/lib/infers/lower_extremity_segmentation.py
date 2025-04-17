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
)
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

class LowerExtremitySegmentationInfer(InferTask):
    """
    Lower Extremity Segmentation using nnUNet
    """
    def __init__(self, networks, roi_size, target_spacing, labels):
        super().__init__(
            path="",
            network=None,
            type=InferType.SEGMENTATION,
            labels=labels,
            dimension=3,
            description="Lower Extremity Segmentation using nnUNet",
        )
        self.networks = networks
        self.roi_size = roi_size
        self.target_spacing = target_spacing

    def pre_transforms(self, data):
        # Define preprocessing transforms
        return Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys="image",
                pixdim=self.target_spacing,
                mode="bilinear",
            ),
            Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys="image"),
        ])

    def post_transforms(self, data):
        # Define postprocessing transforms
        return Compose([
            EnsureTyped(keys="pred"),
        ])

    def inferer(self, data):
        # Implement ensemble prediction logic
        image = data["image"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move image to device
        image = image.to(device)
        
        # Initialize prediction tensor
        pred = torch.zeros((1, len(self.labels), *image.shape[2:]), device=device)
        
        # Perform ensemble prediction
        with torch.no_grad():
            for model in self.networks:
                model = model.to(device)
                # Perform sliding window inference
                output = self.sliding_window_inference(
                    image,
                    self.roi_size,
                    1,
                    model,
                    mode="gaussian",
                    overlap=0.5,
                )
                pred += output
            
            # Average predictions
            pred = pred / len(self.networks)
            
            # Convert to numpy
            pred = pred.cpu().numpy()
            
            # Get final prediction
            pred = np.argmax(pred, axis=1)
            
            # Add to data dictionary
            data["pred"] = pred
            
        return data

    def sliding_window_inference(self, inputs, roi_size, sw_batch_size, predictor, mode="gaussian", overlap=0.5):
        # Implement sliding window inference
        # This is a simplified version, you may need to adjust based on your needs
        return predictor(inputs)
