import os
import json
import torch
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

# Try to import nnUNet v2 network architecture class (PlainConvUNet)
try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
except ImportError:
    PlainConvUNet = None

class LowerExtremitySegmentation(TaskConfig):
    def __init__(self):
        super().__init__()
        # List to store ensemble models
        self.networks = []

    def init(self, name: str, model_dir: str, conf: dict, planner: object = None, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)
        # 1. Load label mapping from dataset.json
        dataset_path = os.path.join(model_dir, "dataset.json")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.dirname(model_dir), "dataset.json")
        with open(dataset_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
        # Extract labels mapping dictionary (name to label index)
        self.labels = dataset_info.get("labels", {})

        # 2. Load spacing and patch_size configurations from plans.json
        plans_path = os.path.join(model_dir, "plans.json")
        if not os.path.exists(plans_path):
            plans_path = os.path.join(os.path.dirname(model_dir), "plans.json")
        with open(plans_path, 'r') as fp:
            plans = json.load(fp)
        fullres_plan = plans["configurations"]["3d_fullres"]
        # Model expected voxel spacing and patch size
        self.target_spacing = tuple(fullres_plan["spacing"])
        self.roi_size = tuple(fullres_plan["patch_size"])

        # 3. Find model weight file paths for each fold
        model_paths = []
        for fold in range(5):
            fold_dir = os.path.join(model_dir, f"fold_{fold}")
            weight_file = None
            # Check common filenames
            for fname in ["checkpoint_final.pth", "model_final.pth", "model.pt", "checkpoint_best.pth"]:
                cand_path = os.path.join(fold_dir, fname)
                if os.path.exists(cand_path):
                    weight_file = cand_path
                    break
            # If no file found, search for any .pth file
            if not weight_file:
                for fname in os.listdir(fold_dir):
                    if fname.endswith(".pth"):
                        weight_file = os.path.join(fold_dir, fname)
                        break
            if weight_file:
                model_paths.append(weight_file)
        if not model_paths:
            raise FileNotFoundError("No fold weight files found, please check the path")

        # 4. Build model and load weights for each fold
        num_classes = (max(self.labels.values()) + 1) if self.labels else 1  # Total number of classes (including background 0)
        for path in model_paths:
            # Initialize model architecture
            if PlainConvUNet is not None:
                # Use PlainConvUNet architecture provided by nnUNet v2
                arch_kwargs = fullres_plan["architecture"]["arch_kwargs"].copy()
                # Convert string-specified classes to actual class objects
                if isinstance(arch_kwargs.get("conv_op"), str) and "Conv3d" in arch_kwargs["conv_op"]:
                    arch_kwargs["conv_op"] = torch.nn.Conv3d
                if isinstance(arch_kwargs.get("norm_op"), str) and "InstanceNorm3d" in arch_kwargs["norm_op"]:
                    arch_kwargs["norm_op"] = torch.nn.InstanceNorm3d
                if isinstance(arch_kwargs.get("nonlin"), str) and "LeakyReLU" in arch_kwargs["nonlin"]:
                    arch_kwargs["nonlin"] = torch.nn.LeakyReLU
                model = PlainConvUNet(**arch_kwargs)
            else:
                # If nnUNet framework is not available, use MONAI DynUNet to approximate the network
                from monai.networks.nets import DynUNet
                arch_args = fullres_plan["architecture"]["arch_kwargs"]
                kernels = arch_args.get("kernel_sizes", [[3,3,3]] * 6)
                strides = arch_args.get("strides", [[2,2,2]] * 5)
                filters = arch_args.get("features_per_stage", [32, 64, 128, 256, 320, 320])
                model = DynUNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=num_classes,
                    kernel_size=kernels,
                    strides=[tuple(s) for s in strides],
                    upsample_kernel_size=[tuple(s) for s in strides],
                    filters=filters,
                    norm_name="instance",
                    act_name=("leakyrelu", {"inplace": True})
                )
            # Load model parameters
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            try:
                model.load_state_dict(state_dict, strict=False)
            except RuntimeError:
                # If necessary, remove "module." prefix that may exist in state_dict
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.networks.append(model)

        # 5. Return self for chaining (optional)
        return self

    def infer(self) -> InferTask:
        # Return inference task instance for lower extremity segmentation
        return LowerExtremitySegmentationInfer(
            networks=self.networks,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels
        )
