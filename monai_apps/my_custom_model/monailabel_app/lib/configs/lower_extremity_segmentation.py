import os
import json
import torch
from lib.infers.lower_extremity_segmentation import LowerExtremitySegmentationInfer
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
        self.networks = []

    def init(self, name: str, model_dir: str, conf: dict, planner: object = None, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)
        # 1. Load label mapping from dataset.json
        dataset_path = os.path.join(model_dir, "dataset.json")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.dirname(model_dir), "dataset.json")
        with open(dataset_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
        self.labels = dataset_info.get("labels", {})
        num_channels = len(dataset_info.get("channel_names", {})) if dataset_info.get("channel_names") else 1

        # 2. Load spacing and patch_size configurations from plans.json
        plans_path = os.path.join(model_dir, "plans.json")
        if not os.path.exists(plans_path):
            plans_path = os.path.join(os.path.dirname(model_dir), "plans.json")
        with open(plans_path, 'r') as fp:
            plans = json.load(fp)
        fullres_plan = plans["configurations"]["3d_fullres"]
        self.target_spacing = tuple(fullres_plan["spacing"])
        self.roi_size = tuple(fullres_plan["patch_size"])

        # 3. only load the first fold model
        model_paths = []
        fold_dir = os.path.join(model_dir, "fold_0")  
        if not os.path.isdir(fold_dir):
            raise FileNotFoundError(f"Model directory 'fold_0' does not exist: {fold_dir}")

        weight_file = None
        for fname in ["checkpoint_final.pth", "model_final.pth", "model.pt", "checkpoint_best.pth"]:
        # for fname in ["checkpoint_best.pth"]:
            cand_path = os.path.join(fold_dir, fname)
            if os.path.exists(cand_path):
                weight_file = cand_path
                break
        # if no specific weight file found, load the first .pth file in the directory
        if not weight_file:
            for fname in os.listdir(fold_dir):
                if fname.endswith(".pth"):
                    weight_file = os.path.join(fold_dir, fname)
                    break
        if not weight_file:
            raise FileNotFoundError(f"No valid model weight file found in {fold_dir}")
        model_paths.append(weight_file)

        # 4. Load the model architecture and weights
        num_classes = (max(self.labels.values()) + 1) if self.labels else 1
        for path in model_paths:
            if PlainConvUNet is not None:
                arch_kwargs = fullres_plan["architecture"]["arch_kwargs"].copy()
                if isinstance(arch_kwargs.get("conv_op"), str) and "Conv3d" in arch_kwargs["conv_op"]:
                    arch_kwargs["conv_op"] = torch.nn.Conv3d
                if isinstance(arch_kwargs.get("norm_op"), str) and "InstanceNorm3d" in arch_kwargs["norm_op"]:
                    arch_kwargs["norm_op"] = torch.nn.InstanceNorm3d
                if isinstance(arch_kwargs.get("nonlin"), str) and "LeakyReLU" in arch_kwargs["nonlin"]:
                    arch_kwargs["nonlin"] = torch.nn.LeakyReLU
                arch_kwargs["input_channels"] = num_channels
                arch_kwargs["num_classes"] = num_classes
                model = PlainConvUNet(**arch_kwargs)
            else:
                from monai.networks.nets import DynUNet
                arch_args = fullres_plan.get("architecture", {}).get("arch_kwargs", {})
                kernels = arch_args.get("kernel_sizes", [[3, 3, 3]] * 6)
                strides = arch_args.get("strides", [[2, 2, 2]] * (len(kernels) - 1))
                filters = arch_args.get("features_per_stage", [32, 64, 128, 256, 320, 320])
                model = DynUNet(
                    spatial_dims=3,
                    in_channels=num_channels,
                    out_channels=num_classes,
                    kernel_size=kernels,
                    strides=[tuple(s) for s in strides],
                    upsample_kernel_size=[tuple(s) for s in strides],
                    filters=filters,
                    norm_name="instance",
                    act_name=("leakyrelu", {"inplace": True})
                )
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            try:
                model.load_state_dict(state_dict, strict=False)
            except RuntimeError:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.networks.append(model)

        return self

    def infer(self) -> InferTask:
        return LowerExtremitySegmentationInfer(
            networks=self.networks,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels
        )

    def trainer(self):
        return None