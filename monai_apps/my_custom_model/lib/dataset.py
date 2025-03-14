import os
import glob
from monai.data import CacheDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Spacingd, CropForegroundd,
    RandFlipd, RandRotate90d, RandShiftIntensityd, ToTensord, Compose
)
from monai.utils import first
import matplotlib.pyplot as plt
from monai.visualize.utils import blend_images

DATASET_PATH = "datasets/my_custom_dataset"

images = sorted(glob.glob(os.path.join(DATASET_PATH, "images", "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(DATASET_PATH, "labels", "*.nii.gz")))

data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
train_files, val_files = data_dicts[:int(0.8 * len(data_dicts))], data_dicts[int(0.8 * len(data_dicts)):]

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ToTensord(keys=["image", "label"]),
])

train_dataset = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1, num_workers=4)
val_dataset = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1, num_workers=4)

# Test visualization
first_sample = first(train_dataset)
image, label = first_sample["image"], first_sample["label"]
overlay = blend_images(image[0], label[0], alpha=0.5)

plt.imshow(overlay, cmap="gray")
plt.title("Sample Segmentation Overlay")
plt.axis("off")
plt.show()
