"""
Convert CT images and segmentation labels into nnUNetv2 format dataset (all cases for training).
This script also creates the necessary base folders for nnU-Net: nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.
It will create three subfolders in the dataset folder: imagesTr, labelsTr, and imagesTs.
"""

import os
import sys
import json
import argparse
import shutil
import gzip
from pathlib import Path
import nibabel as nib
import numpy as np

def create_directory_structure(dataset_name, base_raw_dir, base_preprocessed_dir, base_results_dir):
    # Create base directories if they do not exist
    base_raw_dir.mkdir(parents=True, exist_ok=True)
    base_preprocessed_dir.mkdir(parents=True, exist_ok=True)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset folder under nnUNet_raw
    dataset_dir = base_raw_dir / dataset_name
    imagesTr_dir = dataset_dir / "imagesTr"
    labelsTr_dir = dataset_dir / "labelsTr"
    imagesTs_dir = dataset_dir / "imagesTs"  # This folder will remain empty by default
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)
    imagesTs_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, imagesTr_dir, labelsTr_dir, imagesTs_dir

def copy_nifti_file(src_path: Path, dst_path: Path):
    # If the source file is .nii, compress it to .nii.gz; if it's already .nii.gz, simply copy it.
    if src_path.suffixes == ['.nii']:
        with open(src_path, 'rb') as f_in, gzip.open(dst_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    elif src_path.suffixes == ['.nii', '.gz']:
        shutil.copy(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)

def main():
    parser = argparse.ArgumentParser(description="Convert CT scans and segmentation labels into nnUNetv2 format for training")
    parser.add_argument('--images_dir', type=str, default="images", help='Path to the original CT images folder')
    parser.add_argument('--labels_dir', type=str, default="labels", help='Path to the original segmentation labels folder')
    parser.add_argument('--dataset_id', type=int, default=1, help='Dataset ID (used to generate output folder name)')
    parser.add_argument('--task_name', type=str, default="MYTASK", help='Task name (used to generate output folder name)')
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    dataset_id = args.dataset_id
    task_name = args.task_name
    
    # Check if input folders exist
    if not images_dir.is_dir():
        print(f"Error: Image directory does not exist: {images_dir}")
        return 1
    if not labels_dir.is_dir():
        print(f"Error: Labels directory does not exist: {labels_dir}")
        return 1
    
    # List all NIfTI image and label files
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and (f.name.endswith('.nii') or f.name.endswith('.nii.gz'))])
    label_files = sorted([f for f in labels_dir.iterdir() if f.is_file() and (f.name.endswith('.nii') or f.name.endswith('.nii.gz'))])
    
    if len(image_files) == 0:
        print(f"Error: No .nii or .nii.gz files found in image directory {images_dir}")
        return 1
    if len(label_files) == 0:
        print(f"Error: No .nii or .nii.gz files found in labels directory {labels_dir}")
        return 1
    if len(image_files) != len(label_files):
        print(f"Error: The number of image files ({len(image_files)}) does not match the number of label files ({len(label_files)})")
        return 1
    
    # (Optional) Check if image and label filenames match (if applicable)
    for img_path, lbl_path in zip(image_files, label_files):
        expected_label_name = img_path.name.replace("image", "label")
        if expected_label_name != lbl_path.name:
            print(f"Error: Image file {img_path.name} does not match label file {lbl_path.name}")
            return 1
    
    total_cases = len(image_files)
    
    # Create nnU-Net output directory structure (including nnUNet_raw, nnUNet_preprocessed, and nnUNet_results)
    base_raw_dir = Path("nnUNet_raw")
    base_preprocessed_dir = Path("nnUNet_preprocessed")
    base_results_dir = Path("nnUNet_results")
    
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"
    dataset_dir, imagesTr_dir, labelsTr_dir, imagesTs_dir = create_directory_structure(dataset_name, base_raw_dir, base_preprocessed_dir, base_results_dir)
    
    # Process all cases (training set only): copy images and labels into imagesTr and labelsTr
    for idx, (img_path, lbl_path) in enumerate(zip(image_files, label_files)):
        case_id = f"case_{idx:03d}"
        dst_img = imagesTr_dir / f"{case_id}_0000.nii.gz"
        dst_lbl = labelsTr_dir / f"{case_id}.nii.gz"
        copy_nifti_file(img_path, dst_img)
        copy_nifti_file(lbl_path, dst_lbl)
        print(f"Copied image: {img_path.name} -> {dst_img.name}")
        print(f"Copied label: {lbl_path.name} -> {dst_lbl.name}")
    
    # Build labels dictionary (example, modify based on your task)
    labels_dict = {
        "background": 0,
        "Threshold-200-MAX": 1,
        "Femur_R": 2,
        "Femur_L": 3,
        "Hip_R": 4,
        "Hip_L": 5,
        "Sacrum": 6,
        "Patella_R": 7,
        "Patella_L": 8,
        "Tibia_R": 9,
        "Fibula_R": 10,
        "Tibia_L": 11,
        "Fibula_L": 12,
        "Talus_R": 13,
        "Calcaneus_R": 14,
        "Tarsals_R": 15,
        "Metatarsals_R": 16,
        "Phalanges_R": 17,
        "Talus_L": 18,
        "Calcaneus_L": 19,
        "Tarsals_L": 20,
        "Metatarsals_L": 21,
        "Phalanges_L": 22,
        "Minisci": 23
    }
    
    # Create dataset.json content
    dataset_info = {
        "name": dataset_name,
        "description": "Lower extremity CT segmentation dataset",
        "reference": "Custom dataset",
        "license": "CC-BY-NC",
        "channel_names": { "0": "CT" },
        "labels": labels_dict,
        "file_ending": ".nii.gz",
        "numTraining": total_cases,
        "training": []
    }
    # Populate training list
    for idx in range(total_cases):
        case_id = f"case_{idx:03d}"
        image_rel = f"imagesTr/{case_id}_0000.nii.gz"
        label_rel = f"labelsTr/{case_id}.nii.gz"
        dataset_info["training"].append({"image": image_rel, "label": label_rel})
    
    # Save dataset.json file in the dataset folder
    dataset_json_path = dataset_dir / "dataset.json"
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    print(f"Total cases: {total_cases}")
    print(f"dataset.json saved at: {dataset_json_path}")
    print("Data conversion completed. The 'imagesTs' folder is created but remains empty for future use.")

if __name__ == "__main__":
    sys.exit(main())
