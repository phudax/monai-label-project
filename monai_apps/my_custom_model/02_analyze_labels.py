import os
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse

def analyze_label_distribution(label_file):
    nii = nib.load(label_file)
    data = nii.get_fdata()
    data = data.astype(np.int32)
    labels, counts = np.unique(data, return_counts=True)
    return dict(zip(labels, counts))

def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in NIfTI files")
    parser.add_argument('--labels_dir', type=str, default="labels", help="Path to the folder containing label NIfTI files")
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    
    if not labels_dir.is_dir():
        print(f"Error: Folder {labels_dir} does not exist.")
        return

    all_label_stats = {}
    file_stats = {}
    
    for label_file in labels_dir.glob("*.nii*"):
        stats = analyze_label_distribution(label_file)
        file_stats[label_file.name] = stats

        for label, count in stats.items():
            all_label_stats[label] = all_label_stats.get(label, 0) + count
    
    print("Individual file label distributions:")
    for fname, stats in file_stats.items():
        print(f"{fname}: {stats}")
    
    print("\nOverall label distribution across all files:")
    print(all_label_stats)

if __name__ == "__main__":
    main()
