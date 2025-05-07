#!/usr/bin/env python
# train.py

import os
import json
import boto3
from botocore.exceptions import NoCredentialsError
from monai.apps.auto3dseg import AutoRunner

# def setup_s3_data_from_task_json(task_json_path="task.json", bucket_name="lower-limb-dataset", local_base="datasets"):
#     """
#     Downloads image/label files listed in task.json from S3 to `datasets/`.
#     """
#     print("🔍 Reading task.json and preparing S3 download...")
#     s3 = boto3.client(
#     "s3",
#         aws_access_key_id="AKIAX…WX6",
#         aws_secret_access_key="p7nGu0…r9QS",
#     )
#
#     resp = s3.list_objects_v2(Bucket="lower-limb-dataset", MaxKeys=10)
#     print("HTTP Status:", resp.get("ResponseMetadata", {}).get("HTTPStatusCode"))
#     if "Contents" in resp:
#         for obj in resp["Contents"]:
#             print(obj["Key"])
#     else:
#         print("No objects listed or access denied.")
#
#     # Load filenames from task.json
#     with open(task_json_path, "r") as f:
#         task_data = json.load(f)
#
#     filenames = set()
#     for item in task_data.get("training", []):
#         filenames.add(os.path.basename(item["image"]))
#         filenames.add(os.path.basename(item["label"]))
#
#     os.makedirs(local_base, exist_ok=True)
#
#     for filename in filenames:
#         local_path = os.path.join(local_base, filename)
#         if os.path.exists(local_path):
#             print(f"✅ Already exists: {local_path}")
#             continue
#
#         try:
#             print(f"⬇️  Downloading {filename} to {local_path}...")
#             s3.download_file(bucket_name, filename, local_path)
#         except NoCredentialsError:
#             print("❌ AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
#             return
#         except Exception as e:
#             print(f"❌ Failed to download {filename}: {e}")



def main():

    runner = AutoRunner(
        input="task.yaml",
        work_dir="workdir",
        algos=["segresnet"],
    )
    runner.mlflow_tracking_uri = ""
    runner.mlflow_experiment_name = ""
    runner.set_num_fold(5)

    runner.set_analyze_params({"device": "cuda", "do_ccp": False})
    runner.set_device_info(cuda_visible_devices="0")

    runner.set_training_params({
        "auto_scale_allowed": False,
        "num_epochs": 500,
        "num_epochs_per_validation": 50,
        "amp": True,
    })

    runner.run()

if __name__ == "__main__":
    main()