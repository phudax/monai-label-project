#!/usr/bin/env python
# train.py

import os
import json
import boto3
from botocore.exceptions import NoCredentialsError
from monai.apps.auto3dseg import AutoRunner

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