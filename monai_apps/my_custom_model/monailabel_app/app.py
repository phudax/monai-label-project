import os
import logging
from monailabel.interfaces.app import MONAILabelApp
# Import configuration class for lower extremity segmentation task
from lib.configs.lower_extremity_segmentation import LowerExtremitySegmentation
from lib.configs.segmentation_nnunet import SegmentationNnUNet

logger = logging.getLogger(__name__)

class App(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        # 1) Build TaskConfig
        cfg = SegmentationNnUNet()
        cfg.init(
            name="segmentation_nnunet",
            model_dir=os.path.join(app_dir, "model"),
            conf=conf,
            planner=None,            # nnUNet does not need MONAI's AutoScaler
        )

        # 2) Call parent class to complete registration
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="nnUNet v2 3D App",
            description="Deploy nnUNet v2 3D full-res segmentation via MONAI Label",
            version="0.1.0",
            models={cfg.name: cfg},   # Must be explicitly passed
        )
        # Initialize lower extremity segmentation task configuration
        self.seg_task = LowerExtremitySegmentation()
        task_name = "lower_extremity_segmentation"
        # Call TaskConfig.init() to load model and parameters
        self.seg_task.init(task_name, self.model_dir, conf, planner=None)
        # Register inference task
        infer_task = self.seg_task.infer()
        if isinstance(infer_task, dict):
            # If multiple tasks are returned, add them one by one
            for name, task in infer_task.items():
                self.infers[name] = task
        else:
            # Single task case
            self.infers[task_name] = infer_task

        logger.info(f"MONAI Label task registered: {task_name}")
