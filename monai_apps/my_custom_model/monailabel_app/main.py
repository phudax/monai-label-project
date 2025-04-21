import os
import logging
from monailabel.interfaces.app import MONAILabelApp
from lib.configs.lower_extremity_segmentation import LowerExtremitySegmentation

logger = logging.getLogger(__name__)

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        super().__init__(app_dir=app_dir, studies=studies, conf=conf)
        self.model_dir = os.path.join(app_dir, "model")
        self.seg_task = LowerExtremitySegmentation()
        task_name = "lower_extremity_segmentation"
        self.seg_task.init(task_name, self.model_dir, conf, planner=None)
        
        infer_task = self.seg_task.infer()
        if isinstance(infer_task, dict):
            for name, task in infer_task.items():
                self._infers[name] = task 
        else:
            self._infers[task_name] = infer_task 

        logger.info(f"Registered MONAI Label task: {task_name}")