import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infer
import lib.infer.lowerlimb
import lib.trainers

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

########
from monai.networks.nets.segresnet_ds import SegResNetDS

import lib.trainers.lowerlimb

logger = logging.getLogger(__name__)


class LowerLimb(TaskConfig):
    def __init__(self):
        super().__init__()

        self.epistemic_enabled = None
        self.epistemic_samples = None

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
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

        # Model Files
        self.path = os.path.join(self.model_dir, "model.pt")  


        self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        # Setting ROI size - This is for the image padding
        self.roi_size = (96, 96, 96)

        # Network
        self.network = SegResNetDS(
            init_filters=32,
            blocks_down=(1,2,2,4,4),
            dsdepth=1,
            in_channels=1,
            out_channels=len(self.labels),
            norm="INSTANCE_NVFUSER",
        )

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")
    
    # define infer task
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infer.lowerlimb.LowerLimb(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            model_state_dict="state_dict",
        )
        return task

    # define training task
    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path

        task: TrainTask = lib.trainers.lowerlimb.LowerLimb(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            description="Train Lower Limb Segmentation Model",
            load_path=load_path,
            publish_path=self.path,
            labels=self.labels,
            disable_meta_tracking=False,
        )
        return task

    # define strategy for sample selection
    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    # define scoring methods for model evaluation
    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=SegResNetDS(
                    init_filters=32,
                    blocks_down=(1,2,2,4,4),
                    dsdepth=4,
                    in_channels=1,
                    out_channels=24,
                    norm="INSTANCE_NVFUSER",
                ),
                transforms=lib.infer.LowerLimb(None).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        return methods