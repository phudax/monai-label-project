import torch
from monai.networks.nets import nnUNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

class MySegmentationModel:
    def __init__(self):
        """
        Initializes the nnUNet segmentation model, loss function, optimizer, and evaluation metric.
        """
        self.model = nnUNet(
            dimensions=3,  # Since CT scans are 3D
            in_channels=1,  # Number of input channels (modify if needed)
            out_channels=1,  # Number of output labels (1 for binary segmentation)
            feature_size=32  # Adjust based on computational power
        )

        # Define loss function (Dice + CrossEntropy Loss for segmentation)
        self.loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        # Learning rate scheduler (Optional)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        # Evaluation metric (Dice Score)
        self.metric = DiceMetric(include_background=True, reduction="mean")

    def get_model(self):
        return self.model

    def get_loss_function(self):
        return self.loss_function

    def get_optimizer(self):
        return self.optimizer

    def get_lr_scheduler(self):
        return self.lr_scheduler

    def get_metric(self):
        return self.metric
