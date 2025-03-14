import os
import torch
import monai
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
)
from monai.utils import set_determinism
from model import MySegmentationModel
from dataset import get_training_data

# Set deterministic training for reproducibility
set_determinism(seed=42)

# Hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
train_data, val_data = get_training_data()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model
seg_model = MySegmentationModel()
model = seg_model.get_model().to(DEVICE)
loss_function = seg_model.get_loss_function()
optimizer = seg_model.get_optimizer()
lr_scheduler = seg_model.get_lr_scheduler()
metric = seg_model.get_metric()

# Training Loop
best_metric = -1
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Train Loss: {epoch_loss / len(train_loader):.4f}")

    # Validation Loop
    model.eval()
    dice_score = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(images)
            metric(outputs, labels)
        dice_score = metric.aggregate().item()
        metric.reset()

    print(f"Validation Dice Score: {dice_score:.4f}")

    # Save Best Model
    if dice_score > best_metric:
        best_metric = dice_score
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model!")

    lr_scheduler.step()

print("Training Complete!")
