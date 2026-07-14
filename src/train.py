from __future__ import annotations

from pathlib import Path
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import make_loaders


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = 1 - (2.0 * intersection + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
        )
        dice = dice.mean()
        return self.dice_weight * dice + self.bce_weight * bce


@torch.inference_mode()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(2, 3))
    score = (2.0 * intersection + smooth) / (
        preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
    )
    return score.mean().item()


@torch.inference_mode()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_dice += dice_score(logits, y) * x.size(0)
        total_iou += iou_score(logits, y) * x.size(0)
        n += x.size(0)
        if (batch_idx + 1) % 25 == 0:
            print(
                f"  [epoch {epoch}/{num_epochs}] batch {batch_idx+1}/{len(loader)}  "
                f"loss={loss.item():.4f}",
                flush=True,
            )
    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_dice += dice_score(logits, y) * x.size(0)
        total_iou += iou_score(logits, y) * x.size(0)
        n += x.size(0)
    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


def main():
    data_root = PROJECT_ROOT / "dsb2018" / "train"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    batch_size = 8 if device.type == "cuda" else 4
    num_epochs = 15
    learning_rate = 1e-4

    print("Loading data ...", flush=True)
    train_loader, val_loader = make_loaders(
        root=data_root,
        batch=batch_size,
        workers=0,
        val_split=0.15,
    )
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}",
        flush=True,
    )

    print("Initializing U-Net (resnet34) ...", flush=True)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {"train": [], "val": []}
    best_val_loss = float("inf")

    t_start = datetime.now()
    for epoch in range(1, num_epochs + 1):
        epoch_t0 = datetime.now()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        val_metrics = validate(model, val_loader, criterion, device)

        history["train"].append({**train_metrics, "epoch": epoch})
        history["val"].append({**val_metrics, "epoch": epoch})

        epoch_time = (datetime.now() - epoch_t0).total_seconds()
        elapsed = (datetime.now() - t_start).total_seconds()
        print(
            f"Epoch {epoch:2d}/{num_epochs}  [{epoch_time:.0f}s]  "
            f"Train loss={train_metrics['loss']:.4f}  dice={train_metrics['dice']:.4f}  "
            f"iou={train_metrics['iou']:.4f}  |  "
            f"Val  loss={val_metrics['loss']:.4f}  dice={val_metrics['dice']:.4f}  "
            f"iou={val_metrics['iou']:.4f}  "
            f"[{elapsed:.0f}s elapsed]",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), model_dir / "unet_nuclei.pth")
            print(f"  -> Saved new best model (val_loss={best_val_loss:.4f})", flush=True)

    history["best_val_loss"] = best_val_loss
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    total_time = (datetime.now() - t_start).total_seconds()
    print(f"\nTraining complete in {total_time:.0f}s.", flush=True)
    print(f"Best val_loss={best_val_loss:.4f}", flush=True)
    print(f"Model saved to {model_dir / 'unet_nuclei.pth'}", flush=True)
    print(f"History saved to {model_dir / 'training_history.json'}", flush=True)


if __name__ == "__main__":
    main()
