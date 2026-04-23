"""
train.py — Training loop with early stopping, per-epoch metrics, and confusion matrix.

- Uses 50% stratified subsample of training data
- Class-weighted CrossEntropyLoss for imbalanced data
- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping on validation loss
- Prints Accuracy, Precision, Recall, F1, and Confusion Matrix each epoch
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from tqdm import tqdm

from src.config import (
    MANIFEST_PATH, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR, TRAIN_SUBSET_RATIO, RANDOM_SEED,
    NUM_CLASSES, CLASS_NAMES, MODEL_DIR, BEST_MODEL_PATH,
    TRAINING_HISTORY_PATH
)
from src.dataset import AgriVisionDataset, get_train_transforms, get_val_transforms
from src.model import LateFusionCNN, model_summary, count_parameters


def _get_device():
    """Select the best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"  Using device: CPU")
    return device


def _compute_class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    class_counts = labels.value_counts().sort_index().values.astype(float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize
    print(f"  Class counts:  {dict(zip(CLASS_NAMES.values(), class_counts.astype(int)))}")
    print(f"  Class weights: {dict(zip(CLASS_NAMES.values(), np.round(weights, 3)))}")
    return torch.FloatTensor(weights).to(device)


def _format_confusion_matrix(cm: np.ndarray) -> str:
    """Format confusion matrix as a readable string table."""
    header = f"{'':>15s} | {'Predicted':^42s}"
    subheader = f"{'Actual':>15s} | {'Healthy':>12s} {'Early Stress':>12s} {'Severe Stress':>13s}"
    sep = "-" * 60
    lines = [header, subheader, sep]
    for i, row in enumerate(cm):
        name = CLASS_NAMES[i]
        vals = "".join(f"{v:>13d}" for v in row)
        lines.append(f"{name:>15s} | {vals}")
    return "\n".join(lines)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def _validate(model, loader, criterion, device):
    """
    Validate the model. Returns (avg_loss, accuracy, all_preds, all_labels).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train() -> dict:
    """
    Full training pipeline:
    1. Load manifest, stratified 50% subsample
    2. Create DataLoaders
    3. Initialize model, loss, optimizer, scheduler
    4. Train with early stopping
    5. Return training history dict

    Returns:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                 'val_precision', 'val_recall', 'val_f1', 'epoch_times'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- Load manifest ----
    manifest = pd.read_csv(MANIFEST_PATH)
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]

    print(f"\n  Full training set: {len(train_df)} images")
    print(f"  Validation set:    {len(val_df)} images")

    # ---- Stratified subsample ----
    if TRAIN_SUBSET_RATIO >= 1.0:
        train_subset = train_df.copy()
        print(f"  Training subset (100%): {len(train_subset)} images")
    else:
        train_subset, _ = train_test_split(
            train_df,
            train_size=TRAIN_SUBSET_RATIO,
            stratify=train_df["label"],
            random_state=RANDOM_SEED,
        )
        print(f"  Training subset ({TRAIN_SUBSET_RATIO*100:.0f}% stratified): {len(train_subset)} images")

    # Print subset class distribution
    print(f"\n  Training subset class distribution:")
    for label_val in sorted(train_subset["label"].unique()):
        count = (train_subset["label"] == label_val).sum()
        pct = count / len(train_subset) * 100
        print(f"    {CLASS_NAMES[label_val]:>15s}: {count:5d} ({pct:5.1f}%)")

    # ---- Create DataLoaders ----
    train_dataset = AgriVisionDataset(train_subset, transform=get_train_transforms())
    val_dataset = AgriVisionDataset(val_df, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---- Device ----
    device = _get_device()

    # ---- Model ----
    model = LateFusionCNN(num_classes=NUM_CLASSES).to(device)
    model_summary(model)

    # ---- Loss with class weights ----
    class_weights = _compute_class_weights(train_subset["label"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR
    )

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
        "epoch_times": [], "learning_rates": [],
    }

    print(f"\n  {'='*70}")
    print(f"  TRAINING STARTED — Max {MAX_EPOCHS} epochs, Early stopping patience={EARLY_STOPPING_PATIENCE}")
    print(f"  {'='*70}")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = _validate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        # Compute metrics
        val_precision = precision_score(val_labels, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2])

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["epoch_times"].append(epoch_time)
        history["learning_rates"].append(current_lr)

        # Check for improvement
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            improved = " ★ BEST"
        else:
            patience_counter += 1

        # Print epoch summary
        print(f"\n  Epoch {epoch+1:>2d}/{MAX_EPOCHS} ({epoch_time:.1f}s) | LR: {current_lr:.2e}{improved}")
        print(f"    Train — Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%")
        print(f"    Val   — Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")
        print(f"    Val   — Precision: {val_precision:.4f}  Recall: {val_recall:.4f}  F1: {val_f1:.4f}")
        print(f"\n    Confusion Matrix (Validation):")
        for line in _format_confusion_matrix(cm).split("\n"):
            print(f"    {line}")

        # Step scheduler
        scheduler.step(val_loss)

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  ⚠️  Early stopping triggered after {epoch+1} epochs (patience={EARLY_STOPPING_PATIENCE})")
            break

    # Save training history
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history_serializable, f, indent=2)

    total_time = sum(history["epoch_times"])
    print(f"\n  {'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  {'='*70}")
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {BEST_MODEL_PATH}")
    print(f"  Training history saved to: {TRAINING_HISTORY_PATH}")

    return history


if __name__ == "__main__":
    train()
