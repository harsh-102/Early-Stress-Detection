"""
visualize.py — Data distribution and training visualization.

Pre-training:
  - Class distribution bar chart (by split)
  - Stress ratio histogram
  - Sample images: RGB | NIR | NDVI | GNDVI for each class

Post-training:
  - Training/validation loss curves
  - Training/validation accuracy curves
  - Per-class metric bar chart
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script use

from src.config import (
    MANIFEST_PATH,
    OUTPUT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    RGB_DIR,
    NIR_DIR,
    IMAGE_SIZE,
    TRAINING_HISTORY_PATH,
    HEALTHY_THRESHOLD,
    EARLY_STRESS_THRESHOLD,
)


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_distribution():
    """
    Generate pre-training visualizations:
    1. Class distribution bar chart per split
    2. Stress ratio histogram
    3. Sample images (RGB, NIR, NDVI, GNDVI) per class
    """
    _ensure_output_dir()
    manifest = pd.read_csv(MANIFEST_PATH)

    _plot_class_distribution(manifest)
    _plot_stress_ratio_histogram(manifest)
    _plot_sample_images(manifest)

    print(f"  All pre-training visualizations saved to: {OUTPUT_DIR}/")


def visualize_results():
    """
    Generate post-training visualizations:
    1. Training/validation loss curves
    2. Training/validation accuracy curves
    3. Per-epoch F1 score curve
    """
    _ensure_output_dir()

    if not os.path.exists(TRAINING_HISTORY_PATH):
        print(f"  Training history not found at {TRAINING_HISTORY_PATH}")
        return

    with open(TRAINING_HISTORY_PATH, "r") as f:
        history = json.load(f)

    _plot_training_curves(history)
    _plot_metric_curves(history)

    print(f"  All post-training visualizations saved to: {OUTPUT_DIR}/")


# ============================================================
# Pre-training visualizations
# ============================================================


def _plot_class_distribution(manifest: pd.DataFrame):
    """Bar chart showing class distribution per split."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, Orange, Red

    # Overall
    counts = [len(manifest[manifest["label"] == i]) for i in range(NUM_CLASSES)]
    axes[0].bar(class_names, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Overall Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=11)
    for j, c in enumerate(counts):
        axes[0].text(
            j,
            c + max(counts) * 0.02,
            str(c),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Per split
    for idx, split in enumerate(["train", "val", "test"]):
        split_df = manifest[manifest["split"] == split]
        counts = [len(split_df[split_df["label"] == i]) for i in range(NUM_CLASSES)]
        axes[idx + 1].bar(
            class_names, counts, color=colors, edgecolor="black", linewidth=0.5
        )
        axes[idx + 1].set_title(
            f"{split.capitalize()} (n={len(split_df)})", fontsize=13, fontweight="bold"
        )
        axes[idx + 1].set_ylabel("Count", fontsize=11)
        for j, c in enumerate(counts):
            axes[idx + 1].text(
                j,
                c + max(counts) * 0.02,
                str(c),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Class Distribution Across Splits", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_stress_ratio_histogram(manifest: pd.DataFrame):
    """Histogram of stress ratios (excluding auto-severe planter_skip entries)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter out auto-severe (ratio == -1)
    ratios = manifest[manifest["stress_ratio"] >= 0]["stress_ratio"]

    ax.hist(
        ratios, bins=100, color="#3498db", edgecolor="black", linewidth=0.3, alpha=0.8
    )

    ax.axvline(
        x=HEALTHY_THRESHOLD,
        color="#2ecc71",
        linestyle="--",
        linewidth=2,
        label=f"Healthy threshold ({HEALTHY_THRESHOLD*100:.1f}%)",
    )
    ax.axvline(
        x=EARLY_STRESS_THRESHOLD,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Severe threshold ({EARLY_STRESS_THRESHOLD*100:.1f}%)",
    )

    ax.set_xlabel("Stress Ratio", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    ax.set_title("Distribution of Stress Ratios", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "stress_ratio_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_sample_images(manifest: pd.DataFrame, samples_per_class: int = 3):
    """Show RGB, NIR, NDVI, GNDVI side-by-side for sample images from each class."""
    fig, axes = plt.subplots(
        NUM_CLASSES * samples_per_class,
        4,
        figsize=(16, 4 * NUM_CLASSES * samples_per_class),
    )

    row_idx = 0
    for label_val in range(NUM_CLASSES):
        class_df = manifest[manifest["label"] == label_val]
        if len(class_df) == 0:
            continue

        samples = class_df.sample(
            n=min(samples_per_class, len(class_df)), random_state=42
        )

        for _, sample_row in samples.iterrows():
            image_id = sample_row["image_id"]

            # Load RGB
            rgb = cv2.imread(os.path.join(RGB_DIR, f"{image_id}.jpg"))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

            # Load NIR
            nir = cv2.imread(
                os.path.join(NIR_DIR, f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE
            )
            nir = cv2.resize(nir, (IMAGE_SIZE, IMAGE_SIZE))

            # Compute VIs
            rgb_f = rgb.astype(np.float32) / 255.0
            nir_f = nir.astype(np.float32) / 255.0
            r = rgb_f[:, :, 0]
            g = rgb_f[:, :, 1]
            eps = 1e-7
            ndvi = (nir_f - r) / (nir_f + r + eps)
            gndvi = (nir_f - g) / (nir_f + g + eps)

            # Plot
            axes[row_idx, 0].imshow(rgb)
            axes[row_idx, 0].set_title(
                f"RGB — {CLASS_NAMES[label_val]}", fontsize=10, fontweight="bold"
            )
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(nir, cmap="gray")
            axes[row_idx, 1].set_title("NIR", fontsize=10)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            axes[row_idx, 2].set_title("NDVI", fontsize=10)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(gndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            axes[row_idx, 3].set_title("GNDVI", fontsize=10)
            axes[row_idx, 3].axis("off")

            row_idx += 1

    fig.suptitle(
        "Sample Images: RGB | NIR | NDVI | GNDVI (per class)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sample_images.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Post-training visualizations
# ============================================================


def _plot_training_curves(history: dict):
    """Plot training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Accuracy curves
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct = [a * 100 for a in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, val_acc_pct, "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_metric_curves(history: dict):
    """Plot validation precision, recall, and F1 over epochs."""
    epochs = range(1, len(history["val_precision"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        epochs, history["val_precision"], "g-o", markersize=3, label="Precision (macro)"
    )
    ax.plot(epochs, history["val_recall"], "b-o", markersize=3, label="Recall (macro)")
    ax.plot(epochs, history["val_f1"], "r-o", markersize=3, label="F1-Score (macro)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Metrics Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "metric_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    visualize_distribution()
