"""
evaluate.py — Final evaluation on the test set.

Loads the best saved model, runs inference on the full test set, and produces:
- Overall accuracy
- Per-class precision, recall, F1
- Macro-averaged metrics
- Confusion matrix (printed and saved as heatmap)
- Full classification report
"""

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.config import (
    MANIFEST_PATH, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES,
    BEST_MODEL_PATH, MODEL_DIR, OUTPUT_DIR
)
from src.dataset import AgriVisionDataset, get_val_transforms
from src.model import LateFusionCNN


def evaluate() -> dict:
    """
    Evaluate the best model on the full test set.

    Returns:
        results: dict with accuracy, precision, recall, f1, confusion_matrix
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load manifest ----
    manifest = pd.read_csv(MANIFEST_PATH)
    test_df = manifest[manifest["split"] == "test"]
    print(f"\n  Test set: {len(test_df)} images")

    # Print test class distribution
    print(f"\n  Test set class distribution:")
    for label_val in sorted(test_df["label"].unique()):
        count = (test_df["label"] == label_val).sum()
        pct = count / len(test_df) * 100
        print(f"    {CLASS_NAMES[label_val]:>15s}: {count:5d} ({pct:5.1f}%)")

    # ---- Create DataLoader ----
    test_dataset = AgriVisionDataset(test_df, transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---- Load best model ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = LateFusionCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"\n  Loaded best model from: {BEST_MODEL_PATH}")

    # ---- Run inference ----
    all_preds = []
    all_labels = []

    print(f"  Running inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- Compute Metrics ----
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    class_names_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    # ---- Print Results ----
    report_lines = [
        f"\n  {'='*60}",
        f"  TEST SET EVALUATION RESULTS",
        f"  {'='*60}",
        f"\n  Overall Accuracy:    {accuracy*100:.2f}%",
        f"  Macro Precision:     {precision_macro:.4f}",
        f"  Macro Recall:        {recall_macro:.4f}",
        f"  Macro F1-Score:      {f1_macro:.4f}"
    ]

    # Per-class metrics
    precision_per = np.array(precision_score(all_labels, all_preds, average=None, zero_division=0))
    recall_per = np.array(recall_score(all_labels, all_preds, average=None, zero_division=0))
    f1_per = np.array(f1_score(all_labels, all_preds, average=None, zero_division=0))

    report_lines.extend([
        f"\n  Per-Class Metrics:",
        f"  {'Class':>15s} | {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}",
        f"  {'-'*60}"
    ])
    for i in range(NUM_CLASSES):
        support = (all_labels == i).sum()
        report_lines.append(f"  {CLASS_NAMES[i]:>15s} | {precision_per[i]:>10.4f} {recall_per[i]:>10.4f} {f1_per[i]:>10.4f} {support:>10d}")

    # Full classification report
    report_lines.append(f"\n  Full Classification Report:")
    report = str(classification_report(all_labels, all_preds, target_names=class_names_list, zero_division=0))
    for line in report.split("\n"):
        report_lines.append(f"    {line}")

    final_report_text = "\n".join(report_lines)
    print(final_report_text)

    # Save to text file
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(final_report_text + "\n")
    print(f"  Text report saved to: {report_path}")

    # ---- Confusion Matrix Heatmap ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ax=ax, cbar_kws={"label": "Count"},
        annot_kws={"size": 14}
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Confusion matrix heatmap saved to: {cm_path}")

    # ---- Normalized Confusion Matrix ----
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Oranges",
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ax=ax2, cbar_kws={"label": "Proportion"},
        annot_kws={"size": 14}
    )
    ax2.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax2.set_title("Normalized Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()

    cm_norm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test_normalized.png")
    fig2.savefig(cm_norm_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Normalized confusion matrix saved to: {cm_norm_path}")

    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
    }

    return results


if __name__ == "__main__":
    evaluate()
