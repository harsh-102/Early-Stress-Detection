"""
preprocess.py — Compute stress labels for all images using area-based thresholding.

For each image:
1. Parse field_stats.json to get train/val/test split assignments
2. Check for catastrophic labels (planter_skip) → auto Severe
3. Compute valid field area (bounds ∩ mask) and stressed area (union of stress masks)
4. Calculate stress ratio and apply thresholds
5. Save results to models/label_manifest.csv
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    FIELD_STATS_PATH, BOUNDS_DIR, MASK_DIR, LABELS_DIR,
    STRESS_LABELS, CATASTROPHIC_LABELS,
    HEALTHY_THRESHOLD, EARLY_STRESS_THRESHOLD,
    MODEL_DIR, MANIFEST_PATH, CLASS_NAMES
)


def _extract_image_id(path: str) -> str:
    """Extract image ID from a field_stats.json path key."""
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]


def _load_mask_grayscale(filepath: str) -> np.ndarray | None:
    """Load a mask image as grayscale. Returns None if file doesn't exist."""
    if not os.path.exists(filepath):
        return None
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def _get_split_assignments() -> dict:
    """
    Parse field_stats.json to get image IDs per split and their label counts.
    Returns: {split_name: {image_id: label_counts_dict}}
    """
    print("  Loading field_stats.json (this may take a few seconds)...")
    with open(FIELD_STATS_PATH, "r") as f:
        field_stats = json.load(f)

    splits = {}
    for split_name, images in field_stats.items():
        split_data = {}
        for path, stats in images.items():
            image_id = _extract_image_id(path)
            split_data[image_id] = stats.get("label_counts", {})
        splits[split_name] = split_data

    return splits


def _has_catastrophic_label(label_counts: dict) -> bool:
    """Check if any catastrophic label has non-zero count."""
    for label in CATASTROPHIC_LABELS:
        if label_counts.get(label, 0) > 0:
            return True
    return False


def _has_any_stress_label(label_counts: dict) -> bool:
    """Check if any stress-contributing label has non-zero count."""
    for label in STRESS_LABELS:
        if label_counts.get(label, 0) > 0:
            return True
    return False


def _compute_stress_ratio(image_id: str) -> tuple:
    """
    Compute the stress ratio for an image by loading pixel data.
    Returns: (ratio: float, valid_area: int, stressed_area: int)
    """
    # Load bounds and mask
    bounds = _load_mask_grayscale(os.path.join(BOUNDS_DIR, f"{image_id}.png"))
    mask = _load_mask_grayscale(os.path.join(MASK_DIR, f"{image_id}.png"))

    if bounds is None or mask is None:
        return 0.0, 0, 0

    # Valid area = intersection of bounds and mask
    valid_mask = (bounds > 0) & (mask > 0)
    valid_area = int(np.sum(valid_mask))

    if valid_area == 0:
        return 0.0, 0, 0

    # Build union of all stress masks
    stress_union = np.zeros(bounds.shape, dtype=bool)
    for stress_label in STRESS_LABELS:
        label_path = os.path.join(LABELS_DIR, stress_label, f"{image_id}.png")
        label_mask = _load_mask_grayscale(label_path)
        if label_mask is not None:
            stress_union = stress_union | (label_mask > 0)

    # Only count stressed pixels within valid area
    stressed_area = int(np.sum(stress_union & valid_mask))

    ratio = stressed_area / valid_area
    return ratio, valid_area, stressed_area


def _assign_label(ratio: float) -> int:
    """Apply thresholds to convert stress ratio to class label."""
    if ratio <= HEALTHY_THRESHOLD:
        return 0  # Healthy
    elif ratio <= EARLY_STRESS_THRESHOLD:
        return 1  # Early Stress
    else:
        return 2  # Severe Stress


def preprocess() -> pd.DataFrame:
    """
    Run the full preprocessing pipeline:
    1. Get split assignments from field_stats.json
    2. Compute stress labels for all images
    3. Save manifest CSV to models/label_manifest.csv

    Returns: DataFrame with columns [image_id, split, label, stress_ratio,
             valid_area, stressed_area, has_planter_skip]
    """
    # Check if manifest already exists
    if os.path.exists(MANIFEST_PATH):
        print(f"  Manifest already exists at {MANIFEST_PATH}")
        print("  Loading existing manifest. Delete it to recompute.")
        return pd.read_csv(MANIFEST_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Get split assignments
    split_assignments = _get_split_assignments()

    total_images = sum(len(v) for v in split_assignments.values())
    print(f"  Found {total_images} images across {list(split_assignments.keys())} splits")

    # Step 2: Compute labels for all images
    records = []
    skipped_healthy = 0
    skipped_catastrophic = 0
    pixel_computed = 0

    for split_name, images in split_assignments.items():
        print(f"\n  Processing '{split_name}' split ({len(images)} images)...")

        for image_id, label_counts in tqdm(images.items(), desc=f"    {split_name}"):
            has_planter_skip = _has_catastrophic_label(label_counts)

            # Fast path: catastrophic label → auto severe
            if has_planter_skip:
                records.append({
                    "image_id": image_id,
                    "split": split_name,
                    "label": 2,
                    "stress_ratio": -1.0,  # Sentinel: auto-severe
                    "valid_area": -1,
                    "stressed_area": -1,
                    "has_planter_skip": True,
                })
                skipped_catastrophic += 1
                continue

            # Fast path: no stress labels at all → healthy
            if not _has_any_stress_label(label_counts):
                records.append({
                    "image_id": image_id,
                    "split": split_name,
                    "label": 0,
                    "stress_ratio": 0.0,
                    "valid_area": -1,
                    "stressed_area": 0,
                    "has_planter_skip": False,
                })
                skipped_healthy += 1
                continue

            # Slow path: need pixel-level computation
            ratio, valid_area, stressed_area = _compute_stress_ratio(image_id)
            label = _assign_label(ratio)
            pixel_computed += 1

            records.append({
                "image_id": image_id,
                "split": split_name,
                "label": label,
                "stress_ratio": round(ratio, 6),
                "valid_area": valid_area,
                "stressed_area": stressed_area,
                "has_planter_skip": False,
            })

    # Build DataFrame
    manifest = pd.DataFrame(records)

    # Step 3: Save manifest
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\n  Manifest saved to: {MANIFEST_PATH}")

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  PREPROCESSING SUMMARY")
    print(f"  {'='*50}")
    print(f"  Total images processed: {len(manifest)}")
    print(f"  - Auto-severe (planter_skip): {skipped_catastrophic}")
    print(f"  - Auto-healthy (no stress):   {skipped_healthy}")
    print(f"  - Pixel-level computed:        {pixel_computed}")
    print(f"\n  CLASS DISTRIBUTION (overall):")
    for label_val in sorted(manifest["label"].unique()):
        count = (manifest["label"] == label_val).sum()
        pct = count / len(manifest) * 100
        print(f"    {CLASS_NAMES[label_val]:>15s}: {count:6d} ({pct:5.1f}%)")

    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name]
        print(f"\n  CLASS DISTRIBUTION ({split_name}, n={len(split_df)}):")
        for label_val in sorted(split_df["label"].unique()):
            count = (split_df["label"] == label_val).sum()
            pct = count / len(split_df) * 100
            print(f"    {CLASS_NAMES[label_val]:>15s}: {count:6d} ({pct:5.1f}%)")

    return manifest


if __name__ == "__main__":
    preprocess()
