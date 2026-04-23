"""
main.py — Orchestrator for the Agricultural Stress Detection Pipeline.

Runs the complete pipeline:
1. Preprocessing — Compute stress labels from mask data
2. Visualization — Generate data distribution plots
3. Training — Train the late-fusion dual-branch CNN
4. Evaluation — Test-set evaluation with full metrics
5. Post-training visualization — Loss/accuracy curves, confusion matrix

Usage:
    python main.py
"""

import os
import sys
import time
import torch
import numpy as np
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RANDOM_SEED, OUTPUT_DIR, MODEL_DIR


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    """Run the complete pipeline."""
    pipeline_start = time.time()

    # Set seeds
    set_seed(RANDOM_SEED)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ============================================================
    # STEP 1: Preprocessing
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 1: PREPROCESSING — Computing Stress Labels")
    print("=" * 70)

    from src.preprocess import preprocess
    step_start = time.time()
    manifest = preprocess()
    print(f"\n  Step 1 completed in {time.time() - step_start:.1f}s")

    # ============================================================
    # STEP 2: Pre-training Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 2: PRE-TRAINING VISUALIZATION")
    print("=" * 70)

    from src.visualize import visualize_distribution
    step_start = time.time()
    visualize_distribution()
    print(f"\n  Step 2 completed in {time.time() - step_start:.1f}s")

    # ============================================================
    # STEP 3: Training
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 3: TRAINING — Late Fusion Dual-Branch CNN")
    print("=" * 70)

    from src.train import train
    step_start = time.time()
    history = train()
    print(f"\n  Step 3 completed in {time.time() - step_start:.1f}s")

    # ============================================================
    # STEP 4: Test Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 4: EVALUATION — Test Set")
    print("=" * 70)

    from src.evaluate import evaluate
    step_start = time.time()
    results = evaluate()
    print(f"\n  Step 4 completed in {time.time() - step_start:.1f}s")

    # ============================================================
    # STEP 5: Post-training Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 5: POST-TRAINING VISUALIZATION")
    print("=" * 70)

    from src.visualize import visualize_results
    step_start = time.time()
    visualize_results()
    print(f"\n  Step 5 completed in {time.time() - step_start:.1f}s")

    # ============================================================
    # PIPELINE COMPLETE
    # ============================================================
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Model saved at:      {os.path.join(MODEL_DIR, 'best_model.pth')}")
    print(f"  Outputs saved at:    {OUTPUT_DIR}/")
    print(f"\n  Final Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Final Test F1 (macro): {results['f1_macro']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
