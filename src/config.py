"""
config.py — Central configuration for the Agriculture Stress Detection pipeline.
All paths, hyperparameters, thresholds, and constants live here.
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data2019_miniscale")

RGB_DIR = os.path.join(DATASET_ROOT, "field_images", "rgb")
NIR_DIR = os.path.join(DATASET_ROOT, "field_images", "nir")
MASK_DIR = os.path.join(DATASET_ROOT, "field_masks")
BOUNDS_DIR = os.path.join(DATASET_ROOT, "field_bounds")
LABELS_DIR = os.path.join(DATASET_ROOT, "field_labels")

FIELD_STATS_PATH = os.path.join(DATASET_ROOT, "field_stats.json")
STATS_PATH = os.path.join(DATASET_ROOT, "stats.json")
SPLIT_STATS_PATH = os.path.join(DATASET_ROOT, "split_stats.json")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MANIFEST_PATH = os.path.join(MODEL_DIR, "label_manifest.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ============================================================
# Label Configuration
# ============================================================
# Labels that contribute to stress area (used in ratio calculation)
STRESS_LABELS = ["nutrient_deficiency", "water", "weed_cluster"]

# Labels excluded from stress area (structural farm features)
EXCLUDED_LABELS = ["waterway"]

# Catastrophic labels — auto-assign "Severe Stress"
CATASTROPHIC_LABELS = ["planter_skip"]

# Labels to completely ignore (0 instances in dataset)
IGNORED_LABELS = ["double_plant", "drydown", "endrow", "storm_damage"]

# ============================================================
# Stress Thresholds
# ============================================================
HEALTHY_THRESHOLD = 0.0950  # ratio <= 9.5%  → Healthy
EARLY_STRESS_THRESHOLD = 0.4070  # 9.5% < ratio <= 40.7% → Early Stress
# ratio > 40.7% → Severe Stress

# ============================================================
# Class Mapping
# ============================================================
CLASS_NAMES = {0: "Healthy", 1: "Early Stress", 2: "Severe Stress"}
NUM_CLASSES = 3

# ============================================================
# Training Hyperparameters
# ============================================================
IMAGE_SIZE = 224  # Resize images to 128×128 or 224×224
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
TRAIN_SUBSET_RATIO = 1.0  # To Use 50% of training data - 0.5

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42
