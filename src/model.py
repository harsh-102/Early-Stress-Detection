"""
model.py — Late Fusion Dual-Branch CNN for agricultural stress detection.

Architecture:
- RGB Branch:      3-channel input (R, G, B)       → 4 Conv blocks → Flatten
- Spectral Branch: 3-channel input (NIR, NDVI, GNDVI) → 4 Conv blocks → Flatten
- Fusion: Concatenate flattened features → FC layers → 3-class output

Each Conv block: Conv2d → BatchNorm2d → ReLU → MaxPool2d
Final spatial reduction: AdaptiveAvgPool2d(4×4)
"""

import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    """
    A CNN feature extraction branch with 4 convolutional blocks.

    Input:  (batch, in_channels, H, W)
    Output: (batch, 256, 4, 4) = (batch, 4096) when flattened
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: in_channels → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # → (batch, 256, 4, 4)
        )

    def forward(self, x):
        return self.features(x)


class LateFusionCNN(nn.Module):
    """
    Late Fusion CNN that processes RGB and Spectral data through separate branches,
    then fuses their features at the fully connected layers.

    Input:  (batch, 6, H, W) — [R, G, B, NIR, NDVI, GNDVI]
    Output: (batch, num_classes) — logits (no softmax)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Branch 1: RGB (channels 0, 1, 2)
        self.rgb_branch = CNNBranch(in_channels=3)

        # Branch 2: Spectral (channels 3, 4, 5) = NIR, NDVI, GNDVI
        self.spectral_branch = CNNBranch(in_channels=3)

        # After AdaptiveAvgPool2d(4): each branch outputs (batch, 256, 4, 4)
        # Flattened: 256 * 4 * 4 = 4096 per branch
        # Concatenated: 4096 + 4096 = 8192

        self.classifier = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Split the 6-channel input
        rgb_input = x[:, :3, :, :]       # R, G, B
        spectral_input = x[:, 3:, :, :]  # NIR, NDVI, GNDVI

        # Extract features from each branch
        rgb_features = self.rgb_branch(rgb_input)
        spectral_features = self.spectral_branch(spectral_input)

        # Flatten
        rgb_flat = rgb_features.view(rgb_features.size(0), -1)
        spectral_flat = spectral_features.view(spectral_features.size(0), -1)

        # Late Fusion: concatenate features
        fused = torch.cat([rgb_flat, spectral_flat], dim=1)

        # Classification
        logits = self.classifier(fused)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Print a summary of the model architecture and parameter count."""
    print(f"\n  {'='*60}")
    print(f"  MODEL ARCHITECTURE: Late Fusion Dual-Branch CNN")
    print(f"  {'='*60}")
    print(f"\n  RGB Branch (3ch → structural anomalies):")
    print(f"    Conv2d(3→32→64→128→256) + BN + ReLU + Pool")
    print(f"\n  Spectral Branch (3ch → physiological anomalies):")
    print(f"    Conv2d(3→32→64→128→256) + BN + ReLU + Pool")
    print(f"\n  Fusion Classifier:")
    print(f"    Concat(4096+4096) → FC(8192→512) → FC(512→128) → FC(128→3)")
    print(f"\n  Total trainable parameters: {count_parameters(model):,}")
    print(f"  {'='*60}\n")
