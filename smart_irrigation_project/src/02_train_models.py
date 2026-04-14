"""
Training script for the Smart Irrigation project.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ml_service import train_model


if __name__ == "__main__":
    metadata = train_model()
    print("=" * 60)
    print("SMART IRRIGATION SYSTEM - MODEL TRAINING")
    print("=" * 60)
    print(f"\nBest model: {metadata['best_model']}")
    print(f"Accuracy: {metadata['accuracy']:.4f}")
    print("\nScores:")
    for model_name, score in metadata["scores"].items():
        print(f"  {model_name}: {score:.4f}")
