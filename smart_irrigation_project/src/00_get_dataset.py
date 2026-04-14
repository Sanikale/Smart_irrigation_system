"""
Generate the dataset used by the Smart Irrigation project.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ml_service import DATASET_PATH, generate_dataset


if __name__ == "__main__":
    df = generate_dataset()
    print("=" * 60)
    print("CREATING SMART IRRIGATION DATASET")
    print("=" * 60)
    print(f"\nDataset created successfully with {len(df)} rows")
    print(f"Saved to: {DATASET_PATH}")
    print("\nClass distribution:")
    print(df["irrigation_needed"].value_counts())
