"""
Create visualizations by retraining the model pipeline.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ml_service import get_dashboard_stats, train_model


if __name__ == "__main__":
    train_model()
    stats = get_dashboard_stats()
    print("=" * 60)
    print("SMART IRRIGATION SYSTEM - DATA VISUALIZATION")
    print("=" * 60)
    print("\nVisualizations available in the outputs folder:")
    for file_path in stats["visualizations"]:
        print(f"  {file_path}")
