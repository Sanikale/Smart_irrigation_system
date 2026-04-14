"""
Interactive prediction CLI.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ml_service import predict


def prompt_float(label: str) -> float:
    return float(input(label).strip())


if __name__ == "__main__":
    print("=" * 60)
    print("SMART IRRIGATION SYSTEM - PREDICTION MODULE")
    print("=" * 60)

    while True:
        print("\nEnter crop details or type 'quit' to exit.")
        crop = input("Crop type (Rice/Wheat/Maize/Barley): ").strip()
        if crop.lower() == "quit":
            break

        try:
            payload = {
                "crop": crop,
                "soil_ph": prompt_float("Soil pH (5.5-8.5): "),
                "rainfall_mm": prompt_float("Rainfall (mm): "),
                "temperature_c": prompt_float("Temperature (C): "),
                "humidity_pct": prompt_float("Humidity (%): "),
                "fertilizer_used_kg": 100.0,
                "pesticides_used_kg": 50.0,
                "planting_density": 200.0,
                "irrigation_type": "None",
            }
            result = predict(payload)
            print(f"\nDecision: {result['decision']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Recommendation: {result['recommendation']}")
        except ValueError as exc:
            print(f"Invalid input: {exc}")

    print("\nThank you for using Smart Irrigation System.")
