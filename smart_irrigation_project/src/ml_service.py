"""
Core ML service utilities for the Smart Irrigation project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATASET_PATH = DATA_DIR / "crop_yield_dataset.csv"
MODEL_PATH = MODELS_DIR / "irrigation_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

CROPS = ["Rice", "Wheat", "Maize", "Barley"]
REGIONS = ["North", "South", "East", "West"]
SOIL_TYPES = ["Clay", "Loam", "Sandy", "Silt"]
IRRIGATION_TYPES = ["None", "Drip", "Flood", "Sprinkler"]
PREVIOUS_CROPS = ["Rice", "Wheat", "Maize", "Barley", "None"]

CROP_WATER_REQUIREMENTS = {
    "Rice": 800,
    "Wheat": 450,
    "Maize": 500,
    "Barley": 400,
}

CATEGORICAL_COLUMNS = ["Crop", "Region", "Soil_Type", "Irrigation", "Previous_Crop"]
FEATURE_COLUMNS = [
    "Crop_encoded",
    "Soil_pH",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_pct",
    "Fertilizer_Used_kg",
    "Pesticides_Used_kg",
    "Planting_Density",
    "Irrigation_encoded",
]


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def calculate_irrigation_need(row: pd.Series) -> int:
    crop = row["Crop"]
    water_req = CROP_WATER_REQUIREMENTS[crop]
    rainfall = row["Rainfall_mm"]
    temperature = row["Temperature_C"]
    humidity = row["Humidity_pct"]

    thresholds = {"Rice": 0.40, "Wheat": 0.35, "Maize": 0.38, "Barley": 0.32}
    water_available = rainfall + (humidity / 5)
    water_needed = water_req * (temperature / 35)
    return int(water_available < (water_needed * thresholds[crop]))


def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    ensure_directories()
    np.random.seed(random_state)

    df = pd.DataFrame(
        {
            "Crop": np.random.choice(CROPS, n_samples, p=[0.3, 0.3, 0.2, 0.2]),
            "Region": np.random.choice(REGIONS, n_samples),
            "Soil_Type": np.random.choice(SOIL_TYPES, n_samples),
            "Soil_pH": np.random.uniform(5.5, 8.5, n_samples),
            "Rainfall_mm": np.random.exponential(300, n_samples),
            "Temperature_C": np.random.uniform(15, 45, n_samples),
            "Humidity_pct": np.random.uniform(30, 90, n_samples),
            "Fertilizer_Used_kg": np.random.uniform(50, 200, n_samples),
            "Pesticides_Used_kg": np.random.uniform(10, 100, n_samples),
            "Planting_Density": np.random.uniform(150, 300, n_samples),
            "Irrigation": np.random.choice(IRRIGATION_TYPES, n_samples),
            "Previous_Crop": np.random.choice(PREVIOUS_CROPS, n_samples),
        }
    )
    df["irrigation_needed"] = df.apply(calculate_irrigation_need, axis=1)
    df["Yield_ton_per_ha"] = np.random.uniform(2, 8, n_samples)
    df.to_csv(DATASET_PATH, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    ensure_directories()
    if not DATASET_PATH.exists():
        return generate_dataset()
    df = pd.read_csv(DATASET_PATH, keep_default_na=False)
    for column in ["Irrigation", "Previous_Crop"]:
        df[column] = df[column].replace("", "None").fillna("None")
    return df


def _encode_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded[f"{column}_encoded"] = encoder.fit_transform(encoded[column])
        label_encoders[column] = encoder

    return encoded, label_encoders


def _generate_visuals(y_test: pd.Series, y_pred: np.ndarray, model: Any) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Irrigation", "Irrigation Needed"],
        yticklabels=["No Irrigation", "Irrigation Needed"],
    )
    plt.title("Confusion Matrix - Best Model")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance for Irrigation Prediction")
        plt.bar(range(len(importances)), importances[indices], color="#2f855a")
        plt.xticks(
            range(len(importances)),
            [FEATURE_COLUMNS[i] for i in indices],
            rotation=45,
            ha="right",
        )
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
        plt.close()


def train_model() -> dict[str, Any]:
    ensure_directories()
    df = load_dataset()
    encoded_df, label_encoders = _encode_dataframe(df)

    X = encoded_df[FEATURE_COLUMNS]
    y = encoded_df["irrigation_needed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest (Deep)": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42
        ),
    }

    best_name = ""
    best_model: Any = None
    best_accuracy = -1.0
    best_predictions: np.ndarray | None = None
    report_text = ""
    scores: dict[str, float] = {}

    for name, model in candidate_models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        scores[name] = round(float(accuracy), 4)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_name = name
            best_model = model
            best_predictions = predictions
            report_text = classification_report(
                y_test,
                predictions,
                target_names=["No Irrigation", "Irrigation Needed"],
            )

    if best_model is None or best_predictions is None:
        raise RuntimeError("Model training failed.")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)

    _generate_visuals(y_test, best_predictions, best_model)

    metadata = {
        "best_model": best_name,
        "accuracy": round(best_accuracy, 4),
        "dataset_rows": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "class_balance": {
            "no_irrigation": int((y == 0).sum()),
            "irrigation_needed": int((y == 1).sum()),
        },
        "scores": scores,
        "classification_report": report_text,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def model_ready() -> bool:
    return MODEL_PATH.exists() and SCALER_PATH.exists() and ENCODERS_PATH.exists()


def load_artifacts() -> tuple[Any, StandardScaler, dict[str, LabelEncoder]]:
    if not model_ready():
        train_model()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, label_encoders


def _safe_transform(encoder: LabelEncoder, value: str, field_name: str) -> int:
    if value not in encoder.classes_:
        allowed = ", ".join(map(str, encoder.classes_))
        raise ValueError(f"Invalid {field_name}. Allowed values: {allowed}")
    return int(encoder.transform([value])[0])


def build_recommendation(payload: dict[str, Any], prediction: int, confidence: float) -> str:
    crop = payload["crop"]
    rainfall = payload["rainfall_mm"]
    temperature = payload["temperature_c"]
    humidity = payload["humidity_pct"]
    water_req = CROP_WATER_REQUIREMENTS.get(crop, 500)

    if prediction == 1:
        if rainfall < 150:
            return (
                f"{crop} is under low-rainfall stress. Start irrigation soon and aim for "
                f"roughly {int(water_req * 0.25)} mm supplemental water in this cycle."
            )
        if temperature > 35:
            return (
                f"High temperature is increasing water demand for {crop}. Irrigate during "
                "early morning or evening to reduce evaporation loss."
            )
        return (
            f"The model is {confidence:.1f}% confident irrigation is needed. Provide a "
            "moderate watering cycle and recheck moisture after application."
        )

    if humidity > 70 and rainfall > 250:
        return "Current weather looks supportive. Delay irrigation and monitor for the next day."
    return "Soil-water conditions look acceptable for now. Continue monitoring instead of irrigating immediately."


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    model, scaler, label_encoders = load_artifacts()

    crop_encoded = _safe_transform(label_encoders["Crop"], payload["crop"], "crop")
    irrigation_encoded = _safe_transform(
        label_encoders["Irrigation"], payload["irrigation_type"], "irrigation_type"
    )

    feature_vector = np.array(
        [
            [
                crop_encoded,
                payload["soil_ph"],
                payload["rainfall_mm"],
                payload["temperature_c"],
                payload["humidity_pct"],
                payload["fertilizer_used_kg"],
                payload["pesticides_used_kg"],
                payload["planting_density"],
                irrigation_encoded,
            ]
        ]
    )

    scaled = scaler.transform(feature_vector)
    prediction = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]
    confidence = float(probabilities[prediction] * 100)
    recommended_water_mm = (
        int(CROP_WATER_REQUIREMENTS[payload["crop"]] * 0.25) if prediction == 1 else 0
    )

    return {
        "prediction": prediction,
        "decision": "Irrigation Needed" if prediction == 1 else "No Irrigation Needed",
        "confidence": round(confidence, 2),
        "recommended_water_mm": recommended_water_mm,
        "recommendation": build_recommendation(payload, prediction, confidence),
        "input_summary": payload,
    }


def get_model_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return train_model()
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def get_dashboard_stats() -> dict[str, Any]:
    df = load_dataset()
    metadata = get_model_metadata() if model_ready() or METADATA_PATH.exists() else None

    crop_counts = {
        crop: int(count) for crop, count in df["Crop"].value_counts().sort_index().items()
    }
    irrigation_rate = round(float(df["irrigation_needed"].mean() * 100), 2)

    return {
        "dataset_rows": int(len(df)),
        "crops": crop_counts,
        "irrigation_rate": irrigation_rate,
        "avg_temperature": round(float(df["Temperature_C"].mean()), 2),
        "avg_rainfall": round(float(df["Rainfall_mm"].mean()), 2),
        "model": metadata,
        "visualizations": [
            "/outputs/confusion_matrix.png",
            "/outputs/feature_importance.png",
        ],
    }
