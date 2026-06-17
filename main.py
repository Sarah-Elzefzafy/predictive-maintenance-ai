from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models
classification_model = joblib.load("ai_model.pkl")
rul_model = joblib.load("rul_model.pkl")
metrics = joblib.load("metrics.pkl")

# Normal sensor ranges
NORMAL_RANGES = {
    "sensor_1": (518.67, 518.67),
    "sensor_2": (641.92, 643.58),
    "sensor_3": (1581.11, 1601.47),
    "sensor_4": (1395.62, 1425.67),
    "sensor_5": (14.62, 14.62),
    "sensor_6": (21.61, 21.61),
    "sensor_7": (551.74, 554.69),
    "sensor_8": (2387.99, 2388.22),
    "sensor_9": (9042.55, 9109.98),
    "sensor_10": (1.30, 1.30),
    "sensor_11": (47.15, 48.04),
    "sensor_12": (520.04, 522.50),
    "sensor_13": (2387.99, 2388.23),
    "sensor_14": (8122.50, 8181.40),
    "sensor_15": (8.39, 8.51),
    "sensor_16": (0.03, 0.03),
    "sensor_17": (391.00, 396.00),
    "sensor_18": (2388.00, 2388.00),
    "sensor_19": (100.00, 100.00),
    "sensor_20": (38.49, 39.09),
    "sensor_21": (23.09, 23.45)
}


class SensorData(BaseModel):
    data: list


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}


@app.post("/predict")
def predict(input_data: SensorData):

    # Convert input
    features = np.array(input_data.data)
    reshaped_features = features.reshape(1, -1)

    # Classification
    prediction = classification_model.predict(reshaped_features)[0]

    # Confidence
    probs = classification_model.predict_proba(reshaped_features)[0]
    confidence = float(max(probs))

    # RUL
    rul = float(rul_model.predict(reshaped_features)[0])

    # Find most abnormal sensor
    max_deviation = -1
    problem_sensor = "None"
    current_value = 0
    normal_min = 0
    normal_max = 0

    for i, value in enumerate(features):
        sensor_name = f"sensor_{i+1}"

        low, high = NORMAL_RANGES[sensor_name]

        if value < low:
            deviation = low - value
        elif value > high:
            deviation = value - high
        else:
            deviation = 0

        if deviation > max_deviation:
            max_deviation = deviation
            problem_sensor = sensor_name
            current_value = float(value)
            normal_min = low
            normal_max = high

    
    # If no sensor is abnormal
    if max_deviation == 0:
        problem_sensor = "None"
        current_value = None
        normal_min = None
        normal_max = None

    # Work order
    if prediction == 0:
        problem_sensor = "None"
        current_value = None
        normal_min = None
        normal_max = None

        work_order = "Machine operating normally"

    elif prediction == 1:
        work_order = (
            f"Warning: inspect {problem_sensor}. "
            f"Reading={current_value:.2f}, "
            f"Normal Range=({normal_min:.2f}-{normal_max:.2f})"
        )

    else:
        work_order = (
            f"CRITICAL: inspect {problem_sensor} immediately. "
            f"Reading={current_value:.2f}, "
            f"Normal Range=({normal_min:.2f}-{normal_max:.2f})"
        )

    return {
        "risk_level": int(prediction),
        "confidence": round(confidence, 3),
        "RUL": round(rul, 2),

        "problem_sensor": problem_sensor,
        "current_value": (
            round(current_value, 2)
            if current_value is not None
            else None
        ),
        "normal_min": normal_min,
        "normal_max": normal_max,

        "work_order": work_order,

        "model_metrics": {
            "accuracy": round(float(metrics["accuracy"]), 4),
            "precision": round(float(metrics["precision"]), 4),
            "recall": round(float(metrics["recall"]), 4),
            "f1_score": round(float(metrics["f1_score"]), 4)
        }
    }