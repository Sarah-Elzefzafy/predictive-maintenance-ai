from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load models
classification_model = joblib.load("ai_model.pkl")
rul_model = joblib.load("rul_model.pkl")


# Input schema
class SensorData(BaseModel):
    data: list


# Prediction endpoint
@app.post("/predict")
def predict(input_data: SensorData):

    # Convert input to numpy array
    features = np.array(input_data.data)

    # Reshape for model
    reshaped_features = features.reshape(1, -1)

    # Classification prediction
    prediction = classification_model.predict(reshaped_features)[0]

    # Confidence score
    probs = classification_model.predict_proba(reshaped_features)[0]
    confidence = float(max(probs))

    # RUL prediction
    rul = float(rul_model.predict(reshaped_features)[0])

    # Detect most abnormal sensor
    problem_sensor_index = np.argmax(features)
    problem_sensor = f"sensor_{problem_sensor_index + 1}"

    # Work order logic
    if prediction == 0:
        work_order = "Machine operating normally"

    elif prediction == 1:
        work_order = (
            f"Warning: abnormal behavior detected near "
            f"{problem_sensor}. Schedule inspection."
        )

    else:
        work_order = (
            f"CRITICAL: inspect {problem_sensor} immediately."
        )

    # Final response
    return {
        "risk_level": int(prediction),
        "confidence": round(confidence, 3),
        "RUL": round(rul, 2),
        "problem_sensor": problem_sensor,
        "work_order": work_order
    }