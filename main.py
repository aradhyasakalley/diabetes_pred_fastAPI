# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("models/model.pkl")

# Define the input request structure
class PatientDataIn(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    gender_binary: int
    history_int: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API"}


# Endpoint to receive POST requests for diabetes prediction
@app.post("/predict_diabetes/")
def predict_diabetes(patient_data: PatientDataIn):
    # Convert the input data to a numpy array
    input_data = np.array([
        patient_data.age,
        patient_data.hypertension,
        patient_data.heart_disease,
        patient_data.bmi,
        patient_data.HbA1c_level,
        patient_data.blood_glucose_level,
        patient_data.gender_binary,
        patient_data.history_int
    ]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)

    # Return the prediction result
    if prediction[0] == 0:
        return {"prediction": "No Diabetes"}
    else:
        return {"prediction": "Diabetes"}
