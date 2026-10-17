import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Production API using FastAPI and Scikit-Learn",
    version="1.0.0"
)

# Load the trained model pipeline on startup
model_path = "models/heart_disease_pipeline.pkl"
try:
    pipeline = joblib.load(model_path)
except Exception as e:
    pipeline = None
    print(f"Warning: Could not load model from {model_path}. Error: {e}")

# Define Pydantic model for input data validation (matches the 13 UCI features)
class PatientData(BaseModel):
    age: float = Field(..., example=55.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=2.0)
    trestbps: float = Field(..., example=140.0)
    chol: float = Field(..., example=240.0)
    fbs: float = Field(..., example=0.0)
    restecg: float = Field(..., example=1.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=1.5)
    slope: float = Field(..., example=2.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=2.0)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease MLOps Prediction API! Visit /docs for Swagger UI."}

@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded in memory.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(patient: PatientData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not available.")
    
    try:
        # Convert Pydantic model to Pandas DataFrame
        input_df = pd.DataFrame([patient.dict()])
        
        # Make prediction and get probability
        prediction = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0].tolist()
        
        return {
            "prediction": prediction,
            "prediction_label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "probability_no_disease": round(probabilities[0], 4),
            "probability_disease": round(probabilities[1], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))