import os
import joblib
import pandas as pd
import numpy as np
from src.preprocess import build_preprocessor, ALL_FEATURES, TARGET_COL

def test_cleaned_data_exists():
    """Verify that the cleaned UCI dataset exists and is non-empty."""
    data_path = "data/processed/heart_disease_cleaned.csv"
    assert os.path.exists(data_path), "Cleaned dataset file not found!"
    df = pd.read_csv(data_path)
    assert len(df) > 0, "Dataset is empty!"
    assert TARGET_COL in df.columns, "Target column missing from dataset!"

def test_preprocessor_transformation():
    """Verify that the ColumnTransformer outputs the correct shape without errors."""
    data_path = "data/processed/heart_disease_cleaned.csv"
    df = pd.read_csv(data_path)

    X = df[ALL_FEATURES]
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Check that output is a 2D numpy array and has more columns than raw (due to one-hot encoding)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(df)
    assert X_transformed.shape[1] > len(ALL_FEATURES)

def test_model_artifact_and_inference():
    """Verify that the saved model artifact loads and returns valid predictions."""
    model_path = "models/heart_disease_pipeline.pkl"
    assert os.path.exists(model_path), "Champion model pickle artifact not found!"

    pipeline = joblib.load(model_path)

    # Create a synthetic sample matching the 13 input features
    sample_data = pd.DataFrame([{
        "age": 55.0, "sex": 1.0, "cp": 2.0, "trestbps": 140.0, "chol": 240.0,
        "fbs": 0.0, "restecg": 1.0, "thalach": 150.0, "exang": 0.0,
        "oldpeak": 1.5, "slope": 2.0, "ca": 0.0, "thal": 2.0
    }])

    prediction = pipeline.predict(sample_data)
    probability = pipeline.predict_proba(sample_data)

    # Assertions
    assert prediction[0] in [0, 1], "Prediction must be binary (0 or 1)"
    assert probability.shape == (1, 2), "Probability output shape must be (1, 2)"
    assert 0.0 <= probability[0][1] <= 1.0, "Probability must be between 0 and 1"
