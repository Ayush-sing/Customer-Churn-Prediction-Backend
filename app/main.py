from fastapi import FastAPI, Depends, HTTPException
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://customer-churn-prediction-frontend-three.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model package
MODEL_PATH = os.path.join(BASE_DIR, "../data/processed/churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/telco.csv")

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
threshold = model_data["threshold"]
feature_names = model_data["feature_names"]
explainer = model_data["explainer"]



# ---------------- PREDICT ----------------

@app.post("/predict")
def predict(data: dict):
    try:
        values = np.array([data[feature] for feature in feature_names]).reshape(1, -1)
        values_scaled = scaler.transform(values)

        proba = model.predict_proba(values_scaled)[0][1]
        prediction = int(proba >= threshold)

        return {
            "probability": float(proba),
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- BATCH PREDICT ----------------

@app.get("/batch_predict")
def batch_predict():
    df = pd.read_csv(DATA_PATH)

    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop("Churn", axis=1)

    # Re-encode like training
    from app.ml.preprocess import encode_categorical
    X_encoded = encode_categorical(X)

    values_scaled = scaler.transform(X_encoded)

    probabilities = model.predict_proba(values_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    df["churn_probability"] = probabilities
    df["churn_prediction"] = predictions

    return df.head(10).to_dict(orient="records")


# ---------------- ANALYTICS ----------------

@app.get("/analytics")
def analytics():

    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop("Churn", axis=1)

    from app.ml.preprocess import encode_categorical
    X_encoded = encode_categorical(X)

    values_scaled = scaler.transform(X_encoded)

    probabilities = model.predict_proba(values_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    df["predicted_probability"] = probabilities
    df["predicted_churn"] = predictions

    total_customers = len(df)
    predicted_churn_rate = float(predictions.mean())

    # Revenue at risk (based on prediction, not actual label)
    revenue_at_risk = float(
        df[df["predicted_churn"] == 1]["TotalCharges"].sum()
    )

    # Risk segmentation by probability
    df["risk_segment"] = pd.cut(
        df["predicted_probability"],
        bins=[0, 0.3, 0.6, 1],
        labels=["Low", "Medium", "High"]
    )

    segment_counts = df["risk_segment"].value_counts().to_dict()

    return {
        "total_customers": total_customers,
        "predicted_churn_rate": predicted_churn_rate,
        "revenue_at_risk": revenue_at_risk,
        "risk_segmentation": segment_counts
    }


# ---------------- FEATURE IMPORTANCE ----------------

@app.get("/feature_importance")
def feature_importance():

    importance = model.feature_importances_

    feature_importance_dict = dict(
        sorted(
            [(feature, float(score)) for feature, score in zip(feature_names, importance)],
            key=lambda x: x[1],
            reverse=True
        )
    )

    return feature_importance_dict