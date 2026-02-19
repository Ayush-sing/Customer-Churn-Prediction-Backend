# Customer Churn Prediction – Backend API

This backend powers a production-style Customer Churn Intelligence Dashboard.  
It provides machine learning predictions, analytics, and feature importance using a trained XGBoost model with SHAP explainability.

---

## DEMO Link
``` bash


https://customer-churn-prediction-frontend-three.vercel.app/dashboard

```

---

## 🚀 Tech Stack

- FastAPI
- XGBoost
- Scikit-learn
- SHAP
- Pandas / NumPy
- Uvicorn

---

## 🧠 Model Overview

- Dataset: Telco Customer Churn
- Class imbalance handled using SMOTE
- Model: Tuned XGBoost Classifier
- Threshold optimized using F1-score
- Explainability: SHAP (TreeExplainer)

The API predicts churn probability and provides business-level analytics such as revenue at risk and risk segmentation.

---

## 📂 Project Structure

```bash
backend/
│
├── app/
│ ├── main.py
│ └── ml/
│ ├── preprocess.py
│ └── train.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── requirements.txt
└── start.sh
```
---

## 🔌 API Endpoints

### `GET /analytics`
Returns:
- Total customers
- Predicted churn rate
- Revenue at risk
- Risk segmentation (Low / Medium / High)

---

### `GET /feature_importance`
Returns model feature importance scores sorted by contribution weight.

---

### `POST /predict`
Accepts encoded customer features and returns:
- Churn probability
- Binary churn prediction (based on optimized threshold)

---

### `GET /batch_predict`
Runs predictions on the full dataset and returns sample results.

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
Deployment

Designed for cloud deployment (Railway, etc.) using:

uvicorn app.main:app --host 0.0.0.0 --port $PORT

## Purpose

This backend demonstrates:

End-to-end ML pipeline

Production-ready API design

Business-oriented analytics

Model explainability integration
