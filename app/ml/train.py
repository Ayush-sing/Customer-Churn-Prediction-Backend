import pandas as pd
from preprocess import (
    load_data,
    basic_cleaning,
    encode_target,
    split_features_target,
    encode_categorical,
    train_test_split_data,
    scale_data
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, f1_score
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


DATA_PATH = "../../data/raw/telco.csv"
MODEL_PATH = "../../data/processed/churn_model.pkl"


def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return best_threshold, best_f1


def run_training_pipeline():

    df = load_data(DATA_PATH)
    df = basic_cleaning(df)
    df = encode_target(df)

    X, y = split_features_target(df)
    X_encoded = encode_categorical(X)

    feature_names = X_encoded.columns

    X_train, X_test, y_train, y_test = train_test_split_data(X_encoded, y)

    print("Before SMOTE:", np.bincount(y_train))

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", np.bincount(y_train_smote))

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train_smote, X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train_scaled, y_train_smote)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("\nFinal XGBoost ROC-AUC:", roc_auc)

    best_threshold, best_f1 = find_best_threshold(y_test, y_pred_proba)

    print(f"\nBest Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")

    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    print("\nOptimized Threshold Report:")
    print(classification_report(y_test, y_pred_optimized))

    # 🔥 SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    print("\nGenerating SHAP summary plot...")

    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)

    # Save everything
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "threshold": best_threshold,
        "feature_names": feature_names,
        "explainer": explainer
    }, MODEL_PATH)

    print("\nModel + threshold + explainer saved successfully.")


if __name__ == "__main__":
    run_training_pipeline()