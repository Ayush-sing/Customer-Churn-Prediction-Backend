import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def basic_cleaning(df):
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    return df


def encode_target(df):
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def split_features_target(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y


def encode_categorical(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    encoder = OneHotEncoder(drop="first", sparse_output=False)

    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    X_final = pd.concat([X[numeric_cols].reset_index(drop=True),
                         encoded_df.reset_index(drop=True)], axis=1)

    return X_final


def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler