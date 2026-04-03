"""
src/models/train.py
===================
Baseline training script for Milestone 4.

This script:
  - loads processed train/validation data from Milestone 3
  - trains a simple TF-IDF + Linear Regression baseline
  - logs parameters, metrics, and model artefacts to MLflow

Usage
-----
    cd "Milestone 4"
    python src/models/train.py
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def main():
    # Resolve project paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mlruns_dir = os.path.join(base_dir, "mlruns")
    tracking_uri = "file:///" + mlruns_dir.replace("\\", "/")

    train_path = os.path.join(base_dir, "..", "Milestone 3", "data", "processed", "train.parquet")
    val_path = os.path.join(base_dir, "..", "Milestone 3", "data", "processed", "val.parquet")

    os.makedirs(mlruns_dir, exist_ok=True)

    # MLflow configuration
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Default")

    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = train_df["input_text"].fillna("")
    y_train = train_df["storypoints"]

    X_val = val_df["input_text"].fillna("")
    y_val = val_df["storypoints"]

    with mlflow.start_run():
        # Vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        # Model
        model = LinearRegression()
        model.fit(X_train_vec, y_train)

        # Validation
        y_pred = model.predict(X_val_vec)
        mae = mean_absolute_error(y_val, y_pred)

        # Logging
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("MAE", mae)

        mlflow.sklearn.log_model(model, "model")

        print("Tracking URI:", tracking_uri)
        print("MAE:", mae)


if __name__ == "__main__":
    main()