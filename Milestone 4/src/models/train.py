import os
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.utils.config import load_config


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "unknown"


def _build_pipeline(cfg: dict) -> Pipeline:
    max_features = int(cfg["baseline"]["tfidf"]["max_features"])
    ngram_range = tuple(cfg["baseline"]["tfidf"]["ngram_range"])
    fit_intercept = bool(cfg["baseline"]["linear_regression"].get("fit_intercept", True))

    def select_input_text(df: pd.DataFrame) -> pd.Series:
        return df["input_text"].fillna("").astype(str)

    selector = FunctionTransformer(select_input_text, validate=False)

    return Pipeline(
        steps=[
            ("select_text", selector),
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ("reg", LinearRegression(fit_intercept=fit_intercept)),
        ]
    )


def _load_data(m4_root: Path):
    train_path = m4_root / ".." / "Milestone 3" / "data" / "processed" / "train.parquet"
    val_path = m4_root / ".." / "Milestone 3" / "data" / "processed" / "val.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    return train_df, val_df


def main():
    cfg = load_config()
    m4_root = Path(__file__).resolve().parents[2]
    repo_root = m4_root.parent

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    train_df, val_df = _load_data(m4_root)

    X_train = train_df[["input_text"]]
    y_train = train_df["storypoints"].astype(float)

    X_val = val_df[["input_text"]]
    y_val = val_df["storypoints"].astype(float)

    pipeline = _build_pipeline(cfg)

    with mlflow.start_run() as run:

        mlflow.set_tag("git_commit", _git_sha(repo_root))
        mlflow.set_tag("model_type", "tfidf_linear_regression")

        # Train
        pipeline.fit(X_train, y_train)

        # Validate
        y_val_pred = pipeline.predict(X_val)

        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = val_mse ** 0.5

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)

        # Save feature importance
        tfidf = pipeline.named_steps["tfidf"]
        reg = pipeline.named_steps["reg"]

        feature_names = tfidf.get_feature_names_out()
        coefs = reg.coef_.ravel()

        top_idx = np.argsort(coefs)[-10:]

        report = [
            {"feature": feature_names[i], "coef": float(coefs[i])}
            for i in top_idx
        ]

        output_path = m4_root / "coefficients.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact(str(output_path))

        # Log + REGISTER MODEL
        signature = infer_signature(X_val, y_val_pred)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=cfg["baseline"]["registered_model_name"],
            signature=signature,
            input_example=X_train.head(2),
        )

        print("RUN_ID:", run.info.run_id)
        print("MODEL_URI:", model_info.model_uri)
        print("VAL_MAE:", val_mae)


if __name__ == "__main__":
    main()