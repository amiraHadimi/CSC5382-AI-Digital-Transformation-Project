"""
tracking/mlflow_tracker.py
==========================
MLflow tracking utility for Milestone 4.

Handles:
  - experiment setup
  - run lifecycle
  - logging metrics, params, artefacts
"""

import os
import mlflow


class MLflowTracker:
    def __init__(self, cfg):
        self.cfg = cfg

        # Resolve Milestone 4 root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        mlruns_dir = os.path.join(base_dir, "mlruns")

        os.makedirs(mlruns_dir, exist_ok=True)

        # ✅ FIX: proper file URI for Windows
        tracking_uri = "file:///" + mlruns_dir.replace("\\", "/")

        mlflow.set_tracking_uri(tracking_uri)

        # Use experiment name from config (fallback to Default)
        self.experiment_name = cfg.get("mlflow", {}).get("experiment_name", "Default")
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None, params=None):
        return mlflow.start_run(run_name=run_name)

    def log_model_info(self, model_info: dict):
        for k, v in model_info.items():
            mlflow.log_param(k, v)

    def log_project(self, metrics, params=None):
        mlflow.log_metric(f"{metrics.project}_mae", metrics.mae)
        mlflow.log_metric(f"{metrics.project}_rmse", metrics.rmse)
        mlflow.log_metric(f"{metrics.project}_acc_at_1", metrics.accuracy_at_1)

    def log_summary(self, aggregate: dict, artefact_paths=None):
        for k, v in aggregate.items():
            mlflow.log_metric(k, v)

        if artefact_paths:
            for path in artefact_paths:
                if os.path.exists(path):
                    mlflow.log_artifact(path)

    def register_model(self, model_name="model"):
        # Optional: keep simple for local setup
        pass