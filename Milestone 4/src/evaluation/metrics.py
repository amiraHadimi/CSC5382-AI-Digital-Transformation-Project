"""
evaluation/metrics.py
=====================
Evaluation metrics for story point estimation.

Implements the three metrics defined in Milestone 1:
  - MAE  (Mean Absolute Error)         — primary metric
  - RMSE (Root Mean Squared Error)     — penalises large errors
  - Accuracy@±1                        — within-1-point tolerance
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EvalMetrics:
    """Container for all evaluation metrics for a single project."""
    project: str
    test_size: int
    mae: float
    rmse: float
    accuracy_at_1: float   # proportion of predictions within ±1 story point

    def to_dict(self) -> dict:
        return {
            "project":       self.project,
            "test_size":     self.test_size,
            "mae":           round(self.mae, 4),
            "rmse":          round(self.rmse, 4),
            "accuracy_at_1": round(self.accuracy_at_1, 4),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    project: str,
    tolerance: float = 1.0,
) -> EvalMetrics:
    """
    Compute MAE, RMSE, and Accuracy@±tolerance for a project.

    Args:
        y_true:    Ground-truth story points.
        y_pred:    Predicted story points.
        project:   Project name (for labelling).
        tolerance: Story point tolerance for accuracy metric (default 1.0).

    Returns:
        EvalMetrics: Dataclass with all computed metrics.
    """
    abs_errors = np.abs(y_true - y_pred)
    mae  = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    accuracy_at_1 = float(np.mean(abs_errors <= tolerance))

    return EvalMetrics(
        project=project,
        test_size=len(y_true),
        mae=mae,
        rmse=rmse,
        accuracy_at_1=accuracy_at_1,
    )


def aggregate_metrics(metrics_list: list[EvalMetrics]) -> dict:
    """
    Compute macro-averaged metrics across all projects.

    Args:
        metrics_list: List of per-project EvalMetrics.

    Returns:
        dict: Macro-average of MAE, RMSE, and Accuracy@±1.
    """
    maes  = [m.mae for m in metrics_list]
    rmses = [m.rmse for m in metrics_list]
    accs  = [m.accuracy_at_1 for m in metrics_list]

    return {
        "mean_mae":           round(float(np.mean(maes)), 4),
        "std_mae":            round(float(np.std(maes)), 4),
        "mean_rmse":          round(float(np.mean(rmses)), 4),
        "mean_accuracy_at_1": round(float(np.mean(accs)), 4),
        "num_projects":       len(metrics_list),
    }
