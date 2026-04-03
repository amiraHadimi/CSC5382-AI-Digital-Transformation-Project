"""
pipeline/zenml_pipeline.py
==========================
Milestone 4 ZenML pipeline:
train_tracking_step -> load_model_step -> evaluate_step -> report_step
"""

from zenml import pipeline
from zenml.logger import get_logger
from src.pipeline.zenml_steps import (
    train_tracking_step,
    load_model_step,
    evaluate_step,
    report_step,
)

logger = get_logger(__name__)


@pipeline(name="milestone4_training_pipeline", enable_cache=False)
def training_pipeline():
    """
    Milestone 4 full training and evaluation pipeline.

    Steps:
      1. train_tracking_step  -> runs the classical baseline training with MLflow
      2. load_model_step      -> resolves Llama3SP base model metadata
      3. evaluate_step        -> evaluates all projects with MLflow + CodeCarbon
      4. report_step          -> aggregates final metrics
    """
    train_info = train_tracking_step()
    model_info = load_model_step()
    metrics_df = evaluate_step(model_info=model_info, train_info=train_info)
    summary = report_step(metrics_df=metrics_df)
    return summary


if __name__ == "__main__":
    training_pipeline()