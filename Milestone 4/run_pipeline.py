"""
run_pipeline.py
===============
Milestone 4 – Entry Point
=========================
Runs the full Milestone 4 training and evaluation pipeline:

  train_tracking_step  ──►  load_model_step  ──►  evaluate_step  ──►  report_step

Usage
-----
    cd "Milestone 4"
    python run_pipeline.py
"""

import sys
import os

# Ensure 'Milestone 4' root is on the Python path so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.zenml_pipeline import training_pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("  Milestone 4 – AI Story Point Estimation")
    print("  Model Development and Evaluation Pipeline")
    print("=" * 60)

    training_pipeline()

    print()
    print("=" * 60)
    print("  Pipeline complete!")
    print("  View results:  results/mae_per_project.csv")
    print("  MLflow UI:     mlflow ui --port 5000")
    print("                 http://localhost:5000")
    print("=" * 60)