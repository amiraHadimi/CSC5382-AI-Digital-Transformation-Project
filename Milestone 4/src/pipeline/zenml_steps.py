"""
pipeline/zenml_steps.py
=======================
ZenML steps for the Milestone 4 training and evaluation pipeline.

Four steps are defined:
  1. train_tracking_step – runs the baseline training script with MLflow tracking
  2. load_model_step     – loads base model metadata (no adapter yet)
  3. evaluate_step       – per-project inference + metrics, tracked with MLflow + CodeCarbon
  4. report_step         – aggregates results, saves CSV/JSON artefacts

These steps extend the Milestone 3 data pipeline into a full Milestone 4 MLOps workflow.
"""

import os
import gc
import glob
import json
import time
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from zenml import step
from zenml.logger import get_logger

from src.utils.config import load_config, get_hf_token
from src.pipeline.model_loader import (
    resolve_base_model_id,
    load_tokenizer,
    load_base_model,
    build_peft_model,
    load_adapter_for_project,
)
from src.pipeline.inference import run_inference_on_dataframe
from src.evaluation.metrics import compute_metrics, aggregate_metrics, EvalMetrics
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.carbon_tracker import CarbonTracker

logger = get_logger(__name__)

# Suppress threading warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Step 1: Train baseline model with MLflow ─────────────────────────────────

@step
def train_tracking_step() -> dict:
    """
    ZenML Step 1: Run the baseline training script inside the ZenML pipeline.

    This step executes src/models/train.py using the current Python interpreter
    so that:
      - training is integrated into the MLOps platform
      - MLflow logs are created consistently in Milestone 4/mlruns

    Returns:
        dict: lightweight summary of the training execution
    """
    m4_root = Path(__file__).resolve().parents[2]
    train_script = m4_root / "src" / "models" / "train.py"
    mlruns_dir = m4_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)

    tracking_uri = f"file:///{str(mlruns_dir.resolve()).replace(os.sep, '/')}"
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri

    logger.info("[Step 1] Running baseline training script with MLflow tracking...")
    logger.info(f"[Step 1] Training script: {train_script}")
    logger.info(f"[Step 1] Tracking URI: {tracking_uri}")

    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(m4_root),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.info("[Step 1] Training stdout:\n" + result.stdout)

    if result.returncode != 0:
        if result.stderr:
            logger.error("[Step 1] Training stderr:\n" + result.stderr)
        raise RuntimeError("train_tracking_step failed while running src/models/train.py")

    mae_value = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("MAE:"):
            try:
                mae_value = float(line.split(":", 1)[1].strip())
            except ValueError:
                mae_value = None

    logger.info(f"[Step 1] Baseline training completed. Parsed MAE: {mae_value}")

    return {
        "status": "success",
        "train_script": str(train_script),
        "tracking_uri": tracking_uri,
        "baseline_mae": mae_value,
    }


# ── Step 2: Load model metadata ──────────────────────────────────────────────

@step
def load_model_step() -> dict:
    """
    ZenML Step 2: Resolve the Llama3SP base model metadata.

    Returns a lightweight metadata dict (actual model objects are passed
    to the evaluation step by re-loading — ZenML cannot serialise PyTorch models
    across steps without a custom materializer).

    Returns:
        dict: Model metadata (base_model_id, hf_author, etc.)
    """
    cfg = load_config()
    hf_token = get_hf_token()
    model_cfg = cfg["model"]

    logger.info("[Step 2] Resolving base model ID from HF adapter config...")
    base_model_id = resolve_base_model_id(model_cfg["hf_author"], hf_token)
    logger.info(f"[Step 2] Base model: {base_model_id}")

    return {
        "base_model_id": base_model_id,
        "hf_author": model_cfg["hf_author"],
        "num_labels": model_cfg["num_labels"],
        "problem_type": model_cfg["problem_type"],
    }


# ── Step 3: Evaluate all projects ────────────────────────────────────────────

@step
def evaluate_step(model_info: dict, train_info: dict) -> pd.DataFrame:
    """
    ZenML Step 3: Run per-project inference and evaluation.

    For each Agile project:
      - Loads and activates the project-specific LoRA adapter.
      - Runs batched inference on the test split.
      - Computes MAE, RMSE, and Accuracy@±1.
      - Logs metrics as a nested MLflow child run.
      - Tracks CO₂ emissions with CodeCarbon.

    Args:
        model_info: Metadata dict from load_model_step.
        train_info: Metadata dict from train_tracking_step.

    Returns:
        pd.DataFrame: Per-project metrics table.
    """
    import torch

    cfg = load_config()
    hf_token = get_hf_token()
    inf_cfg = cfg["inference"]
    data_cfg = cfg["data"]
    m4_root = Path(__file__).resolve().parents[2]

    torch.set_num_threads(1)

    logger.info(f"[Step 3] Training step status: {train_info.get('status')}")
    logger.info(f"[Step 3] Baseline training MAE from Step 1: {train_info.get('baseline_mae')}")

    # ── Locate per-project CSV files ─────────────────────────────────────────
    csv_glob = str(m4_root / data_cfg["raw_csv_glob"])
    csv_files = sorted(glob.glob(csv_glob))
    if not csv_files:
        raise RuntimeError(f"No CSV files found at: {csv_glob}")
    logger.info(f"[Step 3] Found {len(csv_files)} project CSV files.")

    # ── Load model objects (re-load here; not serialisable across ZenML steps) ─
    logger.info("[Step 3] Loading tokenizer...")
    tokenizer = load_tokenizer(model_info["hf_author"], hf_token)

    logger.info("[Step 3] Loading base model (CPU)...")
    base_model = load_base_model(
        model_info["base_model_id"], tokenizer.pad_token_id, hf_token
    )

    first_project = os.path.splitext(os.path.basename(csv_files[0]))[0].lower()
    logger.info(f"[Step 3] Building PEFT model with first adapter: {first_project}")
    peft_model = build_peft_model(base_model, model_info["hf_author"], first_project, hf_token)

    # ── MLflow + CodeCarbon setup ─────────────────────────────────────────────
    tracker = MLflowTracker(cfg)
    carbon = CarbonTracker(cfg)
    run_name = f"{cfg['mlflow']['run_name_prefix']}_{time.strftime('%Y%m%d_%H%M%S')}"

    inference_params = {
        "base_model_id": model_info["base_model_id"],
        "hf_author": model_info["hf_author"],
        "max_len": inf_cfg["max_len"],
        "batch_size": inf_cfg["batch_size"],
        "use_description": inf_cfg["use_description"],
        "limit_test_rows": inf_cfg["limit_test_rows"],
        "baseline_train_mae": train_info.get("baseline_mae"),
    }

    results_dir = str(m4_root / data_cfg["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    all_metrics: list[EvalMetrics] = []

    with tracker.start_run(run_name=run_name, params=inference_params):
        tracker.log_model_info(model_info)

        # ── Start CO₂ tracking around the full evaluation loop ────────────────
        carbon.start()

        for csv_path in csv_files:
            project = os.path.splitext(os.path.basename(csv_path))[0].lower()
            logger.info(f"[Step 3] Evaluating project: {project}")

            df = pd.read_csv(csv_path)
            test_df = df[df["split_mark"].astype(str).str.lower().str.contains("test")].copy()
            if test_df.empty:
                logger.warning(f"[Step 3] No test rows for {project} — skipping.")
                continue

            limit = inf_cfg.get("limit_test_rows")
            if limit:
                test_df = test_df.head(limit).copy()

            # Activate this project's LoRA adapter
            load_adapter_for_project(peft_model, model_info["hf_author"], project, hf_token)
            peft_model.eval()

            # Run inference
            y_pred = run_inference_on_dataframe(
                test_df,
                tokenizer,
                peft_model,
                batch_size=inf_cfg["batch_size"],
                max_len=inf_cfg["max_len"],
                use_description=inf_cfg["use_description"],
            )

            # Resolve ground-truth column name
            sp_col = next(
                (
                    c
                    for c in ["storypoint", "storypoints", "story_points", "point"]
                    if c in test_df.columns
                ),
                None,
            )
            if sp_col is None:
                logger.warning(f"[Step 3] No story point column found for {project} — skipping.")
                continue

            y_true = test_df[sp_col].astype(float).to_numpy()
            metrics = compute_metrics(y_true, y_pred, project)
            all_metrics.append(metrics)

            # Log per-project nested MLflow run
            tracker.log_project(metrics, params={"project": project})

            # Save per-project predictions CSV
            pred_df = pd.DataFrame(
                {
                    "project": project,
                    "title": test_df["title"].fillna("").values,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "abs_error": np.abs(y_true - y_pred),
                }
            )
            pred_df.to_csv(f"{results_dir}/predictions_{project}.csv", index=False)

            logger.info(
                f"[Step 3]   {project:20s} | n={metrics.test_size:4d} | "
                f"MAE={metrics.mae:.4f} | RMSE={metrics.rmse:.4f} | "
                f"Acc@±1={metrics.accuracy_at_1:.3f}"
            )
            gc.collect()

        # ── Stop CO₂ tracker ─────────────────────────────────────────────────
        emissions = carbon.stop()
        carbon.log_to_mlflow(emissions)

        # ── Aggregate and log summary ─────────────────────────────────────────
        aggregate = aggregate_metrics(all_metrics)
        logger.info(f"[Step 3] Aggregate metrics: {aggregate}")

        mae_csv = f"{results_dir}/mae_per_project.csv"
        pd.DataFrame([m.to_dict() for m in all_metrics]).to_csv(mae_csv, index=False)

        summary = {
            **aggregate,
            **inference_params,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_json = f"{results_dir}/summary.json"
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        tracker.log_summary(aggregate, artefact_paths=[mae_csv, summary_json])
        tracker.register_model(model_name="Llama3SP-StoryPoints")

    return pd.DataFrame([m.to_dict() for m in all_metrics])


# ── Step 4: Report ───────────────────────────────────────────────────────────

@step
def report_step(metrics_df: pd.DataFrame) -> dict:
    """
    ZenML Step 4: Summarise evaluation results and emit final report.

    Prints a formatted leaderboard and returns the aggregate summary dict.

    Args:
        metrics_df: Per-project metrics DataFrame from evaluate_step.

    Returns:
        dict: Aggregate summary metrics.
    """
    logger.info("[Step 4] Generating final evaluation report...")

    sorted_df = metrics_df.sort_values("mae")

    logger.info("\n" + "=" * 65)
    logger.info("  Milestone 4 – Evaluation Leaderboard (sorted by MAE ↑)")
    logger.info("=" * 65)
    logger.info(f"  {'Project':<22} {'n':>5}  {'MAE':>7}  {'RMSE':>7}  {'Acc@±1':>7}")
    logger.info("-" * 65)
    for _, row in sorted_df.iterrows():
        logger.info(
            f"  {row['project']:<22} {int(row['test_size']):>5}  "
            f"{row['mae']:>7.4f}  {row['rmse']:>7.4f}  {row['accuracy_at_1']:>7.3f}"
        )
    logger.info("=" * 65)

    aggregate = {
        "mean_mae": round(float(metrics_df["mae"].mean()), 4),
        "std_mae": round(float(metrics_df["mae"].std()), 4),
        "mean_rmse": round(float(metrics_df["rmse"].mean()), 4),
        "mean_accuracy_at_1": round(float(metrics_df["accuracy_at_1"].mean()), 4),
        "num_projects": len(metrics_df),
    }

    logger.info(
        f"\n  Macro-average — MAE: {aggregate['mean_mae']:.4f} ± {aggregate['std_mae']:.4f} | "
        f"RMSE: {aggregate['mean_rmse']:.4f} | Acc@±1: {aggregate['mean_accuracy_at_1']:.3f}"
    )
    logger.info("=" * 65)

    return aggregate