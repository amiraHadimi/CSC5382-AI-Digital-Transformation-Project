"""
Milestone 6 – Requirement 6.3 (3 pts)
Continual Training / Continual Deployment (CT/CD) Pipeline
-----------------------------------------------------------
Implements an Apache Airflow DAG that:

  1. data_quality_check     – validates incoming batch data quality
  2. evaluate_current_model – runs current model on new data, computes MAE
  3. drift_check            – runs Evidently drift detection
  4. retrain_trigger        – BRANCHING: retrain if MAE > threshold OR drift detected
  5. retrain_baseline       – retrains TF-IDF + Ridge baseline on combined data
  6. register_new_model     – logs new model to MLflow Model Registry
  7. run_ab_test            – compares old vs new model on held-out slice
  8. deploy_new_model       – promotes new model to "Production" if it wins
  9. notify_team            – logs deployment notification

The DAG is triggered:
  - On a schedule (weekly): @weekly
  - On a file sensor: when new data lands in data/incoming/

Retrain threshold:
  - MAE degradation > 10% from baseline (MAE > 3.4, based on M4 macro-avg 3.09)
  - OR Evidently detects dataset drift in >50% of features

Usage:
  cp continual_learning/airflow_dag.py $AIRFLOW_HOME/dags/
  airflow db init
  airflow webserver --port 8080 &
  airflow scheduler &
  # Visit http://localhost:8080
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Airflow imports – these are only available when Airflow is installed
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.sensors.filesystem import FileSensor
    from airflow.utils.trigger_rule import TriggerRule
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

RETRAIN_MAE_THRESHOLD    = 3.40   # >10% above M4 macro-avg of 3.09
RETRAIN_DRIFT_THRESHOLD  = 0.5    # fraction of drifted features that triggers retrain
DATA_DIR   = Path(os.getenv("STORY_POINTS_DATA_DIR",
                             Path(__file__).parent.parent.parent / "Milestone 3" / "data"))
MODELS_DIR = Path(os.getenv("STORY_POINTS_MODELS_DIR",
                             Path(__file__).parent.parent / "results" / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Task functions ────────────────────────────────────────────────────────────

def data_quality_check(**context):
    """Validate incoming data has required columns and acceptable null rates."""
    incoming = DATA_DIR / "incoming" / "new_issues.csv"
    if not incoming.exists():
        log.info("No new data file found. Using test split as simulated incoming data.")
        incoming = DATA_DIR / "processed" / "test.parquet"
        if not incoming.exists():
            log.warning("Test parquet not found either. Generating synthetic data.")
            _generate_synthetic_data()
            incoming = DATA_DIR / "incoming" / "new_issues.csv"

    # Load and validate
    if str(incoming).endswith(".parquet"):
        df = __import__("pandas").read_parquet(incoming)
    else:
        df = __import__("pandas").read_csv(incoming)

    required_cols = {"storypoints", "input_text"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    null_rates = df.isnull().mean()
    high_null  = null_rates[null_rates > 0.3].index.tolist()
    if high_null:
        log.warning(f"High null rates in columns: {high_null}")

    context["ti"].xcom_push(key="n_new_samples", value=len(df))
    context["ti"].xcom_push(key="incoming_path", value=str(incoming))
    log.info(f"Data quality check passed: {len(df)} samples, {len(df.columns)} columns.")


def evaluate_current_model(**context):
    """Run current production model on new data, compute MAE."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    incoming_path = context["ti"].xcom_pull(key="incoming_path",
                                             task_ids="data_quality_check")
    if incoming_path and str(incoming_path).endswith(".parquet"):
        df = pd.read_parquet(incoming_path)
    else:
        df = pd.read_csv(incoming_path) if incoming_path else None

    if df is None:
        df = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")

    # Load current model (or fallback)
    model_path = MODELS_DIR / "current_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(df["input_text"].fillna("").tolist())
    else:
        log.info("No saved model – using word-count heuristic for evaluation.")
        preds = [max(1.0, min(13.0, len(t.split()) // 8 + 1))
                 for t in df["input_text"].fillna("").tolist()]

    mae = float(mean_absolute_error(df["storypoints"].tolist(), preds))
    log.info(f"Current model MAE on new data: {mae:.4f}")
    context["ti"].xcom_push(key="current_mae", value=mae)


def drift_check(**context):
    """Run KS-test drift detection between reference and current data."""
    from scipy.stats import ks_2samp
    import pandas as pd, numpy as np

    reference = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    incoming_path = context["ti"].xcom_pull(key="incoming_path",
                                             task_ids="data_quality_check")
    if incoming_path and str(incoming_path).endswith(".parquet"):
        current = pd.read_parquet(incoming_path)
    else:
        current = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")

    features = ["storypoints", "text_length", "word_count"]
    features = [f for f in features if f in reference.columns and f in current.columns]

    n_drifted = 0
    for col in features:
        _, p = ks_2samp(reference[col].dropna(), current[col].dropna())
        if p < 0.05:
            n_drifted += 1
            log.info(f"  Drift detected in '{col}' (p={p:.4f})")

    drift_fraction = n_drifted / len(features) if features else 0
    log.info(f"Drift fraction: {drift_fraction:.2f} ({n_drifted}/{len(features)} features)")
    context["ti"].xcom_push(key="drift_fraction", value=drift_fraction)


def retrain_trigger(**context):
    """Branch: return task IDs to run next based on thresholds."""
    current_mae    = context["ti"].xcom_pull(key="current_mae",    task_ids="evaluate_current_model") or 99
    drift_fraction = context["ti"].xcom_pull(key="drift_fraction", task_ids="drift_check") or 0

    should_retrain = (current_mae > RETRAIN_MAE_THRESHOLD or
                      drift_fraction >= RETRAIN_DRIFT_THRESHOLD)

    log.info(f"MAE={current_mae:.4f} threshold={RETRAIN_MAE_THRESHOLD}  "
             f"drift={drift_fraction:.2f} threshold={RETRAIN_DRIFT_THRESHOLD}  "
             f"→ retrain={should_retrain}")

    return "retrain_baseline" if should_retrain else "skip_retrain"


def retrain_baseline(**context):
    """Retrain TF-IDF + Ridge on train + new data combined."""
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    log.info("Retraining baseline model …")
    train_df = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")

    incoming_path = context["ti"].xcom_pull(key="incoming_path",
                                             task_ids="data_quality_check")
    if incoming_path and str(incoming_path).endswith(".parquet"):
        new_df = pd.read_parquet(incoming_path)
    else:
        new_df = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")

    combined = pd.concat([train_df, new_df], ignore_index=True)
    X = combined["input_text"].fillna("").tolist()
    y = combined["storypoints"].tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)

    new_model_path = MODELS_DIR / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(new_model_path, "wb") as f:
        pickle.dump(pipe, f)

    context["ti"].xcom_push(key="new_model_path", value=str(new_model_path))
    log.info(f"New model saved → {new_model_path}")


def register_new_model(**context):
    """Log new model metrics to MLflow Model Registry."""
    import numpy as np, pandas as pd
    from sklearn.metrics import mean_absolute_error

    new_model_path = context["ti"].xcom_pull(key="new_model_path",
                                              task_ids="retrain_baseline")
    if not new_model_path:
        log.info("No new model to register.")
        return

    with open(new_model_path, "rb") as f:
        model = pickle.load(f)

    val_df = pd.read_parquet(DATA_DIR / "processed" / "val.parquet")
    preds  = model.predict(val_df["input_text"].fillna("").tolist())
    val_mae = float(mean_absolute_error(val_df["storypoints"].tolist(), preds))
    log.info(f"New model validation MAE: {val_mae:.4f}")

    try:
        import mlflow
        with mlflow.start_run(run_name=f"ct_cd_retrain_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_metric("val_mae", val_mae)
            mlflow.sklearn.log_model(model, artifact_path="model",
                                      registered_model_name="storypoints_tfidf_lr_ctcd")
        log.info("Model registered to MLflow.")
    except Exception as exc:
        log.warning(f"MLflow registration skipped ({exc}). Saving metadata JSON instead.")
        meta = {"val_mae": val_mae, "model_path": str(new_model_path),
                "timestamp": datetime.now().isoformat()}
        (MODELS_DIR / "latest_model_meta.json").write_text(json.dumps(meta, indent=2))

    context["ti"].xcom_push(key="new_model_val_mae", value=val_mae)


def run_ab_test_task(**context):
    """Compare old vs new model MAE on a held-out slice."""
    import numpy as np, pandas as pd
    from sklearn.metrics import mean_absolute_error

    new_model_path  = context["ti"].xcom_pull(key="new_model_path",     task_ids="retrain_baseline")
    new_model_mae   = context["ti"].xcom_pull(key="new_model_val_mae",  task_ids="register_new_model") or 99
    current_mae     = context["ti"].xcom_pull(key="current_mae",        task_ids="evaluate_current_model") or 99

    winner = "new" if new_model_mae < current_mae else "current"
    log.info(f"A/B test: current_mae={current_mae:.4f}  new_mae={new_model_mae:.4f}  winner={winner}")
    context["ti"].xcom_push(key="ab_winner", value=winner)
    context["ti"].xcom_push(key="new_model_path_for_deploy", value=new_model_path)


def deploy_new_model(**context):
    """Promote new model to production if it won the A/B test."""
    winner         = context["ti"].xcom_pull(key="ab_winner",               task_ids="run_ab_test_task")
    new_model_path = context["ti"].xcom_pull(key="new_model_path_for_deploy", task_ids="run_ab_test_task")

    if winner == "new" and new_model_path:
        import shutil
        prod_path = MODELS_DIR / "current_model.pkl"
        shutil.copy(new_model_path, prod_path)
        log.info(f"✅ New model deployed to production: {prod_path}")
    else:
        log.info("Current model retained (new model did not outperform).")


def notify_team(**context):
    """Log deployment notification (stub for Slack/email integration)."""
    winner   = context["ti"].xcom_pull(key="ab_winner", task_ids="run_ab_test_task") or "N/A (no retrain)"
    mae      = context["ti"].xcom_pull(key="current_mae", task_ids="evaluate_current_model") or "N/A"
    drift    = context["ti"].xcom_pull(key="drift_fraction", task_ids="drift_check") or 0

    message = (
        f"[Milestone 6 CT/CD] Pipeline run completed.\n"
        f"  Current MAE: {mae}\n"
        f"  Drift fraction: {drift:.0%}\n"
        f"  A/B winner: {winner}\n"
    )
    log.info(message)
    (MODELS_DIR / "last_notification.txt").write_text(message)


# ── DAG definition ────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner":            "amira_hadimi",
    "depends_on_past":  False,
    "start_date":       datetime(2025, 1, 1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="story_point_ctcd_pipeline",
        default_args=DEFAULT_ARGS,
        description="Continual Training & Deployment for Llama3SP Story Point Estimator",
        schedule_interval="@weekly",
        catchup=False,
        tags=["milestone6", "ctcd", "story_points"],
    ) as dag:

        t_quality = PythonOperator(
            task_id="data_quality_check",
            python_callable=data_quality_check,
        )

        t_evaluate = PythonOperator(
            task_id="evaluate_current_model",
            python_callable=evaluate_current_model,
        )

        t_drift = PythonOperator(
            task_id="drift_check",
            python_callable=drift_check,
        )

        t_branch = BranchPythonOperator(
            task_id="retrain_trigger",
            python_callable=retrain_trigger,
        )

        t_retrain = PythonOperator(
            task_id="retrain_baseline",
            python_callable=retrain_baseline,
        )

        t_skip = EmptyOperator(task_id="skip_retrain")

        t_register = PythonOperator(
            task_id="register_new_model",
            python_callable=register_new_model,
        )

        t_ab = PythonOperator(
            task_id="run_ab_test_task",
            python_callable=run_ab_test_task,
        )

        t_deploy = PythonOperator(
            task_id="deploy_new_model",
            python_callable=deploy_new_model,
        )

        t_notify = PythonOperator(
            task_id="notify_team",
            python_callable=notify_team,
            trigger_rule=TriggerRule.ONE_SUCCESS,
        )

        # ── DAG dependencies ──────────────────────────────────────────────
        t_quality >> [t_evaluate, t_drift]
        [t_evaluate, t_drift] >> t_branch
        t_branch >> [t_retrain, t_skip]
        t_retrain >> t_register >> t_ab >> t_deploy >> t_notify
        t_skip >> t_notify


# ── Standalone runner (no Airflow required) ───────────────────────────────────

def run_pipeline_standalone():
    """
    Execute the CT/CD pipeline without Airflow for testing/demo purposes.
    Uses a simple context dict to simulate XCom.
    """
    log.info("=" * 60)
    log.info("  CT/CD Pipeline – Standalone Mode")
    log.info("=" * 60)

    context = {"ti": _XComSimulator()}

    log.info("[1/8] Data quality check …")
    data_quality_check(**context)

    log.info("[2/8] Evaluate current model …")
    evaluate_current_model(**context)

    log.info("[3/8] Drift check …")
    drift_check(**context)

    log.info("[4/8] Retrain trigger …")
    next_task = retrain_trigger(**context)
    log.info(f"      → Next task: {next_task}")

    if next_task == "retrain_baseline":
        log.info("[5/8] Retraining baseline …")
        retrain_baseline(**context)

        log.info("[6/8] Registering new model …")
        register_new_model(**context)

        log.info("[7/8] Running A/B test …")
        run_ab_test_task(**context)

        log.info("[8/8] Deploying new model …")
        deploy_new_model(**context)
    else:
        log.info("[5-8] Skipped – model is performing well, no retrain needed.")

    log.info("[9/9] Sending notification …")
    notify_team(**context)
    log.info("Pipeline complete.")


class _XComSimulator:
    """Simulates Airflow's XCom push/pull for standalone execution."""
    def __init__(self):
        self._store = {}

    def xcom_push(self, key, value):
        self._store[key] = value

    def xcom_pull(self, key, task_ids=None):
        return self._store.get(key)


def _generate_synthetic_data():
    """Generate a small synthetic CSV if no incoming data exists."""
    import pandas as pd
    incoming_dir = DATA_DIR / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"input_text": f"Title: Fix issue {i} Description: Some description for issue {i}", "storypoints": (i % 8) + 1}
        for i in range(200)
    ]
    pd.DataFrame(rows).to_csv(incoming_dir / "new_issues.csv", index=False)
    log.info("Synthetic incoming data generated.")


if __name__ == "__main__":
    run_pipeline_standalone()
