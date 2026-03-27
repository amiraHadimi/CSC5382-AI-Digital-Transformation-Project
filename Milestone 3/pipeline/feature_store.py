"""
Milestone 3 - Step 4: Feature Store (Feast)
Registers engineered features in a local Feast feature store.
Features are retrieved at training time and during inference,
ensuring consistent preprocessing across the ML lifecycle.

Feast is used here in offline mode (FileSource) for portability.
For production, swap FileSource for a BigQuery / Redis source.
"""

import os
import subprocess

import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

FEAST_REPO_PATH = "feast_repo/feature_repo"
PROCESSED_DATA_PATH = os.path.abspath("data/processed/processed_data.parquet")
FEATURE_STORE_YAML = os.path.join(FEAST_REPO_PATH, "feature_store.yaml")
FEATURES_DEF_PATH = os.path.join(FEAST_REPO_PATH, "features.py")


FEATURE_STORE_YAML_CONTENT = """\
project: agile_story_points
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
"""


FEATURES_PY_CONTENT = f'''\
"""
Feast Feature View definitions for the Agile Story Point Estimation project.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32

issue_entity = Entity(
    name="issue_id",
    description="Unique identifier for a JIRA issue / user story.",
)

user_story_source = FileSource(
    path=r"{PROCESSED_DATA_PATH}",
    timestamp_field="event_timestamp",
    description="Processed Agile user story features from Milestone 3 pipeline.",
)

user_story_features = FeatureView(
    name="user_story_features",
    entities=[issue_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="text_length", dtype=Int64, description="Character count of input_text"),
        Field(name="word_count", dtype=Int64, description="Word count of input_text"),
        Field(name="has_description", dtype=Int64, description="1 if description is non-empty"),
        Field(name="title_word_count", dtype=Int64, description="Word count of cleaned_title"),
        Field(name="log_storypoints", dtype=Float32, description="log1p of story point target"),
        Field(name="is_fibonacci", dtype=Int64, description="1 if storypoints is Fibonacci"),
    ],
    source=user_story_source,
    description="Engineered features for story point estimation.",
)
'''


def _write_feast_files() -> None:
    """Write Feast config files to the feature repo directory."""
    os.makedirs(FEAST_REPO_PATH, exist_ok=True)

    with open(FEATURE_STORE_YAML, "w", encoding="utf-8", errors="ignore") as f:
        f.write(FEATURE_STORE_YAML_CONTENT)
    logger.info(f"[FeatureStore] Written: {FEATURE_STORE_YAML}")

    with open(FEATURES_DEF_PATH, "w", encoding="utf-8", errors="ignore") as f:
        f.write(FEATURES_PY_CONTENT)
    logger.info(f"[FeatureStore] Written: {FEATURES_DEF_PATH}")


def _add_event_timestamp(parquet_path: str) -> None:
    """
    Feast requires an event_timestamp column for point-in-time joins.
    Add it to the processed Parquet file if not present.
    """
    df = pd.read_parquet(parquet_path)
    updated = False

    if "event_timestamp" not in df.columns:
        df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
        logger.info("[FeatureStore] Added 'event_timestamp' column to processed data.")
        updated = True

    if "issue_id" not in df.columns:
        df["issue_id"] = range(len(df))
        logger.info("[FeatureStore] Added 'issue_id' column as entity key.")
        updated = True

    if updated:
        df.to_parquet(parquet_path, index=False)


def _run_feast_apply() -> bool:
    """Run `feast apply` to register the feature store."""
    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=FEAST_REPO_PATH,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info("[FeatureStore] `feast apply` succeeded.")
            logger.info(result.stdout)
            return True

        logger.error(f"[FeatureStore] `feast apply` failed:\n{result.stderr}")
        return False

    except FileNotFoundError:
        logger.warning(
            "[FeatureStore] `feast` CLI not found. "
            "Install with: pip install feast[local]. "
            "Feature view definitions are written but not applied."
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("[FeatureStore] `feast apply` timed out.")
        return False


@step
def setup_feature_store(df: pd.DataFrame) -> dict:
    """
    ZenML step: Set up the Feast feature store and register features.

    Args:
        df: Processed DataFrame (output of preprocess_and_engineer step).

    Returns:
        dict: Feature store metadata and registration status.
    """
    logger.info("[FeatureStore] Initialising Feast feature store...")

    _write_feast_files()
    _add_event_timestamp(PROCESSED_DATA_PATH)
    applied = _run_feast_apply()

    feature_info = {
        "feast_repo": FEAST_REPO_PATH,
        "feature_view": "user_story_features",
        "entity": "issue_id",
        "num_features": 6,
        "num_registered_rows": len(df),
        "feast_apply_success": applied,
    }

    logger.info(f"[FeatureStore] Registration complete: {feature_info}")
    return feature_info


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    info = setup_feature_store.entrypoint(df=df)
    print(info)