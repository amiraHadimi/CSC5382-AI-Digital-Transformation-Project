"""
Milestone 3 - Step 4: Feature Store Setup with Feast
Registers engineered features using a local Feast feature store.

Uses:
- File offline store
- SQLite online store
"""

import os
import subprocess
from pathlib import Path

import pandas as pd
from zenml import step


FEAST_REPO = Path("feast_repo/feature_repo")
PROCESSED_PATH = Path("data/processed/processed_data.parquet")


@step
def setup_feature_store(df: pd.DataFrame) -> dict:
    print("[FeatureStore] Initialising Feast feature store...")

    FEAST_REPO.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    if "event_timestamp" not in df.columns:
        df["event_timestamp"] = pd.Timestamp.utcnow()
        print("[FeatureStore] Added 'event_timestamp' column.")

    if "issue_id" not in df.columns:
        df["issue_id"] = range(len(df))
        print("[FeatureStore] Added 'issue_id' column as entity key.")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"[FeatureStore] Saved processed data to: {PROCESSED_PATH}")

    feature_store_yaml = """project: story_point_estimation
provider: local
registry: data/registry.db

offline_store:
  type: file

online_store:
  type: sqlite
  path: data/online_store.db

entity_key_serialization_version: 2
"""

    features_py = f'''
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


issue_entity = Entity(
    name="issue_id",
    join_keys=["issue_id"],
    value_type=ValueType.INT64,
    description="Unique issue identifier"
)


user_story_source = FileSource(
    path="{PROCESSED_PATH.as_posix()}",
    event_timestamp_column="event_timestamp",
)


user_story_features = FeatureView(
    name="user_story_features",
    entities=[issue_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="text_length", dtype=Int64),
        Field(name="word_count", dtype=Int64),
        Field(name="has_description", dtype=Int64),
        Field(name="title_word_count", dtype=Int64),
        Field(name="log_storypoints", dtype=Float32),
        Field(name="is_fibonacci", dtype=Int64),
    ],
    source=user_story_source,
)
'''

    feature_store_path = FEAST_REPO / "feature_store.yaml"
    features_path = FEAST_REPO / "features.py"

    feature_store_path.write_text(feature_store_yaml, encoding="utf-8")
    features_path.write_text(features_py, encoding="utf-8")

    print(f"[FeatureStore] Written: {feature_store_path}")
    print(f"[FeatureStore] Written: {features_path}")

    feast_apply_success = False

    try:
        subprocess.run(
            ["feast", "apply"],
            cwd=str(FEAST_REPO),
            check=True,
            capture_output=True,
            text=True,
        )
        feast_apply_success = True
        print("[FeatureStore] feast apply completed successfully.")

    except Exception as e:
        print(f"[FeatureStore] feast apply failed: {e}")

    result = {
        "feast_repo": str(FEAST_REPO),
        "feature_view": "user_story_features",
        "entity": "issue_id",
        "num_features": 6,
        "num_registered_rows": len(df),
        "feast_apply_success": feast_apply_success,
    }

    print(f"[FeatureStore] Registration complete: {result}")
    return result