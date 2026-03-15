"""
Milestone 3 – Step 1: Data Ingestion
=====================================
Ingests raw CSV data from the Agile Story Point dataset,
stores a versioned snapshot, and returns a clean DataFrame.
Wraps TFX-style ExampleGen logic inside a ZenML step.
"""

import os
import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

RAW_OUTPUT_PATH = "data/raw/raw_data.csv"

REQUIRED_COLUMNS = {"title", "description", "storypoints"}
COLUMN_ALIASES = {
    # Map alternative column names to canonical names
    "point":        "storypoints",
    "story_points": "storypoints",
    "storyPoint":   "storypoints",
    "story_point":  "storypoints",
    "concat":       "description",
    "body":         "description",
    "summary":      "title",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alternative column names to canonical schema."""
    df = df.rename(columns=COLUMN_ALIASES)
    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure required columns are present after normalization."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"[Ingestion] Missing required columns after normalization: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


@step
def ingest_data(csv_path: str) -> pd.DataFrame:
    """
    Ingest raw Agile user story data from a CSV file.

    Steps:
    1. Load CSV from disk
    2. Normalize column names to canonical schema
    3. Validate required columns exist
    4. Drop rows with null target (storypoints)
    5. Store raw snapshot for DVC tracking

    Args:
        csv_path: Path to the raw CSV input file.

    Returns:
        pd.DataFrame: Validated raw DataFrame with canonical column names.
    """
    logger.info(f"[Ingestion] Loading data from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Ingestion] CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"[Ingestion] Loaded {len(df)} rows, {len(df.columns)} columns.")
    logger.info(f"[Ingestion] Original columns: {list(df.columns)}")

    # Normalize column names
    df = _normalize_columns(df)

    # Validate required columns
    _validate_required_columns(df)

    # Drop rows where the target label is missing
    before = len(df)
    df = df.dropna(subset=["storypoints"])
    after = len(df)
    if before != after:
        logger.warning(
            f"[Ingestion] Dropped {before - after} rows with null storypoints."
        )

    # Filter to positive story point values only (data quality)
    df = df[df["storypoints"] > 0]
    logger.info(f"[Ingestion] Final dataset size after filtering: {len(df)} rows.")

    # Store raw snapshot for DVC tracking
    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    df.to_csv(RAW_OUTPUT_PATH, index=False)
    logger.info(f"[Ingestion] Raw snapshot saved to: {RAW_OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    # Standalone test run
    sample_df = ingest_data.entrypoint(csv_path="data/raw/raw_data.csv")
    print(sample_df.head())
    print(f"Shape: {sample_df.shape}")
