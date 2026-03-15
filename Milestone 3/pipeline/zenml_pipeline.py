"""
Milestone 3 – ZenML Pipeline: Data Acquisition, Validation & Preparation
=========================================================================
Wires all Milestone 3 steps into a single reproducible ZenML pipeline:

  ingest_data  ──►  validate_data  ──►  preprocess_and_engineer  ──►  setup_feature_store

Run with:
    python pipeline/zenml_pipeline.py

Or via ZenML CLI:
    zenml pipeline run pipeline/zenml_pipeline.py:data_pipeline
"""

import argparse
from zenml import pipeline
from zenml.logger import get_logger

from ingestion     import ingest_data
from validation    import validate_data
from transform     import preprocess_and_engineer
from feature_store import setup_feature_store

logger = get_logger(__name__)


@pipeline(name="milestone3_data_pipeline", enable_cache=True)
def data_pipeline(csv_path: str = "data/raw/raw_data.csv") -> None:
    """
    Full Milestone 3 data pipeline for Agile Story Point Estimation.

    Stages:
      1. Ingestion   – load raw CSV, normalize columns, snapshot to disk
      2. Validation  – TFDV stats, schema inference, anomaly detection & fix
      3. Transform   – text cleaning, feature engineering, Parquet export
      4. FeatureStore – Feast registration of engineered features

    Args:
        csv_path: Path to the raw input CSV file.
    """
    # Step 1: Ingest raw data
    raw_df = ingest_data(csv_path=csv_path)

    # Step 2: Validate data (TFDV)
    val_report = validate_data(df=raw_df)

    # Step 3: Preprocess and engineer features
    processed_df = preprocess_and_engineer(df=raw_df)

    # Step 4: Register features in Feast feature store
    store_info = setup_feature_store(df=processed_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Milestone 3 ZenML data pipeline."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/raw/raw_data.csv",
        help="Path to the raw input CSV file (default: data/raw/raw_data.csv)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Milestone 3 – Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Input CSV: {args.csv_path}")

    data_pipeline(csv_path=args.csv_path)

    logger.info("=" * 60)
    logger.info("  Pipeline complete. Check ZenML dashboard for artifacts.")
    logger.info("=" * 60)
