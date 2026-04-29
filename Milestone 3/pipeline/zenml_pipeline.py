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

from pipeline.ingestion import ingest_data
from pipeline.validation import validate_data
from pipeline.transform import preprocess_and_engineer
from pipeline.feature_store import setup_feature_store

logger = get_logger(__name__)


@pipeline(name="milestone3_data_pipeline_v2", enable_cache=False)
def data_pipeline(csv_path: str = "data/raw/raw_data.csv") -> None:
    """
    Full Milestone 3 data pipeline for Agile Story Point Estimation.

    Stages:
      1. Ingestion     – load raw CSV, normalize columns, snapshot to disk
      2. Validation    – compute stats, infer schema, detect anomalies
      3. Transform     – clean text, engineer features, export Parquet files
      4. FeatureStore  – register engineered features in Feast
    """
    # Step 1: Ingest raw data
    raw_df = ingest_data(csv_path=csv_path)

    # Step 2: Validate data
    validated_df = validate_data(df=raw_df)

    # Step 3: Preprocess only after validation succeeds
    processed_df = preprocess_and_engineer(df=validated_df)

    # Step 4: Register features in Feast
    setup_feature_store(df=processed_df)


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
