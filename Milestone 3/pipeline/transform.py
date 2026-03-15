"""
Milestone 3 – Step 3: Preprocessing & Feature Engineering
===========================================================
Applies text preprocessing and feature engineering to the raw
Agile user story dataset in preparation for Llama3SP fine-tuning.

Engineered features:
  - cleaned_text      : lowercased, de-noised title + description
  - input_text        : final combined field fed to the LLM tokenizer
  - text_length       : character count of cleaned_text
  - word_count        : word count of cleaned_text
  - has_description   : binary flag (1 if description is non-empty)
  - log_storypoints   : log1p transform of the target (reduces skew)

The processed DataFrame is saved as Parquet for downstream use
(feature store, model training).
"""

import os
import re
import math
import pandas as pd
import numpy as np
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

PROCESSED_OUTPUT_PATH = "data/processed/processed_data.parquet"

# Story point values commonly used in Fibonacci planning poker
FIBONACCI_POINTS = {0.5, 1, 2, 3, 5, 8, 13, 20, 40, 100}


# ── Text cleaning helpers ────────────────────────────────────────────────────

def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def _remove_jira_markup(text: str) -> str:
    """Remove JIRA-specific markup (e.g., {code}, {noformat}, [~user])."""
    text = re.sub(r"\{[^}]+\}", " ", text)   # {code}, {noformat}, etc.
    text = re.sub(r"\[~[^\]]+\]", " ", text) # [~username] mentions
    text = re.sub(r"![\w.]+!", " ", text)     # !image.png! attachments
    return text

def _remove_special_chars(text: str) -> str:
    """Keep alphanumeric, spaces, and basic punctuation."""
    return re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\_]", " ", text)

def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str) -> str:
    """Full text cleaning pipeline for Agile user story text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_jira_markup(text)
    text = _remove_special_chars(text)
    text = _collapse_whitespace(text)
    return text


# ── Feature engineering ──────────────────────────────────────────────────────

def build_input_text(title: str, description: str) -> str:
    """
    Combine title and description into a single input string
    for the Llama tokenizer. Uses a separator consistent with
    the Llama3SP paper's prompt format.
    """
    title_clean = clean_text(title)
    desc_clean  = clean_text(description)

    if title_clean and desc_clean:
        return f"Title: {title_clean} Description: {desc_clean}"
    elif title_clean:
        return f"Title: {title_clean}"
    elif desc_clean:
        return f"Description: {desc_clean}"
    return ""


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""

    # Fill nulls before processing
    df["title"]       = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    # 1. Cleaned individual fields
    df["cleaned_title"]       = df["title"].apply(clean_text)
    df["cleaned_description"] = df["description"].apply(clean_text)

    # 2. Combined input text for LLM (primary feature)
    df["input_text"] = df.apply(
        lambda row: build_input_text(row["title"], row["description"]),
        axis=1
    )

    # 3. Structural / meta features
    df["text_length"]     = df["input_text"].apply(len)
    df["word_count"]      = df["input_text"].apply(lambda x: len(x.split()))
    df["has_description"] = (df["cleaned_description"].str.len() > 0).astype(int)
    df["title_word_count"] = df["cleaned_title"].apply(
        lambda x: len(x.split()) if x else 0
    )

    # 4. Target transformation: log1p reduces right skew of story points
    df["log_storypoints"] = df["storypoints"].apply(lambda x: math.log1p(x))

    # 5. Fibonacci alignment flag (is the label a standard planning-poker value?)
    df["is_fibonacci"] = df["storypoints"].apply(
        lambda x: 1 if x in FIBONACCI_POINTS else 0
    )

    # 6. Drop rows with empty input_text (unusable samples)
    before = len(df)
    df = df[df["input_text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.warning(
            f"[Transform] Dropped {before - after} rows with empty input_text."
        )

    return df


# ── ZenML step ───────────────────────────────────────────────────────────────

@step
def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step: Preprocess text and engineer features for the
    Agile story point estimation dataset.

    Args:
        df: Raw ingested DataFrame (output of ingest_data step).

    Returns:
        pd.DataFrame: Processed DataFrame with engineered features.
    """
    logger.info(f"[Transform] Starting preprocessing. Input shape: {df.shape}")

    df = engineer_features(df)

    logger.info(f"[Transform] Preprocessing complete. Output shape: {df.shape}")
    logger.info(
        f"[Transform] Engineered columns: {[c for c in df.columns if c not in ['title','description','storypoints']]}"
    )
    logger.info(
        f"[Transform] Story points stats:\n"
        f"  Mean:   {df['storypoints'].mean():.2f}\n"
        f"  Median: {df['storypoints'].median():.2f}\n"
        f"  Std:    {df['storypoints'].std():.2f}\n"
        f"  Min:    {df['storypoints'].min()}\n"
        f"  Max:    {df['storypoints'].max()}"
    )

    # Save to Parquet for feature store and training pipeline
    os.makedirs(os.path.dirname(PROCESSED_OUTPUT_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_OUTPUT_PATH, index=False)
    logger.info(f"[Transform] Processed data saved to: {PROCESSED_OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    df_raw = pd.read_csv("data/raw/raw_data.csv")
    df_processed = preprocess_and_engineer.entrypoint(df=df_raw)
    print(df_processed[["input_text", "storypoints", "log_storypoints",
                          "text_length", "word_count", "is_fibonacci"]].head(10))
