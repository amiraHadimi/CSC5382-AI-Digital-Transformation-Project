"""
Milestone 3 - Step 2: Data Validation & Schema Management
Uses pandas-based validation checks that work on Python 3.11 / Windows.
"""

import os
import json
import pandas as pd
from zenml import step

SCHEMA_PATH = "schema/schema.json"
ANOMALIES_PATH = "tfdv_output/anomalies_report.json"
STATS_PATH = "tfdv_output/data_statistics.json"
TRAIN_RATIO = 0.8


def compute_statistics(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "count": int(df[col].count()),
            "nulls": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                    "median": round(float(df[col].median()), 4),
                    "p25": round(float(df[col].quantile(0.25)), 4),
                    "p75": round(float(df[col].quantile(0.75)), 4),
                }
            )
        else:
            col_stats.update(
                {
                    "unique": int(df[col].nunique()),
                    "top_value": str(df[col].mode()[0]) if not df[col].mode().empty else None,
                }
            )

        stats[col] = col_stats
    return stats


def infer_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = {
                "type": "numeric",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "max_null_pct": round(df[col].isnull().mean() * 100 + 10, 1),
            }
        else:
            schema[col] = {
                "type": "string",
                "unique_values": int(df[col].nunique()),
                "max_null_pct": round(df[col].isnull().mean() * 100 + 10, 1),
            }
    return schema


def check_anomalies(train_df: pd.DataFrame, eval_df: pd.DataFrame, schema: dict) -> dict:
    anomalies = {}

    for col, rules in schema.items():
        if col not in eval_df.columns:
            anomalies[col] = "Column missing in evaluation set"
            continue

        actual_null_pct = eval_df[col].isnull().mean() * 100
        if actual_null_pct > rules["max_null_pct"]:
            anomalies[col] = (
                f"Null% too high: {actual_null_pct:.1f}% "
                f"(allowed: {rules['max_null_pct']}%)"
            )
            continue

        if rules["type"] == "numeric":
            min_allowed = rules["min"]
            max_allowed = rules["max"] * 1.5

            if ((eval_df[col] < min_allowed) | (eval_df[col] > max_allowed)).any():
                unexpected_pct = (
                    ((eval_df[col] < min_allowed) | (eval_df[col] > max_allowed))
                    .mean()
                    * 100
                )
                anomalies[col] = (
                    f"Values out of range [{min_allowed}, {max_allowed}]: "
                    f"{unexpected_pct:.1f}% unexpected"
                )

    return anomalies


def fix_schema(schema: dict, anomalies: dict, eval_df: pd.DataFrame) -> dict:
    revised = dict(schema)

    for col, issue in anomalies.items():
        if col not in revised or col not in eval_df.columns:
            continue

        if revised[col]["type"] == "numeric":
            revised[col]["min"] = min(revised[col]["min"], float(eval_df[col].min()))
            revised[col]["max"] = max(revised[col]["max"], float(eval_df[col].max()))
        else:
            revised[col]["max_null_pct"] = min(revised[col]["max_null_pct"] + 20, 80)

    return revised


@step
def validate_data(df: pd.DataFrame) -> dict:
    os.makedirs("schema", exist_ok=True)
    os.makedirs("tfdv_output", exist_ok=True)

    train_df = df.sample(frac=TRAIN_RATIO, random_state=42).reset_index(drop=True)
    eval_df = df.drop(train_df.index).reset_index(drop=True)

    train_stats = compute_statistics(train_df)
    eval_stats = compute_statistics(eval_df)

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "eval": eval_stats}, f, indent=2)

    schema = infer_schema(train_df)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    anomalies = check_anomalies(train_df, eval_df, schema)
    with open(ANOMALIES_PATH, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=2)

    if anomalies:
        schema = fix_schema(schema, anomalies, eval_df)
        with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    return {
        "train_size": len(train_df),
        "eval_size": len(eval_df),
        "anomalies_detected": len(anomalies),
        "anomaly_features": list(anomalies.keys()),
        "schema_path": SCHEMA_PATH,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/raw/raw_data.csv")
    report = validate_data.entrypoint(df=df)
    print(report)