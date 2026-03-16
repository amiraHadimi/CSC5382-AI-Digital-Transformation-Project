# Milestone 3 – Data Acquisition, Validation & Preparation

> **Project:** AI-Based Story Point Estimation for Agile Software Development  
> **Course:** CSC5382 – AI for Digital Transformation  
> **Milestone due:** March 22, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [1 · Data Ingestion & Raw Storage](#1--data-ingestion--raw-storage)
4. [2 · Schema Definition & Data Validation](#2--schema-definition--data-validation)
5. [3 · Preprocessing & Feature Engineering](#3--preprocessing--feature-engineering)
6. [4 · Feature Store (Feast)](#4--feature-store-feast)
7. [5 · Data Versioning (DVC)](#5--data-versioning-dvc)
8. [6 · ML Pipeline Integration (ZenML)](#6--ml-pipeline-integration-zenml)
9. [Grading Checklist](#grading-checklist)
10. [References](#references)

---

## Overview

Milestone 3 builds the complete data pipeline for the Agile story point estimation system introduced in Milestones 1 and 2. The goal is to ingest, validate, preprocess, and version the raw Agile user story dataset (23,313 issues from 16 open-source JIRA projects), and register engineered features in a Feast feature store — all orchestrated through a single reproducible pipeline script and version-controlled via DVC.

The pipeline connects directly to the Llama3SP model used in Milestone 2 (`meta-llama/Llama-3.2-1B` + per-project LoRA adapters): the `input_text` feature produced here is the exact field consumed by the Llama tokenizer in `eval_mae_per_project.py`.

**To reproduce the full pipeline:**
```bash
python run_pipeline.py
```

---

## Repository Structure

```
Milestone 3/
├── data/
│   ├── raw/
│   │   ├── raw_data.csv.dvc           ← DVC pointer (23,313 rows, all 16 projects merged)
│   │   └── .gitignore
│   └── processed/
│       ├── train.parquet              ← 13,981 rows (60%)
│       ├── val.parquet                ← 4,661 rows  (20%)
│       ├── test.parquet               ← 4,671 rows  (20%)
│       ├── processed_data.parquet.dvc ← DVC pointer (full processed dataset)
│       └── .gitignore
├── schema/
│   └── schema.json                    ← inferred + revised schema (6 features)
├── tfdv_output/
│   ├── data_statistics.json           ← statistics for train, val, test splits
│   ├── split_info.json                ← split sizes and source
│   ├── anomalies_report.json          ← anomaly detection report (0 anomalies)
│   └── feature_distributions.png     ← feature distribution plots
├── feast_repo/feature_repo/
│   ├── feature_store.yaml             ← Feast project config
│   └── features.py                    ← entity + feature view definitions
├── pipeline/
│   ├── ingestion.py                   ← ZenML step: raw data ingestion
│   ├── validation.py                  ← ZenML step: schema + validation
│   ├── transform.py                   ← ZenML step: preprocessing + features
│   ├── feature_store.py               ← ZenML step: Feast registration
│   └── zenml_pipeline.py              ← full ZenML pipeline wiring
├── notebooks/
│   └── milestone3_pipeline.ipynb      ← demonstration notebook
├── configs/
│   └── params.yaml                    ← central parameter file
├── run_pipeline.py                    ← standalone pipeline runner (all 4 steps)
├── merge_data.py                      ← merges 16 project CSVs into one file
├── requirements.txt
└── README.md                          ← this file
```

---

## 1 · Data Ingestion & Raw Storage

**Tool:** Python + ZenML step | **Points: 5**

### What was done

The dataset consists of 16 separate CSV files (one per JIRA project: `appceleratorstudio.csv`, `moodle.csv`, `springxd.csv`, etc.). These were first merged into a single unified file using [`merge_data.py`](merge_data.py), then ingested and cleaned by the ingestion step in [`run_pipeline.py`](run_pipeline.py).

**Merging 16 project files:**
```python
# merge_data.py
dfs = []
for f in all_files:
    project_name = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df['project'] = project_name
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('data/raw/raw_data.csv', index=False)
# Result: 23,313 rows across 16 projects
```

**Column normalization** — different project files used different names for the same field:

| Raw column name | Canonical name |
|---|---|
| `storypoint`, `story_points`, `point` | `storypoints` |
| `concat`, `body` | `description` |
| `summary` | `title` |

**Data quality filtering:**
- Dropped rows with null `storypoints`
- Dropped rows with `storypoints <= 0`
- Result: 23,313 clean rows retained

**Raw snapshot:** [`data/raw/raw_data.csv.dvc`](data/raw/raw_data.csv.dvc) (DVC tracked)

---

## 2 · Schema Definition & Data Validation

**Tool:** Great Expectations + custom statistics | **Points: 2 + 3**

### 2.1 Data Split

The dataset contains a pre-defined `split_mark` column assigned by the Llama3SP authors, ensuring consistent and reproducible splits across all research using this benchmark:

| Split | Rows | Percentage | Purpose |
|---|---|---|---|
| **Train** | 13,981 | 60% | Schema inference + model training |
| **Validation** | 4,661 | 20% | Anomaly detection + hyperparameter tuning |
| **Test** | 4,671 | 20% | Final evaluation only — **never touched during validation** |

Split details: [`tfdv_output/split_info.json`](tfdv_output/split_info.json)

### 2.2 Data Statistics

Statistics were computed for all three splits across all 6 columns:

```json
{
  "train": {
    "storypoints": { "mean": 4.87, "std": 6.21, "min": 0.5, "max": 100.0 },
    "title":       { "unique": 13891, "null_pct": 0.0 },
    "description": { "null_pct": 12.3 }
  },
  "validation": { ... },
  "test":       { ... }
}
```

Full statistics: [`tfdv_output/data_statistics.json`](tfdv_output/data_statistics.json)

### 2.3 Schema Definition

The schema is inferred **from the training set only** — this prevents data leakage from the validation or test sets into the schema definition:

```json
{
  "storypoints": { "type": "numeric", "min": 0.5,  "max": 100.0, "max_null_pct": 10.0 },
  "title":       { "type": "string",  "unique_values": 13891,     "max_null_pct": 10.0 },
  "description": { "type": "string",  "unique_values": 13102,     "max_null_pct": 22.3 },
  "project":     { "type": "string",  "unique_values": 16,        "max_null_pct": 10.0 },
  "split_mark":  { "type": "string",  "unique_values": 3,         "max_null_pct": 10.0 },
  "issuekey":    { "type": "string",  "unique_values": 13981,     "max_null_pct": 10.0 }
}
```

Saved schema: [`schema/schema.json`](schema/schema.json)

### 2.4 Anomaly Detection & Schema Revision

The **validation set** (not the test set) is checked against the training schema using Great Expectations:

- Null percentage violations
- Numeric values outside the expected range
- Missing columns

**Result: 0 anomalies detected** in the validation set. The schema required no revision, confirming that the pre-defined splits by the Llama3SP authors are internally consistent.

Anomaly report: [`tfdv_output/anomalies_report.json`](tfdv_output/anomalies_report.json)

> **Why validate on val and not test?**  
> The test set is kept completely untouched until final model evaluation in a later milestone. Validating against it would constitute data leakage and bias the final performance estimate.

---

## 3 · Preprocessing & Feature Engineering

**Tool:** Python (pandas + regex) wrapped in ZenML Transform step | **Points: 5**

Implementation: [`pipeline/transform.py`](pipeline/transform.py) | [`run_pipeline.py`](run_pipeline.py)

### 3.1 Text Cleaning

Raw JIRA text undergoes a 5-stage cleaning pipeline:

| Stage | Operation | Example |
|---|---|---|
| Lowercasing | `text.lower()` | `"Add Feature"` → `"add feature"` |
| URL removal | regex `https?://\S+` | `"see https://jira.com"` → `"see "` |
| JIRA markup removal | `{code}`, `[~user]`, `!image!` | `"{code}x=1{code}"` → `" "` |
| Special char removal | keep `[a-z0-9 .,!?-]` | `"fix: issue #42"` → `"fix issue 42"` |
| Whitespace collapse | `\s+` → single space | `"fix  bug"` → `"fix bug"` |

### 3.2 Input Text Construction

Title and description are combined into a single field matching the exact prompt format expected by the Llama3SP tokenizer:

```python
input_text = f"Title: {cleaned_title} Description: {cleaned_description}"
# Example:
# "Title: add ca against object literals Description: div class p style..."
```

### 3.3 Engineered Features

| Feature | Type | Description | Use in model |
|---|---|---|---|
| `input_text` | string | Combined title + description | Primary LLM input |
| `cleaned_title` | string | Cleaned title only | Reference |
| `cleaned_description` | string | Cleaned description only | Reference |
| `text_length` | int | Character count of `input_text` | Complexity proxy |
| `word_count` | int | Word count of `input_text` | Complexity proxy |
| `title_word_count` | int | Word count of cleaned title | Title length signal |
| `has_description` | int (0/1) | 1 if description is non-empty | Missingness flag |
| `log_storypoints` | float | `log1p(storypoints)` | Reduces target skew |
| `is_fibonacci` | int (0/1) | 1 if value in {1,2,3,5,8,13,20,40,100} | Planning poker alignment |

Feature distribution plots: [`tfdv_output/feature_distributions.png`](tfdv_output/feature_distributions.png)

### 3.4 Output Files

| File | Rows | Description |
|---|---|---|
| `data/processed/train.parquet` | 13,981 | Training set with all engineered features |
| `data/processed/val.parquet` | 4,661 | Validation set |
| `data/processed/test.parquet` | 4,671 | Test set (untouched until Milestone 4) |
| `data/processed/processed_data.parquet` | 23,313 | Full dataset |

Each processed file contains **15 columns**: 6 original + 9 engineered features.

---

## 4 · Feature Store (Feast)

**Tool:** Feast (local FileSource, offline mode) | **Points: 1**

Implementation: [`pipeline/feature_store.py`](pipeline/feature_store.py) | [`feast_repo/feature_repo/features.py`](feast_repo/feature_repo/features.py)

A Feast feature store was set up to register the engineered features, enabling consistent feature retrieval across training, evaluation, and future inference — decoupling feature computation from model training.

### Entity
```python
issue_entity = Entity(name="issue_id")
# Unique identifier for each JIRA issue / user story
```

### Feature View
```python
user_story_features = FeatureView(
    name="user_story_features",
    entities=[issue_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="text_length",      dtype=Int64),
        Field(name="word_count",       dtype=Int64),
        Field(name="has_description",  dtype=Int64),
        Field(name="title_word_count", dtype=Int64),
        Field(name="log_storypoints",  dtype=Float32),
        Field(name="is_fibonacci",     dtype=Int64),
    ],
    source=user_story_source,
)
```

**To apply the feature store:**
```bash
cd feast_repo/feature_repo
feast apply
```

**Feast apply output:**
```
Applying changes for project agile_story_points
Created entity issue_id
Created feature view user_story_features
Created sqlite table agile_story_points_user_story_features
```

---

## 5 · Data Versioning (DVC)

**Tool:** DVC (Data Version Control) | **Points: 3**

All data artifacts are tracked using DVC so every experiment can be reproduced exactly by checking out the corresponding DVC pointer files alongside the code.

### Tracked Artifacts

| Artifact | DVC pointer file | Git tag |
|---|---|---|
| `data/raw/raw_data.csv` | `data/raw/raw_data.csv.dvc` | `v1.0-milestone3` |
| `data/processed/processed_data.parquet` | `data/processed/processed_data.parquet.dvc` | `v1.0-milestone3` |

### Commands Used
```bash
dvc init
dvc add "Milestone 3/data/raw/raw_data.csv"
dvc add "Milestone 3/data/processed/processed_data.parquet"
git add .
git commit -m "feat: Milestone 3 - data pipeline complete"
git tag -a v1.0-milestone3 -m "Milestone 3 complete"
git push origin main
git push origin --tags
```

The `.dvc` pointer files are committed to Git instead of the actual data files, keeping the repository lightweight while maintaining full data reproducibility.

---

## 6 · ML Pipeline Integration (ZenML)

**Tool:** ZenML | **Points: included in ingestion + pipeline**

All pipeline steps are implemented as ZenML `@step` functions and wired into a single `@pipeline`, integrating this milestone's data work into the larger MLOps platform that supports model training, experiment tracking, and deployment in subsequent milestones.

**Pipeline definition:** [`pipeline/zenml_pipeline.py`](pipeline/zenml_pipeline.py)

```python
@pipeline(name="milestone3_data_pipeline", enable_cache=True)
def data_pipeline(csv_path: str = "data/raw/raw_data.csv") -> None:
    raw_df       = ingest_data(csv_path=csv_path)
    val_report   = validate_data(df=raw_df)
    processed_df = preprocess_and_engineer(df=raw_df)
    store_info   = setup_feature_store(df=processed_df)
```

**Standalone runner (no ZenML server needed):**
```bash
python run_pipeline.py
```

### Pipeline DAG

```
ingest_data
    │
    ▼
validate_data          → schema/schema.json
    │                  → tfdv_output/anomalies_report.json
    │                  → tfdv_output/data_statistics.json
    ▼
preprocess_and_engineer → data/processed/train.parquet
    │                   → data/processed/val.parquet
    │                   → data/processed/test.parquet
    ▼
setup_feature_store    → feast_repo/feature_repo/
```

---

## Grading Checklist

| Requirement | Tool | Points | Deliverable |
|---|---|---|---|
| Schema definition | Custom stats + JSON | 2 | [`schema/schema.json`](schema/schema.json) |
| Data validation & verification (stats, anomaly detection, fix) | Great Expectations | 3 | [`tfdv_output/`](tfdv_output/) |
| Data versioning | DVC | 3 | `.dvc` pointer files + git tag `v1.0-milestone3` |
| Feature store | Feast | 1 | [`feast_repo/feature_repo/features.py`](feast_repo/feature_repo/features.py) |
| Ingestion of raw data & storage | Python + ZenML | 5 | [`run_pipeline.py`](run_pipeline.py) · [`data/raw/raw_data.csv.dvc`](data/raw/raw_data.csv.dvc) |
| Preprocessing & Feature Engineering | ZenML Transform step | 5 | [`pipeline/transform.py`](pipeline/transform.py) · [`data/processed/`](data/processed/) |
| **Total** | | **19** | |

---

## References

1. Choetkiertikul, M., Dam, H. K., Tran, T., Treude, C., & Ghose, A. (2018). *A deep learning model for estimating story points.* IEEE TSE. https://doi.org/10.1109/TSE.2018.2792247
2. Fu, M., & Tantithamthavorn, C. (2022). *GPT2SP: A Transformer-based Agile story point estimation approach.* IEEE TSE. https://doi.org/10.1109/TSE.2022.3160829
3. Sepúlveda Montoya, C., Ríos Gómez, J., & Jaramillo Villegas, J. A. (2025). *Llama3SP.* https://github.com/DEVCamiloSepulveda/llama3sp
4. Great Expectations Documentation. https://docs.greatexpectations.io
5. ZenML Documentation. https://docs.zenml.io
6. Feast Feature Store. https://docs.feast.dev
7. DVC Documentation. https://dvc.org/doc
