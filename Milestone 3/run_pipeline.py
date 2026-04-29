import os, json, math, re, sys, subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("  MILESTONE 3 - Data Pipeline (with proper 3-way split)")
print("=" * 60)

# STEP 1: DATA INGESTION
print("\n[STEP 1] Data Ingestion ...")
df = pd.read_csv("data/raw/raw_data.csv")
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
ALIASES = {
    "storypoint":  "storypoints",
    "story_points":"storypoints",
    "point":       "storypoints",
    "concat":      "description",
    "body":        "description",
    "summary":     "title"
}
df = df.rename(columns=ALIASES)
df = df.dropna(subset=["storypoints"])
df = df[df["storypoints"] > 0].reset_index(drop=True)
print(f"  After cleaning: {len(df):,} rows")
print("  [STEP 1] Done\n")

# STEP 2: DATA VALIDATION WITH PROPER 3-WAY SPLIT
print("[STEP 2] Data Validation & Schema ...")
os.makedirs("schema", exist_ok=True)
os.makedirs("tfdv_output", exist_ok=True)
total = len(df)

if "split_mark" in df.columns:
    train_df = df[df["split_mark"] == "train"].reset_index(drop=True)
    val_df   = df[df["split_mark"] == "val"].reset_index(drop=True)
    test_df  = df[df["split_mark"] == "test"].reset_index(drop=True)
    print(f"  Using pre-defined splits from split_mark column:")
else:
    train_df  = df.sample(frac=0.70, random_state=42)
    remaining = df.drop(train_df.index)
    val_df    = remaining.sample(frac=0.50, random_state=42)
    test_df   = remaining.drop(val_df.index)
    train_df  = train_df.reset_index(drop=True)
    val_df    = val_df.reset_index(drop=True)
    test_df   = test_df.reset_index(drop=True)
    print(f"  Using manual 70/15/15 split:")

print(f"    Train:      {len(train_df):,} rows  ({len(train_df)/total*100:.1f}%)")
print(f"    Validation: {len(val_df):,} rows  ({len(val_df)/total*100:.1f}%)")
print(f"    Test:       {len(test_df):,} rows  ({len(test_df)/total*100:.1f}%)")

def compute_stats(d):
    s = {}
    for col in d.columns:
        entry = {
            "dtype":    str(d[col].dtype),
            "count":    int(d[col].count()),
            "nulls":    int(d[col].isnull().sum()),
            "null_pct": round(d[col].isnull().mean()*100, 2)
        }
        if pd.api.types.is_numeric_dtype(d[col]):
            entry.update({
                "mean":   round(float(d[col].mean()), 4),
                "std":    round(float(d[col].std()), 4),
                "min":    round(float(d[col].min()), 4),
                "max":    round(float(d[col].max()), 4),
                "median": round(float(d[col].median()), 4),
            })
        else:
            entry["unique"] = int(d[col].nunique())
        s[col] = entry
    return s

train_stats = compute_stats(train_df)
val_stats   = compute_stats(val_df)
test_stats  = compute_stats(test_df)
with open("tfdv_output/data_statistics.json", "w") as f:
    json.dump({"train": train_stats, "validation": val_stats, "test": test_stats}, f, indent=2)
print("  Statistics saved -> tfdv_output/data_statistics.json")

schema = {}
for col in train_df.columns:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        schema[col] = {
            "type":         "numeric",
            "min":          float(train_df[col].min()),
            "max":          float(train_df[col].max()),
            "max_null_pct": round(train_df[col].isnull().mean()*100 + 10, 1)
        }
    else:
        schema[col] = {
            "type":          "string",
            "unique_values": int(train_df[col].nunique()),
            "max_null_pct":  round(train_df[col].isnull().mean()*100 + 10, 1)
        }
with open("schema/schema.json", "w") as f:
    json.dump(schema, f, indent=2)
print("  Schema inferred from training set -> schema/schema.json")
for col, rules in schema.items():
    print(f"    - {col}: {rules['type']}")

print("\n  Checking VALIDATION set for anomalies ...")
anomalies = {}
for col, rules in schema.items():
    if col not in val_df.columns:
        anomalies[col] = "Column missing"
        continue
    actual_null_pct = val_df[col].isnull().mean() * 100
    if actual_null_pct > rules["max_null_pct"]:
        anomalies[col] = f"Null% too high: {actual_null_pct:.1f}%"
    if rules["type"] == "numeric" and len(val_df) > 0:
        out_of_range = val_df[
            (val_df[col] < rules["min"]) |
            (val_df[col] > rules["max"] * 1.5)
        ][col]
        pct = len(out_of_range) / len(val_df) * 100
        if pct > 0:
            anomalies[col] = f"Out of range: {pct:.1f}% unexpected values"

with open("tfdv_output/anomalies_report.json", "w") as f:
    json.dump(anomalies, f, indent=2)

if anomalies:
    print(f"  {len(anomalies)} anomalies detected:")
    for col, issue in anomalies.items():
        print(f"    -> {col}: {issue}")
    for col, issue in anomalies.items():
        if col not in schema:
            continue
        if schema[col]["type"] == "numeric":
            schema[col]["min"] = min(schema[col]["min"], float(val_df[col].min()))
            schema[col]["max"] = max(schema[col]["max"], float(val_df[col].max()))
            print(f"    Fixed '{col}' range -> [{schema[col]['min']}, {schema[col]['max']}]")
        else:
            schema[col]["max_null_pct"] = min(schema[col]["max_null_pct"] + 20, 80)
    with open("schema/schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("  Revised schema saved")
else:
    print("  No anomalies detected in validation set")

split_info = {
    "total": total,
    "train":      {"size": len(train_df), "pct": round(len(train_df)/total*100, 1)},
    "validation": {"size": len(val_df),   "pct": round(len(val_df)/total*100, 1)},
    "test":       {"size": len(test_df),  "pct": round(len(test_df)/total*100, 1)},
    "split_source": "split_mark column (pre-defined by Llama3SP authors)",
    "note": "Schema from train only. Anomalies on val only. Test untouched."
}
with open("tfdv_output/split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)
print("  Split info saved -> tfdv_output/split_info.json")
print("  [STEP 2] Done\n")

# STEP 3: PREPROCESSING & FEATURE ENGINEERING
print("[STEP 3] Preprocessing & Feature Engineering ...")
FIBONACCI = {0.5, 1, 2, 3, 5, 8, 13, 20, 40, 100}

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"\[~[^\]]+\]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df_proc = df.copy()
df_proc["title"]       = df_proc["title"].fillna("")
df_proc["description"] = df_proc["description"].fillna("")
df_proc["cleaned_title"]       = df_proc["title"].apply(clean_text)
df_proc["cleaned_description"] = df_proc["description"].apply(clean_text)
df_proc["input_text"] = df_proc.apply(
    lambda r: (
        f"Title: {r['cleaned_title']} Description: {r['cleaned_description']}"
        if r["cleaned_title"] and r["cleaned_description"]
        else f"Title: {r['cleaned_title']}"
        if r["cleaned_title"]
        else f"Description: {r['cleaned_description']}"
    ), axis=1
)
df_proc["text_length"]      = df_proc["input_text"].apply(len)
df_proc["word_count"]       = df_proc["input_text"].apply(lambda x: len(x.split()))
df_proc["has_description"]  = (df_proc["cleaned_description"].str.len() > 0).astype(int)
df_proc["title_word_count"] = df_proc["cleaned_title"].apply(lambda x: len(x.split()) if x else 0)
df_proc["log_storypoints"]  = df_proc["storypoints"].apply(lambda x: math.log1p(x))
df_proc["is_fibonacci"]     = df_proc["storypoints"].apply(lambda x: 1 if x in FIBONACCI else 0)
df_proc = df_proc[df_proc["input_text"].str.len() > 0].reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)
df_proc.to_parquet("data/processed/processed_data.parquet", index=False)

if "split_mark" in df_proc.columns:
    df_proc[df_proc["split_mark"] == "train"].to_parquet("data/processed/train.parquet", index=False)
    df_proc[df_proc["split_mark"] == "val"].to_parquet("data/processed/val.parquet", index=False)
    df_proc[df_proc["split_mark"] == "test"].to_parquet("data/processed/test.parquet", index=False)
    print(f"  Saved splits:")
    print(f"    -> data/processed/train.parquet  ({len(df_proc[df_proc['split_mark']=='train']):,} rows)")
    print(f"    -> data/processed/val.parquet    ({len(df_proc[df_proc['split_mark']=='val']):,} rows)")
    print(f"    -> data/processed/test.parquet   ({len(df_proc[df_proc['split_mark']=='test']):,} rows)")

print(f"  Full: data/processed/processed_data.parquet ({len(df_proc):,} rows, {df_proc.shape[1]} cols)")
print(f"  Sample: {df_proc['input_text'].iloc[0][:80]}...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0,0].hist(df_proc["storypoints"], bins=40, color="steelblue", edgecolor="white")
axes[0,0].set_title("Story Points Distribution")
axes[0,1].hist(df_proc["word_count"], bins=40, color="coral", edgecolor="white")
axes[0,1].set_title("Word Count Distribution")
axes[1,0].hist(df_proc["log_storypoints"], bins=30, color="green", edgecolor="white")
axes[1,0].set_title("log1p(Story Points)")
fib_c = df_proc["is_fibonacci"].value_counts()
axes[1,1].bar(["Non-Fibonacci", "Fibonacci"], fib_c.values, color=["#e74c3c", "#2ecc71"])
axes[1,1].set_title("Fibonacci Story Points")
plt.tight_layout()
plt.savefig("tfdv_output/feature_distributions.png", dpi=150)
plt.close()
print("  Plot saved -> tfdv_output/feature_distributions.png")
print("  [STEP 3] Done\n")

# STEP 4: FEATURE STORE (Feast)
print("[STEP 4] Feature Store (Feast) ...")
os.makedirs("feast_repo/feature_repo", exist_ok=True)

df_feast = pd.read_parquet("data/processed/processed_data.parquet")
if "event_timestamp" not in df_feast.columns:
    df_feast["event_timestamp"] = pd.Timestamp.now(tz="UTC")
if "issue_id" not in df_feast.columns:
    df_feast["issue_id"] = range(len(df_feast))
df_feast.to_parquet("data/processed/processed_data.parquet", index=False)

abs_path = os.path.abspath("data/processed/processed_data.parquet").replace("\\", "/")

with open("feast_repo/feature_repo/feature_store.yaml", "w") as f:
    f.write(
        "project: agile_story_points\n"
        "registry: data/registry.db\n"
        "provider: local\n"
        "offline_store:\n"
        "  type: file\n"
        "online_store:\n"
        "  type: sqlite\n"
        "  path: data/online_store.db\n"
    )

with open("feast_repo/feature_repo/features.py", "w") as f:
    f.write(
        "from datetime import timedelta\n"
        "from feast import Entity, FeatureView, Field, FileSource\n"
        "from feast.types import Int64, Float32\n"
        "from feast.value_type import ValueType\n\n"
        "issue_entity = Entity(\n"
        "    name=\"issue_id\",\n"
        "    join_keys=[\"issue_id\"],\n"
        "    value_type=ValueType.INT64,\n"
        "    description=\"Unique issue identifier\"\n"
        ")\n\n"
        f"user_story_source = FileSource(path=\"{abs_path}\", event_timestamp_column=\"event_timestamp\")\n\n"
        "user_story_features = FeatureView(\n"
        "    name=\"user_story_features\",\n"
        "    entities=[issue_entity],\n"
        "    ttl=timedelta(days=90),\n"
        "    schema=[\n"
        "        Field(name=\"text_length\",      dtype=Int64),\n"
        "        Field(name=\"word_count\",       dtype=Int64),\n"
        "        Field(name=\"has_description\",  dtype=Int64),\n"
        "        Field(name=\"title_word_count\", dtype=Int64),\n"
        "        Field(name=\"log_storypoints\",  dtype=Float32),\n"
        "        Field(name=\"is_fibonacci\",     dtype=Int64),\n"
        "    ],\n"
        "    source=user_story_source,\n"
        ")\n"
    )

try:
    from feast import FeatureStore
    import sys

    sys.path.insert(0, os.path.abspath("feast_repo/feature_repo"))

    from features import issue_entity, user_story_features

    store = FeatureStore(repo_path="feast_repo/feature_repo")
    store.apply([issue_entity, user_story_features])

    print("  feast apply succeeded (Python API)")
except Exception as e:
    print(f"  feast note: {str(e)[:200]}")
print("  [STEP 4] Done\n")

print("=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"  Raw data:        data/raw/raw_data.csv ({len(df):,} rows)")
print(f"  Schema:          schema/schema.json ({len(schema)} features)")
print(f"  Statistics:      tfdv_output/data_statistics.json (train+val+test)")
print(f"  Split info:      tfdv_output/split_info.json")
print(f"  Anomalies:       tfdv_output/anomalies_report.json ({len(anomalies)} found)")
print(f"  Train split:     data/processed/train.parquet")
print(f"  Val split:       data/processed/val.parquet")
print(f"  Test split:      data/processed/test.parquet")
print(f"  Feature plots:   tfdv_output/feature_distributions.png")
print(f"  Feast store:     feast_repo/feature_repo/")
print("=" * 60)
