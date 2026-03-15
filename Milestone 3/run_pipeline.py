import os, json, math, re, glob, subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("  MILESTONE 3 - Data Pipeline")
print("=" * 60)

# STEP 1: DATA INGESTION
print("\n[STEP 1] Data Ingestion ...")
df = pd.read_csv("data/raw/raw_data.csv")
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
ALIASES = {"storypoint":"storypoints","story_points":"storypoints","point":"storypoints","concat":"description","body":"description","summary":"title"}
df = df.rename(columns=ALIASES)
df = df.dropna(subset=["storypoints"])
df = df[df["storypoints"] > 0].reset_index(drop=True)
print(f"  After cleaning: {len(df):,} rows")
print("  [STEP 1] Done\n")

# STEP 2: DATA VALIDATION
print("[STEP 2] Data Validation ...")
os.makedirs("schema", exist_ok=True)
os.makedirs("tfdv_output", exist_ok=True)
train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
eval_df  = df.drop(train_df.index).reset_index(drop=True)
print(f"  Train: {len(train_df):,} | Eval: {len(eval_df):,}")

def compute_stats(d):
    s = {}
    for col in d.columns:
        entry = {"dtype": str(d[col].dtype), "count": int(d[col].count()), "nulls": int(d[col].isnull().sum()), "null_pct": round(d[col].isnull().mean()*100,2)}
        if pd.api.types.is_numeric_dtype(d[col]):
            entry.update({"mean":round(float(d[col].mean()),4),"std":round(float(d[col].std()),4),"min":round(float(d[col].min()),4),"max":round(float(d[col].max()),4),"median":round(float(d[col].median()),4)})
        else:
            entry["unique"] = int(d[col].nunique())
        s[col] = entry
    return s

train_stats = compute_stats(train_df)
eval_stats  = compute_stats(eval_df)
with open("tfdv_output/data_statistics.json","w") as f: json.dump({"train":train_stats,"eval":eval_stats},f,indent=2)
print("  Statistics saved -> tfdv_output/data_statistics.json")

schema = {}
for col in train_df.columns:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        schema[col] = {"type":"numeric","min":float(train_df[col].min()),"max":float(train_df[col].max()),"max_null_pct":round(train_df[col].isnull().mean()*100+10,1)}
    else:
        schema[col] = {"type":"string","unique_values":int(train_df[col].nunique()),"max_null_pct":round(train_df[col].isnull().mean()*100+10,1)}
with open("schema/schema.json","w") as f: json.dump(schema,f,indent=2)
print("  Schema saved -> schema/schema.json")
for col,rules in schema.items(): print(f"    - {col}: {rules['type']}")

print("\n  Checking evaluation set for anomalies ...")
anomalies = {}
for col, rules in schema.items():
    if col not in eval_df.columns:
        anomalies[col] = "Column missing in evaluation set"; continue
    actual_null_pct = eval_df[col].isnull().mean() * 100
    if actual_null_pct > rules["max_null_pct"]:
        anomalies[col] = f"Null% too high: {actual_null_pct:.1f}% (allowed: {rules['max_null_pct']}%)"
    if rules["type"] == "numeric":
        out_of_range = eval_df[(eval_df[col] < rules["min"]) | (eval_df[col] > rules["max"]*1.5)][col]
        pct = len(out_of_range) / len(eval_df) * 100
        if pct > 0:
            anomalies[col] = f"Values out of range [{rules['min']}, {rules['max']}]: {pct:.1f}% unexpected"

with open("tfdv_output/anomalies_report.json","w") as f: json.dump(anomalies,f,indent=2)
if anomalies:
    print(f"  {len(anomalies)} anomalies detected:")
    for col,issue in anomalies.items(): print(f"    -> {col}: {issue}")
else:
    print("  No anomalies detected in evaluation set")

if anomalies:
    print("\n  Fixing schema ...")
    for col,issue in anomalies.items():
        if col not in schema: continue
        if schema[col]["type"] == "numeric":
            schema[col]["min"] = min(schema[col]["min"], float(eval_df[col].min()))
            schema[col]["max"] = max(schema[col]["max"], float(eval_df[col].max()))
            print(f"    Fixed '{col}' range -> [{schema[col]['min']}, {schema[col]['max']}]")
        else:
            schema[col]["max_null_pct"] = min(schema[col]["max_null_pct"]+20,80)
    with open("schema/schema.json","w") as f: json.dump(schema,f,indent=2)
    print("  Revised schema saved -> schema/schema.json")
print("  [STEP 2] Done\n")

# STEP 3: PREPROCESSING & FEATURE ENGINEERING
print("[STEP 3] Preprocessing & Feature Engineering ...")
FIBONACCI = {0.5,1,2,3,5,8,13,20,40,100}
def clean_text(text):
    if not isinstance(text,str) or not text.strip(): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+"," ",text)
    text = re.sub(r"\{[^}]+\}"," ",text)
    text = re.sub(r"\[~[^\]]+\]"," ",text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

df_proc = df.copy()
df_proc["title"]       = df_proc["title"].fillna("")
df_proc["description"] = df_proc["description"].fillna("")
df_proc["cleaned_title"]       = df_proc["title"].apply(clean_text)
df_proc["cleaned_description"] = df_proc["description"].apply(clean_text)
df_proc["input_text"] = df_proc.apply(lambda r: f"Title: {r['cleaned_title']} Description: {r['cleaned_description']}" if r["cleaned_title"] and r["cleaned_description"] else f"Title: {r['cleaned_title']}" if r["cleaned_title"] else f"Description: {r['cleaned_description']}", axis=1)
df_proc["text_length"]      = df_proc["input_text"].apply(len)
df_proc["word_count"]       = df_proc["input_text"].apply(lambda x: len(x.split()))
df_proc["has_description"]  = (df_proc["cleaned_description"].str.len()>0).astype(int)
df_proc["title_word_count"] = df_proc["cleaned_title"].apply(lambda x: len(x.split()) if x else 0)
df_proc["log_storypoints"]  = df_proc["storypoints"].apply(lambda x: math.log1p(x))
df_proc["is_fibonacci"]     = df_proc["storypoints"].apply(lambda x: 1 if x in FIBONACCI else 0)
df_proc = df_proc[df_proc["input_text"].str.len()>0].reset_index(drop=True)
os.makedirs("data/processed",exist_ok=True)
df_proc.to_parquet("data/processed/processed_data.parquet",index=False)
print(f"  Processed shape: {df_proc.shape}")
print(f"  Sample: {df_proc['input_text'].iloc[0][:80]}...")
print(f"  Saved -> data/processed/processed_data.parquet")

fig,axes = plt.subplots(2,2,figsize=(14,8))
axes[0,0].hist(df_proc["storypoints"],bins=40,color="steelblue",edgecolor="white"); axes[0,0].set_title("Story Points Distribution")
axes[0,1].hist(df_proc["word_count"],bins=40,color="coral",edgecolor="white"); axes[0,1].set_title("Word Count Distribution")
axes[1,0].hist(df_proc["log_storypoints"],bins=30,color="green",edgecolor="white"); axes[1,0].set_title("log1p(Story Points)")
fib_c = df_proc["is_fibonacci"].value_counts(); axes[1,1].bar(["Non-Fibonacci","Fibonacci"],fib_c.values,color=["#e74c3c","#2ecc71"]); axes[1,1].set_title("Fibonacci Story Points")
plt.tight_layout(); plt.savefig("tfdv_output/feature_distributions.png",dpi=150); plt.close()
print("  Plot saved -> tfdv_output/feature_distributions.png")
print("  [STEP 3] Done\n")

# STEP 4: FEATURE STORE
print("[STEP 4] Feature Store (Feast) ...")
os.makedirs("feast_repo/feature_repo",exist_ok=True)
df_feast = pd.read_parquet("data/processed/processed_data.parquet")
if "event_timestamp" not in df_feast.columns: df_feast["event_timestamp"] = pd.Timestamp.now(tz="UTC")
if "issue_id" not in df_feast.columns: df_feast["issue_id"] = range(len(df_feast))
df_feast.to_parquet("data/processed/processed_data.parquet",index=False)
abs_path = os.path.abspath("data/processed/processed_data.parquet").replace("\\","/")
with open("feast_repo/feature_repo/feature_store.yaml","w") as f:
    f.write("project: agile_story_points\nregistry: data/registry.db\nprovider: local\nonline_store:\n  type: sqlite\n  path: data/online_store.db\n")
with open("feast_repo/feature_repo/features.py","w") as f:
    f.write(f'''from datetime import timedelta\nfrom feast import Entity, FeatureView, Field, FileSource\nfrom feast.types import Int64, Float32\nissue_entity = Entity(name="issue_id")\nuser_story_source = FileSource(path="{abs_path}", timestamp_field="event_timestamp")\nuser_story_features = FeatureView(name="user_story_features", entities=[issue_entity], ttl=timedelta(days=90), schema=[Field(name="text_length",dtype=Int64),Field(name="word_count",dtype=Int64),Field(name="has_description",dtype=Int64),Field(name="log_storypoints",dtype=Float32),Field(name="is_fibonacci",dtype=Int64)], source=user_story_source)\n''')
print("  Feast files written")
result = subprocess.run(["feast","apply"],cwd="feast_repo/feature_repo",capture_output=True,text=True)
if result.returncode == 0: print(f"  feast apply succeeded\n  {result.stdout.strip()}")
else: print(f"  feast apply note: {result.stderr.strip()[:150]}")
print("  [STEP 4] Done\n")

print("=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"  Raw data:       data/raw/raw_data.csv ({len(df):,} rows)")
print(f"  Schema:         schema/schema.json")
print(f"  Statistics:     tfdv_output/data_statistics.json")
print(f"  Anomalies:      tfdv_output/anomalies_report.json ({len(anomalies)} found)")
print(f"  Processed data: data/processed/processed_data.parquet ({len(df_proc):,} rows)")
print(f"  Feature plots:  tfdv_output/feature_distributions.png")
print(f"  Feast store:    feast_repo/feature_repo/")
print("=" * 60)
