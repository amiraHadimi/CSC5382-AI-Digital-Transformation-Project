import pandas as pd
import os
import glob

dataset_path = r'..\Milestone 2\data\llama3sp_dataset'
all_files = glob.glob(os.path.join(dataset_path, '*.csv'))

dfs = []
for f in all_files:
    project_name = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df['project'] = project_name
    dfs.append(df)
    print(f'Loaded {project_name}: {len(df)} rows')

combined = pd.concat(dfs, ignore_index=True)
os.makedirs('data/raw', exist_ok=True)
combined.to_csv('data/raw/raw_data.csv', index=False)
print(f'Done! Total rows: {len(combined)}')
print(f'Columns: {list(combined.columns)}')
