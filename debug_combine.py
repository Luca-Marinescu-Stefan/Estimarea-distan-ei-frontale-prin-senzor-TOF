import pandas as pd
import glob

raw_files = sorted(glob.glob('data/raw/*.csv'))
dfs = []
for f in raw_files:
    df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    print('File:', f)
    print('  Columns:', df.columns.tolist())
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print('After concat:', combined.columns.tolist())

col_map = {
    'distanta_bruta': 'distance_raw',
    'intensitate': 'signal_strength',
    'temperatura': 'temperature',
    'distanta_reala': 'distance_ref'
}
combined = combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns})
print('After rename:', combined.columns.tolist())

if 'distance_ref' in combined.columns:
    print('distance_ref non-null:', combined['distance_ref'].notna().sum(), '/', len(combined))
else:
    print('No distance_ref column!')
