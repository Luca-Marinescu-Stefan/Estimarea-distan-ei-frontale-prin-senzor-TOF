"""Combine raw CSVs (data/raw and data/generated) into data/processed/combined.csv"""
import glob
import pandas as pd
from pathlib import Path
import re
import ast

raw_files = sorted(glob.glob('data/raw/*.csv'))
gen_files = sorted(glob.glob('data/generated/*.csv'))
all_files = raw_files + gen_files
out = Path('data/processed')
out.mkdir(parents=True, exist_ok=True)

dfs = []
for f in all_files:
    try:
        df = pd.read_csv(f)
                # Strip whitespace and deduplicate columns
                df.columns = [str(c).strip() for c in df.columns]
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
                dfs.append(df)
    except Exception as e:
        print('skip', f, e)

if dfs:
    # Concat and immediately strip/deduplicate columns
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = [str(c).strip() for c in combined.columns]
    
    # Remove duplicate columns by keeping first occurrence
    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    
    # Normalize column names from Romanian to expected English names
    col_map = {
        'distanta_bruta': 'distance_raw',
        'intensitate': 'signal_strength',
        'temperatura': 'temperature',
        'distanta_reala': 'distance_ref'
    }
    combined = combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns})

    # Ensure we have distance_raw, signal_strength, temperature, distance_ref
    # If missing, try to use any available numeric column
    if 'distance_raw' not in combined.columns and 'distanta_bruta' in combined.columns:
        combined['distance_raw'] = combined['distanta_bruta']
    if 'signal_strength' not in combined.columns and 'intensitate' in combined.columns:
        combined['signal_strength'] = combined['intensitate']
    if 'temperature' not in combined.columns and 'temperatura' in combined.columns:
        combined['temperature'] = combined['temperatura']
    if 'distance_ref' not in combined.columns and 'distanta_reala' in combined.columns:
        combined['distance_ref'] = combined['distanta_reala']

    # Convert distance_ref to numeric
    if 'distance_ref' in combined.columns:
        col = combined['distance_ref']
        
        # Unwrap list/tuple
        def _unwrap(x):
            try:
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    return x[0]
                if isinstance(x, str):
                    s = x.strip()
                    if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                        try:
                            v = ast.literal_eval(s)
                            if isinstance(v, (list, tuple)) and len(v) > 0:
                                return v[0]
                        except:
                            pass
                    return s
            except:
                return x
            return x

        col = col.map(_unwrap)
        
        # Extract numeric value
        number_re = re.compile(r"[-+]?[0-9]*[.,]?[0-9]+")
        def _to_number(x):
            try:
                if x is None:
                    return None
                if isinstance(x, (int, float)):
                    return x
                s = str(x)
                m = number_re.search(s)
                if not m:
                    return None
                num = m.group(0).replace(',', '.')
                return float(num)
            except:
                return None
        
        combined['distance_ref'] = col.map(_to_number)

    # Drop rows missing distance_ref
    before = len(combined)
    if 'distance_ref' in combined.columns:
        combined = combined.dropna(subset=['distance_ref'], how='any')
    else:
        combined = combined.iloc[0:0]
    after = len(combined)

    # Keep only relevant columns
    keep = [c for c in ['timestamp','distance_raw','signal_strength','temperature','distance_ref'] if c in combined.columns]
    combined = combined[keep]

    combined.to_csv(out / 'combined.csv', index=False)
    print('Wrote', len(combined), 'rows to', out / 'combined.csv', f'(dropped {before-after} rows with missing or invalid distance_ref)')
else:
    print('No files to combine')
