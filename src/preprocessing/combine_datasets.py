"""Combine raw CSVs (data/raw and data/generated) into data/processed/combined.csv"""
# -----------------------------------------------------------------------------
# Modul: src/preprocessing/combine_datasets.py
# Scop: combină toate CSV-urile raw + generated într-un singur fișier.
# Input: data/raw/*.csv și data/generated/*.csv.
# Output: data/processed/combined.csv.
# Utilizare: python src/preprocessing/combine_datasets.py
# Pași principali:
#   1) Încărcare și normalizare nume coloane.
#   2) Mapare RO→EN pentru feature-uri.
#   3) Curățare distance_ref și filtrare rânduri invalide.
# Dependențe: pandas, glob, regex.
# Parametri implicați: col_map, bins de curățare.
# Fișiere scrise: data/processed/combined.csv.
# Observații: elimină duplicate de coloane și rânduri fără distance_ref.
# -----------------------------------------------------------------------------
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
        df.columns = [str(c).strip() for c in df.columns]
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        dfs.append(df)
    except Exception as e:
        print('skip', f, e)

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = [str(c).strip() for c in combined.columns]
    
    # For files with both Romanian and English columns, keep only English ones
    # Build a set of columns to keep
    keep_cols = []
    seen = set()
    
    for col in combined.columns:
        # Map Romanian to English names
        mapped = {
            'distanta_bruta': 'distance_raw',
            'intensitate': 'signal_strength',
            'temperatura': 'temperature',
            'distanta_reala': 'distance_ref'
        }.get(col, col)
        
        if mapped not in seen:
            keep_cols.append(col)
            seen.add(mapped)
    
    combined = combined[keep_cols]
    
    # Now rename
    col_map = {
        'distanta_bruta': 'distance_raw',
        'intensitate': 'signal_strength',
        'temperatura': 'temperature',
        'distanta_reala': 'distance_ref'
    }
    combined = combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns})
    
    # Convert distance_ref to numeric
    if 'distance_ref' in combined.columns:
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
        
        col = combined['distance_ref'].map(_unwrap)
        
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
        
        before = len(combined)
        combined = combined.dropna(subset=['distance_ref'], how='any')
        after = len(combined)
    else:
        before = len(combined)
        after = 0
        combined = combined.iloc[0:0]
    
    # Keep only relevant columns
    keep = [c for c in ['timestamp','distance_raw','signal_strength','temperature','distance_ref'] if c in combined.columns]
    combined = combined[keep]
    
    combined.to_csv(out / 'combined.csv', index=False)
    print(f'Wrote {len(combined)} rows to data/processed/combined.csv (dropped {before-after} rows with missing distance_ref)')
else:
    print('No files to combine')
