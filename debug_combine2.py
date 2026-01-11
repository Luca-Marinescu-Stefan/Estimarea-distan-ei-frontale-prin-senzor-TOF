import pandas as pd
import glob
import re
import ast

raw_files = sorted(glob.glob('data/raw/*.csv'))
gen_files = sorted(glob.glob('data/generated/*.csv'))
all_files = raw_files + gen_files

dfs = []
for f in all_files:
    df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.columns = [str(c).strip() for c in combined.columns]
if combined.columns.duplicated().any():
    combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]

print('Columns:', combined.columns.tolist())

col_map = {
    'distanta_bruta': 'distance_raw',
    'intensitate': 'signal_strength',
    'temperatura': 'temperature',
    'distanta_reala': 'distance_ref'
}
combined = combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns})

print('After rename:', combined.columns.tolist())

if 'distance_ref' in combined.columns:
    print('distance_ref before conversion:')
    print('  dtype:', combined['distance_ref'].dtype)
    print('  non-null count:', combined['distance_ref'].notna().sum())
    print('  null count:', combined['distance_ref'].isna().sum())
    print('  sample values:', combined['distance_ref'].head(10).tolist())
    
    # Try conversion
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
    
    col_numeric = col.map(_to_number)
    print('After conversion:')
    print('  non-null count:', col_numeric.notna().sum())
    print('  null count:', col_numeric.isna().sum())
    print('  sample values:', col_numeric.head(10).tolist())
