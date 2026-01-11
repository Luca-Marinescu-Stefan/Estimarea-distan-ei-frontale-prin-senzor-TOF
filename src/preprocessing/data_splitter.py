"""Split combined data into train/validation/test and save CSVs.
Usage: python src/preprocessing/data_splitter.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

infile = 'data/processed/combined.csv'
outdir = Path('data')

try:
    df = pd.read_csv(infile)
except Exception as e:
    print('Error reading combined file:', e)
    raise SystemExit(1)

# for demo, use distance_ref as label (regression -> convert to bins for classification)
# create a simple class by binning distance_ref
bins = [0,150,400,800,1200,2000]
labels = [0,1,2,3,4]
df['label'] = pd.cut(df['distance_ref'], bins=bins, labels=labels)
# Convert label to numeric and drop rows without label
df['label'] = pd.to_numeric(df['label'], errors='coerce')
before = len(df)
df = df.dropna(subset=['label'])
after = len(df)
if before != after:
    print(f'Dropped {before-after} rows without label after binning')

# ensure integer labels
df['label'] = df['label'].astype(int)

# Handle very small datasets deterministically to avoid empty splits
n = len(df)
if n == 0:
    raise RuntimeError('No rows available after binning and dropping NaNs; cannot create splits')

if n == 1:
    # Duplicate the single row into train/val/test so downstream code has something to work with
    print('Only 1 sample available after cleaning — duplicating into train/val/test for demo purposes')
    train = df.copy()
    val = df.copy()
    test = df.copy()
elif n == 2:
    # Use one for train, one for val/test (duplicate second for test)
    print('Only 2 samples available — creating train (1), val (1), test (1 by duplication)')
    train = df.iloc[[0]].copy()
    val = df.iloc[[1]].copy()
    test = df.iloc[[1]].copy()
else:
    # For larger but still small datasets, try stratified split when possible
    stratify_valid = True
    try:
        counts = df['label'].value_counts()
        if counts.min() < 2:
            stratify_valid = False
    except Exception:
        stratify_valid = False

    if stratify_valid:
        try:
            train, rest = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
            # rest -> split equally to val/test
            if len(rest) < 2:
                # fallback to deterministic assignment
                raise ValueError('Not enough samples in rest to split into val/test')
            val, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest['label'])
        except Exception:
            stratify_valid = False

    if not stratify_valid:
        if not stratify_valid:
            print('Warning: stratified split not possible (insufficient samples per class). Using deterministic random split without stratify.')
        # Deterministic shuffled indices to allocate counts without producing empty sets
        rng = __import__('numpy').random.RandomState(42)
        idx = rng.permutation(n)
        train_n = max(1, int(0.7 * n))
        val_n = max(1, int(0.15 * n))
        if train_n + val_n >= n:
            # ensure at least 1 sample reserved for test when possible
            if n - train_n >= 1:
                val_n = max(1, n - train_n - 1)
            else:
                val_n = max(1, n - train_n)
        test_n = n - train_n - val_n
        if test_n < 0:
            test_n = 0
        train_idx = idx[:train_n]
        val_idx = idx[train_n:train_n + val_n]
        test_idx = idx[train_n + val_n: train_n + val_n + test_n]
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy() if len(val_idx) > 0 else df.iloc[[train_idx[0]]].copy()
        test = df.iloc[test_idx].copy() if len(test_idx) > 0 else df.iloc[[train_idx[0]]].copy()

outdir.joinpath('train').mkdir(parents=True, exist_ok=True)
outdir.joinpath('validation').mkdir(parents=True, exist_ok=True)
outdir.joinpath('test').mkdir(parents=True, exist_ok=True)

train.to_csv(outdir / 'train' / 'X_train.csv', index=False)
val.to_csv(outdir / 'validation' / 'X_val.csv', index=False)
test.to_csv(outdir / 'test' / 'X_test.csv', index=False)

print('Saved splits: ', len(train), len(val), len(test))
