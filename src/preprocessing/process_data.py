"""Curățare + normalizare date și salvare parametri de preprocesare."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


INFILE = Path('data/processed/combined.csv')
OUTFILE = Path('data/processed/combined.csv')
CONFIG_JSON = Path('config/preprocessing_params.json')
CONFIG_PKL = Path('config/preprocessing_params.pkl')


def clip_outliers(df: pd.DataFrame, cols: list[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    clipped = df.copy()
    for col in cols:
        if col not in clipped.columns:
            continue
        lo = clipped[col].quantile(lower_q)
        hi = clipped[col].quantile(upper_q)
        clipped[col] = clipped[col].clip(lo, hi)
    return clipped


def procesare() -> None:
    if not INFILE.exists():
        raise RuntimeError('Nu există data/processed/combined.csv. Rulați mai întâi combine_datasets.py')

    df = pd.read_csv(INFILE)
    df.columns = [str(c).strip() for c in df.columns]

    # Păstrăm doar coloanele relevante dacă există
    keep_cols = [c for c in ['timestamp', 'distance_raw', 'signal_strength', 'temperature', 'distance_ref'] if c in df.columns]
    df = df[keep_cols].copy()

    # Convert numeric columns
    numeric_cols = [c for c in ['distance_raw', 'signal_strength', 'temperature', 'distance_ref'] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminare duplicate
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f'Eliminate duplicate: {before - after}')

    # Tratare lipsuri: imputare mediană pe coloane numerice
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Outlieri: clipare percentile 1-99%
    df = clip_outliers(df, numeric_cols)

    # Normalizare/standardizare pe features (fără label)
    feature_cols = [c for c in ['distance_raw', 'signal_strength', 'temperature'] if c in df.columns]
    scaler = StandardScaler()
    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Salvare parametri de preprocesare
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    params = {
        'scaler': 'StandardScaler',
        'features': feature_cols,
        'mean_': scaler.mean_.tolist() if feature_cols else [],
        'scale_': scaler.scale_.tolist() if feature_cols else [],
    }
    CONFIG_JSON.write_text(json.dumps(params, indent=2), encoding='utf-8')
    joblib.dump({'scaler': scaler, 'features': feature_cols}, CONFIG_PKL)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTFILE, index=False)
    print(f'Saved cleaned data to {OUTFILE}')


if __name__ == '__main__':
    procesare()
