"""Basic feature engineering for TOF dataset.
Usage: python src/preprocessing/feature_engineering.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


INFILE = Path('data/processed/cleaned.csv')
OUTFILE = Path('data/processed/combined.csv')


def main() -> None:
	if not INFILE.exists():
		raise RuntimeError('Missing data/processed/cleaned.csv. Run data_cleaner.py first.')

	df = pd.read_csv(INFILE)
	if 'distance_raw' in df.columns and 'distance_ref' in df.columns:
		df['distance_error'] = df['distance_raw'] - df['distance_ref']
		df['distance_abs_error'] = (df['distance_raw'] - df['distance_ref']).abs()

	OUTFILE.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(OUTFILE, index=False)
	print(f'Saved engineered data to {OUTFILE}')


if __name__ == '__main__':
	main()
