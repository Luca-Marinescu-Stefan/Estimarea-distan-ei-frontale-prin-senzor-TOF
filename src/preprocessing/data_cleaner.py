"""Clean combined dataset and save cleaned CSV.
Usage: python src/preprocessing/data_cleaner.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


INFILE = Path('data/processed/combined.csv')
OUTFILE = Path('data/processed/cleaned.csv')


def main() -> None:
	if not INFILE.exists():
		raise RuntimeError('Missing data/processed/combined.csv. Run combine_datasets.py first.')

	df = pd.read_csv(INFILE)
	df.columns = [str(c).strip() for c in df.columns]

	numeric_cols = [c for c in ['distance_raw', 'signal_strength', 'temperature', 'distance_ref'] if c in df.columns]
	for col in numeric_cols:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	before = len(df)
	df = df.drop_duplicates()
	df = df.dropna(subset=numeric_cols)
	after = len(df)
	print(f'Cleaned rows: {before} -> {after}')

	OUTFILE.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(OUTFILE, index=False)
	print(f'Saved cleaned data to {OUTFILE}')


if __name__ == '__main__':
	main()
