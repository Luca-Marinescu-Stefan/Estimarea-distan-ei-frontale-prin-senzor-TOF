"""Generate synthetic sample data for the TOF project.
Run: python src/data_acquisition/generate.py --rows 500 --out data/raw/sample_raw.csv
"""
import argparse
import csv
import random
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=500)
parser.add_argument("--out", type=str, default="data/raw/sample_raw.csv")
args = parser.parse_args()

start = datetime.utcnow()
with open(args.out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp","distance_raw","signal_strength","temperature","distance_ref"]) 
    for i in range(args.rows):
        ts = (start + timedelta(milliseconds=100*i)).isoformat()
        # realistic ranges: distance 50-2000 mm, signal 0-300, temp 20-40 C
        distance_ref = random.choice([100,200,500,1000,1500])
        noise = random.gauss(0, distance_ref*0.02)
        distance_raw = max(10, distance_ref + noise)
        signal = max(0, int(random.gauss(120,30)))
        temp = round(random.uniform(20,35),1)
        writer.writerow([ts, round(distance_raw,2), signal, temp, distance_ref])

print(f"Wrote {args.rows} rows to {args.out}")