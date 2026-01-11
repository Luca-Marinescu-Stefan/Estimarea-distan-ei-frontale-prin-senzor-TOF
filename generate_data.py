#!/usr/bin/env python
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure directory exists
Path('data/generated').mkdir(parents=True, exist_ok=True)

rows = 800
out_file = 'data/generated/sample_800.csv'

start = datetime.utcnow()
with open(out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp','distance_raw','signal_strength','temperature','distance_ref']) 
    for i in range(rows):
        ts = (start + timedelta(milliseconds=100*i)).isoformat()
        distance_ref = random.choice([100,200,500,1000,1500])
        noise = random.gauss(0, distance_ref*0.02)
        distance_raw = max(10, distance_ref + noise)
        signal = max(0, int(random.gauss(120,30)))
        temp = round(random.uniform(20,35),1)
        writer.writerow([ts, round(distance_raw,2), signal, temp, distance_ref])

print(f"Generated {rows} rows in {out_file}")
sys.exit(0)
