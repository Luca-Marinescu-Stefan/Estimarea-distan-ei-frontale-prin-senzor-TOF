"""Train script (demo) using scikit-learn (RandomForest). Expects CSVs from data_splitter.py"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import csv

outdir = Path('models')
outdir.mkdir(parents=True, exist_ok=True)

# ensure results dir exists
Path('results').mkdir(parents=True, exist_ok=True)

train = pd.read_csv('data/train/X_train.csv')
val = pd.read_csv('data/validation/X_val.csv')

# features: distance_raw,signal_strength,temperature -> simple selection
FEATURES = ['distance_raw','signal_strength','temperature']

# Validate label column
for df_name, df in [('train', train), ('val', val)]:
    if 'label' not in df.columns:
        raise RuntimeError(f"Missing 'label' column in data/{df_name}/X_{'train' if df_name=='train' else 'val'}.csv")

train = train.copy()
val = val.copy()
train['label'] = pd.to_numeric(train['label'], errors='coerce')
val['label'] = pd.to_numeric(val['label'], errors='coerce')

# Drop rows with NaN labels
before_train = len(train)
before_val = len(val)
train = train.dropna(subset=['label'])
val = val.dropna(subset=['label'])
after_train = len(train)
after_val = len(val)
print(f"Dropped {before_train-after_train} rows with NaN label from train, {before_val-after_val} from val")

if len(train) == 0 or len(val) == 0:
    raise RuntimeError('Empty train or validation set after dropping rows with missing labels')

X_train = train[FEATURES].fillna(0).values
y_train = train['label'].astype(int).values
X_val = val[FEATURES].fillna(0).values
y_val = val['label'].astype(int).values

# Train a RandomForest classifier (fast and reliable for demo)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Validate
val_pred = clf.predict(X_val)
val_acc = float(accuracy_score(y_val, val_pred))

# Save model
joblib.dump(clf, outdir / 'trained_model.joblib')

# Save a small training summary
with open('results/training_history.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['val_accuracy'])
    writer.writerow([val_acc])

print(f'Training finished. Model saved to {outdir/"trained_model.joblib"}. Validation accuracy: {val_acc:.4f}')