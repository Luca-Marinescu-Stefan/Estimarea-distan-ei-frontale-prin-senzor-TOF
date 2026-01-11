"""Evaluate trained model on test set and save metrics."""
import pandas as pd
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
import os
from pathlib import Path

model_path = Path('models/trained_model.h5')
if not model_path.exists():
    raise RuntimeError('Trained model not found at models/trained_model.h5. Run training first.')
model = load_model(model_path)

test = pd.read_csv('data/test/X_test.csv')
FEATURES = ['distance_raw','signal_strength','temperature']
X_test = test[FEATURES].fillna(0).values
test['label'] = pd.to_numeric(test.get('label'), errors='coerce')
test = test.dropna(subset=['label'])
if len(test) == 0:
    raise RuntimeError('No labeled test examples found in data/test/X_test.csv')
y_test = test['label'].astype(int).values

pred = model.predict(X_test)
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
import joblib

Path('results').mkdir(parents=True, exist_ok=True)

# Prefer scikit-learn joblib model if present
model_path_joblib = Path('models/trained_model.joblib')
model_path_h5 = Path('models/trained_model.h5')

if model_path_joblib.exists():
    model = joblib.load(model_path_joblib)
    model_type = 'sklearn'
elif model_path_h5.exists():
    try:
        from tensorflow.keras.models import load_model
        model = load_model(str(model_path_h5))
        model_type = 'keras'
    except Exception as e:
        raise RuntimeError(f'Failed to load Keras model: {e}')
else:
    raise RuntimeError('Trained model not found: models/trained_model.joblib or models/trained_model.h5')

X_test = pd.read_csv('data/test/X_test.csv')
if 'label' not in X_test.columns:
    raise RuntimeError('Missing label column in data/test/X_test.csv')

X_test = X_test.dropna(subset=['label']).copy()
y_test = X_test['label'].astype(int).values
FEATURES = ['distance_raw','signal_strength','temperature']
X = X_test[FEATURES].fillna(0).values

if model_type == 'sklearn':
    preds = model.predict(X)
    acc = float(accuracy_score(y_test, preds))
    metrics = {'accuracy': acc}
else:
    loss, acc = model.evaluate(X, y_test, verbose=0)
    metrics = {'loss': float(loss), 'accuracy': float(acc)}

with open('results/test_metrics.json','w') as f:
    json.dump(metrics, f, indent=2)

print('Evaluation complete. Results written to results/test_metrics.json')
