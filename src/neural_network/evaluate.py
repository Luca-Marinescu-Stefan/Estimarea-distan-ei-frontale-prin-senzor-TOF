"""Evaluate trained model on test set and save metrics + optional analysis."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='models/trained_model.joblib')
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--out-metrics', type=str, default='results/test_metrics.json')
    parser.add_argument('--confusion-path', type=str, default='docs/confusion_matrix_optimized.png')
    parser.add_argument('--error-analysis-path', type=str, default='results/error_analysis.json')
    return parser.parse_args()


def load_model(model_path: Path) -> tuple[Any, str]:
    if model_path.suffix == '.joblib' and model_path.exists():
        return joblib.load(model_path), 'sklearn'
    if model_path.suffix == '.h5' and model_path.exists():
        try:
            from tensorflow.keras.models import load_model
        except Exception:
            from keras.models import load_model
        return load_model(str(model_path)), 'keras'
    raise RuntimeError(f'Model not found or unsupported format: {model_path}')


def _is_already_scaled(X: np.ndarray) -> bool:
    if X.size == 0:
        return False
    mean_abs = float(np.abs(X.mean(axis=0)).mean())
    std = X.std(axis=0)
    std_mean = float(np.mean(std)) if std.size else 0.0
    return mean_abs < 0.05 and 0.8 <= std_mean <= 1.2


def load_test_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test = pd.read_csv('data/test/X_test.csv')
    if 'label' not in test.columns:
        raise RuntimeError('Missing label column in data/test/X_test.csv')
    test = test.dropna(subset=['label']).copy()
    if len(test) == 0:
        raise RuntimeError('No labeled test examples found in data/test/X_test.csv')
    test['label'] = pd.to_numeric(test['label'], errors='coerce')
    test = test.dropna(subset=['label'])
    features = ['distance_raw', 'signal_strength', 'temperature']
    X = test[features].fillna(0).values
    y = test['label'].astype(int).values
    return test, X, y


def compute_confidence(model_type: str, model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model_type == 'sklearn':
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            preds = np.argmax(proba, axis=1)
            confidence = proba.max(axis=1)
        else:
            preds = model.predict(X)
            confidence = np.full(shape=(len(preds),), fill_value=np.nan)
    else:
        proba = model.predict(X, verbose=0)
        preds = np.argmax(proba, axis=1)
        confidence = proba.max(axis=1)
    return preds, confidence


def save_confusion_matrix(cm: np.ndarray, labels: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix (Optimized)'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def maybe_scale_for_keras(X: np.ndarray) -> np.ndarray:
    if _is_already_scaled(X):
        return X
    scaler_path = Path('config/preprocessing_params.pkl')
    if scaler_path.exists():
        payload = joblib.load(scaler_path)
        scaler = payload.get('scaler') if isinstance(payload, dict) else payload
        return scaler.transform(X)
    return X


def main() -> None:
    args = parse_args()
    Path('results').mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    model, model_type = load_model(model_path)
    test_df, X_test, y_test = load_test_data()

    start = time.perf_counter()
    X_for_model = X_test
    if model_type == 'keras':
        X_for_model = maybe_scale_for_keras(X_test)

    preds, confidence = compute_confidence(model_type, model, X_for_model)
    latency_ms = (time.perf_counter() - start) * 1000.0
    latency_per_sample = latency_ms / max(len(X_test), 1)

    metrics = {
        'model': str(model_path),
        'test_accuracy': float(accuracy_score(y_test, preds)),
        'test_f1_macro': float(f1_score(y_test, preds, average='macro')),
        'test_precision_macro': float(precision_score(y_test, preds, average='macro', zero_division=0)),
        'test_recall_macro': float(recall_score(y_test, preds, average='macro', zero_division=0)),
        'false_negative_rate': None,
        'false_positive_rate': None,
        'inference_latency_ms_total': round(latency_ms, 4),
        'inference_latency_ms_per_sample': round(latency_per_sample, 6),
    }

    labels = sorted(pd.unique(pd.Series(y_test)))
    cm = confusion_matrix(y_test, preds, labels=labels)
    if cm.size > 0:
        if len(labels) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['false_negative_rate'] = float(fn / max(fn + tp, 1))
            metrics['false_positive_rate'] = float(fp / max(fp + tn, 1))

    with open(args.out_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    if args.detailed:
        save_confusion_matrix(cm, labels, Path(args.confusion_path))

        # Error analysis: top 5 misclassified by confidence
        test_df = test_df.reset_index(drop=False).rename(columns={'index': 'row_index'})
        error_mask = preds != y_test
        error_df = test_df.loc[error_mask].copy()
        error_df['predicted'] = preds[error_mask]
        error_df['confidence'] = confidence[error_mask]
        error_df = error_df.sort_values(by='confidence', ascending=False).head(5)
        errors = error_df[['row_index', 'label', 'predicted', 'confidence']].to_dict(orient='records')
        with open(args.error_analysis_path, 'w', encoding='utf-8') as f:
            json.dump({'top_errors': errors}, f, indent=2)

    print(f"Evaluation complete. Metrics written to {args.out_metrics}")


if __name__ == '__main__':
    main()
