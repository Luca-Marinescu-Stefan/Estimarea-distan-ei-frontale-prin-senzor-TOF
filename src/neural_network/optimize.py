"""Run hyperparameter experiments and save optimized model + reports."""
# -----------------------------------------------------------------------------
# Modul: src/neural_network/optimize.py
# Scop: rulare experimente de optimizare și salvare model final.
# Input: data/train/X_train.csv, data/validation/X_val.csv, data/test/X_test.csv.
# Output: models/optimized_model.joblib + results/final_metrics.json.
# Utilizare: python src/neural_network/optimize.py
# Pași principali:
#   1) Rulează experimente cu parametri diferiți.
#   2) Selectează cel mai bun model după F1.
#   3) Evaluează pe test și salvează raport complet.
# Dependențe: scikit-learn, pandas, numpy, matplotlib.
# Parametri implicați: n_estimators, max_depth, min_samples_*.
# Fișiere scrise: results/optimization_experiments.csv, config/optimized_config.yaml.
# Observații: generează și confusion matrix + error_analysis.json.
# -----------------------------------------------------------------------------
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


FEATURES = ['distance_raw', 'signal_strength', 'temperature']


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise RuntimeError(f"Missing label column in {path}")
    df = df.dropna(subset=['label']).copy()
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    return df


def train_and_eval(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    X_train = train_df[FEATURES].fillna(0).values
    y_train = train_df['label'].astype(int).values
    X_val = val_df[FEATURES].fillna(0).values
    y_val = val_df['label'].astype(int).values

    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    preds = clf.predict(X_val)
    acc = float(accuracy_score(y_val, preds))
    f1 = float(f1_score(y_val, preds, average='macro'))

    return {
        'model': clf,
        'accuracy': acc,
        'f1_macro': f1,
        'train_time_sec': round(train_time, 4),
    }


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


def main() -> None:
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('config').mkdir(parents=True, exist_ok=True)
    Path('docs').mkdir(parents=True, exist_ok=True)

    train_df = load_split('data/train/X_train.csv')
    val_df = load_split('data/validation/X_val.csv')
    test_df = load_split('data/test/X_test.csv')

    experiments = [
        {
            'name': 'baseline',
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'name': 'exp1_more_trees',
            'n_estimators': 400,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'name': 'exp2_depth_10',
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'name': 'exp3_min_leaf_2',
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
        },
        {
            'name': 'exp4_split_4',
            'n_estimators': 250,
            'max_depth': 14,
            'min_samples_split': 4,
            'min_samples_leaf': 1,
        },
    ]

    results = []
    best = None
    for exp in experiments:
        out = train_and_eval(train_df, val_df, exp)
        results.append({
            'experiment': exp['name'],
            'n_estimators': exp['n_estimators'],
            'max_depth': exp['max_depth'],
            'min_samples_split': exp['min_samples_split'],
            'min_samples_leaf': exp['min_samples_leaf'],
            'val_accuracy': out['accuracy'],
            'val_f1_macro': out['f1_macro'],
            'train_time_sec': out['train_time_sec'],
        })
        if best is None or out['f1_macro'] > best['f1_macro']:
            best = {**exp, **out}

    results_path = Path('results/optimization_experiments.csv')
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Train final optimized model on train+val
    full_train = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    X_full = full_train[FEATURES].fillna(0).values
    y_full = full_train['label'].astype(int).values

    optimized = RandomForestClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        min_samples_split=best['min_samples_split'],
        min_samples_leaf=best['min_samples_leaf'],
        random_state=42,
        n_jobs=-1,
    )
    optimized.fit(X_full, y_full)

    optimized_path = Path('models/optimized_model.joblib')
    joblib.dump(optimized, optimized_path)

    # Save optimized config (YAML-like)
    config_path = Path('config/optimized_config.yaml')
    config_content = (
        f"model_type: RandomForestClassifier\n"
        f"features: {FEATURES}\n"
        f"n_estimators: {best['n_estimators']}\n"
        f"max_depth: {best['max_depth']}\n"
        f"min_samples_split: {best['min_samples_split']}\n"
        f"min_samples_leaf: {best['min_samples_leaf']}\n"
        f"threshold_alert: 0.35\n"
        f"confidence_min: 0.60\n"
    )
    config_path.write_text(config_content, encoding='utf-8')

    # Final evaluation on test set
    X_test = test_df[FEATURES].fillna(0).values
    y_test = test_df['label'].astype(int).values
    start = time.perf_counter()
    preds = optimized.predict(X_test)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if hasattr(optimized, 'predict_proba'):
        proba = optimized.predict_proba(X_test)
        confidence = proba.max(axis=1)
    else:
        confidence = np.full(shape=(len(preds),), fill_value=np.nan)

    metrics = {
        'model': str(optimized_path),
        'test_accuracy': float(accuracy_score(y_test, preds)),
        'test_f1_macro': float(f1_score(y_test, preds, average='macro')),
        'test_precision_macro': float(precision_score(y_test, preds, average='macro', zero_division=0)),
        'test_recall_macro': float(recall_score(y_test, preds, average='macro', zero_division=0)),
        'false_negative_rate': None,
        'false_positive_rate': None,
        'inference_latency_ms_total': round(latency_ms, 4),
        'inference_latency_ms_per_sample': round(latency_ms / max(len(X_test), 1), 6),
    }

    labels = sorted(pd.unique(pd.Series(y_test)))
    cm = confusion_matrix(y_test, preds, labels=labels)
    if cm.size > 0 and len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['false_negative_rate'] = float(fn / max(fn + tp, 1))
        metrics['false_positive_rate'] = float(fp / max(fp + tn, 1))

    baseline_metrics_path = Path('results/test_metrics.json')
    if baseline_metrics_path.exists():
        try:
            baseline = json.loads(baseline_metrics_path.read_text(encoding='utf-8'))
            base_acc = baseline.get('accuracy') or baseline.get('test_accuracy')
            base_f1 = baseline.get('test_f1_macro')
            if base_acc is not None:
                metrics['improvement_vs_baseline'] = {
                    'accuracy': f"{(metrics['test_accuracy'] - base_acc) * 100:.2f}%",
                    'f1_score': f"{(metrics['test_f1_macro'] - (base_f1 or 0)) * 100:.2f}%",
                    'latency': 'n/a',
                }
        except Exception:
            pass

    final_metrics_path = Path('results/final_metrics.json')
    final_metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    save_confusion_matrix(cm, labels, Path('docs/confusion_matrix_optimized.png'))

    # Error analysis
    errors = []
    mismatch = np.where(preds != y_test)[0]
    if len(mismatch) > 0:
        idxs = mismatch[: min(5, len(mismatch))]
        for idx in idxs:
            errors.append({
                'index': int(idx),
                'true_label': int(y_test[idx]),
                'predicted': int(preds[idx]),
                'confidence': float(confidence[idx]) if not np.isnan(confidence[idx]) else None,
            })
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('results/error_analysis.json').write_text(
        json.dumps({'top_errors': errors}, indent=2),
        encoding='utf-8'
    )

    # Sample predictions for visualization
    sample_preds = {
        'predicted': [int(x) for x in preds[:9]],
        'confidence': [float(x) if not np.isnan(x) else None for x in confidence[:9]],
    }
    Path('results/predictions_sample.json').write_text(
        json.dumps(sample_preds, indent=2),
        encoding='utf-8'
    )

    print('Optimization complete.')
    print(f'Optimized model: {optimized_path}')
    print(f'Experiments: {results_path}')
    print('Final metrics saved to results/final_metrics.json')


if __name__ == '__main__':
    main()
