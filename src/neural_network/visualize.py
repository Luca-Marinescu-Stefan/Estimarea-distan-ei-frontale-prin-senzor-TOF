"""Generate final plots for optimization and results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_f1(experiments: pd.DataFrame) -> None:
    Path('docs/optimization').mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(experiments['experiment'], experiments['val_accuracy'], color='#4C78A8')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Comparison')
    plt.tight_layout()
    plt.savefig('docs/optimization/accuracy_comparison.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(experiments['experiment'], experiments['val_f1_macro'], color='#F58518')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Validation F1-macro')
    plt.title('F1-score Comparison')
    plt.tight_layout()
    plt.savefig('docs/optimization/f1_comparison.png', dpi=150)
    plt.close()


def plot_learning_curve(history_path: Path) -> None:
    if not history_path.exists():
        return
    df = pd.read_csv(history_path)
    Path('docs/results').mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if 'val_accuracy' in df.columns:
        plt.plot(df['val_accuracy'], label='val_accuracy')
    if 'val_f1_macro' in df.columns:
        plt.plot(df['val_f1_macro'], label='val_f1_macro')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Learning Curves (Final Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/results/learning_curves_final.png', dpi=150)
    plt.close()

    # Etapa 5 compatibility
    plt.figure(figsize=(6, 4))
    if 'loss' in df.columns:
        plt.plot(df['loss'], label='loss')
    if 'val_loss' in df.columns:
        plt.plot(df['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/loss_curve.png', dpi=150)
    plt.close()


def plot_metrics_evolution(baseline: dict, final: dict) -> None:
    Path('docs/results').mkdir(parents=True, exist_ok=True)
    stages = ['Etapa 5', 'Etapa 6']
    acc = [baseline.get('accuracy') or baseline.get('test_accuracy'), final.get('test_accuracy')]
    f1 = [baseline.get('test_f1_macro'), final.get('test_f1_macro')]

    plt.figure(figsize=(6, 4))
    plt.plot(stages, acc, marker='o', label='Accuracy')
    plt.plot(stages, f1, marker='o', label='F1-macro')
    plt.ylabel('Score')
    plt.title('Metrics Evolution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/results/metrics_evolution.png', dpi=150)
    plt.close()


def plot_example_predictions(test_path: Path, predictions_path: Path) -> None:
    if not test_path.exists() or not predictions_path.exists():
        return
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_json(predictions_path)
    Path('docs/results').mkdir(parents=True, exist_ok=True)

    sample = test_df.head(9).copy()
    sample['predicted'] = pred_df.get('predicted', pd.Series([None] * len(sample)))
    sample['confidence'] = pred_df.get('confidence', pd.Series([None] * len(sample)))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    table = ax.table(
        cellText=sample[['label', 'predicted', 'confidence']].values,
        colLabels=['True', 'Pred', 'Conf'],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig('docs/results/example_predictions.png', dpi=150)
    plt.close()


def main() -> None:
    experiments_path = Path('results/optimization_experiments.csv')
    if experiments_path.exists():
        experiments = pd.read_csv(experiments_path)
        plot_accuracy_f1(experiments)

    plot_learning_curve(Path('results/training_history.csv'))

    baseline_path = Path('results/test_metrics.json')
    final_path = Path('results/final_metrics.json')
    if baseline_path.exists() and final_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        final = json.loads(final_path.read_text(encoding='utf-8'))
        plot_metrics_evolution(baseline, final)

    predictions_path = Path('results/predictions_sample.json')
    plot_example_predictions(Path('data/test/X_test.csv'), predictions_path)


if __name__ == '__main__':
    main()
