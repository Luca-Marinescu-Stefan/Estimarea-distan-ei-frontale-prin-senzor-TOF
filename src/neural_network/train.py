"""Train script for sklearn (RandomForest) or Keras MLP. Expects CSVs from data_splitter.py"""
# -----------------------------------------------------------------------------
# Modul: src/neural_network/train.py
# Scop: antrenare model (sklearn sau Keras MLP) pe train/val.
# Input: data/train/X_train.csv și data/validation/X_val.csv.
# Output: model antrenat + fișiere de metrici/istoric.
# Utilizare: python src/neural_network/train.py --backend sklearn|keras
# Pași principali:
#   1) Încărcare și validare seturi.
#   2) Standardizare (dacă e necesar).
#   3) Antrenare + evaluare pe setul de validare.
# Dependențe: scikit-learn, tensorflow/keras (opțional).
# Parametri implicați: n_estimators, epochs, batch_size etc.
# Fișiere scrise: models/*, results/training_history.csv, results/hyperparameters.yaml.
# Observații: salvează scaleri în config/ dacă nu există.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train model (sklearn or Keras)')
    parser.add_argument('--backend', type=str, choices=['sklearn', 'keras'], default='sklearn')

    # sklearn params
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--random-state', type=int, default=42)

    # keras params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-units', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--reduce-lr', action='store_true')

    parser.add_argument('--model-out', type=str, default='')
    parser.add_argument('--history-out', type=str, default='results/training_history.csv')
    parser.add_argument('--name', type=str, default='baseline')
    return parser.parse_args()


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv('data/train/X_train.csv')
    val = pd.read_csv('data/validation/X_val.csv')
    for df_name, df in [('train', train), ('val', val)]:
        if 'label' not in df.columns:
            raise RuntimeError(
                f"Missing 'label' column in data/{df_name}/X_{'train' if df_name == 'train' else 'val'}.csv"
            )
    train = train.copy()
    val = val.copy()
    train['label'] = pd.to_numeric(train['label'], errors='coerce')
    val['label'] = pd.to_numeric(val['label'], errors='coerce')
    train = train.dropna(subset=['label'])
    val = val.dropna(subset=['label'])
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError('Empty train or validation set after dropping rows with missing labels')
    return train, val


def save_hyperparams(path: Path, params: dict) -> None:
    lines = [f"{k}: {v}" for k, v in params.items()]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _is_already_scaled(X: np.ndarray) -> bool:
    if X.size == 0:
        return False
    mean_abs = float(np.abs(X.mean(axis=0)).mean())
    std = X.std(axis=0)
    std_mean = float(np.mean(std)) if std.size else 0.0
    return mean_abs < 0.05 and 0.8 <= std_mean <= 1.2


def prepare_features(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    X_train_raw = train[features].fillna(0).values
    X_val_raw = val[features].fillna(0).values

    if _is_already_scaled(X_train_raw):
        X_train = X_train_raw
        X_val = X_val_raw
        scaled = True
    else:
        scaler_path = Path('config/preprocessing_params.pkl')
        if scaler_path.exists():
            payload = joblib.load(scaler_path)
            scaler = payload.get('scaler') if isinstance(payload, dict) else payload
        else:
            scaler = StandardScaler()
            scaler.fit(X_train_raw)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'scaler': scaler, 'features': features}, scaler_path)
            Path('config/preprocessing_params.json').write_text(
                json.dumps({'scaler': 'StandardScaler', 'features': features}, indent=2),
                encoding='utf-8'
            )

        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        scaled = True

    y_train = train['label'].astype(int).values
    y_val = val['label'].astype(int).values
    return X_train, y_train, X_val, y_val, scaled


def main() -> None:
    args = parse_args()
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)

    train, val = load_datasets()
    features = ['distance_raw', 'signal_strength', 'temperature']

    X_train, y_train, X_val, y_val, scaled = prepare_features(train, val, features)

    if args.backend == 'keras':
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except Exception:
            import keras
            from keras import layers

        num_classes = len(np.unique(y_train))
        inputs = keras.Input(shape=(X_train.shape[1],))
        x = layers.Dense(args.hidden_units, activation='relu')(inputs)
        if args.dropout > 0:
            x = layers.Dropout(args.dropout)(x)
        x = layers.Dense(args.hidden_units, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        callbacks = []
        if args.early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
        if args.reduce_lr:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5))

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        val_acc = float(accuracy_score(y_val, val_pred))
        val_f1 = float(f1_score(y_val, val_pred, average='macro'))

        model_out = Path(args.model_out or 'models/trained_model.h5')
        model.save(model_out)

        history_out = Path(args.history_out)
        history_out.parent.mkdir(parents=True, exist_ok=True)
        hist = history.history
        with open(history_out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
            for i in range(len(hist.get('loss', []))):
                writer.writerow([
                    i + 1,
                    hist['loss'][i],
                    hist.get('accuracy', [None])[i],
                    hist.get('val_loss', [None])[i],
                    hist.get('val_accuracy', [None])[i],
                ])

        hyperparams = {
            'backend': 'keras',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_units': args.hidden_units,
            'dropout': args.dropout,
            'early_stopping': args.early_stopping,
            'reduce_lr': args.reduce_lr,
            'input_scaled': scaled,
        }
        save_hyperparams(Path('results/hyperparameters.yaml'), hyperparams)

        print(
            f"Training finished. Model saved to {model_out}. Validation accuracy: {val_acc:.4f}, "
            f"F1-macro: {val_f1:.4f}"
        )
        return

    # sklearn backend (default)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    val_pred = clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_pred))
    val_f1 = float(f1_score(y_val, val_pred, average='macro'))

    model_out = Path(args.model_out or 'models/trained_model.joblib')
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)

    history_out = Path(args.history_out)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    with open(history_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'val_accuracy', 'val_f1_macro', 'train_time_sec'])
        writer.writerow([args.name, val_acc, val_f1, round(train_time, 4)])

    hyperparams = {
        'backend': 'sklearn',
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_state,
    }
    save_hyperparams(Path('results/hyperparameters.yaml'), hyperparams)

    print(
        f"Training finished. Model saved to {model_out}. Validation accuracy: {val_acc:.4f}, "
        f"F1-macro: {val_f1:.4f}"
    )


if __name__ == '__main__':
    main()