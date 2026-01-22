"""Simple Streamlit UI to upload a CSV row and run inference with optimized model."""
# -----------------------------------------------------------------------------
# Modul: src/app/main.py
# Scop: UI Streamlit pentru inferență rapidă pe CSV.
# Input: fișier CSV cu distance_raw, signal_strength, temperature.
# Output: predicții, confidence și decizie (NORMAL/ALERT/CONFIDENCE_CHECK).
# Utilizare: python -m streamlit run src/app/main.py
# Pași principali:
#   1) Încărcare model (optimized/trained).
#   2) Încărcare date și preprocesare minimă.
#   3) Predicție + scor de încredere.
# Dependențe: streamlit, pandas, numpy, joblib.
# Parametri implicați: threshold=0.35, confidence_min=0.60.
# Fișiere citite: models/*.joblib sau models/*.h5.
# Fișiere scrise: nu scrie; doar afișează în UI.
# Observații: potrivit pentru demo și capturi de ecran.
# Checklist UI:
# - CSV cu 3 coloane obligatorii.
# - Model disponibil în models/.
# - Verifică bara de confidence.
# - Salvează screenshot pentru raport.
# -----------------------------------------------------------------------------
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.title('SIA - Demo Inference (Optimized)')

uploaded = st.file_uploader('Upload CSV with columns: distance_raw,signal_strength,temperature', type=['csv'])

def load_any_model():
    if Path('models/optimized_model.joblib').exists():
        return joblib.load('models/optimized_model.joblib'), 'sklearn', 'optimized_model.joblib'
    if Path('models/trained_model.joblib').exists():
        return joblib.load('models/trained_model.joblib'), 'sklearn', 'trained_model.joblib'
    if Path('models/optimized_model.h5').exists():
        from tensorflow.keras.models import load_model
        return load_model('models/optimized_model.h5'), 'keras', 'optimized_model.h5'
    if Path('models/trained_model.h5').exists():
        from tensorflow.keras.models import load_model
        return load_model('models/trained_model.h5'), 'keras', 'trained_model.h5'
    raise RuntimeError('No model available. Run training/optimization first.')


def predict_with_confidence(model, model_type: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model_type == 'sklearn':
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            return np.argmax(proba, axis=1), proba.max(axis=1)
        preds = model.predict(X)
        return preds, np.full(shape=(len(preds),), fill_value=np.nan)
    proba = model.predict(X, verbose=0)
    return np.argmax(proba, axis=1), proba.max(axis=1)


if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('Preview')
    st.write(df.head())

    try:
        model, model_type, model_name = load_any_model()
        st.success(f'Loaded {model_name}')
    except Exception as e:
        st.error(str(e))
        st.stop()

    features = ['distance_raw', 'signal_strength', 'temperature']
    X = df[features].fillna(0).values
    preds, conf = predict_with_confidence(model, model_type, X)
    df['pred_class'] = preds
    df['confidence'] = conf

    # Simple decision logic (State Machine-like)
    threshold = 0.35
    confidence_min = 0.60
    alert_mask = (df['pred_class'] >= 1) & (df['confidence'] >= threshold)
    df['decision'] = np.where(
        df['confidence'] < confidence_min,
        'CONFIDENCE_CHECK',
        np.where(alert_mask, 'ALERT', 'NORMAL')
    )

    st.write(df)
    st.progress(float(np.nanmean(df['confidence'])) if len(df) else 0.0, text='Confidence (avg)')
    st.markdown('Save screenshot to docs/screenshots/inference_optimized.png')
else:
    st.info('Upload a CSV to run inference')
