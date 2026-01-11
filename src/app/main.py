"""Simple Streamlit UI to upload a CSV row and run inference with trained model."""
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

st.title('SIA - Demo Inference')

uploaded = st.file_uploader('Upload CSV with columns: distance_raw,signal_strength,temperature', type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('Preview')
    st.write(df.head())
    try:
        model = load_model('models/trained_model.h5')
        st.success('Loaded trained_model.h5')
    except Exception as e:
        st.warning('Could not load trained model, loading untrained_model.h5 if available')
        try:
            model = load_model('models/untrained_model.h5')
        except Exception:
            st.error('No model available. Run training first.')
            st.stop()

    FEATURES = ['distance_raw','signal_strength','temperature']
    X = df[FEATURES].fillna(0).values
    preds = model.predict(X)
    import numpy as np
    classes = np.argmax(preds, axis=1)
    df['pred_class'] = classes
    df['confidence'] = preds.max(axis=1)
    st.write(df)
    st.markdown('Save screenshot to docs/screenshots/inference_real.png')
else:
    st.info('Upload a CSV to run inference')
