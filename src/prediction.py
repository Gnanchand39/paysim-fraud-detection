"""Model loading and prediction for the Streamlit demo app.

Streamlit's caching decorators are used here (this is the one place the
architecture allows a Streamlit import outside app.py, since caching a
loaded model IS the "prediction module" concern, not UI). No layout/widget
code lives here.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/final_xgboost_fraud_detector.joblib"
METRICS_PATH = "results/final_model_metrics.json"
FEATURE_IMPORTANCE_PATH = "results/final_model_feature_importance.csv"


class ModelArtifactError(RuntimeError):
    """Raised when a required model artifact is missing or malformed --
    distinct from InputValidationError, since this is a system/setup
    problem, not a user-input problem."""


class PredictionError(RuntimeError):
    """Raised when the model produces malformed/unexpected output."""


@st.cache_resource(show_spinner="Loading fraud-detection model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise ModelArtifactError(
            f"Model artifact not found at '{MODEL_PATH}'. Please verify the project structure."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        raise ModelArtifactError(f"Metrics artifact not found at '{METRICS_PATH}'.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        raise ModelArtifactError(f"Feature importance artifact not found at '{FEATURE_IMPORTANCE_PATH}'.")
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def predict_fraud_probability(feature_row: pd.DataFrame, model=None) -> float:
    """feature_row: a single-row DataFrame with exactly the model's expected
    40 columns, in the expected order (see src/app_features.build_feature_vector).
    Returns the predicted fraud probability as a plain float in [0, 1].
    """
    if model is None:
        model = load_model()

    expected_features = list(getattr(model, "feature_names_in_", feature_row.columns))
    if list(feature_row.columns) != list(expected_features):
        raise PredictionError(
            f"Feature mismatch: model expects {len(expected_features)} features "
            f"({expected_features[:3]}...), got {len(feature_row.columns)} "
            f"({list(feature_row.columns)[:3]}...)."
        )

    try:
        proba = model.predict_proba(feature_row)
    except Exception as exc:
        raise PredictionError(f"Model prediction failed: {exc}") from exc

    if proba.ndim != 2 or proba.shape[0] != 1 or proba.shape[1] != 2:
        raise PredictionError(f"Unexpected predict_proba output shape: {proba.shape}")

    fraud_probability = float(proba[0, 1])
    if not (0.0 <= fraud_probability <= 1.0) or np.isnan(fraud_probability):
        raise PredictionError(f"Model returned an invalid probability: {fraud_probability}")

    return fraud_probability
