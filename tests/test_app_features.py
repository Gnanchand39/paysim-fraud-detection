"""Tests for src/app_features.py -- pure Python, no Streamlit, no retraining.

Run with: pytest tests/test_app_features.py -v
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.app_features import (
    ALWAYS_FORBIDDEN,
    AdvancedDemoInput,
    InputValidationError,
    NO_HISTORY_TIME_SENTINEL,
    QuickDemoInput,
    build_feature_vector,
    get_final_feature_columns,
    validate_advanced_input,
    validate_quick_input,
)

VALID_QUICK = QuickDemoInput(
    transaction_type="TRANSFER", amount=10000.0,
    sender_balance=20000.0, receiver_balance=0.0, simulated_step=200,
)


# --- 1 & 2: column count and canonical order ---

def test_feature_count_is_exactly_40():
    columns = get_final_feature_columns()
    assert len(columns) == 40


def test_feature_vector_matches_canonical_order():
    columns = list(get_final_feature_columns())
    row = build_feature_vector(VALID_QUICK)
    assert list(row.columns) == columns


# --- 3: excluded features never appear ---

def test_excluded_features_absent():
    row = build_feature_vector(VALID_QUICK)
    for forbidden in ALWAYS_FORBIDDEN:
        assert forbidden not in row.columns


# --- 4 & 5: no NaN / no infinite values ---

def test_no_nan_values():
    row = build_feature_vector(VALID_QUICK)
    assert not row.isnull().any().any()


def test_no_infinite_values():
    row = build_feature_vector(VALID_QUICK)
    assert np.isfinite(row.to_numpy(dtype=np.float64)).all()


# --- 6: Quick Demo no-history defaults are correct ---

def test_quick_demo_no_history_defaults():
    row = build_feature_vector(VALID_QUICK)
    assert row["prior_sender_transaction_count"].iloc[0] == 0
    assert row["prior_sender_total_amount"].iloc[0] == 0
    assert row["prior_sender_has_history"].iloc[0] == 0
    assert row["prior_receiver_transaction_count"].iloc[0] == 0
    assert row["prior_receiver_has_history"].iloc[0] == 0
    assert row["prior_sender_time_since_last_transaction"].iloc[0] == NO_HISTORY_TIME_SENTINEL
    assert row["prior_receiver_time_since_last_transaction"].iloc[0] == NO_HISTORY_TIME_SENTINEL
    assert row["sender_transactions_previous_1_step"].iloc[0] == 0
    assert row["sender_transactions_previous_24_steps"].iloc[0] == 0
    assert row["receiver_transactions_previous_24_steps"].iloc[0] == 0
    # Non-exposed fields always default too
    assert row["prior_sender_min_amount"].iloc[0] == 0
    assert row["prior_sender_amount_std"].iloc[0] == 0


# --- 7: Advanced historical inputs populate correctly ---

def test_advanced_demo_overrides_populate_correctly():
    advanced = AdvancedDemoInput(
        prior_sender_transaction_count=5,
        prior_sender_total_amount=50000.0,
        prior_receiver_transaction_count=12,
        prior_receiver_time_since_last_transaction=3,
    )
    row = build_feature_vector(VALID_QUICK, advanced)
    assert row["prior_sender_transaction_count"].iloc[0] == 5
    assert row["prior_sender_total_amount"].iloc[0] == 50000.0
    assert row["prior_receiver_transaction_count"].iloc[0] == 12
    assert row["prior_receiver_time_since_last_transaction"].iloc[0] == 3
    # has_history flags derive from the count fields
    assert row["prior_sender_has_history"].iloc[0] == 1
    assert row["prior_receiver_has_history"].iloc[0] == 1
    # Fields not overridden stay at their defaults
    assert row["prior_sender_mean_amount"].iloc[0] == 0
    assert row["sender_transactions_previous_1_step"].iloc[0] == 0


# --- 8: invalid transaction type ---

def test_invalid_transaction_type_raises():
    bad = QuickDemoInput(transaction_type="WIRE", amount=100.0, sender_balance=100.0,
                          receiver_balance=0.0, simulated_step=1)
    with pytest.raises(InputValidationError):
        validate_quick_input(bad)


# --- 9: negative amount ---

def test_negative_amount_raises():
    bad = QuickDemoInput(transaction_type="TRANSFER", amount=-1.0, sender_balance=100.0,
                          receiver_balance=0.0, simulated_step=1)
    with pytest.raises(InputValidationError):
        validate_quick_input(bad)


# --- 10: negative balance ---

def test_negative_sender_balance_raises():
    bad = QuickDemoInput(transaction_type="TRANSFER", amount=100.0, sender_balance=-5.0,
                          receiver_balance=0.0, simulated_step=1)
    with pytest.raises(InputValidationError):
        validate_quick_input(bad)


def test_negative_receiver_balance_raises():
    bad = QuickDemoInput(transaction_type="TRANSFER", amount=100.0, sender_balance=100.0,
                          receiver_balance=-5.0, simulated_step=1)
    with pytest.raises(InputValidationError):
        validate_quick_input(bad)


# --- 11: invalid simulated step ---

def test_simulated_step_below_range_raises():
    bad = QuickDemoInput(transaction_type="TRANSFER", amount=100.0, sender_balance=100.0,
                          receiver_balance=0.0, simulated_step=0)
    with pytest.raises(InputValidationError):
        validate_quick_input(bad)


def test_simulated_step_above_range_raises():
    bad = QuickDemoInput(transaction_type="TRANSFER", amount=100.0, sender_balance=100.0,
                          receiver_balance=0.0, simulated_step=744)
    with pytest.raises(InputValidationError):
        validate_quick_input(bad)


# --- 12: invalid historical count ---

def test_negative_historical_count_raises():
    bad_advanced = AdvancedDemoInput(prior_sender_transaction_count=-1)
    with pytest.raises(InputValidationError):
        validate_advanced_input(bad_advanced)


def test_invalid_time_since_last_raises():
    bad_advanced = AdvancedDemoInput(prior_sender_time_since_last_transaction=0)
    with pytest.raises(InputValidationError):
        validate_advanced_input(bad_advanced)


def test_time_since_last_sentinel_is_valid():
    ok_advanced = AdvancedDemoInput(prior_sender_time_since_last_transaction=-1)
    validated = validate_advanced_input(ok_advanced)
    assert validated.prior_sender_time_since_last_transaction == -1


# --- 13: feature vector remains numeric ---

def test_feature_vector_is_fully_numeric():
    row = build_feature_vector(VALID_QUICK)
    for col in row.columns:
        assert np.issubdtype(row[col].dtype, np.number), f"{col} is not numeric"


# --- Model regression-style smoke test (does not modify the model) ---

def test_model_loads_and_predicts_valid_probability():
    import joblib
    model = joblib.load("models/final_xgboost_fraud_detector.joblib")

    columns = list(get_final_feature_columns())
    test_df = pd.read_parquet("data/processed/test.parquet")
    sample_row = test_df[columns].iloc[[0]]

    assert list(sample_row.columns) == list(model.feature_names_in_)

    proba = model.predict_proba(sample_row)
    assert proba.shape == (1, 2)
    fraud_probability = float(proba[0, 1])
    assert 0.0 <= fraud_probability <= 1.0


def test_app_generated_feature_vector_is_accepted_by_model():
    import joblib
    model = joblib.load("models/final_xgboost_fraud_detector.joblib")
    row = build_feature_vector(VALID_QUICK)
    assert list(row.columns) == list(model.feature_names_in_)
    proba = model.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0, 1]) <= 1.0
