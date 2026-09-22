"""Pure-Python input validation and feature construction for the Streamlit
demo app. No Streamlit imports here -- everything is testable in isolation.

This module turns a small set of user-facing inputs into the exact 40-column
feature vector the final model (models/final_xgboost_fraud_detector.joblib)
expects, using the project's OWN feature-engineering code as the source of
truth for column names/order -- never a hand-typed second list.

NO-HISTORY DEFAULTS
--------------------
For a genuinely first-ever transaction, src/features.py produces:
  - counts, sums, means, maxes, mins, transfer/cashout counts, distinct
    counterparty counts, and all velocity features -> 0
  - *_has_history flags -> 0
  - *_time_since_last_transaction -> -1 (sentinel; verified directly against
    src/features.py's create_sender_history_features / create_receiver_history_features,
    which use `np.where(has_history, step - prior_step, -1)`)
These exact values are reused below, not re-derived or guessed.

QUICK DEMO vs ADVANCED DEMO
-----------------------------
Quick Demo: every historical/velocity feature uses the no-history defaults
above. Advanced Demo: a curated subset of 19 historical/velocity fields may
be overridden with hypothetical demo values; four historical fields
(`prior_sender_min_amount`, `prior_sender_amount_std`, `prior_sender_has_history`,
`prior_receiver_has_history`) are never exposed in the UI -- the two "has_history"
flags are derived from the corresponding transaction-count field, and the
other two always stay at their no-history defaults regardless of Advanced
Demo overrides (a disclosed limitation: Advanced Demo is a curated, partial
hypothetical, not a fully self-consistent synthetic history).
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from src.features import TRANSACTION_TYPES, build_feature_dataset, get_model_feature_columns
from src.model_utils import exclude_features

ARTIFACT_FEATURES_EXCLUDED = ["amount_to_sender_balance", "amount_exceeds_sender_balance"]
ALWAYS_FORBIDDEN = [
    "amount_to_sender_balance", "amount_exceeds_sender_balance",
    "newbalanceOrig", "newbalanceDest", "isFlaggedFraud", "isFraud",
    "nameOrig", "nameDest",
]

NO_HISTORY_TIME_SENTINEL = -1  # matches src/features.py exactly

# Curated Advanced Demo fields (19), each mapped straight to a final feature name.
ADVANCED_HISTORICAL_FIELDS = [
    "prior_sender_transaction_count",
    "prior_sender_total_amount",
    "prior_sender_mean_amount",
    "prior_sender_max_amount",
    "prior_sender_time_since_last_transaction",
    "prior_sender_transfer_count",
    "prior_sender_cashout_count",
    "prior_sender_distinct_destinations",
    "prior_receiver_transaction_count",
    "prior_receiver_total_amount",
    "prior_receiver_mean_amount",
    "prior_receiver_max_amount",
    "prior_receiver_time_since_last_transaction",
    "prior_receiver_distinct_senders",
    "sender_transactions_previous_1_step",
    "sender_transactions_previous_6_steps",
    "sender_transactions_previous_24_steps",
    "sender_amount_previous_24_steps",
    "receiver_transactions_previous_24_steps",
]

# Historical/velocity fields NOT exposed in either UI mode -- always default.
_NON_EXPOSED_ALWAYS_DEFAULT = ["prior_sender_min_amount", "prior_sender_amount_std"]


class InputValidationError(ValueError):
    """Raised when user-supplied input fails validation, BEFORE any model call."""


@dataclass
class QuickDemoInput:
    transaction_type: str
    amount: float
    sender_balance: float
    receiver_balance: float
    simulated_step: int


@dataclass
class AdvancedDemoInput:
    """All fields optional; any left as None fall back to the no-history default."""
    prior_sender_transaction_count: Optional[float] = None
    prior_sender_total_amount: Optional[float] = None
    prior_sender_mean_amount: Optional[float] = None
    prior_sender_max_amount: Optional[float] = None
    prior_sender_time_since_last_transaction: Optional[float] = None
    prior_sender_transfer_count: Optional[float] = None
    prior_sender_cashout_count: Optional[float] = None
    prior_sender_distinct_destinations: Optional[float] = None
    prior_receiver_transaction_count: Optional[float] = None
    prior_receiver_total_amount: Optional[float] = None
    prior_receiver_mean_amount: Optional[float] = None
    prior_receiver_max_amount: Optional[float] = None
    prior_receiver_time_since_last_transaction: Optional[float] = None
    prior_receiver_distinct_senders: Optional[float] = None
    sender_transactions_previous_1_step: Optional[float] = None
    sender_transactions_previous_6_steps: Optional[float] = None
    sender_transactions_previous_24_steps: Optional[float] = None
    sender_amount_previous_24_steps: Optional[float] = None
    receiver_transactions_previous_24_steps: Optional[float] = None


@lru_cache(maxsize=1)
def get_final_feature_columns() -> tuple:
    """The exact 40 column names, in the exact order the final model expects
    -- derived by running a tiny seed transaction through the project's own
    `build_feature_dataset()` pipeline, never a hand-typed second list.

    Cached (lru_cache) because this is a pure, argument-free function that
    always returns the same result -- the underlying seed-dataframe pipeline
    run (~26ms) is otherwise repeated on every single call for no benefit.
    Returns a tuple (hashable, required for lru_cache); callers that need a
    list can wrap with list(...).
    """
    seed = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [100.0],
        "nameOrig": ["SEED_ORIG"], "nameDest": ["SEED_DEST"],
        "oldbalanceOrg": [1000.0], "newbalanceOrig": [900.0],
        "oldbalanceDest": [0.0], "newbalanceDest": [100.0],
        "isFraud": [0], "isFlaggedFraud": [0],
    })
    feature_df = build_feature_dataset(seed)
    all_42 = get_model_feature_columns(feature_df)
    return tuple(exclude_features(all_42, ARTIFACT_FEATURES_EXCLUDED))


def _validate_numeric(name: str, value, allow_negative: bool = False):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise InputValidationError(f"{name} must be numeric (got {value!r}).")
    if np.isnan(numeric_value) or np.isinf(numeric_value):
        raise InputValidationError(f"{name} must be a finite number (got {value!r}).")
    if not allow_negative and numeric_value < 0:
        raise InputValidationError(f"{name} cannot be negative (got {numeric_value}).")
    return numeric_value


def validate_quick_input(raw: QuickDemoInput) -> QuickDemoInput:
    if raw.transaction_type not in TRANSACTION_TYPES:
        raise InputValidationError(
            f"Transaction type must be one of {TRANSACTION_TYPES} (got {raw.transaction_type!r})."
        )
    amount = _validate_numeric("Amount", raw.amount)
    sender_balance = _validate_numeric("Sender previous balance", raw.sender_balance)
    receiver_balance = _validate_numeric("Receiver previous balance", raw.receiver_balance)

    try:
        step = int(raw.simulated_step)
    except (TypeError, ValueError):
        raise InputValidationError(f"Simulated step must be an integer (got {raw.simulated_step!r}).")
    if step != raw.simulated_step and not float(raw.simulated_step).is_integer():
        raise InputValidationError(f"Simulated step must be a whole number (got {raw.simulated_step!r}).")
    if not (1 <= step <= 743):
        raise InputValidationError(f"Simulated step must be between 1 and 743 (got {step}).")

    return QuickDemoInput(
        transaction_type=raw.transaction_type, amount=amount,
        sender_balance=sender_balance, receiver_balance=receiver_balance,
        simulated_step=step,
    )


def _validate_time_since_last(name: str, value) -> float:
    numeric_value = _validate_numeric(name, value, allow_negative=True)
    if numeric_value != NO_HISTORY_TIME_SENTINEL and numeric_value < 1:
        raise InputValidationError(
            f"{name} must be -1 (no prior history) or >= 1 (steps since last transaction); got {numeric_value}."
        )
    return numeric_value


def validate_advanced_input(raw: Optional[AdvancedDemoInput]) -> AdvancedDemoInput:
    if raw is None:
        return AdvancedDemoInput()
    validated = {}
    for field_name in ADVANCED_HISTORICAL_FIELDS:
        value = getattr(raw, field_name)
        if value is None:
            validated[field_name] = None
            continue
        if field_name.endswith("time_since_last_transaction"):
            validated[field_name] = _validate_time_since_last(field_name, value)
        else:
            validated[field_name] = _validate_numeric(field_name, value)
    return AdvancedDemoInput(**validated)


def build_feature_values(quick: QuickDemoInput, advanced: Optional[AdvancedDemoInput] = None) -> dict:
    """Map validated Quick/Advanced input to a {feature_name: value} dict
    covering all 40 final model features. Historical/velocity fields not
    provided in `advanced` use the exact no-history defaults from
    src/features.py."""
    advanced = advanced or AdvancedDemoInput()
    values = {}

    # --- Transaction features ---
    values["amount"] = quick.amount
    values["log_amount"] = float(np.log1p(quick.amount))
    for t in TRANSACTION_TYPES:
        values[f"type_{t}"] = 1 if quick.transaction_type == t else 0
    values["is_transfer"] = 1 if quick.transaction_type == "TRANSFER" else 0
    values["is_cash_out"] = 1 if quick.transaction_type == "CASH_OUT" else 0
    values["is_transfer_or_cash_out"] = int(bool(values["is_transfer"] or values["is_cash_out"]))

    # --- Time features ---
    values["simulated_hour"] = quick.simulated_step
    values["simulated_day"] = (quick.simulated_step - 1) // 24
    values["hour_of_day"] = (quick.simulated_step - 1) % 24

    # --- Pre-transaction balance features (excludes amount_to/exceeds_sender_balance) ---
    values["oldbalanceOrg"] = quick.sender_balance
    values["oldbalanceDest"] = quick.receiver_balance
    values["sender_balance_zero"] = 1 if quick.sender_balance == 0 else 0
    values["destination_balance_zero"] = 1 if quick.receiver_balance == 0 else 0

    # --- Historical sender/receiver + velocity features ---
    def pick(field_name, default):
        v = getattr(advanced, field_name, None)
        return default if v is None else v

    for field_name in ADVANCED_HISTORICAL_FIELDS:
        values[field_name] = pick(field_name, 0 if "time_since_last" not in field_name else NO_HISTORY_TIME_SENTINEL)

    # Non-exposed historical fields: always the no-history default.
    values["prior_sender_min_amount"] = 0
    values["prior_sender_amount_std"] = 0

    # Derived has_history flags (consistent with src/features.py: count > 0).
    values["prior_sender_has_history"] = 1 if values["prior_sender_transaction_count"] > 0 else 0
    values["prior_receiver_has_history"] = 1 if values["prior_receiver_transaction_count"] > 0 else 0

    return values


def build_feature_vector(quick: QuickDemoInput, advanced: Optional[AdvancedDemoInput] = None) -> pd.DataFrame:
    """Validated single-row DataFrame with exactly the 40 canonical columns,
    in canonical order, ready for `model.predict_proba()`."""
    validated_quick = validate_quick_input(quick)
    validated_advanced = validate_advanced_input(advanced)

    values = build_feature_values(validated_quick, validated_advanced)
    columns = list(get_final_feature_columns())

    row = pd.DataFrame([{col: values[col] for col in columns}], columns=columns)

    for forbidden in ALWAYS_FORBIDDEN:
        assert forbidden not in row.columns, f"Forbidden feature '{forbidden}' leaked into the feature vector."
    assert list(row.columns) == columns, "Feature vector column order does not match the canonical order."
    assert row.shape == (1, 40), f"Expected shape (1, 40), got {row.shape}."
    assert not row.isnull().any().any(), "Feature vector contains NaN values."
    assert np.isfinite(row.to_numpy(dtype=np.float64)).all(), "Feature vector contains infinite values."
    for col in row.columns:
        assert np.issubdtype(row[col].dtype, np.number), f"Column '{col}' is not numeric."

    return row
