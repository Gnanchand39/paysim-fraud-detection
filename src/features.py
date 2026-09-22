"""Causal, leakage-safe feature engineering for the PaySim fraud dataset.

PREDICTION POINT
----------------
Every feature here is constructed as if it had to be computed the instant a
transaction is *requested*, before it is processed. Concretely:

Available at decision time (used freely):
    step, type, amount, nameOrig, nameDest, oldbalanceOrg, oldbalanceDest,
    and any aggregate of a customer's STRICTLY EARLIER transactions.

NOT available at decision time for the CURRENT transaction (never used):
    newbalanceOrig, newbalanceDest  (these only exist after the transaction
    is applied), isFlaggedFraud (a rule output, not a transaction attribute),
    isFraud (the target).

ORDERING / CAUSALITY DESIGN DECISION
-------------------------------------
The raw file is sorted by `step`, but rows that share the same `step` have no
reliable sub-step order (verified in the EDA/feature notebook: thousands of
(nameOrig, step) and (nameDest, step) pairs collide). Historical features
therefore only ever look at STRICTLY EARLIER steps -- never at "the previous
row" -- so two transactions sharing a step always see identical history
computed purely from earlier steps. This trades a small amount of granularity
for a causality guarantee that does not depend on an assumption we cannot
verify.

PERFORMANCE DESIGN
-------------------
Historical/velocity features are built by:
  1. Aggregating the raw transactions to one row per (id, step) using only
     vectorized pandas groupby aggregations (size/sum/max/min - no Python
     lambdas, which do not scale to millions of groups).
  2. Taking an inclusive cumulative sum/max/min of those per-step aggregates
     within each id, ordered by step.
  3. Using `pd.merge_asof` (direction="backward", grouped by id) to look up,
     for every transaction, the cumulative state as of `step - offset` for a
     handful of offsets (1, 2, 7, 25). This is a small, fixed number of
     vectorized sorted-merge passes over the full dataset rather than a
     per-row Python loop or a per-group `.apply`.
"""

import numpy as np
import pandas as pd

EPSILON = 1.0  # safe-division constant; avoids blow-up when a balance is 0

# ----------------------------------------------------------------------
# A. Transaction features
# ----------------------------------------------------------------------

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """amount, log_amount, one-hot type encoding, and type-based flags."""
    out = pd.DataFrame(index=df.index)
    out["amount"] = df["amount"]
    out["log_amount"] = np.log1p(df["amount"])

    for t in TRANSACTION_TYPES:
        out[f"type_{t}"] = (df["type"] == t).astype(np.int8)

    out["is_transfer"] = (df["type"] == "TRANSFER").astype(np.int8)
    out["is_cash_out"] = (df["type"] == "CASH_OUT").astype(np.int8)
    out["is_transfer_or_cash_out"] = out["is_transfer"] | out["is_cash_out"]
    return out


# ----------------------------------------------------------------------
# B. Time features
# ----------------------------------------------------------------------


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived purely from PaySim's simulated `step` -- NOT real timestamps.

    step is 1-indexed and represents simulated hours (1 step = 1 hour).
    """
    out = pd.DataFrame(index=df.index)
    out["simulated_hour"] = df["step"]
    out["simulated_day"] = (df["step"] - 1) // 24
    out["hour_of_day"] = (df["step"] - 1) % 24
    return out


# ----------------------------------------------------------------------
# C. Pre-transaction balance features
# ----------------------------------------------------------------------


def create_balance_features(df: pd.DataFrame, epsilon: float = EPSILON) -> pd.DataFrame:
    """Uses ONLY oldbalanceOrg / oldbalanceDest. The current row's
    newbalanceOrig / newbalanceDest are not used as inputs to this
    calculation (those are post-transaction state).
    """
    out = pd.DataFrame(index=df.index)
    out["oldbalanceOrg"] = df["oldbalanceOrg"]
    out["oldbalanceDest"] = df["oldbalanceDest"]
    out["amount_to_sender_balance"] = df["amount"] / (df["oldbalanceOrg"] + epsilon)
    out["amount_exceeds_sender_balance"] = (df["amount"] > df["oldbalanceOrg"]).astype(np.int8)
    out["sender_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(np.int8)
    out["destination_balance_zero"] = (df["oldbalanceDest"] == 0).astype(np.int8)
    return out


# ----------------------------------------------------------------------
# Shared internal machinery for D, E, F (historical + velocity features)
# ----------------------------------------------------------------------


def _first_occurrence_counts(df: pd.DataFrame, id_col: str, partner_col: str) -> pd.DataFrame:
    """For each (id_col, step), count how many DISTINCT partner_col values are
    seen for the first time ever (across the whole dataset) at that step.

    Cumulative-summing this over steps (per id) and then looking it up
    strictly before the current step gives a causal "distinct partners seen
    so far" feature, without ever using a future occurrence of the pair.
    """
    pair_first_step = (
        df.groupby([id_col, partner_col])["step"].min().rename("first_step").reset_index()
    )
    new_counts = (
        pair_first_step.groupby([id_col, "first_step"])
        .size()
        .rename("new_distinct_count")
        .reset_index()
        .rename(columns={"first_step": "step"})
    )
    return new_counts


def _build_step_level_table(
    df: pd.DataFrame,
    id_col: str,
    partner_col: str,
    include_type_counts: bool,
    include_min: bool,
    include_sumsq: bool,
) -> pd.DataFrame:
    """One row per (id_col, step) with vectorized (non-lambda) aggregates,
    plus an inclusive cumulative version of every aggregate, sorted by step
    within each id. This is the table `merge_asof` will query against.
    """
    work = df[[id_col, "step", "amount"]].copy()
    if include_type_counts:
        work["is_transfer"] = (df["type"] == "TRANSFER").astype(np.int64)
        work["is_cash_out"] = (df["type"] == "CASH_OUT").astype(np.int64)
    if include_sumsq:
        work["amount_sq"] = df["amount"] ** 2

    agg_spec = {"amount": ["size", "sum"]}
    if include_min:
        agg_spec["amount"] = ["size", "sum", "max", "min"]
    else:
        agg_spec["amount"] = ["size", "sum", "max"]
    if include_sumsq:
        agg_spec["amount_sq"] = ["sum"]
    if include_type_counts:
        agg_spec["is_transfer"] = ["sum"]
        agg_spec["is_cash_out"] = ["sum"]

    grouped = work.groupby([id_col, "step"]).agg(agg_spec)
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index()

    rename_map = {
        "amount_size": "count",
        "amount_sum": "sum_amount",
        "amount_max": "max_amount",
        "amount_min": "min_amount",
        "amount_sq_sum": "sumsq_amount",
        "is_transfer_sum": "transfer_count",
        "is_cash_out_sum": "cashout_count",
    }
    grouped = grouped.rename(columns=rename_map)

    # Causal distinct-partner counts (new distinct partners introduced at this step)
    new_dist = _first_occurrence_counts(df, id_col, partner_col)
    grouped = grouped.merge(new_dist, on=[id_col, "step"], how="left")
    grouped["new_distinct_count"] = grouped["new_distinct_count"].fillna(0)

    grouped = grouped.sort_values([id_col, "step"]).reset_index(drop=True)

    cumsum_cols = ["count", "sum_amount", "new_distinct_count"]
    if include_sumsq:
        cumsum_cols.append("sumsq_amount")
    if include_type_counts:
        cumsum_cols.extend(["transfer_count", "cashout_count"])

    grp = grouped.groupby(id_col)
    for col in cumsum_cols:
        grouped[f"cum_{col}"] = grp[col].cumsum()
    grouped["cum_max_amount"] = grp["max_amount"].cummax()
    if include_min:
        grouped["cum_min_amount"] = grp["min_amount"].cummin()

    return grouped


def _asof_lookup(
    df: pd.DataFrame,
    step_table: pd.DataFrame,
    id_col: str,
    offset: int,
    value_cols: list,
) -> pd.DataFrame:
    """For every row in df, find the state of `step_table` as of
    (row's step - offset), strictly using the latest step_table row with
    step <= (row_step - offset), grouped by id_col. Rows before an id's
    first recorded step (or with no id in step_table at all) get NaN, which
    callers fill in with an explicit, documented sentinel.
    """
    left = df[[id_col, "step"]].copy()
    left["__lookup_step"] = left["step"] - offset
    left["__orig_order"] = np.arange(len(left))
    left = left.sort_values("__lookup_step")

    non_step_value_cols = [c for c in value_cols if c != "step"]
    right = step_table[[id_col, "step"] + non_step_value_cols].copy()
    right = right.rename(columns={"step": "matched_step"})
    right = right.sort_values("matched_step")

    merged = pd.merge_asof(
        left,
        right,
        left_on="__lookup_step",
        right_on="matched_step",
        by=id_col,
        direction="backward",
    )
    merged = merged.sort_values("__orig_order")

    result_source_cols = ["matched_step" if c == "step" else c for c in value_cols]
    result = merged[result_source_cols].reset_index(drop=True)
    result.columns = value_cols
    result.index = df.index
    return result


# ----------------------------------------------------------------------
# D. Historical sender behavioral features
# ----------------------------------------------------------------------

SENDER_VALUE_COLS = [
    "cum_count",
    "cum_sum_amount",
    "cum_max_amount",
    "cum_min_amount",
    "cum_sumsq_amount",
    "cum_transfer_count",
    "cum_cashout_count",
    "cum_new_distinct_count",
    "step",
]


def _build_sender_step_table(df: pd.DataFrame) -> pd.DataFrame:
    return _build_step_level_table(
        df, id_col="nameOrig", partner_col="nameDest",
        include_type_counts=True, include_min=True, include_sumsq=True,
    )


def create_sender_history_features(df: pd.DataFrame, sender_table: pd.DataFrame = None) -> pd.DataFrame:
    """Causal sender-history features. Only uses transactions with a
    STRICTLY EARLIER step than the current row (offset=1 lookup)."""
    if sender_table is None:
        sender_table = _build_sender_step_table(df)

    prior = _asof_lookup(df, sender_table, "nameOrig", offset=1, value_cols=SENDER_VALUE_COLS)

    out = pd.DataFrame(index=df.index)
    count = prior["cum_count"].fillna(0)
    total = prior["cum_sum_amount"].fillna(0)
    sumsq = prior["cum_sumsq_amount"].fillna(0)

    out["prior_sender_transaction_count"] = count
    out["prior_sender_total_amount"] = total
    out["prior_sender_mean_amount"] = np.where(count > 0, total / count.replace(0, np.nan), 0.0)
    out["prior_sender_max_amount"] = prior["cum_max_amount"].fillna(0)
    out["prior_sender_min_amount"] = prior["cum_min_amount"].fillna(0)

    mean = out["prior_sender_mean_amount"]
    variance = np.where(count > 0, (sumsq / count.replace(0, np.nan)) - mean ** 2, 0.0)
    variance = np.clip(variance, a_min=0, a_max=None)  # guard tiny negative floats
    out["prior_sender_amount_std"] = np.where(count > 1, np.sqrt(variance), 0.0)

    has_history = count > 0
    out["prior_sender_has_history"] = has_history.astype(np.int8)
    out["prior_sender_time_since_last_transaction"] = np.where(
        has_history, df["step"] - prior["step"], -1
    )

    out["prior_sender_transfer_count"] = prior["cum_transfer_count"].fillna(0)
    out["prior_sender_cashout_count"] = prior["cum_cashout_count"].fillna(0)
    out["prior_sender_distinct_destinations"] = prior["cum_new_distinct_count"].fillna(0)
    return out


# ----------------------------------------------------------------------
# E. Historical receiver behavioral features
# ----------------------------------------------------------------------

RECEIVER_VALUE_COLS = [
    "cum_count",
    "cum_sum_amount",
    "cum_max_amount",
    "cum_new_distinct_count",
    "step",
]


def _build_receiver_step_table(df: pd.DataFrame) -> pd.DataFrame:
    return _build_step_level_table(
        df, id_col="nameDest", partner_col="nameOrig",
        include_type_counts=False, include_min=False, include_sumsq=False,
    )


def create_receiver_history_features(df: pd.DataFrame, receiver_table: pd.DataFrame = None) -> pd.DataFrame:
    """Causal receiver-history features. Only uses transactions with a
    STRICTLY EARLIER step than the current row (offset=1 lookup)."""
    if receiver_table is None:
        receiver_table = _build_receiver_step_table(df)

    prior = _asof_lookup(df, receiver_table, "nameDest", offset=1, value_cols=RECEIVER_VALUE_COLS)

    out = pd.DataFrame(index=df.index)
    count = prior["cum_count"].fillna(0)
    total = prior["cum_sum_amount"].fillna(0)

    out["prior_receiver_transaction_count"] = count
    out["prior_receiver_total_amount"] = total
    out["prior_receiver_mean_amount"] = np.where(count > 0, total / count.replace(0, np.nan), 0.0)
    out["prior_receiver_max_amount"] = prior["cum_max_amount"].fillna(0)

    has_history = count > 0
    out["prior_receiver_has_history"] = has_history.astype(np.int8)
    out["prior_receiver_time_since_last_transaction"] = np.where(
        has_history, df["step"] - prior["step"], -1
    )
    out["prior_receiver_distinct_senders"] = prior["cum_new_distinct_count"].fillna(0)
    return out


# ----------------------------------------------------------------------
# F. Velocity features
# ----------------------------------------------------------------------


def create_velocity_features(
    df: pd.DataFrame, sender_table: pd.DataFrame = None, receiver_table: pd.DataFrame = None
) -> pd.DataFrame:
    """Rolling-window transaction counts, defined as:
        sender_transactions_previous_K_steps =
            (# sender txns with step in [current_step - 1, ...]  strictly prior)
            minus (# sender txns strictly before step current_step - K)
          = cum_count(step-1) - cum_count(step-1-K)

    i.e. "how many transactions did this sender make in the K steps
    immediately preceding this one" -- never including the current
    transaction itself. K in {1, 6, 24}.
    """
    if sender_table is None:
        sender_table = _build_sender_step_table(df)
    if receiver_table is None:
        receiver_table = _build_receiver_step_table(df)

    s1 = _asof_lookup(df, sender_table, "nameOrig", offset=1, value_cols=["cum_count", "cum_sum_amount"])
    s2 = _asof_lookup(df, sender_table, "nameOrig", offset=2, value_cols=["cum_count"])
    s7 = _asof_lookup(df, sender_table, "nameOrig", offset=7, value_cols=["cum_count"])
    s25 = _asof_lookup(df, sender_table, "nameOrig", offset=25, value_cols=["cum_count", "cum_sum_amount"])
    r1 = _asof_lookup(df, receiver_table, "nameDest", offset=1, value_cols=["cum_count"])
    r25 = _asof_lookup(df, receiver_table, "nameDest", offset=25, value_cols=["cum_count"])

    out = pd.DataFrame(index=df.index)
    out["sender_transactions_previous_1_step"] = s1["cum_count"].fillna(0) - s2["cum_count"].fillna(0)
    out["sender_transactions_previous_6_steps"] = s1["cum_count"].fillna(0) - s7["cum_count"].fillna(0)
    out["sender_transactions_previous_24_steps"] = s1["cum_count"].fillna(0) - s25["cum_count"].fillna(0)
    out["sender_amount_previous_24_steps"] = s1["cum_sum_amount"].fillna(0) - s25["cum_sum_amount"].fillna(0)
    out["receiver_transactions_previous_24_steps"] = r1["cum_count"].fillna(0) - r25["cum_count"].fillna(0)

    for col in out.columns:
        out[col] = out[col].clip(lower=0)  # guard against any float rounding at the boundary
    return out


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

# Columns that must NEVER be used to construct a feature.
FORBIDDEN_SOURCE_COLUMNS = {"newbalanceOrig", "newbalanceDest", "isFlaggedFraud", "isFraud"}

# Metadata columns kept in the processed dataset for traceability/validation,
# but excluded from the model's feature matrix.
ID_AND_TARGET_COLUMNS = ["nameOrig", "nameDest", "step", "isFraud"]


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full causal feature dataset from raw PaySim transactions.

    Returns a DataFrame containing: nameOrig, nameDest, step, isFraud
    (kept for traceability/validation, NOT model inputs) plus every
    engineered feature. Use `get_model_feature_columns()` to get the
    column list that should actually be fed to a model.
    """
    required = {"step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "oldbalanceDest", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    sender_table = _build_sender_step_table(df)
    receiver_table = _build_receiver_step_table(df)

    parts = [
        df[["nameOrig", "nameDest", "step", "isFraud"]],
        create_transaction_features(df),
        create_time_features(df),
        create_balance_features(df),
        create_sender_history_features(df, sender_table),
        create_receiver_history_features(df, receiver_table),
        create_velocity_features(df, sender_table, receiver_table),
    ]
    result = pd.concat(parts, axis=1)
    return result


def get_model_feature_columns(feature_df: pd.DataFrame) -> list:
    """Every column in `feature_df` EXCEPT the raw IDs and the target --
    i.e. exactly what should be handed to a model."""
    return [c for c in feature_df.columns if c not in ID_AND_TARGET_COLUMNS]
