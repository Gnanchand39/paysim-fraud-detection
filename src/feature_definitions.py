"""Interview-friendly documentation of every column produced by
`src.features.build_feature_dataset`.

Each entry describes: what the feature means, which raw column(s) it comes
from, whether it's available at decision time, whether it uses historical
(cross-row) information, its leakage risk, and its expected dtype.

This is documentation only -- it does not compute anything. See
`src/features.py` for the actual feature logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDoc:
    name: str
    meaning: str
    source_columns: str
    available_at_decision_time: bool
    uses_historical_info: bool
    leakage_risk: str
    expected_dtype: str


FEATURE_DEFINITIONS = [
    # --- Metadata (kept for traceability, excluded from model input) ---
    FeatureDoc(
        "nameOrig", "Sender customer ID. Kept only for building/validating historical features.",
        "nameOrig", True, False,
        "None as a raw value, but MUST be dropped before modeling -- it's a near-unique ID with no generalizable meaning.",
        "object (string, dropped from model matrix)",
    ),
    FeatureDoc(
        "nameDest", "Recipient customer/merchant ID. Kept only for building/validating historical features.",
        "nameDest", True, False,
        "Same as nameOrig -- dropped before modeling.",
        "object (string, dropped from model matrix)",
    ),
    FeatureDoc(
        "step", "Raw simulated-time step, kept for traceability/temporal-split use in Step 4.",
        "step", True, False,
        "None -- available at decision time by definition.",
        "int64 (kept for splitting, not necessarily fed as a raw model feature)",
    ),
    FeatureDoc(
        "isFraud", "TARGET label. Preserved in the processed dataset for training/evaluation. Never used to construct any feature.",
        "isFraud", False, False,
        "N/A -- this is the target, not a feature.",
        "int64 (target)",
    ),
    # --- A. Transaction features ---
    FeatureDoc(
        "amount", "Requested transaction amount.",
        "amount", True, False, "None -- known before approval.", "float64",
    ),
    FeatureDoc(
        "log_amount", "log1p(amount); compresses the heavy right skew found in EDA.",
        "amount", True, False, "None.", "float64",
    ),
    FeatureDoc(
        "type_CASH_IN / type_CASH_OUT / type_DEBIT / type_PAYMENT / type_TRANSFER",
        "One-hot encoding of transaction type.",
        "type", True, False, "None.", "int8 (5 columns)",
    ),
    FeatureDoc(
        "is_transfer", "1 if type == TRANSFER, else 0.",
        "type", True, False, "None.", "int8",
    ),
    FeatureDoc(
        "is_cash_out", "1 if type == CASH_OUT, else 0.",
        "type", True, False, "None.", "int8",
    ),
    FeatureDoc(
        "is_transfer_or_cash_out",
        "1 if type is TRANSFER or CASH_OUT -- the only two types fraud occurs in per EDA (Step 2).",
        "type", True, False, "None.", "int8",
    ),
    # --- B. Time features ---
    FeatureDoc(
        "simulated_hour", "Alias of `step`. PaySim SIMULATED time (1 step = 1 simulated hour), NOT a real timestamp.",
        "step", True, False, "None.", "int64",
    ),
    FeatureDoc(
        "simulated_day", "(step - 1) // 24 -- which simulated day (0-30) the transaction falls in.",
        "step", True, False, "None.", "int64",
    ),
    FeatureDoc(
        "hour_of_day", "(step - 1) % 24 -- hour within the simulated day. EDA found fraud COUNT is flat across this but fraud RATE is confounded by legit-volume seasonality -- feed this to the model rather than a rate.",
        "step", True, False, "None.", "int64",
    ),
    # --- C. Pre-transaction balance features ---
    FeatureDoc(
        "oldbalanceOrg", "Sender's balance BEFORE this transaction.",
        "oldbalanceOrg", True, False, "None -- pre-transaction state, legitimately known before approval.", "float64",
    ),
    FeatureDoc(
        "oldbalanceDest", "Recipient's balance BEFORE this transaction.",
        "oldbalanceDest", True, False, "None.", "float64",
    ),
    FeatureDoc(
        "amount_to_sender_balance", "amount / (oldbalanceOrg + 1). Safe-division ratio of requested amount to sender's available balance.",
        "amount, oldbalanceOrg", True, False, "None -- uses only pre-transaction values.", "float64",
    ),
    FeatureDoc(
        "amount_exceeds_sender_balance", "1 if amount > oldbalanceOrg (the sender doesn't have enough balance to cover it).",
        "amount, oldbalanceOrg", True, False, "None.", "int8",
    ),
    FeatureDoc(
        "sender_balance_zero", "1 if oldbalanceOrg == 0.",
        "oldbalanceOrg", True, False, "None.", "int8",
    ),
    FeatureDoc(
        "destination_balance_zero", "1 if oldbalanceDest == 0.",
        "oldbalanceDest", True, False, "None.", "int8",
    ),
    # --- D. Historical sender features ---
    FeatureDoc(
        "prior_sender_transaction_count", "Count of this sender's transactions with a STRICTLY EARLIER step. 0 for a sender's first transaction.",
        "nameOrig, step (historical)", True, True, "None -- causally computed, excludes current and same-step transactions.", "float64 (whole numbers)",
    ),
    FeatureDoc(
        "prior_sender_total_amount", "Sum of amounts from this sender's strictly-earlier transactions.",
        "nameOrig, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_mean_amount", "prior_sender_total_amount / prior_sender_transaction_count (0 if no history).",
        "derived", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_max_amount", "Maximum amount among this sender's strictly-earlier transactions (0 if no history).",
        "nameOrig, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_min_amount", "Minimum amount among this sender's strictly-earlier transactions (0 if no history).",
        "nameOrig, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_amount_std", "Std. dev. of this sender's strictly-earlier transaction amounts (0 if <2 prior transactions). Computed via cumulative sum/sum-of-squares, not raw values.",
        "nameOrig, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_has_history", "1 if this sender has at least one strictly-earlier transaction, else 0.",
        "nameOrig, step (historical)", True, True, "None -- causal.", "int8",
    ),
    FeatureDoc(
        "prior_sender_time_since_last_transaction", "current step minus the sender's most recent strictly-earlier step. Sentinel -1 if no prior history.",
        "nameOrig, step (historical)", True, True, "None -- causal.", "float64 (>=1 or -1 sentinel)",
    ),
    FeatureDoc(
        "prior_sender_transfer_count", "Count of this sender's strictly-earlier TRANSFER transactions.",
        "nameOrig, step, type (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_cashout_count", "Count of this sender's strictly-earlier CASH_OUT transactions.",
        "nameOrig, step, type (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_sender_distinct_destinations", "Count of DISTINCT recipients this sender has sent to in strictly-earlier transactions.",
        "nameOrig, nameDest, step (historical)", True, True, "None -- causal, computed via first-occurrence-step logic (never looks at a later occurrence of a pair to decide if it's 'new').", "float64",
    ),
    # --- E. Historical receiver features ---
    FeatureDoc(
        "prior_receiver_transaction_count", "Count of transactions received by this account with a strictly-earlier step. 0 for a receiver's first-ever transaction.",
        "nameDest, step (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_receiver_total_amount", "Sum of amounts received in strictly-earlier transactions.",
        "nameDest, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_receiver_mean_amount", "prior_receiver_total_amount / prior_receiver_transaction_count (0 if no history).",
        "derived", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_receiver_max_amount", "Maximum amount received in strictly-earlier transactions (0 if no history).",
        "nameDest, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "prior_receiver_has_history", "1 if this receiver has received at least one strictly-earlier transaction, else 0.",
        "nameDest, step (historical)", True, True, "None -- causal.", "int8",
    ),
    FeatureDoc(
        "prior_receiver_time_since_last_transaction", "current step minus the receiver's most recent strictly-earlier step. Sentinel -1 if no prior history.",
        "nameDest, step (historical)", True, True, "None -- causal.", "float64 (>=1 or -1 sentinel)",
    ),
    FeatureDoc(
        "prior_receiver_distinct_senders", "Count of DISTINCT senders that have sent to this receiver in strictly-earlier transactions.",
        "nameDest, nameOrig, step (historical)", True, True, "None -- causal (same first-occurrence-step logic as sender-side).", "float64",
    ),
    # --- F. Velocity features ---
    FeatureDoc(
        "sender_transactions_previous_1_step", "Count of this sender's transactions in the single step immediately before this one (step-1 only).",
        "nameOrig, step (historical)", True, True, "None -- causal, uses a bounded strictly-prior window.", "float64",
    ),
    FeatureDoc(
        "sender_transactions_previous_6_steps", "Count of this sender's transactions in the 6 steps immediately before this one (steps [step-6, step-1]).",
        "nameOrig, step (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "sender_transactions_previous_24_steps", "Count of this sender's transactions in the 24 steps immediately before this one (steps [step-24, step-1]).",
        "nameOrig, step (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "sender_amount_previous_24_steps", "Sum of this sender's transaction amounts in the 24 steps immediately before this one.",
        "nameOrig, step, amount (historical)", True, True, "None -- causal.", "float64",
    ),
    FeatureDoc(
        "receiver_transactions_previous_24_steps", "Count of transactions received by this account in the 24 steps immediately before this one.",
        "nameDest, step (historical)", True, True, "None -- causal.", "float64",
    ),
]


def print_feature_table():
    """Pretty-print the full feature documentation table."""
    for f in FEATURE_DEFINITIONS:
        print(f"\n{f.name}")
        print(f"  Meaning:            {f.meaning}")
        print(f"  Source column(s):   {f.source_columns}")
        print(f"  At decision time:   {f.available_at_decision_time}")
        print(f"  Uses history:       {f.uses_historical_info}")
        print(f"  Leakage risk:       {f.leakage_risk}")
        print(f"  Expected dtype:     {f.expected_dtype}")


if __name__ == "__main__":
    print_feature_table()
