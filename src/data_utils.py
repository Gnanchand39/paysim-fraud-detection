"""Utilities for loading the PaySim dataset."""

import glob
import os
from typing import Optional

import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Human-readable descriptions of every PaySim column, used in the
# data-understanding notebook so we don't have to re-derive this from
# the Kaggle page every time.
COLUMN_DESCRIPTIONS = {
    "step": "Time unit in the simulation. 1 step = 1 hour of real time. "
            "The dataset spans 744 steps (30 days).",
    "type": "Type of transaction: CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER.",
    "amount": "Transaction amount, in local currency.",
    "nameOrig": "ID of the customer who initiated (originated) the transaction.",
    "oldbalanceOrg": "Sender's account balance before the transaction.",
    "newbalanceOrig": "Sender's account balance after the transaction.",
    "nameDest": "ID of the transaction's recipient.",
    "oldbalanceDest": "Recipient's account balance before the transaction. "
                       "Not populated (0) for merchants (names starting with 'M').",
    "newbalanceDest": "Recipient's account balance after the transaction. "
                       "Same merchant caveat as oldbalanceDest.",
    "isFraud": "Target label. 1 if the transaction is a fraudulent transaction "
               "simulated by PaySim's fraud-injection logic, else 0.",
    "isFlaggedFraud": "A simple rule-based flag PaySim itself sets when a single "
                       "TRANSFER of more than 200,000 is attempted. This is NOT "
                       "our model's prediction — it's a naive baseline rule "
                       "already present in the raw data.",
}


def find_raw_csv() -> str:
    """Locate the PaySim CSV inside data/raw/, regardless of exact filename."""
    candidates = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV file found in {os.path.abspath(RAW_DATA_DIR)}. "
            "Download the PaySim dataset and place the CSV there."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected exactly one CSV in {RAW_DATA_DIR}, found {len(candidates)}: "
            f"{candidates}. Remove the extra file(s)."
        )
    return candidates[0]


def load_raw_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw PaySim CSV into a DataFrame.

    Parameters
    ----------
    nrows : optional
        If set, only reads the first `nrows` rows. Useful for quick checks
        on a laptop before working with the full ~6.3M-row file.
    """
    path = find_raw_csv()
    return pd.read_csv(path, nrows=nrows)
