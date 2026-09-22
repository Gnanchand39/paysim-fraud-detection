"""Chronological (time-aware) train/validation/test splitting for the
PaySim feature dataset.

WHY NOT A RANDOM SPLIT
-----------------------
Fraud detection is deployed against the future: a model is trained on past
transactions and used to score transactions that haven't happened yet. A
random split (or `train_test_split(shuffle=True)`) would let the model
train on transactions that occur chronologically AFTER some of its
validation/test transactions -- an evaluation setup no real deployment
could ever match. A chronological split (train = earliest steps,
validation = middle steps, test = latest steps) is the only split that
mirrors how the model will actually be used.
"""

import pandas as pd


def inspect_temporal_distribution(df: pd.DataFrame, step_col: str = "step", target_col: str = "isFraud") -> pd.DataFrame:
    """Per-step transaction/fraud counts, rates, and cumulative totals."""
    per_step = df.groupby(step_col).agg(
        txn_count=(target_col, "size"), fraud_count=(target_col, "sum")
    )
    per_step["fraud_rate_pct"] = per_step["fraud_count"] / per_step["txn_count"] * 100
    per_step["cum_txn"] = per_step["txn_count"].cumsum()
    per_step["cum_fraud"] = per_step["fraud_count"].cumsum()
    total_txn = len(df)
    total_fraud = df[target_col].sum()
    per_step["cum_txn_pct"] = per_step["cum_txn"] / total_txn * 100
    per_step["cum_fraud_pct"] = per_step["cum_fraud"] / total_fraud * 100
    return per_step


def choose_temporal_boundaries(per_step: pd.DataFrame, train_pct: float = 0.70, val_pct: float = 0.15) -> tuple:
    """Pick the earliest step at which cumulative ROW count reaches each
    target percentage. Boundaries always fall on a complete step -- no
    step's transactions are ever split across two sets.

    Returns (train_end_step, val_end_step). train = [min_step, train_end_step],
    validation = [train_end_step+1, val_end_step], test = [val_end_step+1, max_step].
    """
    train_end_step = per_step[per_step["cum_txn_pct"] >= train_pct * 100].index.min()
    val_end_step = per_step[per_step["cum_txn_pct"] >= (train_pct + val_pct) * 100].index.min()
    return int(train_end_step), int(val_end_step)


def create_temporal_split(df: pd.DataFrame, train_end_step: int, val_end_step: int, step_col: str = "step"):
    """Split df into (train, validation, test) via boolean masks on `step`.
    Each split is a single filtered view -- no intermediate full-dataset
    copies are made beyond the three resulting subsets themselves."""
    train_mask = df[step_col] <= train_end_step
    val_mask = (df[step_col] > train_end_step) & (df[step_col] <= val_end_step)
    test_mask = df[step_col] > val_end_step

    train_df = df.loc[train_mask]
    val_df = df.loc[val_mask]
    test_df = df.loc[test_mask]
    return train_df, val_df, test_df


def validate_temporal_split(
    df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    step_col: str = "step", target_col: str = "isFraud",
) -> bool:
    """Run the 8 explicit assertions required for a defensible temporal
    split. Prints each result and returns True only if all pass."""
    results = {}

    results["max(train_step) < min(validation_step)"] = train_df[step_col].max() < val_df[step_col].min()
    results["max(validation_step) < min(test_step)"] = val_df[step_col].max() < test_df[step_col].min()

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    results["no row appears in more than one split"] = (
        len(train_idx & val_idx) == 0 and len(val_idx & test_idx) == 0 and len(train_idx & test_idx) == 0
    )

    results["row counts sum to full dataset"] = (len(train_df) + len(val_df) + len(test_df)) == len(df)
    results["fraud counts sum to full dataset"] = (
        train_df[target_col].sum() + val_df[target_col].sum() + test_df[target_col].sum()
    ) == df[target_col].sum()

    results["each split's step range falls within its expected bounds"] = (
        train_df[step_col].min() == df[step_col].min()
        and test_df[step_col].max() == df[step_col].max()
        and train_df[step_col].max() < val_df[step_col].min()
        and val_df[step_col].max() < test_df[step_col].min()
    )

    results["no feature columns dropped (same column set in every split)"] = (
        list(train_df.columns) == list(df.columns)
        and list(val_df.columns) == list(df.columns)
        and list(test_df.columns) == list(df.columns)
    )

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print("\n  Target distribution per split:")
    for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        n = len(split)
        f = int(split[target_col].sum())
        print(f"    {name:10s}: {n:>9,} rows, {f:>6,} fraud ({f/n*100:.4f}%)")

    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def summarize_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                     step_col: str = "step", target_col: str = "isFraud") -> pd.DataFrame:
    """Return the split | rows | fraud | legitimate | fraud_rate | min_step |
    max_step | unique_steps summary table."""
    rows = []
    for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        n = len(split)
        fraud = int(split[target_col].sum())
        rows.append({
            "split": name,
            "rows": n,
            "fraud": fraud,
            "legitimate": n - fraud,
            "fraud_rate_pct": round(fraud / n * 100, 4),
            "min_step": int(split[step_col].min()),
            "max_step": int(split[step_col].max()),
            "unique_steps": split[step_col].nunique(),
        })
    return pd.DataFrame(rows)
