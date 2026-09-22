"""Reusable, model-agnostic utilities for loading the chronological splits,
training a classifier, and evaluating it with fraud-appropriate metrics.

Deliberately NOT specific to Logistic Regression -- `evaluate_classifier`,
`evaluate_at_k`, and `load_split_data` are written to be reused by later
models (Random Forest, XGBoost) without modification.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import get_model_feature_columns

TARGET_COL = "isFraud"
FORBIDDEN_IN_X = {"nameOrig", "nameDest", "isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest"}


def load_split_data(processed_dir: str = "data/processed"):
    """Load train/validation/test parquet files and split each into
    (X, y) using the approved model feature columns. Returns a dict with
    keys: X_train, y_train, X_validation, y_validation, X_test, y_test,
    feature_columns.
    """
    train_df = pd.read_parquet(f"{processed_dir}/train.parquet")
    val_df = pd.read_parquet(f"{processed_dir}/validation.parquet")
    test_df = pd.read_parquet(f"{processed_dir}/test.parquet")

    feature_columns = get_model_feature_columns(train_df)

    return {
        "X_train": train_df[feature_columns],
        "y_train": train_df[TARGET_COL],
        "X_validation": val_df[feature_columns],
        "y_validation": val_df[TARGET_COL],
        "X_test": test_df[feature_columns],
        "y_test": test_df[TARGET_COL],
        "feature_columns": feature_columns,
    }


def validate_model_features(X: pd.DataFrame, feature_columns: list, expected_count: int = 42) -> dict:
    """Data-quality + leakage checks required before fitting any model.
    Returns a dict of check_name -> bool and prints nothing itself."""
    checks = {}
    checks["feature_count_matches_expected"] = len(feature_columns) == expected_count
    checks["columns_match_feature_list"] = list(X.columns) == list(feature_columns)
    checks["no_forbidden_columns_present"] = len(FORBIDDEN_IN_X & set(X.columns)) == 0
    checks["all_columns_numeric"] = all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    checks["no_missing_values"] = not X.isnull().any().any()
    checks["no_infinite_values"] = not np.isinf(X.to_numpy(dtype=np.float64)).any()
    return checks


def find_constant_features(X: pd.DataFrame, near_zero_std_threshold: float = 1e-8) -> list:
    """Feature names whose standard deviation is at/near zero (no
    discriminative information for a linear model)."""
    stds = X.std()
    return stds[stds <= near_zero_std_threshold].index.tolist()


def build_logistic_pipeline(class_weight=None, max_iter: int = 1000, solver: str = "lbfgs", random_state: int = 42) -> Pipeline:
    """StandardScaler -> LogisticRegression, bundled in one Pipeline so the
    scaler is always fit/applied together with the model it was fit for."""
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight=class_weight, max_iter=max_iter, solver=solver, random_state=random_state,
        )),
    ])


def exclude_features(feature_columns: list, to_exclude: list) -> list:
    """Feature list with `to_exclude` removed, order preserved. Used for
    controlled ablation experiments (e.g. removing suspect features to
    measure their effect on model performance)."""
    excluded = set(to_exclude)
    missing = excluded - set(feature_columns)
    if missing:
        raise ValueError(f"Cannot exclude columns not present in feature_columns: {missing}")
    return [c for c in feature_columns if c not in excluded]


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """legitimate_count / fraud_count from TRAINING labels only -- the
    standard XGBoost recipe for imbalanced binary classification."""
    n_fraud = int(y_train.sum())
    n_legit = int(len(y_train) - n_fraud)
    return n_legit / n_fraud


def build_xgboost_classifier(scale_pos_weight: float = 1.0, n_estimators: int = 300, max_depth: int = 6,
                              learning_rate: float = 0.1, subsample: float = 0.8, colsample_bytree: float = 0.8,
                              min_child_weight: int = 1, reg_lambda: float = 1.0, tree_method: str = "hist",
                              n_jobs: int = 8, random_state: int = 42):
    """A reasonable baseline XGBClassifier -- fixed boosting rounds (no early
    stopping), not hyperparameter-searched (that's a separate future step)."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        reg_lambda=reg_lambda, objective="binary:logistic", eval_metric="logloss",
        tree_method=tree_method, scale_pos_weight=scale_pos_weight,
        random_state=random_state, n_jobs=n_jobs,
    )


def extract_xgboost_feature_importance(model, feature_columns: list, importance_type: str = "gain") -> pd.DataFrame:
    """feature | importance from a fitted XGBClassifier's gain-based
    importance, sorted descending. Gain measures the average improvement in
    the loss function contributed by splits on that feature -- NOT a causal
    measure, and (like Random Forest's impurity importance) can split credit
    across correlated/redundant features.
    """
    booster = model.get_booster()
    raw_scores = booster.get_score(importance_type=importance_type)
    # When fit on a pandas DataFrame, the booster keys importance by the
    # actual column name; if fit on a bare array it falls back to "f0","f1",...
    # Features never split on are simply absent from raw_scores (importance 0).
    uses_column_names = any(c in raw_scores for c in feature_columns)
    if uses_column_names:
        importances = [raw_scores.get(c, 0.0) for c in feature_columns]
    else:
        importances = [raw_scores.get(f"f{i}", 0.0) for i in range(len(feature_columns))]
    df = pd.DataFrame({"feature": feature_columns, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def train_model(estimator, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit `estimator` on (X_train, y_train) ONLY. Works for a bare
    scikit-learn estimator (e.g. RandomForestClassifier) or a Pipeline
    (e.g. StandardScaler -> LogisticRegression). Returns timing/convergence
    metadata alongside the fitted estimator/pipeline (key "pipeline" is kept
    for backward compatibility with Step 5's code)."""
    t0 = time.time()
    estimator.fit(X_train, y_train)
    elapsed = time.time() - t0

    if isinstance(estimator, Pipeline):
        model = estimator.named_steps.get("model", estimator.steps[-1][1])
    else:
        model = estimator
    n_iter = getattr(model, "n_iter_", None)
    max_iter = getattr(model, "max_iter", None)
    converged = bool(n_iter is not None and max_iter is not None and int(np.max(n_iter)) < max_iter)

    return {
        "pipeline": estimator,
        "training_time_sec": elapsed,
        "n_iter": None if n_iter is None else [int(x) for x in np.atleast_1d(n_iter)],
        "converged": converged,
    }


def evaluate_classifier(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Fraud-appropriate metrics at a fixed threshold: precision, recall,
    F1, ROC-AUC, PR-AUC (average precision), and the confusion matrix.
    ROC-AUC/PR-AUC use the continuous probabilities and are threshold-independent;
    precision/recall/F1/confusion matrix use `threshold` to binarize.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def precision_at_k(y_true: pd.Series, y_proba: np.ndarray, k: int) -> float:
    """Fraction of the top-K highest-scored transactions that are actually fraud."""
    order = np.argsort(-y_proba)[:k]
    return float(np.asarray(y_true)[order].sum() / k)


def recall_at_k(y_true: pd.Series, y_proba: np.ndarray, k: int) -> float:
    """Fraction of ALL frauds in this split that are captured within the top-K."""
    total_fraud = np.asarray(y_true).sum()
    if total_fraud == 0:
        return float("nan")
    order = np.argsort(-y_proba)[:k]
    return float(np.asarray(y_true)[order].sum() / total_fraud)


def evaluate_at_k(y_true: pd.Series, y_proba: np.ndarray, k_values: list) -> pd.DataFrame:
    """Precision@K / Recall@K for every k in k_values that does not exceed
    len(y_true). Larger k values are skipped and reported as such."""
    n = len(y_true)
    rows = []
    for k in k_values:
        if k > n:
            rows.append({"k": k, "precision_at_k": None, "recall_at_k": None, "skipped_reason": f"k={k} exceeds split size ({n} rows)"})
            continue
        rows.append({
            "k": k,
            "precision_at_k": precision_at_k(y_true, y_proba, k),
            "recall_at_k": recall_at_k(y_true, y_proba, k),
            "skipped_reason": None,
        })
    return pd.DataFrame(rows)


def build_random_forest(class_weight=None, n_estimators: int = 200, max_depth: int = 20,
                         min_samples_split: int = 2, min_samples_leaf: int = 1,
                         max_features="sqrt", n_jobs: int = -1, random_state: int = 42):
    """A reasonable baseline RandomForestClassifier -- not scaled (trees don't
    need it) and not hyperparameter-searched (that's a separate future step).
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, class_weight=class_weight,
        n_jobs=n_jobs, random_state=random_state,
    )


def tree_depth_summary(forest) -> dict:
    """Min/mean/max depth across every tree in a fitted forest."""
    depths = [est.get_depth() for est in forest.estimators_]
    return {
        "min_depth": int(np.min(depths)),
        "mean_depth": float(np.mean(depths)),
        "max_depth": int(np.max(depths)),
        "n_trees": len(depths),
    }


def extract_tree_feature_importance(forest, feature_columns: list) -> pd.DataFrame:
    """feature | importance from a fitted tree ensemble's `feature_importances_`,
    sorted descending. This is impurity-based importance: it measures how much
    a feature contributed to reducing impurity across the fitted forest's
    splits. It is NOT a causal measure, can be biased toward continuous /
    high-cardinality features, and gets split across correlated/redundant
    features (e.g. `type_TRANSFER` and `is_transfer` carry the same signal,
    so neither will show the full combined importance on its own).
    """
    df = pd.DataFrame({"feature": feature_columns, "importance": forest.feature_importances_})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def extract_logistic_coefficients(pipeline: Pipeline, feature_columns: list) -> pd.DataFrame:
    """feature | coefficient | odds_ratio, sorted by |coefficient| descending.
    odds_ratio = exp(coefficient): multiplicative change in the odds of
    fraud for a one-standard-deviation increase in the (scaled) feature,
    holding other features fixed WITHIN THIS FITTED MODEL -- not a causal claim.
    """
    model = pipeline.named_steps["model"]
    coefs = model.coef_.ravel()
    df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    })
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient").reset_index(drop=True)
