# Mobile-Money Fraud Detection using PaySim

## Overview

This project builds an end-to-end fraud-detection workflow on **PaySim**, a synthetic mobile-money transaction dataset. Given a transaction, the task is to estimate the probability that it is fraudulent, using only information that would genuinely be available *before* the transaction is executed.

Fraud detection is a hard machine-learning problem: fraud is extremely rare (well under 1% of transactions here), which means naive accuracy is meaningless, class-imbalance handling matters, and evaluation has to be done with rank-aware and precision/recall-aware metrics rather than a single headline number. It's also a problem where it is easy to accidentally leak information that wouldn't exist at prediction time — this project treats leakage prevention as a first-class concern, not an afterthought.

**PaySim is a synthetic dataset.** It is not real banking, UPI, or NPCI transaction data, and no claims in this repository should be read as evidence of real-world fraud-detection performance. The value of this project is the workflow: causal feature engineering, chronological validation, controlled baseline comparisons, a deliberate robustness (ablation) experiment, fixed-model error analysis, and an honestly-scoped final model with a small interactive demo — not a production fraud system.

## Key Highlights

- Causal, leakage-checked feature engineering (verified against a train-only reconstruction)
- Historical sender/receiver behavioral features computed from strictly earlier transactions only
- Chronological train / validation / test split (no random shuffling)
- Severe class-imbalance handling compared across `class_weight` and `scale_pos_weight` strategies
- Logistic Regression, Random Forest, and XGBoost baselines under identical splits and features
- A deliberate artifact-ablation robustness experiment on the two most dominant engineered features
- Fixed-model error analysis (false positives/negatives, transaction type, simulated time, amount bins)
- Precision@K / Recall@K evaluation for review-capacity-constrained scenarios
- A Streamlit demo application with Quick and Advanced modes
- No SMOTE, no test-set-based tuning, no post-hoc threshold optimization

## Problem Statement

Given a transaction at the moment it is requested, estimate whether it is fraudulent.

- **Fraud is rare.** In this dataset, 8,213 of 6,362,620 transactions (0.129%) are labeled fraud.
- **False negatives matter** — a missed fraud transaction is a fraud that succeeds.
- **False positives matter** — flagging legitimate transactions has a real review/friction cost, even though this project does not quantify it in monetary terms.
- **Random train/test splitting would leak time.** A model could otherwise "see" the future during training, which no real deployment could ever match.
- **Fields that only exist after a transaction completes cannot be used to score it beforehand** — using them would silently inflate reported performance.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a synthetic mobile-money transaction simulator (Lopez-Rojas, Elmir & Axelsson, 2016), built to mimic real transaction logs for fraud-detection research.

| Property | Value |
|---|---|
| Total transactions | 6,362,620 |
| Total fraud transactions | 8,213 (0.129%) |
| Transaction types | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| Fraud occurs in | `TRANSFER` and `CASH_OUT` only |
| Simulated time span | 743 steps (1 step = 1 simulated hour, ≈31 days) |
| Key raw columns | `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud` |

**This project uses synthetic PaySim data and does not use NPCI, UPI, bank, or real customer transaction data.**

## Data Understanding & EDA

Full detail: `notebooks/01_data_understanding.ipynb`, `notebooks/02_eda.ipynb`.

- The raw data has no missing values and no exact duplicate rows.
- Fraud is concentrated entirely in `TRANSFER` and `CASH_OUT` transactions; `PAYMENT`, `CASH_IN`, and `DEBIT` contain zero fraud in this dataset.
- `isFlaggedFraud`, PaySim's own built-in rule, flags only 16 of 6,362,620 transactions — far too few to be useful, and excluded from modeling as a rule-derived (not causal) signal.
- A large share of fraud transactions exhibit a "drained sender account" pattern (`oldbalanceOrg == amount`, `newbalanceOrig == 0`) that is almost perfectly separating in the raw data. This is treated as a likely **PaySim simulation artifact** rather than a general fraud law, and is investigated explicitly later in this project (see Artifact Ablation below).
- Legitimate transaction volume varies substantially across the simulated 31 days, while fraud counts stay comparatively steady — this shapes both the temporal split and the interpretation of later results.

These are dataset-specific observations from a synthetic simulator, not general statements about how digital-payment fraud behaves in the real world.

## Leakage Prevention

The model is designed to be scored **before the current transaction is executed.** Concretely:

- `newbalanceOrig`, `newbalanceDest` (only exist *after* a transaction is applied) and `isFlaggedFraud` (a rule output, not a transaction attribute) are **never** used as model inputs.
- Every historical/behavioral feature is computed using only transactions with a **strictly earlier** `step` than the current one — never the current transaction, and never same-step neighboring transactions (verified with a dedicated same-step controlled test, since the raw data has thousands of transactions sharing an identical step with no reliable sub-order).
- **Feature construction was validated by direct recomputation**, not just by code review: features were rebuilt from a train-only subset of the raw data and compared row-for-row against the same rows' features computed on the full dataset — they were identical, confirming no later-period information could influence earlier-period feature values.

Details: `src/features.py`, `notebooks/03_feature_engineering.ipynb`, `notebooks/04_temporal_split.ipynb`.

## Feature Engineering

The final model uses **40 features** (after the artifact-ablation step described below). They fall into five groups:

**Transaction features** — `amount`, `log_amount` (log1p-transformed), one-hot transaction-type indicators, and `TRANSFER`/`CASH_OUT` flags.

**Time features** — `simulated_hour`, `simulated_day`, `hour_of_day`, all derived from PaySim's simulated `step` (not real-world time).

**Balance features** — pre-transaction sender/receiver balances (`oldbalanceOrg`, `oldbalanceDest`) and zero-balance indicator flags.

**Sender historical behavior** — prior transaction count, total/mean/max amount, `TRANSFER`/`CASH_OUT` counts, distinct destination count, time since the sender's last transaction, and short-window transaction velocity (previous 1/6/24 steps).

**Receiver historical behavior** — prior transaction count, total/mean/max amount received, distinct sender count, time since the receiver's last transaction, and 24-step transaction velocity.

Two additional engineered features — `amount_to_sender_balance` and `amount_exceeds_sender_balance` — were built during initial development but are **excluded from the final model** (see Artifact Ablation). These features are not claimed to be universally predictive or causal; they are the features this specific model, on this specific dataset, found useful.

## Temporal Data Split

A chronological split was used — never a random one — because a random split would let the model train on transactions that occur, in simulated time, after some of its own evaluation transactions.

| Split | Steps | Rows | Fraud |
|---|---|---|---|
| Train | 1–323 | 4,463,587 | 3,643 |
| Validation | 324–378 | 980,416 | 564 |
| Test | 379–743 | 918,617 | 4,006 |

Boundaries were chosen as the earliest step at which cumulative row count reached ~70% and ~85% of the dataset, so no step's transactions are split across two sets.

**Important:** the test period has a substantially higher fraud rate (0.436%) than train (0.082%) or validation (0.058%). This is a PaySim simulation characteristic — legitimate transaction volume collapses in the later simulated days while fraud injection stays roughly constant — not a general property of real financial transaction streams, and it is not corrected for.

## Model Experiments

Three model families were trained under identical causal features and identical chronological splits, with no SMOTE, no random re-splitting, and no hyperparameter search. Classification metrics are reported at a fixed 0.5 threshold.

| Model | Imbalance strategy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | none | 0.9984 | 0.3033 | 0.4652 | 0.9934 | 0.7572 |
| Logistic Regression | `class_weight=balanced` | 0.1155 | 0.9576 | 0.2062 | 0.9940 | 0.6604 |
| Random Forest | none | 0.9990 | 0.9940 | 0.9965 | 1.0000 | 0.9997 |
| Random Forest | `class_weight=balanced_subsample` | 0.9997 | 0.9840 | 0.9918 | 1.0000 | 0.9999 |
| XGBoost | none (`scale_pos_weight=1`) | 0.9893 | 0.9019 | 0.9436 | 1.0000 | 0.9971 |
| XGBoost | `scale_pos_weight≈1224` | 0.9689 | 0.9713 | 0.9701 | 1.0000 | 0.9968 |

(Full detail and validation-set numbers: `notebooks/05_logistic_regression_baseline.ipynb`, `notebooks/06_random_forest_baseline.ipynb`, `notebooks/07_xgboost_baseline.ipynb`, `results/model_baseline_comparison_step7.csv`.)

This table is not a leaderboard used to promote a "best" model — the near-perfect Random Forest/XGBoost numbers are exactly what prompted the robustness investigation below, and the final model's methodology (not its raw test score) is what determined which configuration was carried forward.

## Artifact Ablation

The Random Forest and XGBoost models above scored suspiciously well — ROC-AUC and PR-AUC both near 1.0, and precision *and* recall both above 0.98 simultaneously at a single fixed threshold. Investigating rather than accepting this: feature-importance analysis showed two engineered features, `amount_to_sender_balance` and `amount_exceeds_sender_balance`, dominated both models (45%+ combined importance). These features expose the "drained sender account" pattern already flagged in EDA as a likely PaySim simulation artifact, not a confirmed general fraud behavior.

**Robustness experiment:** both features were removed, and Random Forest and XGBoost were retrained on the remaining 40 causal features with identical configurations otherwise.

| Comparison (test set) | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest — before → after | 0.9990 → 0.7389 | 0.9940 → 0.6146 | 0.9965 → 0.6710 | 1.0000 → 0.9904 | 0.9997 → 0.7396 |
| XGBoost — before → after | 0.9893 → 0.9428 | 0.9019 → 0.7526 | 0.9436 → 0.8370 | 1.0000 → 0.9993 | 0.9971 → 0.9191 |

Both models dropped substantially once the two artifact-linked features were removed — confirming a meaningful share of the original performance depended on them — but neither collapsed to a random-guessing floor, meaning the remaining causal features still carry real signal. This is reported as a robustness finding, not proof that the resulting model would generalize to real-world fraud. Detail: `notebooks/08_artifact_ablation.ipynb`, `results/artifact_ablation_comparison.csv`.

## Error Analysis

Rather than reporting only aggregate metrics, the ablated model was analyzed for *where* it errs (`notebooks/09_error_analysis.ipynb`), covering:

- False positives vs. true negatives, and false negatives vs. true positives, compared across transaction amount, balance state, and historical-activity features
- Errors broken down by transaction type (`TRANSFER` vs. `CASH_OUT`)
- Errors broken down by PaySim simulated time (day / hour-of-day)
- Predicted-probability distributions by confusion-matrix group
- The highest-confidence false positives and lowest-confidence false negatives

Key findings: false negatives outnumber false positives at the fixed threshold; missed fraud skews toward smaller amounts and `CASH_OUT` transactions; the model separates classes with a clear probability gap on average, though a subset of false negatives are confidently (not marginally) misclassified. **These error patterns are specific to this fitted model on this synthetic dataset** and are not claimed to represent how real fraud behaves.

## Final Model

The final portfolio model was selected by a **predefined robustness rationale**, not by picking the highest test score:

1. XGBoost was chosen for its nonlinear modeling capacity.
2. The two PaySim-artifact-linked features were excluded, per the ablation finding above.
3. The error analysis above was completed on this exact 40-feature configuration before any final test-set evaluation occurred.

| Property | Value |
|---|---|
| Model | XGBoost (`XGBClassifier`) |
| Features | 40 causal features (see above) |
| Class weighting | `scale_pos_weight=1.0` (unweighted) |
| Training data | Train + validation combined — 5,444,003 rows, 4,207 fraud |
| Test data (held out) | 918,617 rows, 4,006 fraud |
| Threshold | 0.5 (fixed, not tuned) |

**Final held-out test results on synthetic PaySim data:**

| Metric | Value |
|---|---|
| Precision | 0.9498 |
| Recall | 0.7184 |
| F1 | 0.8181 |
| ROC-AUC | 0.9992 |
| PR-AUC | 0.9122 |
| Confusion matrix | TN=914,459 · FP=152 · FN=1,128 · TP=2,878 |

Full detail, hyperparameters, and rationale: `results/final_model_metrics.json`, `results/final_model_card.md`, `notebooks/10_final_model_strategy.ipynb`. These are held-out test results on a synthetic dataset — they are not a measurement of real-world fraud-detection performance.

## Precision@K

In a real fraud-review workflow, analysts can typically only investigate a limited number of transactions. Precision@K / Recall@K ask: if only the top-K highest-scored transactions could be reviewed, what fraction would actually be fraud, and what fraction of all fraud would that capture?

| K | Precision@K | Recall@K |
|---|---|---|
| 100 | 1.0000 | 0.0250 |
| 500 | 1.0000 | 0.1248 |
| 1,000 | 1.0000 | 0.2496 |
| 5,000 | 0.7080 | 0.8837 |
| 10,000 | 0.3847 | 0.9603 |

Source: `results/final_model_precision_recall_at_k.csv`. This describes ranking behavior on the held-out synthetic test set — it does not imply integration into any real transaction-review queue.

## Streamlit Application

A small interactive demo (`app.py`) lets a user score a hypothetical transaction against the final model.

**Quick Demo** — inputs: transaction type, amount, sender previous balance, receiver previous balance, simulated step. All historical/behavioral features are set to their **validated no-history defaults** (the same defaults the underlying feature pipeline produces for a genuinely first-ever transaction) — this assumption is shown directly in the UI.

**Advanced Demo** — a curated subset of historical features can be entered as hypothetical values. **These historical values are hypothetical demonstration inputs, not observed customer history**, and the UI states this explicitly.

**Output** — the app displays the predicted fraud probability, the classification at the fixed 0.5 threshold, a probability visualization (explicitly labeled as illustrative, not a validated business risk policy), model-level (global) feature importance from `results/final_model_feature_importance.csv`, model performance metrics from `results/final_model_metrics.json`, and a persistent synthetic-data disclaimer. The feature-importance chart is global and model-level — it is **not** presented as an explanation of any individual prediction (that would require a method such as SHAP, which is intentionally not part of this project).

## Running the Project Locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd paysim-fraud-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This repository includes the source code, notebooks, figures, result tables, and the **final trained model** (`models/final_xgboost_fraud_detector.joblib`). It does **not** include the raw PaySim CSV (~493 MB), the generated processed Parquet datasets (~831 MB — the full feature set plus train/validation/test splits), or the other experimental model files (Logistic Regression, Random Forest, and intermediate XGBoost variants) — these are excluded due to their size and are not needed to review the code or use the final model. To re-run the data/feature-engineering notebooks from scratch, download the PaySim CSV yourself (see `src/data_utils.py` for the expected location, `data/raw/`) and regenerate the processed datasets by running the notebooks in order.

Run the Streamlit demo:

```bash
streamlit run app.py
```

Run the test suite:

```bash
python -m pytest -q
```

## Project Structure

```
paysim-fraud-detection/
├── app.py                    # Streamlit demo application
├── data/
│   ├── raw/                  # Original PaySim CSV (not committed — see .gitignore)
│   └── processed/            # Engineered features + chronological splits (parquet)
├── figures/
│   ├── eda/                  # EDA charts
│   └── error_analysis/       # Error-analysis charts
├── models/                   # Trained model artifacts (.joblib)
├── notebooks/                # 01–10, one per project stage
├── results/                  # Metrics, feature importance, model card, error tables
├── src/
│   ├── data_utils.py         # Raw data loading
│   ├── features.py           # Causal feature-engineering pipeline
│   ├── feature_definitions.py# Per-feature documentation
│   ├── split_data.py         # Chronological split utilities
│   ├── model_utils.py        # Training/evaluation utilities (all model families)
│   ├── app_features.py       # App-side input validation + feature construction
│   ├── prediction.py         # Cached model loading + prediction for the app
│   └── risk.py                # Threshold-based classification for the app
├── tests/
│   └── test_app_features.py  # App feature-construction and validation tests
├── requirements.txt
└── README.md
```

## Technology Stack

- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, metrics)
- XGBoost
- PyArrow (Parquet storage)
- Matplotlib, Seaborn (visualization)
- Streamlit (demo application)
- Joblib (model persistence)
- Jupyter (analysis notebooks)
- Pytest (application tests)

`requirements.txt` also lists `imbalanced-learn` and `shap`, which were considered but deliberately **not used** anywhere in this project (no SMOTE, no SHAP) — they are not part of the actual technology stack above.

## Reproducibility

- All model training uses fixed random seeds (`random_state=42`).
- The train/validation/test split is deterministic and chronological, not randomly sampled.
- The final model artifact, its metrics, its feature-importance table, and its model card are all saved under `results/` and `models/`, generated directly by `notebooks/10_final_model_strategy.ipynb`.
- Reload reproducibility was explicitly verified: reloading the saved final model and re-predicting on the test set reproduces the original predictions exactly.
- All ten notebooks and the application test suite are included so every step can be re-run.

Exact reproducibility across different hardware, OS, or library versions is not guaranteed — results were produced on the environment described in `requirements.txt`.

## Limitations

1. PaySim is a **synthetic** dataset.
2. Results should **not** be interpreted as real-world NPCI/UPI fraud-detection performance.
3. PaySim contains simulation-specific artifacts (see Artifact Ablation) that make parts of the task unusually easy in ways that may not transfer to real data.
4. Advanced Demo historical values in the Streamlit app are hypothetical, not observed customer history.
5. Quick Demo does not have access to real historical customer context — it uses validated no-history defaults.
6. The application is a **portfolio demonstration**, not a deployed product.
7. No real financial accounts or payment systems are connected to this project.
8. No production deployment has been performed.
9. No real-time transaction stream is connected.
10. Generalization to real-world fraud data is unverified and out of scope for this project.

## Future Improvements

- Evaluate on independent, legally/ethically available real-world fraud datasets, if any become accessible.
- Add calibrated probability evaluation (e.g., reliability diagrams).
- Select an operating threshold using an explicit, stated operational cost framework, rather than the default 0.5.
- Support batch scoring from an uploaded transaction log, so historical features can be constructed correctly rather than assumed.
- Add drift monitoring for a hypothetical deployed setting.
- Add basic model-governance documentation (versioning, approval, retraining triggers).
- Strengthen temporal backtesting with multiple rolling train/test windows.
- More robust historical-state handling for entities with partial or ambiguous history.
- Explore a production deployment architecture (out of scope for this portfolio project).

## Disclaimer

This project is an educational/portfolio demonstration using synthetic PaySim data. It is not connected to NPCI, UPI, banks, or real customer accounts, and its outputs should not be used for real financial decisions.
