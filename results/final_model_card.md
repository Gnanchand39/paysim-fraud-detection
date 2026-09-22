# Model Card: Final PaySim Fraud-Detection Portfolio Model

## 1. Model purpose

A portfolio/demonstration fraud-detection model built to show an end-to-end, causally-sound, honestly-evaluated machine learning workflow on transaction data. It is **not** a production fraud-detection system.

## 2. Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a **synthetic** mobile-money transaction simulator (Lopez-Rojas, Elmir & Axelsson, 2016). 6,362,620 transactions, 8,213 labeled as fraud (0.129%). PaySim is **not real banking, UPI, or NPCI transaction data** — it does not represent real customer behavior, real fraud tactics, or a real payment network's transaction distribution.

## 3. Prediction task

Binary classification: given a transaction, predict whether it is fraudulent (`isFraud`). Fraud in this dataset occurs exclusively in `TRANSFER` and `CASH_OUT` transactions (established in Step 2 EDA).

## 4. Prediction point

The model is designed to be scorable **before** a transaction is executed. Every feature uses only information available at that moment: the transaction's own request-time attributes (type, amount, pre-transaction balances) and a customer's **strictly earlier** transaction history (verified causally in Step 3, with an explicit same-step controlled test). `newbalanceOrig`, `newbalanceDest` (post-transaction state) and `isFlaggedFraud` (a rule output) are never used as inputs.

## 5. Feature set

40 causal features (full definitions in `src/feature_definitions.py` and `src/features.py`):
transaction features (amount, log-amount, one-hot type, type flags), time features (simulated hour/day/hour-of-day), pre-transaction balance features (old balances, zero-balance flags), historical sender/receiver behavioral features (transaction counts, totals, means, time-since-last, distinct counterparties), and velocity features (transactions in the preceding 1/6/24 steps).

## 6. Excluded leakage/artifact features

- `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud` — excluded from the start (Step 3) as post-transaction/rule-derived information not available at decision time.
- `amount_to_sender_balance`, `amount_exceeds_sender_balance` — **deliberately removed for this final model** (Step 8 ablation). These two engineered features were the dominant drivers of the near-perfect performance seen in the original Step 6/7 Random Forest and XGBoost models. They encode a PaySim-specific synthetic pattern: simulated fraud transactions almost always drain the sender's account exactly (`oldbalanceOrg == amount`, `newbalanceOrig == 0`), a pattern present in 97.5% of fraud vs. 0% of legitimate transactions (Step 2 EDA) — very likely an artifact of how PaySim's fraud-injection logic works, not a guaranteed real-world fraudster behavior. Their removal is a robustness choice, not a performance optimization.

## 7. Training protocol

- **Model family:** XGBoost (`XGBClassifier`), gradient-boosted decision trees.
- **Configuration** (fixed, not tuned): `n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=8`.
- **Class weighting:** `scale_pos_weight=1.0` (unweighted) — identical to the Step 8 ablation model; not re-derived or changed for this final model.
- **Training data:** `train.parquet` (steps 1-323) + `validation.parquet` (steps 324-378) combined = 5,444,003 rows, 4,207 fraud. Validation was folded into training here because model selection (which features, which model family, which weighting strategy) was already fixed by the Steps 6-9 robustness rationale *before* this final fit — validation's original purpose (model comparison) was already served.
- **No hyperparameter search, no early stopping, no SMOTE** — this is the same fixed configuration used throughout Steps 7-9.

## 8. Evaluation protocol

Evaluated **once** on the held-out test set (`test.parquet`, steps 379-743, 918,617 rows, 4,006 fraud), which was never used in training or in any prior model-selection decision. Threshold fixed at 0.5 (not tuned). Metrics are reported at this single threshold plus threshold-independent ROC-AUC/PR-AUC and ranking metrics (Precision@K/Recall@K).

## 9. Test metrics (single evaluation, threshold = 0.5)

| Metric | Value |
|---|---|
| TN | 914,459 |
| FP | 152 |
| FN | 1,128 |
| TP | 2,878 |
| Precision | 0.9498 |
| Recall | 0.7184 |
| F1 | 0.8181 |
| ROC-AUC | 0.999173 |
| PR-AUC | 0.912238 |
| False Positive Rate | 0.000166 |
| False Negative Rate | 0.281578 |

Precision@K / Recall@K: K=100 (100.0% / 2.5%), K=500 (100.0% / 12.5%), K=1,000 (100.0% / 25.0%), K=5,000 (70.8% / 88.4%), K=10,000 (38.5% / 96.0%). Full table: `results/final_model_precision_recall_at_k.csv`.

Note: these numbers differ from Step 8's ablation model (trained on train-only) because this final model was trained on more data (train+validation) — a direct, expected consequence of the training protocol, not re-tuning.

## 10. Error characteristics

From Step 9's error analysis of the closely related train-only ablated model (patterns expected to carry over, not re-verified independently for this exact artifact): false negatives outnumber false positives; missed fraud skews toward smaller transaction amounts and CASH_OUT type; false positives concentrate in TRANSFER/CASH_OUT transactions resembling the fraud risk profile; recall increases with transaction amount, reaching 100% in the top percentile.

## 11. Interpretability

Gain-based XGBoost feature importance (`results/final_model_feature_importance.csv`) is led by `is_transfer_or_cash_out`, `destination_balance_zero`, and `prior_receiver_has_history`. **Gain-based feature importance reflects model-specific predictive usefulness and should not be interpreted as causal importance.** No feature is described as a cause of fraud.

## 12. Limitations

- PaySim is synthetic. It should not be described as real banking, UPI, or NPCI transaction data.
- Fraud patterns in this dataset may contain simulation-specific artifacts beyond the two features already removed.
- The removed balance-ratio features were strongly associated with a synthetic drained-account pattern (Steps 2/3); their removal is a documented robustness choice, not proof the remaining model is artifact-free.
- Step 9's error analysis is dataset- and model-specific and was performed on a closely related (train-only) model, not re-run on this exact final artifact.
- Simulated time (`step`/`simulated_day`/`hour_of_day`) is not real-world transaction time.
- Test performance here should not be interpreted as production fraud-detection performance.
- This project does not establish causal relationships between any feature and fraud.
- No claim is made about NPCI's actual fraud-detection systems or their performance.
- No production deployment is claimed or implied by this model or model card.

## 13. Reproducibility

- Model artifact: `models/final_xgboost_fraud_detector.joblib`.
- Verified: reloading the saved model and re-predicting on the test set produces predictions identical to the original evaluation (max absolute probability difference = 0.0, classifications at threshold 0.5 100% identical).
- Full training/evaluation code: `notebooks/10_final_model_strategy.ipynb`, using `src/model_utils.py` and `src/features.py`.
- Random state fixed at 42 throughout.
