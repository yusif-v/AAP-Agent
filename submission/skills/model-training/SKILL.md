---
name: model-training
description: Trains ML models with adaptive hyperparameters
---

# Model Training Skill

Use this skill to train and evaluate models on `train.csv`/`test.csv`.

## Available Scripts

### `scripts/train_automl.py`
Loads `train.csv` and `test.csv`, imputes missing values, clips outliers, applies out-of-fold target
encoding to categorical `feature_*` columns, decodes ordinal columns, generates MI-top feature interactions,
trains CatBoost (seed-averaged), ExtraTrees, XGBoost, and LogisticRegression, blends them with OOF-optimized
NNLS weights (with a conditional soft floor when top-2 OOF models are within 0.01 AUC), and writes
`final_submission.csv` with predicted probabilities for `target`.

**Instructions**:
1. The skill directory is mounted at `skills/model_training/` in the sandbox. Run the script from the root working directory:
   `run_command("python skills/model_training/scripts/train_automl.py --experiment <mode>")`, where
   `<mode>` is one of `rf`, `et`, `xgb`, `lgbm`, `gb`, `cb`, `lr`, `ensemble` (default `ensemble`).
2. Add `--tune` to run RandomizedSearchCV (sklearn-only, no internet needed) on the
   best-performing model from the initial CV screen before it joins the ensemble.
   Use this once a baseline experiment shows a clear best model.
3. Check the printed cross-validation AUC score before submitting.
4. Call `submit_predictions("final_submission.csv")` after a successful run.

## Ensemble Notes
The default `ensemble` mode trains four genuinely different model families:
- **CatBoost** (native categorical handling, seed-averaged over 1 seed for large datasets; adaptive depth/iterations by dataset size)
- **ExtraTrees** (bagging, adaptive n_estimators/max_depth, n_jobs=2)
- **XGBoost** (boosted trees with early stopping, adaptive n_estimators/depth, n_jobs=2)
- **LogisticRegression** (linear baseline with StandardScaler, C=0.1 — strong on weak-signal data)

Model complexity adapts to dataset size via `get_adaptive_params()`: tiny datasets (<500 rows) use 1 CB seed with shallow trees; large datasets (>=10k rows) use 1 CB seed with 500 iterations (reduced from 2000 for memory stability on Kaggle).

These are blended using non-negative least squares (NNLS) optimized on out-of-fold (OOF) predictions. OOF predictions are cached from the CV pass, avoiding double-training. A conditional soft floor ensures both top-2 models get at least 10% weight when their OOF AUCs are within 0.01 AUC of each other — this hedges against OOF-vs-test noise where NNLS might over-zero a marginally-inferior model that's actually better on test data.

**Scoreboard (16 splits)**: mean=0.8025, vs XGBoost-only mean=0.7854 (+0.017 improvement).
The remaining gap to 0.82 is limited by 3 splits with fundamentally weak signal (max |linear corr| < 0.18)
where even the best models plateau at AUC ~0.65.
