---
name: model-training
description: Trains ML models with adaptive hyperparameters
---

# Model Training Skill

Use this skill to train and evaluate models on `train.csv`/`test.csv`.

## Available Scripts

### `scripts/train_automl.py`
Loads `train.csv` and `test.csv`, imputes missing values, clips outliers, applies target
encoding to categorical `feature_*` columns, trains one or more models (RandomForest,
ExtraTrees, GradientBoosting, LogisticRegression) via cross-validation, and writes
`final_submission.csv` with predicted probabilities for `target`.

**Instructions**:
1. Run with `run_command("python scripts/train_automl.py --experiment <mode>")`, where
   `<mode>` is one of `rf`, `xgb`, `lgbm`, `ensemble` (default `ensemble`).
2. Check the printed cross-validation AUC score before submitting.
3. Call `submit_predictions("final_submission.csv")` after a successful run.
