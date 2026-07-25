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
ExtraTrees, XGBoost, LightGBM, GradientBoosting, CatBoost) via 5-fold cross-validation,
stacks them with a logistic-regression meta-learner, and writes `final_submission.csv`
with predicted probabilities for `target`.

**Instructions**:
1. Run with `run_command("python scripts/train_automl.py --experiment <mode>")`, where
   `<mode>` is one of `rf`, `et`, `xgb`, `lgbm`, `gb`, `cb`, `ensemble` (default `ensemble`).
   `cb` (CatBoost) only trains if categorical `feature_*` columns are present.
2. Add `--tune` to run RandomizedSearchCV (sklearn-only, no internet needed) on the
   best-performing model from the initial CV screen before it joins the ensemble/stack.
   Use this once a baseline experiment shows a clear best model.
3. Check the printed cross-validation AUC score before submitting.
4. Call `submit_predictions("final_submission.csv")` after a successful run.
