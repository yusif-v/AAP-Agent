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
trains CatBoost (seed-averaged over 5 seeds), ExtraTrees, and XGBoost, blends them with OOF-optimized
NNLS weights, and writes `final_submission.csv` with predicted probabilities for `target`.

**Instructions**:
1. The skill directory is mounted at `skills/model_training/` in the sandbox. Run the script from the root working directory:
   `run_command("python skills/model_training/scripts/train_automl.py --experiment <mode>")`, where
   `<mode>` is one of `rf`, `et`, `xgb`, `lgbm`, `gb`, `cb`, `ensemble` (default `ensemble`).
2. Add `--tune` to run RandomizedSearchCV (sklearn-only, no internet needed) on the
   best-performing model from the initial CV screen before it joins the ensemble.
   Use this once a baseline experiment shows a clear best model.
3. Check the printed cross-validation AUC score before submitting.
4. Call `submit_predictions("final_submission.csv")` after a successful run.

## Ensemble Notes
The default `ensemble` mode trains three genuinely different model families:
- **CatBoost** (native categorical handling, seed-averaged over 5 seeds)
- **ExtraTrees** (bagging)
- **XGBoost** (boosted trees with early stopping)

These are blended using non-negative least squares (NNLS) optimized on out-of-fold predictions, replacing
the older stacking meta-learner approach for better transparency and stability.
