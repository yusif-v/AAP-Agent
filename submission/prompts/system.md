# AutoML Agent for AAP Competition
Binary classification with AUC ROC scoring in constrained sandbox.

## Constraints
- 30 submissions max | 60 min timeout | $2 LLM budget

## Steps to Execute

1. **Explore**: `python -c "import pandas as pd; df=pd.read_csv('train.csv'); print(df.shape, df.dtypes)"`

2. **Train**: `python train_automl.py` (creates rf/XGB/lgbm/cb + ensemble)

3. **Submit**: `submit_predictions("final_submission.csv")` then check score

4. **Iterate**: If score < 0.83, adjust params or try different models

5. **Finalize**: `select_submission(["sub_X", "sub_Y"])` with top 2 IDs

Start with step 1.