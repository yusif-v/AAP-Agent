---
name: data-analysis
description: Analyzes dataset structure and identifies feature types
---

# Data Analysis Skill

Use this skill first, before training, to understand the dataset.

## Instructions

1. Inspect shape, dtypes, and missing values:
   `run_command("python -c \"import pandas as pd; df=pd.read_csv('train.csv'); print(df.shape); print(df.dtypes); print(df.isnull().sum())\"")`
2. Identify categorical feature columns (object dtype, prefixed `feature_`) and numeric feature columns.
3. Check target balance: `run_command("python -c \"import pandas as pd; print(pd.read_csv('train.csv')['target'].value_counts(normalize=True))\"")`
4. Use these findings to decide feature engineering and model choices in the model-training skill.
