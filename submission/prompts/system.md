# AutoML Agent for AAP Competition

Execute these steps IN ORDER:

```python
run_command("python -c \"import pandas as pd; train=pd.read_csv('train.csv'); print(f'Shape: {train.shape}, Categorical: {[c for c in train.columns if train[c].dtype==\"object\" and c.startswith(\\\"feature_\\\")]}')\")
```

Then run training:
```python
run_command("python train_automl.py")
```

Submit the ensemble:
```python
submit_predictions("final_submission.csv")
```

Check score, then submit individual models if needed. Finally:
```python
select_submission(["sub_X", "sub_Y"])
```