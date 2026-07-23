#!/usr/bin/env python3
"""
Generate synthetic test data mimicking the AAP competition format.
Creates train.csv, test.csv, sample_submission.csv with feature_* columns.
"""
import numpy as np
import pandas as pd
import os

def generate_data(n_train=10000, n_test=5000, n_features=20, n_categorical=5, seed=42):
    rng = np.random.RandomState(seed)
    
    # Numerical features
    X_num = rng.randn(n_train + n_test, n_features - n_categorical)
    
    # Categorical features (as strings)
    cat_levels = [rng.choice([f"cat_{i}_{j}" for j in range(rng.randint(3, 8))], 
                             size=n_train + n_test) 
                  for i in range(n_categorical)]
    X_cat = np.column_stack(cat_levels)
    
    # Target: depends on numerical features with some noise
    # This creates a learnable signal (AUC should be achievable > 0.80)
    logit = (X_num[:, 0] * 0.5 + X_num[:, 1] * 0.3 + X_num[:, 2] * -0.4 + 
             X_num[:, 3] * 0.2 + rng.randn(n_train + n_test) * 0.5)
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    
    # Build DataFrames
    num_cols = [f"feature_{i}" for i in range(n_features - n_categorical)]
    cat_cols = [f"feature_{i}" for i in range(n_features - n_categorical, n_features)]
    all_cols = num_cols + cat_cols
    
    X = np.column_stack([X_num, X_cat])
    df = pd.DataFrame(X, columns=all_cols)
    df.insert(0, "row_id", range(n_train + n_test))
    df["target"] = y
    
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].drop(columns=["row_id", "target"]).copy()
    test_with_id = df.iloc[n_train:][["row_id"] + list(test.columns)]
    
    # Sample submission
    sample_sub = pd.DataFrame({
        "row_id": test_with_id["row_id"],
        "target": 0.5
    })
    
    return train, test_with_id, sample_sub, cat_cols

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(data_dir, exist_ok=True)
    
    train, test, sample_sub, cat_cols = generate_data()
    
    train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    sample_sub.to_csv(os.path.join(data_dir, "sample_submission.csv"), index=False)
    
    # Save cat_cols info
    with open(os.path.join(data_dir, "cat_cols.txt"), "w") as f:
        f.write("\n".join(cat_cols))
    
    print(f"Generated test data in {data_dir}/")
    print(f"  train.csv: {train.shape}")
    print(f"  test.csv: {test.shape}")
    print(f"  sample_submission.csv: {sample_sub.shape}")
    print(f"  Categorical columns: {cat_cols}")
    print(f"  Target distribution: {train['target'].value_counts().to_dict()}")
