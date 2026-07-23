#!/usr/bin/env python3
"""
Local evaluation script for AAP-Agent submission.
Runs train_automl.py locally, validates output, and reports metrics.

Usage:
    python run_local_eval.py [--experiment ENSEMBLE] [--data-dir test_data]
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

def run_training(data_dir, experiment="ensemble"):
    """Run train_automl.py in the data directory."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "submission", 
                               "skills", "model-training", "scripts", "train_automl.py")
    script_path = os.path.abspath(script_path)
    
    print(f"Running training (experiment={experiment}) in {data_dir}...")
    result = subprocess.run(
        [sys.executable, script_path, "--experiment", experiment],
        cwd=data_dir,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return False
    
    return True

def validate_submission(data_dir, submission_file="final_submission.csv"):
    """Validate submission format and compute local AUC."""
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(data_dir, submission_file)
    
    if not os.path.exists(sub_path):
        print(f"ERROR: Submission file {sub_path} not found")
        return False, 0.0
    
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)
    
    # Check format
    if "row_id" not in sub.columns or "target" not in sub.columns:
        print(f"ERROR: Submission must have 'row_id' and 'target' columns")
        return False, 0.0
    
    if len(sub) != len(test):
        print(f"ERROR: Submission has {len(sub)} rows, expected {len(test)}")
        return False, 0.0
    
    # Check for NaN
    if sub["target"].isna().any():
        print("ERROR: Submission contains NaN values")
        return False, 0.0
    
    # Check range
    if sub["target"].min() < 0 or sub["target"].max() > 1:
        print(f"WARNING: Target values out of [0,1] range: [{sub['target'].min()}, {sub['target'].max()}]")
    
    print(f"Submission format: OK ({len(sub)} rows)")
    print(f"Target range: [{sub['target'].min():.4f}, {sub['target'].max():.4f}]")
    print(f"Target mean: {sub['target'].mean():.4f}")
    
    return True, sub["target"].mean()

def compute_cv_auc(data_dir, experiment="ensemble", n_folds=3):
    """Compute cross-validation AUC using the training data."""
    train_path = os.path.join(data_dir, "train.csv")
    train = pd.read_csv(train_path)
    
    y = train["target"].values
    X = train.drop(columns=["row_id", "target"])
    
    # Quick CV with RandomForest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Encode categoricals
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Fill NaN
    X = X.fillna(-999)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=n_folds, scoring="roc_auc")
    
    print(f"CV AUC (RandomForest, {n_folds}-fold): {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local AAP evaluation")
    parser.add_argument("--experiment", default="ensemble", 
                        choices=["rf", "et", "xgb", "lgbm", "cb", "ensemble"])
    parser.add_argument("--data-dir", default="test_data")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, just validate existing submission")
    args = parser.parse_args()
    
    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    
    # Generate test data if not exists
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("Generating test data...")
        gen_script = os.path.join(os.path.dirname(__file__), "generate_test_data.py")
        subprocess.run([sys.executable, gen_script], cwd=os.path.dirname(__file__))
    
    if not args.skip_training:
        success = run_training(data_dir, args.experiment)
        if not success:
            print("Training failed!")
            sys.exit(1)
    
    # Validate submission
    print("\n--- Validation ---")
    valid, mean_pred = validate_submission(data_dir)
    
    # Compute CV AUC
    print("\n--- Cross-Validation ---")
    cv_auc = compute_cv_auc(data_dir)
    
    print(f"\n--- Summary ---")
    print(f"Submission valid: {valid}")
    print(f"Mean prediction: {mean_pred:.4f}")
    print(f"CV AUC: {cv_auc:.4f}")
    
    if cv_auc > 0.80:
        print("PASS: CV AUC > 0.80")
    else:
        print("FAIL: CV AUC < 0.80, needs improvement")

if __name__ == "__main__":
    main()
