#!/usr/bin/env python3
"""
Validate submission CSV format for AAP competition.

Usage:
    python validate_submission.py submission.csv [--test-csv test.csv]
"""
import sys
import pandas as pd
import numpy as np

def validate(submission_path, test_path=None):
    """Validate a submission file."""
    errors = []
    warnings = []
    
    # Load submission
    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        return [f"Cannot read submission file: {e}"], []
    
    # Check required columns
    if "row_id" not in sub.columns:
        errors.append("Missing 'row_id' column")
    if "target" not in sub.columns:
        errors.append("Missing 'target' column")
    
    if errors:
        return errors, warnings
    
    # Check row count
    if test_path:
        test = pd.read_csv(test_path)
        if len(sub) != len(test):
            errors.append(f"Row count mismatch: submission has {len(sub)}, test has {len(test)}")
    
    # Check for NaN
    if sub["target"].isna().any():
        errors.append(f"Submission has {sub['target'].isna().sum()} NaN values")
    
    # Check target range
    if sub["target"].min() < 0:
        errors.append(f"Target min is {sub['target'].min()}, should be >= 0")
    if sub["target"].max() > 1:
        errors.append(f"Target max is {sub['target'].max()}, should be <= 1")
    
    # Check row_id matches test
    if test_path:
        test = pd.read_csv(test_path)
        if "row_id" in test.columns:
            if not set(sub["row_id"]) == set(test["row_id"]):
                errors.append("row_id values don't match test set")
    
    # Warnings
    if sub["target"].std() < 0.01:
        warnings.append("Very low variance in predictions - model may not be learning")
    
    if sub["target"].mean() < 0.1 or sub["target"].mean() > 0.9:
        warnings.append(f"Mean prediction {sub['target'].mean():.4f} is extreme - check for data leakage")
    
    return errors, warnings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate AAP submission")
    parser.add_argument("submission", help="Path to submission CSV")
    parser.add_argument("--test-csv", default=None, help="Path to test.csv for row count check")
    args = parser.parse_args()
    
    errors, warnings = validate(args.submission, args.test_csv)
    
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    if not errors:
        print("VALID: Submission format is correct")
        sys.exit(0)
    else:
        print("INVALID: Submission has errors")
        sys.exit(1)
