#!/usr/bin/env python3
"""
Unit tests for AAP-Agent training pipeline.
Run with: python -m pytest test_submission/test_train_automl.py -v
Or: python test_submission/test_train_automl.py
"""
import os
import sys
import tempfile
import pandas as pd
import numpy as np
import unittest

# Add submission scripts to path
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "submission", 
                          "skills", "model-training", "scripts")
SCRIPT_DIR = os.path.abspath(SCRIPT_DIR)

def create_test_data(n_train=500, n_test=200, n_features=10, n_cat=3, seed=42):
    """Create small test dataset."""
    rng = np.random.RandomState(seed)
    
    X_num = rng.randn(n_train + n_test, n_features - n_cat)
    cat_levels = [rng.choice([f"cat_{i}_{j}" for j in range(4)], size=n_train + n_test) 
                  for i in range(n_cat)]
    X_cat = np.column_stack(cat_levels)
    
    logit = X_num[:, 0] * 0.5 + X_num[:, 1] * 0.3 + rng.randn(n_train + n_test) * 0.5
    y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    
    num_cols = [f"feature_{i}" for i in range(n_features - n_cat)]
    cat_cols = [f"feature_{i}" for i in range(n_features - n_cat, n_features)]
    all_cols = num_cols + cat_cols
    
    X = np.column_stack([X_num, X_cat])
    df = pd.DataFrame(X, columns=all_cols)
    df.insert(0, "row_id", range(n_train + n_test))
    df["target"] = y
    
    return df.iloc[:n_train], df.iloc[n_train:]

class TestTrainAutoml(unittest.TestCase):
    
    def setUp(self):
        """Create temp directory with test data."""
        self.tmpdir = tempfile.mkdtemp()
        train, test = create_test_data()
        train.to_csv(os.path.join(self.tmpdir, "train.csv"), index=False)
        test.drop(columns=["target"]).to_csv(os.path.join(self.tmpdir, "test.csv"), index=False)
    
    def test_train_automl_ensemble(self):
        """Test that ensemble training produces valid output."""
        import subprocess
        
        script = os.path.join(SCRIPT_DIR, "train_automl.py")
        result = subprocess.run(
            [sys.executable, script, "--experiment", "ensemble"],
            cwd=self.tmpdir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        self.assertEqual(result.returncode, 0, 
                         f"Training failed: {result.stderr}")
        
        # Check output file exists
        sub_path = os.path.join(self.tmpdir, "final_submission.csv")
        self.assertTrue(os.path.exists(sub_path), "Submission file not created")
        
        # Validate output format
        sub = pd.read_csv(sub_path)
        self.assertIn("row_id", sub.columns)
        self.assertIn("target", sub.columns)
        self.assertEqual(len(sub), 200)
        self.assertFalse(sub["target"].isna().any())
        self.assertTrue((sub["target"] >= 0).all() and (sub["target"] <= 1).all())
    
    def test_train_automl_rf(self):
        """Test RF-only training."""
        import subprocess
        
        script = os.path.join(SCRIPT_DIR, "train_automl.py")
        result = subprocess.run(
            [sys.executable, script, "--experiment", "rf"],
            cwd=self.tmpdir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        self.assertEqual(result.returncode, 0,
                         f"RF training failed: {result.stderr}")
        
        sub_path = os.path.join(self.tmpdir, "final_submission.csv")
        self.assertTrue(os.path.exists(sub_path))
    
    def test_data_loading(self):
        """Test that data loads correctly."""
        train = pd.read_csv(os.path.join(self.tmpdir, "train.csv"))
        test = pd.read_csv(os.path.join(self.tmpdir, "test.csv"))
        
        self.assertIn("row_id", train.columns)
        self.assertIn("target", train.columns)
        self.assertIn("row_id", test.columns)
        self.assertNotIn("target", test.columns)
    
    def test_categorical_detection(self):
        """Test that categorical columns are detected."""
        train = pd.read_csv(os.path.join(self.tmpdir, "train.csv"))
        cat_cols = [c for c in train.columns 
                    if train[c].dtype == "object" and c.startswith("feature_")]
        self.assertEqual(len(cat_cols), 3, "Should detect 3 categorical columns")

class TestValidateSubmission(unittest.TestCase):
    
    def test_valid_submission(self):
        """Test validation of a valid submission."""
        from validate_submission import validate
        
        tmpdir = tempfile.mkdtemp()
        sub = pd.DataFrame({"row_id": [1, 2, 3], "target": [0.3, 0.7, 0.5]})
        test = pd.DataFrame({"row_id": [1, 2, 3]})
        
        sub_path = os.path.join(tmpdir, "sub.csv")
        test_path = os.path.join(tmpdir, "test.csv")
        sub.to_csv(sub_path, index=False)
        test.to_csv(test_path, index=False)
        
        errors, warnings = validate(sub_path, test_path)
        self.assertEqual(len(errors), 0, f"Valid submission should have no errors: {errors}")
    
    def test_missing_column(self):
        """Test validation catches missing columns."""
        from validate_submission import validate
        
        tmpdir = tempfile.mkdtemp()
        sub = pd.DataFrame({"row_id": [1, 2, 3]})  # Missing target
        sub_path = os.path.join(tmpdir, "sub.csv")
        sub.to_csv(sub_path, index=False)
        
        errors, warnings = validate(sub_path)
        self.assertTrue(any("target" in e for e in errors))
    
    def test_nan_values(self):
        """Test validation catches NaN values."""
        from validate_submission import validate
        
        tmpdir = tempfile.mkdtemp()
        sub = pd.DataFrame({"row_id": [1, 2, 3], "target": [0.3, np.nan, 0.5]})
        sub_path = os.path.join(tmpdir, "sub.csv")
        sub.to_csv(sub_path, index=False)
        
        errors, warnings = validate(sub_path)
        self.assertTrue(any("NaN" in e for e in errors))

if __name__ == "__main__":
    # Add test_submission to path for imports
    sys.path.insert(0, os.path.dirname(__file__))
    unittest.main(verbosity=2)
