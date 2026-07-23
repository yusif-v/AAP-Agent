#!/usr/bin/env python3
"""
Entry point for Kaggle AAP Agent submission.
This script runs the autonomous agent in the Kaggle sandbox.
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import warnings
import itertools
import zipfile
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


def load_data():
    """Load competition data from various sources."""
    # Method 1: Check for mounted competition data at /kaggle/input/
    input_dir = '/kaggle/input/autonomous-agent-prediction-beta'
    if os.path.exists(input_dir):
        print(f"Found mounted data at: {input_dir}")
        # Check for train.csv directly
        train_path = os.path.join(input_dir, 'train.csv')
        test_path = os.path.join(input_dir, 'test.csv')
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Loading data from mounted location")
            return pd.read_csv(train_path), pd.read_csv(test_path)
        
        # Check for multi-split structure
        data_dirs = glob.glob(os.path.join(input_dir, 'train_*'))
        if data_dirs:
            print(f"Found multi-split data: {data_dirs[:3]}...")
            # Use first split
            first_dir = data_dirs[0]
            train_path = os.path.join(first_dir, 'train.csv')
            test_path = os.path.join(first_dir, 'test.csv')
            if os.path.exists(train_path) and os.path.exists(test_path):
                print(f"Loading data from: {first_dir}")
                return pd.read_csv(train_path), pd.read_csv(test_path)
    
    # Method 2: Check current directory
    if os.path.exists('train.csv') and os.path.exists('test.csv'):
        print("Loading data from current directory")
        return pd.read_csv('train.csv'), pd.read_csv('test.csv')
    
    # Method 3: Check data/train_XX pattern
    data_dirs = glob.glob('data/train_*')
    if data_dirs:
        first_dir = data_dirs[0]
        train_path = os.path.join(first_dir, 'train.csv')
        test_path = os.path.join(first_dir, 'test.csv')
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Loading data from: {first_dir}")
            return pd.read_csv(train_path), pd.read_csv(test_path)
    
    # Method 4: Search recursively
    train_files = glob.glob('**/train.csv', recursive=True)
    test_files = glob.glob('**/test.csv', recursive=True)
    if train_files and test_files:
        print(f"Found data via recursive search")
        return pd.read_csv(train_files[0]), pd.read_csv(test_files[0])
    
    return None, None


def download_data():
    """Download competition data using Kaggle API."""
    print("Attempting to download competition data...")
    
    try:
        import kaggle
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Download competition data
        print("Downloading...")
        kaggle.api.competition_download_files(
            'autonomous-agent-prediction-beta',
            path='.',
            force=True,
            quiet=False
        )
        
        # Find and unzip
        zip_files = glob.glob('*.zip')
        for zf in zip_files:
            print(f"Unzipping {zf}...")
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zf)
        
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for training."""
    experiment = 'ensemble'
    if '--experiment' in sys.argv:
        idx = sys.argv.index('--experiment')
        if idx + 1 < len(sys.argv):
            experiment = sys.argv[idx + 1]
    
    print("Loading competition data...")
    
    # Try to load data
    train, test = load_data()
    
    if train is None or test is None:
        print("Data not found, trying to download...")
        if not download_data():
            print("ERROR: Could not load or download data")
            print(f"Current directory contents: {os.listdir('.')}")
            sys.exit(1)
        train, test = load_data()
        if train is None or test is None:
            print("ERROR: Still could not load data after download")
            sys.exit(1)
    
    print(f"Loaded train shape: {train.shape}, test shape: {test.shape}")
    
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c.startswith('feature_')]
    num_cols = [c for c in train.columns if c not in cat_cols and c not in ['row_id', 'target']]
    y = train['target'].values
    X_train = train.drop(columns=['row_id', 'target'])
    X_test = test.drop(columns=['row_id'])
    
    # Impute
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')
        else:
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())
    
    # Outlier handling
    for col in num_cols:
        if col in X_train.columns:
            lower, upper = X_train[col].quantile([0.01, 0.99])
            X_train[col] = X_train[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)
    
    # Target encoding
    for col in cat_cols:
        target_mean = train.groupby(col)['target'].mean()
        global_mean = train['target'].mean()
        X_train[col + '_te'] = X_train[col].map(target_mean).fillna(global_mean)
        X_test[col + '_te'] = X_test[col].map(target_mean).fillna(global_mean)
    
    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)
    
    # Feature interactions
    top_num_cols = num_cols[:5]
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        if col1 in X_train.columns and col2 in X_train.columns:
            new_col = f"{col1}_{col2}_mul"
            X_train[new_col] = (X_train[col1] * X_train[col2]).astype(float)
            X_test[new_col] = (X_test[col1] * X_test[col2]).astype(float)
    
    # Feature selection
    all_features = X_train.columns.tolist()
    mi_scores = mutual_info_classif(X_train, y, random_state=42)
    mi_df = pd.DataFrame({'feature': all_features, 'mi_score': mi_scores})
    top_features = mi_df.nlargest(min(30, len(all_features)), 'mi_score')['feature'].tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    # Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)
    
    def get_model(name, n_samples):
        n_est = 100 if n_samples < 5000 else 150
        if name == 'rf':
            return RandomForestClassifier(n_estimators=n_est, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1)
        if name == 'et':
            return ExtraTreesClassifier(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1)
        if name == 'xgb':
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        if name == 'lgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        if name == 'gb':
            return GradientBoostingClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.1, random_state=42)
        return None

    def get_cv_score(model, X, y, n_folds=3):
        n_folds = min(n_folds, 3)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean(), scores.std()

    n_samples = len(train)
    predictions = []
    model_scores = {}
    trained_models = {}

    models_to_train = [('rf', 'rf'), ('et', 'et'), ('xgb', 'xgb'), ('lgbm', 'lgbm'), ('gb', 'gb')]
    if experiment != 'ensemble':
        models_to_train = [(m, k) for m, k in models_to_train if m == experiment]

    for name, key in models_to_train:
        try:
            model = get_model(name, n_samples)
            if model is None:
                continue
            
            cv_score, cv_std = get_cv_score(model, X_train, y)
            model_scores[name] = cv_score
            print(f"{name}: CV={cv_score:.4f} (+/- {cv_std:.4f})")
            
            model.fit(X_train, y)
            trained_models[name] = model
            pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
            pd.DataFrame({'row_id': test['row_id'], 'target': pred}).to_csv(f'{key}_pred.csv', index=False)
        except Exception as e:
            print(f"{name}: Failed - {e}")
            continue

    # Ensemble
    if len(predictions) >= 3:
        stacking_preds = []
        for name, key in models_to_train[:5]:
            if name in model_scores:
                model = get_model(name, n_samples)
                oof_preds = cross_val_predict(model, X_train, y, cv=3, method='predict_proba')[:, 1]
                stacking_preds.append(oof_preds)
        
        if len(stacking_preds) >= 3:
            meta_X = np.column_stack(stacking_preds)
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            meta_learner.fit(meta_X, y)
            
            test_preds = []
            for name, key in models_to_train[:5]:
                if name in model_scores:
                    model = trained_models.get(name) or get_model(name, n_samples)
                    if model:
                        if name not in trained_models:
                            model.fit(X_train, y)
                            trained_models[name] = model
                        test_preds.append(model.predict_proba(X_test)[:, 1])
            
            if test_preds:
                test_meta_X = np.column_stack(test_preds[:len(stacking_preds)])
                stacking_pred = meta_learner.predict_proba(test_meta_X)[:, 1]
                best_model = max(model_scores, key=model_scores.get)
                best_idx = list(model_scores.keys()).index(best_model)
                best_pred = predictions[best_idx]
                blending_weight = 0.6
                final_pred = blending_weight * stacking_pred + (1 - blending_weight) * best_pred
                pd.DataFrame({'row_id': test['row_id'], 'target': final_pred}).to_csv('final_submission.csv', index=False)
                print(f"Saved final_submission.csv (stacking blend)")
            else:
                weights = [1.0/np.var(p) for p in predictions]
                weights = [w/sum(weights) for w in weights]
                ensemble = np.average(predictions, axis=0, weights=weights)
                pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
                print(f"Saved final_submission.csv (variance-weighted)")
        else:
            weights = [1.0/np.var(p) for p in predictions]
            weights = [w/sum(weights) for w in weights]
            ensemble = np.average(predictions, axis=0, weights=weights)
            pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
            print(f"Saved final_submission.csv (variance-weighted)")
    elif len(predictions) > 1:
        weights = [1.0/np.var(p) for p in predictions]
        weights = [w/sum(weights) for w in weights]
        ensemble = np.average(predictions, axis=0, weights=weights)
        pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv (variance-weighted)")
    else:
        pd.DataFrame({'row_id': test['row_id'], 'target': predictions[0]}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv (1 model)")


if __name__ == '__main__':
    main()