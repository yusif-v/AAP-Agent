#!/usr/bin/env python3
"""
Entry point for Kaggle AAP Agent submission.
Runs the 4-model ensemble pipeline (CatBoost/ExtraTrees/XGBoost/LogisticRegression)
across all 16 data splits and writes a combined final_submission.csv.
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import warnings
import itertools
import re
import zipfile
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')

# --- Model classes (shared) ---

class SeedAveragedCatBoost:
    """Trains one CatBoostClassifier per seed and averages predict_proba."""

    def __init__(self, cat_features=None, seeds=(42,), iterations=2000, **params):
        self.cat_features = cat_features
        self.seeds = seeds
        self.params = params
        self.iterations = iterations
        self.models = []

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.models = []
        for seed in self.seeds:
            X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)
            model = CatBoostClassifier(cat_features=self.cat_features or None,
                                        random_state=seed, verbose=False,
                                        early_stopping_rounds=50, iterations=self.iterations, **self.params)
            model.fit(X_tr, y_tr, eval_set=(X_es, y_es))
            self.models.append(model)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class EarlyStoppingXGB:
    """XGBoost with early stopping on internal validation split."""

    def __init__(self, n_estimators=2000, **params):
        self.params = params
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X, y):
        from xgboost import XGBClassifier
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        self.model = XGBClassifier(n_estimators=self.n_estimators, early_stopping_rounds=50, **self.params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# --- Adaptive hyperparameters ---

def get_adaptive_params(n_samples):
    """Returns model parameters adapted to dataset size."""
    if n_samples < 500:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=4, cb_lr=0.05,
                    xgb_n_est=200, xgb_depth=3, xgb_lr=0.03, et_n_est=100, et_depth=4)
    elif n_samples < 2000:
        return dict(n_seeds=3, cb_iterations=1000, cb_depth=5, cb_lr=0.03,
                    xgb_n_est=300, xgb_depth=4, xgb_lr=0.03, et_n_est=200, et_depth=6)
    elif n_samples < 10000:
        return dict(n_seeds=3, cb_iterations=1500, cb_depth=6, cb_lr=0.03,
                    xgb_n_est=500, xgb_depth=4, xgb_lr=0.03, et_n_est=300, et_depth=8)
    else:
        return dict(n_seeds=3, cb_iterations=2000, cb_depth=6, cb_lr=0.03,
                    xgb_n_est=2000, xgb_depth=4, xgb_lr=0.03, et_n_est=300, et_depth=10)


# --- Data loading ---

def find_data_dirs():
    """Find all split directories, checking Kaggle input path first."""
    search_paths = [
        '/kaggle/input/autonomous-agent-prediction-beta/train_*',
        '/kaggle/input/autonomous-agent-prediction-beta/data/train_*',
        '/kaggle/input/autonomous-agent-prediction-beta/*/train_*',
        'data/train_*',
        'data/*/train_*',
        'train_*',
    ]
    for pattern in search_paths:
        dirs = sorted(glob.glob(pattern))
        if dirs:
            dirs = [d for d in dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'train.csv'))]
            if dirs:
                return dirs
    return []


def download_data():
    """Download competition data using Kaggle API."""
    print("Attempting to download competition data...")
    try:
        import kaggle
        os.makedirs('data', exist_ok=True)
        print("Downloading...")
        kaggle.api.competition_download_files('autonomous-agent-prediction-beta', path='.', force=True, quiet=False)
        zip_files = glob.glob('*.zip')
        for zf in zip_files:
            print(f"Unzipping {zf}...")
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall('data')
            os.remove(zf)
        # If the zip extracted into a nested directory, flatten it
        nested = glob.glob('data/*/train_01')
        if not nested:
            nested_dirs = glob.glob('data/autonomous-agent-prediction-beta')
            if nested_dirs:
                # Move all train_* dirs up one level
                for item in os.listdir(nested_dirs[0]):
                    s = os.path.join(nested_dirs[0], item)
                    d = os.path.join('data', item)
                    if os.path.isdir(s) and item.startswith('train_'):
                        import shutil
                        shutil.move(s, d)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


# --- Core training pipeline (operates on a single split) ---

def train_and_predict(train_df, test_df, experiment='ensemble', cat_features=None):
    """Train the ensemble on one split's data and return test predictions."""
    raw_cat_cols = [c for c in train_df.columns
                    if (pd.api.types.is_object_dtype(train_df[c]) or pd.api.types.is_string_dtype(train_df[c]))
                    and c.startswith('feature_')]

    ORDINAL_RE = re.compile(r'^ord_(\d+)$')
    ordinal_cols = [c for c in raw_cat_cols
                    if train_df[c].dropna().astype(str).str.match(ORDINAL_RE).all()]
    cat_cols = [c for c in raw_cat_cols if c not in ordinal_cols]
    num_cols = [c for c in train_df.columns if c not in cat_cols and c not in ['row_id', 'target']]

    y = train_df['target'].values
    X_train = train_df.drop(columns=['row_id', 'target']).copy()
    X_test = test_df.drop(columns=['row_id']).copy()
    test_row_ids = test_df['row_id'].copy()

    # Decode ordinal columns
    for col in ordinal_cols:
        X_train[col] = X_train[col].str.extract(r'(\d+)$', expand=False).astype(float)
        X_test[col] = X_test[col].str.extract(r'(\d+)$', expand=False).astype(float)

    # Impute
    for col in X_train.columns:
        if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]):
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')
        else:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

    # Outlier clipping
    for col in num_cols:
        if col in X_train.columns:
            lower, upper = X_train[col].quantile([0.01, 0.99])
            X_train[col] = X_train[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

    # CatBoost gets raw categorical strings; others get target-encoded numerics
    cb_X_train = X_train.copy()
    cb_X_test = X_test.copy()

    # OOF target encoding for unordered categoricals
    te_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y.mean()
    for col in cat_cols:
        oof_te = np.zeros(len(X_train))
        for tr_idx, val_idx in te_cv.split(X_train, y):
            fold_means = pd.Series(y[tr_idx]).groupby(X_train[col].iloc[tr_idx].values).mean()
            oof_te[val_idx] = X_train[col].iloc[val_idx].map(fold_means).fillna(global_mean).values
        X_train[col + '_te'] = oof_te
        full_means = pd.Series(y).groupby(X_train[col].values).mean()
        X_test[col + '_te'] = X_test[col].map(full_means).fillna(global_mean)

    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)

    # Feature interactions (top 5 numeric by MI)
    if num_cols:
        num_mi = mutual_info_classif(X_train[num_cols], y, random_state=42)
        top_num_cols = pd.Series(num_mi, index=num_cols).nlargest(min(5, len(num_cols))).index.tolist()
        for col1, col2 in itertools.combinations(top_num_cols, 2):
            if col1 in X_train.columns and col2 in X_train.columns:
                X_train[f"{col1}_{col2}_mul"] = (X_train[col1] * X_train[col2]).astype(float)
                X_test[f"{col1}_{col2}_mul"] = (X_test[col1] * X_test[col2]).astype(float)

    n_samples = len(train_df)
    adaptive = get_adaptive_params(n_samples)
    n_folds = min(5, 3) if n_samples < 2000 else 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cat_feature_indices = [cb_X_train.columns.get_loc(c) for c in cat_cols] if cat_cols else []

    def get_model(name):
        if name == 'et':
            return ExtraTreesClassifier(n_estimators=adaptive['et_n_est'], max_depth=adaptive['et_depth'],
                                        random_state=42, n_jobs=-1)
        if name == 'xgb':
            return EarlyStoppingXGB(n_estimators=adaptive['xgb_n_est'], max_depth=adaptive['xgb_depth'],
                                    learning_rate=adaptive['xgb_lr'], random_state=42, n_jobs=-1, verbosity=0)
        if name == 'lr':
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=5000, C=0.1, random_state=42))
        if name == 'cb':
            cb_seeds = (42, 142, 242, 342, 442)[:adaptive['n_seeds']]
            return SeedAveragedCatBoost(cat_features=cat_feature_indices, seeds=cb_seeds,
                                       iterations=adaptive['cb_iterations'], depth=adaptive['cb_depth'],
                                       learning_rate=adaptive['cb_lr'])

    def get_cv_and_oof(name, X_df):
        model_oof = np.zeros(n_samples)
        scores = []
        for tr_idx, val_idx in cv.split(X_df, y):
            model = get_model(name)
            model.fit(X_df.iloc[tr_idx], y[tr_idx])
            pred = model.predict_proba(X_df.iloc[val_idx])[:, 1]
            model_oof[val_idx] = pred
            scores.append(roc_auc_score(y[val_idx], pred))
        return np.mean(scores), model_oof

    def X_for(name):
        return cb_X_train if name == 'cb' else X_train

    def Xt_for(name):
        return cb_X_test if name == 'cb' else X_test

    models_to_train = []
    if experiment == 'ensemble':
        models_to_train = ['et', 'xgb', 'lr', 'cb']
    elif experiment == 'cb':
        models_to_train = ['cb']
    elif experiment == 'xgb':
        models_to_train = ['xgb']
    elif experiment == 'et':
        models_to_train = ['et']
    elif experiment == 'lr':
        models_to_train = ['lr']

    oof_cache = {}
    predictions = []
    model_ids = []
    model_scores = {}

    for name in models_to_train:
        try:
            cv_score, oof = get_cv_and_oof(name, X_for(name))
            oof_cache[name] = oof
            model_scores[name] = cv_score
            print(f"  {name}: CV={cv_score:.4f}")

            model = get_model(name)
            model.fit(X_for(name), y)
            pred = model.predict_proba(Xt_for(name))[:, 1]
            predictions.append(pred)
            model_ids.append(name)
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            continue

    if len(predictions) >= 2:
        names = [n for n in model_ids]
        oof_matrix = np.column_stack([oof_cache[n] for n in names])
        from scipy.optimize import nnls
        weights, _ = nnls(oof_matrix, y.astype(float))
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Conditional soft floor: if top-2 OOF models within 0.01 AUC, hedge
        oof_aucs = list(roc_auc_score(y, oof_matrix[:, i]) for i in range(len(names)))
        sorted_idx = np.argsort(-np.array(oof_aucs))
        if len(sorted_idx) >= 2 and (oof_aucs[sorted_idx[0]] - oof_aucs[sorted_idx[1]]) < 0.01:
            for idx in sorted_idx[:2]:
                if weights[idx] < 0.10:
                    weights[idx] = 0.10
            weights = weights / weights.sum()

        oof_blend_auc = roc_auc_score(y, oof_matrix @ weights)
        print(f"  OOF-weighted blend: {dict(zip(names, weights.round(3)))}, OOF AUC={oof_blend_auc:.4f}")

        final_pred = np.column_stack(predictions) @ weights
        print(f"  Saved predictions ({len(final_pred)} rows)")
        return test_row_ids, final_pred
    else:
        print(f"  Single model: {model_ids[0]}")
        return test_row_ids, predictions[0]


def main():
    """Main entry point — process all splits and write combined submission."""
    experiment = 'ensemble'
    if '--experiment' in sys.argv:
        idx = sys.argv.index('--experiment')
        if idx + 1 < len(sys.argv):
            experiment = sys.argv[idx + 1]

    data_dirs = find_data_dirs()
    if not data_dirs:
        print("Data not found, trying to download...")
        if not download_data():
            print("ERROR: Could not find or download data")
            sys.exit(1)
        data_dirs = find_data_dirs()

    print(f"Found {len(data_dirs)} split directories")

    all_row_ids = []
    all_preds = []

    for i, split_dir in enumerate(data_dirs):
        print(f"\n[{i+1}/{len(data_dirs)}] Processing {split_dir}...")
        train = pd.read_csv(os.path.join(split_dir, 'train.csv'))
        test = pd.read_csv(os.path.join(split_dir, 'test.csv'))
        print(f"  Loaded train={train.shape}, test={test.shape}")

        row_ids, preds = train_and_predict(train, test, experiment=experiment)
        all_row_ids.extend(row_ids.tolist())
        all_preds.extend(preds.tolist())

    print(f"\nCombined submission: {len(all_row_ids)} rows")
    submission = pd.DataFrame({'row_id': all_row_ids, 'target': all_preds})
    submission.to_csv('final_submission.csv', index=False)
    print(f"Saved final_submission.csv ({len(submission)} rows)")


if __name__ == '__main__':
    main()
