#!/usr/bin/env python3
"""
Entry point for Kaggle AAP Agent submission.
Runs the 4-model ensemble pipeline (CatBoost/ExtraTrees/XGBoost/LogisticRegression)
across all 16 data splits and writes a combined final_submission.csv.

Each split is processed in a subprocess to isolate memory and prevent
segfaults from memory accumulation across 16 splits.
"""

import sys
import os
import gc
import glob
import subprocess
import shutil as _shutil
import pandas as pd
import numpy as np
import warnings
import itertools
import re
import zipfile
import json
import tempfile
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')


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


def get_adaptive_params(n_samples):
    """Returns model parameters adapted to dataset size."""
    if n_samples < 500:
        return dict(n_seeds=1, cb_iterations=300, cb_depth=4, cb_lr=0.1,
                    xgb_n_est=100, xgb_depth=3, xgb_lr=0.1, et_n_est=100, et_depth=4)
    elif n_samples < 2000:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=5, cb_lr=0.05,
                    xgb_n_est=200, xgb_depth=4, xgb_lr=0.05, et_n_est=200, et_depth=6)
    elif n_samples < 10000:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=6, cb_lr=0.03,
                    xgb_n_est=300, xgb_depth=4, xgb_lr=0.03, et_n_est=300, et_depth=8)
    else:
        return dict(n_seeds=1, cb_iterations=300, cb_depth=6, cb_lr=0.1,
                    xgb_n_est=200, xgb_depth=5, xgb_lr=0.1, et_n_est=200, et_depth=6)


def find_data_dirs():
    """Find all split directories, checking Kaggle input path first."""
    search_paths = [
        '/kaggle/input/autonomous-agent-prediction-beta/train_*',
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
                if len(dirs) < 16 and os.path.exists('data/data'):
                    deeper = sorted(glob.glob('data/*/train_*'))
                    deeper = [d for d in deeper if os.path.isdir(d) and os.path.exists(os.path.join(d, 'train.csv'))]
                    if deeper:
                        return deeper
                return dirs
    return []


def download_data():
    """Download competition data using Kaggle API."""
    print("Attempting to download competition data...")
    try:
        import kaggle
        print("Downloading...")
        kaggle.api.competition_download_files('autonomous-agent-prediction-beta', path='.', force=True, quiet=False)
        zip_files = glob.glob('*.zip')
        for zf in zip_files:
            print(f"Unzipping {zf}...")
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zf)
        nested = glob.glob('data/autonomous-agent-prediction-beta/train_*')
        if nested:
            for d in sorted(nested):
                dst_name = os.path.basename(d)
                d2 = os.path.join('data', dst_name)
                if os.path.isdir(d):
                    _shutil.move(d, d2)
            nested_parent = 'data/autonomous-agent-prediction-beta'
            if os.path.isdir(nested_parent) and not os.listdir(nested_parent):
                _shutil.rmtree(nested_parent)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


# --- Subprocess runner code with CatBoost ---

RUNNER_CODE = r'''import sys, os, json, warnings, gc
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import itertools, re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.optimize import nnls
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class SeedAveragedCatBoost:
    def __init__(self, cat_features=None, seeds=(42,), iterations=2000, **params):
        self.cat_features = cat_features
        self.seeds = seeds
        self.params = params
        self.iterations = iterations
        self.models = []
    def fit(self, X, y):
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
    def __init__(self, n_estimators=2000, **params):
        self.params = params
        self.n_estimators = n_estimators
        self.model = None
    def fit(self, X, y):
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        self.model = XGBClassifier(n_estimators=self.n_estimators, early_stopping_rounds=50, **self.params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def get_adaptive_params(n):
    if n < 500:
        return dict(n_seeds=1, cb_iterations=300, cb_depth=4, cb_lr=0.1, xgb_n_est=100, xgb_depth=3, xgb_lr=0.1, et_n_est=100, et_depth=4)
    elif n < 2000:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=5, cb_lr=0.05, xgb_n_est=200, xgb_depth=4, xgb_lr=0.05, et_n_est=200, et_depth=6)
    elif n < 10000:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=6, cb_lr=0.03, xgb_n_est=300, xgb_depth=4, xgb_lr=0.03, et_n_est=300, et_depth=8)
    else:
        return dict(n_seeds=1, cb_iterations=300, cb_depth=6, cb_lr=0.1, xgb_n_est=200, xgb_depth=5, xgb_lr=0.1, et_n_est=200, et_depth=6)

split_dir = sys.argv[1]
experiment = sys.argv[2]

train = pd.read_csv(os.path.join(split_dir, 'train.csv'))
test = pd.read_csv(os.path.join(split_dir, 'test.csv'))

raw_cat_cols = [c for c in train.columns
                if (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
                and c.startswith('feature_')]
ORDINAL_RE = re.compile(r'^ord_(\d+)$')
ordinal_cols = [c for c in raw_cat_cols if train[c].dropna().astype(str).str.match(ORDINAL_RE).all()]
cat_cols = [c for c in raw_cat_cols if c not in ordinal_cols]
num_cols = [c for c in train.columns if c not in cat_cols and c not in ['row_id', 'target']]

y = train['target'].values
X_train = train.drop(columns=['row_id', 'target']).copy()
X_test = test.drop(columns=['row_id']).copy()
test_row_ids = test['row_id'].copy()
del train, test

for col in ordinal_cols:
    X_train[col] = X_train[col].str.extract(r'(\d+)$', expand=False).astype(float)
    X_test[col] = X_test[col].str.extract(r'(\d+)$', expand=False).astype(float)

for col in X_train.columns:
    if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]):
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')
    else:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

for col in num_cols:
    if col in X_train.columns:
        lower, upper = X_train[col].quantile([0.01, 0.99])
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

cb_X_train = X_train.copy()
cb_X_test = X_test.copy()

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

if num_cols:
    num_mi = mutual_info_classif(X_train[num_cols], y, random_state=42)
    top_num_cols = pd.Series(num_mi, index=num_cols).nlargest(min(5, len(num_cols))).index.tolist()
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        if col1 in X_train.columns and col2 in X_train.columns:
            X_train[col1 + "_" + col2 + "_mul"] = (X_train[col1] * X_train[col2]).astype(float)
            X_test[col1 + "_" + col2 + "_mul"] = (X_test[col1] * X_test[col2]).astype(float)

n = len(y)
adaptive = get_adaptive_params(n)
n_folds = 5 if n < 5000 else 3
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cat_feature_indices = [cb_X_train.columns.get_loc(c) for c in cat_cols] if cat_cols else []

def get_model(name):
    if name == 'et':
        return ExtraTreesClassifier(n_estimators=int(adaptive['et_n_est']), max_depth=int(adaptive['et_depth']),
                                    random_state=42, n_jobs=2)
    if name == 'xgb':
        return EarlyStoppingXGB(n_estimators=int(adaptive['xgb_n_est']), max_depth=int(adaptive['xgb_depth']),
                                learning_rate=adaptive['xgb_lr'], random_state=42, n_jobs=2, verbosity=0)
    if name == 'lr':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=0.1, random_state=42))
    if name == 'cb':
        cb_seeds = (42, 142, 242, 342, 442)[:adaptive['n_seeds']]
        return SeedAveragedCatBoost(cat_features=cat_feature_indices, seeds=cb_seeds,
                                   iterations=int(adaptive['cb_iterations']), depth=int(adaptive['cb_depth']),
                                   learning_rate=adaptive['cb_lr'],
                                   thread_count=2, allow_writing_files=False, save_snapshot=False)

if experiment == 'no_cb':
    models_to_train = ['et', 'xgb', 'lr']
elif experiment == 'ensemble':
    models_to_train = ['et', 'xgb', 'lr', 'cb']
else:
    models_to_train = [experiment]

oof_cache = {}
predictions = []
model_ids = []

for name in models_to_train:
    try:
        X_df = cb_X_train if name == 'cb' else X_train
        model_oof = np.zeros(n)
        scores = []
        for tr_idx, val_idx in cv.split(X_df, y):
            model = get_model(name)
            model.fit(X_df.iloc[tr_idx], y[tr_idx])
            pred = model.predict_proba(X_df.iloc[val_idx])[:, 1]
            model_oof[val_idx] = pred
            scores.append(roc_auc_score(y[val_idx], pred))
            del model
            gc.collect()
        oof_cache[name] = model_oof
        cv_score = np.mean(scores)
        print("  " + name + ": CV=" + str(round(cv_score, 4)))

        model = get_model(name)
        model.fit(X_df, y)
        Xt_df = cb_X_test if name == 'cb' else X_test
        pred = model.predict_proba(Xt_df)[:, 1]
        predictions.append(pred)
        model_ids.append(name)
        del model
        gc.collect()
    except Exception as e:
        print("  " + name + ": Failed - " + str(e))

if len(predictions) >= 2:
    names = model_ids
    oof_matrix = np.column_stack([oof_cache[n] for n in names])
    weights, _ = nnls(oof_matrix, y.astype(float))
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    oof_aucs = [roc_auc_score(y, oof_matrix[:, i]) for i in range(len(names))]
    sorted_idx = np.argsort(-np.array(oof_aucs))
    if len(sorted_idx) >= 2 and (oof_aucs[sorted_idx[0]] - oof_aucs[sorted_idx[1]]) < 0.005:
        for idx in sorted_idx[:2]:
            if weights[idx] < 0.10:
                weights[idx] = 0.10
        weights = weights / weights.sum()
    oof_blend_auc = roc_auc_score(y, oof_matrix @ weights)
    print("  OOF-weighted blend: " + str(dict(zip(names, weights.round(3)))) + ", OOF AUC=" + str(round(oof_blend_auc, 4)))
    final_pred = np.column_stack(predictions) @ weights
else:
    final_pred = predictions[0]
    print("  Single model: " + model_ids[0])

result = {
    'row_ids': test_row_ids.tolist(),
    'preds': final_pred.tolist(),
}
print('RESULT_JSON_START' + json.dumps(result) + 'RESULT_JSON_END')
sys.stdout.flush()
'''


# --- No-CatBoost runner code (fallback when CatBoost segfaults) ---

NO_CB_RUNNER_CODE = r'''import sys, os, json, warnings, gc
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import itertools, re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.optimize import nnls
from xgboost import XGBClassifier

class EarlyStoppingXGB:
    def __init__(self, n_estimators=2000, **params):
        self.params = params
        self.n_estimators = n_estimators
        self.model = None
    def fit(self, X, y):
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        self.model = XGBClassifier(n_estimators=self.n_estimators, early_stopping_rounds=50, **self.params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def get_adaptive_params(n):
    if n < 500:
        return dict(n_seeds=1, cb_iterations=300, cb_depth=4, cb_lr=0.1, xgb_n_est=100, xgb_depth=3, xgb_lr=0.1, et_n_est=100, et_depth=4)
    elif n < 2000:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=5, cb_lr=0.05, xgb_n_est=200, xgb_depth=4, xgb_lr=0.05, et_n_est=200, et_depth=6)
    elif n < 10000:
        return dict(n_seeds=1, cb_iterations=500, cb_depth=6, cb_lr=0.03, xgb_n_est=300, xgb_depth=4, xgb_lr=0.03, et_n_est=300, et_depth=8)
    else:
        return dict(n_seeds=1, cb_iterations=300, cb_depth=6, cb_lr=0.1, xgb_n_est=200, xgb_depth=5, xgb_lr=0.1, et_n_est=200, et_depth=6)

split_dir = sys.argv[1]
experiment = sys.argv[2]

train = pd.read_csv(os.path.join(split_dir, 'train.csv'))
test = pd.read_csv(os.path.join(split_dir, 'test.csv'))

raw_cat_cols = [c for c in train.columns
                if (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
                and c.startswith('feature_')]
ORDINAL_RE = re.compile(r'^ord_(\d+)$')
ordinal_cols = [c for c in raw_cat_cols if train[c].dropna().astype(str).str.match(ORDINAL_RE).all()]
cat_cols = [c for c in raw_cat_cols if c not in ordinal_cols]
num_cols = [c for c in train.columns if c not in cat_cols and c not in ['row_id', 'target']]

y = train['target'].values
X_train = train.drop(columns=['row_id', 'target']).copy()
X_test = test.drop(columns=['row_id']).copy()
test_row_ids = test['row_id'].copy()
del train, test

for col in ordinal_cols:
    X_train[col] = X_train[col].str.extract(r'(\d+)$', expand=False).astype(float)
    X_test[col] = X_test[col].str.extract(r'(\d+)$', expand=False).astype(float)

for col in X_train.columns:
    if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]):
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')
    else:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

for col in num_cols:
    if col in X_train.columns:
        lower, upper = X_train[col].quantile([0.01, 0.99])
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

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

if num_cols:
    num_mi = mutual_info_classif(X_train[num_cols], y, random_state=42)
    top_num_cols = pd.Series(num_mi, index=num_cols).nlargest(min(5, len(num_cols))).index.tolist()
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        if col1 in X_train.columns and col2 in X_train.columns:
            X_train[col1 + "_" + col2 + "_mul"] = (X_train[col1] * X_train[col2]).astype(float)
            X_test[col1 + "_" + col2 + "_mul"] = (X_test[col1] * X_test[col2]).astype(float)

n = len(y)
adaptive = get_adaptive_params(n)
n_folds = 5 if n < 5000 else 3
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

def get_model(name):
    if name == 'et':
        return ExtraTreesClassifier(n_estimators=int(adaptive['et_n_est']), max_depth=int(adaptive['et_depth']),
                                    random_state=42, n_jobs=2)
    if name == 'xgb':
        return EarlyStoppingXGB(n_estimators=int(adaptive['xgb_n_est']), max_depth=int(adaptive['xgb_depth']),
                                learning_rate=adaptive['xgb_lr'], random_state=42, n_jobs=2, verbosity=0)
    if name == 'lr':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=0.1, random_state=42))

models_to_train = ['et', 'xgb', 'lr']
oof_cache = {}
predictions = []
model_ids = []

for name in models_to_train:
    try:
        model_oof = np.zeros(n)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y):
            model = get_model(name)
            model.fit(X_train.iloc[tr_idx], y[tr_idx])
            pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            model_oof[val_idx] = pred
            scores.append(roc_auc_score(y[val_idx], pred))
            del model
            gc.collect()
        oof_cache[name] = model_oof
        cv_score = np.mean(scores)
        print("  " + name + ": CV=" + str(round(cv_score, 4)))

        model = get_model(name)
        model.fit(X_train, y)
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
        model_ids.append(name)
        del model
        gc.collect()
    except Exception as e:
        print("  " + name + ": Failed - " + str(e))

if len(predictions) >= 2:
    names = model_ids
    oof_matrix = np.column_stack([oof_cache[n] for n in names])
    weights, _ = nnls(oof_matrix, y.astype(float))
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    oof_aucs = [roc_auc_score(y, oof_matrix[:, i]) for i in range(len(names))]
    sorted_idx = np.argsort(-np.array(oof_aucs))
    if len(sorted_idx) >= 2 and (oof_aucs[sorted_idx[0]] - oof_aucs[sorted_idx[1]]) < 0.005:
        for idx in sorted_idx[:2]:
            if weights[idx] < 0.10:
                weights[idx] = 0.10
        weights = weights / weights.sum()
    oof_blend_auc = roc_auc_score(y, oof_matrix @ weights)
    print("  OOF-weighted blend: " + str(dict(zip(names, weights.round(3)))) + ", OOF AUC=" + str(round(oof_blend_auc, 4)))
    final_pred = np.column_stack(predictions) @ weights
else:
    final_pred = predictions[0]
    print("  Single model: " + model_ids[0])

result = {
    'row_ids': test_row_ids.tolist(),
    'preds': final_pred.tolist(),
}
print('RESULT_JSON_START' + json.dumps(result) + 'RESULT_JSON_END')
sys.stdout.flush()
'''


def process_split_subprocess(split_dir, experiment):
    """Process one split in a subprocess and return (row_ids, predictions).

    Writes a temporary Python script and runs it as a subprocess. Each
    subprocess has its own memory space, so all model memory is released
    when the subprocess exits — preventing segfault from accumulation
    across 16 splits.

    If the subprocess segfaults (exit -11), retries with NO_CB_RUNNER_CODE
    (ET+XGB+LR only, no CatBoost import at all).
    """
    runner_path = os.path.join(tempfile.gettempdir(), f'split_runner_{os.getpid()}_{id(split_dir)}.py')
    with open(runner_path, 'w') as f:
        f.write(RUNNER_CODE)

    try:
        result = subprocess.run(
            [sys.executable, runner_path, split_dir, experiment],
            capture_output=True, text=True, timeout=600,
            cwd=os.getcwd()
        )
    except subprocess.TimeoutExpired:
        print(f"  Subprocess timed out for {split_dir}")
        if os.path.exists(runner_path):
            os.unlink(runner_path)
        return None, None
    finally:
        if os.path.exists(runner_path):
            os.unlink(runner_path)

    if result.returncode != 0:
        if result.returncode == -11:  # SIGSEGV
            print(f"  Subprocess segfaulted, retrying with no_cb...")
            runner_path2 = os.path.join(tempfile.gettempdir(), f'split_runner_{os.getpid()}_{id(split_dir)}_nobr.py')
            with open(runner_path2, 'w') as f:
                f.write(NO_CB_RUNNER_CODE)
            try:
                result2 = subprocess.run(
                    [sys.executable, runner_path2, split_dir, 'no_cb'],
                    capture_output=True, text=True, timeout=600, cwd=os.getcwd()
                )
            finally:
                if os.path.exists(runner_path2):
                    os.unlink(runner_path2)
            if result2.returncode != 0:
                print(f"  no_cb fallback also failed (exit {result2.returncode})")
                return None, None
            stdout = result2.stdout
        else:
            stderr_msg = result.stderr[:500] if result.stderr else "(no stderr)"
            print(f"  Subprocess failed (exit {result.returncode}): {stderr_msg}")
            return None, None
    else:
        stdout = result.stdout

    start = stdout.find('RESULT_JSON_START')
    end = stdout.find('RESULT_JSON_END')
    if start == -1 or end == -1:
        print(f"  Subprocess failed: no result marker in stdout")
        print(f"  stdout: {stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        return None, None

    result_json = stdout[start + len('RESULT_JSON_START'):end]
    data = json.loads(result_json)
    return data['row_ids'], data['preds']


def train_and_predict(train_df, test_df, experiment='ensemble'):
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

    del train_df, test_df

    for col in ordinal_cols:
        X_train[col] = X_train[col].str.extract(r'(\d+)$', expand=False).astype(float)
        X_test[col] = X_test[col].str.extract(r'(\d+)$', expand=False).astype(float)

    for col in X_train.columns:
        if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]):
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')
        else:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

    for col in num_cols:
        if col in X_train.columns:
            lower, upper = X_train[col].quantile([0.01, 0.99])
            X_train[col] = X_train[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

    cb_X_train = X_train.copy()
    cb_X_test = X_test.copy()

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

    if num_cols:
        num_mi = mutual_info_classif(X_train[num_cols], y, random_state=42)
        top_num_cols = pd.Series(num_mi, index=num_cols).nlargest(min(5, len(num_cols))).index.tolist()
        for col1, col2 in itertools.combinations(top_num_cols, 2):
            if col1 in X_train.columns and col2 in X_train.columns:
                X_train[f"{col1}_{col2}_mul"] = (X_train[col1] * X_train[col2]).astype(float)
                X_test[f"{col1}_{col2}_mul"] = (X_test[col1] * X_test[col2]).astype(float)

    n_samples = len(y)
    adaptive = get_adaptive_params(n_samples)
    n_folds = 5 if n_samples < 5000 else 3
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cat_feature_indices = [cb_X_train.columns.get_loc(c) for c in cat_cols] if cat_cols else []

    def get_model(name):
        if name == 'et':
            return ExtraTreesClassifier(n_estimators=int(adaptive['et_n_est']), max_depth=int(adaptive['et_depth']),
                                        random_state=42, n_jobs=2)
        if name == 'xgb':
            return EarlyStoppingXGB(n_estimators=int(adaptive['xgb_n_est']), max_depth=int(adaptive['xgb_depth']),
                                    learning_rate=adaptive['xgb_lr'], random_state=42, n_jobs=2, verbosity=0)
        if name == 'lr':
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=5000, C=0.1, random_state=42))
        if name == 'cb':
            cb_seeds = (42, 142, 242, 342, 442)[:adaptive['n_seeds']]
            return SeedAveragedCatBoost(cat_features=cat_feature_indices, seeds=cb_seeds,
                                       iterations=int(adaptive['cb_iterations']), depth=int(adaptive['cb_depth']),
                                       learning_rate=adaptive['cb_lr'],
                                       thread_count=2, allow_writing_files=False, save_snapshot=False)

    def get_cv_and_oof(name, X_df):
        model_oof = np.zeros(n_samples)
        scores = []
        for tr_idx, val_idx in cv.split(X_df, y):
            model = get_model(name)
            model.fit(X_df.iloc[tr_idx], y[tr_idx])
            pred = model.predict_proba(X_df.iloc[val_idx])[:, 1]
            model_oof[val_idx] = pred
            scores.append(roc_auc_score(y[val_idx], pred))
            del model
            gc.collect()
        return np.mean(scores), model_oof

    models_to_train = []
    if experiment == 'ensemble':
        models_to_train = ['et', 'xgb', 'lr', 'cb']
    elif experiment == 'no_cb':
        models_to_train = ['et', 'xgb', 'lr']
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

    for name in models_to_train:
        try:
            X_df = cb_X_train if name == 'cb' else X_train
            cv_score, oof = get_cv_and_oof(name, X_df)
            oof_cache[name] = oof
            print(f"  {name}: CV={cv_score:.4f}")

            model = get_model(name)
            model.fit(X_df, y)
            Xt_df = cb_X_test if name == 'cb' else X_test
            pred = model.predict_proba(Xt_df)[:, 1]
            predictions.append(pred)
            model_ids.append(name)
            del model
            gc.collect()
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            continue

    if len(predictions) >= 2:
        names = model_ids
        oof_matrix = np.column_stack([oof_cache[n] for n in names])
        from scipy.optimize import nnls
        weights, _ = nnls(oof_matrix, y.astype(float))
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        oof_aucs = list(roc_auc_score(y, oof_matrix[:, i]) for i in range(len(names)))
        sorted_idx = np.argsort(-np.array(oof_aucs))
        if len(sorted_idx) >= 2 and (oof_aucs[sorted_idx[0]] - oof_aucs[sorted_idx[1]]) < 0.005:
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

        row_ids, preds = process_split_subprocess(split_dir, experiment)
        if row_ids is None:
            print(f"  FAILED for {split_dir}")
            continue

        all_row_ids.extend(row_ids)
        all_preds.extend(preds)
        del row_ids, preds
        gc.collect()

    print(f"\nCombined submission: {len(all_row_ids)} rows")
    submission = pd.DataFrame({'row_id': all_row_ids, 'target': all_preds})
    submission.to_csv('final_submission.csv', index=False)
    print(f"Saved final_submission.csv ({len(submission)} rows)")


if __name__ == '__main__':
    main()