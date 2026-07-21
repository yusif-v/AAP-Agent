#!/usr/bin/env python3
"""
AutoML training script - executes in Kaggle sandbox.
Adapts model complexity based on dataset size and feature composition.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c.startswith('feature_')]
y = train['target'].values
X_train = train.drop(columns=['row_id', 'target'])
X_test = test.drop(columns=['row_id'])

for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

n_samples = len(train)
n_folds = 3 if n_samples < 2000 else 5

cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Model config based on size
models = {}
n_est = 100 if n_samples < 1000 else 200
max_d = 5 if n_samples < 1000 else 8
models['rf'] = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)

# XGBoost
try:
    from xgboost import XGBClassifier
    models['xgb'] = XGBClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
except: pass

# LightGBM
try:
    from lightgbm import LGBMClassifier
    models['lgbm'] = LGBMClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
except: pass

# CatBoost for categorical
if len(cat_cols) > 0:
    try:
        from catboost import CatBoostClassifier
        models['cb'] = CatBoostClassifier(iterations=100, depth=6, cat_features=[list(X_train.columns).index(c) for c in cat_cols], verbose=False, random_state=42)
    except: pass

# Train models
all_preds = {}
for name, model in models.items():
    oof = np.zeros(len(X_train))
    scores = []
    for tr_idx, val_idx in cv.split(X_train, y):
        model.fit(X_train.iloc[tr_idx], y[tr_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], oof[val_idx]))
    print(f"{name}: CV={np.mean(scores):.4f}")
    model.fit(X_train, y)
    all_preds[name] = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({'row_id': test['row_id'], 'target': all_preds[name]}).to_csv(f'{name}_submission.csv', index=False)

# Ensemble
if len(all_preds) > 1:
    ens = np.mean(list(all_preds.values()), axis=0)
    pd.DataFrame({'row_id': test['row_id'], 'target': ens}).to_csv('final_submission.csv', index=False)
    print("Saved final_submission.csv (ensemble)")