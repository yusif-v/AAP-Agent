#!/usr/bin/env python3
"""
AutoML training for AAP competition sandbox.
Ensemble of RF/ExtraTrees/XGBoost/LightGBM/CatBoost.
"""

import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import warnings; warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c.startswith('feature_')]
y = train['target'].values
X_train = train.drop(columns=['row_id', 'target']); X_test = test.drop(columns=['row_id'])

# Impute missing
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')
    else:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())

# Encode categoricals
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

n_samples = len(train)
n_folds = 3 if n_samples < 2000 else 5
predictions = []

# RF
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=10, random_state=42, n_jobs=-1)
oof = np.zeros(len(X_train))
for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y):
    rf.fit(X_train.iloc[tr_idx], y[tr_idx]); oof[val_idx] = rf.predict_proba(X_train.iloc[val_idx])[:, 1]
print(f"rf: CV={roc_auc_score(y, oof):.4f}")
rf.fit(X_train, y); predictions.append(rf.predict_proba(X_test)[:, 1])

# ExtraTrees
et = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
oof = np.zeros(len(X_train))
for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y):
    et.fit(X_train.iloc[tr_idx], y[tr_idx]); oof[val_idx] = et.predict_proba(X_train.iloc[val_idx])[:, 1]
print(f"et: CV={roc_auc_score(y, oof):.4f}")
et.fit(X_train, y); predictions.append(et.predict_proba(X_test)[:, 1])

# XGBoost
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    oof = np.zeros(len(X_train))
    for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y):
        xgb.fit(X_train.iloc[tr_idx], y[tr_idx]); oof[val_idx] = xgb.predict_proba(X_train.iloc[val_idx])[:, 1]
    print(f"xgb: CV={roc_auc_score(y, oof):.4f}")
    xgb.fit(X_train, y); predictions.append(xgb.predict_proba(X_test)[:, 1])
except: pass

# LightGBM
try:
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
    oof = np.zeros(len(X_train))
    for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y):
        lgbm.fit(X_train.iloc[tr_idx], y[tr_idx]); oof[val_idx] = lgbm.predict_proba(X_train.iloc[val_idx])[:, 1]
    print(f"lgbm: CV={roc_auc_score(y, oof):.4f}")
    lgbm.fit(X_train, y); predictions.append(lgbm.predict_proba(X_test)[:, 1])
except: pass

# CatBoost
if cat_cols:
    try:
        from catboost import CatBoostClassifier
        cat_idx = [list(X_train.columns).index(c) for c in cat_cols]
        cb = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.1, cat_features=cat_idx, verbose=False, random_state=42)
        oof = np.zeros(len(X_train))
        for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y):
            cb.fit(X_train.iloc[tr_idx], y[tr_idx]); oof[val_idx] = cb.predict_proba(X_train.iloc[val_idx])[:, 1]
        print(f"cb: CV={roc_auc_score(y, oof):.4f}")
        cb.fit(X_train, y); predictions.append(cb.predict_proba(X_test)[:, 1])
    except: pass

# Ensemble
if predictions:
    ensemble = np.mean(predictions, axis=0)
    pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
    print(f"Saved final_submission.csv ({len(predictions)} models)")