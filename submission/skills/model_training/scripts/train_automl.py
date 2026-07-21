#!/usr/bin/env python3
"""
Enhanced AutoML with feature selection and stacking.
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

# Missing imputation
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')
    else:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())

# Encode
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

n_samples = len(train)
n_folds = 3 if n_samples < 2000 else 5

# Quick feature importance to select top features
rf_base = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y)
importances = pd.DataFrame({'feature': X_train.columns, 'importance': rf_base.feature_importances_})
top_features = importances.nlargest(12, 'importance')['feature'].tolist()  # Use top 12 features

print(f"Top features: {top_features}")

# Train on reduced feature set
X_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]

predictions = []

# RF on selected features
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=10, random_state=42, n_jobs=-1)
oof = np.zeros(len(X_reduced))
for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_reduced, y):
    rf.fit(X_reduced.iloc[tr_idx], y[tr_idx]); oof[val_idx] = rf.predict_proba(X_reduced.iloc[val_idx])[:, 1]
print(f"rf_selected: CV={roc_auc_score(y, oof):.4f}")
rf.fit(X_reduced, y); predictions.append(rf.predict_proba(X_test_reduced)[:, 1])

# Full feature RF
rf_full = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
oof = np.zeros(len(X_train))
for tr_idx, val_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y):
    rf_full.fit(X_train.iloc[tr_idx], y[tr_idx]); oof[val_idx] = rf_full.predict_proba(X_train.iloc[val_idx])[:, 1]
print(f"rf_full: CV={roc_auc_score(y, oof):.4f}")
rf_full.fit(X_train, y); predictions.append(rf_full.predict_proba(X_test)[:, 1])

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

# Weighted ensemble
if len(predictions) > 1:
    # Weight by inverse variance of predictions (diversity weighting)
    weights = [1.0/predictions[i].var() for i in range(len(predictions))]
    weights = [w/sum(weights) for w in weights]
    ensemble = np.average(predictions, axis=0, weights=weights)
else:
    ensemble = predictions[0]

pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
print(f"Saved final_submission.csv ({len(predictions)} models)")