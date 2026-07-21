#!/usr/bin/env python3
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

# RF
n_est = 100 if n_samples < 1000 else 200
max_d = 5 if n_samples < 1000 else 8
model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)

oof = np.zeros(len(X_train))
scores = []
for tr_idx, val_idx in cv.split(X_train, y):
    model.fit(X_train.iloc[tr_idx], y[tr_idx])
    oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    scores.append(roc_auc_score(y[val_idx], oof[val_idx]))

print(f"RF CV: {np.mean(scores):.4f}")
model.fit(X_train, y)
pd.DataFrame({'row_id': test['row_id'], 'target': model.predict_proba(X_test)[:, 1]}).to_csv('rf_submission.csv', index=False)