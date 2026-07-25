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
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

# Small, sklearn-only search spaces per model (no optuna - agent sandbox has no internet)
PARAM_DISTS = {
    'rf': {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8, None], 'min_samples_split': [2, 5, 10]},
    'et': {'n_estimators': [100, 200, 300], 'max_depth': [6, 8, 10, None], 'min_samples_split': [2, 5, 10]},
    'xgb': {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5, 6], 'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.85, 1.0], 'colsample_bytree': [0.7, 0.85, 1.0]},
    'lgbm': {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8, -1], 'learning_rate': [0.03, 0.05, 0.1],
             'num_leaves': [15, 31, 63], 'subsample': [0.7, 0.85, 1.0]},
    'gb': {'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.03, 0.05, 0.1]},
    'cb': {'iterations': [100, 200, 300], 'depth': [4, 5, 6, 7], 'learning_rate': [0.03, 0.05, 0.1]},
}


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
        data_dirs = sorted(glob.glob(os.path.join(input_dir, 'train_*')))
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
        # Sort to ensure deterministic order (train_01, train_02, ...)
        data_dirs.sort()
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
    tune = '--tune' in sys.argv

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

    cat_cols = [c for c in train.columns
                if (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
                and c.startswith('feature_')]
    num_cols = [c for c in train.columns if c not in cat_cols and c not in ['row_id', 'target']]
    y = train['target'].values
    X_train = train.drop(columns=['row_id', 'target'])
    X_test = test.drop(columns=['row_id'])

    # Impute
    for col in X_train.columns:
        if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]):
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')
        else:
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())

    # Outlier handling - clip numerical features to [1st, 99th] percentile
    for col in num_cols:
        if col in X_train.columns:
            lower, upper = X_train[col].quantile([0.01, 0.99])
            X_train[col] = X_train[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

    # CatBoost gets its own feature matrix with raw categorical strings intact -
    # target encoding below replaces cat_cols with numeric _te columns for the other models,
    # and CatBoost's whole value-add is native categorical handling, not target-encoded floats.
    cb_X_train = X_train.copy()
    cb_X_test = X_test.copy()

    # Target encoding for categorical columns
    for col in cat_cols:
        target_mean = train.groupby(col)['target'].mean()
        global_mean = train['target'].mean()
        X_train[col + '_te'] = X_train[col].map(target_mean).fillna(global_mean)
        X_test[col + '_te'] = X_test[col].map(target_mean).fillna(global_mean)

    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)

    # Feature interactions - pick the top 5 numerical features by mutual information with
    # the target, not just the first 5 columns in source-file order
    if num_cols:
        num_mi = mutual_info_classif(X_train[num_cols], y, random_state=42)
        top_num_cols = pd.Series(num_mi, index=num_cols).nlargest(min(5, len(num_cols))).index.tolist()
    else:
        top_num_cols = []
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        if col1 in X_train.columns and col2 in X_train.columns:
            new_col = f"{col1}_{col2}_mul"
            X_train[new_col] = (X_train[col1] * X_train[col2]).astype(float)
            X_test[new_col] = (X_test[col1] * X_test[col2]).astype(float)

    # Feature selection using mutual information (limit to reasonable number)
    all_features = X_train.columns.tolist()
    mi_scores = mutual_info_classif(X_train, y, random_state=42)
    mi_df = pd.DataFrame({'feature': all_features, 'mi_score': mi_scores})
    top_features = mi_df.nlargest(min(30, len(all_features)), 'mi_score')['feature'].tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    def get_model(name, n_samples, cat_features=None):
        n_est = 100 if n_samples < 5000 else 150
        if name == 'rf':
            return RandomForestClassifier(n_estimators=n_est, max_depth=6, min_samples_split=5,
                                           random_state=42, n_jobs=-1)
        if name == 'et':
            return ExtraTreesClassifier(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1)
        if name == 'xgb':
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.1,
                                   random_state=42, n_jobs=-1, verbosity=0)
        if name == 'lgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.1,
                                  random_state=42, n_jobs=-1, verbose=-1)
        if name == 'gb':
            return GradientBoostingClassifier(n_estimators=n_est, max_depth=4,
                                                learning_rate=0.1, random_state=42)
        if name == 'cb' and cat_features:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                                      cat_features=cat_features, verbose=False, random_state=42)
        return None

    def get_cv_score(name, X, y, cat_features=None, n_folds=5):
        """Builds a fresh model per fold via get_model() instead of sklearn's clone() -
        CatBoostClassifier with cat_features set in the constructor can't be cloned."""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            model = get_model(name, len(tr_idx), cat_features)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], pred))
        scores = np.array(scores)
        return scores.mean(), scores.std()

    def get_oof_predictions(name, X, y, cat_features=None, n_folds=5):
        """Out-of-fold predict_proba for stacking, without sklearn's clone()."""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(X))
        for tr_idx, val_idx in cv.split(X, y):
            model = get_model(name, len(tr_idx), cat_features)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        return oof

    n_samples = len(train)
    predictions = []
    model_ids = []
    model_scores = {}
    trained_models = {}

    models_to_train = []
    if experiment in ('rf', 'ensemble'):
        models_to_train.append(('rf', 'rf'))
    if experiment in ('et', 'ensemble'):
        models_to_train.append(('et', 'et'))
    if experiment in ('xgb', 'ensemble'):
        models_to_train.append(('xgb', 'xgb'))
    if experiment in ('lgbm', 'ensemble'):
        models_to_train.append(('lgbm', 'lgbm'))
    if experiment in ('gb', 'ensemble'):
        models_to_train.append(('gb', 'gb'))

    # Get categorical feature indices for CatBoost, relative to cb_X_train (which keeps
    # the raw categorical strings rather than the target-encoded numeric columns)
    cat_feature_indices = None
    if cat_cols and experiment in ('cb', 'ensemble'):
        cat_feature_indices = [cb_X_train.columns.get_loc(c) for c in cat_cols]
        models_to_train.append(('cb', 'cb'))

    def X_for(name):
        return cb_X_train if name == 'cb' else X_train

    def Xt_for(name):
        return cb_X_test if name == 'cb' else X_test

    for name, key in models_to_train:
        try:
            model = get_model(name, n_samples, cat_feature_indices)
            if model is None:
                continue

            cv_score, cv_std = get_cv_score(name, X_for(name), y, cat_feature_indices)
            model_scores[name] = cv_score
            print(f"{name}: CV={cv_score:.4f} (+/- {cv_std:.4f})")

            model.fit(X_for(name), y)
            trained_models[name] = model
            pred = model.predict_proba(Xt_for(name))[:, 1]
            predictions.append(pred)
            model_ids.append(name)
            pd.DataFrame({'row_id': test['row_id'], 'target': pred}).to_csv(f'{key}_pred.csv', index=False)
        except Exception as e:
            print(f"{name}: Failed - {e}")
            continue

    # Hyperparameter tuning for the best-performing model (sklearn RandomizedSearchCV only,
    # since the agent sandbox has no internet access and can't pip install optuna)
    if tune and model_scores:
        best_name = max(model_scores, key=model_scores.get)
        dist = PARAM_DISTS.get(best_name)
        if best_name == 'cb':
            # sklearn's RandomizedSearchCV can't clone a CatBoostClassifier with cat_features
            # set, so use CatBoost's own randomized_search to pick candidate params, then
            # re-score those params with our own manual per-fold CV so the comparison against
            # the untuned baseline is apples-to-apples.
            print(f"Tuning cb (baseline CV={model_scores['cb']:.4f}) via CatBoost's native search...")
            search_model = get_model('cb', n_samples, cat_feature_indices)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            search_result = search_model.randomized_search(
                dist, X=cb_X_train, y=y, cv=cv, n_iter=15,
                search_by_train_test_split=False, verbose=False, plot=False)
            best_params = search_result['params']

            def get_tuned_cb():
                params = dict(iterations=100, depth=5, learning_rate=0.1,
                              cat_features=cat_feature_indices, verbose=False, random_state=42)
                params.update(best_params)
                from catboost import CatBoostClassifier
                return CatBoostClassifier(**params)

            scores = []
            for tr_idx, val_idx in cv.split(cb_X_train, y):
                m = get_tuned_cb()
                m.fit(cb_X_train.iloc[tr_idx], y[tr_idx])
                pred = m.predict_proba(cb_X_train.iloc[val_idx])[:, 1]
                scores.append(roc_auc_score(y[val_idx], pred))
            tuned_score = np.mean(scores)
            print(f"Tuned cb: CV={tuned_score:.4f}, params={best_params}")

            if tuned_score > model_scores['cb'] and 'cb' in model_ids:
                final_model = get_tuned_cb()
                final_model.fit(cb_X_train, y)
                trained_models['cb'] = final_model
                predictions[model_ids.index('cb')] = final_model.predict_proba(cb_X_test)[:, 1]
                model_scores['cb'] = tuned_score
        elif dist:
            print(f"Tuning {best_name} (baseline CV={model_scores[best_name]:.4f})...")
            base_model = get_model(best_name, n_samples, cat_feature_indices)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            search = RandomizedSearchCV(base_model, dist, n_iter=15, cv=cv, scoring='roc_auc',
                                         random_state=42, n_jobs=-1)
            search.fit(X_train, y)
            print(f"Tuned {best_name}: CV={search.best_score_:.4f}, params={search.best_params_}")
            if search.best_score_ > model_scores[best_name] and best_name in model_ids:
                trained_models[best_name] = search.best_estimator_
                tuned_pred = search.best_estimator_.predict_proba(X_test)[:, 1]
                predictions[model_ids.index(best_name)] = tuned_pred
                model_scores[best_name] = search.best_score_

    # Stacking ensemble with meta-learner
    if len(predictions) >= 3:
        stacking_preds = []
        for name, key in models_to_train:
            if name in model_scores:
                oof_preds = get_oof_predictions(name, X_for(name), y, cat_feature_indices)
                stacking_preds.append(oof_preds)

        if len(stacking_preds) >= 3:
            meta_X = np.column_stack(stacking_preds)
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            meta_learner.fit(meta_X, y)

            test_preds = []
            for name, key in models_to_train:
                if name in model_scores:
                    model = trained_models.get(name) or get_model(name, n_samples, cat_feature_indices)
                    if model:
                        if name not in trained_models:
                            model.fit(X_for(name), y)
                            trained_models[name] = model
                        test_preds.append(model.predict_proba(Xt_for(name))[:, 1])

            if test_preds:
                test_meta_X = np.column_stack(test_preds[:len(stacking_preds)])
                stacking_pred = meta_learner.predict_proba(test_meta_X)[:, 1]

                best_model = list(model_scores.keys())[0] if not model_scores else max(model_scores, key=model_scores.get)
                best_idx = list(model_scores.keys()).index(best_model)
                best_pred = predictions[best_idx] if best_idx < len(predictions) else predictions[0]

                blending_weight = 0.6
                final_pred = blending_weight * stacking_pred + (1 - blending_weight) * best_pred

                pd.DataFrame({'row_id': test['row_id'], 'target': final_pred}).to_csv('final_submission.csv', index=False)
                print(f"Saved final_submission.csv (stacking blend, {len(predictions)} models)")
            else:
                weights = [1.0/np.var(p) for p in predictions]
                weights = [w/sum(weights) for w in weights]
                ensemble = np.average(predictions, axis=0, weights=weights)
                pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
                print(f"Saved final_submission.csv ({len(predictions)} models, variance-weighted)")
        else:
            weights = [1.0/np.var(p) for p in predictions]
            weights = [w/sum(weights) for w in weights]
            ensemble = np.average(predictions, axis=0, weights=weights)
            pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
            print(f"Saved final_submission.csv ({len(predictions)} models, variance-weighted)")
    elif len(predictions) > 1:
        weights = [1.0/np.var(p) for p in predictions]
        weights = [w/sum(weights) for w in weights]
        ensemble = np.average(predictions, axis=0, weights=weights)
        pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv ({len(predictions)} models, variance-weighted)")
    else:
        pd.DataFrame({'row_id': test['row_id'], 'target': predictions[0]}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv (1 model)")


if __name__ == '__main__':
    main()