#!/usr/bin/env python3
"""
AutoML training script for AAP competition.
Supports multiple experiment modes via --experiment argument.
Enhanced with feature engineering, CV, and stacking ensemble.
"""

import sys, pandas as pd, numpy as np, warnings, itertools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main entry point for training."""
    # Parse experiment mode
    experiment = 'ensemble'
    if '--experiment' in sys.argv:
        idx = sys.argv.index('--experiment')
        if idx + 1 < len(sys.argv):
            experiment = sys.argv[idx + 1]

    # Load data
    train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c.startswith('feature_')]
    num_cols = [c for c in train.columns if c not in cat_cols and c not in ['row_id', 'target']]
    y = train['target'].values
    X_train = train.drop(columns=['row_id', 'target']); X_test = test.drop(columns=['row_id'])

    # Impute
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
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

    # Target encoding for categorical columns
    target_encoded = {}
    for col in cat_cols:
        # Calculate target mean per category
        target_mean = train.groupby(col)['target'].mean()
        global_mean = train['target'].mean()
        
        # Apply encoding
        X_train[col + '_te'] = X_train[col].map(target_mean).fillna(global_mean)
        X_test[col + '_te'] = X_test[col].map(target_mean).fillna(global_mean)
        target_encoded[col] = target_mean

    # Drop original categorical columns after target encoding
    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)

    # Feature interactions (limited to top 5 pairs to avoid explosion)
    interaction_cols = []
    top_num_cols = num_cols[:5]  # Limit to top 5 numerical columns
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        if col1 in X_train.columns and col2 in X_train.columns:
            new_col = f"{col1}_{col2}_mul"
            X_train[new_col] = (X_train[col1] * X_train[col2]).astype(float)
            X_test[new_col] = (X_test[col1] * X_test[col2]).astype(float)
            interaction_cols.append(new_col)

    # Feature selection using mutual information (limit to reasonable number)
    all_features = X_train.columns.tolist()
    mi_scores = mutual_info_classif(X_train, y, random_state=42)
    mi_df = pd.DataFrame({'feature': all_features, 'mi_score': mi_scores})
    top_features = mi_df.nlargest(min(30, len(all_features)), 'mi_score')['feature'].tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Split for internal validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)

    def get_model(name, n_samples, cat_features=None):
        """Get model based on experiment mode."""
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

    def get_cv_score(model, X, y, n_folds=3):
        """Get cross-validation score."""
        n_folds = min(n_folds, 3)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean(), scores.std()

    n_samples = len(train)
    predictions = []
    model_ids = []
    model_scores = {}

    # Train individual models
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

    # Get categorical feature indices for CatBoost
    cat_feature_indices = None
    if cat_cols and experiment in ('cb', 'ensemble'):
        te_cols = [c for c in X_train.columns if c.endswith('_te')]
        if te_cols:
            cat_feature_indices = [list(X_train.columns).index(c) for c in te_cols]

    # Store trained models for stacking
    trained_models = {}

    for name, key in models_to_train:
        try:
            model = get_model(name, n_samples, cat_feature_indices)
            if model is None:
                continue
                
            # Cross-validation score
            cv_score, cv_std = get_cv_score(model, X_train, y)
            model_scores[name] = cv_score
            print(f"{name}: CV={cv_score:.4f} (+/- {cv_std:.4f})")
            
            # Train on full data
            model.fit(X_train, y)
            trained_models[name] = model
            pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
            model_ids.append(name)
            
            # Save individual prediction
            pd.DataFrame({'row_id': test['row_id'], 'target': pred}).to_csv(f'{key}_pred.csv', index=False)
        except Exception as e:
            print(f"{name}: Failed - {e}")
            continue

    # Stacking ensemble with meta-learner
    if len(predictions) >= 3:
        # Get out-of-fold predictions for stacking
        stacking_preds = []
        for name, key in models_to_train[:5]:
            if name in model_scores:
                model = get_model(name, n_samples, cat_feature_indices)
                oof_preds = cross_val_predict(model, X_train, y, cv=3, method='predict_proba')[:, 1]
                stacking_preds.append(oof_preds)
        
        if len(stacking_preds) >= 3:
            # Train meta-learner on OOF predictions
            meta_X = np.column_stack(stacking_preds)
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            meta_learner.fit(meta_X, y)
            
            # Get final predictions for test set from all models
            test_preds = []
            for name, key in models_to_train[:5]:
                if name in model_scores:
                    model = trained_models.get(name) or get_model(name, n_samples, cat_feature_indices)
                    if model:
                        if name not in trained_models:
                            model.fit(X_train, y)
                            trained_models[name] = model
                        test_preds.append(model.predict_proba(X_test)[:, 1])
            
            if test_preds:
                test_meta_X = np.column_stack(test_preds[:len(stacking_preds)])
                stacking_pred = meta_learner.predict_proba(test_meta_X)[:, 1]
                
                # Blend stacking with best individual model
                best_model = list(model_scores.keys())[0] if not model_scores else max(model_scores, key=model_scores.get)
                best_idx = list(model_scores.keys()).index(best_model)
                best_pred = predictions[best_idx] if best_idx < len(predictions) else predictions[0]
                
                # Weighted blend
                blending_weight = 0.6
                final_pred = blending_weight * stacking_pred + (1 - blending_weight) * best_pred
                
                pd.DataFrame({'row_id': test['row_id'], 'target': final_pred}).to_csv('final_submission.csv', index=False)
                print(f"Saved final_submission.csv (stacking blend, {len(predictions)} models)")
            else:
                # Fallback to variance-weighted ensemble
                weights = [1.0/np.var(p) for p in predictions]
                weights = [w/sum(weights) for w in weights]
                ensemble = np.average(predictions, axis=0, weights=weights)
                pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
                print(f"Saved final_submission.csv ({len(predictions)} models, variance-weighted)")
        else:
            # Fallback to variance-weighted ensemble
            weights = [1.0/np.var(p) for p in predictions]
            weights = [w/sum(weights) for w in weights]
            ensemble = np.average(predictions, axis=0, weights=weights)
            pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
            print(f"Saved final_submission.csv ({len(predictions)} models, variance-weighted)")
    elif len(predictions) > 1:
        # Fallback to variance-weighted ensemble
        weights = [1.0/np.var(p) for p in predictions]
        weights = [w/sum(weights) for w in weights]
        ensemble = np.average(predictions, axis=0, weights=weights)
        pd.DataFrame({'row_id': test['row_id'], 'target': ensemble}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv ({len(predictions)} models, variance-weighted)")
    else:
        # Single model fallback
        pd.DataFrame({'row_id': test['row_id'], 'target': predictions[0]}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv (1 model)")


if __name__ == '__main__':
    main()