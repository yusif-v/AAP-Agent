#!/usr/bin/env python3
"""
AutoML training script for AAP competition.
Supports multiple experiment modes via --experiment argument.
Enhanced with feature engineering, CV, and NNLS ensemble blending.
"""

import sys, pandas as pd, numpy as np, warnings, itertools, re
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import warnings
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
    'lr': {'C': [0.01, 0.1, 1.0, 10.0]},
    # No 'iterations' here - early stopping already controls tree count (see
    # SeedAveragedCatBoost/EarlyStoppingXGB/EarlyStoppingLGBM), so tuning searches
    # depth/learning_rate/regularization instead of re-litigating the ceiling.
    'cb': {'depth': [4, 5, 6, 7, 8], 'learning_rate': [0.02, 0.03, 0.05, 0.08],
           'l2_leaf_reg': [1, 3, 5, 7, 9]},
}


def get_adaptive_params(n_samples):
    """Returns model parameters adapted to dataset size.
    Small datasets need simpler models to avoid overfitting; large datasets
    can support higher capacity."""
    if n_samples < 500:
        # Very small - minimal trees, shallow, high regularization
        return {
            'n_seeds': 1,
            'cb_iterations': 500,
            'cb_depth': 4,
            'cb_lr': 0.05,
            'xgb_n_est': 200,
            'xgb_depth': 3,
            'xgb_lr': 0.03,
            'et_n_est': 100,
            'et_depth': 4,
        }
    elif n_samples < 2000:
        # Small - moderate complexity
        return {
            'n_seeds': 3,
            'cb_iterations': 1000,
            'cb_depth': 5,
            'cb_lr': 0.03,
            'xgb_n_est': 300,
            'xgb_depth': 4,
            'xgb_lr': 0.03,
            'et_n_est': 200,
            'et_depth': 6,
        }
    elif n_samples < 10000:
        # Medium
        return {
            'n_seeds': 3,
            'cb_iterations': 1500,
            'cb_depth': 6,
            'cb_lr': 0.03,
            'xgb_n_est': 500,
            'xgb_depth': 4,
            'xgb_lr': 0.03,
            'et_n_est': 300,
            'et_depth': 8,
        }
    else:
        # Large - full capacity but 3 seeds (5 is too slow, 3 gives equivalent variance reduction)
        return {
            'n_seeds': 3,
            'cb_iterations': 2000,
            'cb_depth': 6,
            'cb_lr': 0.03,
            'xgb_n_est': 2000,
            'xgb_depth': 4,
            'xgb_lr': 0.03,
            'et_n_est': 300,
            'et_depth': 10,
        }


class SeedAveragedCatBoost:
    """Trains one CatBoostClassifier per seed and averages predict_proba - a cheap
    variance reducer on a dataset this small (~15k rows), where a single seed's fold
    assignment can swing the score more than a genuine modeling improvement would."""

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
            # Each seed gets its own early-stopping split (varying with the seed) rather
            # than a single hardcoded iteration count - lets iterations run until they
            # actually stop helping instead of guessing a fixed ceiling.
            X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=seed,
                                                     stratify=y)
            model = CatBoostClassifier(cat_features=self.cat_features or None,
                                        random_state=seed, verbose=False,
                                        early_stopping_rounds=50, iterations=self.iterations, **self.params)
            model.fit(X_tr, y_tr, eval_set=(X_es, y_es))
            self.models.append(model)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class EarlyStoppingXGB:
    """XGBoost with a high n_estimators ceiling and early stopping on an internal
    validation split, instead of a small hardcoded n_estimators guess."""

    def __init__(self, n_estimators=2000, **params):
        self.params = params
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X, y):
        from xgboost import XGBClassifier
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        self.model = XGBClassifier(n_estimators=self.n_estimators, early_stopping_rounds=50,
                                   **self.params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class EarlyStoppingLGBM:
    """LightGBM with a high n_estimators ceiling and early stopping on an internal
    validation split, instead of a small hardcoded n_estimators guess."""

    def __init__(self, n_estimators=2000, **params):
        self.params = params
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X, y):
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        self.model = LGBMClassifier(n_estimators=self.n_estimators, **self.params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],
                       callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def main():
    """Main entry point for training."""
    # Parse experiment mode
    experiment = 'ensemble'
    if '--experiment' in sys.argv:
        idx = sys.argv.index('--experiment')
        if idx + 1 < len(sys.argv):
            experiment = sys.argv[idx + 1]
    tune = '--tune' in sys.argv

    # Load data
    train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
    raw_cat_cols = [c for c in train.columns
                    if (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
                    and c.startswith('feature_')]

    # Ordinal columns (e.g. 'ord_0'..'ord_6') carry a monotonic target-rate relationship -
    # decode to their integer rank instead of target-encoding them like an unordered
    # category, which would flatten that order into an arbitrary-looking float.
    ORDINAL_RE = re.compile(r'^ord_(\d+)$')
    ordinal_cols = [c for c in raw_cat_cols
                    if train[c].dropna().astype(str).str.match(ORDINAL_RE).all()]
    cat_cols = [c for c in raw_cat_cols if c not in ordinal_cols]
    num_cols = [c for c in train.columns if c not in cat_cols and c not in ['row_id', 'target']]
    y = train['target'].values
    X_train = train.drop(columns=['row_id', 'target']); X_test = test.drop(columns=['row_id'])

    for col in ordinal_cols:
        X_train[col] = X_train[col].str.extract(r'(\d+)$', expand=False).astype(float)
        X_test[col] = X_test[col].str.extract(r'(\d+)$', expand=False).astype(float)

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

    # CatBoost gets its own feature matrix with raw categorical strings intact (for the
    # remaining unordered cat_cols - ordinal_cols are already numeric by this point) -
    # target encoding below replaces cat_cols with numeric _te columns for the other models,
    # and CatBoost's whole value-add is native categorical handling, not target-encoded floats.
    cb_X_train = X_train.copy()
    cb_X_test = X_test.copy()

    # Out-of-fold (K-fold) target encoding for the remaining unordered categorical columns.
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

    # Drop original categorical columns after target encoding
    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)

    # Feature interactions - pick the top 5 numerical features by mutual information with
    # the target, not just the first 5 columns in source-file order
    interaction_cols = []
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
            interaction_cols.append(new_col)

    n_samples = len(train)
    adaptive = get_adaptive_params(n_samples)

    def get_model(name, n_samples, cat_features=None):
        """Get model based on experiment mode, with size-adaptive hyperparameters."""
        if name == 'rf':
            n_est = 300 if n_samples >= 10000 else (200 if n_samples >= 5000 else 100)
            depth = 8 if n_samples >= 10000 else (6 if n_samples >= 2000 else 4)
            return RandomForestClassifier(n_estimators=n_est, max_depth=depth, min_samples_split=5,
                                         random_state=42, n_jobs=-1)
        if name == 'et':
            n_est = adaptive['et_n_est']
            depth = adaptive['et_depth']
            return ExtraTreesClassifier(n_estimators=n_est, max_depth=depth,
                                        random_state=42, n_jobs=-1)
        if name == 'xgb':
            return EarlyStoppingXGB(n_estimators=adaptive['xgb_n_est'], max_depth=adaptive['xgb_depth'],
                                    learning_rate=adaptive['xgb_lr'],
                                    random_state=42, n_jobs=-1, verbosity=0)
        if name == 'lgbm':
            return EarlyStoppingLGBM(n_estimators=2000, max_depth=6 if n_samples >= 2000 else 4,
                                      learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)
        if name == 'gb':
            return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                              n_iter_no_change=20, validation_fraction=0.15,
                                              random_state=42)
        if name == 'lr':
            # LogisticRegression pipeline with scaling - excellent baseline for weak-signal data
            # and small datasets where tree ensembles overfit. Very cheap to compute.
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=5000, C=0.1, random_state=42))
        if name == 'cb':
            from catboost import CatBoostClassifier
            cb_seeds = (42, 142, 242, 342, 442)[:adaptive['n_seeds']]
            return SeedAveragedCatBoost(cat_features=cat_features, seeds=cb_seeds,
                                       iterations=adaptive['cb_iterations'],
                                       depth=adaptive['cb_depth'],
                                       learning_rate=adaptive['cb_lr'])
        return None

    def get_cv_and_oof(name, X, y, cat_features=None, n_folds=5):
        """Cross-validation score plus out-of-fold predictions, computed in a single
        pass. Builds a fresh model per fold via get_model() instead of sklearn's
        clone() - CatBoostClassifier with cat_features can't be cloned.
        Returns (mean_auc, std_auc, oof_predictions)."""
        n_folds = min(n_folds, 3) if n_samples < 2000 else n_folds
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        oof = np.zeros(len(X))
        for tr_idx, val_idx in cv.split(X, y):
            model = get_model(name, len(tr_idx), cat_features)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            oof[val_idx] = pred
            scores.append(roc_auc_score(y[val_idx], pred))
        scores = np.array(scores)
        return scores.mean(), scores.std(), oof

    predictions = []
    model_ids = []
    model_scores = {}
    trained_models = {}

    # Train individual models. 'ensemble' is pruned to cb (consistently the strongest
    # model by a wide margin) plus et and xgb (the best-performing diverse alternatives -
    # a bagging model and a differently-regularized booster). rf/lgbm/gb are consistently
    # the weakest and most redundant with the others, so they're dropped from the default
    # ensemble but remain available individually via --experiment for comparison.
    # 'lr' is included as a cheap, complementary baseline that often beats tree ensembles
    # on weak-signal data where overfitting is the dominant problem.
    models_to_train = []
    if experiment == 'rf':
        models_to_train.append(('rf', 'rf'))
    if experiment in ('et', 'ensemble'):
        models_to_train.append(('et', 'et'))
    if experiment in ('xgb', 'ensemble'):
        models_to_train.append(('xgb', 'xgb'))
    if experiment in ('lr', 'ensemble'):
        models_to_train.append(('lr', 'lr'))
    if experiment == 'lgbm':
        models_to_train.append(('lgbm', 'lgbm'))
    if experiment == 'gb':
        models_to_train.append(('gb', 'gb'))

    # Get categorical feature indices for CatBoost, relative to cb_X_train (which keeps
    # the raw categorical strings rather than the target-encoded numeric columns). CatBoost
    # still runs even when there are no categorical columns at all - it's a perfectly
    # capable model on purely numeric data too, it just gets an empty cat_features list.
    cat_feature_indices = [cb_X_train.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    if experiment in ('cb', 'ensemble'):
        models_to_train.append(('cb', 'cb'))

    def X_for(name):
        return cb_X_train if name == 'cb' else X_train

    def Xt_for(name):
        return cb_X_test if name == 'cb' else X_test

    # Cache OOF predictions from CV pass to avoid retraining for the NNLS blend
    oof_cache = {}
    for name, key in models_to_train:
        try:
            model = get_model(name, n_samples, cat_feature_indices)
            if model is None:
                continue

            # CV score + OOF predictions in a single pass (avoids double-training)
            cv_score, cv_std, oof = get_cv_and_oof(name, X_for(name), y, cat_feature_indices)
            oof_cache[name] = oof
            model_scores[name] = cv_score
            print(f"{name}: CV={cv_score:.4f} (+/- {cv_std:.4f})")

            # Train on full data
            model.fit(X_for(name), y)
            trained_models[name] = model
            pred = model.predict_proba(Xt_for(name))[:, 1]
            predictions.append(pred)
            model_ids.append(name)

            # Save individual prediction
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
            # re-score those params with our own manual per-fold CV (matching get_cv_score's
            # methodology) rather than trusting CatBoost's internal train/test-split search score.
            print(f"Tuning cb (baseline CV={model_scores['cb']:.4f}) via CatBoost's native search...")
            from catboost import CatBoostClassifier
            search_model = CatBoostClassifier(iterations=adaptive['cb_iterations'], depth=adaptive['cb_depth'],
                                               learning_rate=adaptive['cb_lr'],
                                               cat_features=cat_feature_indices or None,
                                               early_stopping_rounds=50,
                                               verbose=False, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            search_result = search_model.randomized_search(
                dist, X=cb_X_train, y=y, cv=cv, n_iter=15,
                search_by_train_test_split=False, verbose=False, plot=False)
            best_params = search_result['params']

            def get_tuned_cb():
                params = dict(iterations=adaptive['cb_iterations'], depth=adaptive['cb_depth'],
                              learning_rate=adaptive['cb_lr'])
                params.update(best_params)
                return SeedAveragedCatBoost(cat_features=cat_feature_indices, **params)

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
        elif best_name == 'lr':
            # lr pipeline isn't compatible with RandomizedSearchCV directly,
            # but it's already well-tuned with C=0.1 and StandardScaler.
            print(f"Skipping tuning for lr: already uses regularized scaled pipeline (C=0.1).")
        elif best_name in ('xgb', 'lgbm'):
            print(f"Skipping RandomizedSearchCV for {best_name}: not a clonable sklearn "
                  "estimator (uses early stopping internally instead).")
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

    # Ensemble blend: OOF-optimized non-negative weights instead of a stacking
    # meta-learner plus an ad hoc fixed blend weight. With the ensemble pruned to
    # cb/et/xgb/lr (each genuinely different: native-categorical boosting, bagging,
    # a differently-regularized booster, and linear model), a simple weighted average
    # tuned on real out-of-fold performance is more transparent than a learned meta-model
    # and avoids guessing a blend ratio - each model's OOF contribution decides its own weight.
    if len(predictions) >= 2:
        names = [name for name, key in models_to_train if name in model_scores]
        oof_matrix = np.column_stack([oof_cache[name] for name in names])

        from scipy.optimize import nnls
        weights, _ = nnls(oof_matrix, y.astype(float))
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Enforce a conditional soft floor: when the top-2 models have nearly-identical
        # OOF scores (gap < 0.01 AUC), NNLS can zero out the runner-up due to noise rather
        # than genuine inferiority. In that case, ensure the top-2 each get at least 10%
        # weight so the ensemble hedges against this OOF-vs-test mismatch. On splits where
        # one model clearly dominates (gap >= 0.01), pure NNLS is left untouched.
        oof_aucs = list(roc_auc_score(y, oof_matrix[:, i]) for i in range(len(names)))
        sorted_idx = np.argsort(-np.array(oof_aucs))
        if len(sorted_idx) >= 2 and (oof_aucs[sorted_idx[0]] - oof_aucs[sorted_idx[1]]) < 0.01:
            for idx in sorted_idx[:2]:
                if weights[idx] < 0.10:
                    weights[idx] = 0.10
            # Renormalize
            weights = weights / weights.sum()

        names = [name for name, key in models_to_train if name in model_scores]
        oof_blend_auc = roc_auc_score(y, oof_matrix @ weights)
        print(f"OOF-weighted blend: {dict(zip(names, weights.round(3)))}, OOF AUC={oof_blend_auc:.4f}")

        final_pred = np.column_stack(predictions) @ weights
        pd.DataFrame({'row_id': test['row_id'], 'target': final_pred}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv (OOF-weighted blend, {len(predictions)} models)")
    else:
        # Single model fallback
        pd.DataFrame({'row_id': test['row_id'], 'target': predictions[0]}).to_csv('final_submission.csv', index=False)
        print(f"Saved final_submission.csv (1 model)")


if __name__ == '__main__':
    main()
