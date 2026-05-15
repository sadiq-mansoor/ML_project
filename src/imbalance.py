from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np

RANDOM_STATE = 42


def build_smote_pipeline(preprocessing_pipeline, model):
    steps = list(preprocessing_pipeline.steps) + [
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', model),
    ]
    return ImbPipeline(steps)


def build_undersample_pipeline(preprocessing_pipeline, model):
    steps = list(preprocessing_pipeline.steps) + [
        ('undersample', RandomUnderSampler(random_state=RANDOM_STATE)),
        ('model', model),
    ]
    return ImbPipeline(steps)


def build_class_weight_pipeline(preprocessing_pipeline, model_kwargs):
    """Returns a sklearn Pipeline where LogReg uses class_weight='balanced'.
    model_kwargs must include explicit penalty= key.
    """
    from sklearn.pipeline import Pipeline
    m = LogisticRegression(**{**model_kwargs, 'class_weight': 'balanced'})
    steps = list(preprocessing_pipeline.steps) + [('model', m)]
    return Pipeline(steps)


def cross_val_imbalance(imb_pipeline, X, y, cv=None):
    """CV evaluate an imblearn pipeline. Returns dict of metric arrays.
    Uses clone() per fold to prevent state leakage between folds.
    """
    from sklearn.metrics import average_precision_score, f1_score, recall_score
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_results = {'pr_auc': [], 'f1': [], 'recall': []}
    for train_idx, val_idx in cv.split(X, y):
        # Clone per fold — prevents fitted state from leaking across folds
        p = clone(imb_pipeline)
        p.fit(X[train_idx], y[train_idx])
        y_pred = p.predict(X[val_idx])
        y_prob = p.predict_proba(X[val_idx])[:, 1]

        fold_results['pr_auc'].append(average_precision_score(y[val_idx], y_prob))
        fold_results['f1'].append(f1_score(y[val_idx], y_pred, zero_division=0))
        fold_results['recall'].append(recall_score(y[val_idx], y_pred, zero_division=0))

    return {k: np.array(v) for k, v in fold_results.items()}
