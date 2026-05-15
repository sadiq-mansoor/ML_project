import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from scipy import stats
import pandas as pd
import os

RANDOM_STATE = 42
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'pr_auc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }

def cross_val_evaluate(pipeline, model, X, y, cv=None):
    """
    Run stratified 5-fold CV. pipeline is a preprocessing Pipeline,
    model is a LogisticRegression. Returns dict of metric arrays (one per fold).
    Uses clone(model) per fold to prevent state leakage.
    """
    from sklearn.base import clone as _clone
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_results = {'pr_auc': [], 'f1': [], 'recall': [], 'precision': [], 'accuracy': []}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_t = pipeline.fit_transform(X_train, y_train)
        X_val_t = pipeline.transform(X_val)

        m = _clone(model)
        m.fit(X_train_t, y_train)
        y_pred = m.predict(X_val_t)
        y_prob = m.predict_proba(X_val_t)[:, 1]

        for k, v in compute_metrics(y_val, y_pred, y_prob).items():
            fold_results[k].append(v)

    return {k: np.array(v) for k, v in fold_results.items()}

def summarize_cv_results(fold_results: dict) -> dict:
    """Return mean, std, and 95% CI for each metric. (ISSUE-5 fix)"""
    summary = {}
    for k, v in fold_results.items():
        n = len(v)
        mean = np.mean(v)
        se = stats.sem(v)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se) if n > 1 else (mean, mean)
        summary[k] = {
            'mean': mean,
            'std': np.std(v),
            'ci_low': ci[0],
            'ci_high': ci[1],
        }
    return summary

def paired_ttest(scores_a, scores_b):
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {'t_stat': t_stat, 'p_value': p_value}

def count_nonzero_coefs(model):
    if hasattr(model, 'coef_'):
        return int(np.sum(model.coef_ != 0))
    return None

def plot_learning_curve(pipeline, model, X, y, title, save_path):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline as SKPipeline
    steps = list(pipeline.steps) + [('model', clone(model))]
    combined = SKPipeline(steps)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        combined, X, y, cv=cv, scoring='average_precision',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), label='Train PR-AUC')
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
    ax.plot(train_sizes, val_scores.mean(axis=1), label='Val PR-AUC')
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('Training samples')
    ax.set_ylabel('PR-AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_train_val_gap(pipeline, X, y, C_values, title, save_path):
    """Plot train vs val PR-AUC and their gap across C values (bias-variance curve).
    Required by Section 4: Training vs Validation curves across regularization strength C.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline as SKPipeline
    from src.models import make_regularized

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_means, val_means, gaps = [], [], []

    for C in C_values:
        model = make_regularized('L2', C)
        steps = list(pipeline.steps) + [('model', clone(model))]
        combined = SKPipeline(steps)
        train_sizes, train_scores, val_scores = learning_curve(
            combined, X, y, cv=cv, scoring='average_precision',
            train_sizes=[1.0], n_jobs=-1
        )
        t = train_scores[0].mean()
        v = val_scores[0].mean()
        train_means.append(t)
        val_means.append(v)
        gaps.append(t - v)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.semilogx(C_values, train_means, 'o-', label='Train PR-AUC', color='steelblue')
    ax1.semilogx(C_values, val_means, 's--', label='Val PR-AUC', color='tomato')
    ax1.fill_between(C_values, val_means, train_means, alpha=0.15, color='orange', label='Gap')
    ax1.set_ylabel('PR-AUC')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Annotate bias/variance regions
    ax1.axvspan(C_values[0], C_values[1], alpha=0.07, color='blue')
    ax1.text(C_values[0] * 1.1, min(val_means) + 0.01, 'High Bias\n(underfitting)',
             color='blue', fontsize=8)
    ax1.axvspan(C_values[-2], C_values[-1], alpha=0.07, color='red')
    ax1.text(C_values[-2] * 1.05, min(val_means) + 0.01, 'High Variance\n(overfitting)',
             color='red', fontsize=8)

    ax2.semilogx(C_values, gaps, 'D-', color='darkorange', label='Train − Val gap')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('C (inverse regularization strength)')
    ax2.set_ylabel('Gap (Train − Val PR-AUC)')
    ax2.set_title('Gap Analysis — Overfitting Indicator')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_elasticnet_sweep(reg_results, L1_RATIOS, C_VALUES, dataset_name, pipe_name, save_path):
    """Plot ElasticNet l1_ratio sweep: PR-AUC vs l1_ratio for each C value.
    Required by Section 5: Elastic Net sweep visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(C_VALUES)))
    for C, color in zip(C_VALUES, colors):
        means, stds = [], []
        for l1r in L1_RATIOS:
            key = (f'ElasticNet(l1={l1r})', C, dataset_name, pipe_name)
            if key in reg_results:
                means.append(reg_results[key]['pr_auc'].mean())
                stds.append(reg_results[key]['pr_auc'].std())
            else:
                means.append(np.nan); stds.append(0)
        ax.errorbar(L1_RATIOS, means, yerr=stds, marker='o', capsize=4,
                    label=f'C={C}', color=color)
    ax.set_xlabel('l1_ratio  (0 = pure L2 → 1 = pure L1)')
    ax.set_ylabel('PR-AUC (mean ± std, 5-fold CV)')
    ax.set_title(f'Elastic Net l1_ratio Sweep — {dataset_name} Pipeline {pipe_name}')
    ax.legend(title='C value')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_coefficient_stability(pipeline, X, y, reg_type, C, cv, title, save_path):
    """Plot coefficient std across CV folds — Section 5 stability analysis requirement."""
    from src.models import make_regularized
    coef_list = []
    for train_idx, _ in cv.split(X, y):
        X_t = pipeline.fit_transform(X[train_idx], y[train_idx])
        m = make_regularized(reg_type, C)
        m.fit(X_t, y[train_idx])
        coef_list.append(m.coef_[0])

    coef_arr = np.array(coef_list)
    coef_std = np.std(coef_arr, axis=0)
    coef_mean = np.mean(coef_arr, axis=0)

    # Sort by absolute mean coefficient for readability
    order = np.argsort(np.abs(coef_mean))[::-1][:30]  # top-30 features

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(order)), coef_std[order], color='steelblue', alpha=0.8)
    ax.set_xlabel('Feature rank (by |mean coefficient|)')
    ax.set_ylabel('Coefficient std across CV folds')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curves_imbalance(imb_pipelines_dict, X, y, title, save_path):
    """Plot Precision-Recall curves for each imbalance method.
    Required by Section 6: Impact on precision vs recall tradeoff.
    imb_pipelines_dict: {method_name: fitted_imblearn_pipeline}
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    colors = {'SMOTE': 'steelblue', 'Undersample': 'tomato', 'ClassWeight': 'green'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for method_name, imb_pipe in imb_pipelines_dict.items():
        all_probs, all_true = [], []
        for train_idx, val_idx in cv.split(X, y):
            from sklearn.base import clone
            pipe_clone = clone(imb_pipe)
            pipe_clone.fit(X[train_idx], y[train_idx])
            probs = pipe_clone.predict_proba(X[val_idx])[:, 1]
            all_probs.extend(probs)
            all_true.extend(y[val_idx])
        precision, recall, _ = precision_recall_curve(all_true, all_probs)
        ap = average_precision_score(all_true, all_probs)
        color = colors.get(method_name, 'gray')
        ax.plot(recall, precision, label=f'{method_name} (AP={ap:.3f})', color=color, linewidth=1.5)

    # Baseline (no imbalance handling)
    ax.axhline(y=np.mean(y), color='gray', linestyle='--', linewidth=1, label='Random baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_regularization_curve(results_df, metric, title, save_path):
    """results_df: columns = [reg_type, C, mean, std]"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for reg_type, group in results_df.groupby('reg_type'):
        ax.errorbar(group['C'], group['mean'], yerr=group['std'],
                    label=reg_type, marker='o', capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel('C (regularization strength inverse)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_table(df: pd.DataFrame, name: str):
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, f'{name}.csv'), index=False)
