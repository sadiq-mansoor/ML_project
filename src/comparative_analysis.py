import pandas as pd
import numpy as np
from src.evaluation import paired_ttest, save_table


def build_pipeline_dataset_table(all_results: dict) -> pd.DataFrame:
    """
    all_results: {(pipeline, dataset): fold_scores_dict}
    Returns Table 1 from implementation plan.
    """
    rows = []
    for (pipeline, dataset), fold_scores in all_results.items():
        row = {'Pipeline': pipeline, 'Dataset': dataset}
        for metric, scores in fold_scores.items():
            row[f'{metric}_mean'] = np.mean(scores)
            row[f'{metric}_std'] = np.std(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def build_regularization_table(reg_results: dict) -> pd.DataFrame:
    """
    reg_results: {(reg_type, C, dataset, pipeline): fold_scores_dict}
    CRITICAL: 4-tuple keys — 3-tuple keys cause pipeline B to overwrite pipeline A.
    """
    rows = []
    for (reg_type, C, dataset, pipeline), fold_scores in reg_results.items():
        rows.append({
            'Regularization': reg_type,
            'C': C,
            'Dataset': dataset,
            'Pipeline': pipeline,
            'pr_auc_mean': np.mean(fold_scores['pr_auc']),
            'pr_auc_std':  np.std(fold_scores['pr_auc']),
            'f1_mean':     np.mean(fold_scores['f1']),
            'f1_std':      np.std(fold_scores['f1']),
        })
    return pd.DataFrame(rows)


def build_imbalance_interaction_table(imb_results: dict) -> pd.DataFrame:
    """Table 4: imbalance method × regularization."""
    rows = []
    for (imb_method, reg_type, dataset), fold_scores in imb_results.items():
        rows.append({
            'Imbalance_Method': imb_method,
            'Regularization': reg_type,
            'Dataset': dataset,
            'pr_auc_mean': np.mean(fold_scores['pr_auc']),
            'pr_auc_std':  np.std(fold_scores['pr_auc']),
        })
    return pd.DataFrame(rows)


def test_hypothesis(scores_dict: dict, metric: str = 'pr_auc') -> pd.DataFrame:
    """
    scores_dict: {label: fold_scores_dict OR plain array}
    Handles both dict values (with metric key) and plain numpy arrays.
    Runs all pairwise paired t-tests. Returns DataFrame of results.
    """
    def _extract(v):
        if isinstance(v, dict):
            return v[metric]
        return np.asarray(v)

    labels = list(scores_dict.keys())
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            result = paired_ttest(_extract(scores_dict[a]), _extract(scores_dict[b]))
            rows.append({'Group A': a, 'Group B': b, **result})
    return pd.DataFrame(rows)
