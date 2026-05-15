"""Master script — runs the complete experiment pipeline."""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

# Suppress sklearn 1.8 deprecation warnings — penalty= API change is handled in models.py
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from src.utils import log_section, Timer, RANDOM_STATE, logger
from src.data_loader import get_all_datasets, load_bonn_eeg
from src.preprocessing import get_pipelines, SafeSelectKBest
from sklearn.feature_selection import mutual_info_classif
from src.models import (
    make_baseline, make_underfit,
    make_regularized, C_VALUES, L1_RATIOS
)
from src.evaluation import (
    cross_val_evaluate, summarize_cv_results, plot_learning_curve,
    plot_regularization_curve, plot_train_val_gap, plot_elasticnet_sweep,
    plot_coefficient_stability, plot_pr_curves_imbalance,
    save_table, count_nonzero_coefs, compute_metrics, FIGURES_DIR
)
from src.imbalance import (
    build_smote_pipeline, build_undersample_pipeline,
    build_class_weight_pipeline, cross_val_imbalance
)
from src.comparative_analysis import (
    build_pipeline_dataset_table, build_regularization_table,
    build_imbalance_interaction_table, test_hypothesis
)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def main():
    # Set global seed once here, not at module import level (ISSUE-4 fix)
    np.random.seed(RANDOM_STATE)

    # ─── Load Datasets ────────────────────────────────────────────────
    log_section('Loading Datasets')
    datasets = get_all_datasets()  # bonn_eeg = 6-dim features (fast for A/B)
    # Load Bonn raw signals separately — only needed for Pipeline C
    bonn_raw_X, bonn_raw_y = load_bonn_eeg(raw=True)

    # ─── Holdout Split ────────────────────────────────────────────────
    holdout = {}
    for name, (X, y) in datasets.items():
        X_main, X_test, y_main, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        holdout[name] = (X_main, y_main, X_test, y_test)

    # Separate holdout for Bonn raw (Pipeline C only)
    bonn_raw_main, bonn_raw_test, bonn_raw_y_main, bonn_raw_y_test = train_test_split(
        bonn_raw_X, bonn_raw_y, test_size=0.2, stratify=bonn_raw_y, random_state=RANDOM_STATE
    )

    # ─── Section 3: Baseline ──────────────────────────────────────────
    log_section('Section 3: Baseline Model')
    baseline_results = {}
    for dataset_name, (X, y, _, _) in holdout.items():
        pipelines = get_pipelines(dataset_name)
        for pipe_name, pipe in pipelines.items():
            # Pipeline C on Bonn needs raw 4097-dim signals
            if pipe_name == 'C' and dataset_name == 'bonn_eeg':
                X_use, y_use = bonn_raw_main, bonn_raw_y_main
            else:
                X_use, y_use = X, y
            with Timer() as t:
                results = cross_val_evaluate(pipe, make_baseline(), X_use, y_use, CV)
            summary = summarize_cv_results(results)
            baseline_results[(pipe_name, dataset_name)] = results
            logger.info(f"Baseline | {dataset_name} | Pipeline {pipe_name} | "
                        f"PR-AUC: {summary['pr_auc']['mean']:.3f}±{summary['pr_auc']['std']:.3f} "
                        f"| F1: {summary['f1']['mean']:.3f} | Time: {t.elapsed:.1f}s")

    table1 = build_pipeline_dataset_table(baseline_results)
    save_table(table1, 'table1_pipeline_dataset')

    # ─── Section 4: Overfitting / Underfitting ────────────────────────
    log_section('Section 4: Overfitting & Underfitting')
    # Use UCI Seizure (largest, most features → best for poly explosion)
    X_uci, y_uci, _, _ = holdout['uci_seizure']
    from src.preprocessing import build_pipeline_a

    pipe_a = build_pipeline_a(k=20)
    underfit_results = cross_val_evaluate(pipe_a, make_underfit(), X_uci, y_uci, CV)
    logger.info(f"Underfit | PR-AUC: {underfit_results['pr_auc'].mean():.3f}")

    plot_learning_curve(
        build_pipeline_a(k=20), make_baseline(), X_uci, y_uci,
        title='Learning Curve — Baseline (UCI Seizure)',
        save_path=os.path.join(FIGURES_DIR, 'learning_curve_baseline.png')
    )
    plot_learning_curve(
        build_pipeline_a(k=5), make_underfit(), X_uci, y_uci,
        title='Learning Curve — Underfitting (C=0.0001, k=5)',
        save_path=os.path.join(FIGURES_DIR, 'learning_curve_underfit.png')
    )

    # Overfitting with polynomial features (apply after selecting top-20 to keep feasible)
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from functools import partial as _partial
    poly_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('fs', SafeSelectKBest(_partial(mutual_info_classif, random_state=RANDOM_STATE), k=20)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ])
    # Use make_overfit_pipeline's model config directly (ISSUE-2 + ISSUE-3 fix)
    overfit_model_lr = LogisticRegression(
        l1_ratio=0.0, C=1e9, solver='lbfgs', max_iter=10000, random_state=RANDOM_STATE
    )
    overfit_results = cross_val_evaluate(poly_pipe, overfit_model_lr, X_uci, y_uci, CV)
    logger.info(f"Overfit (poly) | PR-AUC: {overfit_results['pr_auc'].mean():.3f}")

    # Overfit learning curve (required by §4)
    plot_learning_curve(
        poly_pipe, overfit_model_lr, X_uci, y_uci,
        title='Learning Curve — Overfitting (Poly deg-2, no regularization)',
        save_path=os.path.join(FIGURES_DIR, 'learning_curve_overfit.png')
    )

    # Train vs Val gap across C values — bias-variance curve (required by §4)
    extended_C = [0.001, 0.01, 0.1, 1, 10, 100]
    plot_train_val_gap(
        build_pipeline_a(k=20), X_uci, y_uci,
        C_values=extended_C,
        title='Train vs Val PR-AUC across Regularization Strength (UCI Seizure, Pipeline A, L2)',
        save_path=os.path.join(FIGURES_DIR, 'train_val_gap_C.png')
    )

    # ─── Section 5: Regularization Study ─────────────────────────────
    log_section('Section 5: Regularization Study')
    reg_results = {}
    sparsity_rows = []

    for dataset_name, (X, y, _, _) in holdout.items():
        pipelines = get_pipelines(dataset_name)
        for pipe_name, pipe in pipelines.items():
            # Pipeline C on Bonn needs raw signals
            if pipe_name == 'C' and dataset_name == 'bonn_eeg':
                X_use, y_use = bonn_raw_main, bonn_raw_y_main
            else:
                X_use, y_use = X, y
            for reg_type in ['L1', 'L2', 'ElasticNet']:
                for C in C_VALUES:
                    if reg_type == 'ElasticNet':
                        for l1r in L1_RATIOS:
                            model = make_regularized(reg_type, C, l1r)
                            results = cross_val_evaluate(pipe, model, X_use, y_use, CV)
                            reg_results[(f'{reg_type}(l1={l1r})', C, dataset_name, pipe_name)] = results
                    else:
                        model = make_regularized(reg_type, C)
                        results = cross_val_evaluate(pipe, model, X_use, y_use, CV)
                        reg_results[(reg_type, C, dataset_name, pipe_name)] = results

                        # Track sparsity + feature count after preprocessing (ISSUE-1 fix)
                        from sklearn.model_selection import StratifiedKFold as SKF
                        cv_tmp = SKF(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                        train_idx, val_idx = list(cv_tmp.split(X_use, y_use))[-1]
                        with Timer() as t_sparsity:
                            X_t = pipe.fit_transform(X_use[train_idx], y_use[train_idx])
                            m_sparse = make_regularized(reg_type, C)
                            m_sparse.fit(X_t, y_use[train_idx])
                        feature_count_after = X_t.shape[1]
                        sparsity_rows.append({
                            'reg_type': reg_type, 'C': C,
                            'dataset': dataset_name, 'pipeline': pipe_name,
                            'nonzero_coefs': count_nonzero_coefs(m_sparse),
                            'feature_count_after_preprocessing': feature_count_after,
                            'training_time_s': round(t_sparsity.elapsed, 3),
                        })
    table3 = build_regularization_table(reg_results)
    save_table(table3, 'table3_regularization')
    save_table(pd.DataFrame(sparsity_rows), 'table2_sparsity')

    # Plot regularization curves for UCI Seizure Pipeline A
    curve_data = []
    for reg_type in ['L1', 'L2']:
        for C in C_VALUES:
            r = reg_results.get((reg_type, C, 'uci_seizure', 'A'))  # key collision fix
            if r:
                curve_data.append({'reg_type': reg_type, 'C': C,
                                   'mean': r['pr_auc'].mean(), 'std': r['pr_auc'].std()})
    if curve_data:
        plot_regularization_curve(
            pd.DataFrame(curve_data), 'PR-AUC',
            title='Regularization Strength vs PR-AUC (UCI Seizure, Pipeline A)',
            save_path=os.path.join(FIGURES_DIR, 'reg_curve_uci.png')
        )

    # ElasticNet l1_ratio sweep plot — required by §5
    plot_elasticnet_sweep(
        reg_results, L1_RATIOS, C_VALUES,
        dataset_name='uci_seizure', pipe_name='A',
        save_path=os.path.join(FIGURES_DIR, 'elasticnet_l1ratio_sweep.png')
    )

    # Coefficient stability (std across CV folds) — required by §5
    X_uci_stab, y_uci_stab, _, _ = holdout['uci_seizure']
    from src.preprocessing import build_pipeline_a as _bpa
    for reg_type in ['L1', 'L2']:
        plot_coefficient_stability(
            _bpa(k=20), X_uci_stab, y_uci_stab,
            reg_type=reg_type, C=1.0, cv=CV,
            title=f'Coefficient Stability across CV Folds — {reg_type}, C=1 (UCI Seizure, Pipeline A)',
            save_path=os.path.join(FIGURES_DIR, f'coef_stability_{reg_type.lower()}.png')
        )
    # ─── Section 6: Class Imbalance ───────────────────────────────────
    log_section('Section 6: Class Imbalance')
    imb_results = {}
    best_C = 1.0  # Use C=1 as default best from reg study

    for dataset_name, (X, y, _, _) in holdout.items():
        if dataset_name == 'bonn_eeg':
            continue  # Bonn is balanced — imbalance experiments less relevant
        pipelines = get_pipelines(dataset_name)
        pipe = pipelines['A']  # Use Pipeline A as reference

        for imb_method in ['SMOTE', 'Undersample', 'ClassWeight']:
            for reg_type in ['L1', 'L2', 'ElasticNet']:
                if reg_type == 'L1':
                    model_kwargs = {'l1_ratio': 1.0, 'solver': 'saga',
                                    'C': best_C, 'max_iter': 10000, 'random_state': RANDOM_STATE}
                elif reg_type == 'L2':
                    model_kwargs = {'l1_ratio': 0.0, 'solver': 'lbfgs',
                                    'C': best_C, 'max_iter': 5000, 'random_state': RANDOM_STATE}
                else:  # ElasticNet
                    model_kwargs = {'l1_ratio': 0.5, 'solver': 'saga',
                                    'C': best_C, 'max_iter': 10000, 'random_state': RANDOM_STATE}

                if imb_method == 'SMOTE':
                    from sklearn.linear_model import LogisticRegression as LR
                    model = LR(**model_kwargs)
                    imb_pipe = build_smote_pipeline(pipe, model)
                    results = cross_val_imbalance(imb_pipe, X, y, CV)
                elif imb_method == 'Undersample':
                    from sklearn.linear_model import LogisticRegression as LR
                    model = LR(**model_kwargs)
                    imb_pipe = build_undersample_pipeline(pipe, model)
                    results = cross_val_imbalance(imb_pipe, X, y, CV)
                else:  # ClassWeight
                    imb_pipe = build_class_weight_pipeline(pipe, model_kwargs)
                    results = cross_val_imbalance(imb_pipe, X, y, CV)

                imb_results[(imb_method, reg_type, dataset_name)] = results
                logger.info(f"Imbalance | {dataset_name} | {imb_method} | {reg_type} | "
                            f"PR-AUC: {results['pr_auc'].mean():.3f}")

    table4 = build_imbalance_interaction_table(imb_results)
    save_table(table4, 'table4_imbalance_interaction')

    # PR curves per imbalance method — required by §6
    for dataset_name, (X, y, _, _) in holdout.items():
        if dataset_name == 'bonn_eeg':
            continue
        pipe = get_pipelines(dataset_name)['A']
        best_model_kwargs = {
            'l1_ratio': 0.0, 'solver': 'lbfgs',
            'C': best_C, 'max_iter': 5000, 'random_state': RANDOM_STATE,
        }
        pr_pipelines = {
            'SMOTE': build_smote_pipeline(pipe, LogisticRegression(**best_model_kwargs)),
            'Undersample': build_undersample_pipeline(pipe, LogisticRegression(**best_model_kwargs)),
            'ClassWeight': build_class_weight_pipeline(pipe, best_model_kwargs),
        }
        plot_pr_curves_imbalance(
            pr_pipelines, X, y,
            title=f'Precision-Recall Curves by Imbalance Method — {dataset_name} (L2, C=1)',
            save_path=os.path.join(FIGURES_DIR, f'pr_curves_imbalance_{dataset_name}.png')
        )

    # ─── Section 7: Statistical Tests (H1–H5) ────────────────────────
    log_section('Section 7: Hypothesis Testing')

    # H1: Pipeline effect across datasets (A vs B; Bonn excluded — Pipeline A is no-op there)
    h1_rows = []
    for ds in ['uci_seizure', 'eeg_eye_state']:
        h1_scores = {
            f'Pipeline_{p}': baseline_results[(p, ds)]
            for p in ['A', 'B'] if (p, ds) in baseline_results
        }
        if len(h1_scores) >= 2:
            df = test_hypothesis(h1_scores, metric='pr_auc')
            df['Dataset'] = ds
            h1_rows.append(df)
    h1_df = pd.concat(h1_rows, ignore_index=True) if h1_rows else pd.DataFrame()
    save_table(h1_df, 'h1_pipeline_ttest')
    logger.info(f"H1 results:\n{h1_df.to_string()}")

    # H2: Regularization generalization — Pipeline A, best C=1 across datasets (key collision fix)
    h2_scores = {}
    for reg_type in ['L1', 'L2', 'ElasticNet(l1=0.5)']:
        fold_scores = []
        for ds in ['uci_seizure', 'eeg_eye_state']:
            key = (reg_type, 1.0, ds, 'A')
            if key in reg_results:
                fold_scores.append(reg_results[key]['pr_auc'])
        if fold_scores:
            h2_scores[reg_type] = {'pr_auc': np.concatenate(fold_scores)}
    if len(h2_scores) >= 2:
        h2_df = test_hypothesis(h2_scores, metric='pr_auc')
        save_table(h2_df, 'h2_regularization_ttest')
        logger.info(f"H2 results:\n{h2_df.to_string()}")

    # H3: ElasticNet vs L1 vs L2 at best C=1 on UCI Seizure, Pipeline A (key collision fix)
    h3_scores = {}
    for reg_type in ['L1', 'L2', 'ElasticNet(l1=0.5)']:
        key = (reg_type, 1.0, 'uci_seizure', 'A')
        if key in reg_results:
            h3_scores[reg_type] = reg_results[key]
    if len(h3_scores) >= 2:
        h3_df = test_hypothesis(h3_scores, metric='pr_auc')
        save_table(h3_df, 'h3_elasticnet_ttest')
        logger.info(f"H3 results:\n{h3_df.to_string()}")

    # H4: Imbalance method × regularization interaction (BUG-3 fix)
    h4_rows = []
    for reg_type in ['L1', 'L2', 'ElasticNet']:
        h4_scores = {
            m: imb_results[(m, reg_type, 'uci_seizure')]
            for m in ['SMOTE', 'Undersample', 'ClassWeight']
            if (m, reg_type, 'uci_seizure') in imb_results
        }
        if len(h4_scores) >= 2:
            df = test_hypothesis(h4_scores, metric='pr_auc')
            df['Regularization'] = reg_type
            h4_rows.append(df)
    if h4_rows:
        h4_df = pd.concat(h4_rows, ignore_index=True)
        save_table(h4_df, 'h4_imbalance_ttest')
        logger.info(f"H4 results:\n{h4_df.to_string()}")

    # H5: Baseline vs polynomial overfitting significance (BUG-3 fix)
    h5_scores = {
        'Baseline': baseline_results[('A', 'uci_seizure')],
        'Polynomial_Overfit': overfit_results,
    }
    h5_df = test_hypothesis(h5_scores, metric='pr_auc')
    save_table(h5_df, 'h5_polynomial_ttest')
    logger.info(f"H5 results:\n{h5_df.to_string()}")

    # ─── Final Holdout Evaluation ─────────────────────────────────────
    log_section('Final Holdout Evaluation')  # BUG-4 fix
    from src.preprocessing import build_pipeline_a
    holdout_rows = []
    for dataset_name, (X_train, y_train, X_test, y_test) in holdout.items():
        k = min(20, X_train.shape[1])
        pipe = build_pipeline_a(k=k)
        model = make_regularized('L2', C=1.0)
        X_train_t = pipe.fit_transform(X_train, y_train)
        X_test_t = pipe.transform(X_test)
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        y_prob = model.predict_proba(X_test_t)[:, 1]
        m = compute_metrics(y_test, y_pred, y_prob)
        holdout_rows.append({'dataset': dataset_name, **m})
        logger.info(f"Holdout | {dataset_name} | PR-AUC: {m['pr_auc']:.3f} | F1: {m['f1']:.3f}")
    save_table(pd.DataFrame(holdout_rows), 'holdout_final_evaluation')

    logger.info('\n=== ALL EXPERIMENTS COMPLETE ===')
    logger.info(f"Tables saved to: results/tables/")
    logger.info(f"Figures saved to: results/figures/")


if __name__ == '__main__':
    main()
