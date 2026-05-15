# Requirements Traceability and Report Guide

This document translates the assignment in [Project_requirements.md](../Project_requirements.md) into repository-ready evidence. It is designed to help with the final report, viva discussion, and presentation preparation.

## 1. Requirement Coverage Matrix

| Requirement from assignment | Repository evidence | Status | Notes for final submission |
| --- | --- | --- | --- |
| Select at least 3 epileptic seizure datasets or related EEG datasets with justification | `src/data_loader.py`, `implementation_plan.md`, `data/` | Implemented | Two seizure-focused datasets plus one independent EEG generalization dataset are present locally |
| Build at least 2 preprocessing pipelines | `src/preprocessing.py` | Implemented | Pipelines A and B apply broadly; Pipeline C is additionally implemented for Bonn raw signals |
| Use logistic regression as the baseline model | `src/models.py`, `run_all.py` | Implemented | The whole study intentionally keeps logistic regression as the central model family |
| Report accuracy, F1-score, and PR-AUC | `src/evaluation.py`, `results/tables/table1_pipeline_dataset.csv`, `results/tables/table3_regularization.csv` | Implemented | Precision and recall are also tracked, which strengthens the analysis |
| Demonstrate underfitting and overfitting | `run_all.py`, `results/figures/learning_curve_underfit.png`, `results/figures/learning_curve_overfit.png`, `results/figures/train_val_gap_C.png` | Implemented | This is one of the better-supported sections of the repo |
| Compare L1, L2, and Elastic Net | `src/models.py`, `results/tables/table2_sparsity.csv`, `results/tables/table3_regularization.csv`, `results/figures/reg_curve_uci.png`, `results/figures/elasticnet_l1ratio_sweep.png` | Implemented | Sparsity and stability are both addressed |
| Analyze sparsity and stability | `results/tables/table2_sparsity.csv`, `results/figures/coef_stability_l1.png`, `results/figures/coef_stability_l2.png` | Implemented | This is a strong section for theory discussion |
| Handle class imbalance using at least two techniques | `src/imbalance.py`, `run_all.py` | Implemented in code; checked-in outputs incomplete | The methods exist, but generated result files for this section are not currently present in `results/` |
| Comparative analysis answering the four main questions | `src/comparative_analysis.py`, `run_all.py`, existing result tables | Partially evidenced | Baseline and regularization comparisons exist; imbalance tables, hypothesis tests, and holdout outputs are referenced in code but not checked in |
| Report with tables and graphs | `results/tables/`, `results/figures/` | Mostly implemented | Enough material exists for a professional draft report, but a complete final submission should include the missing downstream outputs |
| Code should be well documented | Source files in `src/` and `run_all.py` | Implemented | The codebase uses readable module separation and helpful docstrings |
| Presentation | Not included as a slide deck in this repo | Pending | Use the outline below to build the final presentation quickly |

## 2. What the Repository Currently Proves

The current checkout proves that the project is more than a proposal. It already includes:

- three local datasets,
- an executable experiment driver in `run_all.py`,
- modular source code for preprocessing, modeling, evaluation, imbalance handling, and comparative analysis,
- baseline experiment tables,
- regularization study tables,
- figures for learning behavior, regularization, Elastic Net sweeps, and coefficient stability.

It does not yet prove, from checked-in result files alone, that the full imbalance analysis, hypothesis-testing outputs, and holdout evaluation have already been generated in this specific checkout.

That distinction is worth preserving in the report:

- "Implemented in code" is true for those sections.
- "Included as checked-in result artifacts" is not yet true for all sections.

## 3. Requirement-by-Requirement Academic Framing

### 3.1 Dataset collection

Use the following justification in the report:

- UCI Seizure contributes a larger, moderately imbalanced seizure benchmark with 178 input features.
- Bonn EEG contributes a much smaller dataset with a different signal profile and supports both engineered-feature and raw-signal analysis.
- EEG Eye State adds an independent EEG binary classification task to test whether preprocessing and regularization behavior generalizes beyond a seizure-only setting.

This framing is academically stronger than simply listing datasets because it shows why each one is needed.

### 3.2 Preprocessing pipelines

The repo supports a clean conceptual comparison:

- Pipeline A tests supervised feature filtering.
- Pipeline B tests unsupervised variance-preserving compression.
- Pipeline C tests whether explicit wavelet denoising improves performance when raw sequential signal structure is available.

Important nuance for the report:

Pipeline C should be described as a dataset-specific signal-processing extension, not as a universally applicable tabular preprocessing step. The implementation already reflects this by limiting Pipeline C to Bonn EEG.

### 3.3 Baseline model

The baseline is appropriate for the assignment because logistic regression:

- is easy to interpret,
- exposes coefficient shrinkage clearly,
- makes underfitting and overfitting easier to explain,
- supports direct comparison across L1, L2, and Elastic Net.

### 3.4 Overfitting and underfitting

This section is one of the strongest parts of the repository because it includes:

- a deliberately underfit configuration,
- a deliberately high-variance polynomial-feature configuration,
- learning curves,
- train-vs-validation gap analysis.

In the report, tie these visuals directly to bias-variance tradeoff rather than presenting them as isolated graphs.

### 3.5 Regularization study

This section should emphasize three ideas:

1. L1 creates sparsity and can behave like embedded feature selection.
2. L2 usually stabilizes coefficients without forcing exact zeros.
3. Elastic Net balances shrinkage and sparsity and is valuable when pure L1 or pure L2 is not ideal.

The checked-in tables and plots already support this discussion well.

### 3.6 Class imbalance

The code implements:

- SMOTE,
- random undersampling,
- class weighting.

For the final report, include an honest line such as:

> The repository includes a complete implementation for imbalance handling. In the current checked-in artifact set, the imbalance result tables and curves are not yet present and should be regenerated before final submission.

That keeps the documentation credible.

### 3.7 Comparative analysis

The four required questions from the assignment should be answered explicitly in the final report:

1. Does preprocessing order affect results?
2. Which regularization method generalizes best across datasets?
3. Does Elastic Net consistently outperform L1 and L2?
4. How does imbalance handling interact with regularization?

The report should avoid vague conclusions. Even if the answer is mixed, say so directly. For example:

- preprocessing effects may depend on the dataset rather than showing a single universal winner,
- Elastic Net may be competitive without being consistently dominant,
- imbalance strategies may help recall while reducing precision.

## 4. Current Empirical Talking Points

These points are grounded in the checked-in result tables and are safe to use in documentation right now:

- On the current baseline results, Pipeline B is the best baseline on UCI Seizure by PR-AUC, while Pipeline A is best on Bonn EEG and EEG Eye State.
- Bonn EEG currently achieves the strongest overall baseline performance, suggesting the classification boundary is easier for logistic regression under the current feature representation.
- UCI Seizure remains the most challenging dataset in practical terms because recall is low even when precision is very high.
- Bonn Pipeline C does not currently outperform the simpler alternatives, which makes it useful for discussion about whether denoising adds complexity without enough gain.
- The sparsity table shows that L1 meaningfully reduces the number of active coefficients at lower `C` values, directly supporting the regularization theory section.

## 5. Suggested IEEE-Style Report Structure

Use this as the writing sequence for the final report.

### Title

Prefer a title like:

> Preprocessing, Regularization, and Generalization in EEG Classification Using Logistic Regression

### Abstract

State:

- the problem,
- the three datasets,
- the preprocessing pipelines,
- the regularization comparison,
- the main conclusion about generalization.

### 1. Introduction

Cover:

- why seizure and EEG classification matter,
- why generalization is harder than raw accuracy,
- why preprocessing and regularization deserve focused study.

### 2. Related Theory

Keep this short and focused on:

- logistic regression,
- regularization,
- class imbalance,
- bias-variance tradeoff,
- dimensionality reduction and feature selection.

### 3. Datasets

Use a comparison table summarizing:

- source,
- sample count,
- feature count,
- class balance,
- why each dataset was selected.

### 4. Methodology

Include:

- preprocessing pipelines,
- train/test split policy,
- cross-validation policy,
- metrics,
- regularization grid,
- imbalance methods.

### 5. Experiments

Break the section into:

- baseline evaluation,
- underfitting and overfitting,
- regularization study,
- class imbalance study,
- comparative analysis.

### 6. Results and Discussion

This is where the report becomes strong. Do not only restate numbers. Explain:

- why a pipeline worked,
- when sparsity helped,
- where recall was weak,
- which conclusions appear dataset-specific,
- what that means for practical EEG modeling.

### 7. Limitations

Good limitations to include:

- EEG Eye State is an EEG classification dataset but not a seizure dataset,
- not all datasets share identical signal structure,
- logistic regression is intentionally simple and may under-represent nonlinear structure,
- imbalance outputs should be regenerated if the final artifact set is incomplete.

### 8. Conclusion

Summarize what the project teaches about:

- preprocessing order,
- regularization behavior,
- generalization under imbalance,
- interpretability of simple models.

## 6. Suggested Presentation Outline

An 8-10 slide structure is enough.

1. Problem statement and motivation
2. Research question and hypotheses
3. Dataset overview
4. Preprocessing pipelines
5. Baseline model and metrics
6. Overfitting vs underfitting results
7. Regularization and sparsity findings
8. Comparative analysis across datasets
9. Limitations and future work
10. Final takeaways

## 7. Final Submission Checklist

Before submission, make sure the repo or report package includes:

- final report PDF,
- updated result tables for all completed sections,
- all figures referenced in the report,
- clear explanation of which outputs are baseline versus final,
- presentation slides,
- the exact environment and command used to reproduce results.

If you want this repository to look polished for grading, the safest path is:

1. keep the README as the professional project front page,
2. use this document as the report-writing guide,
3. regenerate any missing output files before packaging the final submission.