# ML Theory Project — Implementation Plan (v3 — Final)
## EEG-Based State Classification: Preprocessing, Regularization & Generalization Study

---

## Central Narrative

> **"How do preprocessing choices and regularization jointly affect generalization in imbalanced EEG classification tasks?"**

Every experiment in this project connects to this ONE question. Preprocessing controls what the model sees. Regularization controls how the model learns. Class imbalance determines how hard the task is. The interaction of all three determines whether the model generalizes.

This narrative ties together:
- **Pipelines** → what information reaches the model
- **Regularization** → how the model manages complexity (bias-variance tradeoff)
- **Imbalance handling** → whether the model can learn the minority class
- **Cross-dataset evaluation** → whether findings are robust or dataset-specific

---

## Revision History

**v3 changes (from reviewer feedback round 2):**
- ⚠️ Replaced CHB-MIT with independent EEG Eye State Dataset (avoids EDF/MNE time sink and Kaggle data overlap)
- ⚠️ Reduced C values: 7 → 4 (sufficient for clear trends)
- ⚠️ Reduced Elastic Net l1_ratio sweep: 5 → 3
- ⚠️ Simplified statistical testing (dropped Bonferroni — overkill for semester level)
- ✅ Added central narrative connecting all experiments
- ✅ Added bias-variance theoretical framework
- ✅ Added master summary table
- ✅ Added failure analysis section
- ✅ Better framing for Pipeline C (scientifically neutral)
- ✅ Added experiment count estimate (~540 runs — manageable)

**v2 changes (from reviewer feedback round 1):**
- Real independent datasets, fair pipelines, formal hypotheses, stratified CV, statistical testing, polynomial cap, data leakage prevention, tracking metrics

---

## Project Structure

```
ML Theory Project/
├── Project_requirements.md
├── implementation_plan.md
├── requirements.txt
├── data/
│   ├── uci_seizure/              # UCI Epileptic Seizure Recognition
│   ├── bonn_eeg/                 # Bonn University EEG Dataset
│   └── eeg_eye_state/            # UCI EEG Eye State Dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipelines.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_overfitting_underfitting.ipynb
│   ├── 05_regularization_study.ipynb
│   ├── 06_class_imbalance.ipynb
│   └── 07_comparative_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading & preparation
│   ├── preprocessing.py           # Pipeline A, B, C implementations
│   ├── models.py                  # Logistic regression variants
│   ├── evaluation.py              # Metrics, visualization, stat tests
│   ├── imbalance.py               # SMOTE, undersampling, class weighting
│   ├── comparative_analysis.py    # Cross-dataset/pipeline comparison
│   └── utils.py                   # Helpers, logging, timing
├── results/
│   ├── figures/                   # All generated plots
│   └── tables/                    # CSV result tables
└── run_all.py                     # Master script to run entire pipeline
```

---

## Research Hypotheses

Every experiment is tied to a testable hypothesis. This connects empirical results to the central narrative.

### H1 — Preprocessing Order
**H₀:** The ordering of preprocessing steps (scaling → feature selection vs. scaling → PCA vs. denoising → scaling → PCA) has no statistically significant effect on logistic regression classification performance.
**H₁:** At least one pipeline ordering produces significantly different PR-AUC scores.
**Narrative link:** Tests whether *what the model sees* matters more than *how it learns*.

### H2 — Regularization Generalization
**H₀:** L1, L2, and Elastic Net regularization produce equivalent generalization performance (F1) across all three datasets.
**H₁:** At least one regularization strategy generalizes significantly better across datasets.
**Narrative link:** Tests whether controlling model complexity (variance) transfers across data distributions.

### H3 — Elastic Net Performance Trade-offs
**H₀:** Elastic Net does not significantly differ in generalization performance (PR-AUC) from pure L1 or L2 regularization.
**H₁:** Elastic Net produces significantly different PR-AUC compared to both L1 and L2 across datasets.
**Narrative link:** Tests whether combining bias-inducing strategies (L1/L2) alters generalization behaviors.

### H4 — Imbalance × Regularization Interaction
**H₀:** The choice of class imbalance handling technique does not interact with the choice of regularization.
**H₁:** Certain imbalance techniques (e.g., SMOTE) perform significantly better with specific regularization methods (e.g., L1).
**Narrative link:** Tests whether preprocessing the *data distribution* interacts with how the model controls complexity.

### H5 — Overfitting via Polynomial Features
**H₀:** Adding degree-2 polynomial features to unregularized logistic regression does not cause measurable overfitting.
**H₁:** Polynomial features cause a significant gap between training and validation performance.
**Narrative link:** Directly demonstrates bias-variance tradeoff through increased model complexity.

---

## Theoretical Framework: Bias-Variance Decomposition

This framework unifies ALL experiments under one theory:

```
Total Error = Bias² + Variance + Irreducible Noise
```

### How Each Experiment Maps to Bias-Variance

| Experiment | Bias Effect | Variance Effect | Expected Outcome |
|------------|------------|-----------------|------------------|
| Strong L2 regularization (small C) | ↑ Increases bias | ↓ Reduces variance | Underfitting |
| No regularization (C → ∞) | ↓ Low bias | ↑ High variance | Overfitting |
| L1 regularization | ↑ Increases bias (zeroes features) | ↓ Reduces variance (sparse model) | Feature selection effect |
| PCA preprocessing | Slight ↑ bias (info loss) | ↓ Reduces variance (fewer dims) | Smoother generalization |
| Feature selection (MI) | Depends on k | ↓ If good features selected | Task-dependent |
| Polynomial features (degree 2) | ↓ Reduces bias (more expressive) | ↑ Increases variance (16k features) | Overfitting without regularization |
| SMOTE | ↓ Reduces bias on minority class | ↑ May increase variance (synthetic noise) | Better recall, possibly worse precision |

### Key Insight to Discuss in Report
> Regularization and preprocessing are BOTH forms of variance control. L2 shrinks coefficients (reducing variance directly), PCA reduces input dimensionality (reducing variance indirectly), and feature selection removes noisy inputs (reducing irreducible noise). The question is: **which combination finds the best bias-variance sweet spot for seizure prediction?**

---

## Section 1: Dataset Collection — REAL Independent Datasets

### 1.1 UCI Epileptic Seizure Recognition
| Property | Value |
|----------|-------|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition) |
| Samples | 11,500 |
| Features | 178 (raw time-series EEG values) |
| Classes | 5 → binary: seizure=1 vs non-seizure=2,3,4,5 |
| Imbalance | ~1:4 (seizure is minority at 20%) |
| Feature Type | Raw time-domain EEG amplitudes |
| Format | CSV — clean, ready to use |

### 1.2 Bonn University EEG Dataset
| Property | Value |
|----------|-------|
| Source | [Bonn University Epileptology Dept](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3) |
| Samples | 500 segments (5 sets × 100) |
| Features | 4097 per segment → extract statistical features (mean, std, skew, kurtosis, energy, etc.) |
| Classes | Binary: sets D,E = ictal/pre-ictal vs sets A,B,C = healthy/inter-ictal |
| Imbalance | Balanced or mildly imbalanced depending on grouping |
| Feature Type | Intracranial + scalp EEG, higher SNR |
| Format | Text files — straightforward parsing |

### 1.3 UCI EEG Eye State Dataset
| Property | Value |
|----------|-------|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/264/eeg+eye+state) |
| Samples | 14,980 |
| Features | 14 (continuous EEG measurements from 14 electrodes) |
| Classes | Binary: eye open (1) vs eye closed (0) |
| Imbalance | ~45:55 (Relatively balanced) |
| Feature Type | Continuous raw EEG values |
| Format | ARFF / CSV — clean, ready to load |

**Alternative Task Generalization:** While not strictly "seizure prediction", this dataset tests if the preprocessing and regularization pipelines generalize robustly to other EEG-based binary classification tasks.

### Why NOT CHB-MIT
| Issue | Impact |
|-------|--------|
| EDF file parsing | Requires `mne` library, complex setup |
| Channel inconsistencies | Different patients have different channel montages |
| Seizure annotation parsing | Manual window extraction needed |
| Patient leakage risk | Must split by patient, not by window |
| Computational overhead | Raw EEG files are large |
| **Time risk** | Could consume 80% of project time on data engineering instead of ML analysis |

The project's focus is **preprocessing, regularization, and generalization** — not biomedical EEG engineering. Clean datasets let us focus on the actual ML questions.

### Dataset Justification Table
| Criterion | UCI Seizure | Bonn EEG | EEG Eye State |
|-----------|-------------|----------|---------------|
| Size | Large (11.5k) | Small (500) | Large (15k) |
| Imbalance Ratio | Moderate (1:4) | Mild/Balanced | Balanced (45:55) |
| Feature Type | Raw time-series | Statistical from intracranial | Raw EEG channels |
| Ease of Use | ✓ CSV | ✓ Text files | ✓ ARFF / CSV |
| **Why chosen** | Benchmark, well-studied | Different modality, small-data | Truly independent, task generalization |

All datasets are **independently sourced** and provide variety in size, feature type, and class balance — enabling meaningful cross-dataset comparison.

---

## Section 2: Preprocessing Pipelines — Fair & Interpretable Comparison

### Dataset Compatibility & Adaptation
> **Important Logical Note:** Datasets do not all have the exact same structure (e.g., Bonn has sequential signal data, UCI has tabular segments). Therefore:
> **"Pipelines were adapted minimally to preserve compatibility with dataset feature structure while maintaining the same conceptual preprocessing framework."**

### Data Leakage Prevention Protocol
- All transformations (scaling, PCA, feature selection) are **fit ONLY on training folds**
- Holdout test set is **never touched** during any preprocessing fitting
- Use `sklearn.pipeline.Pipeline` to enforce this automatically
- SMOTE is applied **ONLY inside the training fold** (never on validation/test)

### Pipeline A — Feature Selection Path
```
Raw Data → StandardScaler → Mutual Information Feature Selection (top-k) → Logistic Regression
```
**Rationale:** Supervised dimensionality reduction. Keeps features most relevant to seizure detection.

### Pipeline B — Unsupervised Dimensionality Reduction
```
Raw Data → StandardScaler → PCA (95% variance) → Logistic Regression
```
**Rationale:** Unsupervised dimensionality reduction. Tests if PCA captures seizure-relevant variance without label information.

### Pipeline C — Signal Denoising Investigation
```
Raw Data → Wavelet Denoising (db4, level=4) → StandardScaler → PCA (95% variance) → Logistic Regression
```
**Rationale:** Pipeline C investigates whether explicit signal denoising provides additional benefits beyond what dimensionality reduction (PCA) already achieves. 

**Note on Applicability:** Wavelet denoising mathematically assumes sequential signal structure. Therefore:
* **"Wavelet denoising (Pipeline C) comparisons are restricted strictly to datasets preserving temporally meaningful EEG signal structure (e.g., Bonn EEG)."**
* For datasets with pre-extracted, non-sequential features (e.g., UCI Seizure), Pipeline C is theoretically invalid and will be excluded from comparative evaluation.

### Why This Design is Fair
| Design Choice | Justification |
|---------------|---------------|
| Same model (LogReg) across all | Isolates preprocessing effect |
| Same scaler (StandardScaler) in all | Controls for scaling method |
| Incremental complexity (A → B → C) | Each pipeline adds exactly one variable |
| Same CV strategy across all | Fair evaluation |

### Pipeline Comparison Matrix
| Step | Pipeline A | Pipeline B | Pipeline C |
|------|-----------|-----------|-----------|
| Wavelet Denoising | ✗ | ✗ | ✓ |
| StandardScaler | ✓ | ✓ | ✓ |
| MI Feature Selection | ✓ | ✗ | ✗ |
| PCA (95% var) | ✗ | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ | ✓ |

---

## Section 3: Baseline Model — Logistic Regression

### Configuration
```python
from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression(
    penalty=None,          # No regularization for true baseline
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)
```

### Metrics (Primary)
| Metric | Why |
|--------|-----|
| **PR-AUC** | Best metric for imbalanced data — focuses on positive class |
| **F1-Score** | Balances precision and recall |
| **Recall** | Critical in seizure prediction (missing a seizure = dangerous) |

### Metrics (Secondary)
| Metric | Why |
|--------|-----|
| Accuracy | General overview, but misleading with imbalance |

### Additional Tracking Metrics
| Metric | Why |
|--------|-----|
| **Number of non-zero coefficients** | Measures sparsity (especially for L1) |
| **Training time (seconds)** | Computational cost comparison |
| **Feature count after preprocessing** | Quantifies dimensionality reduction effect |
| **Convergence iterations** | Model complexity indicator |

---

## Section 4: Overfitting & Underfitting Demonstration

### Underfitting Scenario (High Bias)
```python
underfit_model = LogisticRegression(
    penalty='l2',
    C=0.0001,            # Extreme regularization → high bias
    max_iter=5000,
    random_state=42
)
# Use only top-5 features from MI selection → further increases bias
```
**Bias-Variance Interpretation:** Strong regularization forces coefficients toward zero → model cannot capture true patterns → high bias, low variance → underfitting.

### Overfitting Scenario (High Variance)
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False)  # degree 2 ONLY
overfit_model = LogisticRegression(
    penalty=None,          # No regularization → nothing controls variance
    solver='lbfgs',
    max_iter=10000,
    random_state=42
)
```
**Bias-Variance Interpretation:** 178 features → ~16,000 polynomial features with zero regularization → model fits training noise → low bias, high variance → overfitting.

**Polynomial degree capped at 2.** Degree 3 with 178 features would create ~1M+ features — computationally infeasible and unnecessary for demonstrating the concept.

### Visualizations Required
1. **Training vs Validation Accuracy curves** (across regularization strength C)
2. **Learning curves** (performance vs training set size)
3. **Gap analysis** — plot `train_score - val_score` as a function of model complexity
4. **Bias-variance annotation** — label regions of each plot as "high bias" or "high variance"

---

## Section 5: Regularization Study

### Configurations (REDUCED SCOPE)
```python
regularization_configs = {
    'L1 (Lasso)': {'penalty': 'l1', 'solver': 'saga'},
    'L2 (Ridge)': {'penalty': 'l2', 'solver': 'lbfgs'},
    'Elastic Net': {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5}
}

# Hyperparameter grid — REDUCED from 7 to 4 values
C_values = [0.01, 0.1, 1, 10]  # Sufficient for clear trends

# Elastic Net l1_ratio — REDUCED from 5 to 3 values
l1_ratios = [0.3, 0.5, 0.7]  # Enough for academic insight
```

### Analysis Points
1. **Sparsity analysis:** Count non-zero coefficients per regularization type at each C value
2. **Stability analysis:** Coefficient variance across CV folds
3. **Cross-dataset generalization:** Does the best C value transfer across datasets?
4. **Elastic Net sweep:** Is there a sweet spot for l1_ratio?

### Experiment Count Estimate
```
Baseline (no reg): 3 datasets × 3 pipelines × 1 setting × 5 CV folds = 45 runs
L1 & L2 sweep: 3 datasets × 3 pipelines × 2 reg types × 4 C values × 5 folds = 360 runs
Elastic Net sweep: 3 datasets × 3 pipelines × 3 ratios × 4 C values × 5 folds = 540 runs
Imbalance (using best C): 3 datasets × 3 pipelines × 3 imbalance types × 3 reg types × 5 folds = 405 runs

Total: 1,350 training runs (manageable in hours, not days)
```

This is significantly reduced from the previous v2 estimate of 3,780+ runs.

---

## Section 6: Class Imbalance Handling

### Techniques
1. **SMOTE** — Synthetic Minority Oversampling (applied ONLY inside training folds)
2. **Random Undersampling** — Downsample majority class
3. **Class Weighting** — `class_weight='balanced'` in LogisticRegression

### Critical Protocol
```python
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE must be INSIDE the cross-validation loop
pipeline_with_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),   # Only applied to train fold
    ('model', LogisticRegression())
])
```

### Evaluation Focus
- Precision vs Recall tradeoff curves
- How each technique interacts with L1/L2/ElasticNet (→ H4)
- Which technique is most effective on the most imbalanced dataset

---

## Section 7: Comparative Analysis — Deep & Rigorous

### General Policies

#### Reproducibility & Random Seed Policy
To ensure complete reproducibility across all experiments, a strict global random seed policy is enforced:
```python
import numpy as np
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# Passed to all scikit-learn functions as random_state=RANDOM_STATE
```

#### Final Model Selection Rule
To provide a clear, unambiguous methodology for choosing the "best" model:
> **"Models were selected primarily based on mean cross-validated PR-AUC. F1-score and recall were used as secondary criteria."**
This ensures robustness against class imbalance across datasets.

### Validation Strategy

#### Stratified 5-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```
- Used for **all** experiments
- Preserves class distribution in each fold

#### Holdout Test Set
- **20% holdout** stratified split before any CV
- Used **only for final evaluation** — never during model selection
- Reports final unbiased performance estimates

### Statistical Reporting (Appropriate for Semester Level)

#### Format
All results reported as: **mean ± std** across 5 CV folds

#### Statistical Tests
- **Paired t-test** across fold scores for key comparisons
- Report **p-values** for main hypothesis tests (H1-H5)
- **95% confidence intervals** for primary metrics

**Note:** No Bonferroni correction — that level of multiple testing adjustment is overkill for a semester project with ~5 planned hypothesis tests. 
*Caveat on T-tests:* While paired t-tests are used for simplicity in this semester project, we formally acknowledge their statistical limitations in this context (e.g., violation of fold independence in standard CV) and treat them as an approximation of true significance.

### Comparative Analysis Tables

#### Table 1: Pipeline × Dataset (Primary Metrics)
| Pipeline | Dataset | PR-AUC (mean±std) | F1 (mean±std) | Recall (mean±std) | p-value vs best |
|----------|---------|-------------------|---------------|-------------------|-----------------|
| A | UCI | — | — | — | — |
| B | UCI | — | — | — | — |
| C | UCI | — | — | — | — |
| A | Bonn | — | — | — | — |
| B | Bonn | — | — | — | — |
| C | Bonn | — | — | — | — |
| A | Eye State | — | — | — | — |
| B | Eye State | — | — | — | — |
| C | Eye State | — | — | — | — |

#### Table 2: Pipeline × Dataset (Tracking Metrics)
| Pipeline | Dataset | Non-zero Coefs | Feature Count Post-Preproc | Training Time (s) |
|----------|---------|----------------|---------------------------|-------------------|
| A | UCI | — | — | — |
| B | UCI | — | — | — |
| C | UCI | — | — | — |
| ... | ... | ... | ... | ... |

#### Table 3: Regularization × Dataset
| Regularization | Best C | UCI F1 | Bonn F1 | Eye State F1 | Cross-dataset Std |
|----------------|--------|--------|---------|--------------|-------------------|
| None | — | — | — | — | — |
| L1 | — | — | — | — | — |
| L2 | — | — | — | — | — |
| Elastic Net | — | — | — | — | — |

#### Table 4: Imbalance × Regularization Interaction
| Imbalance Method | Regularization | UCI PR-AUC | Eye State PR-AUC | Δ from baseline |
|------------------|---------------|------------|------------------|-----------------|
| None | None | — | — | — |
| SMOTE | L1 | — | — | — |
| SMOTE | L2 | — | — | — |
| SMOTE | Elastic Net | — | — | — |
| Class Weight | L1 | — | — | — |
| Class Weight | L2 | — | — | — |
| Class Weight | Elastic Net | — | — | — |
| Undersampling | L1 | — | — | — |
| Undersampling | L2 | — | — | — |
| Undersampling | Elastic Net | — | — | — |

---

## Section 8: Master Summary & Failure Analysis (NEW)

### 8.1 Master Summary Table

This single table provides the final takeaway for the entire project:

| Method | Model Complexity | Sparsity | Generalization | Stability | Bias Effect | Variance Effect |
|--------|-----------------|----------|----------------|-----------|-------------|-----------------|
| No Regularization | High | None | Low (overfits) | Unstable | Low bias | High variance |
| L1 (Lasso) | Low | High | Medium | Less stable | Higher bias | Lower variance |
| L2 (Ridge) | Medium | None | High | Stable | Moderate bias | Moderate variance |
| Elastic Net | Medium | Medium | Highest (expected) | Stable | Balanced | Balanced |
| PCA preprocessing | Reduces dim | N/A | Improves | Improves | Slight ↑ | ↓ Reduces |
| SMOTE | N/A | N/A | Improves recall | May reduce | ↓ on minority | ↑ synthetic noise |

### 8.2 Failure Analysis (What Strong Submissions Include)

After running all experiments, dedicate a section to analyzing WHERE and WHY things failed:

#### Questions to Answer
1. **Which dataset was hardest?** Why? (size? imbalance? feature quality?)
2. **Where did models produce the most false positives?** What does that mean clinically?
3. **Where did models produce the most false negatives?** This is the dangerous failure mode — missed seizures.
4. **Did any pipeline completely fail on a dataset?** Why?
5. **Did SMOTE ever hurt performance?** (It can, when synthetic samples add noise)
6. **Did denoising (Pipeline C) ever hurt?** (Possible if signal information is removed)

#### Failure Analysis Table (to be filled after experiments)
| Failure Type | Dataset | Pipeline | Regularization | Explanation |
|-------------|---------|----------|----------------|-------------|
| Highest FP rate | — | — | — | — |
| Highest FN rate | — | — | — | — |
| Worst generalization gap | — | — | — | — |
| SMOTE hurt performance | — | — | — | — |
| Denoising hurt performance | — | — | — | — |

#### Why This Matters
> Failure analysis demonstrates **scientific maturity**. Any student can report accuracy numbers. Analyzing *why* things failed shows genuine understanding of the underlying ML theory. This connects directly to bias-variance: if a model fails on small data (Bonn), it likely has high variance. If it fails uniformly, it likely has high bias.

---

## Research Questions → Hypotheses → Sections Map

| Question | Hypothesis | Primary Section | Theory |
|----------|-----------|----------------|--------|
| Does preprocessing order affect results? | H1 | §2, §7 Table 1 | Variance reduction via dim. reduction |
| Which regularization generalizes best? | H2 | §5, §7 Table 3 | Bias-variance tradeoff |
| Does Elastic Net consistently outperform? | H3 | §5, §7 Table 3 | Combined regularization |
| Imbalance × regularization interaction? | H4 | §6, §7 Table 4 | Distribution shift + complexity control |
| Can polynomial features cause overfitting? | H5 | §4 | Variance explosion |
| Where and why do models fail? | — | §8 | Bias-variance diagnosis |

---

## Execution Checklist

- [ ] Download and verify all 3 datasets (UCI, Bonn, Eye State)
- [ ] Implement data loaders with consistent binary labeling
- [ ] Build Pipeline A, B, C using `sklearn.pipeline.Pipeline`
- [ ] Verify no data leakage (fit only on train folds)
- [ ] Run baseline LogReg with Stratified 5-Fold CV
- [ ] Generate overfitting/underfitting curves (polynomial degree=2 only)
- [ ] Run regularization sweep (L1, L2, ElasticNet × 4 C values)
- [ ] Run Elastic Net l1_ratio sweep (3 values only)
- [ ] Run imbalance experiments (SMOTE inside CV loop)
- [ ] Compute all tracking metrics (non-zero coefs, timing, feature counts)
- [ ] Run paired t-tests for H1-H5, report p-values + 95% CIs
- [ ] Build all comparison tables (Tables 1-4)
- [ ] Build master summary table
- [ ] Conduct failure analysis
- [ ] Write IEEE report with bias-variance narrative throughout
- [ ] Prepare presentation

---

## Dependencies

```
scikit-learn>=1.3
imbalanced-learn>=0.11
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
pywt>=1.4        # PyWavelets for wavelet denoising (Pipeline C only)
```

**Removed:** `mne>=1.4` — no longer needed since CHB-MIT was replaced with clean CSV datasets.
