# EEG-Based State Classification
## Preprocessing, Regularization & Generalization Study

**Author:** Sadiq Mansoor  
**Course:** Machine Learning Theory — Semester 8  
**Submission:** Major Assignment

---

## Research Question

> *How do preprocessing choices and regularization jointly affect generalization in imbalanced EEG classification tasks?*

---

## Datasets

| Dataset | Samples | Features | Imbalance | Source |
|---------|---------|----------|-----------|--------|
| UCI Epileptic Seizure Recognition | 11,500 | 178 | 1:4 (seizure minority) | UCI ML Repository |
| Bonn University EEG | 500 | 4,097 (raw) / 6 (features) | Mild | University of Bonn Epileptology Dept |
| UCI EEG Eye State | 14,980 | 14 | ~45:55 | UCI ML Repository |

---

## Project Structure

```
├── run_all.py                  ← Master script — runs all experiments end-to-end
├── requirements.txt            ← Python dependencies
├── implementation_plan.md      ← Full research design and methodology
├── Project_requirements.md     ← Assignment brief
│
├── src/
│   ├── data_loader.py          ← Dataset loading with auto-download
│   ├── preprocessing.py        ← Pipelines A, B, C
│   ├── models.py               ← Logistic regression variants
│   ├── evaluation.py           ← Metrics, CV, all plot functions
│   ├── imbalance.py            ← SMOTE, undersampling, class weighting
│   ├── comparative_analysis.py ← Tables and hypothesis tests
│   └── utils.py                ← Timer, logger, random state
│
├── data/
│   ├── uci_seizure/            ← data.csv (auto-downloaded)
│   ├── bonn_eeg/               ← Z/, O/, N/, F/, S/ (500 .txt files)
│   └── eeg_eye_state/          ← EEG_Eye_State.arff
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipelines.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_overfitting_underfitting.ipynb
│   ├── 05_regularization_study.ipynb
│   ├── 06_class_imbalance.ipynb
│   ├── 07_comparative_analysis.ipynb
│   └── Master_Analysis.ipynb
│
└── results/
    ├── figures/                ← All generated plots (19 files)
    └── tables/                 ← All result CSVs (10 files)
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run all experiments
```bash
python run_all.py
```

This single command runs all 7 sections end-to-end and generates all tables and figures in `results/`.  
Expected runtime: **~25–30 minutes** on a standard laptop.

---

## Preprocessing Pipelines

| Pipeline | Steps | Datasets |
|----------|-------|---------|
| A | StandardScaler → MI Feature Selection (top-k) | All |
| B | StandardScaler → PCA (95% variance) | All |
| C | Wavelet Denoising (db4) → StandardScaler → PCA | Bonn EEG only |

---

## Experiments & Outputs

### Tables (`results/tables/`)

| File | Content |
|------|---------|
| `table1_pipeline_dataset.csv` | Baseline PR-AUC / F1 / Recall per pipeline × dataset |
| `table2_sparsity.csv` | Non-zero coefficients, feature count, training time |
| `table3_regularization.csv` | L1 / L2 / ElasticNet × C values × dataset |
| `table4_imbalance_interaction.csv` | SMOTE / Undersample / ClassWeight × regularization |
| `h1_pipeline_ttest.csv` | H1: Does preprocessing order affect results? |
| `h2_regularization_ttest.csv` | H2: Which regularization generalizes best? |
| `h3_elasticnet_ttest.csv` | H3: Does ElasticNet outperform L1/L2? |
| `h4_imbalance_ttest.csv` | H4: Imbalance × regularization interaction |
| `h5_polynomial_ttest.csv` | H5: Polynomial features cause overfitting? |
| `holdout_final_evaluation.csv` | Final test set evaluation (never seen during training) |

### Figures (`results/figures/`)

| File | Section |
|------|---------|
| `learning_curve_baseline.png` | §3 Baseline |
| `learning_curve_underfit.png` | §4 Underfitting |
| `learning_curve_overfit.png` | §4 Overfitting |
| `train_val_gap_C.png` | §4 Bias-variance tradeoff curve |
| `reg_curve_uci.png` | §5 Regularization strength vs PR-AUC |
| `elasticnet_l1ratio_sweep.png` | §5 ElasticNet l1_ratio sweep |
| `coef_stability_l1.png` | §5 L1 coefficient stability across folds |
| `coef_stability_l2.png` | §5 L2 coefficient stability across folds |
| `pr_curves_imbalance_uci_seizure.png` | §6 Precision-Recall curves (UCI) |
| `pr_curves_imbalance_eeg_eye_state.png` | §6 Precision-Recall curves (Eye State) |

---

## Research Hypotheses

| # | Hypothesis | Result |
|---|-----------|--------|
| H1 | Preprocessing order has no effect on PR-AUC | See `h1_pipeline_ttest.csv` |
| H2 | L1/L2/ElasticNet produce equivalent generalization | See `h2_regularization_ttest.csv` |
| H3 | ElasticNet does not differ from pure L1/L2 | See `h3_elasticnet_ttest.csv` |
| H4 | Imbalance technique does not interact with regularization | See `h4_imbalance_ttest.csv` |
| H5 | Polynomial features do not cause overfitting | **Rejected** — p = 0.000001 |

---

## Key Results

| Dataset | Holdout PR-AUC | Holdout F1 |
|---------|---------------|-----------|
| Bonn EEG | 0.811 | 0.618 |
| EEG Eye State | 0.593 | 0.442 |
| UCI Seizure | 0.452 | 0.022 |

Best configuration: **Pipeline A + L2 regularization (C=1.0)**

---

## Theoretical Framework

All experiments are grounded in the bias-variance decomposition:

```
Total Error = Bias² + Variance + Irreducible Noise
```

- **Strong regularization (small C)** → high bias, low variance → underfitting  
- **No regularization + polynomial features** → low bias, high variance → overfitting  
- **L1** → sparse model, higher bias, lower variance  
- **L2** → dense model, balanced bias-variance  
- **ElasticNet** → combines L1 sparsity and L2 stability  
- **SMOTE** → reduces bias on minority class, may increase variance  

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
pywt>=1.4
jupyter>=1.0
```
