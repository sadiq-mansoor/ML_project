# Preprocessing, Regularization, and Generalization in EEG Classification Using Logistic Regression

**Sadiq Mansoor**  
Semester 8, Department of Computer Science  

---

## Abstract

This project investigates how preprocessing design, regularization strength, and class imbalance handling affect generalization in EEG classification tasks. A logistic regression framework was selected as the core model so that changes in performance remain interpretable and directly attributable to preprocessing and regularization rather than hidden inside a complex nonlinear architecture. Three datasets were used: UCI Epileptic Seizure Recognition, Bonn EEG, and UCI EEG Eye State. Three preprocessing strategies were studied: supervised feature selection, unsupervised dimensionality reduction, and wavelet-based denoising combined with dimensionality reduction. The study further examined underfitting, overfitting, L1, L2, and Elastic Net regularization, as well as SMOTE, undersampling, and class weighting for imbalance mitigation. Results show that preprocessing effects are dataset-dependent, that simpler pipelines often outperform added denoising complexity, and that strong gains from regularization are not universal across all datasets. The seizure dataset remained the most challenging in recall, while Bonn EEG produced the strongest overall classification performance.

## Keywords

EEG classification, seizure prediction, logistic regression, regularization, preprocessing, class imbalance, generalization

## I. Introduction

Electroencephalography-based classification problems are important in biomedical machine learning because predictive performance alone is not sufficient; models must also generalize across different data conditions and class distributions. In seizure-related tasks, false negatives are particularly costly because missed seizures reduce clinical usefulness. For this reason, the project focuses not only on accuracy, but also on precision-recall behavior, generalization, and bias-variance tradeoffs.

The central question of this work is: how do preprocessing choices and regularization jointly affect generalization in EEG classification tasks? This question directly aligns with the semester assignment requirements and motivates a controlled study where logistic regression is kept fixed while preprocessing, regularization, and imbalance treatment are varied systematically.

## II. Datasets

Three EEG-related datasets were used to provide variation in size, representation, and difficulty.

### A. UCI Epileptic Seizure Recognition

This dataset contains 11,500 samples with 178 input features. The original multiclass target was converted into a binary task where seizure class `1` was treated as the positive class and the remaining classes were grouped as non-seizure. The dataset is moderately imbalanced and serves as the main seizure benchmark in the study.

### B. Bonn EEG

The Bonn dataset contains 500 EEG segments. For Pipelines A and B, six statistical features were extracted from each signal: mean, standard deviation, skewness, kurtosis, energy, and zero-crossing rate. For Pipeline C, the raw 4097-point signals were preserved to enable wavelet denoising. Positive samples were defined from sets `F` and `S`, while sets `Z`, `O`, and `N` formed the negative class.

### C. UCI EEG Eye State

This dataset contains 14,980 instances with 14 continuous EEG-derived attributes. Although it is not a seizure dataset, it provides an independent EEG binary classification task and is useful for testing whether preprocessing and regularization conclusions generalize beyond seizure-only data.

## III. Methodology

### A. Preprocessing Pipelines

Three pipelines were implemented.

1. Pipeline A: `StandardScaler -> Mutual Information SelectKBest`
2. Pipeline B: `StandardScaler -> PCA (95% explained variance)`
3. Pipeline C: `Wavelet denoising -> StandardScaler -> PCA (95% explained variance)`

Pipeline C was applied only to Bonn EEG because wavelet denoising assumes temporally meaningful sequential signal structure. Applying it to pre-extracted tabular representations would not be theoretically appropriate.

### B. Baseline and Regularized Models

Logistic regression was used throughout the project. The baseline model employed effectively no regularization. Underfitting was induced using very strong L2 regularization with limited features, while overfitting was induced using degree-2 polynomial feature expansion with minimal effective regularization.

Regularization experiments compared:

1. L1 regularization
2. L2 regularization
3. Elastic Net regularization

The hyperparameter search used `C = [0.01, 0.1, 1, 10]` and Elastic Net `l1_ratio = [0.3, 0.5, 0.7]`.

### C. Imbalance Handling

Three imbalance strategies were implemented:

1. SMOTE
2. Random undersampling
3. Class weighting

These methods were evaluated within the training process to study how class-balance correction interacts with regularization.

### D. Evaluation Protocol

All experiments used stratified 5-fold cross-validation with a fixed random state of `42`. Performance metrics included PR-AUC, F1-score, recall, precision, and accuracy. Additional tracking metrics included coefficient sparsity, feature count after preprocessing, training time, and coefficient stability across folds. A holdout evaluation was also produced for final model inspection.

## IV. Experimental Results

### A. Baseline Pipeline Comparison

Table I summarizes the strongest baseline pipeline for each dataset using the checked-in results.

**Table I**  
Best baseline pipeline per dataset

| Dataset | Best Pipeline | PR-AUC | F1-score | Recall |
| --- | --- | ---: | ---: | ---: |
| UCI Seizure | B | 0.480 | 0.105 | 0.055 |
| Bonn EEG | A | 0.882 | 0.755 | 0.688 |
| EEG Eye State | A | 0.618 | 0.546 | 0.488 |

These results show that preprocessing order matters, but not uniformly. Pipeline B performed best on UCI Seizure, while Pipeline A performed best on Bonn EEG and EEG Eye State. This suggests that the value of supervised feature selection versus PCA depends on dataset structure.

### B. Underfitting and Overfitting

The repository includes learning curves for baseline, underfitting, and overfitting configurations, along with a train-versus-validation gap plot. The hypothesis test comparing the baseline and polynomial overfitting configuration produced a very small p-value (`3.15e-06`), which strongly supports the claim that the polynomial feature expansion altered generalization behavior substantially.

This section illustrates the bias-variance tradeoff clearly:

1. Very strong regularization restricts model flexibility and induces underfitting.
2. High-dimensional polynomial expansion without effective regularization increases variance and widens the training-validation gap.

### C. Regularization Study

The regularization sweep produced several important observations.

1. On Bonn EEG, the best checked-in result was obtained by L1 regularization with `C = 1.0`, achieving PR-AUC `0.884`.
2. On EEG Eye State, the best checked-in result came from L2 regularization with `C = 10.0`, achieving PR-AUC `0.610`.
3. On UCI Seizure, the best scores were close across L1, L2, and Elastic Net, showing no dominant regularizer.

The hypothesis test outputs reinforce this interpretation. For the regularization comparisons already generated in the repository, pairwise p-values were not statistically significant for the selected comparison sets. This indicates that although one method may numerically appear best on a dataset, the differences are not consistently large enough to claim universal superiority.

### D. Sparsity and Stability

The sparsity table shows that L1 regularization reduces the number of active coefficients, especially at smaller `C` values. For example, on UCI Seizure Pipeline A, L1 with `C = 0.01` retained only `5` nonzero coefficients out of `20`, while L2 retained all `20`. This directly supports the theoretical role of L1 as an embedded feature-selection mechanism.

Coefficient stability figures were also generated for L1 and L2, allowing comparison of how shrinkage affects fold-to-fold variation. In general, L2 offers a stable dense solution, while L1 trades stability for sparsity.

### E. Class Imbalance Handling

Imbalance handling results were generated for UCI Seizure and EEG Eye State. The best PR-AUC values in the checked-in table are summarized below.

**Table II**  
Best imbalance result by dataset

| Dataset | Best Method | Regularization | PR-AUC |
| --- | --- | --- | ---: |
| UCI Seizure | Undersample | L1 | 0.466 |
| EEG Eye State | ClassWeight | L2 | 0.603 |

Although numerical differences exist, the hypothesis tests for imbalance strategies did not show statistically significant pairwise differences for the selected UCI comparisons. This suggests that imbalance handling changes performance modestly, but the effect is not strong enough in the current runs to support a broad claim that one method always dominates.

### F. Holdout Evaluation

Final holdout results are listed below.

**Table III**  
Holdout evaluation

| Dataset | PR-AUC | F1-score | Recall | Precision | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI Seizure | 0.452 | 0.043 | 0.022 | 1.000 | 0.804 |
| Bonn EEG | 0.811 | 0.618 | 0.525 | 0.750 | 0.740 |
| EEG Eye State | 0.593 | 0.442 | 0.367 | 0.558 | 0.585 |

These results reinforce the same trend seen in cross-validation: Bonn EEG is the easiest of the three datasets under the current feature representations, while UCI Seizure remains difficult, especially in recall.

## V. Discussion

The most important conclusion is that preprocessing and regularization cannot be evaluated in isolation. Their effectiveness depends on the structure of the underlying dataset.

For UCI Seizure, the major difficulty is not precision but recall. The model can be conservative and still appear accurate, but such behavior is undesirable in a seizure-related setting where missed positives are costly. This makes PR-AUC and recall more informative than accuracy alone.

For Bonn EEG, simple engineered features paired with linear models perform surprisingly well. In the current results, the extra complexity of wavelet denoising in Pipeline C does not outperform the simpler alternatives. This is an important negative result: more preprocessing does not automatically imply better generalization.

For EEG Eye State, Pipeline A and L2 regularization produced the most reliable results. This suggests that even in a non-seizure EEG task, careful preprocessing and moderate regularization can yield stable generalization without requiring a more complex classifier.

## VI. Comparative Analysis

The assignment asked four main comparative questions, which can be answered as follows based on the checked-in outputs.

### A. Does preprocessing order affect results?

Yes, but the effect is dataset-dependent. On EEG Eye State, Pipeline A significantly outperformed Pipeline B in the hypothesis test. On UCI Seizure, the difference between Pipelines A and B was not statistically significant in the current runs.

### B. Which regularization generalizes best across datasets?

No single regularization method emerged as a universal winner. L1 performed strongly on Bonn EEG, L2 performed best on EEG Eye State, and differences on UCI Seizure were small.

### C. Does Elastic Net consistently outperform L1 and L2?

No. Elastic Net was competitive in several settings, but the current hypothesis test results do not support a claim of consistent superiority over pure L1 or pure L2.

### D. How does imbalance handling interact with regularization?

The interaction exists numerically, but the currently checked-in hypothesis tests do not show strong statistical evidence that one imbalance-handling method consistently pairs best with a given regularization type.

## VII. Limitations

This study has several limitations.

1. EEG Eye State is an EEG classification dataset, but it is not a seizure dataset.
2. The datasets differ in representation, which means some pipelines are only valid for certain datasets.
3. Logistic regression was intentionally chosen for interpretability, but this limits nonlinear modeling capacity.
4. Some conclusions are based on modest performance differences that are not always statistically significant.

## VIII. Conclusion

This project demonstrated that generalization in EEG classification depends jointly on preprocessing, regularization, and class imbalance handling. Simpler linear models remain valuable when the study goal is interpretability and controlled comparison rather than raw benchmark maximization. The experiments show that no single preprocessing or regularization strategy dominates across all datasets. Instead, the most reliable conclusion is that the best bias-variance tradeoff is dataset-specific, and that careful experimental design is essential when evaluating EEG classification systems for real-world use.

## References

[1] Semester project specification, `Project_requirements.md`, local course repository copy.  
[2] UCI Machine Learning Repository, "Epileptic Seizure Recognition," referenced in `implementation_plan.md`.  
[3] Department of Epileptology, University of Bonn, Bonn EEG dataset, referenced in `implementation_plan.md`.  
[4] UCI Machine Learning Repository, "EEG Eye State," referenced in `implementation_plan.md`.  
[5] Project implementation source files: `src/data_loader.py`, `src/preprocessing.py`, `src/models.py`, `src/evaluation.py`, `src/imbalance.py`, and `run_all.py`.
