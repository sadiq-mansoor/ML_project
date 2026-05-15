## Semester Major Assignment

I will investigate how students' preprocessing choices, model complexity, and regularisation strategies affect generalisation performance in seizure prediction tasks.

### 1. Dataset Collection

* Select at least 3 epileptic seizure datasets (e.g., EEG-based datasets from Kaggle, UCI, CHB-MIT, etc.)
* Provide justification:
  * Size
  * Class imbalance
  * Feature characteristics (time-series vs extracted features)

### 2. Preprocessing Pipeline (CRITICAL PART)

Students must design at least 2 different preprocessing pipelines, for example:

* Pipeline A:
  * Normalization → Noise removal → Feature selection
* Pipeline B:
  * Feature extraction → Scaling → PCA

This is where you can link to your own research insight:
ordering of preprocessing steps affects performance

### 3. Baseline Model: Logistic Regression

Use logistic regression as the core model:

P(y=1∣x)=11+e−(β0+βTx)**P**(**y**=**1∣**x**)**=**1**+**e**−**(**β**0****+**β**T**x**)**1****

Students should:

* Train baseline model
* Report metrics:
  * Accuracy
  * F1-score
  * PR-AUC (important for imbalance)

### 4. Demonstrate Overfitting & Underfitting

Students must intentionally create scenarios:

* Underfitting:
  * Very strong regularization
  * Limited features
* Overfitting:
  * No regularization
  * High-dimensional features

Require:

* Training vs validation curves
* Learning curve visualisation

### 5. Regularization Study

Students must compare the following:

* L1 (Lasso)
* L2 (Ridge)
* Elastic Net

J(W,b)=1m∑i=1mL(y^(i),y(i))+λ2m∑∥W∥2**J**(**W**,**b**)**=**m**1****i**=**1**∑**m****L**(**y**^****(**i**)**,**y**(**i**)**)**+**2**m**λ****∑**∥**W**∥**2

Key requirement:

* Analyze sparsity (feature selection effect)
* Compare stability across datasets

### 6.  Handling Class Imbalance

Students must apply at least two techniques:

* SMOTE / oversampling
* Undersampling
* Class weighting

Evaluate:

* Impact on precision vs recall tradeoff

### 7. Comparative Analysis (MOST IMPORTANT)

This is where weak students fail and strong ones shine.

Students must answer the following:

* Does preprocessing order affect results?
* Which regularisation generalises best across datasets?
* Does Elastic Net consistently outperform L1/L2?
* How does imbalance handling interact with regularisation?

---

### Instructions

* Report (IEEE format preferred)
* Results tables + graphs
* Code (well-documented)
* Presentation
