# Project Task Division & Work Breakdown Structure

This document outlines the allocation of responsibilities, assigned modules, deliverables, and progress tracking for our 4-person machine learning project on the UCI German Credit Dataset.

---

## Task Allocation Overview

| Team Member | Difficulty / Role | Files Owned | Core Objectives | Key Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Rajeev** | **HARD**<br>(Repo Lead & Integrator) | `src/models.py`<br>`src/evaluation.py`<br>`notebooks/03_modeling_evaluation.ipynb`<br>`notebooks/04_imbalance_threshold.ipynb` | - Build leakage-free `sklearn`/`imblearn` Pipelines.<br>- Implement benchmark models (Logistic Regression, Random Forest, Gradient Boosting).<br>- Repeated Stratified CV setup.<br>- Financial expected-cost metric implementation.<br>- Decision threshold tuning to minimize total cost.<br>- Class weighting vs. SMOTE imbalance comparison.<br>- Unified results reporting table. | - Production-ready `models.py` & `evaluation.py`.<br>- Validated notebooks 03 & 04 with cost curves.<br>- PR code reviews & repository integration.<br>- Section 3 of final project report. |
| **Amlan Kumar Pradhan** | **MODERATE** | `src/preprocessing.py`<br>`src/features.py`<br>`notebooks/02_preprocessing_feature_selection.ipynb` | - Categorical encoding (One-Hot, Ordinal).<br>- Numerical scaling & log-transformation of skewed features (e.g. `credit_amount`).<br>- Assembly of modular `ColumnTransformer`.<br>- Feature selection algorithms (Mutual Information, L1-penalized Logistic Regression, Tree-based importances).<br>- Strict encapsulation of feature selectors inside CV folds. | - Production-ready `preprocessing.py` & `features.py`.<br>- Validated notebook 02 with transformer tests.<br>- Benchmarked feature subsets without leakage.<br>- Section 2 of final project report. |
| **Sohan Prusty** | **MODERATE** | `src/interpret.py`<br>`src/fairness.py`<br>`notebooks/05_interpretability.ipynb`<br>`notebooks/06_fairness.ipynb` | - Global feature importance using SHAP summary plots and Permutation Importance.<br>- Linear model coefficient extraction & log-odds interpretation.<br>- Local sample prediction explanations via waterfall/force plots.<br>- Fairness audits across protected attributes (`personal_status_sex`, `age`, `foreign_worker`).<br>- Compute demographic parity & group approval rates.<br>- Retrain without sensitive features & test proxy variables. | - Production-ready `interpret.py` & `fairness.py`.<br>- Validated notebooks 05 & 06 with SHAP & fairness audits.<br>- Disparate impact analysis and mitigation recommendations.<br>- Section 4 of final project report. |
| **Shubham Behri** | **EASY** | `src/data.py`<br>`notebooks/01_eda.ipynb`<br>`README.md`<br>`reports/final_report.md` | - Load raw space-separated `german.data` without headers.<br>- Map verified column names and schemas.<br>- Recode target: `BAD = 1` (positive), `GOOD = 0` (negative).<br>- Exploratory Data Analysis (distributions, correlations, category checks).<br>- Maintain README and documentation.<br>- Format and synthesize the consolidated final report. | - Clean, validated data loader in `src/data.py`.<br>- Thoroughly annotated `01_eda.ipynb` with written insights.<br>- Polished, coherent `final_report.md`.<br>- Section 1 of final project report. |

---

## Deliverables & Status Checklist

### 1. Shubham Behri (Data Pipeline & EDA)
- [ ] Implement `src/data.py` (`COLUMN_NAMES`, `load_data()`) with raw space-separated ingestion
- [ ] Recode labels to binary integer format (`BAD=1`, `GOOD=0`)
- [ ] Verify category sanity (confirm `A65` and `A124` handled as categories, not nulls)
- [ ] Complete `notebooks/01_eda.ipynb` with distribution plots and written domain interpretations
- [ ] Maintain `README.md` and project instructions
- [ ] Assemble and format Section 1 and overall layout in `reports/final_report.md`

### 2. Amlan Kumar Pradhan (Preprocessing & Feature Selection)
- [ ] Implement `build_preprocessor()` in `src/preprocessing.py` with `ColumnTransformer`
- [ ] Implement log-transformation for skewed `credit_amount`
- [ ] Implement one-hot and ordinal encoders for qualitative attributes
- [ ] Support sensitive attribute exclusion toggle (`drop_sensitive=True`)
- [ ] Implement `select_features()` in `src/features.py` (Mutual Info, L1 Logistic, Tree-based)
- [ ] Ensure feature selection steps are wrapped as `sklearn` transformers to run inside CV
- [ ] Complete `notebooks/02_preprocessing_feature_selection.ipynb` with written interpretations
- [ ] Write Section 2 of `reports/final_report.md`

### 3. Rajeev (Modeling, Integration & Evaluation)
- [ ] Implement `expected_cost()` in `src/evaluation.py` (Cost Matrix: FN=5, FP=1)
- [ ] Implement `cross_validate_model()` using Repeated Stratified K-Fold CV
- [ ] Implement `find_best_threshold()` to tune classification threshold against expected cost
- [ ] Implement `build_pipeline()` in `src/models.py` integrating preprocessor, sampler, and classifier
- [ ] Benchmark Logistic Regression, Random Forest, and Gradient Boosting in `03_modeling_evaluation.ipynb`
- [ ] Evaluate `class_weight='balanced'` vs. SMOTE resampling in `04_imbalance_threshold.ipynb`
- [ ] Generate comparative metric and expected-cost performance tables
- [ ] Review all team PRs into `main`
- [ ] Write Section 3 of `reports/final_report.md`

### 4. Sohan Prusty (Model Interpretability & Fairness)
- [ ] Implement `shap_summary()` in `src/interpret.py` (TreeSHAP & LinearExplainer)
- [ ] Implement `permutation_importance_report()` in `src/interpret.py`
- [ ] Generate global and local prediction explanations in `notebooks/05_interpretability.ipynb`
- [ ] Implement `approval_rate_by_group()` in `src/fairness.py`
- [ ] Implement `compare_with_without_sensitive()` across protected groups (sex, age, foreign worker)
- [ ] Identify and audit proxy variables correlated with protected attributes in `notebooks/06_fairness.ipynb`
- [ ] Propose fairness mitigation strategies and evaluate cost vs. equity tradeoffs
- [ ] Write Section 4 of `reports/final_report.md`

### 5. Shared Team Milestones
- [x] Initial project repository scaffold and directory tree
- [ ] Data placed in `data/raw/` and validated
- [ ] Core modules implemented and reviewed via Pull Requests
- [ ] Notebooks executed with final interpretations
- [ ] Section 5: Conclusions & Business Recommendations synthesized in `reports/final_report.md`
