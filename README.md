# Credit Risk Scoring on the UCI German Credit Dataset

## Project Overview
This repository contains a modular machine learning pipeline and analysis for credit risk scoring using the UCI German Credit Dataset. 

The primary objective is the **binary classification of loan applicants** to assess credit risk:
- **Good Credit Risk**: Label 1 in raw data $\rightarrow$ recoded to **`0`** (Negative class).
- **Bad Credit Risk**: Label 2 in raw data $\rightarrow$ recoded to **`1`** (Positive class).

In financial credit risk modeling, misclassifying a high-risk borrower (false negative) carries much greater financial loss than misclassifying a creditworthy borrower (false positive). Thus, our evaluation framework directly incorporates an asymmetric cost matrix alongside standard classification metrics.

---

## Dataset Facts

- **Instance Count**: 1,000 borrower records.
- **Attributes**: 20 input attributes (7 numerical, 13 categorical) plus 1 binary target label.
- **Data Location & Ingestion**:
  - Raw dataset files (`german.data`, `german.data-numeric`, `german.doc`) must be placed manually in `data/raw/`.
  - **No automated downloads** should be performed.
- **File Format**:
  - `german.data` has **no header row** and values are **space-separated**.
- **Categorical Encodings**:
  - Attributes use qualitative codes (e.g., `A11`, `A32`) whose meanings and mappings are defined in `german.doc`.
  - Certain categories that might look like missing data (e.g., `A65` = *unknown / no savings account*, `A124` = *unknown / no property*) are **legitimate distinct categories**, not missing values (`NaN`).
- **Sensitive Features**:
  - Attribute 9: Personal status and sex (`personal_status_sex`)
  - Attribute 13: Age in years (`age`)
  - Attribute 20: Foreign worker status (`foreign_worker`)
- **Asymmetric Cost Matrix & Evaluation**:
  - Predicting Good when actually Bad (False Negative, credit granted to defaulting borrower): **Cost = 5**.
  - Predicting Bad when actually Good (False Positive, credit denied to creditworthy applicant): **Cost = 1**.
  - Standard accuracy alone is an inappropriate metric for this problem; models must be evaluated and tuned using **expected financial cost** alongside ROC-AUC, PR-AUC, precision, recall, and cost curves.

---

## Team & Module Ownership

| Member | Complexity / Role | Source Modules Owned | Notebooks Owned | Key Scope & Responsibilities |
| :--- | :--- | :--- | :--- | :--- |
| **Rajeev** | **HARD**<br>Repo Lead & Integrator | `src/models.py`<br>`src/evaluation.py` | `03_modeling_evaluation.ipynb`<br>`04_imbalance_threshold.ipynb` | Leakage-safe `sklearn` Pipelines; Logistic Regression, Random Forest, and Gradient Boosting; Repeated Stratified CV; Expected-cost metric implementation; Decision threshold tuning; Class weighting (`class_weight`) vs. SMOTE resampling comparison; Benchmark results table. |
| **Amlan Kumar Pradhan** | **MODERATE** | `src/preprocessing.py`<br>`src/features.py` | `02_preprocessing_feature_selection.ipynb` | Preprocessing transformations (one-hot encoding, ordinal encoding, robust/standard scaling, log-transform of credit amount); Leakage-safe feature selection (Mutual Information, L1-penalized logistic regression, model-based importances) strictly nested within CV folds. |
| **Sohan Prusty** | **MODERATE** | `src/interpret.py`<br>`src/fairness.py` | `05_interpretability.ipynb`<br>`06_fairness.ipynb` | Explainable AI via TreeSHAP/KernelSHAP and Permutation Importance; Linear model coefficient inspection; Local explanations for individual loan decisions; Algorithmic fairness analysis (mitigation by dropping protected attributes: sex, age, foreign worker; demographic parity / approval rates by group; proxy variable detection). |
| **Shubham Behri** | **EASY** | `src/data.py` | `01_eda.ipynb` | Robust data ingestion with standardized column headers; Target recoding (`BAD=1`, `GOOD=0`); Exploratory Data Analysis (EDA) with written interpretations; README and project documentation maintenance; Final report formatting. |

---

## Repository Structure

```text
german-credit-risk/
│
├── README.md                                  <- Project summary, dataset facts, ownership, setup guide
├── requirements.txt                           <- Python package dependencies
├── .gitignore                                 <- Standard ignore rules for Python, notebooks, data, envs
├── CONTRIBUTING.md                            <- Git branching, pull request workflow, code standards
│
├── data/
│   ├── raw/.gitkeep                           <- Raw files (german.data, german.doc) placed here manually
│   └── processed/.gitkeep                     <- Cached processed artifacts (git-ignored)
│
├── docs/
│   └── task_division.md                       <- Detailed ownership table, deliverables, status tracker
│
├── notebooks/
│   ├── 01_eda.ipynb                           <- EDA and initial observations (Shubham)
│   ├── 02_preprocessing_feature_selection.ipynb <- Encoding, scaling, feature selection (Amlan)
│   ├── 03_modeling_evaluation.ipynb           <- Baseline models & repeated stratified CV (Rajeev)
│   ├── 04_imbalance_threshold.ipynb           <- Cost-sensitive threshold tuning & SMOTE (Rajeev)
│   ├── 05_interpretability.ipynb              <- SHAP and permutation importance (Sohan)
│   └── 06_fairness.ipynb                      <- Disparate impact & fairness mitigations (Sohan)
│
├── reports/
│   ├── figures/.gitkeep                       <- Generated plots and visualizations
│   └── final_report.md                        <- Consolidated project report skeleton
│
└── src/
    ├── __init__.py                            <- Package initialization
    ├── data.py                                <- Schema constants and data loader stub
    ├── preprocessing.py                       <- ColumnTransformer preprocessor stub
    ├── features.py                            <- Nested feature selection selector stub
    ├── models.py                              <- Leakage-safe scikit-learn Pipeline factory stub
    ├── evaluation.py                          <- Expected-cost, threshold tuning, and CV stubs
    ├── interpret.py                           <- SHAP summary and permutation importance stubs
    └── fairness.py                            <- Group approval rates and bias assessment stubs
```

---

## Getting Started

### 1. Environment Setup
Clone the repository and set up a virtual environment:
```bash
git clone <repo-url>
cd german-credit-risk
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Adding the Dataset
Place the original dataset files into `data/raw/`:
- `data/raw/german.data`
- `data/raw/german.data-numeric`
- `data/raw/german.doc`

### 3. Development Workflow
Please review [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming conventions, PR guidelines, and data leakage safeguards before submitting any code.
