# Contributing Guidelines & Git Workflow

Welcome to the German Credit Risk Scoring ML project repository! To maintain code quality, prevent data leakage, and ensure clean integration across all four team members, all contributors must adhere to the workflows and standards described below.

---

## 1. Branching Strategy & Protected Main

- **The `main` branch is strictly protected**:
  - Direct pushes or commits to `main` are disallowed.
  - All changes must be merged through a GitHub Pull Request (PR).
- **PR Review & Integration**:
  - **Rajeev** (Repo Lead & Integrator) is the designated reviewer for all pull requests.
  - At least one approved review from Rajeev is required before merging into `main`.
- **Standard Branch Naming Conventions**:
  - `feature/eda` — Owned by **Shubham Behri** (Data loading, schema, exploratory analysis).
  - `feature/preprocessing` — Owned by **Amlan Kumar Pradhan** (Encodings, scalers, feature selection).
  - `feature/modeling` — Owned by **Rajeev** (Pipelines, model training, evaluation, threshold tuning).
  - `feature/interpretability` — Owned by **Sohan Prusty** (SHAP, permutation importance, fairness analysis).

---

## 2. Commit Standards & Notebook Hygiene

- **Commit Messages**:
  - Use clear, descriptive, imperative commit messages (e.g., `feat(preprocessing): implement log transform for credit amount`, `fix(eval): correct cost matrix calculation`).
- **Jupyter Notebook Hygiene**:
  - Before committing any `.ipynb` file, **Restart Kernel and Run All Cells** to ensure reproducibility from top to bottom.
  - Clear excessive or messy intermediate debug output before committing.
  - Never commit credentials, local absolute paths, or unneeded bulky outputs.

---

## 3. Shared Function Signatures & API Contracts

- Shared interfaces are defined across `src/`:
  - `src/data.py`: `COLUMN_NAMES`, `load_data()`
  - `src/preprocessing.py`: `build_preprocessor()`
  - `src/features.py`: `select_features()`
  - `src/models.py`: `build_pipeline()`
  - `src/evaluation.py`: `expected_cost()`, `cross_validate_model()`, `find_best_threshold()`
  - `src/interpret.py`: `shap_summary()`, `permutation_importance_report()`
  - `src/fairness.py`: `approval_rate_by_group()`, `compare_with_without_sensitive()`
- **Rule**: **Do NOT modify any shared function signature, parameter name, or return type without explicit prior discussion and consent from the team.** Any contract breaking change will break dependent notebooks and modules.

---

## 4. Preventing Data Leakage (Strict Requirement)

- **Feature Selection & Transformations**:
  - Feature encoding, imputations, scaling, and feature selection must **never** be fitted on the entire dataset prior to cross-validation.
  - All selectors and transformers must reside inside `sklearn.pipeline.Pipeline` or `imblearn.pipeline.Pipeline` so they are fitted solely on training folds.
- **Resampling (SMOTE / Class Weighting)**:
  - Synthetic oversampling methods such as SMOTE must strictly occur **inside** the CV loop (using `imblearn.pipeline.Pipeline`) so validation folds remain unpolluted and untouched.
