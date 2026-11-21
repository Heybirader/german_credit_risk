"""Model interpretability, SHAP values, and feature importance analyses.

Owner: Sohan Prusty
Scope: Global and local explainability via SHAP (TreeSHAP, LinearExplainer),
       Permutation Importance, Logistic Regression coefficient log-odds interpretation,
       and single-borrower decision breakdown.
"""

from typing import Any, Optional
import pandas as pd


def shap_summary(
    pipeline: Any,
    X: pd.DataFrame,
    plot_type: str = "dot",
    max_display: int = 15,
) -> Any:
    """Generate SHAP summary explanation for a fitted pipeline and feature matrix.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline (with preprocessor and final estimator).
    X : pd.DataFrame
        Evaluation feature set.
    plot_type : {"dot", "bar"}, default="dot"
        SHAP summary plot visualization type.
    max_display : int, default=15
        Maximum number of top influential features to display.

    Returns
    -------
    explanation : Any
        Computed SHAP Explanation object or values.
    """
    # TODO (owner: Sohan Prusty): Extract transformed feature names from preprocessor,
    # instantiate appropriate shap.Explainer (TreeExplainer or LinearExplainer),
    # compute SHAP values on transformed X, and render/return summary plot.
    raise NotImplementedError("shap_summary is not yet implemented.")


def permutation_importance_report(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance on holdout validation data.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline.
    X : pd.DataFrame
        Holdout validation features.
    y : pd.Series
        Validation target vector.
    scoring : str, default="roc_auc"
        Evaluation metric to assess score degradation upon column shuffling.
    n_repeats : int, default=10
        Number of times to permute each feature.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    report_df : pd.DataFrame
        DataFrame indexed by feature names with columns ['importance_mean', 'importance_std']
        sorted descending by mean importance.
    """
    # TODO (owner: Sohan Prusty): Call sklearn.inspection.permutation_importance,
    # map scores to original or preprocessed feature names, sort by importance_mean,
    # and return structured summary DataFrame.
    raise NotImplementedError("permutation_importance_report is not yet implemented.")


def explain_prediction(
    pipeline: Any,
    sample: pd.Series,
) -> Any:
    """Explain an individual borrower's credit prediction using local feature contributions.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline.
    sample : pd.Series
        Single applicant feature row.

    Returns
    -------
    explanation : Any
        Local SHAP waterfall/force object or formatted contribution table.
    """
    # TODO (owner: Sohan Prusty): Compute SHAP values for single sample,
    # extract top positive and negative risk contributors, and display explanation.
    raise NotImplementedError("explain_prediction is not yet implemented.")
