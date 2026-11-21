"""Feature selection transformers and utilities for the German Credit Dataset.

Owner: Amlan Kumar Pradhan
Scope: Supervised and unsupervised feature selection methods (Mutual Information,
       L1-penalized Logistic Regression, model-based/tree importances) structured
       to execute strictly inside CV folds.
"""

from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin


def select_features(
    method: str = "mutual_info",
    k: int | str = 15,
    **kwargs: Any,
) -> BaseEstimator:
    """Build a feature selection transformer compatible with scikit-learn Pipelines.

    Parameters
    ----------
    method : {"mutual_info", "l1_logistic", "model_based"}, default="mutual_info"
        Feature selection technique to employ:
        - "mutual_info": SelectPercentile / SelectKBest using mutual_info_classif
        - "l1_logistic": SelectFromModel using L1-penalized LogisticRegression
        - "model_based": SelectFromModel using RandomForestClassifier / ExtraTrees
    k : int or str, default=15
        Number of top features to retain or threshold rule.
    **kwargs : Any
        Additional hyperparameters passed to underlying estimators or selectors.

    Returns
    -------
    selector : BaseEstimator
        Fitted or fit-ready feature selection transformer implementing fit and transform.

    Notes
    -----
    Feature selection must always occur inside cross-validation pipelines
    to avoid look-ahead data leakage.
    """
    # TODO (owner: Amlan Kumar Pradhan): Instantiate and return the requested selector
    # (e.g., SelectKBest with mutual_info_classif, SelectFromModel with LogisticRegression(penalty='l1'),
    # or tree-based feature selector).
    raise NotImplementedError("select_features is not yet implemented.")
