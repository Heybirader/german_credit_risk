"""Model definition and end-to-end Pipeline factories.

Owner: Rajeev
Scope: Leakage-safe sklearn / imblearn Pipelines integrating preprocessing,
       optional SMOTE resampling, class weighting, and benchmark estimators
       (Logistic Regression, Random Forest, Gradient Boosting).
"""

from typing import Any, Optional
from sklearn.pipeline import Pipeline


def build_pipeline(
    model_name: str = "logistic_regression",
    class_weight: Optional[Any] = None,
    use_smote: bool = False,
    drop_sensitive: bool = False,
    **model_params: Any,
) -> Pipeline:
    """Build an end-to-end leakage-safe classification pipeline.

    Parameters
    ----------
    model_name : {"logistic_regression", "random_forest", "gradient_boosting"}, default="logistic_regression"
        Classifier architecture to instantiate.
    class_weight : dict, "balanced", or None, default=None
        Class weight configuration passed to estimators that support it.
    use_smote : bool, default=False
        If True, includes SMOTE oversampling within an imblearn Pipeline
        so resampling is executed strictly on training folds.
    drop_sensitive : bool, default=False
        Whether to drop sensitive attributes (sex, age, foreign worker) in the preprocessor.
    **model_params : Any
        Additional keyword arguments forwarded to the estimator constructor.

    Returns
    -------
    pipeline : Pipeline
        Complete end-to-end scikit-learn or imblearn Pipeline comprising:
        1. Preprocessing (`build_preprocessor`)
        2. Optional Resampling (SMOTE if use_smote is True)
        3. Classifier (LogisticRegression, RandomForestClassifier, or HistGradientBoostingClassifier)

    Notes
    -----
    If use_smote is True, an imbalanced-learn Pipeline (`imblearn.pipeline.Pipeline`)
    must be returned to properly handle resamplers during cross-validation.
    """
    # TODO (owner: Rajeev): Call build_preprocessor(drop_sensitive=drop_sensitive),
    # instantiate base classifier with class_weight and model_params, optionally insert
    # SMOTE step via imblearn Pipeline, and return the integrated pipeline.
    raise NotImplementedError("build_pipeline is not yet implemented.")
