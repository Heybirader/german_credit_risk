"""Data loading and schema definitions for the UCI German Credit Dataset.

Owner: Shubham Behri
Scope: Loading raw space-separated dataset with standardized column names,
       re-coding labels (BAD = 1, GOOD = 0), and returning feature matrix X
       and target series y.
"""

from typing import Tuple
import pandas as pd

# The 20 attribute names defined in german.doc + "target"
COLUMN_NAMES: list[str] = [
    "checking_status",          # Attribute 1: Status of existing checking account (categorical)
    "duration",                 # Attribute 2: Duration in month (numerical)
    "credit_history",           # Attribute 3: Credit history (categorical)
    "purpose",                  # Attribute 4: Purpose (categorical)
    "credit_amount",            # Attribute 5: Credit amount (numerical)
    "savings_status",           # Attribute 6: Savings account/bonds (categorical, A65 = unknown)
    "employment_since",         # Attribute 7: Present employment since (categorical)
    "installment_rate",         # Attribute 8: Installment rate in % of disposable income (numerical)
    "personal_status_sex",      # Attribute 9: Personal status and sex (categorical, sensitive)
    "other_debtors",            # Attribute 10: Other debtors / guarantors (categorical)
    "residence_since",          # Attribute 11: Present residence since (numerical)
    "property",                 # Attribute 12: Property (categorical, A124 = unknown / no property)
    "age",                      # Attribute 13: Age in years (numerical, sensitive)
    "other_installment_plans",  # Attribute 14: Other installment plans (categorical)
    "housing",                  # Attribute 15: Housing (categorical)
    "existing_credits",         # Attribute 16: Number of existing credits at this bank (numerical)
    "job",                      # Attribute 17: Job (categorical)
    "num_dependents",           # Attribute 18: Number of people liable to provide maintenance for (numerical)
    "telephone",                # Attribute 19: Telephone (categorical)
    "foreign_worker",           # Attribute 20: Foreign worker (categorical, sensitive)
    "target",                   # Label: 1 = Good, 2 = Bad in raw file
]

# Sensitive feature identifiers
SENSITIVE_COLUMNS: list[str] = [
    "personal_status_sex",  # Attribute 9
    "age",                  # Attribute 13
    "foreign_worker",       # Attribute 20
]


def load_data(
    data_path: str = "data/raw/german.data",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the UCI German Credit Dataset, assign column names, and recode target.

    Parameters
    ----------
    data_path : str, default="data/raw/german.data"
        Path to the raw space-separated german.data file (no header).

    Returns
    -------
    X : pd.DataFrame
        DataFrame containing the 20 predictor attributes with standardized column names.
    y : pd.Series
        Binary target series recoded such that:
        - BAD = 1 (positive class, originally coded as 2 in raw data)
        - GOOD = 0 (negative class, originally coded as 1 in raw data)

    Notes
    -----
    - The raw file has NO header and is space-separated.
    - Category codes such as A65 (unknown savings) and A124 (no property)
      are valid categories, NOT missing values.
    """
    # TODO (owner: Shubham Behri): Read data_path with delim_whitespace=True / sep=r'\s+',
    # assign COLUMN_NAMES, map target {1: 0, 2: 1}, and return X, y.
    raise NotImplementedError("load_data is not yet implemented.")
