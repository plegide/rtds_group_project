"""The three swappable parts. Stage 6 replaces only the model.

These are typing Protocols: any class with these methods fits, no inheritance needed.
"""

from typing import Protocol

import numpy as np
import pandas as pd


def visit_columns(n_visits):
    """Column names used everywhere; results are matched by these names, never by position."""
    return [f"visit_{k}" for k in range(1, n_visits + 1)]


class InputBuilder(Protocol):
    def build(self, visits: np.ndarray) -> pd.DataFrame:
        """Raw visits, shape (patients, n_visits), to the model's input columns."""
        ...


class SurvivalModel(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SurvivalModel":
        """y is a sksurv structured array with fields (event, time)."""
        ...

    def predict_survival_function(self, X: pd.DataFrame) -> np.ndarray:
        """One sksurv StepFunction per row of X."""
        ...


class Explainer(Protocol):
    def explain(self, model: SurvivalModel, patient: pd.DataFrame) -> dict[str, float]:
        """One patient (one-row DataFrame) in, one credit per visit column out."""
        ...
