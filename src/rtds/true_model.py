"""The true survival function of the simulator, shaped like a fitted sksurv model.

Used in Run 1: the tool explains the truth, so the driver should get all credit.
"""

import numpy as np
from sksurv.functions import StepFunction


class TrueSurvivalModel:
    def __init__(self, h0, b, driver_column, grid):
        self.h0 = h0
        self.b = b
        self.driver_column = driver_column
        # A StepFunction is exact at its own x-points, so building it on the grid
        # and evaluating it on the same grid returns exp(-h0*exp(b*x)*t) exactly.
        # Off-grid times would be wrong, hence the explainer must always get the grid.
        self.grid = np.asarray(grid, dtype=float)

    def fit(self, X, y):
        return self  # nothing to learn

    def predict_survival_function(self, X):
        # Select the driver by name, so it works on any column order.
        x = X[self.driver_column].to_numpy(dtype=float)
        rate = self.h0 * np.exp(self.b * x)
        surv = np.exp(-np.outer(rate, self.grid))
        return np.array([StepFunction(self.grid, s) for s in surv])
