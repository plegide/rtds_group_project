"""Held-out model quality: Harrell's C-index and the integrated Brier score."""

import numpy as np
from sksurv.metrics import concordance_index_censored, integrated_brier_score


def c_index(y, risk):
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def ibs(model, X_test, y_train, y_test, grid):
    """Integrated Brier score on the fixed grid; censoring is handled by IPCW from the training outcomes."""
    surv = np.array([f(grid) for f in model.predict_survival_function(X_test)])
    # sksurv evaluates the training censoring curve at every test time and fails beyond the
    # last training time. A test patient observed after the grid's end only enters the score
    # as "still at risk at t" for every grid t, so capping their time anywhere above the grid
    # leaves the score unchanged. Cap just below the last training time.
    cap = np.nextafter(y_train["time"].max(), 0)
    assert grid[-1] < cap
    y_capped = y_test.copy()
    y_capped["time"] = np.minimum(y_test["time"], cap)
    return integrated_brier_score(y_train, y_capped, surv, grid)
