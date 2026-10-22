"""Throwaway test data for the Stage 0 checks only.

The real simulator (driver-first Cholesky, censoring, reference sample) is Stage 1.
Here there is no censoring and the time grid is provisional: it only has to be a
reasonable set of times for the tool to evaluate on.
"""

import numpy as np
import pandas as pd
from sksurv.util import Surv

from rtds.interfaces import visit_columns


def correlated_visits(gen, n, n_visits, rho):
    cov = np.full((n_visits, n_visits), rho)
    np.fill_diagonal(cov, 1.0)
    z = gen.standard_normal((n, n_visits))
    return pd.DataFrame(z @ np.linalg.cholesky(cov).T, columns=visit_columns(n_visits))


def event_times(gen, x_driver, h0, b):
    # Exponential event time: T = -log(U) / (h0 * exp(b * x)).
    rate = h0 * np.exp(b * np.asarray(x_driver))
    return -np.log(gen.uniform(size=len(rate))) / rate


def provisional_grid(times, tool_cfg):
    lo, hi = np.quantile(times, [tool_cfg["grid_lower_quantile"], tool_cfg["grid_upper_quantile"]])
    return np.linspace(lo, hi, tool_cfg["n_grid"])


def all_events_y(times):
    return Surv.from_arrays(event=np.ones(len(times), dtype=bool), time=times)


def make_setup(cfg, rho):
    """Training sample, 100-row background, one patient and a grid, all from the master seed."""
    sim, tool = cfg["simulation"], cfg["tool"]
    driver_col = f"visit_{cfg['scenarios']['A']}"
    gen = np.random.default_rng(cfg["master_seed"])
    train = correlated_visits(gen, sim["n_train"], sim["n_visits"], rho)
    times = event_times(gen, train[driver_col], sim["h0"], sim["b"])
    bg_idx = gen.choice(sim["n_train"], tool["n_background"], replace=False)
    background = train.iloc[bg_idx].reset_index(drop=True)
    background_y = all_events_y(times[bg_idx])
    patients = correlated_visits(gen, 2, sim["n_visits"], rho)
    grid = provisional_grid(times, tool)
    return background, background_y, patients, grid, driver_col
