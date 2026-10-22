"""Solve the exponential censoring rate that gives the target censored fraction."""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

# Gauss-Hermite nodes for a standard normal x. Deterministic, so the rate is
# identical on every run and machine (no Monte Carlo noise in a design constant).
_NODES, _WEIGHTS = hermegauss(100)
_WEIGHTS = _WEIGHTS / np.sqrt(2 * np.pi)


def expected_over_driver(fn):
    """E[fn(x)] for x ~ N(0, 1), by quadrature."""
    return np.sum(_WEIGHTS * fn(_NODES))


def censored_fraction(rate, h0, b):
    # With event hazard lam(x) = h0*exp(b*x) and censoring hazard c, both constant,
    # P(censored | x) = P(C < T | x) = c / (c + lam(x)). Average over x ~ N(0, 1).
    return expected_over_driver(lambda x: rate / (rate + h0 * np.exp(b * x)))


def solve_censoring_rate(h0, b, target):
    # censored_fraction rises from 0 to 1 as the rate grows, so a bracketed root exists.
    return brentq(lambda c: censored_fraction(c, h0, b) - target, 1e-8, 1e3, xtol=1e-14)


def marginal_survival(t, h0, b):
    """Population survival curve S(t) = E_x[exp(-h0*exp(b*x)*t)], for comparison with Kaplan-Meier."""
    t = np.atleast_1d(t)
    return np.array([expected_over_driver(lambda x: np.exp(-h0 * np.exp(b * x) * ti)) for ti in t])
