"""Simulate patients: 3 correlated visits, an exponential event time driven by one visit,
and independent exponential censoring.

Common random numbers: all randomness is drawn first as base draws (normals and
uniforms) that depend only on the dataset index. The correlation level and the
scenario only change how the base normals are turned into visits.
"""

from dataclasses import dataclass

import numpy as np
from sksurv.util import Surv

from rtds import seeds
from rtds.censoring import solve_censoring_rate


@dataclass(frozen=True)
class Sample:
    visits: np.ndarray   # (patients, n_visits); column k-1 is visit k
    time: np.ndarray     # observed time = min(event time, censoring time)
    event: np.ndarray    # True if the event was observed

    @property
    def y(self):
        return Surv.from_arrays(event=self.event, time=self.time)


@dataclass(frozen=True)
class Dataset:
    train: Sample
    test: Sample


def draw_base(gen, n, n_visits):
    """All random draws for n patients. Their order inside this function is fixed: never reorder."""
    return {
        "z": gen.standard_normal((n, n_visits)),
        "u_event": gen.uniform(size=n),
        "u_censor": gen.uniform(size=n),
    }


def build_visits(z, rho, driver_visit):
    """Exchangeable-correlated visits with the driver taken directly from z[:, 0].

    The exchangeable matrix looks the same in any visit order, so we order the driver
    first and take its Cholesky factor L. The first row of L is (1, 0, ..., 0), so the
    driver equals z[:, 0] at every rho; the other visits get the remaining columns of
    z @ L.T in increasing visit order.
    """
    n_visits = z.shape[1]
    cov = np.full((n_visits, n_visits), rho)
    np.fill_diagonal(cov, 1.0)
    ordered = z @ np.linalg.cholesky(cov).T      # column 0 is the driver
    others = [k for k in range(1, n_visits + 1) if k != driver_visit]
    visits = np.empty_like(ordered)
    visits[:, driver_visit - 1] = ordered[:, 0]
    for j, k in enumerate(others, start=1):
        visits[:, k - 1] = ordered[:, j]
    return visits


def simulate_from_base(base, rho, driver_visit, h0, b, censor_rate):
    visits = build_visits(base["z"], rho, driver_visit)
    # Inverse-CDF draws for S(t|x) = exp(-h0*exp(b*x)*t). The rate uses z[:, 0], which
    # equals the driver visit, so event times do not depend on rho or the scenario.
    event_time = -np.log(base["u_event"]) / (h0 * np.exp(b * base["z"][:, 0]))
    censor_time = -np.log(base["u_censor"]) / censor_rate
    return Sample(
        visits=visits,
        time=np.minimum(event_time, censor_time),
        event=event_time <= censor_time,
    )


def censor_rate_from_config(cfg):
    sim = cfg["simulation"]
    return solve_censoring_rate(sim["h0"], sim["b"], sim["target_censored_fraction"])


def simulate_sample(cfg, gen, n, rho, scenario):
    sim = cfg["simulation"]
    base = draw_base(gen, n, sim["n_visits"])
    return simulate_from_base(base, rho, cfg["scenarios"][scenario], sim["h0"], sim["b"], censor_rate_from_config(cfg))


def simulate_dataset(cfg, dataset_index, rho, scenario):
    """Dataset i: the first n_train patients are training, the rest held out."""
    sim = cfg["simulation"]
    gen = seeds.rng(cfg["master_seed"], "data", dataset_index)
    full = simulate_sample(cfg, gen, sim["n_train"] + sim["n_test"], rho, scenario)
    split = sim["n_train"]
    part = lambda s: Sample(full.visits[s], full.time[s], full.event[s])
    return Dataset(train=part(slice(0, split)), test=part(slice(split, None)))


def reference_sample(cfg, rho=0.0, scenario="A"):
    """The large reference sample. Its observed times are the same for every rho and
    scenario (see simulate_from_base), so the defaults do not affect the time grid."""
    gen = seeds.rng(cfg["master_seed"], "reference", 0)
    return simulate_sample(cfg, gen, cfg["simulation"]["n_reference"], rho, scenario)
