"""Run 2: how much the model relies on each visit, by shuffling it in the held-out set."""

import numpy as np

from rtds import seeds
from rtds.metrics import c_index


def c_index_drops(cfg, model, X_test, y_test, dataset_index):
    """{visit column: mean drop in held-out C-index over the shuffles of that column}.

    The shuffles come from the dataset's own stream, so they are the same at every rho.
    """
    gen = seeds.rng(cfg["master_seed"], "shuffle", dataset_index)
    base = c_index(y_test, model.predict_risk(X_test))
    drops = {}
    for col in X_test.columns:
        shuffled_c = []
        for _ in range(cfg["runs"]["run2_n_shuffles"]):
            X = X_test.copy()
            X[col] = gen.permutation(X[col].to_numpy())
            shuffled_c.append(c_index(y_test, model.predict_risk(X)))
        drops[col] = base - np.mean(shuffled_c)
    return drops
