"""Turn the master seed into independent random streams.

Every stream is identified by (purpose, index). The correlation level and the
scenario are deliberately NOT part of the key: dataset i gets the same base
draws at every correlation level and in both scenarios (common random numbers).
"""

import numpy as np

# Fixed codes so a stream never changes when new purposes are added later.
# Append new purposes at the end; never renumber.
PURPOSES = {
    "reference": 0,   # large reference sample for the time grid
    "data": 1,        # base normals and uniforms of dataset i
    "background": 2,  # which training rows form the background
    "explained": 3,   # which held-out patients are explained
    "model": 4,       # the forest's own seed
    "shuffle": 5,     # Run 2 column shuffles
    "background_check": 6,  # extra background rows for the Stage 4 size check
    "bootstrap": 7,   # resampling datasets in the analysis
    "lstm": 8,        # LSTM weights, validation split and batch order
}


def seed_sequence(master_seed, purpose, index=0):
    return np.random.SeedSequence(entropy=master_seed, spawn_key=(PURPOSES[purpose], index))


def rng(master_seed, purpose, index=0):
    """A numpy Generator for one (purpose, index) stream."""
    return np.random.default_rng(seed_sequence(master_seed, purpose, index))


def int_seed(master_seed, purpose, index=0):
    """A plain int seed, for libraries such as scikit-learn that want random_state=int."""
    return int(seed_sequence(master_seed, purpose, index).generate_state(1)[0])
