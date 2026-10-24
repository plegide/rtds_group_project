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
    "lstm_seed_check": 9,  # Stage 6e: extra LSTM training seeds on the same data
}


def seed_sequence(master_seed, purpose, index=0, replicate=None):
    # replicate: a third key level for repeated draws on the same dataset (Stage 6e).
    # Without it the key is exactly (purpose, index), as for every other stream.
    key = (PURPOSES[purpose], index) if replicate is None else (PURPOSES[purpose], index, replicate)
    return np.random.SeedSequence(entropy=master_seed, spawn_key=key)


def rng(master_seed, purpose, index=0, replicate=None):
    """A numpy Generator for one (purpose, index) stream."""
    return np.random.default_rng(seed_sequence(master_seed, purpose, index, replicate))


def int_seed(master_seed, purpose, index=0, replicate=None):
    """A plain int seed, for libraries such as scikit-learn that want random_state=int."""
    return int(seed_sequence(master_seed, purpose, index, replicate).generate_state(1)[0])
