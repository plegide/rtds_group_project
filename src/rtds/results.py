"""Write the results tables. The columns are fixed in CLAUDE.md and the same in every version."""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

TABLE1_COLUMNS = [
    "version", "scenario", "rho", "dataset_seed", "patient_id", "run", "true_driver_visit",
    "credit_visit_1", "credit_visit_2", "credit_visit_3", "top_visit", "hit",
    "credit_share", "heldout_c_index",
]
TABLE2_COLUMNS = ["version", "scenario", "rho", "dataset_seed", "visit", "c_index_drop"]

# One row per dataset: model quality and the true-risk C-index as a fixed reference.
DATASET_COLUMNS = ["version", "scenario", "rho", "dataset_seed", "heldout_c_index", "true_risk_c_index"]

# Ties and NaN shares do not fit Table 1's fixed columns, so they go to a separate log.
LOG_COLUMNS = ["version", "scenario", "rho", "dataset_seed", "patient_id", "run", "n_tied", "share_is_nan"]


def start_table(path, columns):
    """Create (or overwrite) a table with just its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_rows(path, rows, columns):
    """Append rows to a table. Called after every dataset, so a crash loses at most one dataset."""
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=False, index=False)
