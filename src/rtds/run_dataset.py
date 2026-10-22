"""One main-run task: one dataset at one rho. Fit the forest, Run 2, Run 3, save its own files.

Each task writes its own small CSV files, so parallel workers never share a file and
a crash loses at most the tasks that were running. The dataset file is written last
and marks the task as finished; finished tasks are skipped on a restart.
"""

import os

import pandas as pd

from rtds.explain_dataset import explain_dataset
from rtds.forest import ForestModel
from rtds.inputs import FlatVisitColumns
from rtds.metrics import c_index
from rtds.reliance import c_index_drops
from rtds.results import DATASET_COLUMNS, LOG_COLUMNS, TABLE1_COLUMNS, TABLE2_COLUMNS
from rtds.simulate import simulate_dataset

VERSION = "forest"
RUN = 3


def task_paths(out_dir, scenario, rho, dataset_index):
    stem = f"{VERSION}_{scenario}_rho{rho}_ds{dataset_index:03d}"
    return {name: out_dir / f"{stem}_{name}.csv" for name in ("table1", "table2", "log", "dataset")}


def _write(path, rows, columns):
    # Write to a temporary name, then rename: a half-written file never looks finished.
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_dataset(cfg, dataset_index, rho, scenario, grid, out_dir):
    paths = task_paths(out_dir, scenario, rho, dataset_index)
    if paths["dataset"].exists():
        return "skipped"
    builder = FlatVisitColumns()
    driver_col = f"visit_{cfg['scenarios'][scenario]}"

    ds = simulate_dataset(cfg, dataset_index, rho, scenario)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
    model = ForestModel(cfg, dataset_index).fit(X_train, ds.train.y)
    model.assert_grid_inside(X_test, grid)
    c = c_index(ds.test.y, model.predict_risk(X_test))
    true_c = c_index(ds.test.y, X_test[driver_col].to_numpy())  # true risk is increasing in the driver

    drops = c_index_drops(cfg, model, X_test, ds.test.y, dataset_index)
    table2 = [[VERSION, scenario, rho, dataset_index, int(k.removeprefix("visit_")), d] for k, d in drops.items()]
    table1, log = explain_dataset(cfg, model, ds, dataset_index, builder, grid, driver_col,
                                  key=(VERSION, scenario, rho, RUN), heldout_c_index=c)

    _write(paths["table1"], table1, TABLE1_COLUMNS)
    _write(paths["table2"], table2, TABLE2_COLUMNS)
    _write(paths["log"], log, LOG_COLUMNS)
    _write(paths["dataset"], [[VERSION, scenario, rho, dataset_index, c, true_c]], DATASET_COLUMNS)
    return "done"
