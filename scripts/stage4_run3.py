"""Stage 4: Run 3, SurvSHAP(t) on the fitted forest, one dataset per rho.

For each rho: fit the forest, Run 2 (for comparison), then explain the 100 chosen
held-out patients. Prints a few patients' credits and the hit rate and credit share.

Writes table1_stage4.csv, log_stage4.csv and table2_stage4.csv in results/scenario_<X>/.
The three rho levels run in parallel, one process each.
Run from the repository root:  python scripts/stage4_run3.py [--scenario B]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd
from joblib import Parallel, delayed

from rtds.cli import parse_scenario
from rtds.config import load_config
from rtds.explain_dataset import explain_dataset
from rtds.forest import ForestModel
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.metrics import c_index
from rtds.reliance import run2, table2_rows
from rtds.results import (LOG_COLUMNS, TABLE1_COLUMNS, TABLE2_COLUMNS,
                          append_rows, scenario_dir, start_table)
from rtds.simulate import simulate_dataset

VERSION = "forest"
RUN = 3
DATASET_INDEX = 0
N_SHOWN = 5  # patients printed per rho


def run_rho(cfg, rho, scenario, grid):
    """Everything for one rho: fit, Run 2, Run 3. Runs in its own process."""
    builder = FlatVisitColumns()
    driver_col = f"visit_{cfg['scenarios'][scenario]}"
    ds = simulate_dataset(cfg, DATASET_INDEX, rho, scenario)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
    model = ForestModel(cfg, DATASET_INDEX).fit(X_train, ds.train.y)
    model.assert_grid_inside(X_test, grid)
    c = c_index(ds.test.y, model.predict_risk(X_test))
    drops = run2(cfg, model, X_test, ds.train.y, ds.test.y, grid, DATASET_INDEX)
    table2 = table2_rows(VERSION, scenario, rho, DATASET_INDEX, drops)
    start = time.perf_counter()
    rows, log = explain_dataset(cfg, model, ds, DATASET_INDEX, builder, grid, driver_col,
                                key=(VERSION, scenario, rho, RUN), heldout_c_index=c)
    return rho, c, drops, table2, rows, log, time.perf_counter() - start


def main():
    scenario = parse_scenario(__doc__)
    cfg = load_config()
    grid = time_grid(cfg)
    out = scenario_dir(scenario)
    paths = {name: out / f"{name}_stage4.csv" for name in ("table1", "log", "table2")}
    for name, cols in (("table1", TABLE1_COLUMNS), ("log", LOG_COLUMNS), ("table2", TABLE2_COLUMNS)):
        start_table(paths[name], cols)

    rhos = cfg["simulation"]["rho_levels"]
    results = Parallel(n_jobs=len(rhos))(delayed(run_rho)(cfg, rho, scenario, grid) for rho in rhos)

    print(f"scenario {scenario}, driver visit_{cfg['scenarios'][scenario]}, dataset {DATASET_INDEX}")
    for rho, c, drops, table2, rows, log, elapsed in results:  # same order as rhos
        append_rows(paths["table2"], table2, TABLE2_COLUMNS)
        append_rows(paths["table1"], rows, TABLE1_COLUMNS)
        append_rows(paths["log"], log, LOG_COLUMNS)

        t = pd.DataFrame(rows, columns=TABLE1_COLUMNS)
        credit_cols = ["credit_visit_1", "credit_visit_2", "credit_visit_3"]
        print(f"\nrho = {rho}: C-index = {c:.4f}; Run 2 drops: "
              + ", ".join(f"{k} = {c:+.4f}" for k, (c, _) in drops.items())
              + "; Brier increase: " + ", ".join(f"{k} = {b:+.4f}" for k, (_, b) in drops.items()))
        print(f"  {len(t)} patients explained in {elapsed:.0f} s ({elapsed / len(t):.2f} s per patient)")
        print(f"  first {N_SHOWN} patients:")
        print(t.head(N_SHOWN)[["patient_id"] + credit_cols + ["top_visit", "hit", "credit_share"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        mean_credit = t[credit_cols].mean()
        print("  mean credit: " + ", ".join(f"{k.removeprefix('credit_')} = {v:.4f}" for k, v in mean_credit.items()))
        print(f"  hit rate = {t['hit'].mean():.4f}, mean credit share = {t['credit_share'].mean():.4f} "
              f"(min {t['credit_share'].min():.4f}); top visit counts: {t['top_visit'].value_counts().to_dict()}; "
              f"ties = {sum(r[-2] > 1 for r in log)}, NaN shares = {sum(bool(r[-1]) for r in log)}")


if __name__ == "__main__":
    main()
