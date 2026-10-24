"""Stage 3: fit the forest on one dataset per rho, check its quality, and run Run 2.

For each rho: the grid lies inside the forest's range, held-out C-index and integrated
Brier score, and Run 2: the drop in C-index and the increase in integrated Brier score
when each visit is shuffled.

Writes results/scenario_<X>/table2_stage3.csv (Table 2 rows).
Run from the repository root:  python scripts/stage3_forest_run2.py [--scenario B]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

from rtds.cli import parse_scenario
from rtds.config import load_config
from rtds.forest import ForestModel
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.metrics import c_index, ibs
from rtds.reliance import run2, table2_rows
from rtds.results import TABLE2_COLUMNS, append_rows, scenario_dir, start_table
from rtds.simulate import simulate_dataset

VERSION = "forest"
DATASET_INDEX = 0  # one dataset per rho, the same one Stage 4 explains


def main():
    scenario = parse_scenario(__doc__)
    cfg = load_config()
    grid = time_grid(cfg)
    builder = FlatVisitColumns()
    driver_col = f"visit_{cfg['scenarios'][scenario]}"
    table_path = scenario_dir(scenario) / "table2_stage3.csv"
    start_table(table_path, TABLE2_COLUMNS)
    print(f"dataset {DATASET_INDEX}, scenario {scenario} (driver {driver_col}), "
          f"grid {grid[0]:.4f} to {grid[-1]:.4f}")

    for rho in cfg["simulation"]["rho_levels"]:
        start = time.perf_counter()
        ds = simulate_dataset(cfg, DATASET_INDEX, rho, scenario)
        X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
        y_train, y_test = ds.train.y, ds.test.y

        model = ForestModel(cfg, DATASET_INDEX).fit(X_train, y_train)
        lo, hi = model.assert_grid_inside(X_test, grid)
        c = c_index(y_test, model.predict_risk(X_test))
        brier = ibs(model, X_test, y_train, y_test, grid)
        drops = run2(cfg, model, X_test, y_train, y_test, grid, DATASET_INDEX)

        append_rows(table_path, table2_rows(VERSION, scenario, rho, DATASET_INDEX, drops), TABLE2_COLUMNS)
        print(f"\nrho = {rho}  ({time.perf_counter() - start:.1f} s)")
        print(f"  forest range [{lo:.4f}, {hi:.4f}] contains the grid: yes")
        print(f"  held-out C-index = {c:.4f}, integrated Brier score = {brier:.4f}")
        print("  Run 2, drop in C-index when shuffled: "
              + ", ".join(f"{col} = {c:+.4f}" for col, (c, _) in drops.items()))
        print("  Run 2, increase in integrated Brier score: "
              + ", ".join(f"{col} = {b:+.4f}" for col, (_, b) in drops.items()))


if __name__ == "__main__":
    main()
