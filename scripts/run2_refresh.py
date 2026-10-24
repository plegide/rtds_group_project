"""Rerun only Run 2 on datasets that are already finished, without redoing Run 3.

Run 2 records two measures per visit (C-index drop and integrated Brier score increase).
For datasets whose Table 2 was written before the Brier column existed, this script:
1. rebuilds each model from its seed (the fits are deterministic),
2. reruns Run 2 with the same shuffles,
3. keeps the new rows only if every C-index drop equals the saved one (to 1e-12; separate
   processes differ only in the last bits, while a different model would differ by far more).
   That proves the rebuilt model is the one that produced Run 3,
4. rewrites the per-dataset and merged Table 2 files.
The Stage 3, Stage 4 and 6c tables are dataset 0 of the same models, so they are rebuilt
from those rows, with the same check against what they held.

Run from the repository root:  python scripts/run2_refresh.py   (6 parallel workers)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rtds.config import load_config
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.models import make_model
from rtds.reliance import run2, table2_rows
from rtds.results import TABLE2_COLUMNS, scenario_dir
from rtds.run_dataset import task_paths
from rtds.simulate import simulate_dataset

SAME = 1e-12  # largest C-index-drop difference still counted as the same model


def refresh(cfg, scenario, version, i, rho, grid):
    path = task_paths(scenario_dir(scenario, version) / "main", scenario, rho, i, version)["table2"]
    old = pd.read_csv(path)
    builder = FlatVisitColumns()
    ds = simulate_dataset(cfg, i, rho, scenario)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
    model = make_model(cfg, version, i, grid).fit(X_train, ds.train.y)
    new = pd.DataFrame(table2_rows(version, scenario, rho, i,
                                   run2(cfg, model, X_test, ds.train.y, ds.test.y, grid, i)), columns=TABLE2_COLUMNS)
    gap = float(np.max(np.abs(new["c_index_drop"].to_numpy() - old["c_index_drop"].to_numpy())))
    if gap <= SAME:
        new.to_csv(path, index=False)
    return scenario, version, i, rho, gap


def merged(scenario, version, tasks):
    out = scenario_dir(scenario, version) / "main"
    return pd.concat([pd.read_csv(task_paths(out, scenario, rho, i, version)["table2"]) for i, rho in tasks],
                     ignore_index=True)


def replace_dataset0(target, source_rows):
    """Rewrite a dataset-0 Table 2 (Stage 3, 4 or 6c) from the refreshed main-run rows, after checking them."""
    old = pd.read_csv(target)
    new = source_rows[source_rows["dataset_seed"] == 0].sort_values(["rho", "visit"]).reset_index(drop=True)
    old = old.sort_values(["rho", "visit"]).reset_index(drop=True)
    assert np.max(np.abs(old["c_index_drop"].to_numpy() - new["c_index_drop"].to_numpy())) <= SAME, target
    new.to_csv(target, index=False)


def main():
    cfg = load_config()
    grid = time_grid(cfg)
    rhos = cfg["simulation"]["rho_levels"]
    full = [(i, rho) for i in range(cfg["runs"]["n_datasets"]) for rho in rhos]
    mirror = [(i, rho) for i in range(cfg["mirror"]["n_datasets"]) for rho in rhos]
    jobs = ([("A", "lstm", t) for t in full] + [("B", "lstm", t) for t in full]
            + [("A", "forest", t) for t in full] + [("B", "forest", t) for t in mirror])

    start = time.perf_counter()
    done = Parallel(n_jobs=cfg["runs"]["n_jobs"], verbose=5)(
        delayed(refresh)(cfg, s, v, i, rho, grid) for s, v, (i, rho) in jobs)
    bad = [d for d in done if d[4] > SAME]
    print(f"{len(done)} datasets rerun in {(time.perf_counter() - start) / 60:.1f} min; "
          f"C-index drops equal to the saved ones in {len(done) - len(bad)} of {len(done)} "
          f"(largest gap {max(d[4] for d in done):.1e})")
    for d in bad:
        print(f"  NOT updated (model differs): scenario {d[0]}, {d[1]}, dataset {d[2]}, rho {d[3]}: max gap {d[4]:.2e}")
    if bad:
        raise SystemExit("some rebuilt models differ; their Table 2 rows were left unchanged")

    forest_a, forest_b = merged("A", "forest", full), merged("B", "forest", mirror)
    lstm_a, lstm_b = merged("A", "lstm", full), merged("B", "lstm", full)
    forest_a.to_csv(scenario_dir("A") / "table2_main.csv", index=False)
    forest_b.to_csv(scenario_dir("B") / "table2_mirror.csv", index=False)
    lstm_a.to_csv(scenario_dir("A", "lstm") / "table2_main.csv", index=False)
    lstm_b.to_csv(scenario_dir("B", "lstm") / "table2_main.csv", index=False)
    for scen, rows in (("A", forest_a), ("B", forest_b)):
        for name in ("table2_stage3.csv", "table2_stage4.csv"):
            replace_dataset0(scenario_dir(scen) / name, rows)
    for scen, rows in (("A", lstm_a), ("B", lstm_b)):
        replace_dataset0(scenario_dir(scen, "lstm") / "table2_pilot.csv", rows)
    print("Merged Table 2 files and the Stage 3, Stage 4 and 6c tables rewritten with both Run 2 measures.")


if __name__ == "__main__":
    main()
