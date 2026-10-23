"""Stage 6c: pilot the LSTM on one dataset. Did it learn the driver?

For dataset 0 at every rho, in both scenarios: fit the LSTM and the forest, then compare
held-out C-index and integrated Brier score, and run Run 2 on the LSTM. The LSTM settings
in config.toml are fixed after this pilot and never tuned per setting.

Writes results/scenario_<X>/lstm/pilot.csv (one row per rho) and
results/scenario_<X>/lstm/table2_pilot.csv (Run 2 rows).
Run from the repository root:  python scripts/stage6c_lstm_pilot.py   (6 fits in parallel)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd
from joblib import Parallel, delayed

from rtds.config import load_config
from rtds.forest import ForestModel
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.lstm_model import LSTMModel
from rtds.metrics import c_index, ibs
from rtds.reliance import c_index_drops
from rtds.results import TABLE2_COLUMNS, scenario_dir
from rtds.simulate import simulate_dataset

DATASET_INDEX = 0


def pilot(cfg, scenario, rho, grid):
    builder = FlatVisitColumns()
    ds = simulate_dataset(cfg, DATASET_INDEX, rho, scenario)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
    start = time.perf_counter()
    lstm = LSTMModel(cfg, DATASET_INDEX, grid).fit(X_train, ds.train.y)
    fit_s = time.perf_counter() - start
    lstm.assert_grid_inside(X_test, grid)
    forest = ForestModel(cfg, DATASET_INDEX).fit(X_train, ds.train.y)
    drops = c_index_drops(cfg, lstm, X_test, ds.test.y, DATASET_INDEX)
    row = {"scenario": scenario, "rho": rho, "epochs_run": len(lstm.history), "best_epoch": lstm.best_epoch,
           "fit_seconds": fit_s,
           "lstm_c_index": c_index(ds.test.y, lstm.predict_risk(X_test)),
           "forest_c_index": c_index(ds.test.y, forest.predict_risk(X_test)),
           "lstm_ibs": ibs(lstm, X_test, ds.train.y, ds.test.y, grid),
           "forest_ibs": ibs(forest, X_test, ds.train.y, ds.test.y, grid)}
    table2 = [["lstm", scenario, rho, DATASET_INDEX, int(k.removeprefix("visit_")), d] for k, d in drops.items()]
    return row, table2


def main():
    cfg = load_config()
    grid = time_grid(cfg)
    jobs = [(s, rho) for s in ("A", "B") for rho in cfg["simulation"]["rho_levels"]]
    done = Parallel(n_jobs=len(jobs))(delayed(pilot)(cfg, s, rho, grid) for s, rho in jobs)

    for scenario in ("A", "B"):
        rows = [r for r, _ in done if r["scenario"] == scenario]
        t2 = [x for r, t in done if r["scenario"] == scenario for x in t]
        out = scenario_dir(scenario, "lstm")
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "pilot.csv", index=False)
        pd.DataFrame(t2, columns=TABLE2_COLUMNS).to_csv(out / "table2_pilot.csv", index=False)

        driver = cfg["scenarios"][scenario]
        print(f"\nScenario {scenario} (driver visit_{driver}), dataset {DATASET_INDEX}")
        for r in rows:
            d = {x[4]: x[5] for x in t2 if x[2] == r["rho"]}
            print(f"  rho = {r['rho']}: LSTM stopped at epoch {r['epochs_run']} (best {r['best_epoch']}, "
                  f"{r['fit_seconds']:.1f} s); C-index LSTM {r['lstm_c_index']:.4f} vs forest {r['forest_c_index']:.4f}; "
                  f"IBS LSTM {r['lstm_ibs']:.4f} vs forest {r['forest_ibs']:.4f}")
            print(f"            LSTM Run 2 drops: " + ", ".join(f"visit_{k} = {v:+.4f}" for k, v in sorted(d.items())))


if __name__ == "__main__":
    main()
