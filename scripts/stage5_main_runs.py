"""Stage 5: main runs for the forest, every rho x every dataset of one scenario, in parallel.

Each (dataset, rho) task saves its own files in results/scenario_<X>/main/ as soon as it finishes.
Re-running the script skips finished tasks, so after a crash just start it again.
At the end the per-task files are merged into table1_main.csv, table2_main.csv,
log_main.csv and datasets_main.csv in results/scenario_<X>/.

Run from the repository root:  python scripts/stage5_main_runs.py [--scenario B]   (about 2.5 hours with 6 workers)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd
from joblib import Parallel, delayed

from rtds.cli import parse_scenario
from rtds.config import load_config
from rtds.grid import time_grid
from rtds.results import scenario_dir
from rtds.run_dataset import run_dataset, task_paths


def main():
    scenario = parse_scenario(__doc__)
    cfg = load_config()
    grid = time_grid(cfg)
    base = scenario_dir(scenario)
    out_dir = base / "main"
    out_dir.mkdir(parents=True, exist_ok=True)
    rhos = cfg["simulation"]["rho_levels"]
    # Dataset-major order: if the run is stopped early, finished datasets are complete at every rho (paired).
    tasks = [(i, rho) for i in range(cfg["runs"]["n_datasets"]) for rho in rhos]

    start = time.perf_counter()
    status = Parallel(n_jobs=cfg["runs"]["n_jobs"], verbose=10)(
        delayed(run_dataset)(cfg, i, rho, scenario, grid, out_dir) for i, rho in tasks
    )
    print(f"{status.count('done')} tasks run, {status.count('skipped')} skipped, "
          f"{(time.perf_counter() - start) / 60:.1f} min")

    for name in ("table1", "table2", "log", "dataset"):
        merged = pd.concat([pd.read_csv(task_paths(out_dir, scenario, rho, i)[name]) for i, rho in tasks],
                           ignore_index=True)
        target = "datasets" if name == "dataset" else name
        merged.to_csv(base / f"{target}_main.csv", index=False)
        print(f"{base.name}/{target}_main.csv: {len(merged)} rows")


if __name__ == "__main__":
    main()
