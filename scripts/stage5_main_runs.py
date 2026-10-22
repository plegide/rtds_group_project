"""Stage 5: main runs for the forest, scenario A, every rho x every dataset, in parallel.

Each (dataset, rho) task saves its own files in results/main/ as soon as it finishes.
Re-running the script skips finished tasks, so after a crash just start it again.
At the end the per-task files are merged into results/table1_main.csv, table2_main.csv,
log_main.csv and datasets_main.csv.

Run from the repository root:  python scripts/stage5_main_runs.py   (about 2 hours with 6 workers)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd
from joblib import Parallel, delayed

from rtds.config import load_config
from rtds.grid import time_grid
from rtds.results import RESULTS_DIR
from rtds.run_dataset import run_dataset, task_paths

SCENARIO = "A"


def main():
    cfg = load_config()
    grid = time_grid(cfg)
    out_dir = RESULTS_DIR / "main"
    out_dir.mkdir(parents=True, exist_ok=True)
    rhos = cfg["simulation"]["rho_levels"]
    # Dataset-major order: if the run is stopped early, finished datasets are complete at every rho (paired).
    tasks = [(i, rho) for i in range(cfg["runs"]["n_datasets"]) for rho in rhos]

    start = time.perf_counter()
    status = Parallel(n_jobs=cfg["runs"]["n_jobs"], verbose=10)(
        delayed(run_dataset)(cfg, i, rho, SCENARIO, grid, out_dir) for i, rho in tasks
    )
    print(f"{status.count('done')} tasks run, {status.count('skipped')} skipped, "
          f"{(time.perf_counter() - start) / 60:.1f} min")

    for name in ("table1", "table2", "log", "dataset"):
        merged = pd.concat([pd.read_csv(task_paths(out_dir, SCENARIO, rho, i)[name]) for i, rho in tasks],
                           ignore_index=True)
        target = "datasets" if name == "dataset" else name
        merged.to_csv(RESULTS_DIR / f"{target}_main.csv", index=False)
        print(f"results/{target}_main.csv: {len(merged)} rows")


if __name__ == "__main__":
    main()
