"""Run 1 (sanity check): SurvSHAP(t) on the TRUE survival function, a handful of datasets per rho.

The truth depends on the driver only, so the driver should get essentially all the
credit and the other visits about zero, at every correlation level.

Writes results/table1_run1.csv (Table 1 rows) and results/log_run1.csv (ties, NaN shares).
Run from the repository root:  python scripts/run1_true_model.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
import pandas as pd

from rtds.config import load_config
from rtds.explain_survshap import SurvShapExplainer
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.results import LOG_COLUMNS, RESULTS_DIR, TABLE1_COLUMNS, append_rows, start_table
from rtds.scores import score_patient
from rtds.selection import explained_patients, tool_inputs
from rtds.simulate import simulate_dataset
from rtds.true_model import TrueSurvivalModel

SCENARIO = "A"
VERSION = "forest"  # Part 1 rows; Run 1 explains the truth, not the forest
RUN = 1


def main():
    cfg = load_config()
    sim, tool = cfg["simulation"], cfg["tool"]
    driver = cfg["scenarios"][SCENARIO]
    driver_col = f"visit_{driver}"
    grid = time_grid(cfg)
    builder = FlatVisitColumns()
    model = TrueSurvivalModel(sim["h0"], sim["b"], driver_col, grid)

    table_path, log_path = RESULTS_DIR / "table1_run1.csv", RESULTS_DIR / "log_run1.csv"
    start_table(table_path, TABLE1_COLUMNS)
    start_table(log_path, LOG_COLUMNS)

    start = time.perf_counter()
    for rho in sim["rho_levels"]:
        for i in range(cfg["runs"]["run1_n_datasets"]):
            dataset = simulate_dataset(cfg, i, rho, SCENARIO)
            background, background_y, patients = tool_inputs(cfg, dataset, i, builder)
            patient_ids = explained_patients(cfg, i)
            explainer = SurvShapExplainer(background, background_y, grid,
                                          tool["calculation_method"], tool["aggregation_method"])
            rows, log = [], []
            for row_pos, pid in enumerate(patient_ids):
                patient = patients.iloc[[row_pos]].reset_index(drop=True)
                credit = explainer.explain(model, patient)
                s = score_patient(credit, driver_col, cfg["scores"]["tie_rel_tol"])
                key = [VERSION, SCENARIO, rho, i, int(pid), RUN]
                rows.append(key + [driver, credit["visit_1"], credit["visit_2"], credit["visit_3"],
                                   s["top_visit"], s["hit"], s["credit_share"], None])
                if s["n_tied"] > 1 or np.isnan(s["credit_share"]):
                    log.append(key + [s["n_tied"], bool(np.isnan(s["credit_share"]))])
            append_rows(table_path, rows, TABLE1_COLUMNS)
            append_rows(log_path, log, LOG_COLUMNS)
            print(f"rho = {rho}, dataset {i}: done ({time.perf_counter() - start:.0f} s elapsed)", flush=True)

    summarise(pd.read_csv(table_path), pd.read_csv(log_path), driver_col)


def summarise(table, log, driver_col):
    others = [c for c in ("credit_visit_1", "credit_visit_2", "credit_visit_3") if c != f"credit_{driver_col}"]
    print(f"\nRun 1 summary ({len(table)} explained patients; average over patients per dataset, then over datasets)")
    for rho, g in table.groupby("rho"):
        per_ds = g.groupby("dataset_seed")[["hit", "credit_share"]].mean()
        print(f"  rho = {rho}: hit = {per_ds['hit'].mean():.4f}, credit share = {per_ds['credit_share'].mean():.9f}, "
              f"lowest single-patient share = {g['credit_share'].min():.9f}, "
              f"largest non-driver credit = {g[others].to_numpy().max():.2e}, "
              f"driver credit range = {g[f'credit_{driver_col}'].min():.3f} to {g[f'credit_{driver_col}'].max():.3f}")
    print(f"  ties logged: {int((log['n_tied'] > 1).sum())}, NaN shares logged: {int(log['share_is_nan'].sum())}")


if __name__ == "__main__":
    main()
