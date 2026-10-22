"""Stage 0: explain one patient with SurvSHAP(t) using the TRUE survival function.

The truth depends on the driver visit only, so with independent filling of masked
visits the other two visits must get (numerically) zero credit, even at rho = 0.9.

Run from the repository root:  python scripts/one_patient_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np

from rtds.config import load_config
from rtds.explain_survshap import SurvShapExplainer
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.selection import tool_inputs
from rtds.simulate import simulate_dataset
from rtds.true_model import TrueSurvivalModel

RHO = 0.9  # the hardest correlation level in the design
SCENARIO = "A"
DATASET_INDEX = 0


def main():
    cfg = load_config()
    sim, tool = cfg["simulation"], cfg["tool"]
    dataset = simulate_dataset(cfg, DATASET_INDEX, RHO, SCENARIO)
    background, background_y, patients = tool_inputs(cfg, dataset, DATASET_INDEX, FlatVisitColumns())
    patient = patients.iloc[[0]].reset_index(drop=True)  # the first explained held-out patient
    grid = time_grid(cfg)
    driver_col = f"visit_{cfg['scenarios'][SCENARIO]}"

    model = TrueSurvivalModel(sim["h0"], sim["b"], driver_col, grid)
    explainer = SurvShapExplainer(background, background_y, grid, tool["calculation_method"], tool["aggregation_method"])

    start = time.perf_counter()
    credit = explainer.explain(model, patient)
    elapsed = time.perf_counter() - start

    result = explainer.explain_full(model, patient).set_index("variable_name")
    tcols = [c for c in result.columns if c.startswith("t = ")]

    print(f"rho = {RHO}, driver = {driver_col}, background rows = {len(background)}, "
          f"grid = {len(grid)} points from {grid[0]:.4f} to {grid[-1]:.4f}")
    print(f"patient: " + ", ".join(f"{c} = {patient[c].iloc[0]:.4f}" for c in patient.columns))
    for c, v in credit.items():
        max_t = np.max(np.abs(result.loc[c, tcols].to_numpy(dtype=float)))
        print(f"  {c}: credit = {v:.3e}   max |SurvSHAP(t)| over grid = {max_t:.3e}")
    total = sum(credit.values())
    share = credit[driver_col] / total if total > 0 else float("nan")
    top = max(credit, key=credit.get)
    print(f"top visit = {top}, driver credit share = {share:.12f}")
    print(f"time for one explanation: {elapsed:.3f} s")
    # Non-drivers are zero in theory; survshap's matrix inverse leaves ~1e-7 relative
    # rounding (see verify_survshap.py, check 2b), so "about zero" means below 1e-6.
    assert top == driver_col and share > 1 - 1e-6


if __name__ == "__main__":
    main()
