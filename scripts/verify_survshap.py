"""Stage 0: check what CLAUDE.md lists as "still to verify" about survshap's kernel method.

1. kernel is the default calculation_method (and integral the default aggregation).
2. It is exact and has no random draws: B, random_state and the global numpy seed
   do not change the result, and it matches a brute-force exact Shapley calculation.
3. Masked visits are filled with background values independently of the patient's
   other values: every input the model sees is "patient value or the background
   row's own value", column by column.

Run from the repository root:  python scripts/verify_survshap.py
"""

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
from survshap import PredictSurvSHAP, SurvivalModelExplainer
from scipy.integrate import trapezoid
from sksurv.functions import StepFunction

from rtds.config import load_config
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.selection import tool_inputs
from rtds.simulate import simulate_dataset

RHO = 0.9  # the strongest correlation in the design, where conditional filling would differ most
SCENARIO = "A"
DATASET_INDEX = 0


class InteractionTestModel:
    """Test fixture only: all three visits matter and two interact, so no Shapley value is trivially 0."""

    def __init__(self, h0, grid):
        self.h0 = h0
        self.grid = grid
        self.inputs_seen = []  # every matrix survshap asks us to predict on

    def predict_survival_function(self, X):
        self.inputs_seen.append(X.to_numpy().copy())
        lp = 0.5 * X["visit_1"] - 0.3 * X["visit_2"] + 0.7 * X["visit_3"] + 0.4 * X["visit_1"] * X["visit_3"]
        surv = np.exp(-np.outer(self.h0 * np.exp(lp.to_numpy()), self.grid))
        return np.array([StepFunction(self.grid, s) for s in surv])


def run_survshap(model, background, background_y, patient, grid, global_seed, **kwargs):
    np.random.seed(global_seed)  # if anything inside drew from the global RNG, this would change results
    explainer = SurvivalModelExplainer(
        model, background, background_y, predict_survival_function=lambda m, d: m.predict_survival_function(d)
    )
    shap = PredictSurvSHAP(**kwargs)
    shap.fit(explainer, patient, timestamps=grid)
    return shap.result.set_index("variable_name")


def time_columns(result):
    return [c for c in result.columns if c.startswith("t = ")]


def brute_force_shapley(model, background, patient, grid):
    """Exact interventional Shapley values per time point, written out from the definition.

    v(S) = mean over background rows of S(t | patient's values on S, the row's own values elsewhere).
    phi_j = sum over S not containing j of |S|!(p-|S|-1)!/p! * (v(S + j) - v(S)).
    """
    cols = list(background.columns)
    p = len(cols)

    def value(coalition):
        X = background.copy()
        for c in coalition:
            X[c] = patient[c].iloc[0]
        return np.mean([f(grid) for f in model.predict_survival_function(X)], axis=0)

    phi = {}
    for j in cols:
        others = [c for c in cols if c != j]
        total = np.zeros(len(grid))
        for size in range(p):
            w = math.factorial(size) * math.factorial(p - size - 1) / math.factorial(p)
            for S in itertools.combinations(others, size):
                total += w * (value(S + (j,)) - value(S))
        phi[j] = total
    return phi


def main():
    cfg = load_config()
    dataset = simulate_dataset(cfg, DATASET_INDEX, RHO, SCENARIO)
    background, background_y, patients = tool_inputs(cfg, dataset, DATASET_INDEX, FlatVisitColumns())
    patient = patients.iloc[[0]].reset_index(drop=True)  # the first explained held-out patient
    grid = time_grid(cfg)
    model = InteractionTestModel(cfg["simulation"]["h0"], grid)
    cols = list(background.columns)

    # 1. Defaults
    defaults = PredictSurvSHAP()
    print(f"[1] default calculation_method = {defaults.calculation_method!r}, "
          f"aggregation_method = {defaults.aggregation_method!r}")
    assert defaults.calculation_method == "kernel" and defaults.aggregation_method == "integral"

    # 2a. No randomness: change B, random_state and the global numpy seed
    runs = [
        run_survshap(model, background, background_y, patient, grid, global_seed=1),
        run_survshap(model, background, background_y, patient, grid, global_seed=999, B=5, random_state=0),
        run_survshap(model, background, background_y, patient, grid, global_seed=7, B=200, random_state=12345),
    ]
    numeric = ["aggregated_change"] + time_columns(runs[0])
    identical = all(np.array_equal(r.loc[cols, numeric].to_numpy(), runs[0].loc[cols, numeric].to_numpy())
                    for r in runs[1:])
    print(f"[2a] three runs with different B / random_state / global seed bit-identical: {identical}")
    assert identical

    # 3. Masking: inspect every matrix survshap handed to the model during one explanation
    model.inputs_seen.clear()
    run_survshap(model, background, background_y, patient, grid, global_seed=1)
    bg = background.to_numpy()
    pt = patient.to_numpy()[0]
    masks = []
    for M in model.inputs_seen:
        if M.shape[0] != len(bg):
            continue  # the single-row call that predicts the patient itself
        from_patient = [bool(np.all(M[:, j] == pt[j])) for j in range(len(cols))]
        from_background = [bool(np.array_equal(M[:, j], bg[:, j])) for j in range(len(cols))]
        # Each column must be exactly one of the two: the patient's value in every row,
        # or the background row's own value (same row, untouched, whatever the patient has).
        assert all(a != b for a, b in zip(from_patient, from_background)), "column filled some other way"
        masks.append(tuple(int(a) for a in from_patient))
    coalitions = sorted(set(masks))
    print(f"[3] model calls: {len(model.inputs_seen)} in total, {len(masks)} on background-sized inputs; "
          f"distinct coalitions seen: {len(coalitions)} of {2 ** len(cols)}")
    print(f"    coalitions (1 = patient's value, 0 = background row's own value): {coalitions}")
    assert len(coalitions) == 2 ** len(cols)

    # 2b. Exact: compare with Shapley values computed straight from the definition
    exact = brute_force_shapley(model, background, patient, grid)
    result = runs[0]
    tcols = time_columns(result)
    max_diff = max(np.max(np.abs(result.loc[c, tcols].to_numpy(dtype=float) - exact[c])) for c in cols)
    max_phi = max(np.max(np.abs(exact[c])) for c in cols)
    agg_diff = max(abs(result.loc[c, "aggregated_change"] - trapezoid(np.abs(exact[c]), grid)) for c in cols)
    print(f"[2b] max |survshap - brute-force exact| over visits and times = {max_diff:.3e} "
          f"(largest |SurvSHAP(t)| = {max_phi:.3e})")
    print(f"     max |aggregated_change - trapezoid(|exact|)| = {agg_diff:.3e}")
    for c in cols:
        print(f"     {c}: aggregated_change survshap = {result.loc[c, 'aggregated_change']:.10f}, "
              f"exact = {trapezoid(np.abs(exact[c]), grid):.10f}")
    # Not bit-exact: survshap gives the empty and full coalitions weight 1e9 and solves the
    # weighted least squares with an explicit matrix inverse (condition number ~4.5e9), which
    # costs about 7 digits. Solving the same system with np.linalg.lstsq gave ~7e-12, so the
    # method is exact and the gap is float rounding. Hence a relative tolerance.
    assert max_diff < 1e-6 * max_phi

    print("All survshap checks passed.")


if __name__ == "__main__":
    main()
