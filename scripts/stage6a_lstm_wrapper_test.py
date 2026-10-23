"""Stage 6a: test the LSTM wrapper with SurvSHAP(t) before using it for anything.

1. It reads the visits by name: shuffling the DataFrame's column order changes nothing.
2. Its survival curves are sksurv StepFunctions on the grid, between 0 and 1, never increasing.
3. Same seed, same fit: two fits give identical predictions.
4. SurvSHAP(t) runs on it unchanged and returns one credit per visit, matched by name.
5. The credits add up (Shapley efficiency): for every grid time, the sum of the visits'
   SurvSHAP(t) values equals the patient's predicted survival minus the background average.
6. Time per explanation.

Uses dataset 0, rho = 0.9, scenario A. Run from the repository root:
python scripts/stage6a_lstm_wrapper_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
from sksurv.functions import StepFunction

from rtds.config import load_config
from rtds.explain_survshap import SurvShapExplainer
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.lstm_model import LSTMModel
from rtds.selection import tool_inputs
from rtds.simulate import simulate_dataset

RHO, SCENARIO, DATASET_INDEX = 0.9, "A", 0


def main():
    cfg = load_config()
    grid = time_grid(cfg)
    builder = FlatVisitColumns()
    ds = simulate_dataset(cfg, DATASET_INDEX, RHO, SCENARIO)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)

    start = time.perf_counter()
    model = LSTMModel(cfg, DATASET_INDEX, grid).fit(X_train, ds.train.y)
    print(f"fit: {time.perf_counter() - start:.1f} s, stopped after {len(model.history)} epochs "
          f"(best epoch {model.best_epoch})")

    # 1. Column order
    shuffled = X_test[["visit_3", "visit_1", "visit_2"]]
    same_order = np.array_equal(model.predict_risk(X_test), model.predict_risk(shuffled))
    print(f"[1] predictions identical after reordering the columns: {same_order}")
    assert same_order

    # 2. Survival curves
    funcs = model.predict_survival_function(X_test)
    vals = np.array([f(grid) for f in funcs])
    ok = (all(isinstance(f, StepFunction) for f in funcs) and vals.min() >= 0 and vals.max() <= 1
          and np.all(np.diff(vals, axis=1) <= 0))
    print(f"[2] {len(funcs)} StepFunctions, values in [{vals.min():.4f}, {vals.max():.4f}], never increasing: {ok}")
    assert ok

    # 3. Reproducible fit
    again = LSTMModel(cfg, DATASET_INDEX, grid).fit(X_train, ds.train.y)
    same_fit = np.array_equal(model.predict_risk(X_test), again.predict_risk(X_test))
    print(f"[3] two fits with the same seed give identical predictions: {same_fit}")
    assert same_fit

    # 4 + 5. SurvSHAP(t) on the wrapper, and the credits add up
    background, background_y, patients = tool_inputs(cfg, ds, DATASET_INDEX, builder)
    explainer = SurvShapExplainer(background, background_y, grid,
                                  cfg["tool"]["calculation_method"], cfg["tool"]["aggregation_method"])
    patient = patients.iloc[[0]].reset_index(drop=True)
    start = time.perf_counter()
    credit = explainer.explain(model, patient)
    elapsed = time.perf_counter() - start
    result = explainer.explain_full(model, patient).set_index("variable_name")
    tcols = [c for c in result.columns if c.startswith("t = ")]
    phi_sum = result.loc[list(background.columns), tcols].to_numpy(dtype=float).sum(axis=0)
    target = model._survival(patient)[0] - model._survival(background).mean(axis=0)
    gap = np.max(np.abs(phi_sum - target))
    print(f"[4] SurvSHAP(t) ran; credit per visit: " + ", ".join(f"{k} = {v:.4f}" for k, v in credit.items()))
    print(f"[5] max |sum of SurvSHAP(t) - (patient survival - background mean)| over the grid = {gap:.2e}")
    assert set(credit) == {"visit_1", "visit_2", "visit_3"} and gap < 1e-6

    # 6. Speed
    print(f"[6] one explanation: {elapsed:.3f} s")
    print("All LSTM wrapper checks passed.")


if __name__ == "__main__":
    main()
