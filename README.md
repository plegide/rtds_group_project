# Do explanation tools credit the right visit?

A simulation study (RTDS 2026, Stockholm University). The patients are simulated, so we know which visit truly drives the risk. We then check whether SurvSHAP(t) gives that visit the most credit, and whether this gets worse when the visits are strongly correlated.

- **Scenario A:** visit 3 drives the risk (the driver is the last visit).
- **Scenario B:** visit 1 drives the risk (the driver is the first visit). The patients are the same as in A with the visits relabelled.
- **Part 1 (Stages 0 to 5):** a random survival forest, which sees the visits as unordered columns.
- **Part 2 (Stages 6 and 7):** an LSTM, which reads the visits in order, compared with the forest.

What changed at each stage, and why, is in [CHANGELOG.md](CHANGELOG.md).

## Install

Use Python 3.12. scikit-survival needs `ecos`, which has no prebuilt wheel for Python 3.13.

```
py -3.12 -m venv .venv            # on macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\activate            # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # includes PyTorch's CPU-only build
```

## Layout

- `config.toml`: every parameter and the master seed.
- `src/rtds/`: the library code (simulator, models, explainer, scores, statistics, figures).
- `scripts/`: runnable checks and runs. Start each one from the repository root.
  - `--scenario A` (the default) or `--scenario B` picks the scenario.
  - `--model forest` (the default) or `--model lstm` picks the model, for the main runs and their analysis.
- `results/scenario_A/` and `results/scenario_B/`: every table, log and figure per scenario. Forest results sit at the top of each folder, LSTM results in its `lstm/` subfolder.
- `results/final/`: the final analysis (Stage 7).

Times below are on a laptop with 8 cores.

## Stage 0: setup checks

```
python scripts/verify_survshap.py    # survshap's kernel method: default, exact, independent masking
python scripts/one_patient_test.py   # one patient explained with the true survival function
```

## Stage 1: simulator

Scenario A:
```
python scripts/check_simulator.py                 # censoring, correlation, paired times, Kaplan-Meier, Cox, reproducibility (about 1 min)
```

Scenario B (B1):
```
python scripts/check_simulator.py --scenario B    # the same checks, plus "same patients as scenario A"
```

## Stage 2: Run 1, the tool on the true survival curve

Scenario A:
```
python scripts/run1_true_model.py                 # 5 datasets x 3 rho (about 4 min)
```

Scenario B (B2):
```
python scripts/run1_true_model.py --scenario B
```

Output: `results/scenario_<X>/table1_run1.csv`.

## Stage 3: forest and Run 2 on one dataset

Scenario A:
```
python scripts/stage3_forest_run2.py              # forest quality and per-visit reliance on dataset 0 at each rho (about 2 min)
```

Scenario B (B3):
```
python scripts/stage3_forest_run2.py --scenario B
```

Output: `results/scenario_<X>/table2_stage3.csv`.

## Stage 4: Run 3, the tool on the forest, one dataset

Scenario A:
```
python scripts/stage4_run3.py                     # dataset 0, the 3 rho levels in parallel
python scripts/stage4_background_check.py         # 50 / 100 / 200 / 500 background rows, dataset 0, rho 0.9 (about 20 min); scenario A only
```

Scenario B (B4):
```
python scripts/stage4_run3.py --scenario B
```

Output: `results/scenario_<X>/table1_stage4.csv` and `table2_stage4.csv`.

## Stage 5: forest main runs and analysis

Scenario A (all 50 datasets):
```
python scripts/stage5_main_runs.py                # 50 datasets x 3 rho, 6 parallel workers (about 2.5 hours); safe to restart
python scripts/stage5_analysis.py                 # summary, bootstrap intervals, paired tests, figures
```

Scenario B (B5, a mirror test on 5 datasets: the forest ignores visit order, so B should equal A relabelled):
```
python scripts/stageB5_forest_mirror.py           # datasets 0-4 x 3 rho, compared patient by patient with scenario A (about 17 min)
```

Output: `results/scenario_A/*_main.csv`, `analysis/` and `figures/`; for B, `results/scenario_B/mirror_check.csv`.

## Stage 6: LSTM

6a, wrapper tested with the tool:
```
python scripts/stage6a_lstm_wrapper_test.py       # dataset 0, rho 0.9, scenario A
```

6b, settings: nothing to run. They are in `config.toml` under `[lstm]`, with the reasons in CHANGELOG.md.

6c, pilot on one dataset (both scenarios in one run):
```
python scripts/stage6c_lstm_pilot.py              # dataset 0, 3 rho x 2 scenarios, 6 fits in parallel
```

6d, main runs. Scenario A:
```
python scripts/stage5_main_runs.py --scenario A --model lstm    # 50 datasets x 3 rho (about 5 min)
python scripts/stage5_analysis.py --scenario A --model lstm
```

6d, main runs. Scenario B:
```
python scripts/stage5_main_runs.py --scenario B --model lstm
python scripts/stage5_analysis.py --scenario B --model lstm
```

Output: `results/scenario_<X>/lstm/`.

## Stage 7: final analysis (forest vs LSTM, scenario A vs B)

```
python scripts/stage7_final_analysis.py           # every summary number, test and figure -> results/final/
```
