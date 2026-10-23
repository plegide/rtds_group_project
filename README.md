# Do explanation tools credit the right visit?

A simulation study (RTDS 2026, Stockholm University). The patients are simulated, so we know which visit truly drives the risk. We then check whether SurvSHAP(t) gives that visit the most credit, and whether this gets worse when the visits are strongly correlated.

## Install

Use Python 3.12. scikit-survival needs `ecos`, which has no prebuilt wheel for Python 3.13.

```
py -3.12 -m venv .venv            # on macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\activate            # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

- `config.toml`: every parameter and the master seed.
- `src/rtds/`: the library code (config loading, seeds, interfaces, true model, SurvSHAP wrapper).
- `scripts/`: runnable checks and runs. Start each one from the repository root.
  The run scripts take `--scenario A` (default; visit 3 drives the risk) or `--scenario B` (visit 1 drives it).
- `results/scenario_A/`, `results/scenario_B/`: every table, log and figure, per scenario.

## Stage 0 checks

```
python scripts/verify_survshap.py    # survshap's kernel method: default, exact, independent masking
python scripts/one_patient_test.py   # one patient explained with the true survival function
```

## Stage 1 checks

```
python scripts/check_simulator.py    # censoring, correlation, paired times, Kaplan-Meier, Cox, reproducibility (about 1 min)
```

## Stage 2: Run 1

```
python scripts/run1_true_model.py    # tool on the true survival curve, 5 datasets x 3 rho (about 4 min); writes results/
```

## Stage 3: forest and Run 2

```
python scripts/stage3_forest_run2.py # forest quality and per-visit reliance on dataset 0 at each rho (about 2 min)
```

## Stage 4: Run 3 on one dataset per rho

```
python scripts/stage4_run3.py              # forest + SurvSHAP(t) on dataset 0 at each rho (about 15 min)
python scripts/stage4_background_check.py  # 50 / 100 / 200 / 500 background rows, dataset 0, rho 0.9 (about 20 min)
```

## Stage 5: main runs and analysis

```
python scripts/stage5_main_runs.py   # 50 datasets x 3 rho, 6 parallel workers (about 2.5 hours); safe to restart
python scripts/stage5_analysis.py    # summary table, bootstrap intervals, paired tests, figures in results/
```

## Scenario B (forest) and Part 2 (LSTM)

```
python scripts/check_simulator.py --scenario B        # B1: simulator checks, plus "same patients as A"
python scripts/run1_true_model.py --scenario B        # B2: Run 1
python scripts/stage3_forest_run2.py --scenario B     # B3: forest + Run 2 on dataset 0
python scripts/stage4_run3.py --scenario B            # B4: Run 3 on dataset 0 (3 rho in parallel)
python scripts/stageB5_forest_mirror.py               # B5: forest on B for 5 datasets, compared with A
python scripts/stage6a_lstm_wrapper_test.py           # 6a: LSTM wrapper tested with SurvSHAP(t)
python scripts/stage6c_lstm_pilot.py                  # 6c: LSTM pilot on dataset 0, both scenarios
python scripts/stage5_main_runs.py --scenario A --model lstm   # 6d: LSTM main runs (about 5 min each)
python scripts/stage5_main_runs.py --scenario B --model lstm
python scripts/stage5_analysis.py --scenario A --model lstm
python scripts/stage5_analysis.py --scenario B --model lstm
python scripts/stage7_final_analysis.py               # 7: every summary number, test and figure -> results/final/
```
