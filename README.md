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

## Stage 0 checks

```
python scripts/verify_survshap.py    # survshap's kernel method: default, exact, independent masking
python scripts/one_patient_test.py   # one patient explained with the true survival function
```

## Stage 1 checks

```
python scripts/check_simulator.py    # censoring, correlation, paired times, Kaplan-Meier, Cox, reproducibility (about 1 min)
```
