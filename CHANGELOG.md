# Changelog

What each stage adds to the project, in plain language.

## Overview

Stages in plan order. The B stages and the LSTM stages were done on the same day, alternating, so the commit order differs from this list; each stage still has its own commit.

| Stage | Scenario and model | What | Date | Commit |
|---|---|---|---|---|
| 0 | - | Setup, pinned environment, checks on the tool | 22 Oct | 8846b4f, 0d50947 |
| 1 | A, - | Simulator | 22 Oct | 0f71030 |
| 2 | A, true curve | Run 1 | 22 Oct | ee959f5 |
| 3 | A, forest | Forest and Run 2 on one dataset | 22 Oct | 391c55e |
| 4 | A, forest | Run 3 on one dataset, background-size check | 22 Oct | 5a559be |
| 5 | A, forest | Main runs (50 datasets) and first analysis | 22 Oct | 6ce21a6 |
| - | - | Results on GitHub, one folder per scenario | 23 Oct | 9828885 |
| B1 | B, - | Simulator, and "same patients as A" | 23 Oct | 5c0e9c0 |
| B2 | B, true curve | Run 1 | 23 Oct | 132e3ab |
| B3 | B, forest | Forest and Run 2 on one dataset | 23 Oct | 5a28bb9 |
| B4 | B, forest | Run 3 on one dataset, mirror check | 23 Oct | d93adbd |
| B5 | B, forest | Mirror test on 5 datasets | 23 Oct | 63cb6e3 |
| 6a | A, LSTM | LSTM and its wrapper, tested with the tool | 23 Oct | b70132e |
| 6b | - | LSTM settings | 23 Oct | ca8324d |
| 6c | A and B, LSTM | Pilot on one dataset | 23 Oct | 1aacbc6 |
| 6d | A and B, LSTM | Main runs (50 datasets per scenario) | 23 Oct | 334cd88 |
| 6e | A, LSTM | Seed check: same data, different training seeds | 24 Oct | faeddea |
| 7 | A and B, both models | Final analysis | 23 Oct, updated 24 Oct | b631fde, 45e1b9e, faeddea |

**How Run 2's second measure was added (24 October, commit bc8c34e).** Run 2 records the increase in integrated Brier score next to the C-index drop. The stages above report both. For datasets that were already finished, `scripts/run2_refresh.py` rebuilt every model from its seed and reran only Run 2 with the same shuffles: 465 models (forest A, forest B mirror, LSTM A and B), 37 minutes. Every rebuilt model reproduced its saved C-index drops (largest difference 1e-16), so the Brier numbers come from exactly the models behind Run 3, and Run 3 did not need to be redone.

## Part 1: the forest, scenario A (Stages 0 to 5)

Visit 3 drives the risk. Each stage was reviewed before the next one started.

### Stage 0: setup and checks on the explanation tool

**Added**
- A single settings file (`config.toml`) with every number the study uses: the correlation levels, the hazard, the dataset sizes, the forest settings, the explanation settings and one master seed. Changing a setting means editing this file only.
- Seed handling. Every random draw in the project comes from the master seed, split into separate streams by purpose (data, background rows, explained patients, model, shuffles) and by dataset number. The correlation level is deliberately not part of the stream, so dataset 7 at correlation 0 and dataset 7 at correlation 0.9 start from the same random draws. This is what makes the correlation levels a paired comparison.
- Three small interfaces: the input builder, the model and the explainer. The LSTM in Part 2 will replace only the model.
- The true survival curve of the simulated patients, packaged so the explanation tool can use it like a fitted model. Run 1 will use it.
- A wrapper around SurvSHAP(t) that takes one patient and returns one credit per visit, matched by visit name.
- A pinned list of every installed package (`requirements.txt`), so anyone can rebuild the same environment.

**Checked**
- The tool's default method is `kernel`. It tries every combination of visits (8 for 3 visits) and uses no randomness, so the same input always gives the same output.
- When the tool hides a visit, it fills it with the value from a background patient, regardless of the explained patient's other visits.
- The tool's numbers match an exact calculation written out by hand, apart from rounding of about 1 in 10 million.
- When the tool explains the true survival curve for one patient at correlation 0.9, the driver visit gets 99.99999% of the credit, and the other two visits get about 0.0000002 each.
- A fresh install from `requirements.txt` gives the same results.

**Changed from the original plan**
- Python 3.12 instead of 3.13. A package that scikit-survival needs (`ecos`) cannot be installed on Python 3.13 without a C compiler.

### Stage 1: the simulator

**Added**
- The simulator. Each patient gets 3 visit measurements that are correlated with each other at the chosen level (0, 0.5 or 0.9). The time to the event depends only on the driver visit (visit 3 in scenario A). Patients can also be censored, at random and unrelated to their visits, so that about 30% of them have no observed event.
- The driver visit is built first, directly from its own random number, and the other two visits are built around it. Because of this, a dataset has exactly the same driver values, event times and censoring at every correlation level. Only the other two visits change. This makes the three correlation levels a fair paired comparison.
- The censoring rate is calculated exactly from the settings, not estimated by trial. It comes out at 0.0392.
- The fixed time grid: 50 evenly spaced times between the 5th and 80th percentile of observed times in a reference sample of 100,000 patients. It runs from 0.31 to 11.90 and is the same for every dataset and run.
- The input builder for Part 1, which turns each visit into a plain column (`visit_1`, `visit_2`, `visit_3`).

**Checked** (in a sample of 100,000 patients unless stated)
- Censoring is 30.0% (target 30%). Across the 50 training sets it is 30.1% on average, ranging from 27.3% to 33.1%.
- The correlation between visits matches the chosen level to within 0.003 at all three levels.
- For all 50 datasets, the event times, censoring and driver values are identical across the three correlation levels.
- The Kaplan-Meier survival curve is never more than 0.003 away from the true curve on the time grid.
- A Cox model finds an effect of 0.690 to 0.692 for visit 3 (true value 0.693) and at most 0.007 in size for the other visits, at every correlation level.
- Running the same dataset twice gives identical data, and a different dataset number gives different data.
- The Stage 0 checks now use the real simulator instead of their own test data: dataset 0 at correlation 0.9, the real time grid, and the background rows and explained patients chosen by seed (new file `src/rtds/selection.py`). The throwaway test-data file was removed. The results are unchanged: the tool still matches the exact calculation to about 1 in 10 million, and the driver still gets 99.99999% of the credit.

### Stage 2: Run 1, the tool on the true survival curve

**Added**
- Scoring for each explained patient (`src/rtds/scores.py`): the visit with the most credit, the hit (1 if it is the driver; 1/2 or 1/3 if the driver is tied for most credit), and the driver's share of the total credit.
- Credits that differ by less than one part in a million count as tied. Stage 0 showed the tool's own rounding is about one part in ten million, so closer credits cannot be told apart. This setting is in `config.toml`.
- A writer for the results tables (`src/rtds/results.py`), with the fixed Table 1 and Table 2 columns. Tables are saved after every dataset. Ties and undefined shares go to a separate log, because Table 1's columns are fixed.
- The Run 1 script: the tool explains the true survival curve for 100 held-out patients in each of 5 datasets, at each of the 3 correlation levels (1,500 explanations, about 4 minutes).

**Checked**
- At every correlation level the driver had the most credit for all 1,500 patients (hit rate 1.0), with no ties and no undefined shares.
- The driver's share is 0.999999875 for every patient. The other two visits get credits of at most 0.0000005, against 0.16 to 5.5 for the driver.
- The results are exactly the same at all three correlation levels. This is expected: the true curve uses only the driver, and the driver values and background patients are the same at every level.
- The tiny credit for the other visits is always the same fraction of the driver's credit (0.000000096 for visit 1 and 0.000000029 for visit 2). This shows it is the tool's fixed rounding, not a real effect.

### Stage 3: the forest and Run 2

**Added**
- The random survival forest (`src/rtds/forest.py`): 200 trees, all three visits considered at every split, at least 15 patients per leaf. Its seed depends only on the dataset number, so a dataset gets the same forest seed at every correlation level. It checks that the fixed time grid lies inside the range where the forest can predict.
- Model quality on the held-out patients (`src/rtds/metrics.py`): Harrell's C-index (how well the model ranks who has the event first; 0.5 is guessing, 1 is perfect) and the integrated Brier score (average squared error of the predicted survival over the time grid; lower is better).
- Run 2 (`src/rtds/reliance.py`): shuffle one visit at a time in the held-out set, 5 times each, and measure two things from the same shuffles. The drop in C-index shows whether the forest needs the visit to rank patients. The increase in integrated Brier score shows whether it needs the visit to get the survival curve right. The second is the closer comparison with SurvSHAP(t), whose credit reacts to any change in the curve, not only to changes in ranking. A big value means the forest relies on that visit. The shuffles are the same at every correlation level.
- The Stage 3 script, which does all of this on dataset 0 at each correlation level (about 35 seconds per level).

**Checked** (dataset 0)
- The grid (0.31 to 11.90) lies inside the forest's range (0 to 63.38) at every correlation level.
- Held-out C-index: 0.667, 0.665 and 0.675 at correlation 0, 0.5 and 0.9. For comparison, the true risk gives 0.684 on the same patients. Integrated Brier score: 0.187, 0.188 and 0.185.
- Run 2: shuffling the driver (visit 3) drops the C-index by 0.174, 0.168 and 0.164. Shuffling visit 1 or 2 changes it by at most 0.006. So the forest relies on the driver and hardly at all on the other visits, even when they are strongly correlated with it.
- Run 2, Brier score: shuffling the driver makes it worse by 0.067, 0.064 and 0.054; shuffling visit 1 or 2 changes it by at most 0.003. Same picture.

**Fixed along the way**
- The integrated Brier score crashed when a held-out patient's time was later than every training time. Such patients only count as "still at risk" on our grid, so their time is now capped just below the last training time. This does not change the score.

### Stage 4: Run 3, the tool on the forest (one dataset per correlation level)

**Added**
- One shared piece of code (`src/rtds/explain_dataset.py`) that explains a dataset's 100 chosen held-out patients and turns the credits into Table 1 rows. Run 1, Run 3 and the main runs all use it. Rerunning Run 1 with it gave a byte-identical table.
- The Run 3 script: fit the forest on dataset 0 at each correlation level, run Run 2 for comparison, then explain the 100 patients with SurvSHAP(t).
- The background-size check: the same Run 3 on dataset 0 at correlation 0.9, with 50, 100, 200 and 500 background patients. The backgrounds are nested, so each one contains the smaller ones. Its settings are in `config.toml`.

**Checked** (dataset 0)
- Hit rate (how often the driver gets the most credit): 0.96, 0.98 and 0.91 at correlation 0, 0.5 and 0.9.
- The driver's share of the credit: 0.776, 0.742 and 0.700. The other visits get more credit as they become more similar to the driver, even though Run 2 shows the forest hardly uses them.
- One explanation on the forest takes about 2 to 4 seconds.
- Background size, compared with the largest (500 rows): with 100 rows the top visit is the same for 94% of patients, and the driver's share differs by 0.057 on average. The hit rate falls slightly as the background grows: 0.93, 0.91, 0.89 and 0.87 for 50, 100, 200 and 500 rows. Time per patient grows from 0.8 to 8.8 seconds. We keep 100 rows, as designed.

### Stage 5: main runs and first analysis of Part 1

**Added**
- The main runs (`scripts/stage5_main_runs.py`, `src/rtds/run_dataset.py`): 50 datasets at each of the 3 correlation levels. For each dataset the forest is refitted, Run 2 and Run 3 are done, and 100 patients are explained, giving 15,000 explanations. Each dataset is saved on its own as soon as it finishes, and a restart skips finished datasets. Six datasets run in parallel. The full run took 160 minutes.
- For each dataset, the C-index of the true risk score as a fixed yardstick for the forest (requested after Stage 3).
- The analysis (`scripts/stage5_analysis.py`, `src/rtds/stats.py`, `src/rtds/figures.py`): average per dataset, then over datasets. 95% intervals come from resampling the 50 datasets 10,000 times, never the patients. Paired Wilcoxon tests compare correlation levels on the same datasets: 2 scores x 3 pairs of levels = 6 comparisons. Three figures.

**Results** (means over 50 datasets, 95% intervals in brackets)

| Correlation | Hit rate | Driver's credit share | Share for visit 1 | Share for visit 2 | Run 2 drop, visit 3 | Forest C-index | True-risk C-index |
|---|---|---|---|---|---|---|---|
| 0 | 0.906 [0.888, 0.923] | 0.718 [0.706, 0.729] | 0.143 | 0.139 | 0.169 | 0.669 | 0.679 |
| 0.5 | 0.900 [0.883, 0.915] | 0.715 [0.703, 0.726] | 0.147 | 0.139 | 0.168 | 0.669 | 0.679 |
| 0.9 | 0.876 [0.848, 0.900] | 0.672 [0.653, 0.689] | 0.167 | 0.161 | 0.154 | 0.668 | 0.679 |

- No ties and no undefined shares among the 15,000 patients.
- Between correlation 0 and 0.5 nothing changes clearly (hit p = 0.14, share p = 0.33).
- At correlation 0.9 the tool does worse. Compared with correlation 0, the hit rate is 0.03 lower (p = 0.001) and the driver's share is 0.046 lower (p = 4e-9). Compared with 0.5, the share is lower (p = 8e-13) and the hit rate is lower (p = 0.018). With 6 comparisons, the strict Bonferroni threshold is 0.05 / 6 = 0.0083, and only that last hit-rate result (p = 0.018) does not pass it.
- Run 2 shows the forest hardly uses visits 1 and 2 at any level (drops of 0.002 or less). Its reliance on visit 3 also falls a little at 0.9 (0.169 to 0.154). So part of the change at 0.9 may come from the forest itself, not only from the tool.
- The forest's C-index is about 0.01 below the true-risk C-index at every level, so the model is close to the best possible.
- Run 2, Brier score: shuffling the driver makes it worse by 0.069, 0.069 and 0.059. Shuffling visit 1 or 2 changes it by at most 0.0004, and every 95% interval includes 0. So the forest does not use the innocent visits to shape its curves either. The roughly 14% of credit each of them gets at correlation 0 is noise that the tool turns into positive credit. Its credit is an integral of absolute values, so a small wobble always counts as credit and never cancels out.
- Check: dataset 0 in the main runs gives the same hits, shares and C-indexes as Stage 4. 5 of its 300 credit values differ in the last binary digit (4e-16).

## Between Part 1 and Part 2

### Before scenario B: results on GitHub, one folder per scenario

- The `results/` folder is now stored in git (about 5.5 MB), so teammates can use the tables and figures without rerunning anything.
- Each scenario has its own folder: `results/scenario_A/` (everything from Stages 1 to 5, moved there unchanged) and `results/scenario_B/`. Rerunning the scenario A analysis from the new folder gave byte-identical tables.
- The run scripts take `--scenario A` (the default) or `--scenario B`.

## Scenario B with the forest (Stages B1 to B5)

Visit 1 drives the risk. Each B stage repeats the Part 1 stage with the same number, for scenario B. The forest ignores visit order, so these stages check that B mirrors A. They are not new results.

### Stage B1 (Stage 1 for scenario B): the simulator

In scenario B, visit 1 drives the risk and visit 3 is an innocent bystander that is correlated with it. The forest sees the visits as unordered columns, so for the forest B should mirror A. Order starts to matter only for the LSTM in Part 2.

**Checked** (`python scripts/check_simulator.py --scenario B`, output in `results/scenario_B/check_simulator_output.txt`)
- Censoring 30.0%, correlations within 0.003 of each level, and Kaplan-Meier within 0.003 of the true curve. These are the same numbers as in scenario A.
- A Cox model now puts the effect on visit 1 (0.690 to 0.692; true value 0.693) and at most 0.007 in size on visits 2 and 3.
- New check: for all 50 datasets at all 3 correlation levels, the patients' event times, censoring and driver values are identical to scenario A's (visit 1 in B equals visit 3 in A). Scenario A and B therefore use exactly the same patients and can be compared in pairs.

### Stage B2 (Stage 2 for scenario B): Run 1, the tool on the true curve

**Checked** (`python scripts/run1_true_model.py --scenario B`, 1,500 explanations, about 4 minutes)
- Visit 1 (the B driver) got the most credit for every patient at every correlation level: hit rate 1.0, with no ties and no undefined shares.
- The driver's share is 0.999999914 for every patient. The other visits get at most 0.0000004.
- The driver's credits are the same as in scenario A (0.155 to 5.502), because these are the same patients. The tiny leftover differs slightly from A's (0.999999875). It is the tool's fixed rounding error, which depends on which column the driver sits in.

### Stage B3 (Stage 3 for scenario B): the forest and Run 2

**Checked** (`python scripts/stage3_forest_run2.py --scenario B`, dataset 0)
- The forest relies on visit 1, the B driver. Shuffling it drops the C-index by 0.171, 0.166 and 0.162 at correlation 0, 0.5 and 0.9; the other visits change it by at most 0.005.
- The held-out C-index (0.667, 0.665, 0.675) and integrated Brier score (0.187, 0.188, 0.185) are exactly the same as in scenario A. B is the same patients with the visits relabelled, and a forest that considers every visit at every split, with the same seed, builds the same trees. That is the "mirror" the design expects. The Run 2 drops differ slightly from A's because the shuffles are applied in column order, so the driver gets different random shuffles.
- Run 2, Brier score: shuffling visit 1 makes it worse by 0.068, 0.064 and 0.055; the other visits change it by at most 0.002.

### Stage B4 (Stage 4 for scenario B): Run 3 on one dataset, and the mirror check

**Added:** `src/rtds/mirror.py` compares a scenario's results with scenario A's after relabelling the visits. In B, visit 1 is A's visit 3, visit 2 is A's visit 1 and visit 3 is A's visit 2.

**Checked** (`python scripts/stage4_run3.py --scenario B`, dataset 0; output and check in `results/scenario_B/`)
- Hit rate 0.96, 0.98 and 0.91, and driver share 0.776, 0.742 and 0.700 at correlation 0, 0.5 and 0.9: exactly scenario A's numbers.
- Patient by patient (300 patients), B's credits equal A's relabelled credits to within 1 part in 10 million of each patient's total credit, the tool's own rounding. All hits and C-indexes are identical, and the shares agree to 0.00000015.
- So for the forest, which ignores visit order, it makes no difference whether the driver is the first or the last visit.

**Fixed before this stage ran:** the scenario change (commit 9828885) left a broken line in `scripts/stage4_run3.py`, so it could not run. It was fixed in the Stage 6a commit (b70132e), and this stage was run with the fixed script.

### Stage B5 (Stage 5 for scenario B): the forest as a mirror test on 5 datasets

You decided on 23 October that the forest on scenario B is a mirror test only, not 50 datasets.

**Added:** `scripts/stageB5_forest_mirror.py` runs the full main-run pipeline (forest, Run 2, Run 3) for datasets 0 to 4 at every correlation level in scenario B (15 tasks, 6 in parallel, 17 minutes). It then compares every explained patient with scenario A after relabelling the visits. Output is in `results/scenario_B/` (`table1_mirror.csv`, `table2_mirror.csv`, `mirror_check.csv`, `stageB5_mirror_output.txt`).

**Checked** (1,500 patients)
- Hit rate and credit share on these 5 datasets are identical to scenario A's to 4 decimals at every level (for example 0.928 and 0.715 at correlation 0.9).
- Every hit and every held-out C-index is identical.
- In 14 of the 15 dataset-level runs, the credits match A's relabelled credits to within 1 part in 10 million, the tool's own rounding.
- The exception is dataset 1 at correlation 0.9, where the credits differ by up to 0.09% of a patient's total and the shares by up to 0.0006. The forests themselves differ slightly there: their predictions differ for 7 of 500 test patients, by at most 0.0024 in survival probability. The most likely cause is an exact tie between two equally good splits, which the forest breaks by column order. That is a detail of the software, not a difference between the scenarios.
- Conclusion: for the forest it does not matter whether the driver is the first or the last visit. The final analysis uses the forest's scenario A results, relabelled, as its scenario B results.

## Part 2: the LSTM, scenarios A and B (Stage 6)

The LSTM reads the visits in order, so here the two scenarios can differ.

### Stage 6a: the LSTM and its wrapper, tested with the tool

**Added**
- PyTorch, CPU-only build, in `requirements.txt`. The file now points pip to PyTorch's own package index, so a clean install still works.
- `src/rtds/lstm_net.py`: the network. One small LSTM layer reads the three visits in order (visit 1 first, visit 3 last). A head turns the final state into 50 hazards, one per time interval of the grid. The survival curve is the running product of (1 - hazard). It is trained on the censored likelihood: an event in interval k counts the hazard of interval k and survival through all earlier intervals; a censored patient counts survival through the intervals before censoring.
- `src/rtds/lstm_model.py`: the wrapper. It takes the same flat columns as the forest, puts them in visit order by name inside, trains with early stopping on 20% of the training patients, and returns sksurv step functions on the grid. So SurvSHAP(t), Run 2 and the scoring code work on it unchanged. Its risk score for the C-index is the area under its survival curve, with the sign flipped.
- `src/rtds/models.py`: picks the forest or the LSTM by name. The main-run and analysis scripts take `--model forest` (the default) or `--model lstm`. LSTM results go to `results/scenario_<X>/lstm/`.
- `scripts/stage6a_lstm_wrapper_test.py`.

**Checked** (dataset 0, correlation 0.9, scenario A; output in `results/scenario_A/lstm/stage6a_wrapper_test_output.txt`)
- The wrapper reads the visits by name: shuffling the column order gives identical predictions.
- 500 survival curves are proper sksurv step functions, between 0 and 1 and never increasing.
- Two fits with the same seed give identical predictions.
- SurvSHAP(t) runs on the wrapper unchanged and returns one credit per visit.
- The credits add up: at every grid time, the sum of the three visits' SurvSHAP(t) values equals the patient's predicted survival minus the background average, to within 0.00000005.
- One fit takes about 7 seconds and one explanation 0.11 seconds, about 20 times faster than on the forest.

### Stage 6b: the LSTM settings

You delegated the choice on 23 October. The settings are fixed after the pilot (6c) and never tuned per setting. They are in `config.toml` under `[lstm]`.

| Setting | Value | Why |
|---|---|---|
| LSTM layers | 1 | Fixed by the design. |
| Hidden size | 16 | Each patient has only 3 numbers and there are 800 training patients, so a small layer is enough and is less likely to overfit. |
| Learning rate | 0.005 (Adam) | Fast enough for a network this small, and stable with 800 patients. |
| Batch size | 64 | About 13 updates per epoch: enough updates without very noisy steps. |
| Maximum epochs | 300 | Only an upper bound; early stopping decides when to stop. In the pilot it never came close (52 to 92 epochs). |
| Early-stopping patience | 20 epochs | Stops once the validation loss has not improved for 20 epochs, then keeps the best weights. |
| Validation share | 20% of the training patients (200) | Used only for early stopping; the 500 held-out patients are never touched during training. |
| Seed | from the master seed, by dataset number | Same weights, validation split and batch order every run; one thread per model for reproducibility. |

The forest's settings were also fixed in advance, so neither model is tuned.

### Stage 6c: LSTM pilot on one dataset

**Checked** (`python scripts/stage6c_lstm_pilot.py`, dataset 0, both scenarios, all correlation levels, 6 fits in parallel; output in `results/scenario_<X>/lstm/`)

| Scenario | Correlation | LSTM C-index | Forest C-index | LSTM Brier (IBS) | Forest Brier | Run 2 drop, driver | Run 2 drop, other two visits |
|---|---|---|---|---|---|---|---|
| A | 0 | 0.682 | 0.667 | 0.182 | 0.187 | 0.186 | -0.001, -0.001 |
| A | 0.5 | 0.683 | 0.665 | 0.181 | 0.188 | 0.177 | -0.000, 0.000 |
| A | 0.9 | 0.676 | 0.675 | 0.186 | 0.185 | 0.035 | 0.023, 0.014 |
| B | 0 | 0.673 | 0.667 | 0.184 | 0.187 | 0.172 | 0.000, -0.002 |
| B | 0.5 | 0.675 | 0.665 | 0.184 | 0.188 | 0.136 | 0.010, 0.000 |
| B | 0.9 | 0.678 | 0.675 | 0.185 | 0.185 | 0.049 | 0.018, 0.013 |

- The LSTM predicts as well as the forest or slightly better, at every level and in both scenarios. Training stopped after 52 to 92 epochs (about 9 to 11 seconds).
- At correlation 0 and 0.5 the LSTM clearly learned the driver: shuffling it costs 0.14 to 0.19 of C-index, and shuffling the others costs 0.01 or less.
- At correlation 0.9 the LSTM spreads its reliance over the three similar visits. The driver still matters most, but only by a little (0.035 against 0.023 in A; 0.049 against 0.018 in B), whereas the forest relies on the driver by about 0.16. So at 0.9, credit given to visits 1 and 2 partly reflects what this LSTM really uses, not only a mistake by the tool. Run 2 in the main runs measures this for every dataset, so the two can be told apart.
- Run 2, Brier score, tells the same story. At 0 and 0.5 shuffling the driver makes it worse by 0.038 to 0.062, and the others by at most 0.004. At 0.9 the three visits are close: 0.006, 0.004 and 0.009 in A (driver last), and 0.012, 0.006 and 0.003 in B (driver first).
- The settings are kept as they are; nothing was changed after the pilot.

### Stage 6d: LSTM main runs, scenarios A and B

**Done** (`python scripts/stage5_main_runs.py --scenario A --model lstm`, then `--scenario B`, followed by `stage5_analysis.py` for each): 50 datasets x 3 correlation levels per scenario, the LSTM refitted for every dataset, 100 explained patients each, giving 30,000 explanations. Six datasets ran in parallel, about 5 minutes per scenario. Results are in `results/scenario_<X>/lstm/`.

**Results** (means over 50 datasets, 95% intervals in brackets)

| Scenario | Correlation | Hit rate | Driver's share | Run 2 drop: driver | Run 2 drop: other visits | LSTM C-index |
|---|---|---|---|---|---|---|
| A (driver = visit 3, read last) | 0 | 0.938 [0.915, 0.955] | 0.801 | 0.176 | 0.000, -0.000 | 0.675 |
| A | 0.5 | 0.919 [0.881, 0.945] | 0.770 | 0.156 | 0.000, 0.001 | 0.675 |
| A | 0.9 | 0.336 [0.251, 0.423] | 0.323 | 0.031 | visit 1: 0.032, visit 2: 0.014 | 0.671 |
| B (driver = visit 1, read first) | 0 | 0.853 [0.837, 0.869] | 0.691 | 0.170 | -0.001, -0.001 | 0.669 |
| B | 0.5 | 0.777 [0.754, 0.800] | 0.595 | 0.126 | 0.004, 0.001 | 0.668 |
| B | 0.9 | 0.769 [0.749, 0.789] | 0.483 | 0.064 | 0.014, 0.006 | 0.675 |

No ties and no undefined shares among the 30,000 patients. The true-risk C-index is 0.679 throughout.

**Flag: in scenario A at correlation 0.9 the LSTM did not learn the driver specifically.** Shuffling visit 1 hurts it as much as shuffling the true driver, visit 3 (0.032 against 0.031). Its predictions are still good (C-index 0.671), because with visits this similar any of them predicts the risk. SurvSHAP(t)'s hit rate there is 0.336, which is chance level. That mostly reflects a model that does not single out the driver, rather than a tool error, and the results must be read that way.

**What stands out**
- Order matters for the LSTM. With the driver read first (scenario B), the tool gives visit 2 about 21% of the credit even at correlation 0, while shuffling visit 2 does not change the C-index at all. The hit rate is 0.853 in B against 0.938 in A at correlation 0.
- In scenario B the LSTM keeps relying mostly on the driver even at 0.9 (drop 0.064 against at most 0.014 for the others), and the hit rate stays at 0.77.
- Run 2, Brier score, settles the question about visit 2 in scenario B. At correlation 0, shuffling it changes the Brier score by 0.0001, with an interval from -0.0003 to 0.0005, while the tool gives it 21% of the credit. So the LSTM does not use visit 2 for its curve either: that credit comes from the tool and the model's noise, not from real use. At 0.9 in scenario A, the Brier increases for visits 1, 2 and 3 are 0.009, 0.006 and 0.010, confirming that this LSTM spreads its use over the three look-alike visits.
- Within each scenario, the scores get worse as correlation rises (paired Wilcoxon p-values are in `results/scenario_<X>/lstm/analysis/paired_tests.csv`). The only exceptions are the hit rate between 0 and 0.5 in A (p = 0.13) and between 0.5 and 0.9 in B (p = 0.82).

**Fixed:** the analysis printout labelled the C-index line "forest" for every model; it now shows the model's name.

### Stage 6e: does the data or the training seed decide which look-alike visit the LSTM uses?

Added 24 October with your approval. In 6d (scenario A, correlation 0.9) the LSTM relied on visit 1 as much as on the driver, visit 3. This stage asks whether that choice comes from the data or from training randomness.

**Added:** `scripts/stage6e_lstm_seeds.py`. For datasets 0 to 9 it trains 5 extra LSTMs on the same data with the same settings; only the training seed differs. The seeds come from a new stream, `lstm_seed_check`, added at the end of the seed list, so no existing result changes. Each model goes through Run 2 and Run 3 exactly as in 6d. The 6d model of the same dataset is read back from its saved tables as a sixth model. It ran 50 trainings in parallel in 2 minutes. Settings are in `config.toml` under `[seed_stability]`; output is in `results/scenario_A/lstm/seed_check/`.

**Checked** (10 datasets x 6 models)
- In 9 of 10 datasets, the visit the tool favours flips between visit 1 and visit 3 depending only on the training seed.
- The models are almost equally good: within a dataset the C-index differs by at most 0.017 (averages over the 10 datasets: 0.675 to 0.679 per seed).
- On the same data, the hit rate swings with the seed. In dataset 9 it goes from 0.01 to 0.92, and in 8 of the 10 datasets its lowest value is 0.08 or less. The one dataset that does not flip (dataset 7) has the driver favoured by every seed, with hit rates from 0.38 to 0.98.
- For a given patient, two seeds name the same top visit only 38% of the time on average, close to the 1 in 3 expected by chance.
- The tool follows the model faithfully: in 57 of 60 models, the visit Run 3 favours is the visit whose shuffling hurts that model's Brier score most.

**What this means:** when visits are this similar, two equally accurate LSTMs trained on the same patients can explain those patients in opposite ways. The explanation is faithful to each model, but which visit a model relies on is decided by chance during training, not by the data.

## Final analysis (Stage 7)

### Stage 7: final analysis

**Added:** `scripts/stage7_final_analysis.py`, one script that regenerates every summary number, test and figure from the saved tables. Output is in `results/final/`: `summary.csv`, `tests.csv`, `run1_summary.csv`, `analysis_output.txt` and `figures/`.
- The forest's scenario B numbers are scenario A's numbers relabelled, as shown in Stages B4 and B5. The summary marks them with `source = relabelled from forest A`.
- Intervals come from resampling the 50 datasets, never patients.
- Tests are paired Wilcoxon signed-rank tests on the shared datasets: 36 comparisons in total. That is 18 between correlation levels (forest A, LSTM A, LSTM B), 12 forest vs LSTM (each scenario and level), and 6 LSTM scenario A vs B (each level), each for the hit rate and the credit share. Every p-value is also given Holm-adjusted for all 36.
- Figures: hit rate and credit share for forest vs LSTM in both scenarios. For forest A, LSTM A and LSTM B, the per-visit credit (Run 3) is shown next to both Run 2 measures (C-index drop and Brier increase).
- Both Run 2 measures are in `summary.csv` with bootstrap intervals. They are a reading aid and are not tested, so the number of tests stays at 36.
- The output ends with the Stage 6e seed check, read from its saved tables.

**Main results** (hit rate / driver's credit share; means over 50 datasets)

| Correlation | Forest (A and B) | LSTM, scenario A (driver read last) | LSTM, scenario B (driver read first) |
|---|---|---|---|
| 0 | 0.906 / 0.718 | 0.938 / 0.801 | 0.853 / 0.691 |
| 0.5 | 0.900 / 0.715 | 0.919 / 0.770 | 0.777 / 0.595 |
| 0.9 | 0.876 / 0.672 | 0.336 / 0.323 | 0.769 / 0.483 |

- Forest vs LSTM: every comparison differs (Holm-adjusted p at most 0.025). With the driver read last (A), the LSTM's explanations are better than the forest's at 0 and 0.5. With the driver read first (B), they are worse at every level.
- LSTM, scenario A vs B: they differ at every level (Holm-adjusted p at most 7e-7). At 0 and 0.5, A is better. At 0.9, B is better, because in A the LSTM itself stopped singling out the driver (see Stage 6d).
- Between correlation levels, 12 of 18 comparisons stay below 0.05 after Holm adjustment. The 6 that do not:
  - the forest's hit rate and credit share from 0 to 0.5;
  - the forest's hit rate from 0.5 to 0.9 (raw p 0.018);
  - the LSTM's hit rate and credit share from 0 to 0.5 in scenario A (credit share raw p 0.009);
  - the LSTM's hit rate from 0.5 to 0.9 in scenario B.

  So moving from 0 to 0.5 changes little. The clear changes come at 0.9.
- Correction: the first version of this section said 15 of 18. Recounting from `results/final/tests.csv` gives 12.
- Run 1 (true curve) gives the driver all the credit in both scenarios (share 0.999999875 in A, 0.999999914 in B).
- Stage 6e: in 9 of 10 datasets the visit the LSTM favours flips with the training seed (see 6e).
