# Changelog

What each stage adds to the project, in plain language.

## Stage 0: setup and checks on the explanation tool

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

## Stage 1: the simulator

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

## Stage 2: Run 1, the tool on the true survival curve

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

## Stage 3: the forest and Run 2

**Added**
- The random survival forest (`src/rtds/forest.py`): 200 trees, all three visits considered at every split, at least 15 patients per leaf. Its seed depends only on the dataset number, so a dataset gets the same forest seed at every correlation level. It checks that the fixed time grid lies inside the range where the forest can predict.
- Model quality on the held-out patients (`src/rtds/metrics.py`): Harrell's C-index (how well the model ranks who has the event first; 0.5 is guessing, 1 is perfect) and the integrated Brier score (average squared error of the predicted survival over the time grid; lower is better).
- Run 2 (`src/rtds/reliance.py`): shuffle one visit at a time in the held-out set, 5 times each, and measure how much the C-index drops. A big drop means the forest relies on that visit. The shuffles are the same at every correlation level.
- The Stage 3 script, which does all of this on dataset 0 at each correlation level (about 35 seconds per level).

**Checked** (dataset 0)
- The grid (0.31 to 11.90) lies inside the forest's range (0 to 63.38) at every correlation level.
- Held-out C-index: 0.667, 0.665 and 0.675 at correlation 0, 0.5 and 0.9. For comparison, the true risk gives 0.684 on the same patients. Integrated Brier score: 0.187, 0.188 and 0.185.
- Run 2: shuffling the driver (visit 3) drops the C-index by 0.174, 0.168 and 0.164. Shuffling visit 1 or 2 changes it by at most 0.006. So the forest relies on the driver and hardly at all on the other visits, even when they are strongly correlated with it.

**Fixed along the way**
- The integrated Brier score crashed when a held-out patient's time was later than every training time. Such patients only count as "still at risk" on our grid, so their time is now capped just below the last training time. This does not change the score.

## Stage 4: Run 3, the tool on the forest (one dataset per correlation level)

**Added**
- One shared piece of code (`src/rtds/explain_dataset.py`) that explains a dataset's 100 chosen held-out patients and turns the credits into Table 1 rows. Run 1, Run 3 and the main runs all use it. Rerunning Run 1 with it gave a byte-identical table.
- The Run 3 script: fit the forest on dataset 0 at each correlation level, run Run 2 for comparison, then explain the 100 patients with SurvSHAP(t).
- The background-size check: the same Run 3 on dataset 0 at correlation 0.9, with 50, 100, 200 and 500 background patients. The backgrounds are nested, so each one contains the smaller ones. Its settings are in `config.toml`.

**Checked** (dataset 0)
- Hit rate (how often the driver gets the most credit): 0.96, 0.98 and 0.91 at correlation 0, 0.5 and 0.9.
- The driver's share of the credit: 0.776, 0.742 and 0.700. The other visits get more credit as they become more similar to the driver, even though Run 2 shows the forest hardly uses them.
- One explanation on the forest takes about 2 to 4 seconds.
- Background size, compared with the largest (500 rows): with 100 rows the top visit is the same for 94% of patients, and the driver's share differs by 0.057 on average. The hit rate falls slightly as the background grows: 0.93, 0.91, 0.89 and 0.87 for 50, 100, 200 and 500 rows. Time per patient grows from 0.8 to 8.8 seconds. We keep 100 rows, as designed.

## Stage 5: main runs and first analysis of Part 1

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
- Check: dataset 0 in the main runs gives the same hits, shares and C-indexes as Stage 4. 5 of its 300 credit values differ in the last binary digit (4e-16).

## Before scenario B: results on GitHub, one folder per scenario

- The `results/` folder is now stored in git (about 5.5 MB), so teammates can use the tables and figures without rerunning anything.
- Each scenario has its own folder: `results/scenario_A/` (everything from Stages 1 to 5, moved there unchanged) and `results/scenario_B/`. Rerunning the scenario A analysis from the new folder gave byte-identical tables.
- The run scripts take `--scenario A` (the default) or `--scenario B`.

## Stage B1: the simulator for scenario B

In scenario B, visit 1 drives the risk and visit 3 is an innocent bystander that is correlated with it. The forest sees the visits as unordered columns, so for the forest B should mirror A. Order starts to matter only for the LSTM in Part 2.

**Checked** (`python scripts/check_simulator.py --scenario B`, output in `results/scenario_B/check_simulator_output.txt`)
- Censoring 30.0%, correlations within 0.003 of each level, and Kaplan-Meier within 0.003 of the true curve. These are the same numbers as in scenario A.
- A Cox model now puts the effect on visit 1 (0.690 to 0.692; true value 0.693) and at most 0.007 in size on visits 2 and 3.
- New check: for all 50 datasets at all 3 correlation levels, the patients' event times, censoring and driver values are identical to scenario A's (visit 1 in B equals visit 3 in A). Scenario A and B therefore use exactly the same patients and can be compared in pairs.
