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
