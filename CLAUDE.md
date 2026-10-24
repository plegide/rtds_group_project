# Project: do explanation tools credit the right visit? (RTDS 2026, Stockholm University)

## Role and rules
You are the implementation engineer for a small university research project. I am a master's student and two teammates work with me. The research design is ours and is fixed below. You implement it, explain it in plain language, and flag problems. You do not plan or add experiments, and you do not write report or poster text.

1. I must understand every line of code. Write small files with one job each, comment the reason for non-obvious choices, and after each stage give me a short plain explanation of what each file does and how to run it.
2. Do not add experiments, ablations, sweeps or metrics that are not listed here. The course penalizes many irrelevant experiments. If you think something is missing, propose it in one or two sentences and wait.
3. Never invent a reference, a result or a number. Every number you report must come from a run you actually did. If you cite a paper or tool, it must be something you found and can link.
4. Everything must be reproducible from seeds and one config file. The repository will be public on GitHub and linked from a poster. Course PDFs stay out of git.
5. Work in stages. Stop after each stage, report, and wait for my go-ahead. Commit to git at the end of each stage with a clear message, then push to GitHub (origin).
6. Ask before deleting files, installing system packages, or running anything that will take more than a few minutes.
7. Time is short (poster session on 2 November) and compute is a laptop CPU. Prefer the simplest thing that answers the question.

## The project
Question: when we know which visit truly drives a patient's risk, how often does an explanation tool give the most credit to that visit, and does this get worse when the visits are strongly correlated?
Approach: a simulation with a known answer key, in two parts:
- Part 1 (Stages 0 to 5): a random survival forest that sees the visits as plain columns.
- Part 2 (Stage 6): an LSTM that reads the visits in order, compared with the forest on the same simulated patients. This is where the order of the visits matters, so scenario B is added here. Part 2 starts only after I accept Part 1. If time runs out, Part 2 is the part we cut, so Part 1 must stand on its own.

## Fixed design decisions (do not change without asking)
Simulated data
- Each patient has one measurement at K = 3 visits, stored as 3 columns, standard normal marginals, exchangeable correlation rho in {0, 0.5, 0.9}.
- Follow-up starts after the last visit. The visits are baseline columns.
- Hazard is constant: h(t | x) = h0 * exp(b * x_driver), with h0 = 0.1 and b = ln 2. The survival function is S(t | x) = exp(-h0 * exp(b * x_driver) * t), so event times can be drawn directly.
- Scenario A: visit 3 is the driver. Scenario B: visit 1 is the driver and visit 3 is a correlated bystander. Part 1 uses scenario A only. Part 2 uses both A and B for both models (the forest is run on B too, where it is expected to mirror A).
- Common random numbers: dataset index i uses the same base draws at every correlation level. Generate the driver visit first from its own base normal, then the other two visits conditionally on it (Cholesky factor with the driver ordered first). Then event times are identical across correlation levels and only the other columns change, so settings are truly paired.
- Censoring is exponential and independent of x, with the rate solved once per config to give about 30% censored. No administrative cutoff.
- No control feature.
- Each dataset has 1,000 training and 500 held-out patients.

Forest
- scikit-survival RandomSurvivalForest with 200 trees, max_features = all columns (not the default), min_samples_leaf = 15, fixed seed. Fixed in advance, not tuned.

Tool and credit
- SurvSHAP(t) from the survshap package, default kernel method.
- Time grid: 50 evenly spaced points between the 5th and 80th percentile of observed times in a large reference sample. Fixed per config, identical across datasets and runs, and inside the model's prediction range (assert this at fit time).
- Background: 100 training rows, sampled with a seed.
- Credit per visit: the aggregated_change value with the default "integral" aggregation, on the fixed grid.
- Explained patients: 100 random held-out patients per dataset, chosen by seed before fitting.
- Match output rows by variable_name, never by position.

Scores
- Hit = the visit with the most credit is the true driver. Ties count as 1 divided by the number of visits tied at the maximum, and the tie count is logged. Chance is 1 in 3.
- Credit share = credit of the true driver divided by the total credit over the three visits. If all three credits are 0, record NaN and log it.
- Average over explained patients per dataset, then over 50 re-simulated datasets per setting (refit the model each time).

Three runs
- Run 1 (sanity check only): pass the TRUE survival function to the tool, as a custom predict_survival_function returning sksurv-style step functions. Expected to give the driver all the credit at every correlation level, so it says nothing about correlation effects. Run it on a handful of datasets per correlation level (for example 5), not on all datasets, and check that the other visits get about zero.
- Run 2 (no tool): on the held-out set, shuffle one visit column at a time, 5 shuffles per column, and record two measures from the same shuffles: the drop in Harrell's C-index and the increase in integrated Brier score on the fixed grid. This shows which visit the model relies on. The C-index only measures ranking, while SurvSHAP(t) credit reacts to any change in the survival curve, so the Brier increase is the closer comparison for Run 3 (added 24 October with the user's approval).
- Run 3: SurvSHAP(t) on the fitted model. Compare with the truth and with Run 2. Note: shuffling a correlated column creates the same unrealistic combinations as the tool's masking, so agreement between Runs 2 and 3 is partly expected and is not independent confirmation.
- The only background-size check is done on Run 3, on one dataset, to confirm the setting of 100 rows is reasonable. It is not a result.

Statistics
- Intervals for hit rate and credit share come from a bootstrap over datasets only, never from pooling patients (patients in one dataset share one model).
- Paired tests between correlation levels use the shared datasets (for example Wilcoxon signed-rank). In Part 2, forest versus LSTM is also a paired comparison on the same datasets. State how many comparisons were made.

LSTM (Part 2 only, Stage 6)
- PyTorch on CPU. The input is the 3 visits as a sequence in visit order, shape (patients, 3, 1). One small LSTM layer and a head that outputs a discrete-time hazard for each of the 50 grid intervals, so the survival curve is the cumulative product of (1 - hazard). Trained on the censored likelihood, with early stopping on a validation split taken from the training patients, and a fixed seed.
- It is wrapped so that it accepts the same flat columns as the forest and rearranges them into a sequence inside, and returns sksurv-style step functions, so SurvSHAP(t) runs on it unchanged. This wrapper is untested with the tool: test it first and tell me.
- Do not choose the size, learning rate or epochs yourself. Propose defaults with a one-line reason each and wait for my approval. Fix them after one pilot on a single dataset, and do not tune them per setting.
- Before comparing, check the LSTM learned the driver: held-out C-index close to the forest's and a Run 2 reliance on the driver visit. If it did not, say so before going on.
- Seed check (6e, added 24 October with the user's approval): in scenario A at rho = 0.9, retrain the LSTM with 5 extra training seeds on each of 10 datasets (same data, same settings, a new seed stream added at the end of the list), then Run 2 and Run 3 as in 6d. Question: is the choice between look-alike visits decided by the data or by training randomness? If the favoured visit flips between equally accurate seeds, two equally good models explain the same patients differently.

## What was already checked in a clean environment (2 cores, rough speeds)
- Python 3.12.10 with scikit-survival 0.28.0, survshap 0.4.2, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1. Pin these. (Changed from 3.13.16 at Stage 0: sksurv needs ecos, which has no prebuilt wheel for 3.13, and Windows has no compiler.)
- survshap failed to import until matplotlib was installed separately. Install whatever else the import needs, and pin the full pip freeze output.
- PredictSurvSHAP.fit takes exactly one patient as a one-row DataFrame, so loop over patients. The result has a variable_name column, an aggregated_change column and one column per time point named like "t = 0.5".
- aggregated_change with aggregation_method="integral" is the trapezoid integral of the absolute credit over the time grid. It is never negative.
- RandomSurvivalForest defaults are max_features="sqrt" and min_samples_leaf=3, which is why max_features is set above.
- A step function from the forest raises ValueError if evaluated outside its observed time range.
- Speed: about 5 to 6 seconds per explained patient with a 1,000-row background and about 0.7 seconds with a 100-row background.
- Reported from reading the source, still to verify at Stage 0: kernel is survshap's default calculation_method, it enumerates all coalitions (exact, no random draws; B and random_state matter only for the sampling method), and it fills masked features with background values independently of the patient's other values.

## Stages (stop after each one)
0. Setup and verification. Repository structure, pinned requirements.txt (full freeze), config file, seed handling. Verify the items above marked "still to verify", and test a one-patient run of the tool with a custom survival function. Check: runs from a clean install.
1. Simulator. Check: event and censoring rates match the config; correlation between visits matches the chosen level; event times are identical across correlation levels for the same dataset index; the Kaplan-Meier curve of a large sample matches the closed-form survival curve; a Cox model on a large sample puts the effect on the driver and about zero on the others; same seed gives identical data.
2. Run 1 on a handful of datasets. Check: the driver gets essentially all the credit and the other visits about zero.
3. Fit the forest and Run 2. Check: held-out C-index and integrated Brier score, per-visit reliance (C-index drop and Brier increase), and the time grid lies inside the forest's range. If the model barely uses the true driver, say so before going on.
4. Run 3 on one dataset at each correlation level, plus the one-dataset background-size check. Show per-visit credit for a few patients and the hit rate and credit share. Stop for my review.
5. Main runs for the forest, scenario A: 3 correlation levels x 50 datasets, model refitted each time, results saved after every dataset so a crash loses nothing, parallel where useful.
   Also record, per dataset, the held-out C-index of the true risk score (the driver visit) as a fixed reference for the forest's C-index (added after Stage 3). Then a first analysis of Part 1 (tables, figures, the tests above). Stop for my review. Part 1 must be complete and usable on its own at this point.
6. LSTM (Part 2), only after I accept Part 1. (Part 1 accepted on 23 October. Order agreed then: first the forest on scenario B as stages B1-B5, mirroring Stages 1-5 with --scenario B; then 6a-6d and Stage 7 without stopping between stages, one commit and push per stage. The LSTM settings in 6b were delegated to you on 23 October: choose them, fix them after the pilot, and record each with a one-line reason in CHANGELOG.md. Also decided 23 October: the forest on scenario B is a mirror test only (5 datasets x 3 rho, compared with scenario A relabelled), not 50 datasets; the full main runs in 6d are for the LSTM in scenarios A and B.) 6a: build the wrapper and test it with the tool. 6b: propose the LSTM settings and wait. 6c: pilot on one dataset, check it learned the driver. 6d: main runs for both models, scenarios A and B, 3 correlation levels x 50 datasets. 6e: seed check (see the LSTM section). Stop after each sub-step.
7. Final analysis: tables, figures and the tests above, including forest versus LSTM and scenario A versus B, with both Run 2 measures next to Run 3, and the 6e seed check. Every number in the report must be regenerable by one script.

## Results tables (same columns in every version)
All results live in results/scenario_<X>/ and are committed to git (decided 23 October).
Table 1, one row per explained patient per dataset: version (forest or lstm), scenario, correlation level, dataset seed, patient id, run (1 or 3), true driver visit, credit for visit 1, 2 and 3, top visit, hit, credit share of the true driver, held-out C-index of that dataset (blank for Run 1 rows).
Table 2, one row per visit per dataset (Run 2): version, scenario, correlation level, dataset seed, visit, drop in C-index when that visit is shuffled, increase in integrated Brier score when that visit is shuffled.

## Swappable parts
Keep the input builder (raw data to columns), the model (fit, predict a survival function) and the explainer (one patient in, one credit per visit out) behind small interfaces, so Stage 6 swaps only the model. Do not implement any image version.

## How to report after each stage
What was built, how to run it, the check results with the real numbers from your run, anything surprising, and your questions. No filler.
Also add a section for the stage to CHANGELOG.md: what was added, checked and changed, in plain language.
Commits carry no Claude co-author line and use the git identity already configured on the machine (no overrides).
