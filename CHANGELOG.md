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
