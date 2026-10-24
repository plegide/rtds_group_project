"""Stage 7: final analysis. Regenerates every summary number, test and figure from the saved tables.

Sources (all in results/):
- forest, scenario A: scenario_A/*_main.csv (50 datasets per rho)
- forest, scenario B: the forest ignores visit order, and Stages B4 and B5 showed its scenario B
  results equal scenario A's after relabelling the visits. So the forest's B numbers are A's
  numbers relabelled (scenario_B/mirror_check.csv holds the evidence).
- LSTM, scenarios A and B: scenario_<X>/lstm/*_main.csv (50 datasets per rho each)
- Run 1: scenario_<X>/table1_run1.csv
- Stage 6e (LSTM seed check): scenario_A/lstm/seed_check/models.csv and datasets.csv

Statistics: per-dataset averages first; bootstrap intervals over datasets; paired Wilcoxon
signed-rank tests on the shared datasets; raw and Holm-adjusted p-values over all tests.

Writes results/final/summary.csv, tests.csv, run1_summary.csv, analysis_output.txt and
results/final/figures/*.png.
Run from the repository root:  python scripts/stage7_final_analysis.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd

from rtds import seeds
from rtds.config import load_config
from rtds.figures import models_by_rho, per_visit_panels
from rtds.mirror import visit_map
from rtds.results import RESULTS_DIR, scenario_dir
from rtds.stats import bootstrap_ci, holm, paired_wilcoxon, per_dataset

SCORES = ["hit", "credit_share"]  # the two scores that are tested
METRICS = SCORES + ["share_visit_1", "share_visit_2", "share_visit_3", "drop_visit_1", "drop_visit_2",
                    "drop_visit_3", "ibs_visit_1", "ibs_visit_2", "ibs_visit_3",
                    "heldout_c_index", "true_risk_c_index"]


def load_per_dataset(scenario, model):
    """One row per (rho, dataset): scores averaged over patients, Run 2 drops, C-indexes."""
    base = scenario_dir(scenario, model)
    t1 = pd.read_csv(base / "table1_main.csv")
    t2 = pd.read_csv(base / "table2_main.csv")
    ds = pd.read_csv(base / "datasets_main.csv")
    log = pd.read_csv(base / "log_main.csv")
    drops = (t2.pivot_table(index=["rho", "dataset_seed"], columns="visit", values="c_index_drop")
             .rename(columns=lambda k: f"drop_visit_{k}").reset_index())
    brier = (t2.pivot_table(index=["rho", "dataset_seed"], columns="visit", values="ibs_increase")
             .rename(columns=lambda k: f"ibs_visit_{k}").reset_index())
    drops = drops.merge(brier, on=["rho", "dataset_seed"])
    out = (per_dataset(t1).merge(drops, on=["rho", "dataset_seed"])
           .merge(ds[["rho", "dataset_seed", "heldout_c_index", "true_risk_c_index"]], on=["rho", "dataset_seed"]))
    counts = {rho: (int((t1["rho"] == rho).sum()), int(((log["rho"] == rho) & (log["n_tied"] > 1)).sum()),
                    int(((log["rho"] == rho) & log["share_is_nan"]).sum())) for rho in out["rho"].unique()}
    return out, counts


def relabel_from_a(cfg, per_ds_a, scenario):
    """Scenario A's per-dataset values with the visits renamed to `scenario`'s labels."""
    out = per_ds_a.copy()
    for k, ka in visit_map(cfg, scenario).items():
        for kind in ("share", "drop", "ibs"):
            out[f"{kind}_visit_{k}"] = per_ds_a[f"{kind}_visit_{ka}"]
    return out


def main():
    cfg = load_config()
    a = cfg["analysis"]
    rhos = list(cfg["simulation"]["rho_levels"])
    final = RESULTS_DIR / "final"
    figs = final / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    per = {}
    counts = {}
    per[("forest", "A")], counts[("forest", "A")] = load_per_dataset("A", "forest")
    per[("forest", "B")], counts[("forest", "B")] = relabel_from_a(cfg, per[("forest", "A")], "B"), counts[("forest", "A")]
    for scen in ("A", "B"):
        per[("lstm", scen)], counts[("lstm", scen)] = load_per_dataset(scen, "lstm")

    # Summary with bootstrap intervals over datasets
    gen = seeds.rng(cfg["master_seed"], "bootstrap", 1)  # stream 1: stream 0 is the Stage 5 analysis
    rows = []
    for (model, scen), p in per.items():
        for rho in rhos:
            g = p[p["rho"] == rho]
            n_pat, n_ties, n_nan = counts[(model, scen)][rho]
            row = {"model": model, "scenario": scen, "rho": rho, "driver": cfg["scenarios"][scen],
                   "n_datasets": len(g), "n_patients": n_pat, "n_ties": n_ties, "n_nan_shares": n_nan,
                   "source": "relabelled from forest A" if (model, scen) == ("forest", "B") else "main runs"}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(g[m], gen, a["n_bootstrap"], a["ci_level"])
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(final / "summary.csv", index=False)

    # Paired tests on the shared datasets
    tests = []

    def test(family, label, metric, rho, x, y):
        n, ma, mb, med, stat, pv = paired_wilcoxon(x.set_index("dataset_seed")[metric], y.set_index("dataset_seed")[metric])
        tests.append({"family": family, "comparison": label, "metric": metric, "rho": rho, "n_datasets": n,
                      "mean_first": ma, "mean_second": mb, "median_diff_second_minus_first": med,
                      "wilcoxon_stat": stat, "p_value": pv})

    for model, scen in (("forest", "A"), ("lstm", "A"), ("lstm", "B")):  # forest B = forest A, not retested
        p = per[(model, scen)]
        for r1, r2 in itertools.combinations(rhos, 2):
            for m in SCORES:
                test("between rho levels", f"{model} {scen}: rho {r1:g} vs {r2:g}", m, f"{r1:g} vs {r2:g}",
                     p[p["rho"] == r1], p[p["rho"] == r2])
    for scen in ("A", "B"):
        for rho in rhos:
            for m in SCORES:
                f, l = per[("forest", scen)], per[("lstm", scen)]
                test("forest vs LSTM", f"scenario {scen}: forest vs lstm", m, rho, f[f["rho"] == rho], l[l["rho"] == rho])
    for rho in rhos:
        for m in SCORES:
            la, lb = per[("lstm", "A")], per[("lstm", "B")]
            test("scenario A vs B", "lstm: scenario A vs B", m, rho, la[la["rho"] == rho], lb[lb["rho"] == rho])
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p_value"])
    tests.to_csv(final / "tests.csv", index=False)

    # Run 1 (true curve), both scenarios
    run1 = []
    for scen in ("A", "B"):
        t = pd.read_csv(scenario_dir(scen) / "table1_run1.csv")
        for rho, g in t.groupby("rho"):
            run1.append({"scenario": scen, "rho": rho, "n_patients": len(g), "hit": g["hit"].mean(),
                         "credit_share": g["credit_share"].mean(), "lowest_share": g["credit_share"].min()})
    run1 = pd.DataFrame(run1)
    run1.to_csv(final / "run1_summary.csv", index=False)

    # Figures
    models_by_rho(summary, "hit", "Hit rate (driver gets most credit)", "Hit rate of SurvSHAP(t), forest vs LSTM",
                  figs / "fig1_hit_rate.png", chance=1 / 3)
    models_by_rho(summary, "credit_share", "Driver's share of credit", "Driver's credit share, forest vs LSTM",
                  figs / "fig2_credit_share.png")
    for model, scen in (("forest", "A"), ("lstm", "A"), ("lstm", "B")):
        s = summary[(summary["model"] == model) & (summary["scenario"] == scen)].reset_index(drop=True)
        per_visit_panels(s, figs / f"fig3_per_visit_{model}_{scen}.png", cfg["scenarios"][scen])

    # Printed report
    pct = int(a["ci_level"] * 100)
    say(f"Final analysis. Means over datasets with {pct}% bootstrap intervals ({a['n_bootstrap']} resamples of datasets).")
    say("Forest scenario B = forest scenario A relabelled (mirror shown in Stages B4 and B5).")
    for _, r in summary.iterrows():
        f = lambda m, d=3: f"{r[m + '_mean']:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]"
        say(f"\n{r['model']}, scenario {r['scenario']} (driver visit_{r['driver']}), rho = {r['rho']:g}: "
            f"{r['n_datasets']} datasets, {r['n_patients']} patients, ties = {r['n_ties']}, NaN shares = {r['n_nan_shares']}")
        say(f"  hit rate {f('hit')}   credit share {f('credit_share')}")
        say(f"  share per visit: 1 = {f('share_visit_1')}, 2 = {f('share_visit_2')}, 3 = {f('share_visit_3')}")
        say(f"  Run 2 drop: 1 = {f('drop_visit_1', 4)}, 2 = {f('drop_visit_2', 4)}, 3 = {f('drop_visit_3', 4)}")
        say(f"  Run 2 Brier increase: 1 = {f('ibs_visit_1', 4)}, 2 = {f('ibs_visit_2', 4)}, 3 = {f('ibs_visit_3', 4)}")
        say(f"  C-index model {f('heldout_c_index', 4)}, true risk {f('true_risk_c_index', 4)}")
    say(f"\nPaired Wilcoxon signed-rank tests: {len(tests)} comparisons in total "
        f"({', '.join(f'{k}: {v}' for k, v in tests['family'].value_counts(sort=False).items())}). "
        f"Holm-adjusted p-values account for all {len(tests)}.")
    say(tests.drop(columns=["family"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    say("\nRun 1 (true survival curve):")
    say(run1.to_string(index=False, formatters={"rho": "{:g}".format, "hit": "{:.4f}".format,
                                               "credit_share": "{:.9f}".format, "lowest_share": "{:.9f}".format}))
    # Stage 6e (LSTM seeds), read from its saved tables
    sc = cfg["seed_stability"]
    seed_dir = scenario_dir(sc["scenario"], "lstm") / "seed_check"
    models = pd.read_csv(seed_dir / "models.csv")
    ds6e = pd.read_csv(seed_dir / "datasets.csv")
    driver = cfg["scenarios"][sc["scenario"]]
    say(f"\nStage 6e, LSTM seeds (scenario {sc['scenario']}, rho = {sc['rho']}, {len(ds6e)} datasets x "
        f"{int(ds6e['n_models'].iloc[0])} models): the visit Run 3 favours changes between seeds in "
        f"{int(ds6e['run3_flips'].sum())} of {len(ds6e)} datasets; two seeds name the same top visit for "
        f"{ds6e['patient_top_visit_agreement'].mean():.3f} of patients on average; C-index spread within a dataset "
        f"at most {(ds6e['c_index_max'] - ds6e['c_index_min']).max():.4f}; Run 3 favours the driver in "
        f"{int((models['run3_favoured'] == driver).sum())} of {len(models)} models.")
    (final / "analysis_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
