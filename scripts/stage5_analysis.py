"""Stage 5: analysis of the main runs for one scenario and one model.

Reads the merged main-run tables, averages over patients within each dataset, then
summarises over datasets per rho with bootstrap intervals (datasets resampled, never
patients), runs paired Wilcoxon tests between rho levels, and draws the figures.

Writes analysis/summary_by_rho.csv, analysis/paired_tests.csv, analysis/per_dataset.csv and
figures/*.png in the results folder of that scenario and model.
Run from the repository root after stage5_main_runs.py:  python scripts/stage5_analysis.py [--scenario B] [--model lstm]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd

from rtds import seeds
from rtds.cli import parse_scenario_and_model
from rtds.config import load_config
from rtds.figures import interval_by_rho, per_visit_panels
from rtds.results import scenario_dir
from rtds.stats import bootstrap_ci, paired_tests, per_dataset

TESTED = ["hit", "credit_share"]  # the two scores compared between rho levels


def main():
    scenario, version = parse_scenario_and_model(__doc__)
    cfg = load_config()
    base = scenario_dir(scenario, version)
    driver = cfg["scenarios"][scenario]
    a = cfg["analysis"]
    rhos = list(cfg["simulation"]["rho_levels"])
    table1 = pd.read_csv(base / "table1_main.csv")
    table2 = pd.read_csv(base / "table2_main.csv")
    log = pd.read_csv(base / "log_main.csv")
    datasets = pd.read_csv(base / "datasets_main.csv")

    per_ds = per_dataset(table1)
    drops = (table2.pivot_table(index=["rho", "dataset_seed"], columns="visit", values="c_index_drop")
             .rename(columns=lambda k: f"drop_visit_{k}").reset_index())
    brier = (table2.pivot_table(index=["rho", "dataset_seed"], columns="visit", values="ibs_increase")
             .rename(columns=lambda k: f"ibs_visit_{k}").reset_index())
    drops = drops.merge(brier, on=["rho", "dataset_seed"])
    per_ds = (per_ds.merge(drops, on=["rho", "dataset_seed"])
              .merge(datasets[["rho", "dataset_seed", "heldout_c_index", "true_risk_c_index"]], on=["rho", "dataset_seed"]))

    metrics = ["hit", "credit_share", "share_visit_1", "share_visit_2", "share_visit_3",
               "drop_visit_1", "drop_visit_2", "drop_visit_3", "ibs_visit_1", "ibs_visit_2", "ibs_visit_3",
               "heldout_c_index", "true_risk_c_index"]
    gen = seeds.rng(cfg["master_seed"], "bootstrap", 0)
    summary = []
    for rho in rhos:
        g = per_ds[per_ds["rho"] == rho]
        row = {"rho": rho, "n_datasets": len(g), "n_patients": int((table1["rho"] == rho).sum()),
               "n_ties": int(((log["rho"] == rho) & (log["n_tied"] > 1)).sum()),
               "n_nan_shares": int(((log["rho"] == rho) & log["share_is_nan"]).sum())}
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(g[m], gen, a["n_bootstrap"], a["ci_level"])
        summary.append(row)
    summary = pd.DataFrame(summary)
    tests = paired_tests(per_ds, TESTED, rhos)

    out = base / "analysis"
    figs = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary_by_rho.csv", index=False)
    tests.to_csv(out / "paired_tests.csv", index=False)
    per_ds.to_csv(out / "per_dataset.csv", index=False)

    interval_by_rho(summary, "hit", "Hit rate (driver gets most credit)",
                    f"Hit rate ({version}, scenario {scenario})", figs / "fig1_hit_rate.png",
                    per_ds=per_ds, chance=1 / 3)
    interval_by_rho(summary, "credit_share", "Driver's share of credit",
                    f"Driver's credit share ({version}, scenario {scenario})", figs / "fig2_credit_share.png",
                    per_ds=per_ds)
    per_visit_panels(summary, figs / "fig3_per_visit.png", driver)

    pct = int(a["ci_level"] * 100)
    print(f"{version}, scenario {scenario} (driver visit_{driver}). Means over datasets with {pct}% bootstrap intervals "
          f"({a['n_bootstrap']} resamples of datasets).")
    for _, r in summary.iterrows():
        f = lambda m, d=3: f"{r[m + '_mean']:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]"
        print(f"\nrho = {r['rho']:g}: {int(r['n_datasets'])} datasets, {int(r['n_patients'])} patients, "
              f"ties = {int(r['n_ties'])}, NaN shares = {int(r['n_nan_shares'])}")
        print(f"  hit rate      {f('hit')}")
        print(f"  credit share  {f('credit_share')}")
        print(f"  share per visit: 1 = {f('share_visit_1')}, 2 = {f('share_visit_2')}, 3 = {f('share_visit_3')}")
        print(f"  Run 2 drop:      1 = {f('drop_visit_1', 4)}, 2 = {f('drop_visit_2', 4)}, 3 = {f('drop_visit_3', 4)}")
        print(f"  Run 2 Brier up:  1 = {f('ibs_visit_1', 4)}, 2 = {f('ibs_visit_2', 4)}, 3 = {f('ibs_visit_3', 4)}")
        print(f"  C-index {version} {f('heldout_c_index', 4)}, true risk {f('true_risk_c_index', 4)}")

    print(f"\nPaired Wilcoxon signed-rank tests between rho levels, pairing on dataset: "
          f"{len(tests)} comparisons ({len(TESTED)} scores x {len(tests) // len(TESTED)} pairs of rho levels).")
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
