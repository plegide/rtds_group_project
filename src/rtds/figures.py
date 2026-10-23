"""Static figures for Part 1 (PNG, light background, for the poster and the report)."""

import matplotlib

matplotlib.use("Agg")  # no window; write files only
import matplotlib.pyplot as plt

# Reference palette (dataviz skill, light mode). The first three categorical slots are
# validated for all pairs, including colour-blind vision, so each visit keeps one hue.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
VISIT_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
POINT = TEXT  # single-series charts use ink, not a series hue

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def interval_by_rho(summary, metric, ylabel, title, path, per_ds=None, chance=None, ylim=(0, 1)):
    """One point per rho with its bootstrap interval; optional per-dataset dots and chance line."""
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = range(len(summary))
    if per_ds is not None:
        for x, rho in zip(xs, summary["rho"]):
            vals = per_ds.loc[per_ds["rho"] == rho, metric]
            jitter = [(j % 9 - 4) * 0.025 for j in range(len(vals))]  # fixed offsets, no randomness
            ax.scatter([x + d for d in jitter], vals, s=10, color=TEXT_2, alpha=0.35, linewidths=0, zorder=2)
    ax.errorbar(xs, summary[f"{metric}_mean"],
                yerr=[summary[f"{metric}_mean"] - summary[f"{metric}_lo"], summary[f"{metric}_hi"] - summary[f"{metric}_mean"]],
                fmt="o", color=POINT, ms=8, capsize=4, elinewidth=2, zorder=3)
    for x, m in zip(xs, summary[f"{metric}_mean"]):
        ax.annotate(f"{m:.2f}", (x, m), xytext=(10, 0), textcoords="offset points", va="center", color=TEXT)
    if chance is not None:
        ax.axhline(chance, color=TEXT_2, linestyle="--", linewidth=1.2)
        ax.annotate("chance", (len(summary) - 0.5, chance), xytext=(0, 4), textcoords="offset points",
                    ha="right", color=TEXT_2)
    ax.set_xticks(list(xs), [f"ρ = {r:g}" for r in summary["rho"]])
    ax.set_xlim(-0.5, len(summary) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def per_visit_panels(summary, path, driver_visit):
    """Left: share of SurvSHAP(t) credit per visit (Run 3). Right: C-index drop per visit (Run 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    width = 0.26
    panels = [("share_visit_{k}", "Share of credit (Run 3)", "Mean share of credit"),
              ("drop_visit_{k}", "Reliance (Run 2)", "Drop in held-out C-index")]
    for ax, (pattern, title, ylabel) in zip(axes, panels):
        for j, k in enumerate((1, 2, 3)):
            col = pattern.format(k=k)
            xs = [x + (j - 1) * (width + 0.02) for x in range(len(summary))]  # small surface gap between bars
            mean = summary[f"{col}_mean"]
            ax.bar(xs, mean, width=width, color=VISIT_COLORS[k], label=f"visit {k}" + (" (driver)" if k == driver_visit else ""),
                   zorder=2)
            ax.errorbar(xs, mean, yerr=[mean - summary[f"{col}_lo"], summary[f"{col}_hi"] - mean],
                        fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3, zorder=3)
        ax.set_xticks(range(len(summary)), [f"ρ = {r:g}" for r in summary["rho"]])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=TEXT)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
    # Legend above both panels, so it never covers a bar.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)
