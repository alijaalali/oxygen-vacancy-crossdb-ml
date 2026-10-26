"""
Figure for the Phase-2d robustness checks: duplicate-composition
reweighting, feature-space distance, and the redox-element-list
sensitivity is reported as a table (no figure needed for that one).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project")
MDIR = ROOT / "15th - OxygenVacancy-CrossDB-ML-project"
FIG = MDIR / "figures"
DATA = ROOT / "data"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BLUE = "#2166ac"
RED = "#b2182b"
GRAY = "#666666"
ORANGE = "#e08214"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


res = json.load(open(DATA / "phase2d_robustness_checks.json"))
dup = res["duplicate_check"]
dist = res["distance_check"]
nn = pd.read_csv(DATA / "nn_distance_source2.csv")


def fig11():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    cats = ["Row-weighted\n(n=795)", "Per-unique-\ncompound (n=182)"]
    r2_overall = [dup["row_weighted_r2"], dup["dedup_r2"]]
    x = np.arange(2)
    w = 0.6
    ax.bar(x, r2_overall, w, color=[GRAY, ORANGE])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("$R^2$, Source 1 $\\to$ Source 2 zero-shot")
    ax.set_title("(a) Row-weighted vs.\nper-compound aggregation")
    for i, v in enumerate(r2_overall):
        ax.text(i, v - 0.3, f"{v:.2f}", ha="center", fontsize=9)

    ax = axes[1]
    logd = np.log10(nn["nn_distance"].clip(lower=0.05))
    bins = np.linspace(logd.min(), logd.max(), 30)
    ax.hist(logd[nn["redox_active"]], bins=bins, alpha=0.6, color=BLUE,
            label=f"Redox-active\n(median={dist['median_dist_redox']:.2f})", density=True)
    ax.hist(logd[~nn["redox_active"]], bins=bins, alpha=0.6, color=RED,
            label=f"Fixed-valence only\n(median={dist['median_dist_fixed']:.2f})", density=True)
    ax.set_xlabel("$\\log_{10}$(distance to nearest Source-1 neighbor)\n(standardized Magpie feature space)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"(b) Feature-space distance by class\n(Mann-Whitney $p$={dist['mannwhitney_p']:.2f}, n.s.)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    colors = np.where(nn["redox_active"], BLUE, RED)
    dist_clipped = nn["nn_distance"].clip(lower=0.05)
    ax.scatter(dist_clipped, nn["abs_error"], s=8, alpha=0.3, c=colors)
    ax.set_xscale("log")
    ax.set_xlim(left=0.05)
    ax.set_xlabel("Distance to nearest Source-1 neighbor (log scale)")
    ax.set_ylabel("$|$Transfer error$|$ (eV/O)")
    ax.set_title(f"(c) Distance vs. error, all compounds\n(Spearman $\\rho$={dist['spearman_dist_vs_error_rho']:.2f}, "
                 f"$p$={dist['spearman_dist_vs_error_p']:.1e})")

    fig.tight_layout()
    save(fig, "fig11_robustness_checks")


if __name__ == "__main__":
    fig11()
    print("PHASE 2D FIGURE DONE")
