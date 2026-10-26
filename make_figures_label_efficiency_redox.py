"""
Figures for the Phase-2b additions: few-shot fine-tuning / label-efficiency
curve, and the redox-active-cation transfer-failure diagnosis.
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
GREEN = "#1a9850"
GRAY = "#666666"
ORANGE = "#e08214"
PURPLE = "#762a83"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


res = json.load(open(DATA / "phase2b_results.json"))
ft = res["fine_tuning"]
rx = res["redox_diagnosis"]
raw_curve = pd.read_csv(DATA / "finetuning_curve_raw.csv")
redox_df = pd.read_csv(DATA / "redox_diagnosis_source2.csv")


# ---------- Figure 7: fine-tuning / label-efficiency curve ----------
def fig7():
    k = np.array(ft["k_values"])
    r2m = np.array(ft["r2_mean"])
    r2s = np.array(ft["r2_std"])
    maem = np.array(ft["mae_mean"])
    maes = np.array(ft["mae_std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.errorbar(k, r2m, yerr=r2s, fmt="o-", color=BLUE, capsize=3, label="Source 1 + $k$ Source-2 examples")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ft["source2_internal_r2"], color=GREEN, ls="--", lw=1.2,
               label=f"Source-2-internal ceiling ($R^2$={ft['source2_internal_r2']:.2f},\n"
                     f"trained on Source 2 alone, no Source 1)")
    ax.set_xlabel("Labeled Source-2 fine-tuning examples ($k$)")
    ax.set_ylabel("$R^2$ on fixed held-out Source-2 test set")
    ax.set_title(f"(a) A few dozen labels flip $R^2$ positive\n"
                 f"($R^2$=0 crossed at $k\\approx${ft['k_for_r2_positive']:.0f} "
                 f"of {ft['n_pool_total']} available)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.errorbar(k, maem, yerr=maes, fmt="s-", color=ORANGE, capsize=3)
    ax.axhline(ft["source2_internal_mae"], color=GREEN, ls="--", lw=1.2,
               label=f"Source-2-internal ceiling\n(MAE={ft['source2_internal_mae']:.2f} eV)")
    ax.set_xlabel("Labeled Source-2 fine-tuning examples ($k$)")
    ax.set_ylabel("MAE on fixed held-out Source-2 test set (eV/O)")
    ax.set_title("(b) Error falls steadily with a modest\nnumber of target-domain labels")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save(fig, "fig7_finetuning_curve")


# ---------- Figure 8: redox-active-cation diagnosis ----------
def fig8():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    ax = axes[0]
    cats = ["Redox-active\ncation present", "Fixed-valence\ncations only"]
    maes = [rx["mae_redox"], rx["mae_fixed"]]
    ns = [rx["n_redox_active"], rx["n_fixed_valence"]]
    ax.bar(cats, maes, color=[BLUE, RED])
    for i, (m, n) in enumerate(zip(maes, ns)):
        ax.text(i, m + 0.08, f"n={n}", ha="center", fontsize=9)
    ax.set_ylabel("Mean absolute error (eV/O)")
    ax.set_title("(a) Transfer error by cation chemistry")

    ax = axes[1]
    cats2 = ["Share of\ntest compounds", "Share of total\nabsolute error"]
    redox_vals = [rx["frac_redox_compounds_pct"], rx["frac_redox_error_pct"]]
    fixed_vals = [100 - rx["frac_redox_compounds_pct"], 100 - rx["frac_redox_error_pct"]]
    x = np.arange(2)
    w = 0.6
    ax.bar(x, redox_vals, w, color=BLUE, label="Redox-active")
    ax.bar(x, fixed_vals, w, bottom=redox_vals, color=RED, label="Fixed-valence only")
    ax.set_xticks(x)
    ax.set_xticklabels(cats2)
    ax.set_ylabel("Percent")
    ax.set_title("(b) Fixed-valence compounds are a\nminority but a disproportionate\nerror share")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[2]
    bins = np.linspace(0, 11, 30)
    ax.hist(redox_df[redox_df["redox_active"]]["abs_residual"], bins=bins, alpha=0.6,
            color=BLUE, label=f"Redox-active (median={redox_df[redox_df['redox_active']]['abs_residual'].median():.2f} eV)",
            density=True)
    ax.hist(redox_df[~redox_df["redox_active"]]["abs_residual"], bins=bins, alpha=0.6,
            color=RED, label=f"Fixed-valence only (median={redox_df[~redox_df['redox_active']]['abs_residual'].median():.2f} eV)",
            density=True)
    ax.set_xlabel("Absolute transfer error (eV/O)")
    ax.set_ylabel("Probability density")
    ax.set_title("(c) Full error distributions\nby cation chemistry")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    save(fig, "fig8_redox_diagnosis")


if __name__ == "__main__":
    fig7()
    fig8()
    print("PHASE 2B FIGURES DONE")
