"""
Figures for the Phase-2c multi-model robustness check.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


bench = json.load(open(DATA / "phase2c_multimodel_benchmark.json"))
ft = json.load(open(DATA / "phase2c_finetuning_multimodel.json"))

MODEL_ORDER = ["GradientBoosting", "XGBoost", "RandomForest", "AdaBoost", "SVR",
               "kNN", "GaussianProcess", "Ridge", "Lasso", "MLP"]
MODEL_LABELS = ["Grad.\nBoosting", "XGBoost", "Random\nForest", "AdaBoost", "SVR",
                "$k$NN", "Gaussian\nProcess", "Ridge", "Lasso", "MLP"]


# ---------- Figure 9: 10-model redox-active-cation diagnosis ----------
def fig9():
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    r2_redox = [bench[m]["r2_redox"] for m in MODEL_ORDER]
    r2_fixed = [bench[m]["r2_fixed"] for m in MODEL_ORDER]
    ax.bar(x - w / 2, r2_redox, w, color=BLUE, label="Redox-active cation present")
    ax.bar(x + w / 2, r2_fixed, w, color=RED, label="Fixed-valence cations only")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    ax.set_ylabel("$R^2$, Source 1 $\\to$ Source 2 zero-shot transfer")
    ax.set_title("The redox-active-vs-fixed-valence asymmetry holds across\n"
                 "10 independent regressors spanning tree ensembles, kernel\n"
                 "methods, linear models, and a neural network")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    save(fig, "fig9_redox_multimodel")


# ---------- Figure 10: multi-model fine-tuning curves ----------
def fig10():
    fig, ax = plt.subplots(figsize=(7, 5.2))
    colors = {"GradientBoosting": BLUE, "XGBoost": ORANGE, "RandomForest": GREEN,
              "Ridge": PURPLE, "MLP": RED}
    markers = {"GradientBoosting": "o", "XGBoost": "s", "RandomForest": "^",
               "Ridge": "D", "MLP": "v"}
    for name in ["GradientBoosting", "XGBoost", "RandomForest", "Ridge", "MLP"]:
        s = ft["models"][name]
        k = np.array(s["k_values"])
        r2m = np.array(s["r2_mean"])
        r2s = np.array(s["r2_std"])
        ax.errorbar(k, r2m, yerr=r2s, fmt=markers[name] + "-", color=colors[name],
                    capsize=3, label=name, markersize=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Labeled Source-2 fine-tuning examples ($k$)")
    ax.set_ylabel("$R^2$ on fixed held-out Source-2 test set")
    ax.set_title("The fine-tuning benefit is model-agnostic:\n"
                 "5 model families all recover positive $R^2$ by $k\\approx100$--200")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig10_finetuning_multimodel")


if __name__ == "__main__":
    fig9()
    fig10()
    print("PHASE 2C FIGURES DONE")
