"""
Figure generation for the Oxygen-Vacancy Cross-Database ML manuscript (Paper 15).
Each figure shows a distinct concept -- no two figures repeat the same
comparison. No gridlines anywhere.
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
FIG.mkdir(exist_ok=True, parents=True)
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


results = json.load(open(DATA / "phase1_results.json"))
f1 = pd.read_csv(DATA / "featurized_source1.csv")
f2 = pd.read_csv(DATA / "featurized_source2.csv")
f3 = pd.read_csv(DATA / "featurized_source3.csv")
pred_internal = pd.read_csv(DATA / "predictions_internal.csv")
pred_s2 = pd.read_csv(DATA / "predictions_source2.csv")
pred_s3 = pd.read_csv(DATA / "predictions_source3.csv")
uq_internal = pd.read_csv(DATA / "uq_internal.csv")
uq_s2 = pd.read_csv(DATA / "uq_source2.csv")

# raw (pre-filter) Source 1 for the SI stability-filter diagnostic
raw1 = pd.read_csv(DATA / "emery_wolverton" / "wolverton_oxides.csv")
raw1["target"] = pd.to_numeric(raw1["Vacancy energy [eV/O atom]"], errors="coerce")
raw1["stability"] = pd.to_numeric(raw1["Stability [eV/atom]"], errors="coerce")
raw1 = raw1.dropna(subset=["target", "stability"]).reset_index(drop=True)


# ---------- Figure 1: dataset overview ----------
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    counts = [results["source1_n_after_filter"], results["source2_n"], results["source3_n"]]
    ax.bar(["Source 1\n(Emery &\nWolverton)", "Source 2\n(Witman/\nNREL)", "Source 3\n(JARVIS/\nNIST)"],
           counts, color=[BLUE, RED, GREEN])
    ax.set_ylabel("Number of compounds")
    ax.set_yscale("log")
    ax.set_title("(a) Dataset size (after filtering)")
    for i, v in enumerate(counts):
        ax.text(i, v * 1.15, str(v), ha="center", fontsize=10)

    ax = axes[1]
    bins = np.linspace(-10, 10, 50)
    ax.hist(f1["target"], bins=bins, alpha=0.6, color=BLUE, label="Source 1 (filtered)", density=True)
    ax.hist(f2["target"], bins=bins, alpha=0.6, color=RED, label="Source 2", density=True)
    ax.hist(f3["target"], bins=bins, alpha=0.6, color=GREEN, label="Source 3", density=True)
    ax.set_xlabel("Oxygen-vacancy formation energy (eV/O)")
    ax.set_ylabel("Probability density")
    ax.set_title("(b) Target distribution, all sources")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    d = results["distribution"]
    names = ["Source 1", "Source 2", "Source 3"]
    means = [d["source1"]["mean"], d["source2"]["mean"], d["source3"]["mean"]]
    medians = [d["source1"]["median"], d["source2"]["median"], d["source3"]["median"]]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, means, w, label="Mean", color=ORANGE)
    ax.bar(x + w / 2, medians, w, label="Median", color=PURPLE)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Vacancy formation energy (eV/O)")
    ax.set_title("(c) Mean vs. median by source")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save(fig, "fig1_dataset_overview")


# ---------- Figure 2: in-distribution parity plot ----------
def fig2():
    fig, ax = plt.subplots(figsize=(5.2, 5))
    t, p = pred_internal["true"], pred_internal["pred"]
    ax.scatter(t, p, s=14, alpha=0.5, color=BLUE)
    lims = [min(t.min(), p.min()) - 0.5, max(t.max(), p.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.text(0.05, 0.92,
             f"$R^2$ = {results['transfer']['internal_r2']:.2f}\nMAE = {results['transfer']['internal_mae']:.2f} eV",
             transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlabel("True vacancy formation energy (eV/O)")
    ax.set_ylabel("Predicted vacancy formation energy (eV/O)")
    ax.set_title("Source 1-internal random 80/20 split\n(in-distribution reference)")
    fig.tight_layout()
    save(fig, "fig2_internal_parity")


# ---------- Figure 3: cross-database transfer summary ----------
def fig3():
    tr = results["transfer"]
    conditions = ["Internal\n(Source 1)", "Source 2\n(raw)", "Source 2\n(bias-corrected)",
                  "Source 3\n(raw)", "Source 3\n(bias-corrected)"]
    r2_vals = [tr["internal_r2"], tr["source2_r2"], tr["source2_r2_corrected"],
               tr["source3_r2"], tr["source3_r2_corrected"]]
    mae_vals = [tr["internal_mae"], tr["source2_mae"], tr["source2_mae_corrected"],
                tr["source3_mae"], tr["source3_mae_corrected"]]
    colors = [GRAY, RED, "#e8a0a0", GREEN, "#a8d8a8"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.bar(conditions, r2_vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$R^2$")
    ax.set_title("(a) Transfer accuracy")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(r2_vals):
        ax.text(i, v + (0.1 if v >= 0 else -0.25), f"{v:.2f}", ha="center", fontsize=9)

    ax = axes[1]
    ax.bar(conditions, mae_vals, color=colors)
    ax.set_ylabel("MAE (eV/O)")
    ax.set_title("(b) Transfer error")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(mae_vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    save(fig, "fig3_crossdb_transfer_summary")


# ---------- Figure 4: offset-correction test ----------
def fig4():
    tr = results["transfer"]
    res_raw = pred_s2["pred"] - pred_s2["true"]
    res_corr = res_raw - tr["source2_bias"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(-8, 4, 50)
    ax.hist(res_raw, bins=bins, alpha=0.6, color=BLUE,
            label=f"Raw residual\nMAE={tr['source2_mae']:.2f} eV, $R^2$={tr['source2_r2']:.2f}")
    ax.hist(res_corr, bins=bins, alpha=0.6, color=RED,
            label=f"After removing\nmean offset\nMAE={tr['source2_mae_corrected']:.2f} eV, "
                  f"$R^2$={tr['source2_r2_corrected']:.2f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual, vacancy formation energy (predicted $-$ true, eV/O)")
    ax.set_ylabel("Count")
    ax.set_title("A constant offset improves MAE but not ranking:\n"
                  "$R^2$ stays negative after bias correction (Source 1 $\\to$ Source 2)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_offset_correction_test")


# ---------- Figure 5: uncertainty calibration ----------
def fig5():
    uq = results["uq"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    err_in = (uq_internal["true"] - uq_internal["pred_mean"]).abs()
    ax.scatter(uq_internal["pred_std"], err_in, s=16, alpha=0.5, color=BLUE)
    ax.set_xlabel("Ensemble predicted std (eV/O)")
    ax.set_ylabel("$|$Actual error$|$ (eV/O)")
    ax.set_title(f"(a) In-distribution\nSpearman $\\rho$ = {uq['in_dist_spearman']:.2f} "
                 f"($p$ = {uq['in_dist_p']:.1e})")

    ax = axes[1]
    err_out = (uq_s2["true"] - uq_s2["pred_mean"]).abs()
    ax.scatter(uq_s2["pred_std"], err_out, s=10, alpha=0.35, color=RED)
    ax.set_xlabel("Ensemble predicted std (eV/O)")
    ax.set_ylabel("$|$Actual error$|$ (eV/O)")
    ax.set_title(f"(b) Cross-database (Source 2)\nSpearman $\\rho$ = {uq['cross_db_spearman']:.2f} "
                 f"($p$ = {uq['cross_db_p']:.1e})")

    fig.tight_layout()
    save(fig, "fig5_uq_calibration")


# ---------- Figure 6: active learning ----------
def fig6():
    al = results["active_learning"]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot(al["n_random"], al["mae_random"], "o-", color=GRAY, label="Random selection")
    ax.plot(al["n_uncertainty"], al["mae_uncertainty"], "s-", color=ORANGE, label="Uncertainty-based selection")
    ax.axhline(al["target_mae"], color="k", ls="--", lw=1, label=f"Target MAE ({al['target_mae']:.2f} eV)")
    ax.axvline(al["points_random"], color=GRAY, ls=":", lw=1)
    ax.set_xlabel("Labeled training points")
    ax.set_ylabel("Held-out MAE (eV/O)")
    ax.set_title("Active learning: uncertainty-based querying shows\nno efficiency gain over random selection")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_active_learning")


# ---------- SI Figure 1: stability filter diagnostic ----------
def figs1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    ax = axes[0]
    bins = np.linspace(-20, 10, 60)
    ax.hist(raw1["target"], bins=bins, alpha=0.6, color=GRAY, label=f"All (n={len(raw1)})")
    ax.hist(raw1[raw1["stability"] <= 0.1]["target"], bins=bins, alpha=0.7, color=BLUE,
            label=f"Stability $\\leq$ 0.1 eV/atom (n={results['source1_n_after_filter']})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Vacancy formation energy (eV/O)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Effect of the stability filter\non Source 1")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    d = results["source1_in_literature_subset_diagnostic"]
    cats = ["Raw Source 1\n(all, pre-filter)", "Stability-filtered\n(this study)", "\"In literature\"\nsubset"]
    means = [raw1["target"].mean(), raw1[raw1["stability"] <= 0.1]["target"].mean(), d["mean"]]
    stds = [raw1["target"].std(), raw1[raw1["stability"] <= 0.1]["target"].std(), d["std"]]
    ax.bar(cats, means, yerr=stds, color=[GRAY, BLUE, PURPLE], capsize=4)
    ax.axhline(results["distribution"]["source2"]["mean"], color=RED, ls="--", lw=1.2,
               label=f"Source 2 mean ({results['distribution']['source2']['mean']:.2f} eV)")
    ax.set_ylabel("Vacancy formation energy, mean $\\pm$ std (eV/O)")
    ax.set_title("(b) The literature-reported subset of\nSource 1 matches Sources 2/3")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    save(fig, "figs1_stability_filter_diagnostic")


# ---------- SI Figure 2: Source 2 and Source 3 raw parity plots ----------
def figs2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    t, p = pred_s2["true"], pred_s2["pred"]
    ax.scatter(t, p, s=10, alpha=0.3, color=RED)
    lims = [min(t.min(), p.min()) - 0.5, max(t.max(), p.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.text(0.05, 0.92,
             f"$R^2$ = {results['transfer']['source2_r2']:.2f}\nMAE = {results['transfer']['source2_mae']:.2f} eV",
             transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlabel("True (eV/O)")
    ax.set_ylabel("Predicted (eV/O)")
    ax.set_title(f"(a) Source 1 $\\to$ Source 2\n(n={results['source2_n']}, zero-shot)")

    ax = axes[1]
    t, p = pred_s3["true"], pred_s3["pred"]
    ax.scatter(t, p, s=24, alpha=0.6, color=GREEN)
    lims = [min(t.min(), p.min()) - 0.5, max(t.max(), p.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.text(0.05, 0.92,
             f"$R^2$ = {results['transfer']['source3_r2']:.2f}\nMAE = {results['transfer']['source3_mae']:.2f} eV",
             transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlabel("True (eV/O)")
    ax.set_ylabel("Predicted (eV/O)")
    ax.set_title(f"(b) Source 1 $\\to$ Source 3\n(n={results['source3_n']}, small-sample caveat)")

    fig.tight_layout()
    save(fig, "figs2_raw_parity_sources23")


# ---------- SI Figure 3: UQ coverage (within 1/2 std) ----------
def figs3():
    uq = results["uq"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cats = ["Within 1 std\n(in-dist.)", "Within 1 std\n(cross-DB)",
            "Within 2 std\n(in-dist.)", "Within 2 std\n(cross-DB)"]
    vals = [uq["in_dist_within1std"] * 100, uq["cross_db_within1std"] * 100,
            uq["in_dist_within2std"] * 100, uq["cross_db_within2std"] * 100]
    ideal = [68.3, 68.3, 95.4, 95.4]
    x = np.arange(4)
    ax.bar(x, vals, color=[BLUE, RED, BLUE, RED], alpha=0.85)
    ax.plot(x, ideal, "k_", markersize=25, markeredgewidth=2, label="Ideal (Gaussian)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Fraction of points (%)")
    ax.set_title("Uncertainty is a useful relative ranking signal\n(Figure 5) but poorly calibrated in absolute terms")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "figs3_uq_coverage")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    figs1(); figs2(); figs3()
    print("ALL FIGURES DONE")
