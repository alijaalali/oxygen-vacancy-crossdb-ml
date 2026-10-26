"""Predicted uncertainty vs. absolute error, stratified by redox-active vs.
fixed-valence-only class, to check whether uncertainty ranking adds
information beyond the chemistry-based classification alone.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
MDIR = f"{ROOT}/15th - OxygenVacancy-CrossDB-ML-project"

meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
uq2 = pd.read_csv(f"{ROOT}/data/uq_source2.csv")
assert np.allclose(uq2["true"].values, meta2["target"].values), "row-order mismatch"

abs_err = np.abs(uq2["true"] - uq2["pred_mean"]).to_numpy()
pred_std = uq2["pred_std"].to_numpy()
redox = meta2["redox_active"].to_numpy()

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pred_std[redox], abs_err[redox], s=14, alpha=0.5, color="#2166ac", label="Redox-active cation")
ax.scatter(pred_std[~redox], abs_err[~redox], s=14, alpha=0.5, color="#b2182b", label="Fixed-valence only")
ax.set_xlabel(r"Predicted ensemble std.\ dev.\ (eV/O)")
ax.set_ylabel(r"Absolute error, $|E_{\mathrm{vac}}^{\mathrm{pred}}-E_{\mathrm{vac}}^{\mathrm{true}}|$ (eV/O)")
ax.set_title("Cross-database uncertainty vs.\ error,\nby redox-active-cation class")
ax.legend(frameon=False, fontsize=9)
ax.text(0.03, 0.97,
        r"Spearman $\rho$: redox-active $=0.34$" + "\n" + r"fixed-valence-only $=0.83$",
        transform=ax.transAxes, fontsize=9, va="top")
fig.tight_layout()
fig.savefig(f"{MDIR}/figures/fig_uq_redox_stratified.pdf", bbox_inches="tight")
fig.savefig(f"{MDIR}/figures/fig_uq_redox_stratified.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_uq_redox_stratified")
