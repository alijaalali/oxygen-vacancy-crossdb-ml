# Figure: composition-grouped vs entry-random label-efficiency curves, joint vs target-only training.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
d = json.load(open(f"{ROOT}/data/extended_robustness_results.json"))["finetuning_and_source2only_comparison"]

er = d["entry_random"]
cg = d["composition_grouped"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
ax.axhline(0, color="gray", lw=0.8, ls=":", label="_nolegend_")
for proto, style, label in [
    (er, dict(ls="-", marker="o"), "Entry-random (may leak)"),
    (cg, dict(ls="--", marker="s"), "Composition-grouped (leak-free)"),
]:
    k = proto["combined_source1_plus_source2"]["k_values"]
    r2 = proto["combined_source1_plus_source2"]["r2_mean"]
    r2sd = proto["combined_source1_plus_source2"]["r2_std"]
    ax.errorbar(k, r2, yerr=r2sd, **style, capsize=2, markersize=4, label=label)
ax.set_xlabel("Labeled Source-2 compounds added, $k$")
ax.set_ylabel("$R^2$ on held-out Source-2 test set")
ax.set_title("(a) Entry-random vs. composition-grouped split")
ax.legend(fontsize=8, loc="lower right")

ax2 = axes[1]
k_cg = cg["combined_source1_plus_source2"]["k_values"]
r2_comb_cg = cg["combined_source1_plus_source2"]["r2_mean"]
k_cg_s2 = cg["source2_only"]["k_values"]
r2_s2only_cg = cg["source2_only"]["r2_mean"]
ax2.plot(k_cg, r2_comb_cg, "-o", markersize=4, label="Source 1 + Source 2 (joint)")
ax2.plot(k_cg_s2, r2_s2only_cg, "-^", markersize=4, label="Source 2 only")
ax2.axhline(0, color="gray", lw=0.8, ls=":")
ax2.set_xlabel("Labeled Source-2 compounds used, $k$")
ax2.set_ylabel("$R^2$ on held-out Source-2 test set")
ax2.set_title("(b) Joint vs. target-only training, composition-grouped")
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
out_pdf = f"{ROOT}/15th - OxygenVacancy-CrossDB-ML-project/figures/fig12_finetuning_protocols.pdf"
out_png = f"{ROOT}/15th - OxygenVacancy-CrossDB-ML-project/figures/fig12_finetuning_protocols.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved", out_pdf)
