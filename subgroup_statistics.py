"""Subgroup statistics for the redox-active vs. fixed-valence-only split:
RMSE alongside MAE/R2, subgroup target mean/std, and a composition-
grouped bootstrap CI for the subgroup MAE difference (so no compound's
multiple vacancy sites are split across resample draws).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
pred2 = pd.read_csv(f"{ROOT}/data/predictions_source2.csv")
assert np.allclose(pred2["true"].values, meta2["target"].values), "row-order mismatch"

df = meta2.copy()
df["pred"] = pred2["pred"].values

out = {}
for label, mask in [("redox_active", df["redox_active"]), ("fixed_valence_only", ~df["redox_active"])]:
    sub = df[mask]
    y, p = sub["target"].to_numpy(), sub["pred"].to_numpy()
    out[label] = {
        "n": int(len(sub)),
        "target_mean": float(y.mean()), "target_std": float(y.std()),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
    }
    print(f"{label}: n={out[label]['n']} target_mean={out[label]['target_mean']:.2f} "
          f"target_std={out[label]['target_std']:.2f} MAE={out[label]['mae']:.3f} "
          f"RMSE={out[label]['rmse']:.3f} R2={out[label]['r2']:.3f}")

# Composition-grouped bootstrap CI for the MAE difference (fixed-valence - redox-active),
# resampling whole compositions (not individual rows) so a compound's several
# vacancy sites always move together.
rng = np.random.RandomState(0)
groups = df.groupby("composition")
comp_ids = df["composition"].unique()
comp_row_idx = {c: g.index.to_numpy() for c, g in groups}
comp_is_redox = {c: bool(g["redox_active"].iloc[0]) for c, g in groups}
target_arr = df["target"].to_numpy()
pred_arr = df["pred"].to_numpy()

n_boot = 5000
diffs = []
for _ in range(n_boot):
    samp_comps = rng.choice(comp_ids, size=len(comp_ids), replace=True)
    redox_idx, fixed_idx = [], []
    for c in samp_comps:
        idx = comp_row_idx[c]
        (redox_idx if comp_is_redox[c] else fixed_idx).extend(idx)
    if not redox_idx or not fixed_idx:
        continue
    mae_redox = mean_absolute_error(target_arr[redox_idx], pred_arr[redox_idx])
    mae_fixed = mean_absolute_error(target_arr[fixed_idx], pred_arr[fixed_idx])
    diffs.append(mae_fixed - mae_redox)
diffs = np.array(diffs)
ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
out["mae_difference_fixed_minus_redox"] = {
    "mean": float(diffs.mean()), "ci95_composition_grouped_bootstrap": ci, "n_boot": int(len(diffs)),
}
print(f"\nMAE difference (fixed-valence - redox-active): mean={diffs.mean():.3f}, 95% CI {ci}")

json.dump(out, open(f"{ROOT}/data/phase5_subgroup_statistics.json", "w"), indent=2)
print("Saved data/phase5_subgroup_statistics.json")
