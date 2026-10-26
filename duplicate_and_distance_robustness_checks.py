# Duplicate-composition reweighting and feature-space distance-to-training-distribution checks.
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, mannwhitneyu

RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

# ---------- Reload data (identical to phase2b/2c) ----------
df1 = pd.read_csv(f"{ROOT}/data/emery_wolverton/wolverton_oxides.csv")
df1["target"] = pd.to_numeric(df1["Vacancy energy [eV/O atom]"], errors="coerce")
df1["stability"] = pd.to_numeric(df1["Stability [eV/atom]"], errors="coerce")
df1 = df1.dropna(subset=["target", "stability"]).reset_index(drop=True)
df1 = df1[df1["stability"] <= 0.1].reset_index(drop=True)
df1["composition"] = df1["Chemical formula"]

df2 = pd.read_csv(f"{ROOT}/data/witman_nrel/witman_oxygen_vacancies.csv")
df2["target"] = df2["dH_eV"]
df2["composition"] = df2["formula"]

feat = ElementProperty.from_preset("magpie")


def featurize(df):
    comps, ok = [], []
    for c in df["composition"]:
        try:
            comps.append(Composition(c))
            ok.append(True)
        except Exception:
            ok.append(False)
    df = df[ok].reset_index(drop=True)
    X = feat.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition",
                                  ignore_errors=True)
    Xf = X.drop(columns=["composition"])
    mask = Xf.notna().all(axis=1)
    return (Xf[mask].reset_index(drop=True), df.loc[mask, "target"].reset_index(drop=True),
            df.loc[mask].reset_index(drop=True))


X1, y1, meta1 = featurize(df1)
X2, y2, meta2 = featurize(df2)
feature_cols = list(X1.columns)
X2 = X2[feature_cols]
print(f"Reloaded: source1={len(X1)} source2={len(X2)} ({meta2['composition'].nunique()} unique compositions)")

REDOX_ACTIVE = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                 "Mo", "W", "Ce", "Pr", "Tb", "U", "Sn", "Pb"}


def classify(formula, active_set):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    els.discard("O")
    return len(els & active_set) > 0


meta2 = meta2.copy()
meta2["redox_active"] = meta2["composition"].apply(lambda f: classify(f, REDOX_ACTIVE))

# ---------- Reproduce the primary Phase-1 model exactly ----------
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=RNG)
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.02,
                                 subsample=0.8, random_state=RNG)
gbm.fit(X1_tr, y1_tr)
pred2 = gbm.predict(X2)

ref = json.load(open(f"{ROOT}/data/phase1_results.json"))
r2_check = r2_score(y2, pred2)
mae_check = mean_absolute_error(y2, pred2)
assert abs(r2_check - ref["transfer"]["source2_r2"]) < 1e-6
assert abs(mae_check - ref["transfer"]["source2_mae"]) < 1e-6
print(f"Reproduces Phase-1 numbers exactly: R2={r2_check:.3f} MAE={mae_check:.3f}\n")

out = {}

# ============================================================
# Check 1: duplicate-composition reweighting
# ============================================================
print("=" * 70)
print("CHECK 1: duplicate-composition reweighting")
print("=" * 70)

dup = meta2.copy()
dup["pred"] = pred2
dup["true"] = y2.values

vc = dup["composition"].value_counts()
print(f"795 rows, {dup['composition'].nunique()} unique compositions")
print("Top 5 most-duplicated compositions:")
print(vc.head(5).to_string())

# Per-unique-composition average (true and pred averaged over duplicate rows)
agg = dup.groupby("composition").agg(
    true=("true", "mean"), pred=("pred", "mean"),
    redox_active=("redox_active", "first"), n_rows=("true", "size")
).reset_index()

r2_dedup = r2_score(agg["true"], agg["pred"])
mae_dedup = mean_absolute_error(agg["true"], agg["pred"])
print(f"\nRow-weighted (n=795):        R2={r2_check:.3f}  MAE={mae_check:.3f}")
print(f"Per-unique-compound (n={len(agg)}): R2={r2_dedup:.3f}  MAE={mae_dedup:.3f}")

redox_agg = agg[agg["redox_active"]]
fixed_agg = agg[~agg["redox_active"]]
r2_redox_dedup = r2_score(redox_agg["true"], redox_agg["pred"])
r2_fixed_dedup = r2_score(fixed_agg["true"], fixed_agg["pred"])
mae_redox_dedup = mean_absolute_error(redox_agg["true"], redox_agg["pred"])
mae_fixed_dedup = mean_absolute_error(fixed_agg["true"], fixed_agg["pred"])
print(f"\nRedox-active, per-compound (n={len(redox_agg)}):  R2={r2_redox_dedup:.3f}  MAE={mae_redox_dedup:.3f}")
print(f"Fixed-valence, per-compound (n={len(fixed_agg)}): R2={r2_fixed_dedup:.3f}  MAE={mae_fixed_dedup:.3f}")

out["duplicate_check"] = {
    "n_rows": 795, "n_unique": int(dup["composition"].nunique()),
    "top_duplicates": vc.head(5).to_dict(),
    "row_weighted_r2": float(r2_check), "row_weighted_mae": float(mae_check),
    "dedup_r2": float(r2_dedup), "dedup_mae": float(mae_dedup),
    "dedup_n_redox": int(len(redox_agg)), "dedup_n_fixed": int(len(fixed_agg)),
    "dedup_r2_redox": float(r2_redox_dedup), "dedup_mae_redox": float(mae_redox_dedup),
    "dedup_r2_fixed": float(r2_fixed_dedup), "dedup_mae_fixed": float(mae_fixed_dedup),
}
agg.to_csv(f"{ROOT}/data/redox_diagnosis_source2_dedup.csv", index=False)

# ============================================================
# Check 2: feature-space distance to training distribution
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: feature-space distance to Source-1 training distribution")
print("=" * 70)

scaler = StandardScaler().fit(X1_tr)
X1_tr_s = scaler.transform(X1_tr)
X2_s = scaler.transform(X2)

# Nearest-neighbor distance in standardized Magpie feature space
from scipy.spatial.distance import cdist
dmat = cdist(X2_s, X1_tr_s, metric="euclidean")
nn_dist = dmat.min(axis=1)

abs_err = np.abs(pred2 - y2.values)
redox_mask = meta2["redox_active"].values

med_dist_redox = np.median(nn_dist[redox_mask])
med_dist_fixed = np.median(nn_dist[~redox_mask])
mw_stat, mw_p = mannwhitneyu(nn_dist[redox_mask], nn_dist[~redox_mask], alternative="less")
rho_dist_err, p_dist_err = spearmanr(nn_dist, abs_err)

print(f"Median nearest-Source-1-neighbor distance: redox-active={med_dist_redox:.2f}, "
      f"fixed-valence-only={med_dist_fixed:.2f}")
print(f"Mann-Whitney U test (redox-active distances < fixed-valence distances): "
      f"U={mw_stat:.0f}, p={mw_p:.2e}")
print(f"Spearman(nearest-neighbor distance, |transfer error|): rho={rho_dist_err:.3f}, p={p_dist_err:.2e}")

out["distance_check"] = {
    "median_dist_redox": float(med_dist_redox), "median_dist_fixed": float(med_dist_fixed),
    "mannwhitney_u": float(mw_stat), "mannwhitney_p": float(mw_p),
    "spearman_dist_vs_error_rho": float(rho_dist_err), "spearman_dist_vs_error_p": float(p_dist_err),
}
pd.DataFrame({"composition": meta2["composition"], "redox_active": meta2["redox_active"],
              "nn_distance": nn_dist, "abs_error": abs_err}).to_csv(
    f"{ROOT}/data/nn_distance_source2.csv", index=False)

# ============================================================
# Check 3: redox-active element-list sensitivity
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: redox-active element-list sensitivity")
print("=" * 70)

ALT_LISTS = {
    "primary_16": REDOX_ACTIVE,
    "narrow_9_firstrow_plus_Ce": {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ce"},
    "broad_19_plus_NbTaBi": REDOX_ACTIVE | {"Nb", "Ta", "Bi"},
    "no_mainGroup_14_transitionOnly": REDOX_ACTIVE - {"Sn", "Pb"},
}

sensitivity = {}
for name, els in ALT_LISTS.items():
    cls = meta2["composition"].apply(lambda f: classify(f, els))
    mask = cls.values
    n_redox, n_fixed = int(mask.sum()), int((~mask).sum())
    r2_redox = r2_score(y2.values[mask], pred2[mask])
    r2_fixed = r2_score(y2.values[~mask], pred2[~mask])
    mae_redox = np.abs(pred2[mask] - y2.values[mask]).mean()
    mae_fixed = np.abs(pred2[~mask] - y2.values[~mask]).mean()
    sensitivity[name] = {
        "elements": sorted(els), "n_redox": n_redox, "n_fixed": n_fixed,
        "r2_redox": float(r2_redox), "r2_fixed": float(r2_fixed),
        "mae_redox": float(mae_redox), "mae_fixed": float(mae_fixed),
    }
    print(f"{name:32s} (n_elements={len(els):2d}): "
          f"redox n={n_redox:3d} R2={r2_redox:7.3f} MAE={mae_redox:.3f} | "
          f"fixed n={n_fixed:3d} R2={r2_fixed:8.3f} MAE={mae_fixed:.3f}")

out["sensitivity_check"] = sensitivity

with open(f"{ROOT}/data/phase2d_robustness_checks.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nSaved data/phase2d_robustness_checks.json, "
      "data/redox_diagnosis_source2_dedup.csv, data/nn_distance_source2.csv")
print("DONE")
