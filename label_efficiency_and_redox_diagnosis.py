# Label-efficiency (fine-tuning) curve and redox-active-cation transfer-error diagnosis.
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

# ---------- Reload raw data + featurize, keeping composition strings ----------
df1 = pd.read_csv(f"{ROOT}/data/emery_wolverton/wolverton_oxides.csv")
df1["target"] = pd.to_numeric(df1["Vacancy energy [eV/O atom]"], errors="coerce")
df1["stability"] = pd.to_numeric(df1["Stability [eV/atom]"], errors="coerce")
df1 = df1.dropna(subset=["target", "stability"]).reset_index(drop=True)
df1 = df1[df1["stability"] <= 0.1].reset_index(drop=True)
df1["composition"] = df1["Chemical formula"]

df2 = pd.read_csv(f"{ROOT}/data/witman_nrel/witman_oxygen_vacancies.csv")
df2["target"] = df2["dH_eV"]
df2["composition"] = df2["formula"]

d3 = json.load(open(f"{ROOT}/data/jarvis_vacancydb/vacancydb.json"))
ov3 = [x for x in d3 if x.get("symbol") == "O"]
df3 = pd.DataFrame({"composition": [x["bulk_formula"] for x in ov3],
                     "target": [x["ef"] for x in ov3]})

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
X3, y3, meta3 = featurize(df3)
feature_cols = list(X1.columns)
X2 = X2[feature_cols]
X3 = X3[feature_cols]

print(f"Reloaded: source1={len(X1)} source2={len(X2)} source3={len(X3)}")

# Sanity check against the saved featurized tables used in Phase 1 / make_predictions.py
saved1 = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
saved2 = pd.read_csv(f"{ROOT}/data/featurized_source2.csv")
assert len(saved1) == len(X1) and len(saved2) == len(X2), "row-count mismatch vs Phase 1 tables"
assert np.allclose(saved1["target"].values, y1.values), "target mismatch vs Phase 1 source1"
assert np.allclose(saved2["target"].values, y2.values), "target mismatch vs Phase 1 source2"
print("Reload matches Phase 1 featurized tables exactly (row count + targets).")

GBM_KW = dict(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8)

# ============================================================
# Analysis 1: few-shot fine-tuning / label-efficiency curve
# ============================================================
# Fixed held-out Source-2 test set (20%), never used for fine-tuning at any k.
X2_pool, X2_test, y2_pool, y2_test, meta2_pool, meta2_test = train_test_split(
    X2, y2, meta2, test_size=0.2, random_state=RNG)
print(f"\nSource-2 fine-tuning pool: {len(X2_pool)}, fixed held-out test: {len(X2_test)}")

# Zero-shot baseline on this exact (new) split, for a clean comparison point
gbm0 = GradientBoostingRegressor(random_state=RNG, **GBM_KW)
gbm0.fit(X1, y1)
pred0 = gbm0.predict(X2_test)
r2_zero = r2_score(y2_test, pred0)
mae_zero = mean_absolute_error(y2_test, pred0)
print(f"Zero-shot on this split's held-out test (n={len(X2_test)}): R2={r2_zero:.3f} MAE={mae_zero:.3f}")

# Source-2-internal baseline (trained AND tested within Source 2 only, no Source-1 data)
# -- the ceiling this fine-tuning curve should approach as k grows.
gbm_s2_only = GradientBoostingRegressor(random_state=RNG, **GBM_KW)
gbm_s2_only.fit(X2_pool, y2_pool)
pred_s2_only = gbm_s2_only.predict(X2_test)
r2_s2_internal = r2_score(y2_test, pred_s2_only)
mae_s2_internal = mean_absolute_error(y2_test, pred_s2_only)
print(f"Source-2-internal (trained on {len(X2_pool)} Source-2 pool only, no Source-1): "
      f"R2={r2_s2_internal:.3f} MAE={mae_s2_internal:.3f}")

k_values = [0, 5, 10, 20, 50, 100, 200, 300, len(X2_pool)]
n_seeds = 10
ft_results = {k: {"r2": [], "mae": []} for k in k_values}

for seed in range(n_seeds):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X2_pool))
    for k in k_values:
        if k == 0:
            Xtr, ytr = X1, y1
        else:
            idx = perm[:k]
            Xtr = pd.concat([X1, X2_pool.iloc[idx]], ignore_index=True)
            ytr = pd.concat([y1, y2_pool.iloc[idx]], ignore_index=True)
        m = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        m.fit(Xtr, ytr)
        pred = m.predict(X2_test)
        ft_results[k]["r2"].append(r2_score(y2_test, pred))
        ft_results[k]["mae"].append(mean_absolute_error(y2_test, pred))
    print(f"seed {seed} done")

ft_summary = {
    "k_values": k_values,
    "r2_mean": [float(np.mean(ft_results[k]["r2"])) for k in k_values],
    "r2_std": [float(np.std(ft_results[k]["r2"])) for k in k_values],
    "mae_mean": [float(np.mean(ft_results[k]["mae"])) for k in k_values],
    "mae_std": [float(np.std(ft_results[k]["mae"])) for k in k_values],
    "n_seeds": n_seeds,
    "n_test_fixed": len(X2_test),
    "n_pool_total": len(X2_pool),
    "source2_internal_r2": float(r2_s2_internal),
    "source2_internal_mae": float(mae_s2_internal),
    "zero_shot_r2_this_split": float(r2_zero),
    "zero_shot_mae_this_split": float(mae_zero),
}
print("\n--- Fine-tuning / label-efficiency summary ---")
for k, r2m, r2s, maem, maes in zip(k_values, ft_summary["r2_mean"], ft_summary["r2_std"],
                                     ft_summary["mae_mean"], ft_summary["mae_std"]):
    print(f"k={k:4d}: R2={r2m:.3f}+-{r2s:.3f}  MAE={maem:.3f}+-{maes:.3f}")

# Points needed to cross R2=0 (useful) and to reach within 90% of the
# Source-2-internal ceiling MAE, interpolated linearly between the nearest
# sampled k values.
def interp_crossing(xs, ys, target, mode="above"):
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        crossed = (y0 < target <= y1) if mode == "above" else (y0 > target >= y1)
        if crossed and y1 != y0:
            frac = (target - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0)
    return None

k_for_r2_positive = interp_crossing(k_values, ft_summary["r2_mean"], 0.0, mode="above")
mae_90pct_ceiling = mae_s2_internal / 0.9
k_for_90pct_ceiling = interp_crossing(k_values, ft_summary["mae_mean"], mae_90pct_ceiling, mode="below")
ft_summary["k_for_r2_positive"] = None if k_for_r2_positive is None else float(k_for_r2_positive)
ft_summary["mae_90pct_ceiling_eV"] = float(mae_90pct_ceiling)
ft_summary["k_for_90pct_ceiling"] = None if k_for_90pct_ceiling is None else float(k_for_90pct_ceiling)
print(f"\nLabeled Source-2 points needed for R2>0 (interpolated): {k_for_r2_positive}")
print(f"Labeled Source-2 points needed for MAE within 90% of the Source-2-internal "
      f"ceiling ({mae_90pct_ceiling:.3f} eV, interpolated): {k_for_90pct_ceiling}")

# ============================================================
# Analysis 2: redox-active-cation diagnosis
# ============================================================
# Cations with an experimentally well-documented accessible reduced
# oxidation state under typical oxygen-vacancy-forming (reducing)
# conditions in oxides: first-row and selected heavier transition metals,
# redox-active lanthanides/actinides, and the main-group ions with a
# known lower-valence reduced state (Sn, Pb). This is the standard
# distinction drawn in the oxide-defect-chemistry literature between
# cations that can locally charge-compensate a vacancy via a valence
# change versus cations that cannot (Deml et al., J. Phys. Chem. Lett.
# 2015, already cited in the main text for the same physical mechanism).
REDOX_ACTIVE = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                 "Mo", "W", "Ce", "Pr", "Tb", "U", "Sn", "Pb"}


def has_redox_active_cation(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    els.discard("O")
    return len(els & REDOX_ACTIVE) > 0


meta2_full = meta2.copy()
meta2_full["redox_active"] = meta2_full["composition"].apply(has_redox_active_cation)
meta2_full["pred"] = gbm0.predict(X2) if False else None  # placeholder, filled below

# Use the ORIGINAL Phase-1 model/split (full X2, not the fine-tuning test
# split) so this diagnosis is directly comparable to the main Phase-1
# Source-1->Source-2 zero-shot numbers already reported.
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=RNG)
gbm_main = GradientBoostingRegressor(random_state=RNG, **GBM_KW)
gbm_main.fit(X1_tr, y1_tr)
pred_main_internal = gbm_main.predict(X1_te)
pred_main_s2 = gbm_main.predict(X2)
r2_check = r2_score(y2, pred_main_s2)
mae_check = mean_absolute_error(y2, pred_main_s2)
ref = json.load(open(f"{ROOT}/data/phase1_results.json"))
print(f"\nRedox-diagnosis base-model check: R2={r2_check:.6f} (saved {ref['transfer']['source2_r2']:.6f}), "
      f"MAE={mae_check:.6f} (saved {ref['transfer']['source2_mae']:.6f})")
assert abs(r2_check - ref["transfer"]["source2_r2"]) < 1e-6
assert abs(mae_check - ref["transfer"]["source2_mae"]) < 1e-6
print("Matches exactly -- using this model for the redox diagnosis.")

meta2_full["abs_residual"] = np.abs(pred_main_s2 - y2.values)
meta2_full["residual"] = pred_main_s2 - y2.values
n_unresolved = meta2_full["redox_active"].isna().sum()
print(f"Formulas that failed to parse for element classification: {n_unresolved}")
meta2_valid = meta2_full.dropna(subset=["redox_active"]).copy()

redox = meta2_valid[meta2_valid["redox_active"]]
fixed = meta2_valid[~meta2_valid["redox_active"]]
n_redox, n_fixed = len(redox), len(fixed)
frac_redox_compounds = n_redox / len(meta2_valid) * 100
frac_redox_error = redox["abs_residual"].sum() / meta2_valid["abs_residual"].sum() * 100
mae_redox = redox["abs_residual"].mean()
mae_fixed = fixed["abs_residual"].mean()
r2_redox = r2_score(redox["target"] if "target" in redox else y2.loc[redox.index], pred_main_s2[redox.index])
r2_fixed = r2_score(y2.loc[fixed.index], pred_main_s2[fixed.index])

print(f"\n--- Redox-active-cation diagnosis (Source 1 -> Source 2) ---")
print(f"Redox-active-cation compounds: n={n_redox} ({frac_redox_compounds:.1f}% of {len(meta2_valid)}), "
      f"MAE={mae_redox:.3f} eV, R2={r2_redox:.3f}, share of total abs. error={frac_redox_error:.1f}%")
print(f"Fixed-valence-only compounds: n={n_fixed} ({100-frac_redox_compounds:.1f}%), "
      f"MAE={mae_fixed:.3f} eV, R2={r2_fixed:.3f}")

# Also check bias direction by class (does the model over- or under-predict
# each class differently?)
bias_redox = redox["residual"].mean()
bias_fixed = fixed["residual"].mean()
print(f"Mean signed residual (pred - true): redox-active={bias_redox:.3f} eV, fixed-valence={bias_fixed:.3f} eV")

redox_summary = {
    "n_redox_active": int(n_redox), "n_fixed_valence": int(n_fixed),
    "n_unresolved": int(n_unresolved),
    "frac_redox_compounds_pct": float(frac_redox_compounds),
    "frac_redox_error_pct": float(frac_redox_error),
    "mae_redox": float(mae_redox), "mae_fixed": float(mae_fixed),
    "r2_redox": float(r2_redox), "r2_fixed": float(r2_fixed),
    "bias_redox": float(bias_redox), "bias_fixed": float(bias_fixed),
    "redox_active_elements": sorted(REDOX_ACTIVE),
}

# Save per-compound table for the SI / figure script
meta2_valid[["composition", "target", "redox_active", "residual", "abs_residual"]].to_csv(
    f"{ROOT}/data/redox_diagnosis_source2.csv", index=False)

# ============================================================
# Save everything
# ============================================================
out = {"fine_tuning": ft_summary, "redox_diagnosis": redox_summary}
with open(f"{ROOT}/data/phase2b_results.json", "w") as f:
    json.dump(out, f, indent=2)

# Save full per-seed/per-k fine-tuning data for the figure script
rows = []
for k in k_values:
    for s in range(n_seeds):
        rows.append({"k": k, "seed": s, "r2": ft_results[k]["r2"][s], "mae": ft_results[k]["mae"][s]})
pd.DataFrame(rows).to_csv(f"{ROOT}/data/finetuning_curve_raw.csv", index=False)

print("\nSaved data/phase2b_results.json, data/finetuning_curve_raw.csv, "
      "data/redox_diagnosis_source2.csv")
print("DONE")
