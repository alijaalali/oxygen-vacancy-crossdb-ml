# Stability-threshold sensitivity sweep, size-matched subgroup control, composition-overlap check, bootstrap CIs.
import json, warnings
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

# ---------- Load raw Source 1 (before any stability filter) ----------
df1_raw = pd.read_csv(f"{ROOT}/data/emery_wolverton/wolverton_oxides.csv")
df1_raw["target"] = pd.to_numeric(df1_raw["Vacancy energy [eV/O atom]"], errors="coerce")
df1_raw["stability"] = pd.to_numeric(df1_raw["Stability [eV/atom]"], errors="coerce")
df1_raw = df1_raw.dropna(subset=["target", "stability"]).reset_index(drop=True)
df1_raw["composition"] = df1_raw["Chemical formula"]
n_before = len(df1_raw)
print(f"Raw Source 1 (valid target+stability): {n_before}")

df2 = pd.read_csv(f"{ROOT}/data/witman_nrel/witman_oxygen_vacancies.csv")
df2["target"] = df2["dH_eV"]
df2["composition"] = df2["formula"]

d3 = json.load(open(f"{ROOT}/data/jarvis_vacancydb/vacancydb.json"))
ov3 = [x for x in d3 if x.get("symbol") == "O"]
df3 = pd.DataFrame({"composition": [x["bulk_formula"] for x in ov3],
                     "target": [x["ef"] for x in ov3]})

# ---------- Featurize the FULL raw Source 1 once (cache), then slice by threshold ----------
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
    return Xf[mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)

print("Featurizing full raw Source 1 (cached, reused across all thresholds)...")
X1_full, meta1_full = featurize(df1_raw)
print(f"Featurized Source 1 (all stabilities): {len(X1_full)} rows")

X2, meta2 = featurize(df2)
X3, meta3 = featurize(df3)
feature_cols = list(X1_full.columns)
X2 = X2[feature_cols]
X3 = X3[feature_cols]
y2 = meta2["target"].reset_index(drop=True)
y3 = meta3["target"].reset_index(drop=True)
print(f"Featurized Source2={len(X2)} Source3={len(X3)}")

GBM_KW = dict(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8, random_state=RNG)

# =========================================================================
# ITEM 2: stability-threshold sensitivity sweep
# =========================================================================
print("\n" + "=" * 70)
print("ITEM 2: stability-threshold sensitivity sweep")
print("=" * 70)

thresholds = [0.05, 0.1, 0.2]
sweep_results = []
for thresh in thresholds:
    keep = meta1_full["stability"] <= thresh
    Xt = X1_full[keep].reset_index(drop=True)
    yt = meta1_full.loc[keep, "target"].reset_index(drop=True)
    n = len(Xt)
    Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.2, random_state=RNG)
    gbm = GradientBoostingRegressor(**GBM_KW)
    gbm.fit(Xtr, ytr)
    r2_int = r2_score(yte, gbm.predict(Xte))
    mae_int = mean_absolute_error(yte, gbm.predict(Xte))
    pred2 = gbm.predict(X2)
    r2_2 = r2_score(y2, pred2)
    mae_2 = mean_absolute_error(y2, pred2)
    bias_2 = (pred2 - y2).mean()
    r2_2_corr = r2_score(y2, pred2 - bias_2)
    mae_2_corr = mean_absolute_error(y2, pred2 - bias_2)
    row = dict(threshold=thresh, n_compounds=n, internal_r2=float(r2_int), internal_mae=float(mae_int),
               source2_r2=float(r2_2), source2_mae=float(mae_2), source2_bias=float(bias_2),
               source2_r2_corrected=float(r2_2_corr), source2_mae_corrected=float(mae_2_corr))
    sweep_results.append(row)
    print(f"threshold={thresh:.2f} eV/atom: n={n}, internal R2={r2_int:.3f} MAE={mae_int:.3f}, "
          f"Source2 R2={r2_2:.3f} MAE={mae_2:.3f} bias={bias_2:.3f}, "
          f"Source2 corrected R2={r2_2_corr:.3f} MAE={mae_2_corr:.3f}")

# =========================================================================
# ITEM 4: composition-overlap/leakage check, Source1 (0.1 filter) vs Source2
# =========================================================================
print("\n" + "=" * 70)
print("ITEM 4: composition-overlap / leakage check, Source1 vs Source2")
print("=" * 70)

keep01 = meta1_full["stability"] <= 0.1
X1 = X1_full[keep01].reset_index(drop=True)
meta1 = meta1_full.loc[keep01].reset_index(drop=True)
y1 = meta1["target"].reset_index(drop=True)
assert len(X1) == 644, f"expected 644, got {len(X1)}"

def reduced_formula_set(formulas):
    out = {}
    for f in formulas:
        try:
            rf = Composition(f).reduced_formula
            out.setdefault(rf, []).append(f)
        except Exception:
            pass
    return out

s1_reduced = reduced_formula_set(meta1["composition"])
s2_reduced = reduced_formula_set(df2["composition"])
overlap_formulas = sorted(set(s1_reduced.keys()) & set(s2_reduced.keys()))
print(f"Source1 (644, filtered): {len(s1_reduced)} distinct reduced formulas")
print(f"Source2 (795 rows): {len(s2_reduced)} distinct reduced formulas")
print(f"Exact reduced-formula overlap: {len(overlap_formulas)} formulas: {overlap_formulas}")

# Which Source-2 ROWS correspond to an overlapping reduced formula?
overlap_set = set(overlap_formulas)
def is_overlap(f):
    try:
        return Composition(f).reduced_formula in overlap_set
    except Exception:
        return False

df2_overlap_mask = df2["composition"].apply(is_overlap)
n_overlap_rows = int(df2_overlap_mask.sum())
print(f"Source2 rows whose reduced formula overlaps Source1: {n_overlap_rows} of {len(df2)}")

# Retrain the SAME headline model (0.1 eV/atom filter, same split) and recompute
# Source-2 transfer metrics EXCLUDING the overlapping rows
Xtr1, Xte1, ytr1, yte1 = train_test_split(X1, y1, test_size=0.2, random_state=RNG)
gbm_headline = GradientBoostingRegressor(**GBM_KW)
gbm_headline.fit(Xtr1, ytr1)
pred2_all = gbm_headline.predict(X2)
r2_2_all = r2_score(y2, pred2_all)
mae_2_all = mean_absolute_error(y2, pred2_all)

mask_noleak = ~df2_overlap_mask.reset_index(drop=True).values
X2_noleak = X2[mask_noleak].reset_index(drop=True)
y2_noleak = y2[mask_noleak].reset_index(drop=True)
pred2_noleak = gbm_headline.predict(X2_noleak)
r2_2_noleak = r2_score(y2_noleak, pred2_noleak)
mae_2_noleak = mean_absolute_error(y2_noleak, pred2_noleak)
print(f"Source2 all rows (n={len(y2)}): R2={r2_2_all:.3f} MAE={mae_2_all:.3f}")
print(f"Source2 excluding overlap rows (n={len(y2_noleak)}): R2={r2_2_noleak:.3f} MAE={mae_2_noleak:.3f}")

# =========================================================================
# ITEM 3 & 5a: use the real per-row redox diagnosis predictions already on disk
# =========================================================================
print("\n" + "=" * 70)
print("ITEM 3: size-matched control, redox-active vs fixed-valence")
print("ITEM 5a: grouped bootstrap CIs on headline R2 numbers")
print("=" * 70)

redox_df = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
print(f"Loaded redox_diagnosis_source2.csv: {len(redox_df)} rows "
      f"(redox_active={redox_df['redox_active'].sum()}, fixed={len(redox_df) - redox_df['redox_active'].sum()})")

redox_df["pred"] = redox_df["target"] + redox_df["residual"]
r2_overall_check = r2_score(redox_df["target"], redox_df["pred"])
redox_sub = redox_df[redox_df["redox_active"]]
fixed_sub = redox_df[~redox_df["redox_active"]]
r2_redox_check = r2_score(redox_sub["target"], redox_sub["pred"])
r2_fixed_check = r2_score(fixed_sub["target"], fixed_sub["pred"])
print(f"Sanity check against headline numbers: overall R2={r2_overall_check:.3f} "
      f"(expect -1.898), redox R2={r2_redox_check:.3f} (expect -1.309), "
      f"fixed R2={r2_fixed_check:.3f} (expect -19.868)")

# --- Item 3: size-matched control ---
n_fixed = len(fixed_sub)
rng = np.random.RandomState(RNG)
N_SUB = 1000
redox_vals = redox_sub[["target", "pred"]].values
subsample_r2 = np.zeros(N_SUB)
for i in range(N_SUB):
    idx = rng.choice(len(redox_vals), n_fixed, replace=False)
    sub = redox_vals[idx]
    subsample_r2[i] = r2_score(sub[:, 0], sub[:, 1])

sub_median = float(np.median(subsample_r2))
sub_p5 = float(np.percentile(subsample_r2, 5))
sub_p95 = float(np.percentile(subsample_r2, 95))
sub_min = float(subsample_r2.min())
sub_max = float(subsample_r2.max())
frac_below_fixed = float(np.mean(subsample_r2 <= r2_fixed_check))
print(f"\nSize-matched control (redox-active subsampled to n={n_fixed}, {N_SUB} draws):")
print(f"  R2 distribution: median={sub_median:.3f}, 5-95pct=[{sub_p5:.3f}, {sub_p95:.3f}], "
      f"range=[{sub_min:.3f}, {sub_max:.3f}]")
print(f"  Actual fixed-valence R2 = {r2_fixed_check:.3f}")
print(f"  Fraction of size-matched redox-active draws with R2 <= actual fixed-valence R2: {frac_below_fixed:.4f}")

# --- Item 5a: grouped bootstrap CIs (resample whole compositions together) ---
def grouped_bootstrap_r2(df, n_boot=2000, seed=RNG):
    groups = df["composition"].unique()
    rng = np.random.RandomState(seed)
    out = np.zeros(n_boot)
    grouped = {g: df[df["composition"] == g] for g in groups}
    for b in range(n_boot):
        chosen = rng.choice(groups, len(groups), replace=True)
        parts = [grouped[g] for g in chosen]
        resampled = pd.concat(parts, ignore_index=True)
        out[b] = r2_score(resampled["target"], resampled["pred"])
    return out

boot_overall = grouped_bootstrap_r2(redox_df, seed=RNG)
boot_redox = grouped_bootstrap_r2(redox_sub, seed=RNG + 1)
boot_fixed = grouped_bootstrap_r2(fixed_sub, seed=RNG + 2)

def ci(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

ci_overall = ci(boot_overall)
ci_redox = ci(boot_redox)
ci_fixed = ci(boot_fixed)
print(f"\nGrouped bootstrap 95% CIs (resampling whole compositions, n_boot=2000):")
print(f"  Overall R2 = {r2_overall_check:.3f} [{ci_overall[0]:.3f}, {ci_overall[1]:.3f}]")
print(f"  Redox-active R2 = {r2_redox_check:.3f} [{ci_redox[0]:.3f}, {ci_redox[1]:.3f}]")
print(f"  Fixed-valence R2 = {r2_fixed_check:.3f} [{ci_fixed[0]:.3f}, {ci_fixed[1]:.3f}]")

# =========================================================================
# Save all results
# =========================================================================
out = {
    "item2_threshold_sweep": sweep_results,
    "item4_composition_overlap": {
        "source1_n_distinct_reduced_formulas": len(s1_reduced),
        "source2_n_distinct_reduced_formulas": len(s2_reduced),
        "overlap_formulas": overlap_formulas,
        "n_overlap_formulas": len(overlap_formulas),
        "n_source2_overlap_rows": n_overlap_rows,
        "source2_all_r2": float(r2_2_all), "source2_all_mae": float(mae_2_all),
        "source2_noleak_n": int(mask_noleak.sum()),
        "source2_noleak_r2": float(r2_2_noleak), "source2_noleak_mae": float(mae_2_noleak),
    },
    "item3_size_matched_control": {
        "n_fixed": int(n_fixed), "n_redox_full": int(len(redox_sub)), "n_draws": N_SUB,
        "subsample_r2_median": sub_median, "subsample_r2_p5": sub_p5, "subsample_r2_p95": sub_p95,
        "subsample_r2_min": sub_min, "subsample_r2_max": sub_max,
        "actual_fixed_r2": float(r2_fixed_check),
        "frac_subsample_draws_at_or_below_actual_fixed_r2": frac_below_fixed,
    },
    "item5a_grouped_bootstrap_ci": {
        "overall_r2": float(r2_overall_check), "overall_ci95": ci_overall,
        "redox_r2": float(r2_redox_check), "redox_ci95": ci_redox,
        "fixed_r2": float(r2_fixed_check), "fixed_ci95": ci_fixed,
        "n_boot": 2000,
    },
}
with open(f"{ROOT}/data/phase3_strengthening_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to data/phase3_strengthening_results.json")
print("DONE")
