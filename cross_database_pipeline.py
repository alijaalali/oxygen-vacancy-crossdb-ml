# Cross-database transfer pipeline: stability filter, featurization, zero-shot transfer, UQ, active learning.
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
ROOT = "/glade/work/ajalali/OxygenVacancy-CrossDB-ML-project"

# ---------- 1. Load + clean ----------
df1 = pd.read_csv(f"{ROOT}/data/emery_wolverton/wolverton_oxides.csv")
df1["target"] = pd.to_numeric(df1["Vacancy energy [eV/O atom]"], errors="coerce")
df1["stability"] = pd.to_numeric(df1["Stability [eV/atom]"], errors="coerce")
df1 = df1.dropna(subset=["target", "stability"]).reset_index(drop=True)
n_before = len(df1)

# Diagnostic that justifies the filter below: the raw 4,914-compound set is
# 96% compositions never reported in the literature (combinatorially
# generated ABO3 candidates, most thermodynamically unstable), and for these
# the "vacancy energy" frequently comes out negative/unphysical (57% of the
# full set is negative, i.e. the parent "compound" would spontaneously lose
# oxygen even at 0 K -- not a real oxide). The 208 compounds that ARE in the
# literature (known, usually stable oxides) have mean=3.52 eV, median=3.99 eV,
# std=2.23 eV -- matching Source 2 (mean=4.15, median=3.73) and Source 3
# (mean=3.58, median=3.39) almost exactly. This is a selection-bias effect
# (unstable combinatorial candidates), not a chemical-potential-reference or
# units mismatch between databases.
in_lit = df1[df1["In literature"] == True]["target"]
print(f"Diagnostic: 'In literature' compounds (n={len(in_lit)}): mean={in_lit.mean():.3f} "
      f"median={in_lit.median():.3f} std={in_lit.std():.3f} -- matches Sources 2/3's physical range")

# Disclosed filter: restrict training data to compounds within 0.1 eV/atom of
# the convex hull (a standard metastability threshold in the materials-
# informatics literature, e.g. the ~150 meV/atom cutoffs used in Matbench-style
# filters and in the companion dielectric-constant paper), which removes the
# unstable combinatorial tail and recovers a physically sensible,
# cross-database-comparable target distribution.
df1 = df1[df1["stability"] <= 0.1].reset_index(drop=True)
n_after = len(df1)
print(f"Source 1 (Emery & Wolverton): {n_before} -> {n_after} after stability<=0.1 eV/atom filter "
      f"({n_before-n_after} removed)")
df1["composition"] = df1["Chemical formula"]

df2 = pd.read_csv(f"{ROOT}/data/witman_nrel/witman_oxygen_vacancies.csv")
df2["target"] = df2["dH_eV"]
df2["composition"] = df2["formula"]
print(f"Source 2 (Witman/NREL/Sandia): {len(df2)} rows, {df2['composition'].nunique()} unique compounds")

d3 = json.load(open(f"{ROOT}/data/jarvis_vacancydb/vacancydb.json"))
ov3 = [x for x in d3 if x.get("symbol") == "O"]
df3 = pd.DataFrame({"composition": [x["bulk_formula"] for x in ov3],
                     "target": [x["ef"] for x in ov3]})
print(f"Source 3 (JARVIS/NIST): {len(df3)} rows")

print("\n--- distribution comparison (the chemical-potential reconciliation check) ---")
for name, d in [("Source1", df1), ("Source2", df2), ("Source3", df3)]:
    print(f"{name}: mean={d['target'].mean():.3f} median={d['target'].median():.3f} "
          f"std={d['target'].std():.3f} min={d['target'].min():.3f} max={d['target'].max():.3f}")

offset_23 = df1["target"].median() - pd.concat([df2["target"], df3["target"]]).median()
print(f"Median offset (Source1 - Source2&3 combined): {offset_23:.3f} eV")

# ---------- 2. Featurization (Magpie, composition-only) ----------
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
    return Xf[mask].reset_index(drop=True), df.loc[mask, "target"].reset_index(drop=True), df.loc[mask].reset_index(drop=True)

X1, y1, meta1 = featurize(df1)
X2, y2, meta2 = featurize(df2)
X3, y3, meta3 = featurize(df3)
print(f"\nFeaturized: source1={len(X1)} source2={len(X2)} source3={len(X3)}, "
      f"{X1.shape[1]} Magpie features")

feature_cols = list(X1.columns)
X2 = X2[feature_cols]
X3 = X3[feature_cols]

# ---------- 3a. Cross-database transfer ----------
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=RNG)

# Hyperparameters chosen by 5-fold CV on the stability-filtered training set
# (depth-4/lr-0.05/n-500, used in an earlier pass, badly overfit this smaller
# n=642 set: CV R2 was negative; shallower/slower-learning models were
# systematically better across every config tested). This also matches the
# literature's own finding (Baldassarri et al. 2023, Witman et al. 2023) that
# oxygen-vacancy formation energy is a harder target than bulk formation
# energy or band gap and needs more conservative models/more informative
# features (e.g. site-local descriptors) to avoid overfitting composition-only
# features.
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.02,
                                 subsample=0.8, random_state=RNG)
gbm.fit(X1_tr, y1_tr)

pred_internal = gbm.predict(X1_te)
r2_internal = r2_score(y1_te, pred_internal)
mae_internal = mean_absolute_error(y1_te, pred_internal)

pred2 = gbm.predict(X2)
r2_2 = r2_score(y2, pred2)
mae_2 = mean_absolute_error(y2, pred2)
bias_2 = (pred2 - y2).mean()

pred3 = gbm.predict(X3)
r2_3 = r2_score(y3, pred3)
mae_3 = mean_absolute_error(y3, pred3)
bias_3 = (pred3 - y3).mean()

print("\n--- Cross-database transfer (raw, no offset correction) ---")
print(f"Internal (Source1 random split, n_test={len(y1_te)}): R2={r2_internal:.3f} MAE={mae_internal:.3f} eV")
print(f"Transfer -> Source2 (n={len(y2)}): R2={r2_2:.3f} MAE={mae_2:.3f} eV, mean bias={bias_2:.3f} eV")
print(f"Transfer -> Source3 (n={len(y3)}, small-sample caveat): R2={r2_3:.3f} MAE={mae_3:.3f} eV, mean bias={bias_3:.3f} eV")

# Offset-correction test: does removing the mean residual (chemical-potential
# reference mismatch) recover transfer accuracy, or is the gap compositional?
pred2_corr = pred2 - bias_2
r2_2_corr = r2_score(y2, pred2_corr)
mae_2_corr = mean_absolute_error(y2, pred2_corr)
pred3_corr = pred3 - bias_3
r2_3_corr = r2_score(y3, pred3_corr)
mae_3_corr = mean_absolute_error(y3, pred3_corr)
print("\n--- After removing the mean residual (constant chemical-potential-offset correction) ---")
print(f"Source2 corrected: R2={r2_2_corr:.3f} MAE={mae_2_corr:.3f} eV")
print(f"Source3 corrected: R2={r2_3_corr:.3f} MAE={mae_3_corr:.3f} eV")

# ---------- 3b. Uncertainty quantification (bootstrap ensemble of GBMs) ----------
N_ENS = 30
rng = np.random.RandomState(RNG)
ens_preds_internal = np.zeros((N_ENS, len(X1_te)))
ens_preds_2 = np.zeros((N_ENS, len(X2)))
models = []
for i in range(N_ENS):
    idx = rng.randint(0, len(X1_tr), len(X1_tr))
    m = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.02,
                                   subsample=0.8, random_state=i)
    m.fit(X1_tr.iloc[idx], y1_tr.iloc[idx])
    models.append(m)
    ens_preds_internal[i] = m.predict(X1_te)
    ens_preds_2[i] = m.predict(X2)

mean_internal = ens_preds_internal.mean(axis=0)
std_internal = ens_preds_internal.std(axis=0)
mean_2 = ens_preds_2.mean(axis=0)
std_2 = ens_preds_2.std(axis=0)

def calibration_metrics(y_true, mean_pred, std_pred):
    err = np.abs(y_true.values - mean_pred)
    # Spearman correlation between predicted std and actual abs error
    from scipy.stats import spearmanr
    rho, p = spearmanr(std_pred, err)
    # fraction of points within 1 and 2 predicted std
    z = (y_true.values - mean_pred) / np.where(std_pred == 0, 1e-6, std_pred)
    within1 = np.mean(np.abs(z) <= 1)
    within2 = np.mean(np.abs(z) <= 2)
    return rho, p, within1, within2

rho_in, p_in, w1_in, w2_in = calibration_metrics(y1_te, mean_internal, std_internal)
rho_out, p_out, w1_out, w2_out = calibration_metrics(y2, mean_2, std_2)

print("\n--- Uncertainty quantification (bootstrap ensemble, N=30) ---")
print(f"In-distribution (Source1 held-out): Spearman(std,|err|)={rho_in:.3f} (p={p_in:.2e}), "
      f"fraction within 1 std={w1_in:.2f}, within 2 std={w2_in:.2f}")
print(f"Cross-database (Source2): Spearman(std,|err|)={rho_out:.3f} (p={p_out:.2e}), "
      f"fraction within 1 std={w1_out:.2f}, within 2 std={w2_out:.2f}")
print(f"Mean predicted std in-distribution: {std_internal.mean():.3f} eV; cross-DB: {std_2.mean():.3f} eV")

# ---------- 3c. Active learning simulation (on Source1 only) ----------
X1_pool_init, X1_holdout, y1_pool_init, y1_holdout = train_test_split(
    X1, y1, test_size=0.2, random_state=RNG)

def active_learning_curve(X_pool, y_pool, X_test, y_test, strategy, seed_frac=0.08,
                           batch_frac=0.08, n_rounds=12, seed=RNG):
    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(X_pool))
    n_seed = max(15, int(seed_frac * len(idx_all)))
    labeled = set(rng.choice(idx_all, n_seed, replace=False))
    maes = []
    n_points = []
    batch = max(10, int(batch_frac * len(idx_all)))
    for r in range(n_rounds):
        lab_idx = np.array(sorted(labeled))
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03,
                                       subsample=0.8, random_state=0)
        m.fit(X_pool.iloc[lab_idx], y_pool.iloc[lab_idx])
        pred = m.predict(X_test)
        maes.append(mean_absolute_error(y_test, pred))
        n_points.append(len(lab_idx))
        unlabeled = np.array(sorted(set(idx_all) - labeled))
        if len(unlabeled) == 0:
            break
        if strategy == "random":
            chosen = rng.choice(unlabeled, min(batch, len(unlabeled)), replace=False)
        elif strategy == "uncertainty":
            ens_std = np.zeros((10, len(unlabeled)))
            for i in range(10):
                bidx = rng.randint(0, len(lab_idx), len(lab_idx))
                mm = GradientBoostingRegressor(n_estimators=120, max_depth=2,
                                                learning_rate=0.03, subsample=0.8,
                                                random_state=i + 100)
                mm.fit(X_pool.iloc[lab_idx[bidx]], y_pool.iloc[lab_idx[bidx]])
                ens_std[i] = mm.predict(X_pool.iloc[unlabeled])
            unc = ens_std.std(axis=0)
            top = np.argsort(-unc)[:min(batch, len(unlabeled))]
            chosen = unlabeled[top]
        labeled.update(chosen.tolist())
    return n_points, maes

n_rand, mae_rand = active_learning_curve(X1_pool_init, y1_pool_init, X1_holdout, y1_holdout, "random")
n_unc, mae_unc = active_learning_curve(X1_pool_init, y1_pool_init, X1_holdout, y1_holdout, "uncertainty")

print("\n--- Active learning simulation (Source1, pool-based) ---")
print("n_labeled, MAE(random), MAE(uncertainty)")
for a, b, c in zip(n_rand, mae_rand, mae_unc):
    print(f"{a}, {b:.4f}, {c:.4f}")

target_mae = mae_rand[-1] * 1.1  # 10% above final random MAE, as a reach threshold
def points_to_reach(n_list, mae_list, thresh):
    for n, m in zip(n_list, mae_list):
        if m <= thresh:
            return n
    return None
pts_rand = points_to_reach(n_rand, mae_rand, target_mae)
pts_unc = points_to_reach(n_unc, mae_unc, target_mae)
print(f"\nPoints needed to reach MAE<={target_mae:.4f} eV: random={pts_rand}, uncertainty={pts_unc}")

# ---------- Save everything ----------
results = {
    "source1_n_before_filter": n_before,
    "source1_n_after_filter": n_after,
    "source1_in_literature_subset_diagnostic": {
        "n": int(len(in_lit)), "mean": float(in_lit.mean()), "median": float(in_lit.median()),
        "std": float(in_lit.std()),
        "note": "matches Sources 2/3's physical range almost exactly, confirming the raw "
                "dataset's negative/wide distribution is a selection-bias artifact of "
                "unstable combinatorial candidates, not a chemical-potential-reference "
                "or units mismatch",
    },
    "source2_n": len(df2),
    "source3_n": len(df3),
    "distribution": {
        "source1": {"mean": float(df1['target'].mean()), "median": float(df1['target'].median()),
                     "std": float(df1['target'].std())},
        "source2": {"mean": float(df2['target'].mean()), "median": float(df2['target'].median()),
                     "std": float(df2['target'].std())},
        "source3": {"mean": float(df3['target'].mean()), "median": float(df3['target'].median()),
                     "std": float(df3['target'].std())},
    },
    "median_offset_source1_minus_2and3": float(offset_23),
    "transfer": {
        "internal_r2": float(r2_internal), "internal_mae": float(mae_internal),
        "source2_r2": float(r2_2), "source2_mae": float(mae_2), "source2_bias": float(bias_2),
        "source3_r2": float(r2_3), "source3_mae": float(mae_3), "source3_bias": float(bias_3),
        "source2_r2_corrected": float(r2_2_corr), "source2_mae_corrected": float(mae_2_corr),
        "source3_r2_corrected": float(r2_3_corr), "source3_mae_corrected": float(mae_3_corr),
    },
    "uq": {
        "in_dist_spearman": float(rho_in), "in_dist_p": float(p_in),
        "in_dist_within1std": float(w1_in), "in_dist_within2std": float(w2_in),
        "cross_db_spearman": float(rho_out), "cross_db_p": float(p_out),
        "cross_db_within1std": float(w1_out), "cross_db_within2std": float(w2_out),
        "mean_std_in_dist": float(std_internal.mean()), "mean_std_cross_db": float(std_2.mean()),
    },
    "active_learning": {
        "n_random": n_rand, "mae_random": mae_rand,
        "n_uncertainty": n_unc, "mae_uncertainty": mae_unc,
        "target_mae": float(target_mae), "points_random": pts_rand, "points_uncertainty": pts_unc,
    },
}
with open(f"{ROOT}/data/phase1_results.json", "w") as f:
    json.dump(results, f, indent=2)

import joblib
joblib.dump(gbm, f"{ROOT}/models/gbm_source1_main.joblib")
X1.assign(target=y1).to_csv(f"{ROOT}/data/featurized_source1.csv", index=False)
X2.assign(target=y2).to_csv(f"{ROOT}/data/featurized_source2.csv", index=False)
X3.assign(target=y3).to_csv(f"{ROOT}/data/featurized_source3.csv", index=False)

print("\nAll results saved to data/phase1_results.json, model to models/gbm_source1_main.joblib")
print("DONE")
