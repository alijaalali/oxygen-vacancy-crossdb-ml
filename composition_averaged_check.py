"""Composition-averaged Source-2 check: aggregates the 795 Source-2
vacancy-site rows to 182 one-row-per-composition means (since Magpie
composition features are already identical for every site of the same
composition, this only changes the target, averaging over different
site-specific vacancy energies), and re-runs the zero-shot transfer and a
joint-vs-target-only label-efficiency check at this composition-averaged
level, to see whether the headline conclusions change.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
RNG = 0
GBM_KW = dict(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8)

X1_full = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
y1 = X1_full.pop("target")
X1 = X1_full

X2_full = pd.read_csv(f"{ROOT}/data/featurized_source2.csv")
y2 = X2_full.pop("target")
X2 = X2_full[X1.columns]
meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
assert np.allclose(meta2["target"].values, y2.values)

# Composition-averaged Source-2: one row per unique composition, feature
# vector unchanged (identical across sites of the same composition),
# target = mean of the site-specific vacancy energies.
df2 = X2.copy()
df2["composition"] = meta2["composition"].values
df2["target"] = y2.values
site_counts = df2.groupby("composition").size()
agg = df2.groupby("composition").agg({**{c: "first" for c in X1.columns}, "target": "mean"}).reset_index()
within_comp_std = df2.groupby("composition")["target"].std().fillna(0.0)
print(f"Source-2: {len(df2)} site-level rows -> {len(agg)} unique compositions")
print(f"Mean within-composition target std (sites sharing identical features): {within_comp_std.mean():.4f} eV/O")
print(f"Compositions with >1 site: {(site_counts > 1).sum()} / {len(site_counts)}")

X2_agg = agg[X1.columns]
y2_agg = agg["target"]

# ---- Zero-shot transfer, composition-averaged ----
gbm_main = joblib.load(f"{ROOT}/models/gbm_source1_main.joblib")
pred_agg = gbm_main.predict(X2_agg)
r2_agg = r2_score(y2_agg, pred_agg)
mae_agg = mean_absolute_error(y2_agg, pred_agg)
bias_agg = (pred_agg - y2_agg).mean()
r2_agg_corr = r2_score(y2_agg, pred_agg - bias_agg)
mae_agg_corr = mean_absolute_error(y2_agg, pred_agg - bias_agg)
print(f"\nZero-shot (composition-averaged, n={len(y2_agg)}): "
      f"raw R2={r2_agg:.3f} MAE={mae_agg:.3f}, bias-corrected R2={r2_agg_corr:.3f} MAE={mae_agg_corr:.3f}")

# site-level reference (same model, same compounds, un-aggregated)
r2_site = r2_score(y2, gbm_main.predict(X2))
mae_site = mean_absolute_error(y2, gbm_main.predict(X2))
print(f"Zero-shot (site-level, n={len(y2)}): raw R2={r2_site:.3f} MAE={mae_site:.3f} (for comparison)")

# ---- Joint-vs-target-only at composition-averaged level, k=10,30 (pool is only 182 compositions now) ----
rng_split = np.random.RandomState(RNG)
comps = agg["composition"].to_numpy()
perm = rng_split.permutation(len(comps))
n_test = int(0.2 * len(comps))
test_idx, pool_idx = perm[:n_test], perm[n_test:]
X2a_test, y2a_test = X2_agg.iloc[test_idx].reset_index(drop=True), y2_agg.iloc[test_idx].reset_index(drop=True)
X2a_pool, y2a_pool = X2_agg.iloc[pool_idx].reset_index(drop=True), y2_agg.iloc[pool_idx].reset_index(drop=True)
print(f"\nComposition-averaged pool={len(X2a_pool)}, test={len(X2a_test)}")

k_list = [10, 30]
n_seeds = 10
results = {"joint": {}, "target_only": {}}
for k in k_list:
    k_eff = min(k, len(X2a_pool))
    results["joint"][k] = {"r2": [], "mae": []}
    results["target_only"][k] = {"r2": [], "mae": []}
    for seed in range(n_seeds):
        rr = np.random.RandomState(seed)
        idx = rr.permutation(len(X2a_pool))[:k_eff]
        X2k, y2k = X2a_pool.iloc[idx], y2a_pool.iloc[idx]

        Xtr = pd.concat([X1, X2k], ignore_index=True)
        ytr = pd.concat([y1, y2k], ignore_index=True)
        m = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        m.fit(Xtr, ytr)
        pred = m.predict(X2a_test)
        results["joint"][k]["r2"].append(r2_score(y2a_test, pred))
        results["joint"][k]["mae"].append(mean_absolute_error(y2a_test, pred))

        ms = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        ms.fit(X2k, y2k)
        preds2 = ms.predict(X2a_test)
        results["target_only"][k]["r2"].append(r2_score(y2a_test, preds2))
        results["target_only"][k]["mae"].append(mean_absolute_error(y2a_test, preds2))
    print(f"k={k} (n_eff={k_eff}): joint R2={np.mean(results['joint'][k]['r2']):.3f} "
          f"target-only R2={np.mean(results['target_only'][k]['r2']):.3f}")

out = {
    "n_site_rows": int(len(df2)), "n_unique_compositions": int(len(agg)),
    "mean_within_composition_target_std": float(within_comp_std.mean()),
    "n_compositions_multi_site": int((site_counts > 1).sum()),
    "zero_shot_composition_averaged": {"n": int(len(y2_agg)), "r2_raw": float(r2_agg), "mae_raw": float(mae_agg),
                                        "r2_bias_corrected": float(r2_agg_corr), "mae_bias_corrected": float(mae_agg_corr)},
    "zero_shot_site_level_reference": {"n": int(len(y2)), "r2_raw": float(r2_site), "mae_raw": float(mae_site)},
    "label_efficiency_composition_averaged": {
        key: {str(k): {"r2_mean": float(np.mean(d["r2"])), "r2_std": float(np.std(d["r2"])),
                        "mae_mean": float(np.mean(d["mae"])), "mae_std": float(np.std(d["mae"]))}
              for k, d in byk.items()}
        for key, byk in results.items()
    },
}
json.dump(out, open(f"{ROOT}/data/phase5_composition_averaged_source2.json", "w"), indent=2)
print("\nSaved data/phase5_composition_averaged_source2.json")
