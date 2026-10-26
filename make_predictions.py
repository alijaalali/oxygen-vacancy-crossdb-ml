"""
Regenerate per-compound predictions (parity-plot data + UQ ensemble) from the
already-featurized tables and the exact same model configuration used in
phase1_pipeline.py. Deterministic given the same random_state, so the
resulting summary metrics are checked against data/phase1_results.json
before anything downstream trusts them.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

df1 = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
df2 = pd.read_csv(f"{ROOT}/data/featurized_source2.csv")
df3 = pd.read_csv(f"{ROOT}/data/featurized_source3.csv")

feature_cols = [c for c in df1.columns if c != "target"]
X1, y1 = df1[feature_cols], df1["target"]
X2, y2 = df2[feature_cols], df2["target"]
X3, y3 = df3[feature_cols], df3["target"]

X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=RNG)

gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.02,
                                 subsample=0.8, random_state=RNG)
gbm.fit(X1_tr, y1_tr)

pred_internal = gbm.predict(X1_te)
pred2 = gbm.predict(X2)
pred3 = gbm.predict(X3)

r2_internal = r2_score(y1_te, pred_internal)
mae_internal = mean_absolute_error(y1_te, pred_internal)
r2_2 = r2_score(y2, pred2)
mae_2 = mean_absolute_error(y2, pred2)
bias_2 = (pred2 - y2).mean()
r2_3 = r2_score(y3, pred3)
mae_3 = mean_absolute_error(y3, pred3)
bias_3 = (pred3 - y3).mean()

ref = json.load(open(f"{ROOT}/data/phase1_results.json"))
checks = [
    ("internal_r2", r2_internal), ("internal_mae", mae_internal),
    ("source2_r2", r2_2), ("source2_mae", mae_2), ("source2_bias", bias_2),
    ("source3_r2", r2_3), ("source3_mae", mae_3), ("source3_bias", bias_3),
]
print("--- reproducibility check against data/phase1_results.json ---")
all_ok = True
for key, val in checks:
    ref_val = ref["transfer"][key]
    ok = abs(val - ref_val) < 1e-6
    all_ok &= ok
    print(f"{key}: recomputed={val:.6f} saved={ref_val:.6f} match={ok}")
if not all_ok:
    raise SystemExit("MISMATCH vs saved phase1_results.json -- do not trust these predictions")
print("All metrics match exactly. Proceeding to save predictions.")

pd.DataFrame({"true": y1_te.values, "pred": pred_internal}).to_csv(
    f"{ROOT}/data/predictions_internal.csv", index=False)
pd.DataFrame({"true": y2.values, "pred": pred2, "composition": df2.index}).to_csv(
    f"{ROOT}/data/predictions_source2.csv", index=False)
pd.DataFrame({"true": y3.values, "pred": pred3}).to_csv(
    f"{ROOT}/data/predictions_source3.csv", index=False)

# ---- UQ bootstrap ensemble (matches phase1_pipeline.py exactly) ----
N_ENS = 30
rng = np.random.RandomState(RNG)
ens_preds_internal = np.zeros((N_ENS, len(X1_te)))
ens_preds_2 = np.zeros((N_ENS, len(X2)))
for i in range(N_ENS):
    idx = rng.randint(0, len(X1_tr), len(X1_tr))
    m = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.02,
                                   subsample=0.8, random_state=i)
    m.fit(X1_tr.iloc[idx], y1_tr.iloc[idx])
    ens_preds_internal[i] = m.predict(X1_te)
    ens_preds_2[i] = m.predict(X2)

mean_internal = ens_preds_internal.mean(axis=0)
std_internal = ens_preds_internal.std(axis=0)
mean_2 = ens_preds_2.mean(axis=0)
std_2 = ens_preds_2.std(axis=0)

rho_in, p_in = spearmanr(std_internal, np.abs(y1_te.values - mean_internal))
rho_out, p_out = spearmanr(std_2, np.abs(y2.values - mean_2))
print(f"\nUQ reproducibility check: in_dist rho={rho_in:.6f} (saved {ref['uq']['in_dist_spearman']:.6f}), "
      f"cross_db rho={rho_out:.6f} (saved {ref['uq']['cross_db_spearman']:.6f})")

pd.DataFrame({
    "true": y1_te.values, "pred_mean": mean_internal, "pred_std": std_internal,
}).to_csv(f"{ROOT}/data/uq_internal.csv", index=False)
pd.DataFrame({
    "true": y2.values, "pred_mean": mean_2, "pred_std": std_2,
}).to_csv(f"{ROOT}/data/uq_source2.csv", index=False)

print("\nSaved: predictions_internal.csv, predictions_source2.csv, predictions_source3.csv, "
      "uq_internal.csv, uq_source2.csv")
print("DONE")
