# Re-estimates the offset correction from a held-out calibration subset only, using the canonical Source-1 model.
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

X2_full = pd.read_csv(f"{ROOT}/data/featurized_source2.csv")
y2 = X2_full.pop("target")
X2 = X2_full
meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
assert np.allclose(meta2["target"].values, y2.values)
pred2 = pd.read_csv(f"{ROOT}/data/predictions_source2.csv")
assert np.allclose(pred2["true"].values, y2.values)

gbm_main = joblib.load(f"{ROOT}/models/gbm_source1_main.joblib")
pred_full_s2 = gbm_main.predict(X2)
assert np.allclose(pred_full_s2, pred2["pred"].values, atol=1e-6), "still mismatched vs canonical predictions!"
resid_full_s2 = pred_full_s2 - y2.values
orig_r2 = r2_score(y2.values, pred_full_s2)
orig_mae = mean_absolute_error(y2.values, pred_full_s2)
orig_offset = float(np.mean(resid_full_s2))
print(f"Canonical-model raw R2={orig_r2:.4f} MAE={orig_mae:.4f} offset={orig_offset:.4f} "
      f"(should match Table 1: R2=-1.898, MAE=2.045)")


def composition_grouped_split(meta, test_frac_rows, seed):
    comp_sizes = meta.groupby("composition").size()
    unique_comps = comp_sizes.index.to_numpy()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(unique_comps)
    target_rows = int(test_frac_rows * len(meta))
    test_comps, cum = [], 0
    for c in perm:
        if cum >= target_rows:
            break
        test_comps.append(c)
        cum += comp_sizes[c]
    return meta["composition"].isin(set(test_comps)).values


N_CAL_REPEATS = 50
cal_r2s, cal_maes, cal_offsets = [], [], []
for rep in range(N_CAL_REPEATS):
    cal_mask = composition_grouped_split(meta2, 0.2, seed=1000 + rep)
    cal_resid = resid_full_s2[cal_mask]
    eval_resid = resid_full_s2[~cal_mask]
    eval_true = y2.values[~cal_mask]
    eval_pred = pred_full_s2[~cal_mask]
    offset_est = float(np.mean(cal_resid))
    corrected_pred = eval_pred - offset_est
    cal_r2s.append(r2_score(eval_true, corrected_pred))
    cal_maes.append(mean_absolute_error(eval_true, corrected_pred))
    cal_offsets.append(offset_est)

item5 = {
    "diagnostic_test_label_informed": {
        "note": "The ORIGINAL offset correction already reported elsewhere in "
                "this paper (Table 2): the offset is estimated from the mean "
                "residual of the SAME compounds used for evaluation. "
                "Retrospective diagnostic only, not a deployable estimate.",
        "r2": float(orig_r2), "mae": float(orig_mae), "offset_eV": orig_offset,
    },
    "realistic_heldout_calibration": {
        "note": "Offset estimated from an independent ~20%-of-rows, "
                "composition-grouped CALIBRATION subset (never seen by the "
                "eval set), applied to the untouched remaining ~80% EVAL "
                "subset. Repeated over 50 independent random calibration/eval "
                "splits; reporting the distribution, not a single lucky split.",
        "n_repeats": N_CAL_REPEATS,
        "r2_mean": float(np.mean(cal_r2s)), "r2_std": float(np.std(cal_r2s)),
        "r2_p5": float(np.percentile(cal_r2s, 5)), "r2_p95": float(np.percentile(cal_r2s, 95)),
        "mae_mean": float(np.mean(cal_maes)), "mae_std": float(np.std(cal_maes)),
        "offset_mean_eV": float(np.mean(cal_offsets)), "offset_std_eV": float(np.std(cal_offsets)),
    },
}
print(f"Realistic held-out calibration (50 repeats): R2={np.mean(cal_r2s):.3f}+-{np.std(cal_r2s):.3f} "
      f"[{np.percentile(cal_r2s,5):.3f}, {np.percentile(cal_r2s,95):.3f}], "
      f"MAE={np.mean(cal_maes):.3f}+-{np.std(cal_maes):.3f}")

results = json.load(open(f"{ROOT}/data/phase4_v1_revisions.json"))
results["item5_calibration_subset_offset"] = item5
with open(f"{ROOT}/data/phase4_v1_revisions.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Fixed and saved data/phase4_v1_revisions.json")
