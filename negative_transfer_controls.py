"""Negative-transfer robustness controls.
Tests whether Source-2-only beating joint Source-1+Source-2 training is
genuine negative transfer, or partly an artifact of Source-1's full
644-compound pool dominating a small Source-2 budget k in the joint pool.
Uses the SAME 644-compound Source-1 pool as the main-text joint-training/
label-efficiency curve (no held-out Source-1 partition here -- evaluation
is always against Source-2's held-out test set, so including the full
Source-1 pool carries no leakage risk).

Control (a): source-data downsampling -- downsample Source-1's
contribution to the joint pool to a scale comparable to k, instead of
always using the full 644 rows, at k=10, 50, 100 under the
composition-grouped (leakage-free) protocol.
Control (b): target-domain sample weighting -- upweight Source-2 rows in
the joint GBM fit (sample_weight supported natively by
GradientBoostingRegressor) so their total weight equals Source-1's.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pymatgen.core import Composition

RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
GBM_KW = dict(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8)

X1_full = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
y1 = X1_full.pop("target")
X1 = X1_full  # full 644-compound Source-1 pool, matching the main-text joint-training protocol

df2 = pd.read_csv(f"{ROOT}/data/featurized_source2.csv")
y2 = df2.pop("target")
X2 = df2[X1.columns]
meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
assert np.allclose(meta2["target"].values, y2.values), "row-order mismatch vs featurized_source2"


def composition_grouped_split(meta, test_frac_rows, seed):
    """Greedy composition-level split targeting ~test_frac_rows of TOTAL ROWS
    in the held-out partition, with every row of a given composition going
    to the same side. Identical to the split used for the main-text joint-
    training/label-efficiency curve (phase4_v1_revisions.py), so this
    script's test partition matches the main curve's exactly."""
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
    test_mask = meta["composition"].isin(set(test_comps))
    return test_mask.values


mask_test = composition_grouped_split(meta2, 0.2, seed=RNG)
X2_test, y2_test = X2[mask_test].reset_index(drop=True), y2[mask_test].reset_index(drop=True)
X2_pool, y2_pool = X2[~mask_test].reset_index(drop=True), y2[~mask_test].reset_index(drop=True)
print(f"Pool={len(X2_pool)} rows, Test={len(X2_test)} rows (composition-grouped, leakage-free)")

k_list = [10, 50, 100]
n_seeds = 10
results = {"downsampled_source1": {}, "weighted_source2": {}, "baseline_full_source1_joint": {}, "source2_only": {}}

for k in k_list:
    for key in results:
        results[key][k] = {"r2": [], "mae": []}
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X2_pool))
        idx = perm[:k]
        X2k, y2k = X2_pool.iloc[idx], y2_pool.iloc[idx]

        # baseline: full 644-row Source-1 + k Source-2 rows (as in main text)
        Xtr = pd.concat([X1, X2k], ignore_index=True)
        ytr = pd.concat([y1, y2k], ignore_index=True)
        m = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        m.fit(Xtr, ytr)
        pred = m.predict(X2_test)
        results["baseline_full_source1_joint"][k]["r2"].append(r2_score(y2_test, pred))
        results["baseline_full_source1_joint"][k]["mae"].append(mean_absolute_error(y2_test, pred))

        # control (a): downsample Source-1 to 5x k (capped at 644)
        n_s1_down = min(len(X1), max(5 * k, 20))
        rs1 = np.random.RandomState(1000 + seed)
        s1_idx = rs1.choice(len(X1), size=n_s1_down, replace=False)
        Xtr_d = pd.concat([X1.iloc[s1_idx], X2k], ignore_index=True)
        ytr_d = pd.concat([y1.iloc[s1_idx], y2k], ignore_index=True)
        md = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        md.fit(Xtr_d, ytr_d)
        predd = md.predict(X2_test)
        results["downsampled_source1"][k]["r2"].append(r2_score(y2_test, predd))
        results["downsampled_source1"][k]["mae"].append(mean_absolute_error(y2_test, predd))

        # control (b): sample-weight Source-2 rows so total S2 weight == total S1 weight
        w1 = np.ones(len(X1))
        w2 = np.full(k, len(X1) / k)
        Xtr_w = pd.concat([X1, X2k], ignore_index=True)
        ytr_w = pd.concat([y1, y2k], ignore_index=True)
        w = np.concatenate([w1, w2])
        mw = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        mw.fit(Xtr_w, ytr_w, sample_weight=w)
        predw = mw.predict(X2_test)
        results["weighted_source2"][k]["r2"].append(r2_score(y2_test, predw))
        results["weighted_source2"][k]["mae"].append(mean_absolute_error(y2_test, predw))

        # source2-only reference (same split, for direct comparison)
        ms = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        ms.fit(X2k, y2k)
        preds2 = ms.predict(X2_test)
        results["source2_only"][k]["r2"].append(r2_score(y2_test, preds2))
        results["source2_only"][k]["mae"].append(mean_absolute_error(y2_test, preds2))
    print(f"k={k} done")

summary = {}
for key, byk in results.items():
    summary[key] = {}
    for k, d in byk.items():
        summary[key][k] = {
            "r2_mean": float(np.mean(d["r2"])), "r2_std": float(np.std(d["r2"])),
            "mae_mean": float(np.mean(d["mae"])), "mae_std": float(np.std(d["mae"])),
        }

# Paired comparison, per seed, of source2-only vs. the full-Source-1 joint
# baseline (same seed = same k Source-2 rows drawn in both conditions, so
# this is a genuine paired difference, not an independent-samples comparison).
paired = {}
for k in k_list:
    diffs = np.array(results["source2_only"][k]["mae"]) - np.array(results["baseline_full_source1_joint"][k]["mae"])
    boot_rng = np.random.RandomState(42)
    boot_means = [boot_rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(5000)]
    paired[k] = {
        "mean_mae_diff_source2only_minus_joint": float(diffs.mean()),
        "std_mae_diff": float(diffs.std()),
        "ci95_bootstrap_over_seeds": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
        "n_seeds": len(diffs),
    }
    print(f"k={k} paired (source2-only - joint) MAE diff: {diffs.mean():.3f} +- {diffs.std():.3f}, "
          f"95% CI {paired[k]['ci95_bootstrap_over_seeds']}")
summary["paired_source2only_vs_joint_mae_diff"] = paired

for key in ["downsampled_source1", "weighted_source2", "baseline_full_source1_joint", "source2_only"]:
    for k in k_list:
        s = summary[key][k]
        print(f"{key} k={k}: R2={s['r2_mean']:.3f}+-{s['r2_std']:.3f} MAE={s['mae_mean']:.3f}")

json.dump(summary, open(f"{ROOT}/data/phase5_negative_transfer_controls.json", "w"), indent=2)
print("Saved data/phase5_negative_transfer_controls.json")
