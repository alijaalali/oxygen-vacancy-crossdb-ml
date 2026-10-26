"""Standalone rerun of the 15-seed active-learning labels-to-threshold
comparison, computing the across-seed standard deviation (not just the
mean) of how many labeled points each strategy needs to reach the target
MAE threshold. Reuses the exact same pool/holdout split, strategy
functions, and hyperparameters as the multiseed active-learning check in
extended_robustness_analyses.py.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split as tts

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
RNG = 0
t0 = time.time()

X1_full = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
y1 = X1_full.pop("target")
X1 = X1_full

X1_pool_init, X1_holdout, y1_pool_init, y1_holdout = tts(X1, y1, test_size=0.2, random_state=RNG)


def active_learning_curve(X_pool, y_pool, X_test, y_test, strategy, seed, seed_frac=0.08,
                           batch_frac=0.08, n_rounds=12):
    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(X_pool))
    n_seed = max(15, int(seed_frac * len(idx_all)))
    labeled = set(rng.choice(idx_all, n_seed, replace=False))
    maes, n_points = [], []
    batch = max(10, int(batch_frac * len(idx_all)))
    for r in range(n_rounds):
        lab_idx = np.array(sorted(labeled))
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03,
                                       subsample=0.8, random_state=seed)
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
                                                random_state=seed * 1000 + i + 100)
                mm.fit(X_pool.iloc[lab_idx[bidx]], y_pool.iloc[lab_idx[bidx]])
                ens_std[i] = mm.predict(X_pool.iloc[unlabeled])
            unc = ens_std.std(axis=0)
            top = np.argsort(-unc)[:min(batch, len(unlabeled))]
            chosen = unlabeled[top]
        labeled.update(chosen.tolist())
    return n_points, maes


N_AL_SEEDS = 15
al_rand_runs, al_unc_runs = [], []
for seed in range(N_AL_SEEDS):
    n_r, mae_r = active_learning_curve(X1_pool_init, y1_pool_init, X1_holdout, y1_holdout, "random", seed)
    n_u, mae_u = active_learning_curve(X1_pool_init, y1_pool_init, X1_holdout, y1_holdout, "uncertainty", seed)
    al_rand_runs.append(mae_r)
    al_unc_runs.append(mae_u)
    print(f"  seed {seed} done ({time.time()-t0:.1f}s)")

al_rand_arr = np.array(al_rand_runs)
al_unc_arr = np.array(al_unc_runs)
n_points_ref = n_r

target_mae = 1.1 * float(np.mean(al_rand_arr[:, -1]))
k_cross_rand = [next((n_points_ref[i] for i in range(len(n_points_ref)) if al_rand_arr[s, i] <= target_mae), None)
                for s in range(N_AL_SEEDS)]
k_cross_unc = [next((n_points_ref[i] for i in range(len(n_points_ref)) if al_unc_arr[s, i] <= target_mae), None)
               for s in range(N_AL_SEEDS)]

k_rand_valid = [k for k in k_cross_rand if k is not None]
k_unc_valid = [k for k in k_cross_unc if k is not None]

out = {
    "target_mae_eV": target_mae,
    "k_random_per_seed": k_cross_rand,
    "k_uncertainty_per_seed": k_cross_unc,
    "k_random_mean": float(np.mean(k_rand_valid)), "k_random_std": float(np.std(k_rand_valid)),
    "k_uncertainty_mean": float(np.mean(k_unc_valid)), "k_uncertainty_std": float(np.std(k_unc_valid)),
    "n_seeds_random_reached": len(k_rand_valid), "n_seeds_uncertainty_reached": len(k_unc_valid),
}
print(f"\nRandom: mean k={out['k_random_mean']:.1f} +- {out['k_random_std']:.1f}")
print(f"Uncertainty: mean k={out['k_uncertainty_mean']:.1f} +- {out['k_uncertainty_std']:.1f}")

json.dump(out, open(f"{ROOT}/data/phase5_active_learning_variability.json", "w"), indent=2)
print("Saved data/phase5_active_learning_variability.json")
