# Repeats the zero-shot transfer test, redox diagnosis, and label-efficiency curve across 10 regressors.
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

RNG = 0
ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

# ---------- Reload data (identical to phase2b_analysis.py) ----------
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
print(f"Reloaded: source1={len(X1)} source2={len(X2)}")

REDOX_ACTIVE = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                 "Mo", "W", "Ce", "Pr", "Tb", "U", "Sn", "Pb"}


def has_redox_active_cation(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    els.discard("O")
    return len(els & REDOX_ACTIVE) > 0


meta2 = meta2.copy()
meta2["redox_active"] = meta2["composition"].apply(has_redox_active_cation)
redox_mask = meta2["redox_active"].values

# ============================================================
# Part A: 10-model zero-shot transfer benchmark + redox split
# ============================================================
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=RNG)

scaler = StandardScaler().fit(X1_tr)
X1_tr_s = scaler.transform(X1_tr)
X2_s = scaler.transform(X2)

SCALE_FREE = {"GradientBoosting", "XGBoost", "RandomForest", "AdaBoost"}

models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                    learning_rate=0.02, subsample=0.8,
                                                    random_state=RNG),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.7, random_state=RNG,
                             verbosity=0),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RNG,
                                           n_jobs=4),
    "AdaBoost": AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=RNG),
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    "kNN": KNeighborsRegressor(n_neighbors=8, weights="distance"),
    "GaussianProcess": GaussianProcessRegressor(
        kernel=RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0),
        alpha=1e-6, normalize_y=True, random_state=RNG),
    "Ridge": Ridge(alpha=1.0, random_state=RNG),
    "Lasso": Lasso(alpha=0.01, random_state=RNG, max_iter=5000),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RNG,
                         early_stopping=True),
}

bench = {}
preds_by_model = {}
for name, m in models.items():
    if name in SCALE_FREE:
        m.fit(X1_tr, y1_tr)
        pred = m.predict(X2)
    else:
        m.fit(X1_tr_s, y1_tr)
        pred = m.predict(X2_s)
    r2 = r2_score(y2, pred)
    mae = mean_absolute_error(y2, pred)
    abs_res = np.abs(pred - y2.values)
    r2_redox = r2_score(y2.values[redox_mask], pred[redox_mask])
    r2_fixed = r2_score(y2.values[~redox_mask], pred[~redox_mask])
    mae_redox = abs_res[redox_mask].mean()
    mae_fixed = abs_res[~redox_mask].mean()
    bench[name] = {
        "r2": float(r2), "mae": float(mae),
        "r2_redox": float(r2_redox), "mae_redox": float(mae_redox),
        "r2_fixed": float(r2_fixed), "mae_fixed": float(mae_fixed),
    }
    preds_by_model[name] = pred
    print(f"{name:18s} zero-shot: R2={r2:7.3f} MAE={mae:.3f} | "
          f"redox-active: R2={r2_redox:7.3f} MAE={mae_redox:.3f} | "
          f"fixed-valence: R2={r2_fixed:8.3f} MAE={mae_fixed:.3f}")

# Sanity check: GradientBoosting here should reproduce the Phase-1/Phase-2b number exactly
ref = json.load(open(f"{ROOT}/data/phase1_results.json"))
assert abs(bench["GradientBoosting"]["r2"] - ref["transfer"]["source2_r2"]) < 1e-6
assert abs(bench["GradientBoosting"]["mae"] - ref["transfer"]["source2_mae"]) < 1e-6
print("\nGradientBoosting reproduces Phase-1 numbers exactly -- 10-model benchmark trustworthy.")

with open(f"{ROOT}/data/phase2c_multimodel_benchmark.json", "w") as f:
    json.dump(bench, f, indent=2)

# ============================================================
# Part B: fine-tuning curve across 5 representative model families
# ============================================================
X2_pool, X2_test, y2_pool, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=RNG)
scaler_ft = StandardScaler().fit(pd.concat([X1, X2_pool], ignore_index=True))

k_values = [0, 50, 100, 200, 300, 636]
n_seeds = 5

ft_model_specs = {
    "GradientBoosting": (GradientBoostingRegressor, dict(n_estimators=300, max_depth=2,
                          learning_rate=0.02, subsample=0.8), False, True),
    "XGBoost": (XGBRegressor, dict(n_estimators=300, max_depth=2, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.7, verbosity=0), False, True),
    "RandomForest": (RandomForestRegressor, dict(n_estimators=300, max_depth=6, n_jobs=4), False, True),
    "Ridge": (Ridge, dict(alpha=1.0), True, True),
    "MLP": (MLPRegressor, dict(hidden_layer_sizes=(64, 32), max_iter=2000,
            early_stopping=True), True, True),
}
# tuple = (class, fixed_kwargs, needs_feature_scaling, accepts_random_state)

ft_multimodel = {}
for name, (cls, kw, needs_scale, has_seed) in ft_model_specs.items():
    ft_multimodel[name] = {k: {"r2": [], "mae": []} for k in k_values}
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
            m = cls(random_state=seed, **kw) if has_seed else cls(**kw)
            if needs_scale:
                sc = StandardScaler().fit(Xtr)
                m.fit(sc.transform(Xtr), ytr)
                pred = m.predict(sc.transform(X2_test))
            else:
                m.fit(Xtr, ytr)
                pred = m.predict(X2_test)
            ft_multimodel[name][k]["r2"].append(float(r2_score(y2_test, pred)))
            ft_multimodel[name][k]["mae"].append(float(mean_absolute_error(y2_test, pred)))
        print(f"{name} seed {seed} done")

ft_multimodel_summary = {}
for name in ft_model_specs:
    ft_multimodel_summary[name] = {
        "k_values": k_values,
        "r2_mean": [float(np.mean(ft_multimodel[name][k]["r2"])) for k in k_values],
        "r2_std": [float(np.std(ft_multimodel[name][k]["r2"])) for k in k_values],
        "mae_mean": [float(np.mean(ft_multimodel[name][k]["mae"])) for k in k_values],
        "mae_std": [float(np.std(ft_multimodel[name][k]["mae"])) for k in k_values],
    }
    print(f"\n{name}:")
    for k, r2m in zip(k_values, ft_multimodel_summary[name]["r2_mean"]):
        print(f"  k={k:4d}: R2={r2m:.3f}")

with open(f"{ROOT}/data/phase2c_finetuning_multimodel.json", "w") as f:
    json.dump({"n_seeds": n_seeds, "models": ft_multimodel_summary}, f, indent=2)

print("\nSaved data/phase2c_multimodel_benchmark.json, data/phase2c_finetuning_multimodel.json")
print("DONE")
