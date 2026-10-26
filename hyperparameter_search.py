# 5-fold CV hyperparameter sweep on the Source-1 training set.
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
X1_full = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
y1 = X1_full.pop("target")
X1 = X1_full

grid = [
    dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=1.0),
    dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8),
    dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8),
    dict(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8),  # selected
    dict(n_estimators=500, max_depth=2, learning_rate=0.02, subsample=0.8),
    dict(n_estimators=300, max_depth=2, learning_rate=0.01, subsample=0.8),
    dict(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=1.0),
]

kf = KFold(n_splits=5, shuffle=True, random_state=0)
rows = []
for params in grid:
    m = GradientBoostingRegressor(random_state=0, **params)
    scores = cross_val_score(m, X1, y1, cv=kf, scoring="r2")
    rows.append({**params, "cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())})
    print(params, f"R2={scores.mean():.3f}+-{scores.std():.3f}")

json.dump(rows, open(f"{ROOT}/data/phase4_hyperparam_sweep.json", "w"), indent=2)
print("Saved data/phase4_hyperparam_sweep.json")
