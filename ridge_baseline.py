# Ridge-regression and mean-prediction baselines on the same Source-1 split used for the headline in-distribution result.
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
RNG = 0


def main():
    X1_full = pd.read_csv(f"{ROOT}/data/featurized_source1.csv")
    y1 = X1_full.pop("target")
    X1 = X1_full

    Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.2, random_state=RNG)

    gbm = joblib.load(f"{ROOT}/models/gbm_source1_main.joblib")
    pred_gbm = gbm.predict(Xte)
    r2_gbm, mae_gbm = r2_score(yte, pred_gbm), mean_absolute_error(yte, pred_gbm)
    print(f"GBM (reference): R2={r2_gbm:.3f} MAE={mae_gbm:.3f}")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    ridge = Ridge(alpha=1.0, random_state=RNG).fit(Xtr_s, ytr)
    pred_ridge = ridge.predict(Xte_s)
    r2_ridge, mae_ridge = r2_score(yte, pred_ridge), mean_absolute_error(yte, pred_ridge)
    print(f"Ridge: R2={r2_ridge:.3f} MAE={mae_ridge:.3f}")

    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    pred_mean = dummy.predict(Xte)
    r2_mean, mae_mean = r2_score(yte, pred_mean), mean_absolute_error(yte, pred_mean)
    print(f"Mean baseline: R2={r2_mean:.3f} MAE={mae_mean:.3f}")

    out = {
        "gbm_reference": {"r2": float(r2_gbm), "mae": float(mae_gbm)},
        "ridge": {"alpha": 1.0, "r2": float(r2_ridge), "mae": float(mae_ridge)},
        "mean_baseline": {"r2": float(r2_mean), "mae": float(mae_mean)},
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    with open(f"{ROOT}/data/ridge_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved data/ridge_baseline.json")


if __name__ == "__main__":
    main()
