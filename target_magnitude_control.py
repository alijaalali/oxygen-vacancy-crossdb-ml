# Matched-subset regression of |residual| on redox activity, structural family, feature-space distance, and target-energy magnitude.
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.linear_model import LinearRegression

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"

meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
nn2 = pd.read_csv(f"{ROOT}/data/nn_distance_source2.csv")


def structural_family(formula):
    try:
        c = Composition(formula)
    except Exception:
        return None
    els = c.get_el_amt_dict()
    n_o = els.get("O", 0.0)
    cations = {e: a for e, a in els.items() if e != "O"}
    n_cations_distinct = len(cations)
    n_cation_atoms = sum(cations.values())
    if n_o == 0 or n_cation_atoms == 0:
        return "other"
    ratio = n_o / n_cation_atoms
    if n_cations_distinct == 1:
        return "binary_oxide"
    if n_cations_distinct == 2 and abs(ratio - 1.5) <= 1.5 * 0.15:
        return "perovskite_like_ABO3"
    return "other_multinary"


meta2["structural_family"] = meta2["composition"].apply(structural_family)

valid_reg = meta2["redox_active"].notna() & meta2["structural_family"].notna()

# Baseline regression, reproduced for a direct side-by-side comparison
Xbase = pd.DataFrame({
    "redox_active": meta2.loc[valid_reg, "redox_active"].astype(float).values,
    "is_binary": (meta2.loc[valid_reg, "structural_family"] == "binary_oxide").astype(float).values,
    "is_perovskite_like": (meta2.loc[valid_reg, "structural_family"] == "perovskite_like_ABO3").astype(float).values,
    "nn_distance": nn2.loc[valid_reg, "nn_distance"].values,
})
y = meta2.loc[valid_reg, "abs_residual"].values
Xbase_std = (Xbase - Xbase.mean()) / Xbase.std()
lr_base = LinearRegression().fit(Xbase_std, y)
base_coefs = dict(zip(Xbase.columns, [float(c) for c in lr_base.coef_]))
base_r2 = float(lr_base.score(Xbase_std, y))

# Extended regression: add target-energy magnitude (|true vacancy energy|)
Xext = Xbase.copy()
Xext["target_magnitude"] = meta2.loc[valid_reg, "target"].abs().values
Xext_std = (Xext - Xext.mean()) / Xext.std()
lr_ext = LinearRegression().fit(Xext_std, y)
ext_coefs = dict(zip(Xext.columns, [float(c) for c in lr_ext.coef_]))
ext_r2 = float(lr_ext.score(Xext_std, y))

out = {
    "description": "Does the redox_active coefficient survive "
                    "adding |target vacancy energy| (target-energy magnitude) as "
                    "an additional standardized covariate, alongside the original "
                    "regression's structural-family indicators and feature-space "
                    "(nn_distance) control? Same 795-row Source-2 zero-shot set, "
                    "same standardization convention as the original regression.",
    "baseline_regression_no_magnitude": {
        "standardized_coefficients_eV_per_std": base_coefs,
        "r2_of_regression": base_r2,
        "n": int(valid_reg.sum()),
    },
    "extended_regression_with_target_magnitude": {
        "standardized_coefficients_eV_per_std": ext_coefs,
        "r2_of_regression": ext_r2,
        "n": int(valid_reg.sum()),
    },
}
print(json.dumps(out, indent=2))
with open(f"{ROOT}/data/phase5_target_magnitude_control.json", "w") as f:
    json.dump(out, f, indent=2)
