# Mean number of unique Source-2 compositions represented at each label-efficiency checkpoint k, under composition-grouped sampling.
import numpy as np
import pandas as pd
import json

ROOT = "/glade/work/ajalali/15 - OxygenVacancy-CrossDB-ML-project"
RNG = 0
K_VALUES = [5, 10, 20, 50, 60, 70, 80, 90, 100, 200, 300]
N_SEEDS = 10


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
    test_mask = meta["composition"].isin(set(test_comps))
    return test_mask.values


def main():
    meta2 = pd.read_csv(f"{ROOT}/data/redox_diagnosis_source2.csv")
    test_mask = composition_grouped_split(meta2, 0.2, seed=RNG)
    pool = meta2[~test_mask].reset_index(drop=True)
    print(f"pool rows: {len(pool)}, pool unique compositions: {pool['composition'].nunique()}")

    out = {}
    for k in K_VALUES:
        k_eff = min(k, len(pool))
        counts = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(pool))
            idx = perm[:k_eff]
            counts.append(int(pool.iloc[idx]["composition"].nunique()))
        out[str(k)] = {
            "mean_unique_compositions": float(np.mean(counts)),
            "min": int(min(counts)), "max": int(max(counts)),
        }
        print(f"k={k}: mean unique compositions = {np.mean(counts):.1f} "
              f"(range {min(counts)}-{max(counts)})")

    result = {
        "pool_rows": len(pool),
        "pool_unique_compositions": int(pool["composition"].nunique()),
        "n_seeds": N_SEEDS,
        "by_k": out,
    }
    with open(f"{ROOT}/data/k_composition_counts.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved data/k_composition_counts.json")


if __name__ == "__main__":
    main()
